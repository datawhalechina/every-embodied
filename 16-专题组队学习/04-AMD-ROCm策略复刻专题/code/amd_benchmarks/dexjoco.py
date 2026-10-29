"""DexJoCo native ROCm JAX LoRA training and official simulator evaluation."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from .common import bind_run, disk_gate, fingerprint, memory_gate, reconcile_metrics, require_files, require_free_port, run_logged, sha256, write_json


def source_root(c):
    return Path(c['source_root']) / 'openpi'


def setup(c):
    source = source_root(c)
    sys.path.insert(0, str(source / 'src'))
    sys.path.insert(0, str(source / 'packages/openpi-client/src'))
    if c.get('lerobot_source'):
        sys.path.insert(0, str(Path(c['lerobot_source'])))
    if c.get('asset_cache'):
        os.environ['OPENPI_DATA_HOME'] = c['asset_cache']
    os.environ['JAX_PLATFORMS'] = 'rocm'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_disable_hlo_passes=' not in flags:
        flags += ' --xla_disable_hlo_passes=priority-fusion'
    # The gfx1151 device has 64 KiB shared memory; this compiler's fused
    # 1048-token softmax requests 256 KiB. Use the unfused implementation.
    os.environ['XLA_FLAGS'] = flags.strip()
    os.environ['DEXJOCO_PYTORCH_COMPILE_MODE'] = 'none'
    # Import upstream configuration from a private working directory. Never
    # overwrite the upstream global config.yaml used by other jobs.
    import yaml
    root = Path(c['run_root'])
    sandbox = root / 'config_sandbox'
    sandbox.mkdir(parents=True, exist_ok=True)
    settings = yaml.safe_load((source / 'config.yaml').read_text())
    settings.update(pretrained_model_path=str(Path(c['base_model']) / 'params'),
        pretrained_model_action_dim_44_path=str(Path(c['base_model']) / 'params'),
        dataset_root=str(Path(c['dataset_root']).parent), rand_full_dataset_root=str(Path(c['dataset_root']).parent),
        ckpts_root=str(root / 'checkpoints'), rand_full_ckpts_root=str(root / 'checkpoints'),
        wandb_enabled=False, batch_size=c['batch_size'], single_arm_steps=c['steps'], dual_arm_steps=c['steps'])
    (sandbox / 'config.yaml').write_text(yaml.safe_dump(settings), encoding='utf-8')
    os.chdir(sandbox)
    from openpi.shared import download
    def cached_asset(url, *, force_download=False, **kwargs):
        from urllib.parse import urlparse
        parsed = urlparse(str(url))
        path = (Path(url) if not parsed.scheme else
                download.get_cache_dir() / parsed.netloc / parsed.path.strip('/'))
        if force_download or not path.exists():
            raise FileNotFoundError(f'Prefetch this asset through DOWNLOAD_ENTRY before training: {url}; cache target: {path}')
        return path.resolve()
    # Model/data stages may read cached assets, but never start implicit downloads.
    download.maybe_download = cached_asset


def get_config(c):
    setup(c)
    from openpi.training import config as configs
    from openpi.training import optimizer
    from openpi.training import weight_loaders
    base = configs.get_config(c['task'])
    data = dataclasses.replace(base.data, root=Path(c['dataset_root']))
    return dataclasses.replace(base, exp_name='course', data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(c['base_model']) / 'params')),
        batch_size=c['batch_size'], num_workers=0, num_train_steps=c['steps'], seed=c['seed'],
        checkpoint_base_dir=str(Path(c['run_root']) / 'checkpoints'),
        assets_base_dir=str(Path(c['run_root']) / 'assets'),
        log_interval=1, save_interval=max(1, c.get('save_every', 1000)), keep_period=None,
        resume=c.get('resume', False), overwrite=False, wandb_enabled=False,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=int(c.get('warmup_steps', 100)),
            peak_lr=c['learning_rate'], decay_steps=max(c['steps'], 1000), decay_lr=c['learning_rate'] * .1))


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight(c):
    require_files(c['source_root'], ['openpi/config.yaml', 'openpi/scripts/train.py',
                                      'configs/rand_obj/' + c['task'] + '.yaml'])
    require_files(c['dataset_root'], ['meta/info.json', 'meta/tasks.parquet'])
    require_files(c['base_model'], ['params'])
    setup(c)
    from openpi.shared import download
    tokenizer = download.get_cache_dir() / 'big_vision/paligemma_tokenizer.model'
    require_files(tokenizer.parent, [tokenizer.name])
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    platform_versions = [str(getattr(d.client, 'platform_version', '')).lower() for d in devices]
    # Recent PJRT reports the generic backend name 'gpu' for ROCm devices.
    if not devices or not all('rocm' in v for v in platform_versions):
        raise RuntimeError('Native ROCm JAX is required; no CPU fallback is accepted.')
    x = jnp.ones((64, 64))
    total = float((x @ x).sum().block_until_ready())
    assert total == 262144.0
    return {'jax': jax.__version__, 'backend': jax.default_backend(), 'devices': [str(x) for x in jax.devices()],
            'matmul_sum': total, 'tokenizer_sha256': sha256(tokenizer),
            'free_disk_gib': disk_gate(c['run_root'], c.get('minimum_free_gib', 10)),
            'note': 'Backend and data preflight only, not task success.'}


def prepare(c):
    root = Path(c['run_root'])
    identity = fingerprint(c['dataset_root'], ['meta/info.json', 'meta/tasks.parquet',
                                               'meta/episodes/**/*.parquet', 'data/**/*.parquet'])
    base = fingerprint(c['base_model'], ['params/**/*', 'assets/**/*.json'])
    bind_run(c, identity, base)
    if (root / 'prepared.json').exists():
        previous = json.loads((root / 'prepared.json').read_text())
        if previous.get('normalization_frames') != c.get('normalization_frames'):
            raise RuntimeError('Normalization selection changed; use a new run_root.')
        if previous.get('normalization_batch_size', 1) != c.get('normalization_batch_size', 1):
            raise RuntimeError('Normalization batching changed; use a new run_root.')
        if Path(previous['stats_path']).is_file() and sha256(previous['stats_path']) == previous['normalization_sha256']:
            return previous
        raise RuntimeError('Prepared normalization asset is missing or changed; restore it or use a fresh run_root.')
    cfg = get_config(c)
    disk_gate(cfg.checkpoint_dir.parent, c.get('minimum_checkpoint_free_gib', c.get('minimum_free_gib', 24)))
    sys.path.insert(0, str(source_root(c) / 'scripts'))
    norm = load_script(source_root(c) / 'scripts/compute_norm_stats.py', 'compute_norm_stats')
    normalization_batch_size = int(c.get('normalization_batch_size', 1))
    total_frames = json.loads((Path(c['dataset_root']) / 'meta/info.json').read_text())['total_frames']
    selected_frames = min(c.get('normalization_frames') or total_frames, total_frames)
    if normalization_batch_size < 1 or selected_frames % normalization_batch_size:
        raise ValueError('Normalization batch size must divide the selected frame count; no frames may be dropped.')
    norm_cfg = dataclasses.replace(cfg, batch_size=normalization_batch_size,
                                   num_workers=int(c.get('normalization_workers', 0)))
    norm.main(norm_cfg, max_frames=c.get('normalization_frames'))
    stats = cfg.assets_dirs / 'local_repo/norm_stats.json'
    if not stats.is_file():
        raise FileNotFoundError(stats)
    result = {'dataset_identity': identity['sha256'], 'base_identity': base['sha256'],
              'normalization_frames': c.get('normalization_frames', 'all'),
              'normalization_frames_processed': selected_frames,
              'normalization_batch_size': normalization_batch_size,
              'normalization_workers': norm_cfg.num_workers,
              'normalization_sha256': sha256(stats), 'stats_path': str(stats),
              'model_config': repr(cfg.model), 'training_recipe': 'upstream single-task or bimanual LoRA'}
    write_json(root / 'prepared.json', result)
    return result


def train(c):
    root = Path(c['run_root'])
    require_files(root, ['identity.json', 'prepared.json'])
    prepared = json.loads((root / 'prepared.json').read_text())
    if prepared.get('normalization_frames') != c.get('normalization_frames'):
        raise RuntimeError('Normalization selection changed; use a new run_root.')
    if sha256(prepared['stats_path']) != prepared['normalization_sha256']:
        raise RuntimeError('Normalization assets changed after preparation.')
    disk_gate(root, c.get('minimum_free_gib', 10))
    memory_gate(c.get('minimum_available_ram_gib', 24))
    cfg = get_config(c)
    disk_gate(cfg.checkpoint_dir.parent, c.get('minimum_checkpoint_free_gib', c.get('minimum_free_gib', 24)))
    os.environ['DEXJOCO_METRICS_PATH'] = str(root / 'metrics.jsonl')
    script = load_script(source_root(c) / 'scripts/train.py', 'course_dex_train')
    def compatible_state_map(state, selected, fn):
        flat = state.filter(selected).flat_state()
        # Older Flax returns a mapping; newer Flax returns (path, variable) pairs.
        keys = set(flat.keys()) if hasattr(flat, 'keys') else {path for path, _ in flat}
        return state.map(lambda path, value: fn(value) if path in keys else value)
    script.nnx_utils.state_map = compatible_state_map
    import jax
    import numpy as np
    import hashlib
    import inspect
    if 'axis_types' in inspect.signature(jax.make_mesh).parameters:
        def automatic_mesh(num_fsdp_devices):
            if jax.device_count() % num_fsdp_devices:
                raise ValueError('Device count must be divisible by fsdp_devices.')
            return jax.make_mesh((jax.device_count() // num_fsdp_devices, num_fsdp_devices),
                ('batch', 'fsdp'), axis_types=(jax.sharding.AxisType.Auto,) * 2)
        # Preserve the automatic partitioning expected by this upstream trainer.
        script.sharding.make_mesh = automatic_mesh
    existing_steps = [int(p.name) for p in cfg.checkpoint_dir.glob('*') if p.is_dir() and p.name.isdigit()]
    skip_batches = max(existing_steps) + 1 if c.get('resume') and existing_steps else 0
    original_loader = script._data_loader.create_data_loader
    class ResumedLoader:
        def __init__(self, inner):
            self.inner = inner
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def __iter__(self):
            iterator = iter(self.inner)
            for _ in range(skip_batches):
                next(iterator)
            yield from iterator
    def resumed_loader(*args, **kwargs):
        return ResumedLoader(original_loader(*args, **kwargs))
    script._data_loader.create_data_loader = resumed_loader
    measurements = {'metrics_recovery': reconcile_metrics(root, skip_batches)}
    def parameter_digest(state):
        leaves = jax.tree.leaves(state.params.filter(cfg.trainable_filter))
        h = hashlib.sha256()
        count = 0
        for leaf in leaves:
            array = np.asarray(jax.device_get(leaf))
            count += int(array.size)
            h.update(array.tobytes())
        return h.hexdigest(), count
    original_init = script.init_train_state
    def measured_init(*args, **kwargs):
        state, sharding = original_init(*args, **kwargs)
        if not kwargs.get('resume', False):
            digest, count = parameter_digest(state)
            measurements.update(parameter_digest_before=digest, trainable_parameters=count,
                                steps_start=int(state.step))
            write_json(root / 'training_measurements.json', measurements)
        return state, sharding
    script.init_train_state = measured_init
    original_restore = script._checkpoints.restore_state
    def measured_restore(*args, **kwargs):
        state = original_restore(*args, **kwargs)
        if int(state.step) != skip_batches:
            raise RuntimeError('Checkpoint index and restored optimizer step disagree; refusing sampler replay.')
        if int(state.step) >= c['steps']:
            raise ValueError('Checkpoint already reached the requested total steps.')
        digest, count = parameter_digest(state)
        measurements.update(parameter_digest_before=digest, trainable_parameters=count,
                            steps_start=int(state.step))
        write_json(root / 'training_measurements.json', measurements)
        return state
    script._checkpoints.restore_state = measured_restore
    original_save = script._checkpoints.save_state
    def measured_save(manager, state, loader, step):
        result = original_save(manager, state, loader, step)
        digest, count = parameter_digest(state)
        measurements.update(parameter_digest_after=digest, trainable_parameters=count)
        write_json(root / 'training_measurements.json', measurements)
        return result
    script._checkpoints.save_state = measured_save
    original_log = None
    def metric_log(data, step=None, **kwargs):
        if 'loss' in data:
            row = {k: float(v) for k, v in data.items() if np.ndim(v) == 0}
            row.update(step=int(step) + 1, time_unix=time.time())
            with (root / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
        return original_log(data, step=step, **kwargs)
    original_wandb_init = script.init_wandb
    def measured_wandb_init(*args, **kwargs):
        nonlocal original_log
        result = original_wandb_init(*args, **kwargs)
        # wandb.init replaces the module-level log function, even in disabled mode.
        original_log = script.wandb.log
        script.wandb.log = metric_log
        return result
    script.init_wandb = measured_wandb_init
    started = time.time()
    script.main(cfg)
    if measurements.get('parameter_digest_before') == measurements.get('parameter_digest_after'):
        raise RuntimeError('Training produced no measured parameter update.')
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if not rows or not all(np.isfinite(row['loss']) for row in rows):
        raise RuntimeError('Training metrics are absent or non-finite.')
    candidates = sorted([p for p in cfg.checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if not candidates:
        raise RuntimeError('Training produced no checkpoint.')
    checkpoint = candidates[-1]
    require_files(checkpoint, ['params', 'train_state', 'assets'])
    identity = fingerprint(checkpoint, ['params/**/*', 'assets/**/*.json'])
    logical_checkpoint = root / 'checkpoints' / cfg.name / cfg.exp_name / checkpoint.name
    result = {'checkpoint': str(logical_checkpoint), 'checkpoint_sha256': identity['sha256'],
              'checkpoint_index': int(checkpoint.name), 'trained_steps_target': c['steps'],
              'duration_s': time.time() - started, 'batch_size': c['batch_size'],
              'gradient_accumulation': 1, 'effective_batch': c['batch_size'],
              'learning_rate': c['learning_rate'], 'freeze_scope': repr(cfg.freeze_filter),
              'identity_files': identity['files'], 'resume_replayed_batches': skip_batches}
    result.update(measurements)
    result['device_memory_stats'] = jax.devices()[0].memory_stats()
    write_json(root / 'trained_model.json', result)
    return result


def policy_record(c):
    root = Path(c['run_root'])
    if c.get('policy_variant') == 'base':
        identity = json.loads((root / 'identity.json').read_text())
        return {'checkpoint': c['base_model'], 'checkpoint_sha256': identity['base_model_identity']['sha256'],
                'trained_steps_target': 0, 'policy_variant': 'base'}
    record = json.loads((root / 'trained_model.json').read_text())
    return dict(record, policy_variant='trained')


def checkpoint_model_config(model, checkpoint):
    """Keep task adapters present in the checkpoint and reject silent parameter dropping."""
    metadata = json.loads((Path(checkpoint) / 'params/_METADATA').read_text())
    adapter_keys = [key for key in metadata['tree_metadata'] if 'lora_' in key]
    if not adapter_keys:
        model = dataclasses.replace(model,
            paligemma_variant=model.paligemma_variant.removesuffix('_lora'),
            action_expert_variant=model.action_expert_variant.removesuffix('_lora'))
    elif not all('lora' in value for value in [model.paligemma_variant, model.action_expert_variant]):
        raise ValueError('Checkpoint contains task adapters, but the model disables them.')

    class StrictCheckpointConfig(type(model)):
        def load(self, params, *, remove_extra_params=False):
            return super().load(params, remove_extra_params=False)

    checked = StrictCheckpointConfig(**{field.name: getattr(model, field.name)
                                       for field in dataclasses.fields(model) if field.init})
    return checked, {'adapter_tensor_count': len(adapter_keys),
                     'paligemma_variant': checked.paligemma_variant,
                     'action_expert_variant': checked.action_expert_variant,
                     'remove_extra_params': False, 'loader_contract': 'strict-adapters-v2'}


def serve(c):
    cfg = get_config(c)
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    root = Path(c['run_root'])
    record = policy_record(c)
    checkpoint = Path(record['checkpoint'])
    identity = fingerprint(checkpoint, ['params/**/*', 'assets/**/*.json'])
    if identity['sha256'] != record['checkpoint_sha256']:
        raise RuntimeError('Trained checkpoint identity changed.')
    model_config, load_audit = checkpoint_model_config(cfg.model, checkpoint)
    cfg = dataclasses.replace(cfg, model=model_config)
    policy = policy_config.create_trained_policy(cfg, checkpoint)
    if c.get('load_audit_record'):
        write_json(c['load_audit_record'], load_audit)
    if c.get('warmup_observation'):
        import numpy as np
        from openpi_client import msgpack_numpy
        observation = msgpack_numpy.unpackb(Path(c['warmup_observation']).read_bytes())
        saved_rng = policy._rng
        started = time.perf_counter()
        output = policy.infer(observation)
        policy._rng = saved_rng
        if not np.isfinite(output['actions']).all():
            raise RuntimeError('Policy warmup returned non-finite actions.')
        write_json(c['warmup_record'], {'seconds': time.perf_counter() - started,
            'rng_restored': True, 'action_shape': list(output['actions'].shape),
            'scope': 'Real reset observation; compile before any evaluated episode starts.'})
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host='127.0.0.1',
        port=c.get('port', 8122), metadata=policy.metadata).serve_forever()


def evaluate(c):
    require_free_port(c.get('port', 8122))
    if not c.get('eval_seeds', [100, 101, 102, 103, 104]):
        raise ValueError('Evaluation seeds must not be empty.')
    root = Path(c['run_root'])
    require_files(root, ['identity.json'] if c.get('policy_variant') == 'base' else ['trained_model.json'])
    record = policy_record(c)
    import uuid
    prefix = 'baseline_' if record['policy_variant'] == 'base' else 'eval_'
    directory = root / (prefix + record['checkpoint_sha256'][:12] + '_' + uuid.uuid4().hex[:8])
    directory.mkdir(exist_ok=False)
    c = dict(c, load_audit_record=str(directory / 'model_load_audit.json'))
    if c.get('warmup_policy', False):
        c = dict(c, warmup_observation=str(directory / 'warmup_observation.msgpack'),
                 warmup_record=str(directory / 'warmup.json'))
        probe = '''import os,sys,yaml
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from openpi_client import msgpack_numpy
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
env=DexJoCoOpenPIEnv(env_name=cfg['env_name'],camera_mapping=cfg['camera_mapping'],
    seed=int(sys.argv[3]),rand_full=False,randomize_dynamics=False,
    dual_arm=cfg['robot_type']=='dual_arm',prompt=cfg['prompt'],render_mode='rgb_array',
    pad_state_dim46=sys.argv[4]=='true',password=cfg.get('password'))
try:
    env.start();env.reset()
    Path(sys.argv[2]).write_bytes(msgpack_numpy.packb(env.get_obs()))
finally:
    env.close()
'''
        probe_env = os.environ.copy()
        probe_env['PYTHONPATH'] = str(source_root(c) / 'packages/openpi-client/src')
        run_logged([str(Path(c['simulator_eval']).parent / 'python'), '-c', probe,
            str(Path(c['source_root']) / 'configs/rand_obj' / (c['task'] + '.yaml')),
            c['warmup_observation'], str(c['eval_seeds'][0]),
            str(bool(c.get('dual_arm', False))).lower()], c['source_root'],
            directory / 'warmup_probe.log', probe_env)
    path = root / 'evaluation_config.json'
    write_json(path, c)
    log = (directory / 'server.log').open('w')
    p = subprocess.Popen([sys.executable, '-m', 'amd_benchmarks.dexjoco', '--serve', str(path)], stdout=log, stderr=subprocess.STDOUT)
    port = int(c.get('port', 8122))
    try:
        for _ in range(600):
            if p.poll() is not None:
                raise RuntimeError('Policy server failed; inspect server.log.')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Policy server did not open the port.')
        rows = []
        clock = c.get('evaluation_clock', 'official_async')
        if clock not in {'official_async', 'pause_while_waiting'}:
            raise ValueError('Unsupported evaluation clock condition.')
        adapted_client = None
        if clock == 'pause_while_waiting':
            from .clock_control import materialize_client
            adapted_client = directory / 'clock_controlled_client.py'
            materialize_client(Path(c['source_root']) / 'dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py', adapted_client)
        seeds = c.get('eval_seeds', [100, 101, 102, 103, 104])
        if len(set(seeds)) != len(seeds):
            raise ValueError('Evaluation seeds must be unique.')
        for seed in seeds:
            dest = directory / f'seed_{seed}'
            command = [c['simulator_eval'], '--config=' + str(Path(c['source_root']) / 'configs/rand_obj' / (c['task'] + '.yaml')),
                       '--seed=' + str(seed), '--host=127.0.0.1', '--port=' + str(port),
                       '--output=' + str(dest), '--render-mode=rgb_array', '--episodes=1']
            if adapted_client:
                command = [str(Path(c['simulator_eval']).parent / 'python'), str(adapted_client), *command[1:]]
            if c.get('dual_arm', False):
                command.append('--pad-state-dim46')
            env = os.environ.copy()
            # Expose only the protocol client; framework source overrides stay isolated.
            env['PYTHONPATH'] = str(source_root(c) / 'packages/openpi-client/src')
            run_logged(command, c['source_root'], directory / 'eval.log', env)
            markers = list(dest.glob('success_rate_*_1.txt'))
            if len(markers) != 1:
                raise RuntimeError('Missing or ambiguous official result marker.')
            count = int(re.fullmatch(r'success_rate_(\d+)_1.txt', markers[0].name)[1])
            if count not in (0, 1):
                raise ValueError('Invalid episode success count.')
            row = {'seed': seed, 'success': bool(count), 'checkpoint_sha256': record['checkpoint_sha256'],
                   'policy_variant': record['policy_variant'], 'directory': str(dest)}
            rows.append(row)
            with (directory / 'results.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
        result = {'episodes': len(rows), 'successes': sum(x['success'] for x in rows),
                  'success_rate': sum(x['success'] for x in rows) / len(rows),
                  'seeds': [x['seed'] for x in rows], 'protocol': 'official task termination and success predicate',
                  'checkpoint_sha256': record['checkpoint_sha256'], 'directory': str(directory),
                  'policy_variant': record['policy_variant'],
                  'warmup_policy': bool(c.get('warmup_policy', False)),
                  'loader_contract': 'strict-adapters-v2',
                  'evaluation_clock': clock,
                  'model_load_audit': json.loads(Path(c['load_audit_record']).read_text()),
                  'warmup': json.loads(Path(c['warmup_record']).read_text()) if c.get('warmup_policy') else None,
                  'validation_only': len(rows) < 20 or (record['policy_variant'] == 'trained' and record['trained_steps_target'] < 100)}
        if clock == 'pause_while_waiting':
            result['protocol'] = 'official task stepping and success; physics pauses during inference waits (separate deployment condition)'
        write_json(directory / 'summary.json', result)
        return result
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
        log.close()


def baseline(c):
    return evaluate(dict(c, policy_variant='base'))


def summary(c):
    root = Path(c['run_root'])
    return {p.name: json.loads(p.read_text()) for p in root.glob('*_status.json')}


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--serve':
        serve(json.loads(Path(sys.argv[2]).read_text()))
