"""RoboCasa365 GR00T projector adaptation with composite checkpoints.

Frozen base weights are referenced by hash. Only trained tensors and optimizer
state are saved, avoiding a duplicate multi-gigabyte frozen backbone per save.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

from .common import bind_run, disk_gate, fingerprint, reconcile_metrics, require_files, require_free_port, sha256, torch_preflight, write_json


def setup(c):
    source = Path(c['source_root'])
    for path in [source, Path(c['robocasa_root']), Path(c['robosuite_root'])]:
        sys.path.insert(0, str(path))
    # Numba's optional coverage integration is incompatible with some images.
    # Disable that optional integration in this worker only.
    sys.modules['coverage'] = None
    # HIP allocation before torchvision/timm avoids the known gfx1151 import issue.
    import torch
    if torch.cuda.is_available():
        torch.empty(1, device='cuda')
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.siglip.modeling_siglip import eager_attention_forward
    ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = eager_attention_forward


def validate_eval_split(c):
    if c.get("eval_split", "pretrain") not in {"pretrain", "target", "all", None}:
        raise ValueError("This pinned RoboCasa version supports pretrain, target, all or None; not test.")


def preflight(c):
    validate_eval_split(c)
    require_files(c['source_root'], ['gr00t/model/policy.py', 'gr00t/data/dataset.py'])
    require_files(c['dataset_root'], ['meta/info.json', 'meta/tasks.parquet'])
    require_files(c['base_model'], ['config.json', 'model.safetensors.index.json', 'experiment_cfg/metadata.json'])
    setup(c)
    result = torch_preflight()
    result['free_disk_gib'] = disk_gate(c['run_root'], c.get('minimum_free_gib', 3))
    info = json.loads((Path(c['dataset_root']) / 'meta/info.json').read_text())
    assert info['features']['observation.state']['shape'] == [16]
    assert info['features']['action']['shape'] == [12]
    result['dataset_episodes'] = info['total_episodes']
    result['note'] = 'Device and data contract only; not a learned-policy success.'
    return result


def prepare(c):
    """Create episode-based GR00T views of v3 data without copying videos.

    Preserve absolute video timestamps. Fail if cameras have different time
    origins instead of silently pairing images from different episodes.
    """
    import numpy as np
    import pandas as pd
    source = Path(c['dataset_root'])
    root = Path(c['run_root'])
    dst = root / 'prepared'
    info = json.loads((source / 'meta/info.json').read_text())
    di = fingerprint(source, ['meta/info.json', 'meta/tasks.parquet', 'meta/episodes/**/*.parquet'])
    bi = fingerprint(c['base_model'], ['config.json', '*.safetensors', 'experiment_cfg/metadata.json'])
    bind_run(c, di, bi)
    if (dst / 'prepare_manifest.json').exists():
        prepared = json.loads((dst / 'prepare_manifest.json').read_text())
        expected = min(int(c.get('training_episodes', 32)), info['total_episodes'])
        if prepared['episodes'] != expected:
            raise RuntimeError('Training episode selection changed; use a new run_root.')
        return prepared
    dst.mkdir(parents=True, exist_ok=False)
    (dst / 'meta').mkdir()
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((source / 'meta/episodes').rglob('*.parquet'))])
    episodes = episodes.sort_values('episode_index').head(int(c.get('training_episodes', 32)))
    if len(episodes) < 2:
        raise ValueError('At least two real episodes are required.')
    cameras = [k for k in info['features'] if k.startswith('observation.images.')]
    tables = {}
    metadata = []
    all_state, all_action = [], []
    for _, ep in episodes.iterrows():
        episode = int(ep['episode_index'])
        key = (int(ep['data/chunk_index']), int(ep['data/file_index']))
        if key not in tables:
            tables[key] = pd.read_parquet(source / info['data_path'].format(chunk_index=key[0], file_index=key[1]))
        frame = tables[key][tables[key].episode_index == episode].copy()
        if len(frame) != int(ep['length']):
            raise ValueError(f'Episode {episode}: missing or truncated data.')
        offsets = [float(ep[f'videos/{cam}/from_timestamp']) for cam in cameras]
        if not np.allclose(offsets, offsets[0], atol=1e-4):
            raise ValueError('Camera time origins differ; explicit video segmentation is required.')
        frame['timestamp'] = frame['timestamp'] + offsets[0]
        target = dst / f'data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet'
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        for cam in cameras:
            video = source / info['video_path'].format(video_key=cam,
                chunk_index=int(ep[f'videos/{cam}/chunk_index']), file_index=int(ep[f'videos/{cam}/file_index']))
            if not video.is_file():
                raise FileNotFoundError(video)
            link = dst / f'videos/chunk-{episode // 1000:03d}/{cam}/episode_{episode:06d}.mp4'
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(video.resolve())
        metadata.append({'episode_index': episode, 'length': len(frame), 'tasks': list(ep['tasks'])})
        all_state.append(np.stack(frame['observation.state']))
        all_action.append(np.stack(frame['action']))
    modality_path = Path(c['robocasa_root']) / 'robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json'
    modality = json.loads(modality_path.read_text())
    write_json(dst / 'meta/modality.json', modality)
    # The published v3 task dataset uses the canonical PandaOmron sorted schema.
    assert modality['state']['base_position']['start'] == 0
    assert modality['action']['base_motion']['start'] == 0
    info.update(codebase_version='v2.1', total_episodes=len(metadata),
                total_frames=sum(x['length'] for x in metadata),
                data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
    write_json(dst / 'meta/info.json', info)
    tasks = pd.read_parquet(source / 'meta/tasks.parquet')
    task_rows = [{'task_index': int(row.task_index), 'task': str(text)} for text, row in tasks.iterrows()]
    (dst / 'meta/tasks.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in task_rows))
    (dst / 'meta/episodes.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in metadata))
    stats = {}
    for name, arrays in [('observation.state', all_state), ('action', all_action)]:
        values = np.concatenate(arrays)
        stats[name] = {k: v.tolist() for k, v in {
            'mean': values.mean(0), 'std': values.std(0), 'min': values.min(0), 'max': values.max(0),
            'q01': np.quantile(values, .01, axis=0), 'q99': np.quantile(values, .99, axis=0)}.items()}
    write_json(dst / 'meta/stats.json', stats)
    result = {'episodes': len(metadata), 'frames': info['total_frames'],
              'episode_ids': [x['episode_index'] for x in metadata], 'video_mode': 'shared source + absolute timestamps',
              'modality_sha256': sha256(modality_path), 'dataset_identity': di['sha256'], 'base_identity': bi['sha256']}
    write_json(dst / 'prepare_manifest.json', result)
    return result


def tensor_digest(state):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def train(c):
    setup(c)
    import numpy as np
    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.model.transforms import DefaultDataCollator
    root = Path(c['run_root'])
    require_files(root, ['identity.json', 'prepared/prepare_manifest.json'])
    prepared = json.loads((root / 'prepared/prepare_manifest.json').read_text())
    info = json.loads((Path(c['dataset_root']) / 'meta/info.json').read_text())
    if prepared['episodes'] != min(int(c.get('training_episodes', 32)), info['total_episodes']):
        raise RuntimeError('Training episode selection changed; use a new run_root.')
    disk_gate(root, c.get('minimum_free_gib', 3))
    config = DATA_CONFIG_MAP['panda_omron']
    dataset = LeRobotSingleDataset(dataset_path=root / 'prepared', modality_configs=config.modality_config(),
                                  transforms=config.transform(), embodiment_tag=EmbodimentTag('new_embodiment'),
                                  video_backend='opencv')
    model = GR00T_N1_5.from_pretrained(c['base_model'], tune_llm=False, tune_visual=False,
                                     tune_projector=True, tune_diffusion_model=False).to('cuda')
    # The upstream projector flag also enables large embodiment-specific heads.
    # Restrict this course recipe to the shared VL projector explicitly.
    model.requires_grad_(False)
    model.action_head.vlln.requires_grad_(True)
    model.action_head.vl_self_attention.requires_grad_(True)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable:
        raise RuntimeError('No trainable projector parameters.')
    optimizer = torch.optim.AdamW(trainable.values(), lr=c['learning_rate'])
    checkpoint = root / 'checkpoint.pt'
    step = 0
    if checkpoint.exists():
        if not c.get('resume', False):
            raise FileExistsError('Checkpoint exists: use resume=True or a fresh run_root.')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['identity'] != json.loads((root / 'identity.json').read_text()):
            raise RuntimeError('Checkpoint identity does not match this run.')
        model.load_state_dict(saved['weights'], strict=False)
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to('cuda')
        step = saved['step']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['random_rng'])
        np.random.set_state(saved['numpy_rng'])
        if step >= c['steps']:
            raise ValueError(f'Checkpoint already has {step} steps; set steps above this value to resume.')
    else:
        torch.manual_seed(c['seed']); random.seed(c['seed']); np.random.seed(c['seed'])
    steps_start = step
    recovery = reconcile_metrics(root, step)
    before = tensor_digest(trainable)
    model.train()
    collate = DefaultDataCollator()
    started = time.time()
    final_loss = None
    metrics = root / 'metrics.jsonl'
    print(json.dumps({'trainable_parameters': sum(p.numel() for p in trainable.values()),
                      'batch_size': c['batch_size'], 'gradient_accumulation': 1,
                      'steps_start': step, 'steps_target': c['steps'], 'learning_rate': c['learning_rate']}), flush=True)
    while step < c['steps']:
        # Step-indexed sampling makes sample order reproducible across resume.
        g = torch.Generator().manual_seed(c['seed'] + step)
        indices = torch.randint(len(dataset), (c['batch_size'],), generator=g).tolist()
        batch = collate([dataset[i] for i in indices])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(batch)
            loss = result['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        final_loss = float(loss.detach().cpu())
        record = {'step': step, 'loss': final_loss, 'elapsed_s': time.time() - started,
                  'gpu_peak_bytes': torch.cuda.max_memory_allocated()}
        with metrics.open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if step % int(c.get('save_every', 100)) == 0 or step == c['steps']:
            full_state = model.state_dict()
            buffer_names = {n for n, _ in model.named_buffers()}
            state = {n: value.detach().cpu() for n, value in full_state.items()
                     if n in trainable or n in buffer_names}
            payload = {'step': step, 'weights': state, 'optimizer': optimizer.state_dict(),
                       'identity': json.loads((root / 'identity.json').read_text()),
                       'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                       'random_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
            temp = checkpoint.with_suffix('.tmp')
            torch.save(payload, temp)
            temp.replace(checkpoint)
    write_json(root / 'metadata.json', dataset.metadata.model_dump(mode='json'))
    digest = sha256(checkpoint)
    result = {'step': step, 'steps_start': steps_start, 'metrics_recovery': recovery,
              'loss': final_loss, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
              'parameter_digest_before': before, 'parameter_digest_after': tensor_digest(trainable),
              'trainable_parameters': sum(p.numel() for p in trainable.values()),
              'gpu_peak_bytes': torch.cuda.max_memory_allocated(), 'duration_s': time.time() - started,
              'recipe': 'GR00T vlln + vl_self_attention only; all other parameters frozen; no gradient accumulation'}
    if step and result['parameter_digest_before'] == result['parameter_digest_after']:
        raise RuntimeError('Training produced no parameter update.')
    write_json(root / 'trained_model.json', result)
    return result


def policy_record(c):
    root = Path(c['run_root'])
    if c.get('policy_variant') == 'base':
        identity = json.loads((root / 'identity.json').read_text())
        return {'checkpoint_sha256': identity['base_model_identity']['sha256'],
                'step': 0, 'policy_variant': 'base'}
    return dict(json.loads((root / 'trained_model.json').read_text()), policy_variant='trained')


def serve(c):
    setup(c)
    import torch
    import numpy as np
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.data.schema import DatasetMetadata
    from gr00t.eval.robot import RobotInferenceServer
    torch.manual_seed(c['seed']); np.random.seed(c['seed']); random.seed(c['seed'])
    root = Path(c['run_root'])
    record = policy_record(c)
    config = DATA_CONFIG_MAP['panda_omron']
    policy = Gr00tPolicy(model_path=c['base_model'], modality_config=config.modality_config(),
                         modality_transform=config.transform(), embodiment_tag='new_embodiment', denoising_steps=4)
    if record['policy_variant'] == 'trained':
        if sha256(root / 'checkpoint.pt') != record['checkpoint_sha256']:
            raise RuntimeError('Checkpoint hash changed.')
        saved = torch.load(root / 'checkpoint.pt', map_location='cpu', weights_only=False)
        target_keys = set(policy.model.state_dict())
        weights = dict(saved['weights'])
        for key in set(weights) - target_keys:
            if key.endswith(('.position_ids', '.rotary_emb.inv_freq')):
                weights.pop(key)
            else:
                raise RuntimeError(f'Unknown trained tensor: {key}')
        result = policy.model.load_state_dict(weights, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(result.unexpected_keys)
        policy.modality_transform.set_metadata(DatasetMetadata.model_validate(json.loads((root / 'metadata.json').read_text())))
    RobotInferenceServer(policy, port=int(c.get('port', 8121))).run()


def evaluate(c):
    validate_eval_split(c)
    require_free_port(c.get('port', 8121))
    setup(c)
    import numpy as np
    from unittest.mock import patch
    from gr00t.eval import simulation
    from gr00t.eval.simulation import SimulationInferenceClient, SimulationConfig, VideoConfig, MultiStepConfig
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    root = Path(c['run_root'])
    require_files(root, ['identity.json'] if c.get('policy_variant') == 'base' else
                  ['trained_model.json', 'checkpoint.pt', 'metadata.json'])
    trained = policy_record(c)
    seeds = c.get('eval_seeds', [None] * int(c.get('eval_episodes', 5)))
    if not seeds or len(seeds) != int(c.get('eval_episodes', len(seeds))):
        raise ValueError('Evaluation episode count must match the seed list.')
    if any(seed is not None for seed in seeds) and len(set(seeds)) != len(seeds):
        raise ValueError('Explicit evaluation seeds must be unique.')
    import uuid
    prefix = 'baseline_' if trained['policy_variant'] == 'base' else 'eval_'
    evaluation = root / (prefix + trained['checkpoint_sha256'][:12] + '_' + uuid.uuid4().hex[:8])
    evaluation.mkdir(exist_ok=False)
    config_path = evaluation / 'evaluation_config.json'
    write_json(config_path, c)
    log = (evaluation / 'server.log').open('w')
    p = subprocess.Popen([sys.executable, '-m', 'amd_benchmarks.robocasa', '--serve', str(config_path)],
                         stdout=log, stderr=subprocess.STDOUT)
    port = int(c.get('port', 8121))
    try:
        for _ in range(600):
            if p.poll() is not None:
                raise RuntimeError('Policy server exited; see server.log.')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Policy server did not become ready.')
        official_horizon = get_task_horizon(c['task'])
        horizon = int(c.get('eval_horizon', official_horizon))
        original_make = simulation.gym.make
        rows = []
        for episode, seed in enumerate(seeds):
            video_dir = evaluation / ('seed_' + str(seed) if seed is not None else f'episode_{episode:03d}')
            sim = SimulationConfig(env_name='robocasa/' + c['task'], split=c.get('eval_split', 'pretrain'),
                n_episodes=1, n_envs=1, video=VideoConfig(video_dir=str(video_dir)),
                multistep=MultiStepConfig(n_action_steps=8, max_episode_steps=horizon))
            def seeded_make(env_id, *args, **kwargs):
                if env_id == sim.env_name and seed is not None:
                    kwargs['seed'] = int(seed)
                return original_make(env_id, *args, **kwargs)
            if seed is not None:
                np.random.seed(int(seed)); random.seed(int(seed))
            client = SimulationInferenceClient(host='127.0.0.1', port=port)
            # Only the constructor seed is injected; official stepping and success logic stay unchanged.
            with patch.object(simulation.gym, 'make', seeded_make):
                _, successes = client.run_simulation(sim)
            if len(successes) != 1:
                raise RuntimeError('Incomplete per-seed evaluation.')
            row = {'episode': episode, 'seed': seed, 'success': bool(successes[0]),
                   'checkpoint_sha256': trained['checkpoint_sha256'], 'policy_variant': trained['policy_variant']}
            rows.append(row)
            with (evaluation / 'results.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        result = {'episodes': len(rows), 'successes': sum(x['success'] for x in rows),
                  'success_rate': sum(x['success'] for x in rows) / len(rows),
                  'horizon': horizon, 'official_horizon': official_horizon, 'seeds': seeds,
                  'split': c.get('eval_split', 'pretrain'),
                  'protocol': 'official simulator stepping and success; fresh environment per explicit constructor seed',
                  'policy_variant': trained['policy_variant'],
                  'validation_only': horizon != official_horizon or len(rows) < 20 or
                     (trained['policy_variant'] == 'trained' and trained['step'] < 100),
                  'checkpoint_sha256': trained['checkpoint_sha256'], 'directory': str(evaluation)}
        write_json(evaluation / 'summary.json', result)
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
    else:
        raise SystemExit('Use python -m amd_benchmarks.common --config CONFIG --stage STAGE')
