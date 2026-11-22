"""Run bounded G0.5 interface qualification on captured BEHAVIOR observations.

This file-based worker loads the real checkpoint through the official loader.
It never controls a simulator. Requests bind captured arrays by SHA256; a
durable reservation counts every model attempt, including interrupted ones.
"""
import argparse
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
import psutil
import yaml

HERE = Path(__file__).resolve().parent
GPU_FREE_MIB = 0
CALL_LIMIT = 8
SOURCE_COMMIT = '89f2322b4ad016e192437adc1a2c253b05bab246'


from bridge import PART_WIDTHS, SEMANTICS, joint_tool_arguments
from observation_input import CAMERAS, build_input


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def gpu_headroom():
    row = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    return bool(row) and int(row[0]) >= GPU_FREE_MIB


def child_file(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('captured input path must be relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('captured input escapes this qualification directory')
    return path


def read_request(path, input_root):
    raw = path.read_bytes()
    if len(raw) > 65536:
        raise ValueError('request metadata exceeds its bound')
    request = json.loads(raw)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', request.get('request_id', '')):
        raise ValueError('invalid request identifier')
    if request.get('fixture_only') is not False or request.get('source') != 'official_allowed_observation':
        raise ValueError('model qualification requires captured official observations')
    if ('seed' in request) != ('candidate_index' in request):
        raise ValueError('seed and candidate index must be supplied together')
    seed, candidate_index = request.get('seed', 0), request.get('candidate_index', 0)
    if type(seed) is not int or not 0 <= seed < 2**32 or type(candidate_index) is not int or not 0 <= candidate_index < 3:
        raise ValueError('invalid candidate seed or index')
    entries = request.get('history', [])
    if not 1 <= len(entries) <= 6:
        raise ValueError('one to six causal captured frames are required')
    history = []
    for entry in entries:
        captured = child_file(input_root, entry['path'])
        if captured.stat().st_size > 64*1024**2:
            raise ValueError('captured frame exceeds its bound')
        blob = captured.read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError('captured observation digest mismatch')
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 64*1024**2:
                raise ValueError('decoded captured frame exceeds its bound')
        with np.load(io.BytesIO(blob), allow_pickle=False) as frame:
            sequence, step = frame['sequence'].item(), frame['control_step'].item()
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (sequence, step)):
                raise ValueError('captured sequence and physical step must be integers')
            row = {'episode_id': str(frame['episode_id'].item()),
                   'sequence': sequence, 'control_step': step,
                   'proprio': frame['proprio'].reshape(-1).tolist(),
                   'images': {camera: frame[camera].copy() for camera in CAMERAS}}
        history.append(row)
    data, binding = build_input(history, request['task'])
    if binding['episode_id'] != request['episode_id'] or binding['sequence'] != request['sequence']:
        raise ValueError('request does not match its latest captured observation')
    return request, hashlib.sha256(raw).hexdigest(), data, binding, history[-1]


class CallLedger:
    """Keep one reservation per request and count interrupted attempts."""

    def __init__(self, root, limit=CALL_LIMIT):
        self.root, self.limit = root, limit
        self.root.mkdir(parents=True, exist_ok=True)

    def reserve(self, request_id, digest):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', request_id):
            raise ValueError('invalid request identifier')
        path = self.root / (request_id + '.json')
        with (self.root/'counter.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                row = json.loads(path.read_text())
                if row['request_sha256'] != digest:
                    raise ValueError('request identifier was reused with another input')
                if row['status'] == 'completed':
                    return path, row, True
                raise RuntimeError('interrupted or failed request requires inspection; not replayed')
            count = len(list(self.root.glob('*.json')))
            if count >= self.limit:
                raise RuntimeError('qualification model-call budget exhausted')
            row = {'at': now(), 'request_id': request_id, 'request_sha256': digest,
                   'status': 'reserved', 'attempt_number': count + 1}
            with path.open('x') as stream:
                json.dump(row, stream); stream.flush(); os.fsync(stream.fileno())
        return path, row, False


class QualificationWorker:
    def __init__(self, output):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger = CallLedger(output/'calls')
        self.policy = self.processor = self.inferencer = None
        self.camera_shapes = self.fingerprint = None
        self.model_cfg = self.checkpoint_identity = None

    def status(self, phase, **fields):
        atomic_json(self.output/'status.json', {'at': now(), 'phase': phase, 'pid': os.getpid(),
            'process_create_time': psutil.Process().create_time(),
            'qualification_model_calls_reserved': len(list(self.ledger.root.glob('*.json'))),
            'call_limit': CALL_LIMIT, 'model_loaded': self.policy is not None,
            'processor_ready': self.processor is not None,
            'simulator_control': False, **fields})

    def base_configuration(self):
        from g05.utils.checkpoint.ckpt_utils import load_config_from_run_dir
        from g05.utils.config.config_resolvers import register_default_resolvers
        register_default_resolvers()
        cfg = load_config_from_run_dir(BASE, str(CHECKPOINT), [
            f'model.model_arch.hf_processor_path={PROCESSOR}',
            f'model.model_arch.AT_CONFIG.ckpt_dir={TOKENIZER}',
            f'model.model_arch.predict_cot={str(NATIVE_REASONING).lower()}', 'model.model_arch.discrete_action=false',
            'model.model_arch.continuous_action=true', 'model.model_arch.return_continuous_action=true',
            'model.use_torch_compile=false'])
        return cfg

    def configure(self, latest):
        from omegaconf import OmegaConf
        from behavior_processor import processor_configuration
        cfg=self.model_cfg if self.model_cfg is not None else self.base_configuration()
        self.camera_shapes = {name: list(latest['images'][name].transpose(2, 0, 1).shape) for name in CAMERAS}
        processor_cfg = processor_configuration(OmegaConf.to_container(cfg, resolve=True), self.camera_shapes)
        return cfg, processor_cfg

    def load_checkpoint(self):
        """Load authentic parameters without constructing observations or inferring."""
        if self.policy is not None:return
        if not gpu_headroom():
            raise BlockingIOError('waiting for the fixed GPU headroom gate')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        from g05.utils.checkpoint import checkpoint_utils as loader
        cfg = self.base_configuration()
        source_commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        if source_commit != SOURCE_COMMIT:
            raise ValueError('model source differs from the registered version')
        self.status('loading_real_checkpoint')
        checkpoint_digest = sha256(CHECKPOINT)
        if not gpu_headroom():
            raise BlockingIOError('GPU capacity changed before checkpoint loading')
        initial_free_mib=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if initial_free_mib<GPU_FREE_MIB:raise BlockingIOError('GPU changed before allocation')
        # CPU execution does not allocate CUDA memory.
        load_report = {}
        original = loader.load_state_dict_safely
        def inspected(*args, **kwargs):
            kwargs['return_info'] = True
            report = original(*args, **kwargs)
            load_report.update({key: value for key, value in report.items() if key != 'model'})
            return report['model']
        loader.load_state_dict_safely = inspected
        try:
            policy = loader.load_model_from_checkpoint(cfg.model.model_arch, str(CHECKPOINT),
                device='cpu', use_meta_device=True, eval_mode=False, extra_prefixes=['normalizer.'])
        finally:
            loader.load_state_dict_safely = original
        atomic_json(self.output/'checkpoint-load-report.json', {
            key: str(value) if key == 'mismatched_keys' else value for key, value in load_report.items()})
        missing = [key for key in load_report.get('truly_missing', [])
                   if not key.startswith(('normalizer.', 'action_tokenizer.'))]
        if missing or load_report.get('partial_loaded_keys') or load_report.get('mismatched_keys'):
            raise RuntimeError('checkpoint shape or missing-weight mismatch requires inspection')
        if cfg.model.get('model_weights_to_bf16', True):
            policy = policy.to(torch.bfloat16)
        policy.apply_fp32_params(); policy.eval()
        from tokenizer_device import align_action_tokenizer
        tokenizer_device=align_action_tokenizer(policy,'cpu')
        if bool(policy.predict_cot) != NATIVE_REASONING or policy.discrete_action or not policy.continuous_action:
            raise RuntimeError('qualification must use continuous actions and the selected reasoning mode')
        # No CUDA allocator is used by the CPU worker.
        self.policy,self.model_cfg=policy,cfg
        self.checkpoint_identity={'checkpoint_sha256':checkpoint_digest,'source_commit':source_commit,
            'pid':os.getpid(),'birth':psutil.Process().create_time(),
            'gpu_allocated_mib':0,
            'gpu_reserved_mib':0,
            'gpu_peak_allocated_mib':0,
            'gpu_free_before_load_mib':initial_free_mib,'worker_sha256':sha256(Path(__file__)),
            'action_tokenizer_placement':tokenizer_device,
            'native_reasoning':NATIVE_REASONING,'continuous_action':True,'model_inferences':0,
            'input_observations_constructed':False,'initial_gpu_free_gate_mib':GPU_FREE_MIB}
        atomic_json(self.output/'checkpoint-warmup.json',self.checkpoint_identity)
        self.status('weights_loaded_waiting_for_real_observation')

    def load(self, latest):
        self.load_checkpoint()
        import torch
        from hydra.utils import instantiate
        from g05.models.g05.inferencer import PolicyInferencer
        from g05.utils.data.normalizer import load_dataset_stats_from_json
        cfg,processor_cfg=self.configure(latest)
        processor = instantiate(processor_cfg)
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(BASE/'dataset_stats.json')['behavior_r1pro'])
        processor.eval(); processor.action_horizon = 32
        identity = {'checkpoint_sha256': self.checkpoint_identity['checkpoint_sha256'],
                    'source_commit': self.checkpoint_identity['source_commit'],
                    'config_sha256': sha256(BASE/'.hydra/config.yaml'), 'stats_sha256': sha256(BASE/'dataset_stats.json'),
                    'tokenizer_sha256': sha256(TOKENIZER), 'processor_config': processor_cfg,
                    'bridge_sha256': sha256(HERE/'bridge.py'),
                    'adaptive_policy_sha256': sha256(HERE/'adaptive_policy_service.py'),
                    'processor_adapter_sha256': sha256(HERE/'behavior_processor.py'),
                    'observation_input_sha256': sha256(HERE/'observation_input.py'),
                    'worker_sha256': sha256(Path(__file__)),
                    'tokenizer_device_adapter_sha256':sha256(HERE/'tokenizer_device.py'),
                    'gripper_domain_adapter_sha256':sha256(HERE/'action_domain.py'),
                    'joint_domain_adapter_sha256':sha256(HERE/'joint_domain.py'),
                    'hf_processor_sidecars': {path.name: sha256(path) for path in sorted(PROCESSOR.iterdir())
                        if path.is_file() and path.suffix in ('.json', '.jinja', '.model', '.txt')},
                    'torch': torch.__version__, 'inference_device': 'cpu', 'request_seed': 'per_request',
                    'native_reasoning': NATIVE_REASONING, 'action_source': 'fm', 'scope': 'manipulation_only_hold_base_torso'}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        identity['policy_fingerprint'] = self.fingerprint
        atomic_json(self.output/'policy-identity.json', identity)
        self.processor = processor
        from g05.utils.common.pytorch_utils import dict_apply
        class CPUInferencer(PolicyInferencer):
            def forward_batch(self, batch):
                batch = dict_apply(batch, lambda x: x.to('cpu') if isinstance(x, torch.Tensor) else x)
                with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                    batch = self.policy.predict_action(batch)
                return dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        self.inferencer = CPUInferencer(self.policy, processor, device='cpu')
        self.status('ready_for_captured_qualification_input')

    def process(self, request_path, input_root):
        request, digest, data, binding, latest = read_request(request_path, input_root)
        old = self.ledger.root/(request['request_id']+'.json')
        if old.exists():
            _, row, cached = self.ledger.reserve(request['request_id'], digest)
            result = self.output/row['result_file']
            if cached and sha256(result) == row['result_sha256']:
                return json.loads(result.read_text())
            raise RuntimeError('completed request result is unavailable')
        if self.processor is None:
            self.load(latest)
        shapes = {name: list(latest['images'][name].transpose(2, 0, 1).shape) for name in CAMERAS}
        if shapes != self.camera_shapes:
            raise ValueError('camera dimensions changed after processor construction')
        path, reservation, _ = self.ledger.reserve(request['request_id'], digest)
        import torch
        seed = request.get('seed', 0)
        torch.manual_seed(seed); np.random.seed(seed)
        started = time.monotonic()
        self.status('inferring', request_id=request['request_id'], **binding)
        try:
            from adaptive_policy_service import AdaptiveProprioAdapter, PromptPurifier
            original_task = data.get('task', '')
            data['task'] = PromptPurifier.purify(original_task)
            arms = set(re.findall(r'\b(left|right)\s+(?:arm|hand)\b', original_task.lower()))
            active_arm = next(iter(arms)) if len(arms) == 1 else None
            data, adaptive_state = AdaptiveProprioAdapter.adapt_observation(data, list(latest['proprio']), active_arm)
            outputs, timing = self.inferencer.infer_with_timing([data])
            outputs = AdaptiveProprioAdapter.adapt_action_output(outputs, list(latest['proprio']), adaptive_state['idle_arm'])
            action = outputs[0]
            reasoning_text = action.get('_cot_text')
            if action.get('_absent_keys') or (NATIVE_REASONING and not isinstance(reasoning_text, str)):
                raise ValueError('native reasoning or action parts are absent')
            if not NATIVE_REASONING and reasoning_text:
                raise ValueError('unexpected native reasoning in the disabled mode')
            arrays = {}
            for key, width in PART_WIDTHS.items():
                value = action[key].detach().cpu().numpy()
                if value.ndim == 3 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim != 2 or value.shape[1] != width or len(value) < 1 or not np.isfinite(value).all():
                    raise ValueError('invalid postprocessed action tensor')
                arrays[key] = value
            if len({len(value) for value in arrays.values()}) != 1:
                raise ValueError('action parts have inconsistent horizons')
            raw_file = self.output/(request['request_id']+'.actions.npz')
            with raw_file.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            parts = {key: value[0].tolist() for key, value in arrays.items()}
            from action_domain import gripper_domain
            parts,domain_changes=gripper_domain(parts)
            from joint_domain import arm_position_domain
            parts,joint_changes=arm_position_domain(parts,latest['proprio'])
            # Geometric tools own all base and torso commands in the first comparison.
            parts['lower_body'] = [0., 0., 0., *latest['proprio'][57:61]]
            proposal = {'semantics': SEMANTICS, 'policy_fingerprint': self.fingerprint,
                        'proposal_id': request['request_id'], 'episode_id': binding['episode_id'],
                        'sequence': binding['sequence'], 'action': parts}
            result = {'status': 'candidate', 'request_id': request['request_id'], 'request_sha256': digest,
                      'binding': binding, 'seed': seed, 'candidate_index': request.get('candidate_index', 0),
                      'adaptive_input': {'original_task': original_task, 'model_task': data['task'], 'state': adaptive_state},
                      'raw_action_file': raw_file.name, 'raw_action_sha256': sha256(raw_file),
                      'native_reasoning': NATIVE_REASONING, 'native_reasoning_text': reasoning_text,
                      'scope_projection': 'hold measured torso and zero base velocity',
                      'gripper_domain_projection':domain_changes,
                      'joint_domain_projection':joint_changes,
                      'model_seconds': time.monotonic()-started, 'timing_ms': timing}
            try:
                joint_tool_arguments(proposal, latest['proprio'], binding['episode_id'], binding['sequence'])
                result['proposal'] = proposal
            except ValueError as error:
                result.update(status='candidate_rejected', rejection=str(error))
            result_file = self.output/(request['request_id']+'.result.json')
            atomic_json(result_file, result)
            reservation.update(status='completed', result_file=result_file.name, result_sha256=sha256(result_file))
            atomic_json(path, reservation)
            self.status('ready_for_captured_qualification_input')
            return result
        except Exception as error:
            reservation.update(status='failed', error_type=type(error).__name__)
            atomic_json(path, reservation)
            self.status('qualification_fault', error_type=type(error).__name__)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--validate-request', action='store_true',
                        help='validate recorded input without starting model inference')
    parser.add_argument('--source', type=Path, help='GalaxeaVLA checkout')
    parser.add_argument('--base', type=Path, help='extracted g05-base directory')
    parser.add_argument('--processor', type=Path, help='processor sidecar directory')
    parser.add_argument('--tokenizer', type=Path, help='action tokenizer .pt file')
    parser.add_argument('--output', type=Path, help='new worker output directory')
    parser.add_argument('--reasoning', choices=['on', 'off'], default='on')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--warmup', action='store_true',help='Load real weights before simulation, without input or inference')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--input-root', type=Path)
    args = parser.parse_args()
    if args.validate_request:
        if not args.request or not args.input_root:
            parser.error('--validate-request requires --request and --input-root')
        _, _, _, binding, _ = read_request(args.request, args.input_root)
        print(json.dumps(binding, indent=2))
        return
    if not all((args.source, args.base, args.processor, args.tokenizer, args.output)):
        parser.error('model operation requires --source, --base, --processor, --tokenizer and --output')
    global SOURCE, BASE, CHECKPOINT, PROCESSOR, TOKENIZER, OUTPUT, NATIVE_REASONING
    SOURCE, BASE, PROCESSOR, TOKENIZER, OUTPUT = (
        value.resolve() for value in (args.source, args.base, args.processor, args.tokenizer, args.output)
    )
    CHECKPOINT = BASE / 'checkpoints/model_state_dict.pt'
    NATIVE_REASONING = args.reasoning == 'on'
    args.input_root = (args.input_root or OUTPUT / 'inputs').resolve()
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE)]
    worker = QualificationWorker(OUTPUT)
    with (OUTPUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.preflight:
            import importlib
            modules = ['g05.models.g05.g05_policy_qwen35', 'g05.models.g05.inferencer', 'behavior_processor']
            for name in modules:
                importlib.import_module(name)
            for path in (CHECKPOINT, BASE/'.hydra/config.yaml', BASE/'dataset_stats.json', TOKENIZER, PROCESSOR/'config.json'):
                if not path.is_file():
                    raise FileNotFoundError(str(path))
            worker.status('preflight_passed', real_model_loading_authorized_now=gpu_headroom(),
                imports=modules, checkpoint_bytes=CHECKPOINT.stat().st_size, model_inferences=0)
            return
        if args.request:
            worker.process(args.request, args.input_root)
            return
        if not args.serve:
            parser.error('choose --preflight, --request, or --serve')
        if args.warmup:worker.load_checkpoint()
        inbox = args.input_root/'requests'
        inbox.mkdir(parents=True, exist_ok=True)
        worker.status('waiting_for_captured_input')
        while True:
            phase = 'ready_for_captured_qualification_input' if worker.policy is not None else 'waiting_for_captured_input'
            for path in sorted(inbox.glob('*.json')):
                try:
                    worker.process(path, args.input_root)
                except BlockingIOError:
                    phase = 'waiting_for_gpu_headroom'
                    break
                except Exception as error:
                    worker.status('qualification_fault', error_type=type(error).__name__)
                    raise
            if len(list(worker.ledger.root.glob('*.json'))) >= CALL_LIMIT:
                worker.status('qualification_call_budget_consumed')
                return
            worker.status(phase, minimum_free_mib=GPU_FREE_MIB)
            time.sleep(30)


if __name__ == '__main__':
    main()
