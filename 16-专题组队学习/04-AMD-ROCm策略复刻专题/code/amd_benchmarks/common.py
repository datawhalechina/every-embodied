"""Provenance, stage execution and checkpoint contracts for course runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(root, patterns):
    root = Path(root)
    paths = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    if not paths:
        raise FileNotFoundError(f'No identity files under {root}: {patterns}')
    records = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
                'sha256': sha256(p)} for p in paths]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': records}


def require_files(root, names):
    missing = [str(Path(root) / p) for p in names if not (Path(root) / p).exists()]
    if missing:
        raise FileNotFoundError('Missing prerequisites:\n' + '\n'.join(missing))


def disk_gate(root, minimum_gib):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / (1 << 30)
    if free < minimum_gib:
        raise RuntimeError(f'Free disk {free:.2f} GiB < required {minimum_gib} GiB at {root}')
    return free


def memory_gate(minimum_gib):
    """Reject a new unified-memory workload when Linux available RAM is too low."""
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].split()[0]) / (1 << 20)
    if available < minimum_gib:
        raise RuntimeError(f'Available system RAM {available:.2f} GiB < required {minimum_gib} GiB. '
                           'Resolve competing memory use before starting this workload.')
    return available


def require_free_port(port):
    """Refuse an occupied endpoint before starting a checkpoint-specific server."""
    import socket
    with socket.socket() as endpoint:
        # Permit reuse after a completed server; an active listener still fails bind.
        endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            endpoint.bind(('127.0.0.1', int(port)))
        except OSError as exc:
            raise RuntimeError(f'Port {port} is occupied; choose an unused port.') from exc


def reconcile_metrics(root, committed_step):
    """Preserve abandoned attempts and retain one metric row per committed update."""
    path = Path(root) / 'metrics.jsonl'
    if not path.is_file():
        return {'retained_rows': 0, 'archived_rows': 0}
    lines = path.read_text().splitlines()
    retained = {}
    parsed = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            continue
        parsed += 1
        if int(row['step']) <= committed_step:
            retained[int(row['step'])] = row
    if len(retained) == parsed and parsed == len(lines):
        return {'retained_rows': len(retained), 'archived_rows': 0}
    backup = path.with_name(f'metrics_before_resume_{time.time_ns()}.jsonl')
    shutil.copy2(path, backup)
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(retained[k]) + '\n' for k in sorted(retained)))
    temp.replace(path)
    return {'retained_rows': len(retained), 'archived_rows': len(lines) - len(retained),
            'raw_attempt_log': str(backup)}


def bind_run(config, dataset_identity, base_identity):
    root = Path(config['run_root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract = {k: config[k] for k in ['benchmark', 'task', 'batch_size', 'learning_rate', 'seed']}
    contract.update({k: config[k] for k in ['source_root', 'dataset_root', 'base_model']})
    contract.update(dataset=dataset_identity, base_model_identity=base_identity,
                    training_recipe=config.get('training_recipe', 'upstream_lora'))
    path = root / 'identity.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != contract:
        raise RuntimeError('Run identity changed. Use a new run_root; old results will not be reused.')
    write_json(path, contract)
    return root


def torch_preflight():
    import torch
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError('This workflow requires a working ROCm PyTorch GPU, not CPU/CUDA fallback.')
    x = torch.ones((64, 64), device='cuda')
    total = float((x @ x).sum().cpu())
    assert total == 262144.0
    return {'torch': torch.__version__, 'hip': torch.version.hip,
            'device': torch.cuda.get_device_name(0), 'matmul_sum': total,
            'python': sys.version, 'hostname': os.uname().nodename}


def run_logged(command, cwd, log, env=None):
    """Run a real command, stream output, preserve failure and exit status."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as f:
        p = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in p.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            code = p.wait()
        except BaseException:
            p.terminate()
            p.wait(timeout=30)
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def stage_cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=['preflight', 'prepare', 'baseline', 'train', 'evaluate', 'summary'], required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ['source_root', 'dataset_root', 'base_model', 'run_root']:
        config[key] = str(Path(config[key]).expanduser().resolve())
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from . import dexjoco, robocasa
    module = {'dexjoco': dexjoco, 'robocasa': robocasa}[config['benchmark']]
    root = Path(config['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        identity_path = root / 'identity.json'
        if args.stage in {'baseline', 'train', 'evaluate'}:
            if not identity_path.is_file():
                raise FileNotFoundError('Run prepare first; identity.json is missing.')
            bound = json.loads(identity_path.read_text(encoding='utf-8'))
            for key in ['benchmark', 'task', 'batch_size', 'learning_rate', 'seed',
                        'source_root', 'dataset_root', 'base_model', 'training_recipe']:
                if bound.get(key) != config.get(key):
                    raise RuntimeError(f'Prepared identity mismatch for {key}; use a new run_root.')
            patterns = (['config.json', '*.safetensors', 'experiment_cfg/metadata.json']
                        if config['benchmark'] == 'robocasa' else ['params/**/*', 'assets/**/*.json'])
            if fingerprint(config['base_model'], patterns) != bound['base_model_identity']:
                raise RuntimeError('Frozen base weight contents changed after preparation.')
        result = getattr(module, args.stage)(config)
    except BaseException as exc:
        write_json(root / (args.stage + '_status.json'), {
            'state': 'failed', 'error': str(exc), 'started_unix': started,
            'finished_unix': time.time(), 'stage': args.stage})
        raise
    write_json(root / (args.stage + '_status.json'), {
        'state': 'complete', 'started_unix': started, 'finished_unix': time.time(),
        'stage': args.stage, 'result': result})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    stage_cli()
