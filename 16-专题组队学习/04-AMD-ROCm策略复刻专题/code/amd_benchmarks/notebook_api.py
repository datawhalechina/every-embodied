"""Standard-library notebook API; each workload uses its dedicated interpreter."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import write_json


def run_stage(config, stage, python, enabled=False):
    if not enabled:
        print(f'{stage}: disabled. No workload was executed and no result is claimed.')
        return None
    python = Path(python).expanduser()
    if not python.is_file():
        raise FileNotFoundError(f'Interpreter not found: {python}')
    config_path = Path(config['run_root']) / 'notebook_config.json'
    write_json(config_path, config)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    command = [str(python), '-u', '-m', 'amd_benchmarks.common', '--config', str(config_path), '--stage', stage]
    log = Path(config['run_root']) / (stage + '.log')
    with log.open('a', encoding='utf-8') as f:
        p = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in p.stdout:
                print(line, end='', flush=True)
                f.write(line); f.flush()
            code = p.wait()
        except BaseException:
            p.terminate(); p.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    return json.loads((Path(config['run_root']) / (stage + '_status.json')).read_text())


def download_assets(config, commands, enabled=False):
    """Run explicit resource downloads through the caller's download launcher."""
    if not enabled:
        print('Resource downloads disabled; reuse existing local assets or enable this stage explicitly.')
        return
    launcher = Path(os.environ.get('DOWNLOAD_ENTRY', str(Path.home() / '.local/bin/lab-download')))
    if not launcher.is_file():
        raise FileNotFoundError('Set DOWNLOAD_ENTRY to the configured resource-download launcher before downloading.')
    for command in commands:
        subprocess.run([str(launcher), *command], check=True)


def inspect_run(config):
    root = Path(config['run_root'])
    for name in ['preflight_status.json', 'prepared.json', 'prepare_status.json', 'trained_model.json', 'evaluate_status.json']:
        p = root / name
        if p.is_file():
            print(name)
            print(json.dumps(json.loads(p.read_text()), ensure_ascii=False, indent=2)[:5000])
        else:
            print(f'{name}: not produced by this run.')


def plot_metrics(config):
    p = Path(config['run_root']) / 'metrics.jsonl'
    if not p.is_file():
        print('No training metrics from this run. Historical values are not substituted.')
        return
    records = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 3.2))
    plt.plot([r['step'] for r in records], [r['loss'] for r in records], marker='.')
    plt.title(config['benchmark'] + ': measured training loss (logged steps only)')
    plt.xlabel('Training step'); plt.ylabel('Loss'); plt.grid(alpha=.25)
    plt.tight_layout(); plt.show()
    print(f'{len(records)} logged updates; absent steps are not interpolated.')


def show_videos(config):
    root = Path(config['run_root'])
    status = root / 'evaluate_status.json'
    if not status.is_file():
        print('No evaluation artifact from this run.')
        return
    result = json.loads(status.read_text())
    if result['state'] != 'complete':
        raise RuntimeError('Evaluation failed; inspect the log instead of showing an old video.')
    trained = json.loads((root / 'trained_model.json').read_text())
    if result['result']['checkpoint_sha256'] != trained['checkpoint_sha256']:
        raise RuntimeError('Weights changed after evaluation. Evaluate the current checkpoint before showing its videos.')
    directory = Path(result['result']['directory'])
    videos = sorted(directory.rglob('*.mp4'))
    print(json.dumps(result['result'], ensure_ascii=False, indent=2))
    if not videos:
        raise FileNotFoundError('Evaluation completed without a video artifact.')
    from IPython.display import Video, display
    for p in videos[:3]:
        print(p.relative_to(root))
        display(Video(str(p), embed=True, width=640))
