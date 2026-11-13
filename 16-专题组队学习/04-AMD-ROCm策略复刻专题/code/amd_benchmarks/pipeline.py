"""Run complete benchmark recipes serially and preserve restartable stage state."""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
from .common import write_json, memory_gate


def read(path):
    return json.loads(Path(path).read_text())


def expected_training_step(config, result):
    return result.get('step', result.get('trained_steps_target')) == config['steps']


def completed(config, stage):
    root = Path(config['run_root'])
    status = root / (stage + '_status.json')
    if not status.is_file():
        return False
    record = read(status)
    if record.get('state') != 'complete':
        return False
    if stage == 'train':
        return (root / 'trained_model.json').is_file() and expected_training_step(config, read(root / 'trained_model.json'))
    if stage in {'baseline', 'evaluate'}:
        result = record['result']
        if result.get('evaluation_clock', 'official_async') != config.get('evaluation_clock', 'official_async'):
            return False
        if config.get('policy_loader_contract') and result.get('loader_contract') != config['policy_loader_contract']:
            return False
        if bool(result.get('warmup_policy', False)) != bool(config.get('warmup_policy', False)):
            return False
        if result.get('seeds') != config['eval_seeds'] or result.get('validation_only', True):
            return False
        if stage == 'evaluate':
            return result['checkpoint_sha256'] == read(root / 'trained_model.json')['checkpoint_sha256']
    return True


def validate_final(config):
    root = Path(config['run_root'])
    model = read(root / 'trained_model.json')
    assert expected_training_step(config, model)
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    assert [int(row['step']) for row in rows] == list(range(1, config['steps'] + 1))
    assert all(math.isfinite(row['loss']) for row in rows)
    info = read(Path(config['dataset_root']) / 'meta/info.json')
    if config['benchmark'] == 'dexjoco':
        prepared = read(root / 'prepared.json')
        assert prepared['normalization_frames'] is None
        assert prepared['normalization_frames_processed'] == info['total_frames']
    else:
        prepared = read(root / 'prepared/prepare_manifest.json')
        assert prepared['episodes'] == info['total_episodes']
        assert prepared['frames'] == info['total_frames']
    baseline = read(root / 'baseline_status.json')['result']
    final = read(root / 'evaluate_status.json')['result']
    assert baseline['policy_variant'] == 'base' and final['policy_variant'] == 'trained'
    assert baseline.get('evaluation_clock', 'official_async') == final.get('evaluation_clock', 'official_async') == config.get('evaluation_clock', 'official_async')
    if config.get('policy_loader_contract'):
        assert baseline['loader_contract'] == final['loader_contract'] == config['policy_loader_contract']
    assert bool(baseline.get('warmup_policy', False)) == bool(final.get('warmup_policy', False)) == bool(config.get('warmup_policy', False))
    assert baseline['episodes'] == final['episodes'] == len(config['eval_seeds'])
    assert not baseline['validation_only'] and not final['validation_only']
    base_rows = [json.loads(line) for line in (Path(baseline['directory']) / 'results.jsonl').read_text().splitlines()]
    final_rows = [json.loads(line) for line in (Path(final['directory']) / 'results.jsonl').read_text().splitlines()]
    assert [row['seed'] for row in base_rows] == [row['seed'] for row in final_rows] == config['eval_seeds']
    result = {'task': config['task'], 'steps': config['steps'], 'metrics_rows': len(rows),
              'baseline': baseline, 'trained': final,
              'improved_seeds': [a['seed'] for a, b in zip(base_rows, final_rows) if not a['success'] and b['success']],
              'regressed_seeds': [a['seed'] for a, b in zip(base_rows, final_rows) if a['success'] and not b['success']]}
    write_json(root / 'formal_comparison.json', result)
    return result


def run(manifest_path):
    manifest = read(manifest_path)
    if manifest.get('prerequisite_status'):
        prerequisite = read(manifest['prerequisite_status'])
        assert prerequisite['state'] == 'complete' and prerequisite['execution_counts'] == list(range(1, 11))
    job = Path(manifest['job_root'])
    job.mkdir(parents=True, exist_ok=True)
    lock = (job / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'state': 'running', 'supervisor_pid': os.getpid(), 'started_unix': time.time(),
             'manifest': str(manifest_path), 'completed_stages': []}
    write_json(job / 'pipeline_state.json', state)
    env = os.environ.copy()
    env['PYTHONPATH'] = manifest['code_root']
    results = []
    try:
        for recipe in manifest['runs']:
            config = read(recipe['config'])
            root = Path(config['run_root'])
            root.mkdir(parents=True, exist_ok=True)
            for stage in manifest.get('stages', ['preflight', 'prepare', 'train', 'baseline', 'evaluate']):
                label = recipe['name'] + '/' + stage
                if completed(config, stage):
                    state['completed_stages'].append(label)
                    print('Reusing completed stage:', label, flush=True)
                    continue
                actual = dict(config)
                if stage == 'train':
                    native = root / 'checkpoints' / config['task'] / 'course'
                    actual['resume'] = (root / 'checkpoint.pt').is_file() or any(
                        path.is_dir() and path.name.isdigit() for path in native.glob('*'))
                config_file = root / ('pipeline_' + stage + '_config.json')
                write_json(config_file, actual)
                state.update(current_run=recipe['name'], current_stage=stage, child_pid=None,
                             stage_started_unix=time.time(), log=str(root / (stage + '.log')))
                write_json(job / 'pipeline_state.json', state)
                minimum_ram = float(config.get('minimum_available_ram_gib', 0))
                while memory_gate(0) < minimum_ram:
                    state.update(state='waiting_resources', available_ram_gib=memory_gate(0),
                                 required_ram_gib=minimum_ram, heartbeat_unix=time.time())
                    write_json(job / 'pipeline_state.json', state)
                    time.sleep(30)
                state.update(state='running', available_ram_gib=memory_gate(0))
                write_json(job / 'pipeline_state.json', state)
                command = [recipe['python'], '-u', '-m', 'amd_benchmarks.common',
                           '--config', str(config_file), '--stage', stage]
                print('Starting stage:', label, flush=True)
                with (root / (stage + '.log')).open('a') as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = process.pid
                    while True:
                        state['heartbeat_unix'] = time.time()
                        write_json(job / 'pipeline_state.json', state)
                        try:
                            code = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                if code:
                    raise RuntimeError(f'{label} exited {code}; inspect {root / (stage + ".log")}')
                if not completed(config, stage):
                    raise RuntimeError(f'{label} did not satisfy the formal stage contract.')
                state['completed_stages'].append(label)
                write_json(job / 'pipeline_state.json', state)
            results.append(validate_final(config))
        state.update(state='complete', finished_unix=time.time(), child_pid=None,
                     results_file=str(job / 'formal_results.json'))
        write_json(job / 'formal_results.json', results)
        write_json(job / 'pipeline_state.json', state)
    except BaseException as exc:
        state.update(state='failed', error=str(exc), failed_unix=time.time())
        write_json(job / 'pipeline_state.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    run(parser.parse_args().manifest.resolve())
