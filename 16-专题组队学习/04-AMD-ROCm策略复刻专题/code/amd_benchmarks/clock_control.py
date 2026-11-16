"""Materialize an auditable client that pauses physics only while waiting for inference."""
from pathlib import Path
import hashlib
import json
import shutil


def prepare_reused_run(config, destination):
    """Create an independent evaluation directory referencing the existing trained checkpoint."""
    origin, destination = Path(config['run_root']), Path(destination)
    for name in ['identity.json', 'prepared.json', 'trained_model.json', 'metrics.jsonl']:
        if not (origin / name).is_file():
            raise FileNotFoundError(origin / name)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ['identity.json', 'prepared.json', 'trained_model.json', 'metrics.jsonl', 'training_measurements.json']:
        if (origin / name).is_file():
            shutil.copy2(origin / name, destination / name)
    for path in (origin / 'assets').rglob('*.json'):
        target = destination / path.relative_to(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    result = dict(config, run_root=str(destination), evaluation_clock='pause_while_waiting',
                  training_origin=str(origin))
    (destination / 'training_origin.json').write_text(json.dumps({
        'origin': str(origin), 'training_repeated': False,
        'checkpoint_sha256': json.loads((origin / 'trained_model.json').read_text())['checkpoint_sha256']}, indent=2)+'\n')
    return result


def materialize_client(source, destination):
    source, destination = Path(source), Path(destination)
    original = source.read_text(encoding='utf-8')
    text = original.replace('from .dexjoco_openpi_env import DexJoCoOpenPIEnv',
                            'from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv')
    old = '''                else:
                    pressed_digits = env.stay(continue_stay=in_stay_state)
                    in_stay_state = True
'''
    if text.count(old) != 1:
        raise ValueError('Official client waiting branch changed; review before applying the clock adapter.')
    replacement = '''                else:
                    if not inference_proc.is_alive():
                        raise RuntimeError('Inference process exited while physics was paused.')
                    if time.monotonic() - clock_last_action_wall > 120:
                        raise TimeoutError('Inference did not deliver an action within 120 seconds.')
                    clock_wait_started = time.monotonic()
                    time.sleep(0.001)
                    clock_wait_seconds += time.monotonic() - clock_wait_started
                    continue
'''
    text = text.replace(old, replacement)
    anchor = '            # Episode loop.\n'
    if text.count(anchor) != 1:
        raise ValueError('Official episode loop anchor changed.')
    text = text.replace(anchor, '            clock_wait_seconds = 0.0\n'
        '            clock_started_wall = time.monotonic()\n'
        '            clock_last_action_wall = clock_started_wall\n' + anchor)
    text = text.replace('                    pressed_digits = env.step(action)\n',
        '                    pressed_digits = env.step(action)\n'
        '                    clock_last_action_wall = time.monotonic()\n', 1)
    text = text.replace('                if env.is_done:\n',
        '                if env.is_done:\n'
        '                    import json\n'
        '                    (output_dir / "clock_audit.json").write_text(json.dumps({\n'
        '                        "evaluation_clock": "pause_while_waiting", "physics_steps": timestamp,\n'
        '                        "inference_wait_seconds": clock_wait_seconds,\n'
        '                        "wall_seconds": time.monotonic() - clock_started_wall,\n'
        '                        "success": bool(env.is_success)}))\n', 1)
    text += '\nif __name__ == "__main__":\n    import tyro\n    tyro.cli(main)\n'
    compile(text, str(destination), 'exec')
    destination.write_text(text, encoding='utf-8')
    record = {'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
              'client_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'evaluation_clock': 'pause_while_waiting',
              'contract': 'Official action merging, timestamps, task stepping and termination preserved. '
                          'Physics and its timestamp stop when the action buffer is empty; wall time continues. '
                          'This is a separate deployment condition, not the official asynchronous score.'}
    destination.with_suffix('.source.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
