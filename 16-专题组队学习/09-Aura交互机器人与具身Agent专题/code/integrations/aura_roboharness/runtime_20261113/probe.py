"""Read-only environment discovery, usable on Ubuntu without extra packages.

No simulator imports, model requests, policy-port connections or process control.
The environment and code are inspected separately: file presence is not a run.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def execute(args, env=None, timeout=30):
    p = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=timeout, env=env)
    return {'exit_code': p.returncode, 'stdout': p.stdout[-12000:], 'stderr': p.stderr[-2000:]}


def active_simulators(proc=Path('/proc')):
    """Return only IDs and entry names; never expose argv or process secrets."""
    result = []
    names = {'native_counter_entry.py', 'native_counter_entry_compact_v2.py',
             'official_evaluator_entrypoint.py', 'evaluator_entry.py'}
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            args = (item/'cmdline').read_bytes().split(b'\0')
            text = [a.decode('utf-8', 'replace') for a in args if a]
            # Shell command text does not count as an evaluator process.
            if not text or not Path(text[0]).name.startswith('python'):
                continue
            entry = next((Path(a).name for a in text[1:3] if Path(a).name in names), None)
            if not entry and '-m' in text[:3] and any(a.endswith('official_evaluator_entrypoint') for a in text[:4]):
                entry = 'official_evaluator_entrypoint'
            if entry:
                result.append({'pid': int(item.name), 'entry': entry})
        except (OSError, ProcessLookupError):
            continue
    return result


def inspect(profile):
    repo, work, data = (Path(profile[k]) for k in ('repo', 'workspace', 'data'))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    # Package metadata and find_spec do not initialize SimulationApp or GPU kernels.
    code = """import importlib.util,importlib.metadata,json,sys
names={'numpy':'numpy','torch':'torch','isaacsim':'isaacsim','omnigibson':'omnigibson','cv2':'opencv-python-headless','scipy':'scipy','websockets':'websockets','PIL':'pillow','mcp':'mcp'}
out={'python':sys.executable,'version':sys.version.split()[0],'modules':{}}
for module,package in names.items():
 try: version=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError: version=None
 out['modules'][module]={'available':importlib.util.find_spec(module) is not None,'distribution_version':version}
print(json.dumps(out))"""
    runtimes = {}
    for role, python in [('workspace', work/'.venv/bin/python'), ('evaluator', Path(profile['sim_python']))]:
        if python.is_file():
            result = execute([str(python), '-B', '-c', code], env)
            runtimes[role] = {'exists': True, **result}
            if result['exit_code'] == 0:
                runtimes[role]['metadata'] = json.loads(result['stdout'])
        else:
            runtimes[role] = {'exists': False}
    files = {}
    for relative in ('capture_driver.py', 'observation_capture.py', 'observation_input.py',
                     'tools/util/driver.py', 'tools/util/observation.py', 'tools/read_proprioception.py'):
        path = work/relative
        files[relative] = {'exists': path.is_file()}
        if path.is_file():
            files[relative]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    commits = {}
    for label, path in [('source', repo), ('behavior', repo/'BEHAVIOR')]:
        result = execute(['git', '-C', str(path), 'rev-parse', 'HEAD'])
        commits[label] = result['stdout'].strip() if not result['exit_code'] else None
    # Read LISTEN metadata only. Never connect to the other task's policy socket.
    listeners = execute(['ss', '-H', '-ltn'])
    occupied = []
    for line in listeners['stdout'].splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try: occupied.append(int(parts[3].rsplit(':', 1)[1]))
            except ValueError: pass
    gpu = execute(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                   '--format=csv,noheader,nounits'])
    active = active_simulators()
    paths = {name: (data/name).exists() for name in
             ('behavior-1k-assets', 'omnigibson-robot-assets', '2026-challenge-task-instances', 'omnigibson.key')}
    code_ready = all(x['exists'] for x in files.values())
    required = {'workspace': ('numpy','torch','cv2','scipy','websockets','PIL'),
                'evaluator': ('torch','isaacsim','omnigibson','numpy','websockets')}
    runtimes_ready = all(x.get('exit_code') == 0 and
        all(x.get('metadata',{}).get('modules',{}).get(m,{}).get('available')
            for m in required[role]) for role,x in runtimes.items())
    return {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'reference_thread': profile['reference_thread'], 'profile': profile,
            'source_commits': commits, 'runtimes': runtimes, 'workspace_files': files,
            'data_entries_present': paths, 'key_contents_read': False,
            'gpu': gpu, 'active_simulators': active, 'occupied_tcp_ports': sorted(set(occupied)),
            'environment_paths_verified': code_ready and runtimes_ready and all(paths.values()),
            'execution_started': False, 'model_calls': 0,
            'note': 'Path/package checks only. No simulator, tool action or household task was run.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(json.loads(args.profile.read_text(encoding='utf-8'))), ensure_ascii=False, indent=2))
