"""Execute bounded course lessons and paired native benchmarks on the AMD host."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import nbformat
from nbclient import NotebookClient

RUN = Path('/dev/shm/amd-acceleration-20261028-v1')
COURSE = RUN / 'course'
EVIDENCE = COURSE / 'notebooks/results/acceleration'
CURRENT = EVIDENCE / 'amd-device-20261028-v1'
CURRENT.mkdir(exist_ok=True)
NATIVE_ROOT = Path('/home/aup/tmp/mujoco-warp-amd-conditional-graph-20260821')
NATIVE_PYTHON = Path('/home/aup/envs/mujoco-warp-amd-py312/bin/python')
LIBRARY = Path('/home/aup/tmp/rocm-hip-conditional-node-20260821/clr-build/hipamd/lib/libamdhip64.so.7')
ENV = os.environ.copy()
ENV.update({'MPLCONFIGDIR': str(RUN/'matplotlib'), 'JUPYTER_RUNTIME_DIR': str(RUN/'jupyter'),
            'TMPDIR': str(RUN/'tmp'), 'PYTHONDONTWRITEBYTECODE': '1'})
for key in ['MPLCONFIGDIR', 'JUPYTER_RUNTIME_DIR', 'TMPDIR']:
    Path(ENV[key]).mkdir(exist_ok=True)
os.environ.update(ENV)

import torch
assert torch.cuda.is_available() and torch.version.hip, 'Native AMD GPU required'
device = {'hostname': platform.node(), 'python': sys.version, 'torch': torch.__version__,
          'hip': torch.version.hip, 'gpu': torch.cuda.get_device_name(0),
          'client_time': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'work_mount': '/dev/shm', 'work_free_bytes': os.statvfs(RUN).f_bavail * os.statvfs(RUN).f_frsize,
          'scope': 'shared AMD device; other services left running; no training or robot task evaluation'}
assert device['work_free_bytes'] > 12 * 1024**3
(CURRENT/'device.json').write_text(json.dumps(device, indent=2)+'\n')


def read_last_json(path):
    for line in reversed(path.read_text().splitlines()):
        if line.strip().startswith('{'):
            return json.loads(line)
    raise ValueError(f'No benchmark JSON in {path}')


def native_benchmark():
    target = CURRENT / 'native'
    target.mkdir(exist_ok=True)
    env = ENV.copy()
    env.update({'PYTHONPATH': f'{NATIVE_ROOT}/upstream/warp:{NATIVE_ROOT}/upstream/mujoco_warp',
                'LD_LIBRARY_PATH': f'{LIBRARY.parent}:{NATIVE_ROOT}/upstream/warp/warp/bin:/opt/rocm/lib:/opt/rocm/lib64',
                'WARP_CACHE_PATH': str(RUN/'warp-cache'), 'MUJOCO_GL': 'egl',
                'MJW_HIP_SINGLE_STREAM': '1', 'WP_HIP_GRAPH_ENABLE': '0',
                'WP_HIP_CONDITIONAL_EMULATION': '0', 'WP_HIP_CONDITIONAL_MAX_ITERS': '100'})
    model = NATIVE_ROOT/'upstream/mujoco_warp/mujoco_warp/test_data/humanoid/humanoid.xml'
    records = []
    for repeat in range(3):
        for variant in (['native', 'eager'] if repeat % 2 == 0 else ['eager', 'native']):
            env['WP_HIP_CONDITIONAL_NATIVE'] = '1' if variant == 'native' else '0'
            log = target / f'{variant}-{repeat}.log'
            command = [str(NATIVE_PYTHON), '-m', 'mujoco_warp.testspeed', str(model),
                       '--function=step', '--nworld=1024', '--nstep=1000',
                       '--nconmax=128', '--njmax=128', '--device=cuda:0', '--format=json']
            start = time.perf_counter()
            with log.open('w') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=600)
            row = read_last_json(log)
            row.update({'variant': variant, 'repeat': repeat,
                        'process_wall_s_including_setup': time.perf_counter()-start,
                        'command': command})
            records.append(row)
            (target/'timing-records.json').write_text(json.dumps(records, indent=2)+'\n')
            print('Native benchmark', variant, repeat, row.get('run_time'), flush=True)
    states = {}
    for variant in ['eager', 'native']:
        env['WP_HIP_CONDITIONAL_NATIVE'] = '1' if variant == 'native' else '0'
        state_path = target / f'{variant}-state.json'
        with (target/f'{variant}-state.log').open('w') as stream:
            subprocess.run([str(NATIVE_PYTHON), str(NATIVE_ROOT/'scripts/run_mjwarp_variant.py'),
                            str(model), '--variant', variant, '--worlds', '256', '--steps', '100',
                            '--output', str(state_path)], env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=600)
        states[variant] = json.loads(state_path.read_text())
    import numpy as np
    errors = {}
    for field in ['qpos', 'qvel', 'time']:
        left, right = (np.asarray(states[x][field]) for x in ['native', 'eager'])
        assert left.shape == right.shape
        assert np.isfinite(left).all() and np.isfinite(right).all()
        errors[field] = float(np.max(np.abs(left-right)))
    median = {v: float(np.median([r['run_time'] for r in records if r['variant'] == v])) for v in ['eager','native']}
    result = {'executed': True, 'device': device, 'performance_worlds': 1024,
              'performance_steps': 1000, 'timing_repeats_each': 3,
              'paired_state_worlds': 256, 'paired_state_steps': 100,
              'median_run_s': median, 'speed_ratio': median['eager']/median['native'],
              'max_abs_state_errors': errors, 'correctness_tolerance': 1e-5,
              'correctness_passed': max(errors.values()) <= 1e-5,
              'runtime_file': str(LIBRARY.resolve()),
              'runtime_sha256': hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
              'records': records}
    (CURRENT/'native-result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


records = []
for path in sorted((COURSE/'07-推理与仿真加速/03-Notebook').glob('*.ipynb')):
    book = nbformat.read(path, as_version=4)
    # Refresh only this new small exercise group; previous competition notebooks remain untouched.
    if path.name.startswith('04_'):
        try:
            native = native_benchmark()
        except Exception as exc:
            native = {'executed': False, 'error_type': type(exc).__name__, 'error': str(exc),
                      'scope': 'current attempt failed; historical results kept separately'}
            (CURRENT/'native-result.json').write_text(json.dumps(native, indent=2)+'\n')
        insertion = [nbformat.v4.new_markdown_cell(
            '## 本次原生设备实测\n\n本节显示本次设备实际执行的对照，历史结果在后文单独保留。'
            '主基准为1024世界、1000步，每种变体3次交替计时；状态检查为256世界、100步。'
            '共享设备的其他服务保持运行。'),
            nbformat.v4.new_code_cell('''current = json.loads((EVIDENCE / "amd-device-20261028-v1/native-result.json").read_text())
display(current if not current.get("executed") else {k:v for k,v in current.items() if k != "records"})
if current.get("executed"):
    values = current["median_run_s"]
    fig, ax = plt.subplots()
    ax.bar(values.keys(), values.values(), color=["#526b87", "#238267"])
    ax.bar_label(ax.containers[0], fmt="%.4f", padding=3)
    ax.set(ylabel="Steady workload time (s)", title="Current AMD: 1024 worlds x 1000 steps", ylim=(0,max(values.values())*1.2))
    fig.tight_layout(); plt.show()
''')]
        # The first code cell establishes COURSE and EVIDENCE.
        first_code = next(i for i,c in enumerate(book.cells) if c.cell_type == 'code')
        book.cells[first_code+1:first_code+1] = insertion
    first_code = next(c for c in book.cells if c.cell_type == 'code')
    first_code.source += '\nprint(json.loads((EVIDENCE / "amd-device-20261028-v1/device.json").read_text()))\n'
    client = NotebookClient(book, timeout=240, kernel_name='python3',
                            resources={'metadata': {'path': str(path.parent)}})
    client.execute()
    nbformat.validate(book)
    cells = [c for c in book.cells if c.cell_type == 'code']
    assert [c.execution_count for c in cells] == list(range(1,len(cells)+1))
    assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
    metadata = book.metadata.setdefault('amd_course', {})
    metadata.update({'exercise_execution_complete': True, 'execution_interpreter': sys.executable,
                     'execution_host': platform.node(), 'execution_backend': 'AMD ROCm',
                     'device_evidence': 'notebooks/results/acceleration/amd-device-20261028-v1/device.json'})
    path.write_text(json.dumps(book, ensure_ascii=True, indent=1)+'\n')
    measures = []
    for cell in cells:
        for output in cell.outputs:
            text = ''.join(output.get('data', {}).get('text/plain', []))
            if "'speed_ratio':" in text:
                try:
                    measures.append(ast.literal_eval(text))
                except (ValueError, SyntaxError):
                    pass
    row = {'file': path.name, 'executed_code_cells':len(cells), 'measurements': measures,
           'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    records.append(row)
    (CURRENT/'notebook-execution.json').write_text(json.dumps(records, ensure_ascii=True, indent=2)+'\n')
    print('NOTEBOOK COMPLETE', path.name, flush=True)
(RUN/'COMPLETE.json').write_text(json.dumps({'device':device, 'notebooks':records}, indent=2)+'\n')
