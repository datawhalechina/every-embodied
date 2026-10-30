"""Read archived task evidence and preview explicitly selected reproduction stages."""
from pathlib import Path
import html
import json
import os
import re
import subprocess


def load_record(topic, relative):
    """Read a bundled source record without contacting the original website."""
    return json.loads((Path(topic) / 'reproducibility/showcase' / relative).read_text(encoding='utf-8'))


def show_media(label, url):
    """Display the original online video with a direct source link."""
    from IPython.display import HTML, display
    display(HTML(f'<p><strong>{html.escape(label)}</strong></p><video controls preload="none" width="640" style="max-width:100%"><source src="{html.escape(url)}" type="video/mp4"></video><p><a href="{html.escape(url)}">打开原站完整视频（需要网络）</a></p>'))


def preview_stage(stage, language, script, reference_only=False):
    """Show commands; execute only the explicitly selected stage in a configured environment."""
    from IPython.display import Markdown, display
    display(Markdown(f'**阶段 `{stage}`**\n\n```{language}\n{script}\n```'))
    if reference_only:
        print('参考阶段：当前设备尚未完成完整训练验证，本单元不自动执行。')
        return
    if os.environ.get('AMD_REPRO_STAGE') != stage:
        print('未选择本阶段，不启动计算或下载。')
        return
    required = ['SRC_ROOT', 'DATA_ROOT', 'RUN_ROOT', 'ENV_ROOT', 'EXPERIMENT_PYTHON']
    missing = [key for key in required if not os.environ.get(key) or '/path/to' in os.environ[key]]
    if missing:
        raise ValueError('Configure explicit experiment paths first: ' + ', '.join(missing))
    env = dict(os.environ)
    env['PATH'] = str(Path(env['EXPERIMENT_PYTHON']).parent) + os.pathsep + env.get('PATH', '')
    env.setdefault('EVERY_EMBODIED_ROOT', str(Path(__file__).resolve().parents[4]))
    env.setdefault('DOWNLOAD_ENTRY', '/home/ubuntu/.local/bin/lab-download')
    for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'UV_OFFLINE']:
        env[key] = '1'
    if language == 'python':
        subprocess.run([env['EXPERIMENT_PYTHON'], '-c', script], env=env, check=True)
        return
    # Route explicit downloads only. Training remains on its normal runtime path.
    script = re.sub(r'(?m)^(\s*)(python(?:3(?:\.\d+)?)? -m pip |hf download |uv sync)',
                    r'\1"$DOWNLOAD_ENTRY" \2', script)
    script = re.sub(r'(?m)^(\s*)(git clone |git submodule update )',
                    r'\1"$DOWNLOAD_ENTRY" --download-exec \2', script)
    script = script.replace('python scripts/setup_submodules.py', '"$DOWNLOAD_ENTRY" --download-exec python scripts/setup_submodules.py')
    if 'hf download ' in script:
        env['HF_HUB_OFFLINE'] = '0'
    if 'uv sync' in script:
        env['UV_OFFLINE'] = '0'
    subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', script], env=env, check=True)
