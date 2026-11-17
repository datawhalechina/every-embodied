"""Future fresh workspace creation, invoked only after the owned driver is ready."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--session-id',required=True);a=p.parse_args()
    sys.path.insert(0,str(a.template))
    from tools.util.session import create
    create(a.template,a.workspace,a.socket,session_id=a.session_id,task_name='picking_up_trash',gpu=0)
    env=dict(os.environ,ROBOT_WORKSPACE=str(a.workspace),PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    subprocess.run([str(a.workspace/'.venv/bin/python'),'tools/capture_head_camera.py','{}'],cwd=a.workspace,env=env,check=True)
    if not(a.workspace/'perception/latest.json').is_file():raise RuntimeError('Initial current camera not published')

if __name__=='__main__':main()
