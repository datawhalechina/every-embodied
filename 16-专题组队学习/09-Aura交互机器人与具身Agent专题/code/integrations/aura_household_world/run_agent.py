"""Entry for a future owned evaluator session. Default performs preparation only."""
import argparse,json,signal,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--workspace',type=Path);p.add_argument('--evidence',type=Path)
    p.add_argument('--task-file',type=Path);p.add_argument('--run',action='store_true')
    a=p.parse_args();config=json.loads(a.config.read_text())
    if not a.run:
        print(json.dumps({'phase':'prepared_not_started','model_calls':0,'simulation_started':False,
            'same_world_scene':config['scene'],'task':config['task_name'],'execution_enabled':config['execution_enabled']}))
        return 0
    from .agent import require_execution,make_agent,public_event,publish_aura
    require_execution(config,explicit_run=True)
    def terminated(signum,frame):raise KeyboardInterrupt('Owned agent stopping')
    signal.signal(signal.SIGTERM,terminated)  # Lets WorkspaceAgent close its private model subprocess.
    if not all([a.workspace,a.evidence,a.task_file]):raise ValueError('Owned workspace, evidence and task prompt are required')
    sys.path.insert(0,str(a.workspace))
    from tools.util.protocol import request
    session=json.loads((a.workspace/'.session.json').read_text())
    def live_status():return request(Path(session['socket']),{'operation':'status'})
    instance=make_agent(a.workspace,a.evidence,a.task_file.read_text(),config,live_status,explicit_run=True)
    public_event(a.evidence,'agent_started',session_id=session['session_id'],episode_id=session['episode_id'])
    result=instance.run()
    # Model completion is only a request to stop. Official result verifies success.
    publish_aura(a.evidence,'verifying','正在核对实际任务结果。')
    request(Path(session['socket']),{'operation':'finish'})
    print(json.dumps({'phase':'verifying','model_requests_finish':bool(result['done']),
        'household_success_verified':False}))
    return 0

if __name__=='__main__':sys.exit(main())
