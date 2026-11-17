"""Explicit Sol exec entry, on an already qualified and bound owned episode."""
import argparse,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--codex',type=Path,required=True);p.add_argument('--max-turns',type=int,default=80)
    p.add_argument('--task',required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--context-mode',choices=['legacy','compact'],default='legacy')
    p.add_argument('--goal-macros',action='store_true');p.add_argument('--native-completion',type=Path)
    p.add_argument('--canonical-placement',action='store_true')
    p.add_argument('--durable-checkpoints',action='store_true')
    p.add_argument('--archive-public-context',action='store_true')
    p.add_argument('--completion-scope',choices=['one_can','three_cans'],default='one_can');a=p.parse_args()
    # Reference host remains unchanged. Numerical dependencies are the workspace's own.
    reference=Path(__file__).resolve().parents[1]/'aura_household_world/reference'
    sys.path.insert(0,str(reference));sys.path.insert(0,str(a.workspace))
    from tools.util.protocol import request
    from workspace_agent_host import WorkspaceHost
    session=json.loads((a.workspace/'.session.json').read_text())
    def status():return request(Path(session['socket']),{'operation':'status'})
    from .controller import run
    host=WorkspaceHost(a.workspace,a.evidence,allow_execution=True,live_status=status)
    if a.canonical_placement:
        from .placement_contract import PlacementHost
        host=PlacementHost(host)
    if a.goal_macros:
        from .macros import MacroHost
        host=MacroHost(host)
    if a.durable_checkpoints:
        from .durable import DurableHost
        host=DurableHost(host)
    try:
        result=run(host,a.task,str(a.codex),qualification=a.qualification,max_turns=a.max_turns,resume=a.resume,context_mode=a.context_mode,completion_file=a.native_completion,completion_scope=a.completion_scope,archive_context=a.archive_public_context)
    except Exception as error:
        (a.evidence/'controller-result.json').write_text(json.dumps({'status':'stopped','error_type':type(error).__name__,'message':str(error)[:1000],'native_success_verified':False},ensure_ascii=False,indent=2))
        raise
    (a.evidence/'controller-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
