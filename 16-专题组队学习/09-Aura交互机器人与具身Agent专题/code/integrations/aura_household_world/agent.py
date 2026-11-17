"""Wire the existing visual Coding Agent into Aura feedback, without auto-run."""
import datetime,hashlib,importlib,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def public_event(root,kind,**payload):
    folder=Path(root)/'public-events';folder.mkdir(parents=True,exist_ok=True)
    number=len(list(folder.glob('*.json')))
    value={'kind':kind,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),**payload}
    with (folder/f'{number:06d}.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def publish_aura(root,phase,reply,**extra):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    value={'phase':phase,'reply':reply,'voice_source':'none_not_synthesized',**extra}
    temp=root/'aura-state.tmp';temp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temp.replace(root/'aura-state.json')


def require_execution(config,*,explicit_run=False):
    if not explicit_run or config.get('execution_enabled') is not True:
        raise PermissionError('当前为场景与代码准备阶段，不启动模型、仿真或家务动作')
    if config.get('placement',{}).get('clearance_verified') is not True:
        raise PermissionError('同场景人物／Aura摆位尚未经过实际碰撞与可见性检查')


def make_agent(workspace,evidence,task_text,config,live_status,*,explicit_run=False):
    require_execution(config,explicit_run=explicit_run)
    sys.path.insert(0,str(HERE/'reference'))
    from workspace_agent_host import WorkspaceHost
    from astra_workspace_client import WorkspaceAgent
    evidence=Path(evidence)
    host=WorkspaceHost(workspace,evidence/'host',allow_execution=True,live_status=live_status)
    # Reuses exact current-image binding, one tool/turn, execution ledger and
    # same-thread recovery from the installed reference implementation.
    class HouseholdAgent(WorkspaceAgent):
        def decision(self):
            publish_aura(evidence,'working','正在观察家庭场景并清理垃圾。')
            outcome=super().decision()
            public_event(evidence,'agent_decision',turn=outcome['turn_number'],
                sequence=outcome['observation_sequence'],tool_execution_ids=outcome['tool_execution_ids'],
                agent_requests_finish=outcome['done'])
            latest=Path(workspace)/'perception/latest.json'
            if latest.exists():
                # Camera metadata are actual source records, not generated events.
                row=json.loads(latest.read_text())
                public_event(evidence,'camera_feedback',episode_id=row['episode_id'],
                    sequence=row['sequence'],head_image_id=row['head_image_id'],images=row['images'])
            if outcome['done']:
                publish_aura(evidence,'verifying','动作流程已结束，正在核对实际任务结果。')
            return outcome
    return HouseholdAgent(host,task_text,max_turns=config['max_model_turns'])


def close_with_native_result(evidence,result_path):
    from .inbox import verify_native_result
    evidence=Path(evidence).resolve();source=Path(result_path).resolve()
    if not source.is_relative_to(evidence.parent/'native-output'):
        raise ValueError('Native verification must use this episode output, not an older result')
    raw=source.read_bytes();outcome=verify_native_result(json.loads(raw))
    public_event(evidence,'native_verification',source_sha256=hashlib.sha256(raw).hexdigest(),**outcome)
    publish_aura(evidence,outcome['phase'],outcome['reply'],native_result=str(source),native_success=outcome['native_success'])
    return outcome
