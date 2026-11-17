import os
"""Same-episode navigation-only finale; all original tools remain immutable."""
import argparse,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--job-name',choices=['pipeline-v14','pipeline-v15'],default='pipeline-v14');a=p.parse_args();base=a.base
    workspace=base/'r1-workspace-v6';runtime=base/'runtime-v12';job=base/a.job_name
    if not (base/'pipeline-v13/r1-agent-process-result.json').is_file():raise RuntimeError('Cleanup policy is still running')
    if json.loads((base/'pipeline-v13/r1-agent-process-result.json').read_text())['returncode']!=0:raise RuntimeError('Inspect cleanup failure first')
    proof=json.loads((runtime/'three-cans-accepted.json').read_text())
    if not proof.get('passed') or not proof.get('official_task_complete'):raise RuntimeError('Three-can native goal not accepted')
    reference=Path(__file__).resolve().parents[1]/'aura_household_world/reference';sys.path[:0]=[str(reference),str(workspace)]
    from workspace_agent_host import WorkspaceHost
    from tools.util.protocol import request
    from .controller import run
    from .durable import DurableHost
    session=json.loads((workspace/'.session.json').read_text());sock=Path(session['socket'])
    def status():return request(sock,{'operation':'status'})
    if status()['episode_id']!=proof['episode_id']:raise RuntimeError('Episode changed before return')
    host=WorkspaceHost(workspace,job/'host',allow_execution=True,live_status=status)
    allowed={'deactivate_skill','capture_head_camera','capture_left_wrist_camera','capture_right_wrist_camera',
        'adjust_chassis','adjust_pitch','reset_body','spin_to_facing_point','move_chassis_to_floor_point','measure_shoulder_distance','track_object_distance','save_native_checkpoint'}
    host.tools={name:path for name,path in host.tools.items() if name in allowed};host=DurableHost(host)
    task='''三个饮料罐已经全部释放并稳定在原垃圾桶内，清理目标原生验收通过。现在只做返回：从厨房返回客厅，在能看见Aura或访客的一块可见空地停稳，方便Aura向人报告结果。不要重做搜索垃圾、抓取或投放。白色圆屏两轮物体是Aura，人物和Aura都是协作者。
只用当前相机检查通道，选择可见空地导航；不要把人物或Aura当作直接接近的点击目标。目标是在Aura约1.2至2.3米范围内，距访客至少1米的空地停稳；不能只因远处看见人就结束。上一段已经走到客厅入口，继续当前位置，不重走厨房路径。用当前RGB-D和measure_shoulder_distance实际测量Aura（白色圆屏两轮）的距离，再选择相邻空地前进；测量是只读，不能把协作者当成抓取目标。优先较长的可见通行段。双臂保持空手收拢，禁止操作物体。满足范围后才能done，宿主未通过时会给出公开反馈并自动继续；每次运动仍自动存档，完整公开输入、相机和回执归档。'''
    exe=os.environ.get('CODEX_EXEC','codex')
    from .return_goal import goal_status
    try:result=run(host,task,exe,qualification=runtime/'qualification-accepted.json',max_turns=30,context_mode='compact',archive_context=True,goal_guard=lambda:goal_status(runtime)['passed'])
    except Exception as error:
        (job/'host/controller-result.json').write_text(json.dumps({'status':'stopped','error_type':type(error).__name__,'message':str(error),'native_success_verified':False}));raise
    (job/'host/controller-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
