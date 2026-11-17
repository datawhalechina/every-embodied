"""Create a concrete Aura request + scene/agent launch recipe, never execute it."""
import argparse,hashlib,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def prepare(output,text='帮我捡垃圾',scene_config=None):
    from .inbox import receive,write_new
    config=json.loads(Path(scene_config or HERE/'world.json').read_text(encoding='utf-8'))
    if config.get('scene')!='house_double_floor_lower' or config.get('task_name')!='picking_up_trash':raise ValueError('Use this prepared household/task pair')
    if config['execution_enabled'] is not False:raise ValueError('Preparation must keep execution disabled')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    request=receive(text,output/'requests')
    task=json.loads((ROOT/'external/RoboHarness/tasks/task01.json').read_text())
    case=next(c for c in task['cases'] if c['instance_id']==config['instance_id'])
    prompt=ROOT/'external/RoboHarness'/case['prompt']
    if hashlib.sha256(prompt.read_bytes()).hexdigest()!=case['prompt_sha256']:raise ValueError('Task prompt changed')
    (output/'task-prompt.txt').write_bytes(prompt.read_bytes())
    write_new(output/'world.json',config)
    runtime=json.loads((ROOT/'config/runtime.json').read_text())
    remote=config['remote_root']+'/episodes/'+request['id']
    code_root=config['remote_root']+'/code'
    recipe={'request':request,'phase':'prepared_not_started','execution_enabled':False,
        'world_config':config,'remote_episode':remote,'code_root':code_root,
        'workspace_template':runtime['workspace'],'native_task_budget':task['max_steps'],
        'evaluator_python':runtime['sim_python'],'workspace_python':runtime['workspace']+'/.venv/bin/python',
        'data_path':runtime['data'],'upstream':runtime['repo'],
        'wrapper':'integrations.aura_household_world.runtime_wrapper.HouseholdRGBDWrapper',
        'views':['observer','r1_head','aura_head'],
        'roles':{'person':'same-scene animated visitor; typed request in this version',
                 'aura':'same-scene interface prop with screen and camera',
                 'r1_pro':'the single official task robot with native physics'},
        'pending_runtime_checks':['placement_clearance','human_animation_playback','camera_visibility_and_sync','actual_model_turn_and_tool_loop'],
        'task_prompt':prompt.read_text(encoding='utf-8'),
        'model_calls':0,'simulation_started':False,'physical_robot_access':False}
    write_new(output/'launch-recipe.json',recipe)
    return recipe

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--text',default='帮我捡垃圾');p.add_argument('--scene-config',type=Path);a=p.parse_args()
    recipe=prepare(a.output,a.text,a.scene_config)
    print(json.dumps({'request_id':recipe['request']['id'],'phase':recipe['phase'],'simulation_started':False,'model_calls':0},ensure_ascii=False))

if __name__=='__main__':main()
