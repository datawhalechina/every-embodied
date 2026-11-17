"""Future single-world launcher. No execution without both explicit/config gates.

This module is authored for review; this release's world.json disables running.
Only the child process groups created by this launcher can be stopped here.
"""
import argparse,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path


def stop_owned(p):
    if p is not None and p.poll() is None:
        try:
            if os.getpgid(p.pid)!=p.pid:raise RuntimeError('Owned process group changed')
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
        except ProcessLookupError:pass


def launch(config,recipe,*,explicit_run=False):
    from .agent import require_execution,close_with_native_result,publish_aura
    require_execution(config,explicit_run=explicit_run)  # Before imports, auth or Popen.
    from integrations.aura_roboharness.runtime_20261113.probe import active_simulators
    if active_simulators():raise RuntimeError('Another simulation is active; left untouched')
    import fcntl,shutil
    import socket
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        check.bind(('127.0.0.1',config['ports']['policy']))  # No connection to any existing policy server.
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(free.splitlines()[0])<49152:raise RuntimeError('Insufficient spare GPU memory; no other task stopped')
    base=Path(config['remote_root']);base.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(base).free<64*1024**3:raise RuntimeError('Insufficient spare disk capacity')
    with (base/'active-world.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _launch(config,recipe)


def _launch(config,recipe):
    from .agent import close_with_native_result,publish_aura
    root=Path(recipe['remote_episode']);root.mkdir(parents=True,exist_ok=False)
    launch_time=time.time()
    upstream=Path(recipe['upstream']);code=Path(recipe['code_root'])
    config_path=root/'runtime-world.json';config_path.write_text(json.dumps(config))
    evidence=root/'agent';workspace=root/'workspace';socket=Path('/tmp')/('aura-hh-'+uuid.uuid4().hex[:12])/'control.sock'
    python=recipe['workspace_python'];port=config['ports']['policy']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
        AURA_HOUSEHOLD_WORLD_CONFIG=str(config_path),AURA_HOUSEHOLD_EXPLICIT_RUN='1',
        AURA_HOUSEHOLD_CAPTURE_ROOT=str(root/'camera-streams'),AURA_HOUSEHOLD_STATE_FILE=str(evidence/'aura-state.json'),
        OMNIGIBSON_DATA_PATH=recipe['data_path'],OMNIGIBSON_APPDATA_PATH=str(root/'appdata'),
        OMNIGIBSON_HEADLESS='1',OMNI_KIT_ACCEPT_EULA='YES',OMNIGIBSON_GPU_ID='0',
        BEHAVIOR_EVAL_TEST_PHYSICAL_GPU='0',BEHAVIOR_EVAL_TEST_UNMASK_CUDA='0',
        BEHAVIOR_EVAL_TEST_ENABLE_PRIVILEGED_GRASP_ORACLE='0',BEHAVIOR_EVAL_TEST_RESTORE_TASK_OBJECTS='1',
        BEHAVIOR_INTERFACE_AUTO_STORAGE_CLEANUP='0',BEHAVIOR_INTERFACE_TEXCACHE_MAX_GIB='8',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='2',
        XDG_CACHE_HOME=str(root/'cache'),CUDA_CACHE_PATH=str(root/'cache/cuda'),TORCHINDUCTOR_CACHE_DIR=str(root/'cache/inductor'))
    paths=[str(code),recipe['workspace_template'],str(upstream/'eval/behavior'),str(upstream/'BEHAVIOR/OmniGibson'),str(upstream/'BEHAVIOR/bddl3'),str(upstream/'BEHAVIOR/joylo')]
    env['PYTHONPATH']=':'.join(paths)
    children=[];logs=[];result={'phase':'starting','root':str(root)}
    def spawn(role,args,cwd=root):
        log=(root/(role+'.log')).open('x');logs.append(log)
        p=subprocess.Popen(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    try:
        publish_aura(evidence,'prepared_not_started','已收到捡垃圾指令，正在准备同一个家庭场景。')
        driver=spawn('driver',[python,'-B','-m','integrations.aura_household_world.driver','--policy-port',str(port),'--socket',str(socket),'--status',str(root/'driver-status.json'),'--cap',str(config['native_step_budget'])])
        deadline=time.monotonic()+40
        while not socket.exists():
            if driver.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Owned driver not ready')
            time.sleep(.2)
        profile=upstream/'eval/behavior/behavior_interface_eval_test/robot_profiles/r1pro_8dof_hf250/evaluator_robot.yaml'
        evaluator=spawn('evaluator',[recipe['evaluator_python'],'-B','-m','integrations.aura_household_world.evaluator_entry',
            '--task-name',config['task_name'],'--robot-config',str(profile),'--host','127.0.0.1','--port',str(port),
            '--mode','public_test','--instance-indices',str(config['instance_id']-301),'--num-rollouts','1',
            '--max-steps',str(config['native_step_budget']),'--headless','--env-wrapper',recipe['wrapper'],'--output-dir',str(root/'native-output')])
        deadline=time.monotonic()+1200
        while True:
            if evaluator.poll() is not None:raise RuntimeError('Owned evaluator exited before initialization')
            try:status=json.loads((root/'driver-status.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):status={}
            if status.get('connected') and status.get('initialization',{}).get('ready'):break
            if time.monotonic()>deadline:raise TimeoutError('Scene initialization exceeded1200s')
            time.sleep(2)
        binding=spawn('binding',[python,'-B','-m','integrations.aura_household_world.bind_workspace','--template',recipe['workspace_template'],'--workspace',str(workspace),'--socket',str(socket),'--session-id',recipe['request']['id']])
        if binding.wait()!=0:raise RuntimeError('Fresh isolated workspace binding failed')
        prompt=root/'task-prompt.txt'
        # Source hash was checked during prepare; ship the actual prompt with recipe.
        prompt.write_text(recipe['task_prompt'],encoding='utf-8')
        agent=spawn('agent',[str(workspace/'.venv/bin/python'),'-B','-m','integrations.aura_household_world.run_agent','--config',str(config_path),'--workspace',str(workspace),'--evidence',str(evidence),'--task-file',str(prompt),'--run'])
        native=root/'native-output/json'/f"{config['task_name']}_{config['instance_id']}_0.json"
        while evaluator.poll() is None:
            if agent.poll() is not None and agent.returncode!=0:raise RuntimeError('Agent stopped; inspect its ledger before retry')
            time.sleep(2)
        if not native.is_file() or native.stat().st_mtime<launch_time:raise RuntimeError('No fresh native terminal result')
        result=close_with_native_result(evidence,native)
    except Exception as error:
        publish_aura(evidence,'incomplete','本次流程已暂停，请查看实际回执。')
        result={'phase':'needs_inspection','error':type(error).__name__+': '+str(error)}
        raise
    finally:
        for p in reversed(children):stop_owned(p)
        for log in logs:log.close()
        (root/'run-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--run',action='store_true')
    a=p.parse_args();recipe=json.loads(a.recipe.read_text());config=json.loads(a.config.read_text())
    if not a.run:
        print(json.dumps({'phase':'prepared_not_started','simulation_started':False,'model_calls':0}));return
    launch(config,recipe,explicit_run=True)

if __name__=='__main__':main()
