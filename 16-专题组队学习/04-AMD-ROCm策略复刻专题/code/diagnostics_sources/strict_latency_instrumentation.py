"""Measure the unchanged official asynchronous evaluator in an isolated diagnostic run."""
import os,json,time,socket,subprocess,hashlib,sys
from pathlib import Path
root=Path('/home/aup/amd-notebook-work/latency-audit-strict-20261025-v1');root.mkdir(exist_ok=False)
formal=Path('/home/aup/amd-notebook-work/runs/dexjoco-formal-20261023-v1')
c=json.loads((formal/'pipeline_baseline_config.json').read_text());c['policy_variant']='base'
comparison=json.loads((formal/'formal_comparison.json').read_text())
base_dir=Path(comparison['baseline']['directory'])
c['warmup_observation']=str(base_dir/'warmup_observation.msgpack')
c['warmup_record']=str(root/'warmup.json')
c['load_audit_record']=str(root/'model_load_audit.json')
(root/'config.json').write_text(json.dumps(c,indent=2)+'\n')
source=Path(c['source_root'])/'dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py'
s=source.read_text()
s=s.replace('from .dexjoco_openpi_env import DexJoCoOpenPIEnv','from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv')
helper='''\nimport json\ndef audit_event(event, **fields):\n    record=dict(event=event,pid=os.getpid(),time_unix=time.time(),**fields)\n    with open(os.environ["DEX_AUDIT_LOG"],"a") as f:\n        f.write(json.dumps(record)+"\\n")\n\n'''
s=s.replace('@dataclass\nclass Observation:',helper+'@dataclass\nclass Observation:',1)
old='        result = client.infer(obs.obs)\n'
assert s.count(old)==1
s=s.replace(old,'        audit_started=time.perf_counter()\n'+old+'        audit_event("inference", observation_step=int(obs.timestamp), wall_ms=(time.perf_counter()-audit_started)*1000, model_ms=float(result.get("policy_timing",{}).get("infer_ms",-1)))\n')
old='            if action_chunk_timestamp_range[1] <= now_timestamp:\n'
assert s.count(old)==1
s=s.replace(old,'            audit_event("received_chunk", now_step=int(now_timestamp), observation_step=int(action_chunk.timestamp), length=int(action_chunk.action.shape[0]), expired=bool(action_chunk_timestamp_range[1] <= now_timestamp), usable_steps=max(0,int(action_chunk_timestamp_range[1]-now_timestamp)))\n'+old)
s=s.replace('            # Episode loop.\n','            audit_applied=0\n            audit_held=0\n            # Episode loop.\n',1)
s=s.replace('                    pressed_digits = env.step(action)\n','                    audit_applied+=1\n                    pressed_digits = env.step(action)\n',1)
s=s.replace('                    pressed_digits = env.stay(continue_stay=in_stay_state)\n','                    audit_held+=1\n                    pressed_digits = env.stay(continue_stay=in_stay_state)\n',1)
s=s.replace('                if env.is_done:\n','                if env.is_done:\n                    audit_event("episode_end", steps=int(timestamp), applied=audit_applied, held=audit_held, success=bool(env.is_success))\n',1)
s+='\nif __name__ == "__main__":\n    import tyro\n    tyro.cli(main)\n'
(root/'instrumented_client.py').write_text(s)
manifest={'official_client_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'instrumented_client_sha256':hashlib.sha256(s.encode()).hexdigest(),'scope':'one correctly loaded prewarmed baseline diagnostic; official scheduling and termination unchanged; excluded from formal denominators','seed':100,'checkpoint':c['base_model']}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
env=os.environ.copy();env['PYTHONPATH']='/home/aup/amd-notebook-work/rootcause-20261024-v1/code'
serverlog=(root/'server.log').open('w')
server=subprocess.Popen(['/home/aup/envs/openpi-amd-jax010/bin/python','-u','-m','amd_benchmarks.dexjoco','--serve',str(root/'config.json')],env=env,stdout=serverlog,stderr=subprocess.STDOUT)
(root/'status.json').write_text(json.dumps({'state':'starting','runner_pid':os.getpid(),'server_pid':server.pid}))
try:
    for _ in range(600):
        if server.poll() is not None:raise RuntimeError('Diagnostic server failed')
        try:
            with socket.create_connection(('127.0.0.1',8122),timeout=1):break
        except OSError:time.sleep(1)
    else:raise TimeoutError('Diagnostic server startup timeout')
    env['PYTHONPATH']=str(Path(c['source_root'])/'openpi/packages/openpi-client/src')
    env['DEX_AUDIT_LOG']=str(root/'events.jsonl')
    cmd=['/home/aup/envs/dexjoco-amd/bin/python','-u',str(root/'instrumented_client.py'),'--config='+str(Path(c['source_root'])/'configs/rand_obj/water_plant.yaml'),'--seed=100','--host=127.0.0.1','--port=8122','--output='+str(root/'episode'),'--render-mode=rgb_array','--episodes=1']
    with (root/'client.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    rows=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
    infer=[x for x in rows if x['event']=='inference'];chunks=[x for x in rows if x['event']=='received_chunk']
    import statistics
    summary={'state':'complete','episode':[x for x in rows if x['event']=='episode_end'],'inference_count':len(infer),'inference_wall_ms':[x['wall_ms'] for x in infer],'steady_inference_median_ms':statistics.median([x['wall_ms'] for x in infer[1:]]) if len(infer)>1 else None,'received_chunks':len(chunks),'expired_chunks':sum(x['expired'] for x in chunks),'usable_steps_per_chunk':[x['usable_steps'] for x in chunks],'scope':manifest['scope']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'status.json').write_text(json.dumps({'state':'complete'}))
    print(json.dumps(summary),flush=True)
except Exception as e:
    (root/'status.json').write_text(json.dumps({'state':'failed','error':repr(e)}));raise
finally:
    server.terminate()
    try:server.wait(timeout=30)
    except subprocess.TimeoutExpired:server.kill();server.wait()
    serverlog.close()
