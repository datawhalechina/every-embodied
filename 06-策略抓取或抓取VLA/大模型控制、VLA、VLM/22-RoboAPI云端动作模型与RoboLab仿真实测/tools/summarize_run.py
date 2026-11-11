#!/usr/bin/env python3
"""Summarize saved RoboAPI episode records and HTTP timing distributions."""
import json,math,statistics,sys
from pathlib import Path

def distribution(values):
    vals=sorted(float(v) for v in values if v is not None)
    if not vals:return None
    def pct(p):
        pos=(len(vals)-1)*p; lo=math.floor(pos); hi=math.ceil(pos)
        return vals[lo] + (vals[hi]-vals[lo])*(pos-lo)
    return {'n':len(vals),'min':vals[0],'mean':statistics.mean(vals),'median':statistics.median(vals),'p90':pct(.90),'p95':pct(.95),'max':vals[-1]}
root=Path(sys.argv[1]); task_dirs=sorted(p for p in root.glob('*/run_*_api') if (p/'summary.json').exists())
for ep in task_dirs:
 s=json.loads((ep/'summary.json').read_text()); cloud=[]; statuses={}; shapes=[]; success_http=[]; failed_http=[]
 for p in sorted(ep.glob('http_*_response.json')):
  obj=json.loads(p.read_text()); status=str(obj.get('http_status'));statuses[status]=statuses.get(status,0)+1
  request=json.loads(p.with_name(p.name.replace('_response.json','_request.json')).read_text())
  if request.get('path','').endswith('/chat/completions'):
   (success_http if obj.get('http_status')==200 else failed_http).append(obj.get('http_latency_s'))
  u=obj.get('body',{}).get('inference_usage') or {}; timing=u.get('timings_ms') or {}
  if 'total_ms' in timing:cloud.append(float(timing['total_ms'])/1000)
  if u.get('output_shape'):shapes.append(u['output_shape'])
 seedfiles=list(ep.parent.glob('run_*_seed.json')); seeds=[json.loads(p.read_text()) for p in seedfiles]
 out={'task':ep.parent.name,'status':s.get('status'),'requests':s.get('requests'),'responses':s.get('responses'),'chunks_completed':s.get('chunks_completed'),'executed_steps':s.get('executed_steps'),'errors':s.get('errors'),'model':s.get('model'),'resolved_model':s.get('resolved_model'),'http_latency_s':distribution(s.get('api_http_latency_s',[])),'successful_inference_http_latency_s':distribution(success_http),'failed_inference_http_latency_s':distribution(failed_http),'client_latency_s':distribution(s.get('api_latency_s',[])),'server_latency_s':distribution(cloud),'http_status_counts':statuses,'action_shapes':shapes,'episode_records':seeds}
 (ep/'analysis_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(out,ensure_ascii=False,indent=2))
