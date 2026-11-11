#!/usr/bin/env python3
"""Export non-sensitive per-inference timing data from task archives."""
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]); rows=[]
for task_dir in sorted(root.glob('*/run_*_api')):
 request_index=0
 for p in sorted(task_dir.glob('http_*_response.json')):
  rq=p.with_name(p.name.replace('_response.json','_request.json'))
  if not rq.exists() or not json.loads(rq.read_text()).get('path','').endswith('/chat/completions'):continue
  request_index+=1
  o=json.loads(p.read_text());u=o.get('body',{}).get('inference_usage') or {};t=u.get('timings_ms') or {};shape=u.get('output_shape') or []
  rows.append({'task':task_dir.parent.name,'request_index':request_index,'http_status':o.get('http_status'),'http_latency_s':o.get('http_latency_s'),'server_latency_s':t.get('total_ms',0)/1000 if t.get('total_ms') is not None else None,'action_rows':shape[0] if shape else None})
with out.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['task','request_index','http_status','http_latency_s','server_latency_s','action_rows'],lineterminator='\n');w.writeheader();w.writerows(rows)
print('exported',len(rows),'inference rows to',out)
