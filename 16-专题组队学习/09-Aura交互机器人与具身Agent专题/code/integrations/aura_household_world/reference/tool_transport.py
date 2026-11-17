"""Validate an original atomic tool receipt when CLI stdout is not clean JSON.

Only a unique new history item matching the exact invocation may replace the
text transport. Completed native command records must be contiguous. Nothing
in this module sends a command, reruns a tool, or changes an original record.
"""
import json
from pathlib import Path


def stored_receipt(workspace,before,name,arguments,binding,compact):
    workspace=Path(workspace).resolve();candidates=[]
    for path in (workspace/'tool_call_history').glob('*'):
        if path.name in before or not path.is_dir():continue
        inp=path/'input.json';result=path/'result.json'
        if not inp.is_file() or not result.is_file():continue
        i=json.loads(inp.read_bytes());r=json.loads(result.read_bytes())
        if (i.get('tool')!=name or i.get('arguments')!=arguments
                or any(i.get(k)!=binding[k] for k in ['episode_id','session_id'])
                or i.get('observation_sequence')!=binding['sequence']):continue
        if r.get('tool_name')!=name or Path(r.get('record','')).resolve()!=result.resolve():
            raise ValueError('Atomic receipt identity differs from its original record')
        count=r.get('joint_commands_sent')
        if type(count) is not int or count<0:raise ValueError('Atomic command count is missing')
        session=json.loads((workspace/'.session.json').read_bytes())
        if any(session.get(k)!=binding[k] for k in ['episode_id','session_id']):
            raise ValueError('Atomic receipt belongs to another workspace session')
        commands=Path(session['trajectory_dir'])/'calls'/path.name/'joint_commands.jsonl'
        rows=[json.loads(l) for l in commands.read_text().splitlines()] if commands.exists() else []
        if len(rows)!=count:raise ValueError('Atomic physical command count is incomplete')
        sequence=binding['sequence']
        for row in rows:
            if row['sequence_before']!=sequence or row['sequence_after']!=sequence+1:
                raise ValueError('Atomic physical sequence is not contiguous')
            sequence+=1
        if count and r.get('rollout_budget',{}).get('used_ticks')!=sequence-1:
            raise ValueError('Atomic native counter differs from command acknowledgments')
        candidates.append((r,result))
    if len(candidates)!=1:raise ValueError('No unique completed original atomic tool receipt')
    receipt,path=candidates[0]
    return compact(receipt,workspace),path
