#!/usr/bin/env python3
"""Serve archived actions to an isolated evaluator without a model session."""
import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tools.util.driver import Driver, serve
from tools.util.model import validate_action, PROPRIO_SLICES
from tools.util.observation import Snapshot, filter_observation

POSITION_GROUPS=('trunk_qpos','arm_left_qpos','arm_right_qpos','gripper_left_qpos','gripper_right_qpos')


def checkpoint_metrics(actual,expected,actual_rgb=None,expected_image=None):
    """Compare measured positions and, when present, the unmodified head image."""
    a=np.asarray(actual,dtype=float); b=np.asarray(expected,dtype=float)
    if a.shape!=(65,) or b.shape!=(65,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected two finite 65-value observations')
    errors={key:float(np.max(np.abs(a[PROPRIO_SLICES[key]]-b[PROPRIO_SLICES[key]]))) for key in POSITION_GROUPS}
    result={'joint_position_max_abs':max(errors.values()),'group_max_abs':errors}
    if actual_rgb is not None and expected_image is not None:
        aimg=np.asarray(actual_rgb,dtype=np.uint8)[...,:3]
        with Image.open(expected_image) as image:bimg=np.asarray(image.convert('RGB'),dtype=np.uint8)
        if aimg.shape!=bimg.shape:result['head_image_shape_mismatch']=[list(aimg.shape),list(bimg.shape)]
        else:result['head_rgb_mean_absolute_difference']=float(np.mean(np.abs(aimg.astype(np.int16)-bimg.astype(np.int16))))
    return result


class ReplayDriver(Driver):
    def __init__(self,actions,checkpoints,out,limit):
        super().__init__()
        self.actions=actions[:limit];self.by_sequence={int(r['sequence']):r for r in checkpoints}
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.sent=0;self.checks=0;self.last_phase='prepared';self.connected=False
        self.write_replay_status('prepared')

    def write_replay_status(self,phase,**extra):
        self.last_phase=phase
        row={'phase':phase,'sequence':self.sequence,'commands_sent':self.sent,'commands_limit':len(self.actions),
             'checkpoints_checked':self.checks,'model_calls':0,**extra}
        tmp=self.out/'status.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(self.out/'status.json')

    def write_status(self):
        pass

    def observe(self,raw):
        with self.condition:
            self.sequence+=1
            self.adapter.current=Snapshot(filter_observation(raw),self.sequence,time.time(),self.world.episode_id())
            self.world.reconcile_base_odometry_from_proprio(1./30.)
            if not self.initializer.ready():
                action=self.initializer.step(self.world)
                self.world.set_episode_initialized(self.initializer.ready())
                self.ticks+=1
                return action
            self.world.set_episode_initialized(True)
            if self.sent>=len(self.actions):
                self.write_replay_status('exhausted')
                raise RuntimeError('Replay command limit reached; no extra action issued')
            row=self.actions[self.sent]
            if self.sequence!=row['sequence_before']:
                self.write_replay_status('sequence_mismatch',expected=row['sequence_before'])
                raise RuntimeError('Replay sequence does not match saved action')
            ref=self.by_sequence.get(self.sequence)
            if ref:
                expected=json.loads(Path(ref['joint_state']).read_text())['proprioception']
                actual_head=self.adapter.camera_rgb('head')
                metrics=checkpoint_metrics(self.adapter.proprio_vector(),expected,actual_head,ref.get('head_image'))
                observation={'sequence':self.sequence,'source_image_id':ref.get('head_image_id'),**metrics}
                if metrics.get('head_rgb_mean_absolute_difference',0)>25:
                    directory=self.out/'head-differences';directory.mkdir(exist_ok=True)
                    replay_image=directory/f'seq-{self.sequence}.png'
                    Image.fromarray(np.asarray(actual_head,dtype=np.uint8)[...,:3]).save(replay_image)
                    observation.update(source_head_image=ref.get('head_image'),replay_head_image=str(replay_image))
                with (self.out/'checkpoint-comparisons.jsonl').open('a') as f:f.write(json.dumps(observation)+'\n')
                self.checks+=1
                if self.sent==0 and (metrics['joint_position_max_abs']>0.02 or metrics.get('head_rgb_mean_absolute_difference',256)>25):
                    self.write_replay_status('initial_alignment_failed',metrics=metrics)
                    raise RuntimeError('Initial replay observation differs from source')
            action=validate_action(row['action'])
            self.sent+=1;self.ticks+=1
            if self.sent==1 or self.sent%100==0:self.write_replay_status('replaying')
            return action

    def disconnected(self):
        with self.condition:
            self.connected=False
            if self.last_phase not in {'sequence_mismatch', 'initial_alignment_failed', 'exhausted'}:
                self.write_replay_status('disconnected')
            self.condition.notify_all()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--limit',type=int,required=True)
    a=p.parse_args()
    with gzip.open(a.input/'actions.jsonl.gz', 'rt', encoding='utf-8') as file:
        actions = [json.loads(line) for line in file]
    checkpoint = json.loads((a.input/'checkpoint.json').read_text(encoding='utf-8'))
    for key in ('head_image', 'joint_state'):
        checkpoint[key] = str((a.input/checkpoint[key]).resolve())
    checkpoints = [checkpoint]
    if not 1<=a.limit<=len(actions):raise ValueError('Invalid replay action bound')
    if not any(row.get('sequence')==actions[0]['sequence_before'] and row.get('head_image') for row in checkpoints):
        raise ValueError('The initial sequence lacks its reference image and state')
    if any(r['sequence_after']!=r['sequence_before']+1 for r in actions):raise ValueError('Invalid action sequence')
    if any(x['sequence_after']!=y['sequence_before'] for x,y in zip(actions,actions[1:])):raise ValueError('Action stream has gaps')
    driver=ReplayDriver(actions,checkpoints,a.out,a.limit)
    try:asyncio.run(serve(driver,'127.0.0.1',a.port))
    finally:driver.disconnected()

if __name__=='__main__':main()
