"""Single-episode G0.5 service and client using typed local array transport."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
from hybrid_rollout.robodojo.robodojo_server.protocol import RPCClient


def validate_actions(actions):
    values = np.asarray(actions, dtype=np.float32)
    if values.shape != (16, 14) or not np.isfinite(values).all():
        raise ValueError('Expected finite G0.5 actions with shape (16, 14)')
    result = values.copy()
    result[:, [6, 13]] = np.clip(result[:, [6, 13]], 0, 1)
    return result


class G05Client:
    def __init__(self, port, checkpoint):
        self.client = RPCClient('127.0.0.1', port, timeout=900)
        self.metadata = self.client.request('metadata')
        if (self.metadata.get('backend') != 'G0.5/PyTorch'
                or self.metadata.get('checkpoint') != str(Path(checkpoint).resolve())
                or self.metadata.get('inferences') != 0
                or self.metadata.get('action_horizon') != 16):
            self.close()
            raise ValueError('Fresh identified G0.5 server required')
        self.index = 0

    def infer(self, observation, output):
        started = time.monotonic()
        result = self.client.request('infer', observation=observation, index=self.index)
        if (result['index'] != self.index or
                result['checkpoint_sha256'] != self.metadata['checkpoint_sha256']):
            raise ValueError('G0.5 prediction identity mismatch')
        actions = validate_actions(result['actions'])
        np.savez_compressed(output, raw_actions=result['actions'], actions=actions,
            **{k: observation[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist','states')})
        self.index += 1
        return actions, dict(prediction_id=uuid.uuid4().hex, inference_index=self.index-1,
                             inference_seconds=time.monotonic()-started)

    def close(self):
        self.client.close()


class G05Session:
    def __init__(self, checkpoint, identity_output):
        import torch
        import yaml
        from XPolicyLab.policy.G05.model import Model
        from XPolicyLab.utils.process_data import unpack_robot_state
        from hybrid_rollout.robodojo.io import sha256
        adapter = Path(os.environ['G05_XPOLICY_ROOT']).resolve() / 'policy/G05'
        checkpoint = Path(checkpoint).resolve()
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        cfg = yaml.safe_load((adapter/'deploy.yml').read_text())
        cfg.update(ckpt_path=str(checkpoint), action_steps=16, action_source='fm',
                   inference_batch_size=1, frequency=25,
                   g05_root=str(adapter/'G05'))
        self.model = Model(cfg)
        self.model.reset()
        self.unpack = unpack_robot_state
        self.index = 0
        self.metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            backend='G0.5/PyTorch', config='g05_robodojo_fm', action_horizon=16, action_dim=14,
            inferences=0, policy_rng_seed=0, frequency=25, action_source='fm',
            norm_stats_sha256=sha256(checkpoint.parent.parent/'dataset_stats.json'),
            action_space='left_joints6_opening1_right_joints6_opening1')
        Path(identity_output).write_text(json.dumps(self.metadata,indent=2))
        self.poisoned = self.terminated = self.truncated = False

    def dispatch(self, op, args):
        if op == 'metadata':
            return self.metadata
        if op != 'infer' or self.poisoned or args['index'] != self.index:
            raise ValueError('Invalid or stale G0.5 inference request')
        observation = args['observation']
        obs = dict(instruction=observation['instruction'], additional_info={'frequency':25},
            state=self.unpack(observation['states'],'joint',self.model.robot_action_dim_info,source_type='obs'),
            vision={k:{'color':observation[k]} for k in ('cam_high','cam_left_wrist','cam_right_wrist')})
        self.model.update_obs(obs)
        chunk = self.model.get_action()
        raw = np.asarray([np.concatenate([row[k] for k in (
            'left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state')])
            for row in chunk], dtype=np.float32)
        validate_actions(raw)
        result = dict(actions=raw,index=self.index,checkpoint_sha256=self.metadata['checkpoint_sha256'])
        self.index += 1; self.metadata['inferences'] = self.index
        return result

    def _write_summary(self, reason):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--identity-output', required=True)
    parser.add_argument('--port',type=int,default=18830)
    args = parser.parse_args()
    from hybrid_rollout.robodojo.robodojo_server.rpc import serve
    serve(G05Session(args.checkpoint,args.identity_output),args.port)


if __name__ == '__main__':
    main()
