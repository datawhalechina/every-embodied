"""Run pi05 alone with a fixed 15-step receding horizon on the paired RPC recorder."""
import argparse
import json
from pathlib import Path
import signal

from hybrid_rollout.robodojo.pi05_server.client import Pi05Client
from hybrid_rollout.robodojo.robodojo_server.protocol import RPCClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sim-port', type=int, default=19113)
    parser.add_argument('--student-port', type=int, default=18830)
    args = parser.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt('Stopping this baseline episode')
    signal.signal(signal.SIGTERM, terminate)
    args.output.mkdir(parents=True, exist_ok=False)
    import os
    if os.environ.get('ROLLOUT_STUDENT') == 'g05':
        from g05_astra.student import G05Client
        student = G05Client(args.student_port, args.checkpoint)
    else:
        student = Pi05Client(args.student_port, args.checkpoint)
    sim = None
    state = None
    try:
        sim = RPCClient('127.0.0.1', args.sim_port, timeout=900)
        case_file = args.output.parent/'case.json'
        reset_seed = json.loads(case_file.read_text())['identity']['reset_seed'] if case_file.exists() else 0
        state = sim.request('reset', seed=reset_seed, source='student',
                            policy_version=student.metadata['checkpoint_sha256'])
        def request(op, **kwargs):
            return sim.request(op, episode_id=state['episode_id'], step_id=state['step_id'], **kwargs)
        (args.output/'method.json').write_text(json.dumps(dict(
            method=os.environ.get('ROLLOUT_STUDENT', 'pi05')+'_only_fixed15', horizon=15, checkpoint=student.metadata,
            scope='Local matched-recorder baseline, not an official leaderboard score'), indent=2))
        index = 0
        while True:
            obs = request('teacher_observation')
            actions, prediction = student.infer(obs, args.output/f'proposal_{index:05d}.npz')
            result = request('chunk_step', actions=actions[:15],
                             student_prediction_id=prediction['prediction_id'])
            state['step_id'] = result['step_id']
            last = result['steps'][-1]
            if last['terminated'] or last['truncated']:
                request('finish_pilot', reason='terminal')
                break
            index += 1
    except BaseException:
        if sim is not None and state is not None and sim.sock is not None:
            try:
                request('finish_pilot', reason='controller_error')
            except Exception:
                pass
        raise
    finally:
        if sim is not None:
            sim.close()
        student.close()


if __name__ == '__main__':
    main()
