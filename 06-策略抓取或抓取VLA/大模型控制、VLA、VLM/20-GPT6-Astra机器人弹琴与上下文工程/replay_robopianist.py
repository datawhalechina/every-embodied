"""Replay the official RoboPianist action fixture, not an online GPT policy."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/robopianist-replay"))
    args = parser.parse_args()
    import numpy as np
    from dm_env_wrappers import CanonicalSpecWrapper
    from mujoco_utils import composer_utils
    from robopianist import music
    from robopianist.suite.tasks import piano_with_shadow_hands
    from robopianist.wrappers import PianoSoundVideoWrapper

    actions = np.load(args.repo / "examples/twinkle_twinkle_actions.npy", allow_pickle=False)
    if actions.ndim != 2 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected a non-empty finite 2D action sequence")
    # Match the official tutorial task and wrapper, not an arbitrary suite preset.
    task = piano_with_shadow_hands.PianoWithShadowHands(
        change_color_on_activation=True,
        midi=music.load("TwinkleTwinkleRousseau"),
        trim_silence=True,
        control_timestep=0.05,
        gravity_compensation=True,
        primitive_fingertip_collisions=False,
        reduced_action_space=False,
        n_steps_lookahead=10,
        disable_fingering_reward=False,
        disable_forearm_reward=False,
        disable_colorization=False,
        disable_hand_collisions=False,
        attachment_yaw=0.0,
    )
    base = composer_utils.Environment(
        task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
    )
    args.out.mkdir(parents=True, exist_ok=True)
    video = PianoSoundVideoWrapper(
        base, record_every=1, camera_id="piano/back", record_dir=str(args.out.resolve())
    )
    env = CanonicalSpecWrapper(video)
    try:
        spec = env.action_spec()
        if actions.shape[1:] != spec.shape:
            raise ValueError(f"Action shape {actions.shape[1:]} != environment {spec.shape}")
        if (actions < spec.minimum).any() or (actions > spec.maximum).any():
            raise ValueError("Actions outside the canonical action range")
        timestep = env.reset()
        steps = 0
        while not timestep.last():
            if steps >= len(actions):
                raise ValueError("Sequence exhausted before episode ended; check task/version")
            timestep = env.step(actions[steps])
            steps += 1
        report = {
            "mode": "official_recorded_action_replay",
            "model_calls": 0,
            "steps": steps,
            "control_timestep_seconds": 0.05,
            "physical_success": "not_measured",
            "video_files": [p.name for p in args.out.glob("*.mp4")],
        }
        (args.out / "replay_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
