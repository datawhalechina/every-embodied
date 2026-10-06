"""Local process bridge for fresh observations and bounded G0.5 arm posture priors."""

import json
from pathlib import Path
import subprocess

from g05_native_scene import load_prediction
from g05_piano_probe import sha256


class PianoPolicy:
    def __init__(self, args, task, physics):
        self.task, self.physics = task, physics
        self.folder = args.out.resolve() / "g05-observations"
        self.folder.mkdir()
        self.log = (args.out / "g05-worker.log").open("w", encoding="utf-8")
        command = [str(args.g05_python), "-u", str(Path(__file__).with_name("g05_piano_worker.py")),
                   "--repo", str(args.g05_repo), "--shared-models", str(args.g05_models),
                   "--checkpoint", str(args.g05_checkpoint), "--out", str(args.out.resolve() / "g05")]
        command.extend(["--config-mode", getattr(args, "g05_config_mode", "legacy-ar")])
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        self.calls = []
        try:
            if not self._read().get("ready"):
                raise RuntimeError("G0.5 worker did not initialize")
        except BaseException:
            self.close()
            raise

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("G0.5 worker exited; inspect g05-worker.log")
        return json.loads(line)

    def infer(self, time_seconds, score):
        import numpy as np

        observation = {}
        for name, size in (("head_rgb", (640, 360)), ("left_wrist_rgb", (640, 480)), ("right_wrist_rgb", (640, 480))):
            camera = self.task.robot.mjcf_model.find("camera", name).full_identifier
            pixels = self.physics.render(width=size[0], height=size[1], camera_id=camera)
            observation[name] = pixels.transpose(2, 0, 1)
        for side in ("left", "right"):
            observation[side + "_arm"] = np.asarray([self.physics.bind(j).qpos[0] for j in self.task.robot.arm_joints[side]], dtype=np.float32)
            tips = self.task.robot.hands[side].fingertip_sites
            aperture = np.linalg.norm(self.physics.bind(tips[0]).xpos - self.physics.bind(tips[1]).xpos)
            observation[side + "_gripper"] = np.array([np.clip(aperture * 1000, 0, 100)], dtype=np.float32)
        beat = 60 / score["tempo_bpm"]
        upcoming = [{"hand": n["hand"], "midi": n["midi"], "finger": n["finger"]} for n in score["notes"]
                    if 1 + (n["start_beat"] + n["duration_beats"]) * beat >= time_seconds
                    and 1 + n["start_beat"] * beat < time_seconds + 2]
        return self.infer_observation(time_seconds, observation,
            "Keep both arms comfortably above the piano keyboard while playing. Adjust the arm posture gently to support the fingers pressing the keys.",
            "The local finger controller handles independent key presses. Provide arm posture proposals for the next score events: " + json.dumps(upcoming))

    def infer_observation(self, time_seconds, observation, instruction, plan, *, chunk=False, seed=None):
        """Allow named-scene experiments to reuse the same audited model bridge."""
        import numpy as np

        for side in ("left", "right"):
            value = np.asarray(observation[side + "_arm"])
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError("Expected seven finite measured arm angles")
        observation_file = self.folder / f"observation-{len(self.calls)}.npz"
        np.savez_compressed(observation_file, **observation)
        request = {"observation": str(observation_file), "time_seconds": time_seconds,
                   "instruction": instruction, "plan": plan}
        if seed is not None:
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("Invalid inference seed")
            request["seed"] = seed
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self._read()
        actions, absent, report = load_prediction(Path(response["actions"]), Path(response["report"]), response["call_index"])
        call = report["calls"][response["call_index"]]
        if call["observation_sha256"] != sha256(observation_file):
            raise ValueError("G0.5 proposal does not belong to the current observation")
        proposed, clipped = {}, {}
        for side in ("left", "right"):
            if side + "_control" in absent:
                continue
            current = observation[side + "_arm"]
            raw = actions[side + "_arm"] if chunk else np.mean(actions[side + "_arm"][:4], axis=0)
            target = np.clip(raw, current - 0.25, current + 0.25)
            proposed[side] = target
            clipped[side] = int(np.count_nonzero(target != raw))
        self.calls.append({"time_seconds": time_seconds, "call_index": response["call_index"],
            "observation_sha256": call["observation_sha256"], "seed": seed, "chunk": chunk,
            "arms_proposed": sorted(proposed), "clipped_coordinates": clipped,
            "bounded_targets": {k: v.tolist() for k, v in proposed.items()},
            "inference_seconds": call["inference_seconds"], "actions_sha256": call["actions_sha256"]})
        return proposed

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()
