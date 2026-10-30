"""Run the cup course's collection, replay audit, training and held-out evaluation.

All heavy outputs live outside the course source tree. Completed stages resume
only under the same source/configuration fingerprint. Failed logs are retained.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("CUP_RUN_ROOT", "/data/Data14TB/runs/every_embodied/cup-deterministic-v2")).resolve()
PYTHON = os.environ.get("EXPERIMENT_PYTHON", sys.executable)
RECIPE = {"collection_seeds": list(range(1000, 1040)), "evaluation_seeds": list(range(2000, 2030)),
          "steps": 20000, "batch_size": 8, "gradient_accumulation": 1,
          "lr": 0.0001, "chunk_size": 20, "n_action_steps": 20,
          "horizon": 1200, "control_hz": 20, "task": "red mug to white plate",
          "input": "two RGB cameras and end-effector pose", "action": "six joint targets and gripper",
          "initialization": "ACT transformer from scratch, cached ImageNet ResNet18 backbone",
          "sampling": "uniform", "object_truth_as_policy_input": False,
          "use_vae": False, "fresh_batches_each_epoch": True,
          "reused_verified_data_run": "cup-full-20261027-v1"}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, data):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def command(stage):
    data = str(RUN / "dataset")
    if stage == "collect":
        return ["collect_pnp_scripted_oracle.py", "--seed-start", "1000", "--num-demos", "40",
                "--root", data, "--repo-id", "amd_cup_red_v1", "--write-dataset",
                "--video-dir", str(RUN / "collection-videos"), "--width", "640", "--height", "480",
                "--frame-every", "2", "--video-fps", "10"]
    if stage == "audit":
        return ["audit_pnp_dataset_physical.py", "--root", data, "--repo-id", "amd_cup_red_v1",
                "--output-jsonl", str(RUN / "replay.jsonl"), "--summary-json", str(RUN / "replay-summary.json")]
    if stage == "train":
        return ["train_act_chunk.py", "--root", data, "--repo-id", "amd_cup_red_v1",
                "--output-dir", str(RUN / "policy"), "--steps", "20000", "--batch-size", "8",
                "--chunk-size", "20", "--n-action-steps", "20", "--lr", "0.0001",
                "--keep-wrist-image", "--num-workers", "2", "--log-freq", "25", "--eval-freq", "1000",
                "--device", "cuda", "--disable-vae"]
    if stage == "evaluate":
        return ["eval_policy_success.py", "--policy", "act", "--act-policy-path", str(RUN / "policy"),
                "--act-dataset-root", data, "--act-repo-id", "amd_cup_red_v1", "--episodes", "30",
                "--seed-start", "2000", "--max-action-steps", "1200", "--hz", "20", "--physical-success",
                "--video-dir", str(RUN / "evaluation-videos"), "--trace-dir", str(RUN / "evaluation-traces"),
                "--record-all-episodes", "--output-jsonl", str(RUN / "evaluation.jsonl"), "--device", "cuda"]
    raise ValueError(stage)


def validate(stage):
    if stage == "collect":
        rows = read_rows(RUN / "collection-videos/scripted_oracle_summary.jsonl")
        assert len(rows) == 40 and [r["seed"] for r in rows] == RECIPE["collection_seeds"]
        assert not any("error" in r for r in rows), "Collection contains errors; inspect the retained log"
        saved = [r for r in rows if r["dataset_saved"]]
        assert len(saved) >= 20, "Fewer than 20 successful physical demonstrations"
        return {"attempts": len(rows), "saved_episodes": len(saved), "frames": sum(r["action_steps"] for r in saved)}
    if stage == "audit":
        result = json.loads((RUN / "replay-summary.json").read_text())
        assert result["episodes"] >= 20
        assert result["physical_success_count"] == result["episodes"], "Replay failed; repair action/data alignment before training"
        return result
    if stage == "train":
        rows = []
        for line in (RUN / "train.log").read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("event") == "train":
                rows.append(item)
        assert rows and rows[-1]["step"] == 20000
        assert all(math.isfinite(r["loss"]) for r in rows)
        weights = RUN / "policy/model.safetensors"
        assert weights.is_file()
        write_json(RUN / "training-metrics.json", rows)
        return {"updates": rows[-1]["step"], "last_loss": rows[-1]["loss"],
                "elapsed_s": rows[-1]["elapsed_s"], "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    rows = read_rows(RUN / "evaluation.jsonl")
    assert [r["seed"] for r in rows] == RECIPE["evaluation_seeds"]
    assert all(r.get("video_path") and Path(r["video_path"]).is_file() for r in rows)
    result = {"episodes": len(rows), "physical_successes": sum(r["physical_success"] for r in rows),
              "seeds": RECIPE["evaluation_seeds"], "horizon": 1200,
              "weights_sha256": hashlib.sha256((RUN / "policy/model.safetensors").read_bytes()).hexdigest()}
    write_json(RUN / "evaluation-summary.json", result)
    return result


def run(stage, train_callable=None, stage_callable=None):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in SOURCE.rglob("*.py") if "__pycache__" not in p.parts}
        contract = {"recipe": RECIPE, "sources": sources}
        contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        state_path = RUN / "pipeline_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": {}}
        if state.get("contract_hash", contract_hash) != contract_hash:
            raise RuntimeError("Source/recipe changed: use a new run directory to preserve prior evidence")
        state["contract_hash"] = contract_hash
        if stage in state["completed"]:
            result = validate(stage)
            print(json.dumps({"stage": stage, "reused_verified_result": result}, ensure_ascii=False))
            return result
        order = ["collect", "audit", "train", "evaluate"]
        for prerequisite in order[:order.index(stage)]:
            assert prerequisite in state["completed"], f"Missing completed stage: {prerequisite}"
        assert shutil.disk_usage(RUN).free > 12 * 2**30, "At least 12 GiB free disk required"
        if not (SOURCE / "asset").exists():
            assets = Path(os.environ["PNP_ASSET_ROOT"]).resolve()
            assert (assets / "example_scene_y.xml").is_file()
            (SOURCE / "asset").symlink_to(assets, target_is_directory=True)
        write_json(RUN / "recipe.json", contract)
        state.update(status="running", stage=stage, pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(state_path, state)
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", MUJOCO_GL="glfw", SUCCESS_REQUIRE_EE_HEIGHT="0")
        # Legacy LeRobot is selected explicitly so its dataset and ACT APIs match.
        env["PYTHONPATH"] = os.environ["LEROBOT_SRC"]
        argv = [PYTHON, "-u", *command(stage)]
        if stage != "train" and not env.get("DISPLAY"):
            argv = ["xvfb-run", "-a", *argv]
        log = RUN / (stage + ".log")
        if log.exists():
            log.rename(RUN / (stage + ".failed-" + str(time.time_ns()) + ".log"))
        try:
            with log.open("w") as stream:
                if stage_callable is not None or (stage == "train" and train_callable is not None):
                    import contextlib
                    old_cwd = Path.cwd()
                    old_env = os.environ.copy()
                    virtual_display = None
                    try:
                        if stage != "train" and not env.get("DISPLAY"):
                            reader, writer = os.pipe()
                            try:
                                virtual_display = subprocess.Popen(
                                    ["Xvfb", "-displayfd", str(writer), "-screen", "0", "1280x720x24", "-nolisten", "tcp"],
                                    pass_fds=(writer,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            finally:
                                os.close(writer)
                            with os.fdopen(reader) as pipe:
                                display_number = pipe.readline().strip()
                            if not display_number.isdigit():
                                raise RuntimeError("Private virtual display failed to start")
                            env["DISPLAY"] = ":" + display_number
                        os.chdir(SOURCE)
                        os.environ.update(env)
                        class Tee:
                            def __init__(self, file, notebook):
                                self.file, self.notebook = file, notebook
                            def write(self, value):
                                self.file.write(value)
                                self.notebook.write(value)
                            def flush(self):
                                self.file.flush()
                                self.notebook.flush()
                        stdout, stderr = Tee(stream, sys.stdout), Tee(stream, sys.stderr)
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            (stage_callable or train_callable)(command(stage))
                    finally:
                        if virtual_display is not None:
                            virtual_display.terminate()
                            virtual_display.wait(timeout=10)
                        os.chdir(old_cwd)
                        os.environ.clear()
                        os.environ.update(old_env)
                else:
                    subprocess.run(argv, cwd=SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = validate(stage)
        except BaseException as exc:
            state.update(status="failed", error=repr(exc), log=str(log))
            write_json(state_path, state)
            raise
        state["completed"][stage] = result
        state.update(status="complete" if stage == "evaluate" else "stage_complete", error=None)
        write_json(state_path, state)
        print(json.dumps({"stage": stage, "result": result, "log": str(log)}, ensure_ascii=False, indent=2))
        return result


if __name__ == "__main__":
    run(sys.argv[1])
