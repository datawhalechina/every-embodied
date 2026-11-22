"""Replay recorded simulator actions once the external environments are installed."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def env_path(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} before starting this replay")
    path = Path(value).expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return str(listener.getsockname()[1])


@contextmanager
def service(command, cwd, env, log):
    with log.open("xb") as file:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=file,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def wait_ready(process, ready, log, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Replay service exited. See {log}")
        if ready():
            return
        time.sleep(1)
    raise TimeoutError(f"Replay service did not become ready. See {log}")


def fold(output):
    native = env_path("ROBODOJO_ROOT")
    controller = env_path("ASTRA_ROOT")
    python = env_path("SIM_PYTHON")
    code = ROOT / "code/robodojo"
    panel = controller / "hybrid_rollout/robodojo/eval_panels/robodojo_panel60_v1.json"
    plan = json.loads((ROOT / "data/fold/case-plan.json").read_text(encoding="utf-8"))
    env = dict(os.environ, ROLLOUT_STUDENT="g05", PYTHONUNBUFFERED="1",
               ROLLOUT_EVAL_MANIFEST_SHA256=plan["panel_sha256"],
               PYTHONPATH=os.pathsep.join(map(str, [code, controller, native, native / "XPolicyLab"])))
    port = free_port()
    case_file = output / "case.json"
    subprocess.run([sys.executable, code / "select_case.py", "--panel", panel,
                    "--case", "fold_clothes__random__g0__l0", "--output", case_file],
                   env=env, check=True)
    cmd = [python, "-m", "hybrid_rollout.robodojo.robodojo_server.server",
           "--task", "fold_clothes_random", "--output", output / "sim",
           "--eval-manifest", panel, "--case-file", case_file, "--port", port]
    log = output / "sim.log"
    with service(cmd, native, env, log) as process:
        wait_ready(process, lambda: '"event": "ready"' in log.read_text(errors="replace"), log)
        subprocess.run([python, code / "replay_actions.py", "--port", port,
                        "--report", output / "replay.json"], env=env, check=True, timeout=600)


def halloween(output, method, limit):
    harness = env_path("HARNESS")
    driver_python = env_path("DRIVER_PYTHON")
    evaluator_python = env_path("EVAL_PYTHON")
    data_path = env_path("OMNIGIBSON_DATA_PATH")
    code = ROOT / "code/roboharness"
    folder = ROOT / "data/halloween"
    row = next(run for run in json.loads((folder / "runs.json").read_text(encoding="utf-8"))
               if run["method"] == method)
    commands = row["steps"] - row["initial_sequence"]
    limit = commands if limit is None else limit
    if not 1 <= limit <= commands:
        raise ValueError(f"{method} contains {commands} archived commands")
    port, operator_port = free_port(), free_port()
    (output / "operator").mkdir()
    (output / "tmp").mkdir()
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMNIGIBSON_DATA_PATH=str(data_path),
               OMNIGIBSON_APPDATA_PATH=str(output / "kit"), OMNIGIBSON_HEADLESS="1",
               BEHAVIOR_INTERFACE_RUNTIME_TMP_ROOT=str(output / "tmp"),
               BEHAVIOR_EVAL_TEST_PORT=operator_port, BEHAVIOR_EVAL_TEST_POLICY_PORT=port,
               BEHAVIOR_EVAL_OPERATOR_DIR=str(output / "operator"),
               BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE="cpu",
               BEHAVIOR_EVAL_TEST_RESTORE_TASK_OBJECTS="1",
               BEHAVIOR_EVAL_TEST_ENABLE_PRIVILEGED_GRASP_ORACLE="0")
    workspace = harness / "harness/v2/R1_pro"
    driver_env = dict(os.environ, PYTHONPATH=str(workspace), PYTHONUNBUFFERED="1")
    driver_env.pop("OMNIGIBSON_APPDATA_PATH", None)
    evaluator_paths = [harness / "eval/behavior", *(harness / "BEHAVIOR" / name
                       for name in ("OmniGibson", "bddl3", "joylo"))]
    evaluator_env = dict(env, PYTHONPATH=os.pathsep.join(map(str, evaluator_paths)))
    profile = (harness / "eval/behavior/behavior_interface_eval_test/robot_profiles"
               / "r1pro_8dof_hf250/evaluator_robot.yaml")

    def healthy():
        try:
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                return response.status == 200
        except (URLError, TimeoutError):
            return False

    driver_cmd = [driver_python, code / "replay_driver.py", "--input", folder / method,
                  "--out", output / "driver", "--port", port, "--limit", str(limit)]
    driver_log = output / "driver.log"
    with service(driver_cmd, workspace, driver_env, driver_log) as driver:
        wait_ready(driver, healthy, driver_log, timeout=45)
        evaluator_cmd = [
            evaluator_python, code / "replay_evaluator.py",
            "--task-name", "putting_away_Halloween_decorations", "--robot-config", profile,
            "--host", "127.0.0.1", "--port", port, "--mode", "public_test", "--instance-indices", "0",
            "--num-rollouts", "1", "--max-steps", str(row["initial_sequence"] - 1 + limit), "--headless",
            "--env-wrapper", "behavior_interface_eval_test.official_rgbd_wrapper.OfficialRGBDFullResWrapper",
            "--output-dir", output / "evaluator", "--write-video", "--video-fps", "30",
        ]
        evaluator_log = output / "evaluator.log"
        with service(evaluator_cmd, harness, evaluator_env, evaluator_log) as evaluator:
            code = evaluator.wait()
        status = json.loads((output / "driver/status.json").read_text(encoding="utf-8"))
        if status["commands_sent"] != limit or status["checkpoints_checked"] < 1:
            raise RuntimeError(f"Replay stopped at {status['commands_sent']}/{limit}. See {evaluator_log}")
        print(f"Replayed {limit} commands with initial alignment. Evaluator exit status: {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=["fold", "halloween"])
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--method", choices=["astra", "purple", "pink"], default="pink")
    parser.add_argument("--limit", type=int, help="Halloween command limit. Default is the full stream.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(output, flush=True)
    if args.project == "fold":
        fold(output)
    else:
        halloween(output, args.method, args.limit)


if __name__ == "__main__":
    main()
