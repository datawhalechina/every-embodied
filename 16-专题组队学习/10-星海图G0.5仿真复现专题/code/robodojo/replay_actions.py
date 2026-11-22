"""Send recorded joint targets to the RoboDojo simulator for their original case."""
import argparse
import gzip
import json
from pathlib import Path

from hybrid_rollout.robodojo.robodojo_server.protocol import RPCClient

DATA = Path(__file__).resolve().parents[2] / "data/fold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=19113)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(DATA / "fold-mixed-trace.json.gz", "rt", encoding="utf-8") as file:
        trace = json.load(file)
    plan = json.loads((DATA / "case-plan.json").read_text(encoding="utf-8"))
    identity = next(row["identity"] for row in plan["cases"]
                    if row["identity"]["case_id"] == trace["case_id"])
    client = RPCClient("127.0.0.1", args.port, timeout=180)
    step = 0
    try:
        reset = client.request("reset", seed=identity["reset_seed"], source="student",
                               policy_version="recorded_action_trace")
        if reset["metadata"].get("evaluation_case") != identity:
            raise ValueError("Simulator case metadata differs from the recorded case")
        episode = reset["episode_id"]
        terminal = False
        errors = []
        for chunk in trace["chunks"]:
            if terminal:
                break
            result = client.request("chunk_step", actions=chunk["actions"],
                                    episode_id=episode, step_id=step)
            for row, reference in zip(result["steps"], chunk["states"]):
                errors.append(max(abs(actual - expected)
                                  for actual, expected in zip(row["obs"]["states"], reference)))
            step = result["step_id"]
            last = result["steps"][-1]
            terminal = last["terminated"] or last["truncated"]
        client.request("finish_pilot", episode_id=episode, step_id=step,
                       reason="terminal" if terminal else "recorded_trace_exhausted")
    finally:
        client.close()
    result = {"case_id": trace["case_id"], "model_calls": 0, "expected_steps": trace["control_steps"],
              "observed_steps": step, "terminal": terminal,
              "max_absolute_state_difference": max(errors, default=None)}
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not terminal or step != trace["control_steps"]:
        raise SystemExit("Physical replay ended at a different step. Inspect the simulator output.")


if __name__ == "__main__":
    main()
