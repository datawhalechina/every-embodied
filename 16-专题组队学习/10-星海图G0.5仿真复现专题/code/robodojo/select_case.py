"""Select one frozen RoboDojo case and write its original launcher input."""
import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data/fold/case-plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(DATA.read_text(encoding="utf-8"))
    from hybrid_rollout.robodojo.evaluation import case_identity, read_panel

    panel = read_panel(args.panel, plan["panel_sha256"])
    identity = next(row["identity"] for row in plan["cases"] if row["identity"]["case_id"] == args.case)
    case = next(row for row in panel["cases"] if row["case_id"] == args.case)
    if case_identity(panel, case) != identity:
        raise ValueError("Recorded case does not match the benchmark panel")
    args.output.write_text(json.dumps({"case": case, "identity": identity}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
