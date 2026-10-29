#!/usr/bin/env python3
"""Add a real 14-episode media review cell to the six workflow notebooks."""

from __future__ import annotations

import json
from pathlib import Path


TOPIC = Path(__file__).resolve().parents[1]
WORKFLOWS = TOPIC / "notebooks" / "workflows"


def patch_common_helper(source: str) -> str:
    # Evaluation cells run in a fresh kernel too.  Keep the compact iterator
    # available globally instead of defining it only inside the training
    # function.
    progress_helper = '''\n\nclass _NotebookCompactProgress:\n    def __init__(self, iterable, desc, total=None):\n        self.iterable = iterable\n        self.desc = desc\n        self.total = int(total if total is not None else len(iterable))\n        self.every = max(1, int(os.environ.get("NOTEBOOK_PROGRESS_EVERY", "1")))\n        self.postfix = ""\n\n    def set_postfix(self, **kwargs):\n        self.postfix = ", ".join(f"{key}={value}" for key, value in kwargs.items())\n\n    def __iter__(self):\n        for index, value in enumerate(self.iterable, start=1):\n            yield value\n            if index == 1 or index == self.total or index % self.every == 0:\n                suffix = f" | {self.postfix}" if self.postfix else ""\n                print(f"{self.desc}: {index}/{self.total}{suffix}", flush=True)\n\ndef notebook_progress(iterable, desc, total=None):\n    return _NotebookCompactProgress(iterable, desc, total=total)\n'''
    if "class _NotebookCompactProgress" not in source:
        source = source.replace(
            "def train_lerobot_config_in_notebook(",
            progress_helper + "\n\ndef train_lerobot_config_in_notebook(",
            1,
        )
    source = source.replace(
        "    dataset_root=None,\n):\n",
        "    dataset_root=None,\n    video_dir=None,\n    trace_dir=None,\n    seed_list=None,\n):\n",
        1,
    )
    source = source.replace(
        "        output_jsonl=result_path,\n        device=",
        "        output_jsonl=result_path,\n        video_dir=Path(video_dir) if video_dir else None,\n        trace_dir=Path(trace_dir) if trace_dir else None,\n        device=",
        1,
    )
    source = source.replace(
        "    with pushd(PROJECT_ROOT):\n",
        "    with pushd(PROJECT_ROOT):\n        selected_cases = set()\n",
        1,
    )
    source = source.replace(
        "            seed = args.seed_start + offset\n            row = rollout(args, policy, seed)\n            rows.append(row)\n",
        "            seed = (list(seed_list)[offset] if seed_list is not None else args.seed_start + offset)\n            row = rollout(args, policy, seed)\n            row = module.finalize_case_media(args, row, selected_cases)\n            rows.append(row)\n",
        1,
    )
    return source


def media_cell(model: str) -> tuple[str, str]:
    if model == "smolvla":
        code = r'''# FULL_14_EVAL_WITH_MEDIA
FULL_EVAL_EPISODES = 14
FULL_EVAL_SEEDS = list(range(14))
FULL_EVAL_ROOT = OUTPUT_ROOT / "full_eval_14"
FULL_EVAL_RESULT = FULL_EVAL_ROOT / "result.jsonl"
FULL_EVAL_POLICY = resolve_eval_policy(PRETRAINED_POLICY, LONG_OUTPUT, "SMOLVLA_EVAL_POLICY_PATH")
FULL_EVAL_ROWS = run_eval_policy_in_notebook(
    "smolvla",
    FULL_EVAL_POLICY,
    FULL_EVAL_RESULT,
    episodes=FULL_EVAL_EPISODES,
    seed_start=0,
    seed_list=FULL_EVAL_SEEDS,
    render=True,
    enabled=RUN_EVAL,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
    video_dir=FULL_EVAL_ROOT / "videos",
    trace_dir=FULL_EVAL_ROOT / "traces",
)
'''
    elif model == "pi0":
        code = r'''# FULL_14_EVAL_WITH_MEDIA
FULL_EVAL_EPISODES = 14
FULL_EVAL_SEEDS = [3007, 3002, 3004, 3006, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019]
FULL_EVAL_ROOT = OUTPUT_ROOT / "full_eval_14"
FULL_EVAL_RESULT = FULL_EVAL_ROOT / "result.jsonl"
FULL_EVAL_POLICY = resolve_eval_policy(PI0_POLICY_PATH, LONG_OUTPUT, "PI0_EVAL_POLICY_PATH")
FULL_EVAL_ROWS = run_pi0_eval_native_in_notebook(
    FULL_EVAL_POLICY,
    FULL_EVAL_RESULT,
    FULL_EVAL_SEEDS,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
    render=True,
    enabled=RUN_EVAL,
    video_dir=FULL_EVAL_ROOT / "videos",
    trace_dir=FULL_EVAL_ROOT / "traces",
)
'''
    else:
        code = r'''# FULL_14_EVAL_WITH_MEDIA
FULL_EVAL_EPISODES = 14
FULL_EVAL_SEEDS = list(range(1030, 1044))
FULL_EVAL_ROOT = OUTPUT_ROOT / "full_eval_14"
FULL_EVAL_RESULT = FULL_EVAL_ROOT / "result.jsonl"
FULL_EVAL_POLICY = resolve_eval_policy(
    Path(os.environ.get("ACT_EVAL_POLICY_PATH", str(globals().get("ACT_REPAIR_NATIVE_OUTPUT", ACT_POLICY_PATH)))),
    LONG_OUTPUT,
    "ACT_EVAL_POLICY_PATH",
)
FULL_EVAL_ROWS = run_eval_policy_in_notebook(
    "act",
    FULL_EVAL_POLICY,
    FULL_EVAL_RESULT,
    episodes=FULL_EVAL_EPISODES,
    seed_start=1030,
    seed_list=FULL_EVAL_SEEDS,
    render=True,
    enabled=RUN_EVAL,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
    video_dir=FULL_EVAL_ROOT / "videos",
    trace_dir=FULL_EVAL_ROOT / "traces",
)
'''
    display = r'''# FULL_14_MEDIA_REVIEW
from IPython.display import Video
import matplotlib.pyplot as plt

def display_full_eval_media(result_path, title="Full 14-episode evaluation"):
    result_path = Path(result_path)
    if not result_path.exists():
        print("Evaluation has not run yet:", public_path(result_path))
        return
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    success = sum(bool(row.get("physical_success", row.get("success"))) for row in rows)
    print(f"{title}: physical_success={success}/{len(rows)}")
    md_table([["metric", "value"]][0], [["physical success", f"{success}/{len(rows)}"]])
    for kind in ("success", "failure"):
        row = next((item for item in rows if bool(item.get("physical_success", item.get("success"))) == (kind == "success") and item.get("video_path")), None)
        if row is None:
            print(f"No {kind} episode was found.")
            continue
        print(f"{kind} case: seed={row['seed']}")
        video_path = Path(row["video_path"])
        if video_path.exists():
            display(Video(filename=str(video_path), embed=False, width=960))
        trace_path = Path(row.get("trace_path", ""))
        if not trace_path.exists():
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        actions = np.asarray([item.get("action", []) for item in trace], dtype=np.float32)
        if actions.ndim != 2 or not len(actions):
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        channels = min(actions.shape[1], 7)
        for channel in range(channels):
            label = "gripper" if channel == 6 else f"action[{channel}]"
            ax.plot(actions[:, channel], label=label, linewidth=1.2)
        ax.set_title(f"{title} - {kind} sequence, seed={row['seed']}")
        ax.set_xlabel("policy step")
        ax.set_ylabel("normalized action / command")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        display(fig)
        plt.close(fig)

display_full_eval_media(FULL_EVAL_RESULT)
'''
    return code, display


def patch_notebook(path: Path, model: str) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def run_eval_policy_in_notebook(" in source:
            source = patch_common_helper(source)
            cell["source"] = source.splitlines(keepends=True)
        if model == "act" and "def train_act_repair15_in_notebook(" in source:
            source = source.replace(
                "    from lerobot.policies.act.configuration_act import ACTConfig\n    from lerobot.policies.act.modeling_act import ACTPolicy\n",
                "    import sys\n    lerobot_src = os.environ.get(\"LEROBOT_SRC\", \"/home/aup/jiahang/lerobot_10b7_legacy\")\n    if Path(lerobot_src).exists():\n        sys.path.insert(0, lerobot_src)\n    try:\n        from lerobot.policies.act.configuration_act import ACTConfig\n        from lerobot.policies.act.modeling_act import ACTPolicy\n    except ModuleNotFoundError:\n        from lerobot.common.policies.act.configuration_act import ACTConfig\n        from lerobot.common.policies.act.modeling_act import ACTPolicy\n",
                1,
            )
            cell["source"] = source.splitlines(keepends=True)
        if model == "pi0" and "def run_pi0_eval_native_in_notebook(" in source:
            source = source.replace(
                "    enabled=False,\n    render=False,\n):\n",
                "    enabled=False,\n    render=False,\n    video_dir=None,\n    trace_dir=None,\n):\n",
                1,
            )
            source = source.replace(
                "    from lerobot.policies.act.configuration_act import ACTConfig\n    from lerobot.policies.act.modeling_act import ACTPolicy\n",
                "    import sys\n    lerobot_src = os.environ.get(\"LEROBOT_SRC\", \"/home/aup/lerobot/src\")\n    if Path(lerobot_src).exists():\n        sys.path.insert(0, lerobot_src)\n    try:\n        from lerobot.policies.act.configuration_act import ACTConfig\n        from lerobot.policies.act.modeling_act import ACTPolicy\n    except ModuleNotFoundError:\n        from lerobot.common.policies.act.configuration_act import ACTConfig\n        from lerobot.common.policies.act.modeling_act import ACTPolicy\n",
                1,
            )
            source = source.replace(
                "    from contextlib import contextmanager\n",
                "    from contextlib import contextmanager\n    import argparse\n    media_module = load_eval_module()\n",
                1,
            )
            source = source.replace(
                "        rows = list(existing_rows)\n        for seed in notebook_progress",
                "        rows = list(existing_rows)\n        media_args = argparse.Namespace(policy=\"pi0\", video_dir=Path(video_dir) if video_dir else None, trace_dir=Path(trace_dir) if trace_dir else None)\n        selected_cases = set()\n        for seed in notebook_progress",
                1,
            )
            source = source.replace(
                "                start = time.time()\n                while action_steps",
                "                start = time.time()\n                frames = [] if video_dir is not None else None\n                trace = []\n                while action_steps",
                1,
            )
            source = source.replace(
                "                    image, wrist_image = env.grab_image()\n                    batch = {",
                "                    image, wrist_image = env.grab_image()\n                    media_module._capture_panel(frames, image, wrist_image)\n                    batch = {",
                1,
            )
            source = source.replace(
                "                    env.step(_pi0_native_eef_abs_to_env(action, env))\n                    action_steps += 1\n",
                "                    env.step(_pi0_native_eef_abs_to_env(action, env))\n                    action_steps += 1\n                    trace.append({\"step\": int(action_steps), \"action\": [float(x) for x in action]})\n",
                1,
            )
            source = source.replace(
                "                rows.append(row)\n                with result_path.open",
                "                row[\"_media\"] = media_module._write_episode_media(media_args, seed, frames, trace)\n                row = media_module.finalize_case_media(media_args, row, selected_cases)\n                rows.append(row)\n                with result_path.open",
                1,
            )
            cell["source"] = source.splitlines(keepends=True)
    code, display = media_cell(model)
    if not any("# FULL_14_EVAL_WITH_MEDIA" in "".join(c.get("source", [])) for c in nb["cells"]):
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
            "## Full 14-episode closed-loop evaluation\n\n",
            "This cell runs all 14 seeds in the Notebook kernel, saves one successful and one failed episode video, and plots their action sequences.\n",
        ]})
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)})
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": display.splitlines(keepends=True)})
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    for mode in ("ordinary", "protected"):
        for model, filename in (("smolvla", "14_smolvla_end_to_end.ipynb"), ("pi0", "15_pi0_end_to_end.ipynb"), ("act", "16_act_end_to_end.ipynb")):
            patch_notebook(WORKFLOWS / mode / filename, model)


if __name__ == "__main__":
    main()
