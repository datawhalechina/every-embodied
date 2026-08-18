#!/usr/bin/env python3
"""Append idempotent post-long-training eval cells to model notebooks.

These cells are deliberately placed at the end of each notebook so a finished
long-training transcript is not overwritten. Execute them with:

  RUN_POST_LONG_EVAL=1 EVAL_USE_LONG_TRAIN=1 \
  python code/execute_tutorial_notebooks.py \
    --prepare-marker "from pathlib import Path" \
    --prepare-marker "def md_table(headers, rows)" \
    --prepare-marker "<model config marker>" \
    --record-marker "POST_LONG_EVAL_CELL" notebooks/...
"""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"
MARKER = "POST_LONG_EVAL_CELL"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def append_cells(notebook_name: str, cells: list[dict]) -> None:
    path = NOTEBOOK_DIR / notebook_name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell
        for cell in notebook.get("cells", [])
        if MARKER not in ("".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", "")))
    ]
    notebook["cells"].extend(cells)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {path}")


def smolvla_cells() -> list[dict]:
    return [
        md(
            """
            ## 长训后追加评估：本轮 checkpoint 的 strict closed-loop

            这个单元只用于长训完成后的收尾执行。它会强制使用本次 `LONG_OUTPUT` 里的最新 checkpoint，并把结果写到 `post_long_eval/`，避免和前面“历史保护权重”结果混在一起。
            """
        ),
        code(
            r'''
# POST_LONG_EVAL_CELL
post_eval_enabled = env_flag("RUN_POST_LONG_EVAL", False)
os.environ.setdefault("EVAL_USE_LONG_TRAIN", "1")
post_eval_dir = OUTPUT_ROOT / "post_long_eval"
post_eval_dir.mkdir(parents=True, exist_ok=True)
eval_episodes = os.environ.get("SMOLVLA_POST_EVAL_EPISODES", os.environ.get("SMOLVLA_EVAL_EPISODES", "10"))
checkpoints = list_checkpoints(LONG_OUTPUT)
if post_eval_enabled and not checkpoints:
    raise RuntimeError("没有找到本轮 SmolVLA 长训 checkpoint，不能回退到旧权重冒充本轮 eval。")
eval_policy = checkpoints[-1] if checkpoints else resolve_eval_policy(PRETRAINED_POLICY, LONG_OUTPUT, "SMOLVLA_EVAL_POLICY_PATH")
result_path = post_eval_dir / f"smolvla_post_long_seed0_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "smolvla",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=os.environ.get("SMOLVLA_EVAL_SEED_START", "0"),
    render=env_flag("RENDER_EVAL"),
    enabled=post_eval_enabled,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
rows = summarize_jsonl(result_path)
display(Markdown("**视频复核入口**：下面展示教程内已准备好的成功/失败视频，真实 eval 的 JSONL 路径见上表。"))
show_video("smolvla_weighted500_red_success_seed0.mp4", "SmolVLA 红杯成功参考视频")
show_video("smolvla_weighted500_blue_success_seed0.mp4", "SmolVLA 蓝杯成功参考视频")
show_video("smolvla_weighted500_red_failure_seed8.mp4", "SmolVLA 失败参考视频：观察 release/upright")
''',
        ),
    ]


def pi0_cells() -> list[dict]:
    return [
        md(
            """
            ## 长训后追加评估：Pi0 本轮 checkpoint 的 strict closed-loop

            这个单元只在 Pi0 长训完成后执行。它会强制使用本轮 `LONG_OUTPUT` 最新 checkpoint，按 Pi0 数据集对应的 8 维 `observation.state` 协议评估，并把结果写到 `post_long_eval/`。
            """
        ),
        code(
            r'''
# POST_LONG_EVAL_CELL
post_eval_enabled = env_flag("RUN_POST_LONG_EVAL", False)
os.environ.setdefault("EVAL_USE_LONG_TRAIN", "1")
post_eval_dir = OUTPUT_ROOT / "post_long_eval"
post_eval_dir.mkdir(parents=True, exist_ok=True)
eval_episodes = os.environ.get("PI0_POST_EVAL_EPISODES", os.environ.get("PI0_EVAL_EPISODES", "14"))
eval_seed_start = os.environ.get("PI0_EVAL_SEED_START", "3000")
checkpoints = list_checkpoints(LONG_OUTPUT)
if post_eval_enabled and not checkpoints:
    raise RuntimeError("没有找到本轮 Pi0 长训 checkpoint，不能回退到旧权重冒充本轮 eval。")
eval_policy = checkpoints[-1] if checkpoints else resolve_eval_policy(PI0_POLICY_PATH, LONG_OUTPUT, "PI0_EVAL_POLICY_PATH")
result_path = post_eval_dir / f"pi0_post_long_seed{eval_seed_start}_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "pi0",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=eval_seed_start,
    render=env_flag("RENDER_EVAL"),
    enabled=post_eval_enabled,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
rows = summarize_jsonl(result_path)
display(Markdown("**视频复核入口**：Pi0 raw 结果必须和 scaffold/finisher 分开读，下面是教程内诊断参考视频。"))
show_video("pi0_ep2_raw_vs_finisher_side_by_side.mp4", "Pi0 raw 与诊断 finisher 对照视频")
show_image("pi0_raw_vs_finisher_diagnostic.png", "Pi0 诊断图：raw、scaffold 与严格判定")
''',
        ),
    ]


def act_cells() -> list[dict]:
    return [
        md(
            """
            ## 长训后追加评估：ACT 本轮 checkpoint 的 strict closed-loop

            这个单元只用于 ACT 长训完成后的收尾执行。它会使用本次 `LONG_OUTPUT` 里的最新 checkpoint，并把结果写到 `post_long_eval/`。ACT 当前是诊断反例，结果要以 strict physical success 和视频复核为准。
            """
        ),
        code(
            r'''
# POST_LONG_EVAL_CELL
post_eval_enabled = env_flag("RUN_POST_LONG_EVAL", False)
os.environ.setdefault("EVAL_USE_LONG_TRAIN", "1")
post_eval_dir = OUTPUT_ROOT / "post_long_eval"
post_eval_dir.mkdir(parents=True, exist_ok=True)
eval_episodes = os.environ.get("ACT_POST_EVAL_EPISODES", os.environ.get("ACT_EVAL_EPISODES", "6"))
eval_seed_start = os.environ.get("ACT_EVAL_SEED_START", "1030")
checkpoints = list_checkpoints(LONG_OUTPUT)
if post_eval_enabled and not checkpoints:
    raise RuntimeError("没有找到本轮 ACT 长训 checkpoint，不能回退到旧权重冒充本轮 eval。")
eval_policy = checkpoints[-1] if checkpoints else resolve_eval_policy(ACT_POLICY_PATH, LONG_OUTPUT, "ACT_EVAL_POLICY_PATH")
result_path = post_eval_dir / f"act_post_long_seed{eval_seed_start}_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "act",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=eval_seed_start,
    render=env_flag("RENDER_EVAL"),
    enabled=post_eval_enabled,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
rows = summarize_jsonl(result_path)
display(Markdown("**视频复核入口**：ACT 这条线重点看失败阶段，不只看 loss。"))
show_image("act_dagger_progress_curve.png", "ACT DAgger 进展曲线")
display(Markdown("本轻量分支没有提交原始 ACT rollout 视频；请使用自己的 OUTPUT_ROOT 重新生成视频后再做逐帧复核。"))
''',
        ),
    ]


def main() -> None:
    append_cells("14_smolvla_end_to_end.ipynb", smolvla_cells())
    append_cells("15_pi0_end_to_end.ipynb", pi0_cells())
    append_cells("16_act_end_to_end.ipynb", act_cells())


if __name__ == "__main__":
    main()
