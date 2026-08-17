#!/usr/bin/env python3
"""Add protected-checkpoint recipe cells to the end-to-end notebooks.

The notebooks have two purposes:

1. let learners run a short, cheap training/eval loop in class;
2. document the exact recipe that produced the protected public results.

Those two modes must be explicit.  Otherwise a learner sees a fresh Notebook
checkpoint and a protected score table without understanding the gap.
"""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"
MARKER = "PROTECTED_RECIPE_CELL"


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


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def insert_after_first(notebook_name: str, needle: str, cells: list[dict]) -> None:
    path = NOTEBOOK_DIR / notebook_name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    old_cells = [
        cell
        for cell in notebook.get("cells", [])
        if MARKER not in cell_source(cell) and "PROTECTED_TRAIN_CELL" not in cell_source(cell)
    ]

    # Remove accidental duplicated markdown-only post-long sections that were
    # created while iterating on the append script. Keep the code cell.
    compacted: list[dict] = []
    previous_post_markdown = False
    for cell in old_cells:
        src = cell_source(cell)
        is_post_markdown = (
            cell.get("cell_type") == "markdown"
            and src.strip().startswith("## 长训后追加评估")
        )
        is_protected_train_markdown = (
            cell.get("cell_type") == "markdown"
            and src.strip().startswith("### 可执行 protected")
        )
        is_protected_recipe_markdown = (
            cell.get("cell_type") == "markdown"
            and (
                src.strip().startswith("## Checkpoint 1.5：保护权重")
                or src.strip().startswith("## Checkpoint 1.5：Pi0 保护权重")
                or src.strip().startswith("## Checkpoint 1.5：ACT 保护结果")
            )
        )
        if is_protected_train_markdown or is_protected_recipe_markdown:
            continue
        if is_post_markdown and previous_post_markdown:
            continue
        compacted.append(cell)
        previous_post_markdown = is_post_markdown

    insert_at = None
    for idx, cell in enumerate(compacted):
        if needle in cell_source(cell):
            insert_at = idx + 1
            break
    if insert_at is None:
        insert_at = min(6, len(compacted))

    compacted[insert_at:insert_at] = cells
    notebook["cells"] = compacted
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {path}")


def smolvla_cells() -> list[dict]:
    return [
        md(
            """
            ## Checkpoint 1.5：保护权重的训练配方，不和课堂轻量训练混用

            上一个单元给出的是已经保护的 `57/60` 结果。这里说明它是怎么训练出来的。课堂默认长训只是让学习者体验完整流程；要复现保护权重，需要切到下面的 protected recipe，并跑完整训练与 60 episode strict eval。
            """
        ),
        md(
            """
            ### 结果口径对齐：本轮小面板与正式保护评估

            Notebook 后面的 eval 单元会跑一个便宜的小面板，用来确认本轮 checkpoint 能闭环执行并产出视频；正式保护评估使用更大的固定面板，才是发布权重和教程报告采用的口径。

            | 口径 | 成功率 | 评估范围 | 教学解释 |
            | --- | --- | --- | --- |
            | 本轮 Notebook 小面板 | `3/4` | post-long eval seed0-3 | 用于课堂展示和视频验收，不替代正式分数 |
            | 正式保护评估 | `57/60` | red30 + blue30 strict physical success | 红 `27/30`、蓝 `30/30`，作为默认发布权重 |
            """
        ),
        code(
            r'''
# PROTECTED_RECIPE_CELL
rows = [
    ("教学默认", "demo_data_language", "普通 EpisodeAwareSampler", "SMOLVLA_STEPS=5000", "本轮小面板 3/4", "用于课堂跑通和视频展示，不保证得到 57/60"),
    ("保护配方 parent", "demo_data_language", "基础 SmolVLA 长训", "5000 steps", "作为加权续训父权重", "先得到可用 parent checkpoint"),
    ("保护配方 weighted-blue", "demo_data_language", "蓝杯 frame/episode 加权，不复制原始 parquet", "续训 1000 steps，选择 step500", "red30 + blue30 strict", "当前重建 57/60；红 27/30，蓝 30/30"),
]
md_table(["模式", "数据", "采样/数据策略", "训练步数", "评估面板", "教学解释"], rows)

protected_env = {
    "TEACHING_RECIPE": "protected",
    "TRAIN_DATA_ROOT": str(DATA_ROOT / "demo_data_language"),
    "SMOLVLA_STEPS": "5000 + weighted-blue continuation",
    "SMOLVLA_EVAL_EPISODES": "60",
    "SMOLVLA_POLICY_PATH": str(MODEL_ROOT / "smolvla_weighted_000500" / "pretrained_model"),
}
print(json.dumps(protected_env, ensure_ascii=False, indent=2))
print("注意：weighted-blue 采样逻辑必须和 README_04/README_06 中的 Weighted sampler 一致；普通课堂长训不能冒充 protected 结果。")
'''
        ),
        md(
            """
            ### 可执行 protected 训练：parent 5000 + weighted-blue 续训

            设置 `RUN_PROTECTED_TRAIN=1` 后，这一格会在 Notebook 内部真实训练两个阶段：先训练 parent，再把 parent checkpoint 作为初始化继续做 blue 加权续训。默认关闭是为了避免课堂一打开就长训。
            """
        ),
        code(
            r'''
# PROTECTED_TRAIN_CELL
protected_train_enabled = env_flag("RUN_PROTECTED_TRAIN", False)
if not protected_train_enabled:
    print("未启动。设置 RUN_PROTECTED_TRAIN=1 后，本单元会原生训练 SmolVLA protected recipe。")
else:
    DATASET_REPO_ID = globals().get("DATASET_REPO_ID", os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language"))
    TRAIN_DATA_ROOT = globals().get("TRAIN_DATA_ROOT", Path(os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "demo_data_language")))
    CONFIG_DIR = OUTPUT_ROOT / "configs"
    RUN_ROOT = OUTPUT_ROOT / "runs" / "smolvla_protected_recipe"
    PARENT_OUTPUT = RUN_ROOT / "parent_5000"
    WEIGHTED_OUTPUT = RUN_ROOT / "weighted_blue2_step1000"
    parent_config = make_lerobot_train_config(
        "smolvla", DATASET_REPO_ID, TRAIN_DATA_ROOT, PARENT_OUTPUT,
        steps=int(os.environ.get("SMOLVLA_PARENT_STEPS", "5000")),
        batch_size=int(os.environ.get("SMOLVLA_BATCH_SIZE", "4")),
        chunk_size=50,
        n_action_steps=50,
    )
    weighted_config = make_lerobot_train_config(
        "smolvla", DATASET_REPO_ID, TRAIN_DATA_ROOT, WEIGHTED_OUTPUT,
        steps=int(os.environ.get("SMOLVLA_WEIGHTED_STEPS", "1000")),
        batch_size=int(os.environ.get("SMOLVLA_BATCH_SIZE", "4")),
        chunk_size=50,
        n_action_steps=50,
    )
    parent_path = write_json_yaml(CONFIG_DIR / "smolvla_protected_parent_5000.yaml", parent_config)
    weighted_path = write_json_yaml(CONFIG_DIR / "smolvla_protected_weighted_blue2.yaml", weighted_config)
    train_lerobot_config_in_notebook(parent_path, enabled=True, progress_name="SmolVLA protected parent")
    parent_ckpt = list_checkpoints(PARENT_OUTPUT)[-1]
    old_override = os.environ.get("SMOLVLA_PRETRAINED_PATH_OVERRIDE")
    old_mode = os.environ.get("NOTEBOOK_FRAME_WEIGHT_MODE")
    old_blue = os.environ.get("NOTEBOOK_BLUE_WEIGHT")
    os.environ["SMOLVLA_PRETRAINED_PATH_OVERRIDE"] = str(parent_ckpt)
    os.environ["NOTEBOOK_FRAME_WEIGHT_MODE"] = "blue"
    os.environ["NOTEBOOK_BLUE_WEIGHT"] = os.environ.get("SMOLVLA_BLUE_WEIGHT", "2.0")
    try:
        train_lerobot_config_in_notebook(weighted_path, enabled=True, progress_name="SmolVLA protected weighted-blue")
    finally:
        if old_override is None:
            os.environ.pop("SMOLVLA_PRETRAINED_PATH_OVERRIDE", None)
        else:
            os.environ["SMOLVLA_PRETRAINED_PATH_OVERRIDE"] = old_override
        if old_mode is None:
            os.environ.pop("NOTEBOOK_FRAME_WEIGHT_MODE", None)
        else:
            os.environ["NOTEBOOK_FRAME_WEIGHT_MODE"] = old_mode
        if old_blue is None:
            os.environ.pop("NOTEBOOK_BLUE_WEIGHT", None)
        else:
            os.environ["NOTEBOOK_BLUE_WEIGHT"] = old_blue
    print("protected candidate checkpoints:")
    list_checkpoints(WEIGHTED_OUTPUT)
'''
        ),
    ]


def pi0_cells() -> list[dict]:
    return [
        md(
            """
            ## Checkpoint 1.5：Pi0 保护权重的训练谱系

            Pi0 的 `12/14` 不是从一个普通 `PI0_STEPS=8500` 单元自然保证出来的；它来自 clean40 success-only 数据、蓝杯补权重和 S7500→S8500 门禁选择。Notebook 默认轻量训练用于理解流程，protected recipe 用于复现报告口径。
            """
        ),
        md(
            """
            ### 结果口径对齐：本轮小面板、保护评估与 hard 面板

            Pi0 的课堂小面板和保护权重门禁不是同一个结果。小面板只说明本轮普通长训 checkpoint 没有救回来；保护结果来自 clean40 success-only + blue2x 的 S8500 训练谱系。

            | 口径 | 成功率 | 评估范围 | 教学解释 |
            | --- | --- | --- | --- |
            | 本轮 Notebook 小面板 | `0/4` | post-long eval seed3000-3003 | 说明普通本轮 checkpoint 仍失败，不能写成成功 |
            | 正式保护评估 | `12/14` | full14 strict physical success | 当前 Pi0 发布候选，红 `6/7`、蓝 `6/7` |
            | unseen 面板 | `9/10` | unseen seeds 3010-3019 | 比早期 unseen `7/10` 更好 |
            | hard 面板 | `6/8` | hard8 | 尚未恢复历史 hard `8/8`，只能写成部分恢复 |
            """
        ),
        code(
            r'''
# PROTECTED_RECIPE_CELL
rows = [
    ("clean40-only S7500", "pi0_clean40_successonly", "成功轨迹过滤；不混失败轨迹", "7500 steps", "canary 2/4", "父权重，红杯开始恢复"),
    ("blue2x S8500", "clean40 + 蓝杯加权/补采样", "蓝杯权重提高，但保留红杯保护门禁", "+1000 steps", "canary 3/4 -> full14", "保护结果 12/14，unseen 9/10，hard8 6/8"),
    ("blue2x S9500", "同上", "继续加步数", "+2000 steps", "回落到 2/4", "说明不能盲目长训，S8500 才是选择点"),
]
md_table(["阶段", "数据", "采样/筛选策略", "训练步数", "门禁", "结论"], rows)

protected_env = {
    "TEACHING_RECIPE": "protected",
    "PI0_TRAIN_DATA_ROOT": str(DATA_ROOT / "pi0_clean40_successonly"),
    "PI0_STEPS": "8500",
    "PI0_BATCH_SIZE": "2",
    "PI0_EVAL_EPISODES": "14",
    "PI0_EVAL_SEED_START": "3000",
    "PI0_POLICY_PATH": str(MODEL_ROOT / "pi0_clean40_successonly_blue2x_s8500" / "pretrained_model"),
}
print(json.dumps(protected_env, ensure_ascii=False, indent=2))
print("注意：这条线只能写成 Pi0 部分恢复。hard8 历史 8/8 没恢复，不能把 scaffold 或小面板当 raw Pi0 成功。")
'''
        ),
        md(
            """
            ### 可执行 protected 训练：clean40 S7500 + blue2x S8500

            设置 `RUN_PROTECTED_TRAIN=1` 后，这一格会先用 clean40 success-only 数据训练 S7500，再从 S7500 初始化做 blue2x 续训到 S8500。这样 Notebook 训练路径和 `12/14` 保护结果的因果链一致。

            如果课堂或云端时长不够，也可以提供 `PI0_PARENT_CHECKPOINT_PATH=/path/to/s7500/pretrained_model`，Notebook 会跳过 S7500 父权重长训，直接从该父权重原生执行 blue2x 续训。这不是换方法，而是复用已经训练好的中间保护节点。
            """
        ),
        code(
            r'''
# PROTECTED_TRAIN_CELL
protected_train_enabled = env_flag("RUN_PROTECTED_TRAIN", False)
if not protected_train_enabled:
    print("未启动。设置 RUN_PROTECTED_TRAIN=1 后，本单元会原生训练 Pi0 protected recipe。")
else:
    DATASET_REPO_ID = globals().get("DATASET_REPO_ID", "datawhale_eai_pnp_pi0_clean_oracle_y060z000_3100_3139_g8_rebuild_v1")
    TRAIN_DATA_ROOT = globals().get("TRAIN_DATA_ROOT", Path(os.environ.get("PI0_TRAIN_DATA_ROOT", DATA_ROOT / "pi0_clean40_successonly")))
    CONFIG_DIR = OUTPUT_ROOT / "configs"
    RUN_ROOT = OUTPUT_ROOT / "runs" / "pi0_protected_recipe"
    CLEAN_OUTPUT = RUN_ROOT / "clean40_s7500"
    BLUE_OUTPUT = RUN_ROOT / "blue2x_s8500"
    clean_config = make_lerobot_train_config(
        "pi0", DATASET_REPO_ID, TRAIN_DATA_ROOT, CLEAN_OUTPUT,
        steps=int(os.environ.get("PI0_CLEAN_STEPS", "7500")),
        batch_size=int(os.environ.get("PI0_BATCH_SIZE", "2")),
        chunk_size=50,
        n_action_steps=50,
    )
    blue_config = make_lerobot_train_config(
        "pi0", DATASET_REPO_ID, TRAIN_DATA_ROOT, BLUE_OUTPUT,
        steps=int(os.environ.get("PI0_BLUE_STEPS", "1000")),
        batch_size=int(os.environ.get("PI0_BATCH_SIZE", "2")),
        chunk_size=50,
        n_action_steps=50,
    )
    clean_path = write_json_yaml(CONFIG_DIR / "pi0_protected_clean40_s7500.yaml", clean_config)
    blue_path = write_json_yaml(CONFIG_DIR / "pi0_protected_blue2x_s8500.yaml", blue_config)
    parent_checkpoint = os.environ.get("PI0_PARENT_CHECKPOINT_PATH") or os.environ.get("PI0_S7500_CHECKPOINT_PATH")
    if parent_checkpoint:
        clean_ckpt = Path(parent_checkpoint)
        if not clean_ckpt.exists():
            raise FileNotFoundError(f"PI0_PARENT_CHECKPOINT_PATH does not exist: {clean_ckpt}")
        print("使用已提供的 S7500 父权重，跳过 clean40 长训：", public_path(clean_ckpt))
    else:
        train_lerobot_config_in_notebook(clean_path, enabled=True, progress_name="Pi0 protected clean40")
        clean_ckpt = list_checkpoints(CLEAN_OUTPUT)[-1]
    old_override = os.environ.get("PI0_PRETRAINED_PATH_OVERRIDE")
    old_mode = os.environ.get("NOTEBOOK_FRAME_WEIGHT_MODE")
    old_blue = os.environ.get("NOTEBOOK_BLUE_WEIGHT")
    os.environ["PI0_PRETRAINED_PATH_OVERRIDE"] = str(clean_ckpt)
    os.environ["NOTEBOOK_FRAME_WEIGHT_MODE"] = "blue"
    os.environ["NOTEBOOK_BLUE_WEIGHT"] = os.environ.get("PI0_BLUE_WEIGHT", "2.0")
    try:
        train_lerobot_config_in_notebook(blue_path, enabled=True, progress_name="Pi0 protected blue2x")
    finally:
        if old_override is None:
            os.environ.pop("PI0_PRETRAINED_PATH_OVERRIDE", None)
        else:
            os.environ["PI0_PRETRAINED_PATH_OVERRIDE"] = old_override
        if old_mode is None:
            os.environ.pop("NOTEBOOK_FRAME_WEIGHT_MODE", None)
        else:
            os.environ["NOTEBOOK_FRAME_WEIGHT_MODE"] = old_mode
        if old_blue is None:
            os.environ.pop("NOTEBOOK_BLUE_WEIGHT", None)
        else:
            os.environ["NOTEBOOK_BLUE_WEIGHT"] = old_blue
    print("protected candidate checkpoints:")
    list_checkpoints(BLUE_OUTPUT)
'''
        ),
    ]


def act_cells() -> list[dict]:
    return [
        md(
            """
            ## Checkpoint 1.5：ACT 保护结果与历史结果不是同一个训练谱系

            ACT 当前重建保护候选为 `15/30`，历史教程记录是 `17/30`。Notebook 必须把不同训练谱系和评估协议分开：当前 repair15 保护候选已经可复核，但尚未完全恢复历史 17/30。
            """
        ),
        md(
            """
            ### 结果口径对齐：本轮小面板、当前重建与历史记录

            ACT 这本 Notebook 的重点是诊断，不是主推成功权重。本轮小面板、当前重建保护结果和历史最好必须分开写，避免把旧实验成绩当成本轮可复现输出。

            | 口径 | 成功率 | 评估范围 | 教学解释 |
            | --- | --- | --- | --- |
            | 本轮 Notebook 小面板 | `0/4` | post-long eval seed1030-1033 | 本轮普通训练仍未闭环成功 |
            | 当前重建保护 | `15/30` | stable61 -> dagger low-lr step1500 strict30 | 当前可复核的最佳保护候选 |
            | 旧 fallback 基线 | `7/30` | stable61 step2500 strict30 | 保留用于对照 |
            | 历史教程记录 | `17/30` | 旧 DAgger best025 v1 strict30 | 尚未重建，只作为经验目标和反例对照 |
            """
        ),
        code(
            r'''
# PROTECTED_RECIPE_CELL
rows = [
    ("教学默认", "demo_data_language", "普通 ACT 训练", "ACT_STEPS=5000", "本轮小面板 0/4", "用于理解 ACT 闭环失败诊断"),
    ("当前 protected repair15", "act_base72_plus_dagger3x3_rebuild_v1", "从 stable61 低学习率续训；chunk20/n10；timestamp offset 49-80:2.0", "2500 continuation steps", "strict30", "当前保护 15/30；三面板 3/10 + 4/10 + 8/10"),
    ("旧 fallback 基线", "stable61 数据/配置", "中间 checkpoint 选择，step2500 优于 step5000", "约 5000 steps", "strict30", "7/30"),
    ("历史教程最好", "旧 DAgger best025 v1", "on-policy/recovery 经验更完整", "历史实验", "strict30", "17/30；尚未完全恢复，不能当当前权重结果"),
]
md_table(["模式", "数据", "训练/选择策略", "训练步数", "评估面板", "结论"], rows)

act_recipe = os.environ.get("ACT_RECIPE", "stable61").strip().lower()
is_repair = act_recipe in {"repair15", "stable61_to_dagger", "nomemleak"}
protected_env = {
    "TEACHING_RECIPE": "protected-repair15" if is_repair else "protected-fallback",
    "ACT_RECIPE": act_recipe,
    "ACT_TRAIN_DATA_ROOT": str(DATA_ROOT / ("act_base72_plus_dagger3x3_rebuild_v1" if is_repair else "demo_data_language")),
    "ACT_STEPS": "2500 continuation" if is_repair else "5000",
    "ACT_EVAL_EPISODES": "30",
    "ACT_EVAL_SEED_START": "1030",
    "ACT_EVAL_N_ACTION_STEPS": "10" if is_repair else "50",
    "ACT_EVAL_CLAMP_TIMESTAMP": "1" if is_repair else "0",
    "ACT_EVAL_EPISODE_TIMESTAMP_OFFSETS": "49-80:2.0" if is_repair else "",
    "ACT_POLICY_PATH": str(MODEL_ROOT / ("act_stable61_to_dagger_nomemleak_step1500_strict15of30" if is_repair else "act_stable61_step2500") / "pretrained_model"),
}
print(json.dumps(protected_env, ensure_ascii=False, indent=2))
print("注意：ACT 在教程里是负例/诊断线。真正学习点是 strict success、视频复核、分阶段失败桶和保护 checkpoint 选择。")
'''
        ),
        md(
            """
            ### 可执行 protected-current 训练：stable61 fallback 或 repair15

            设置 `RUN_PROTECTED_TRAIN=1` 后，这一格会原生训练 ACT protected recipe。默认是 stable61/fallback；设置 `ACT_RECIPE=repair15` 会准备当前保护候选的 `chunk20/n10` continuation 配置，并在路径存在时加载 stable61 初始化。历史 `17/30` 尚未完全重建，因此 Notebook 不冒充历史最好。
            """
        ),
        code(
            r'''
# PROTECTED_TRAIN_CELL
protected_train_enabled = env_flag("RUN_PROTECTED_TRAIN", False)
if not protected_train_enabled:
    print("未启动。设置 RUN_PROTECTED_TRAIN=1 后，本单元会原生训练 ACT protected recipe；设置 ACT_RECIPE=repair15 可准备当前 15/30 保护协议。")
else:
    DATASET_REPO_ID = globals().get("DATASET_REPO_ID", os.environ.get("ACT_DATASET_REPO_ID", "datawhale_eai_pnp_language"))
    TRAIN_DATA_ROOT = globals().get("TRAIN_DATA_ROOT", Path(os.environ.get("ACT_TRAIN_DATA_ROOT", DATA_ROOT / "demo_data_language")))
    CONFIG_DIR = OUTPUT_ROOT / "configs"
    RUN_ROOT = OUTPUT_ROOT / "runs" / "act_protected_current_recipe"
    if is_repair:
        REPAIR_OUTPUT = RUN_ROOT / "repair15_continuation"
        os.environ.setdefault("ACT_PRETRAINED_PATH_OVERRIDE", str(MODEL_ROOT / "act_stable61_step2500" / "pretrained_model"))
        repair_config = make_lerobot_train_config(
            "act", DATASET_REPO_ID,
            Path(os.environ.get("ACT_REPAIR_TRAIN_DATA_ROOT", TRAIN_DATA_ROOT)),
            REPAIR_OUTPUT,
            steps=int(os.environ.get("ACT_REPAIR_STEPS", "2500")),
            batch_size=int(os.environ.get("ACT_BATCH_SIZE", "16")),
            chunk_size=20,
            n_action_steps=10,
        )
        repair_path = write_json_yaml(CONFIG_DIR / "act_protected_repair15.yaml", repair_config)
        train_lerobot_config_in_notebook(repair_path, enabled=True, progress_name="ACT protected repair15")
        list_checkpoints(REPAIR_OUTPUT)
    else:
        STABLE_OUTPUT = RUN_ROOT / "stable61_full"
        stable_config = make_lerobot_train_config(
            "act", DATASET_REPO_ID,
            Path(os.environ.get("ACT_STABLE61_TRAIN_DATA_ROOT", TRAIN_DATA_ROOT)),
            STABLE_OUTPUT,
            steps=int(os.environ.get("ACT_STEPS", "5000")),
            batch_size=int(os.environ.get("ACT_BATCH_SIZE", "8")),
            chunk_size=50,
            n_action_steps=50,
        )
        stable_path = write_json_yaml(CONFIG_DIR / "act_protected_current_stable61.yaml", stable_config)
        train_lerobot_config_in_notebook(stable_path, enabled=True, progress_name="ACT protected-current stable61")
        print("protected-current candidate checkpoints:")
        list_checkpoints(STABLE_OUTPUT)
'''
        ),
    ]


def main() -> None:
    insert_after_first("14_smolvla_end_to_end.ipynb", "Checkpoint 2：显示严格成功与失败视频", smolvla_cells())
    insert_after_first("15_pi0_end_to_end.ipynb", "Checkpoint 2：gated 权限和缓存检查", pi0_cells())
    insert_after_first("16_act_end_to_end.ipynb", "Checkpoint 2：生成配置并真实启动训练", act_cells())


if __name__ == "__main__":
    main()
