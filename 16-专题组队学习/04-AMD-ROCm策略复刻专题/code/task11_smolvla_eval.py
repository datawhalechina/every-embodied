"""Task 11 的 SmolVLA 四 seed 脚本入口。

原 PR 通过执行 Notebook 的固定 cell 编号来复用逻辑，但 Notebook 后续增加
单元格后容易失效。现在统一调用 run_closed_loop.py，因此路径、严格成功判定、
视频编码和结果字段只维护一份。
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("POLICY_TYPE", "smolvla")
os.environ.setdefault("EVAL_SEEDS", "1000,1001,1002,1003")
os.environ.setdefault(
    "MODEL_RUN_DIR",
    str(
        Path(os.environ.get("MODEL_ROOT", TOPIC_ROOT / "checkpoints"))
        / "smolvla_weighted_000500"
    ),
)

runpy.run_path(
    str(Path(__file__).with_name("run_closed_loop.py")),
    run_name="__main__",
)
