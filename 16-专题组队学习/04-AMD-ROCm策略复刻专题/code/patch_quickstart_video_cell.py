#!/usr/bin/env python3
"""Idempotently add the zero-training video preview to Notebook 11."""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = TOPIC_ROOT / "notebooks" / "11_mujoco_closed_loop_deploy.ipynb"
MARKER = "## 零训练成功预览：先看正确行为"
VIDEO_SOURCE = """
SUCCESS_PREVIEW = ASSET_DIR / "pnp_four_view_strict_success.mp4"
if not SUCCESS_PREVIEW.exists():
    raise FileNotFoundError(f"缺少成功预览视频：{SUCCESS_PREVIEW}")

try:
    from IPython.display import Video
except ImportError:
    print("当前是纯 Python 执行器；在 Jupyter 中运行本单元格即可嵌入播放视频。")
else:
    display(Video(filename=str(SUCCESS_PREVIEW), embed=True, html_attributes="controls muted"))

print("严格成功预览：", SUCCESS_PREVIEW)
"""


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if MARKER in "".join(cell.get("source", [])):
            if index + 1 >= len(cells) or cells[index + 1].get("cell_type") != "code":
                raise RuntimeError("预览说明后缺少视频代码单元格。")
            expected_source = VIDEO_SOURCE.strip() + "\n"
            current_source = "".join(cells[index + 1].get("source", []))
            if current_source == expected_source:
                print("Notebook 11 already contains the current zero-training preview.")
                return
            cells[index + 1]["source"] = expected_source
            cells[index + 1]["execution_count"] = None
            cells[index + 1]["outputs"] = []
            NOTEBOOK.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            print("Updated the existing zero-training preview cell.")
            return

    insert_at = 3
    cells[insert_at:insert_at] = [
        markdown_cell(
            """
## 零训练成功预览：先看正确行为

这一格不加载模型、不启动 MuJoCo，也不会消耗训练额度。它直接显示仓库内置的四视角严格成功回放，先确认接近、夹取、抬升、搬运、释放和稳定放置的正确顺序。视频来自固定环境上的 `pi0 + visual/history learned head`，不是 raw pi0，也不能代表随机位置泛化。
"""
        ),
        code_cell(VIDEO_SOURCE),
    ]
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Patched {NOTEBOOK}")


if __name__ == "__main__":
    main()
