#!/usr/bin/env python3
"""Execute tutorial notebooks without requiring Jupyter/nbclient.

The AMD learning image contains the project runtime but may not include the
Jupyter execution stack. This small executor runs code cells in one shared
Python namespace and persists stream/error outputs in standard nbformat JSON.
Interactive collection and long training remain controlled by their RUN_*
flags inside the notebooks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import types
import traceback
from pathlib import Path


MAX_STREAM_CHARS = 100_000
REDACT_ENV_NAMES = [
    "NOTEBOOK_PYTHON",
    "TRAIN_DATA_ROOT",
    "COLLECTION_ROOT",
    "EVAL_DATA_ROOT",
    "OUTPUT_ROOT",
    "MODEL_ROOT",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "NOTEBOOK_TOPIC_ROOT",
]


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def redact_paths(value: str) -> str:
    replacements = []
    for name in REDACT_ENV_NAMES:
        path = os.environ.get(name)
        if path:
            replacements.append((str(Path(path).expanduser()), f"${name}"))
    for path, label in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        value = value.replace(path, label)
    return value


def stream_output(name: str, value: str) -> dict | None:
    if not value:
        return None
    value = redact_paths(value)
    if len(value) > MAX_STREAM_CHARS:
        value = value[:MAX_STREAM_CHARS] + "\n... output truncated by tutorial executor ...\n"
    return {"name": name, "output_type": "stream", "text": value}


class Markdown(str):
    pass


class HTML(str):
    pass


class Image:
    def __init__(self, filename: str | None = None, width: int | None = None, **_: object) -> None:
        self.filename = filename
        self.width = width


class DisplayCapture:
    def __init__(self) -> None:
        self.outputs: list[dict] = []

    def display(self, *objects: object, **_: object) -> None:
        for obj in objects:
            self.outputs.append(self.to_output(obj))

    def to_output(self, obj: object) -> dict:
        if isinstance(obj, HTML):
            data = {"text/html": redact_paths(str(obj))}
        elif isinstance(obj, Markdown):
            data = {"text/markdown": redact_paths(str(obj))}
        elif isinstance(obj, Image):
            if obj.filename:
                width = f" width='{int(obj.width)}'" if obj.width else ""
                data = {"text/html": f"<img src='{redact_paths(str(obj.filename))}'{width}>"}
            else:
                data = {"text/plain": "[image]"}
        else:
            data = {"text/plain": redact_paths(str(obj))}
        return {"output_type": "display_data", "data": data, "metadata": {}}


_CURRENT_DISPLAY_CAPTURE: DisplayCapture | None = None


def display_proxy(*objects: object, **kwargs: object) -> None:
    if _CURRENT_DISPLAY_CAPTURE is None:
        print(*objects)
        return
    _CURRENT_DISPLAY_CAPTURE.display(*objects, **kwargs)


def install_display_shim(capture: DisplayCapture) -> None:
    """Provide a tiny IPython.display module for teaching images/videos.

    The AMD learning environment often does not install IPython/nbclient. The
    notebooks still use ``display(HTML(...))`` for videos, so the lightweight
    executor records those calls as normal notebook ``display_data`` outputs.
    """

    ipython_mod = sys.modules.setdefault("IPython", types.ModuleType("IPython"))
    display_mod = types.ModuleType("IPython.display")
    display_mod.HTML = HTML
    display_mod.Markdown = Markdown
    display_mod.Image = Image
    global _CURRENT_DISPLAY_CAPTURE
    _CURRENT_DISPLAY_CAPTURE = capture
    display_mod.display = display_proxy
    ipython_mod.display = display_mod
    sys.modules["IPython.display"] = display_mod


def cell_matches(cell: dict, markers: list[str]) -> bool:
    if not markers:
        return True
    source = cell_source(cell)
    return any(marker in source for marker in markers)


def execute_notebook(
    path: Path,
    stop_on_error: bool = True,
    record_markers: list[str] | None = None,
    prepare_markers: list[str] | None = None,
) -> tuple[int, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__notebook__",
        "__file__": str(path),
    }
    execution_count = 0
    errors = 0
    record_markers = record_markers or []
    prepare_markers = prepare_markers or []

    old_cwd = Path.cwd()
    os.chdir(path.parent.parent)
    try:
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            should_record = cell_matches(cell, record_markers)
            should_prepare = bool(prepare_markers and cell_matches(cell, prepare_markers))
            if record_markers and not should_record and not should_prepare:
                continue
            if should_record:
                execution_count += 1
                cell["execution_count"] = execution_count
                cell["outputs"] = []
            stdout = io.StringIO()
            stderr = io.StringIO()
            source = cell_source(cell)
            capture = DisplayCapture()
            install_display_shim(capture)
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(source, f"{path.name}:cell-{cell_index}", "exec"), namespace)
            except Exception as exc:
                errors += 1
                out = stream_output("stdout", stdout.getvalue())
                err = stream_output("stderr", stderr.getvalue())
                if should_record:
                    if out:
                        cell["outputs"].append(out)
                    if err:
                        cell["outputs"].append(err)
                    cell["outputs"].extend(capture.outputs)
                    cell["outputs"].append(
                        {
                            "ename": type(exc).__name__,
                            "evalue": redact_paths(str(exc)),
                            "output_type": "error",
                            "traceback": redact_paths(traceback.format_exc()).splitlines(),
                        }
                    )
                if stop_on_error:
                    break
            else:
                out = stream_output("stdout", stdout.getvalue())
                err = stream_output("stderr", stderr.getvalue())
                if should_record:
                    if out:
                        cell["outputs"].append(out)
                    if err:
                        cell["outputs"].append(err)
                    cell["outputs"].extend(capture.outputs)
    finally:
        os.chdir(old_cwd)

    metadata = notebook.setdefault("metadata", {})
    metadata["amd_rocm_tutorial_execution"] = {
        "executor": "code/execute_tutorial_notebooks.py",
        "python": sys.version.split()[0],
        "code_cells_executed": execution_count,
        "errors": errors,
        "record_markers": record_markers,
        "prepare_markers": prepare_markers,
        "long_tasks_enabled": False,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return execution_count, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", type=Path, nargs="+")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later cells after an exception.",
    )
    parser.add_argument(
        "--record-marker",
        action="append",
        default=[],
        help="Only record outputs for code cells whose source contains this marker. Can be repeated.",
    )
    parser.add_argument(
        "--prepare-marker",
        action="append",
        default=[],
        help="Execute matching cells silently before recorded cells. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_errors = 0
    for path in args.notebooks:
        cells, errors = execute_notebook(
            path.resolve(),
            stop_on_error=not args.keep_going,
            record_markers=args.record_marker,
            prepare_markers=args.prepare_marker,
        )
        total_errors += errors
        print(f"{path}: executed={cells}, errors={errors}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
