"""Shared utilities for video2robot."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from video2robot.config import DATA_DIR


def resolve_project_dir(name: str, *, create: bool = False) -> Path:
    """Resolve a project name to a direct child of DATA_DIR.

    Web requests should pass project names, not arbitrary filesystem paths.
    Keeping this validation in one place prevents path traversal in upload,
    download, delete, and background-task entry points.
    """
    if not isinstance(name, str):
        raise ValueError("Project name must be a string.")

    name = name.strip()
    if not name:
        raise ValueError("Project name cannot be empty.")
    if name in {".", ".."}:
        raise ValueError("Project name cannot be '.' or '..'.")
    if "\x00" in name:
        raise ValueError("Project name cannot contain NUL bytes.")
    if "/" in name or "\\" in name:
        raise ValueError("Project name cannot contain path separators.")
    if len(name) > 128:
        raise ValueError("Project name is too long.")

    data_root = DATA_DIR.resolve()
    project_dir = (data_root / name).resolve()
    if project_dir.parent != data_root:
        raise ValueError("Project name must resolve inside the data directory.")

    if create:
        project_dir.mkdir(parents=True, exist_ok=True)

    return project_dir


def resolve_project_file(project_name: str, filename: str) -> Path:
    """Resolve a single downloadable file inside a validated project."""
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")

    filename = filename.strip()
    if not filename:
        raise ValueError("Filename cannot be empty.")
    if filename in {".", ".."}:
        raise ValueError("Filename cannot be '.' or '..'.")
    if "\x00" in filename:
        raise ValueError("Filename cannot contain NUL bytes.")
    if "/" in filename or "\\" in filename:
        raise ValueError("Filename cannot contain path separators.")

    project_dir = resolve_project_dir(project_name)
    file_path = (project_dir / filename).resolve()
    if file_path.parent != project_dir:
        raise ValueError("Filename must resolve inside the project directory.")

    return file_path


def emit_progress(stage: str, value: float, message: str, **kwargs):
    """Emit a standardized progress marker for TaskManager parsing.

    Format: [Progress] stage=<name> value=<0.0-1.0> message=<text> [key=value ...]

    Args:
        stage: Stage identifier (e.g., "init", "generating", "done")
        value: Progress value between 0.0 and 1.0
        message: Human-readable status message
        **kwargs: Additional key-value pairs (e.g., frames="100/200")
    """
    value = max(0.0, min(1.0, value))
    parts = [f"[Progress] stage={stage} value={value:.2f} message={message}"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


def get_next_project_dir(prefix: str = "video") -> Path:
    """Get next available project directory (video_001, video_002, ...)"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(DATA_DIR.glob(f"{prefix}_*"))

    if not existing:
        num = 1
    else:
        nums = []
        for d in existing:
            try:
                nums.append(int(d.name.split("_")[-1]))
            except ValueError:
                continue
        num = max(nums) + 1 if nums else 1

    return DATA_DIR / f"{prefix}_{num:03d}"


def ensure_project_dir(
    project_path: Optional[str | Path] = None,
    name: Optional[str] = None,
) -> Path:
    """Ensure project directory exists and return its path.

    Args:
        project_path: Explicit project path (takes priority)
        name: Folder name under DATA_DIR (used if project_path is None)

    Returns:
        Resolved project directory path

    Examples:
        >>> ensure_project_dir("/path/to/project")  # Use explicit path
        >>> ensure_project_dir(name="my_project")   # DATA_DIR/my_project
        >>> ensure_project_dir()                    # Auto-generate video_XXX
    """
    if project_path:
        path = Path(project_path)
    elif name:
        path = resolve_project_dir(name)
    else:
        path = get_next_project_dir()

    path.mkdir(parents=True, exist_ok=True)
    return path


def run_in_conda(env_name: str, argv: list[str], cwd: Path, *, raise_on_error: bool = True):
    """Run a command inside a conda environment.

    Args:
        env_name: Conda environment name
        argv: Command arguments
        cwd: Working directory
        raise_on_error: If True, raise RuntimeError on failure; if False, print error
    """
    # Resolve conda executable robustly (interactive shells may have conda as a function only).
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_exe and Path("/opt/conda/bin/conda").exists():
        conda_exe = "/opt/conda/bin/conda"

    if conda_exe:
        cmd = [conda_exe, "run", "--no-capture-output", "-n", env_name, *argv]
    else:
        msg = (
            "Could not locate `conda` executable. "
            "Set CONDA_EXE or ensure conda is in PATH."
        )
        if raise_on_error:
            raise RuntimeError(msg)
        print(f"[Error] {msg}")
        print("[Info] Falling back to current environment.")
        cmd = argv
    try:
        result = subprocess.run(cmd, cwd=str(cwd))
    except KeyboardInterrupt:
        if raise_on_error:
            raise RuntimeError("Interrupted by user.") from None
        print("\n[Info] Interrupted.")
        return
    if result.returncode != 0:
        msg = f"Command failed (env={env_name}): {' '.join(argv)}"
        if raise_on_error:
            raise RuntimeError(msg)
        print(f"[Error] {msg}")
