"""Background task management."""

import asyncio
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from video2robot.config import PROJECT_ROOT
from video2robot.utils import resolve_project_dir


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    GENERATE_VIDEO = "generate_video"
    EXTRACT_POSE = "extract_pose"
    RETARGET = "retarget"


@dataclass
class Task:
    """Background task."""
    id: str
    type: TaskType
    project: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    stage_name: str = ""
    stage_index: int = 0
    stage_total: int = 0
    stage_progress_min: float = 0.0
    stage_progress_max: float = 1.0
    stage_expected_seconds: float = 0.0
    stage_started_at: Optional[datetime] = None
    elapsed_seconds: int = 0
    total_frames: int = 0
    current_frame: int = 0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "project": self.project,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "stage_name": self.stage_name,
            "stage_index": self.stage_index,
            "stage_total": self.stage_total,
            "elapsed_seconds": self.elapsed_seconds,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TaskManager:
    """Manage background tasks."""

    VIDEO_STAGE_TOTAL = 5
    POSE_STAGE_PADDING = 1  # init stage
    RETARGET_STAGE_PADDING = 1  # init stage

    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self._running_processes: dict[str, asyncio.subprocess.Process] = {}
    
    def create_task(self, task_type: TaskType, project: str) -> Task:
        """Create a new task."""
        task_id = str(uuid.uuid4())[:8]
        task = Task(id=task_id, type=task_type, project=resolve_project_dir(project).name)
        self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self, project: Optional[str] = None) -> list[Task]:
        """List all tasks, optionally filtered by project."""
        tasks = list(self.tasks.values())
        if project:
            tasks = [t for t in tasks if t.project == project]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def _enter_stage(
        self,
        task: Task,
        *,
        name: str,
        index: int,
        total: int,
        min_progress: float,
        max_progress: float,
        expected_seconds: float = 0.0,
    ):
        """Update task stage metadata and make progress snap to minimum."""
        changed = task.stage_name != name or task.stage_index != index
        task.stage_name = name
        task.stage_index = index
        task.stage_total = total
        task.stage_progress_min = min_progress
        task.stage_progress_max = max_progress
        task.stage_expected_seconds = max(expected_seconds, 0.0)
        if changed:
            task.stage_started_at = datetime.now()
        if task.progress < min_progress:
            task.progress = min_progress

    def _update_stage_progress(self, task: Task, ratio: float):
        """Clamp progress to the configured stage range."""
        ratio = max(0.0, min(1.0, ratio))
        span = max(task.stage_progress_max - task.stage_progress_min, 0.0)
        value = task.stage_progress_min + span * ratio
        if value > task.progress:
            task.progress = value

    def _update_stage_progress_from_elapsed(
        self,
        task: Task,
        measured_seconds: Optional[float] = None,
    ):
        """Advance progress based on elapsed time within the current stage."""
        expected = task.stage_expected_seconds
        if measured_seconds is None:
            if task.stage_started_at:
                measured_seconds = (datetime.now() - task.stage_started_at).total_seconds()
            else:
                measured_seconds = 0.0
        if expected <= 0:
            ratio = 1.0
        else:
            ratio = min(max(measured_seconds / expected, 0.0), 1.0)
        self._update_stage_progress(task, ratio)

    # Regex pattern for [Progress] markers
    _PROGRESS_PATTERN = re.compile(
        r'^\[Progress\]\s+stage=(\S+)\s+value=([\d.]+)\s+message=(.+?)(?:\s+\w+=|$)'
    )

    def _parse_progress_marker(self, line: str) -> Optional[dict]:
        """Parse [Progress] marker line.

        Format: [Progress] stage=<name> value=<0.0-1.0> message=<text> [key=value ...]

        Returns:
            dict with 'stage', 'value', 'message' if valid marker, else None
        """
        if not line.startswith("[Progress]"):
            return None

        match = self._PROGRESS_PATTERN.match(line)
        if not match:
            return None

        stage = match.group(1)
        try:
            value = float(match.group(2))
        except ValueError:
            return None
        message = match.group(3).strip()

        # Parse additional key=value pairs
        result = {"stage": stage, "value": value, "message": message}

        # Extract frames=X/Y if present
        frames_match = re.search(r'frames=(\d+)/(\d+)', line)
        if frames_match:
            result["current_frame"] = int(frames_match.group(1))
            result["total_frames"] = int(frames_match.group(2))

        return result

    def _apply_progress_marker(self, task: Task, marker: dict):
        """Apply parsed progress marker to task."""
        task.progress = max(0.0, min(1.0, marker["value"]))
        task.stage_name = marker["message"]
        if "current_frame" in marker:
            task.current_frame = marker["current_frame"]
        if "total_frames" in marker:
            task.total_frames = marker["total_frames"]

    # Veo model mapping (UI value → CLI --veo-model value)
    VEO_MODEL_MAP = {
        "veo-3.1-fast": "veo-3.1-fast-generate-preview",
        "veo-3.1": "veo-3.1-generate-preview",
        "veo-3.0-fast": "veo-3.0-fast-generate-001",
        "veo-3.0": "veo-3.0-generate-001",
        "veo-2.0": "veo-2.0-generate-001",
    }

    def _resolve_conda_executable(self) -> str:
        """Resolve conda executable robustly for non-interactive subprocesses."""
        conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not conda_exe and Path("/opt/conda/bin/conda").exists():
            conda_exe = "/opt/conda/bin/conda"
        if not conda_exe:
            raise RuntimeError(
                "Could not locate `conda` executable. "
                "Set CONDA_EXE or ensure conda is in PATH."
            )
        return conda_exe

    async def run_generate_video(
        self,
        task: Task,
        action: Optional[str] = None,
        raw_prompt: Optional[str] = None,
        model: str = "seedance",
        duration: int = 8,
        phmr_env: str = "phmr",
    ):
        """Run video generation."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.message = "Generating video..."
        video_stage_total = self.VIDEO_STAGE_TOTAL
        generation_expected = max(duration * 30, 240)
        self._enter_stage(
            task,
            name="Initializing",
            index=0,
            total=video_stage_total,
            min_progress=0.0,
            max_progress=0.05,
            expected_seconds=5,
        )

        try:
            conda_exe = self._resolve_conda_executable()
            script = PROJECT_ROOT / "scripts" / "generate_video.py"

            # Determine model parameters
            if model in self.VEO_MODEL_MAP:
                cli_model = "veo"
                veo_model_id = self.VEO_MODEL_MAP[model]
            elif model == "seedance":
                cli_model = "seedance"
                veo_model_id = None
            else:
                cli_model = model  # sora, sora-pro
                veo_model_id = None

            cmd = [
                conda_exe, "run", "-n", phmr_env, "--no-capture-output",
                "python", str(script),
                "--model", cli_model,
                "--name", task.project,
                "--duration", str(duration),
            ]

            if veo_model_id:
                cmd.extend(["--veo-model", veo_model_id])

            if action:
                cmd.extend(["--action", action])
            elif raw_prompt:
                cmd.extend(["--raw-prompt", raw_prompt])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            self._running_processes[task.id] = process

            # Read output
            start_time = datetime.now()
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str:
                    continue

                # Check for explicit progress marker first
                marker = self._parse_progress_marker(line_str)
                if marker:
                    self._apply_progress_marker(task, marker)
                    task.elapsed_seconds = int((datetime.now() - start_time).total_seconds())
                    continue

                line_lower = line_str.lower()
                task.message = line_str[-100:]
                # Update elapsed time
                task.elapsed_seconds = int((datetime.now() - start_time).total_seconds())

                if "[veo] starting" in line_lower or "[sora] creating" in line_lower:
                    self._enter_stage(
                        task,
                        name="API Request",
                        index=1,
                        total=video_stage_total,
                        min_progress=0.05,
                        max_progress=0.15,
                        expected_seconds=8,
                    )
                    self._update_stage_progress_from_elapsed(task)

                if (
                    "[veo] operation started" in line_lower
                    or "[sora] job created" in line_lower
                    or "[veo] model" in line_lower
                ):
                    self._enter_stage(
                        task,
                        name="API Request",
                        index=1,
                        total=video_stage_total,
                        min_progress=0.05,
                        max_progress=0.15,
                        expected_seconds=8,
                    )
                    self._update_stage_progress_from_elapsed(task)

                if "[veo] waiting" in line_lower or "[sora] status:" in line_lower:
                    # Generation stage (waiting + generating)
                    self._enter_stage(
                        task,
                        name="Generating",
                        index=2,
                        total=video_stage_total,
                        min_progress=0.15,
                        max_progress=0.9,
                        expected_seconds=generation_expected,
                    )
                    measured = None
                    match = re.search(r'\((\d+)\.?\d*s\)', line_str)
                    if match:
                        measured = float(match.group(1))
                        task.elapsed_seconds = int(measured)
                    status_part = ""
                    if "[sora] status:" in line_lower:
                        status_part = line_str.split("Status:")[-1].strip()
                        lower_status = status_part.lower()
                        if "completed" in lower_status:
                            measured = generation_expected
                    self._update_stage_progress_from_elapsed(task, measured_seconds=measured)

                elif "[veo] completed" in line_lower or "[sora] downloading" in line_lower:
                    self._enter_stage(
                        task,
                        name="Downloading",
                        index=3,
                        total=video_stage_total,
                        min_progress=0.9,
                        max_progress=0.97,
                        expected_seconds=30,
                    )
                    self._update_stage_progress_from_elapsed(task)

                if "[sora] status:" in line_lower and "completed" in line_lower:
                    self._enter_stage(
                        task,
                        name="Downloading",
                        index=3,
                        total=video_stage_total,
                        min_progress=0.9,
                        max_progress=0.97,
                        expected_seconds=30,
                    )
                    self._update_stage_progress_from_elapsed(task)

                if "[veo] saved" in line_lower or "[sora] saved" in line_lower:
                    self._enter_stage(
                        task,
                        name="Saving",
                        index=4,
                        total=video_stage_total,
                        min_progress=0.97,
                        max_progress=0.99,
                        expected_seconds=15,
                    )
                    self._update_stage_progress_from_elapsed(task)

            await process.wait()

            if process.returncode == 0:
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                task.message = "Video generation complete"
                task.stage_name = "Complete"
            else:
                task.status = TaskStatus.FAILED
                task.error = f"Exit code: {process.returncode}"

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)

        finally:
            task.completed_at = datetime.now()
            self._running_processes.pop(task.id, None)
    
    async def run_extract_pose(
        self,
        task: Task,
        static_camera: bool = False,
        phmr_env: str = "phmr",
    ):
        """Run pose extraction."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.message = "Extracting pose..."
        pose_stage_flow = [
            {"name": "Preparing", "keywords": ["[prompthmr] running"], "min": 0.03, "max": 0.1, "seconds": 20},
            {"name": "Video Analysis", "keywords": ["detect, segment"], "min": 0.1, "max": 0.25, "seconds": 60},
            {"name": "Camera Tracking", "keywords": ["camera motion"], "min": 0.25, "max": 0.4, "seconds": 90},
            {"name": "2D Keypoints", "keywords": ["2d keypoint"], "min": 0.4, "max": 0.55, "seconds": 80},
            {"name": "3D Pose Estimation", "keywords": ["mesh estimation"], "min": 0.55, "max": 0.7, "seconds": 120},
            {"name": "Coordinate Transform", "keywords": ["world coordinate"], "min": 0.7, "max": 0.82, "seconds": 80},
            {"name": "Optimization", "keywords": ["post optimization"], "min": 0.82, "max": 0.9, "seconds": 100},
            {"name": "SMPL-X Conversion", "keywords": ["[poseextractor]"], "min": 0.9, "max": 0.96, "seconds": 60},
            {"name": "Exporting", "keywords": ["exported", "export"], "min": 0.96, "max": 0.99, "seconds": 20},
        ]
        pose_stage_total = len(pose_stage_flow) + 1
        self._enter_stage(
            task,
            name="Initializing",
            index=0,
            total=pose_stage_total,
            min_progress=0.0,
            max_progress=0.03,
            expected_seconds=5,
        )

        try:
            conda_exe = self._resolve_conda_executable()
            project_dir = resolve_project_dir(task.project)

            script = PROJECT_ROOT / "scripts" / "extract_pose.py"
            cmd = [
                conda_exe, "run", "-n", phmr_env, "--no-capture-output",
                "python", str(script),
                "--project", str(project_dir),
            ]

            if static_camera:
                cmd.append("--static-camera")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            self._running_processes[task.id] = process

            start_time = datetime.now()
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str:
                    continue

                # Check for explicit progress marker first
                marker = self._parse_progress_marker(line_str)
                if marker:
                    self._apply_progress_marker(task, marker)
                    task.elapsed_seconds = int((datetime.now() - start_time).total_seconds())
                    continue

                task.message = line_str[-100:]
                # Update elapsed time
                task.elapsed_seconds = int((datetime.now() - start_time).total_seconds())

                line_lower = line_str.lower()
                matched = False
                for idx, stage in enumerate(pose_stage_flow):
                    if any(keyword in line_lower for keyword in stage["keywords"]):
                        self._enter_stage(
                            task,
                            name=stage["name"],
                            index=idx + 1,
                            total=pose_stage_total,
                            min_progress=stage["min"],
                            max_progress=stage["max"],
                            expected_seconds=stage["seconds"],
                        )
                        self._update_stage_progress_from_elapsed(task)
                        matched = True
                        break

                if not matched and task.stage_name and task.stage_name not in ("Complete",):
                    self._update_stage_progress_from_elapsed(task)

            await process.wait()

            if process.returncode == 0:
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                task.message = "Pose extraction complete"
                task.stage_name = "Complete"
            else:
                task.status = TaskStatus.FAILED
                task.error = f"Exit code: {process.returncode}"

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)

        finally:
            task.completed_at = datetime.now()
            self._running_processes.pop(task.id, None)
    
    async def run_retarget(
        self,
        task: Task,
        robot_type: str = "unitree_g1",
        gmr_env: str = "gmr",
        all_tracks: bool = True,
    ):
        """Run robot retargeting."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.message = "Converting to robot motion..."
        retarget_stage_total = 5 + self.RETARGET_STAGE_PADDING
        self._enter_stage(
            task,
            name="Initializing",
            index=0,
            total=retarget_stage_total,
            min_progress=0.0,
            max_progress=0.04,
            expected_seconds=5,
        )

        try:
            conda_exe = self._resolve_conda_executable()
            project_dir = resolve_project_dir(task.project)

            script = PROJECT_ROOT / "scripts" / "convert_to_robot.py"
            cmd = [
                conda_exe, "run", "-n", gmr_env, "--no-capture-output",
                "python", str(script),
                "--project", str(project_dir),
                "--robot", robot_type,
            ]

            if all_tracks:
                cmd.append("--all-tracks")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            self._running_processes[task.id] = process

            start_time = datetime.now()
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str:
                    continue

                # Check for explicit progress marker first
                marker = self._parse_progress_marker(line_str)
                if marker:
                    self._apply_progress_marker(task, marker)
                    task.elapsed_seconds = int((datetime.now() - start_time).total_seconds())
                    continue

                line_lower = line_str.lower()
                task.message = line_str[-100:]
                # Update elapsed time
                task.elapsed_seconds = int((datetime.now() - start_time).total_seconds())

                match = re.search(r'Processed\s+(\d+)/(\d+)', line_str)
                if match:
                    task.current_frame = int(match.group(1))
                    task.total_frames = int(match.group(2))
                    self._enter_stage(
                        task,
                        name="Retargeting",
                        index=3,
                        total=retarget_stage_total,
                        min_progress=0.25,
                        max_progress=0.9,
                        expected_seconds=0,
                    )
                    ratio = task.current_frame / max(task.total_frames, 1)
                    self._update_stage_progress(task, ratio)
                    continue

                if "[robotretargeter] input" in line_lower:
                    self._enter_stage(
                        task,
                        name="Loading SMPL-X",
                        index=1,
                        total=retarget_stage_total,
                        min_progress=0.04,
                        max_progress=0.12,
                        expected_seconds=10,
                    )
                    self._update_stage_progress_from_elapsed(task)
                    continue

                if "[robotretargeter] loading" in line_lower or "loading robot" in line_lower:
                    self._enter_stage(
                        task,
                        name="Loading Robot Model",
                        index=2,
                        total=retarget_stage_total,
                        min_progress=0.12,
                        max_progress=0.25,
                        expected_seconds=20,
                    )
                    self._update_stage_progress_from_elapsed(task)
                    continue

                if "[robotretargeter] retargeting" in line_lower:
                    frames_match = re.search(r'(\d+)\s+frames', line_str)
                    if frames_match:
                        task.total_frames = int(frames_match.group(1))
                    self._enter_stage(
                        task,
                        name="Retargeting",
                        index=3,
                        total=retarget_stage_total,
                        min_progress=0.25,
                        max_progress=0.9,
                        expected_seconds=max(task.total_frames / 15, 60) if task.total_frames else 120,
                    )
                    self._update_stage_progress_from_elapsed(task)
                    continue

                if "[twist]" in line_lower:
                    self._enter_stage(
                        task,
                        name="TWIST Conversion",
                        index=4,
                        total=retarget_stage_total,
                        min_progress=0.9,
                        max_progress=0.97,
                        expected_seconds=30,
                    )
                    self._update_stage_progress_from_elapsed(task)
                    continue

                if "saved" in line_lower:
                    self._enter_stage(
                        task,
                        name="Saving",
                        index=5,
                        total=retarget_stage_total,
                        min_progress=0.97,
                        max_progress=0.99,
                        expected_seconds=10,
                    )
                    self._update_stage_progress_from_elapsed(task)

            await process.wait()

            if process.returncode == 0:
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                task.message = "Robot motion conversion complete"
                task.stage_name = "Complete"
            else:
                task.status = TaskStatus.FAILED
                task.error = f"Exit code: {process.returncode}"

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)

        finally:
            task.completed_at = datetime.now()
            self._running_processes.pop(task.id, None)


# Global task manager instance
task_manager = TaskManager()
