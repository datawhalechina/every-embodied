"""Pipeline execution API."""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video2robot.utils import resolve_project_dir
from ..tasks import task_manager, TaskType

router = APIRouter()


class GenerateVideoRequest(BaseModel):
    """Request to generate video."""
    project: str
    action: Optional[str] = None
    raw_prompt: Optional[str] = None
    model: str = "seedance"  # seedance, veo-3.1-fast, veo-3.1, veo-3.0-fast, veo-3.0, veo-2.0, sora, sora-pro
    duration: int = 8


class ExtractPoseRequest(BaseModel):
    """Request to extract pose."""
    project: str
    static_camera: bool = False


class RetargetRequest(BaseModel):
    """Request to retarget to robot."""
    project: str
    robot_type: str = "unitree_g1"
    all_tracks: bool = True


class RunPipelineRequest(BaseModel):
    """Request to run full pipeline."""
    project: Optional[str] = None
    action: Optional[str] = None
    raw_prompt: Optional[str] = None
    video_path: Optional[str] = None
    model: str = "seedance"
    duration: int = 8
    robot_type: str = "unitree_g1"
    static_camera: bool = False


def _project_dir_or_400(name: str, *, create: bool = False) -> Path:
    try:
        return resolve_project_dir(name, create=create)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/generate-video")
async def generate_video(request: GenerateVideoRequest, background_tasks: BackgroundTasks):
    """Start video generation task."""
    if not request.action and not request.raw_prompt:
        raise HTTPException(status_code=400, detail="Either action or raw_prompt is required")
    
    # Create project dir if needed
    project_dir = _project_dir_or_400(request.project, create=True)
    
    # Create and start task
    task = task_manager.create_task(TaskType.GENERATE_VIDEO, project_dir.name)
    
    background_tasks.add_task(
        task_manager.run_generate_video,
        task,
        action=request.action,
        raw_prompt=request.raw_prompt,
        model=request.model,
        duration=request.duration,
    )
    
    return task.to_dict()


@router.post("/extract-pose")
async def extract_pose(request: ExtractPoseRequest, background_tasks: BackgroundTasks):
    """Start pose extraction task."""
    project_dir = _project_dir_or_400(request.project)
    
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not (project_dir / "original.mp4").exists():
        raise HTTPException(status_code=400, detail="Video not found. Upload or generate video first.")
    
    task = task_manager.create_task(TaskType.EXTRACT_POSE, project_dir.name)
    
    background_tasks.add_task(
        task_manager.run_extract_pose,
        task,
        static_camera=request.static_camera,
    )
    
    return task.to_dict()


@router.post("/retarget")
async def retarget(request: RetargetRequest, background_tasks: BackgroundTasks):
    """Start robot retargeting task."""
    project_dir = _project_dir_or_400(request.project)
    
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check for pose data
    has_pose = (project_dir / "smplx.npz").exists() or (project_dir / "results.pkl").exists()
    if not has_pose:
        raise HTTPException(status_code=400, detail="Pose data not found. Extract pose first.")
    
    task = task_manager.create_task(TaskType.RETARGET, project_dir.name)
    
    background_tasks.add_task(
        task_manager.run_retarget,
        task,
        robot_type=request.robot_type,
        all_tracks=request.all_tracks,
    )
    
    return task.to_dict()


@router.get("/tasks")
async def list_tasks(project: Optional[str] = None):
    """List all tasks."""
    tasks = task_manager.list_tasks(project)
    return [t.to_dict() for t in tasks]


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status."""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task.to_dict()
