"""Project management API."""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video2robot.config import DATA_DIR
from video2robot.utils import resolve_project_dir

router = APIRouter()


class ProjectInfo(BaseModel):
    """Project information."""
    name: str
    has_video: bool = False
    has_pose: bool = False
    has_robot: bool = False
    has_robot_twist: bool = False
    prompt: Optional[str] = None
    created_at: Optional[str] = None


class ProjectDetail(ProjectInfo):
    """Detailed project information."""
    video_size_mb: Optional[float] = None
    num_tracks: int = 0
    robot_type: Optional[str] = None


class CreateProjectRequest(BaseModel):
    """Request to create a new project."""
    name: Optional[str] = None


def _project_dir_or_400(name: str, *, create: bool = False) -> Path:
    try:
        return resolve_project_dir(name, create=create)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_project_info(project_dir: Path) -> ProjectInfo:
    """Get basic project info."""
    info = ProjectInfo(name=project_dir.name)
    
    # Check files
    info.has_video = (project_dir / "original.mp4").exists()
    info.has_pose = (project_dir / "smplx.npz").exists() or (project_dir / "results.pkl").exists()
    info.has_robot = (project_dir / "robot_motion.pkl").exists() or (project_dir / "robot_motion_track_1.pkl").exists()
    info.has_robot_twist = (project_dir / "robot_motion_twist.pkl").exists()
    
    # Load config if exists
    config_path = project_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            info.prompt = config.get("prompt")
            info.created_at = config.get("created_at")
        except Exception:
            pass
    
    # Try prompt.txt as fallback
    if not info.prompt:
        prompt_path = project_dir / "prompt.txt"
        if prompt_path.exists():
            try:
                info.prompt = prompt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
    
    return info


def _get_project_detail(project_dir: Path) -> ProjectDetail:
    """Get detailed project info."""
    basic = _get_project_info(project_dir)
    detail = ProjectDetail(**basic.model_dump())
    
    # Video size
    video_path = project_dir / "original.mp4"
    if video_path.exists():
        detail.video_size_mb = video_path.stat().st_size / (1024 * 1024)
    
    # Track count
    tracks_path = project_dir / "smplx_tracks.json"
    if tracks_path.exists():
        try:
            with open(tracks_path, "r", encoding="utf-8") as f:
                tracks = json.load(f)
            detail.num_tracks = len(tracks.get("tracks", []))
        except Exception:
            pass
    
    # Robot type from motion file
    robot_motion_path = project_dir / "robot_motion.pkl"
    if not robot_motion_path.exists():
        robot_motion_path = project_dir / "robot_motion_track_1.pkl"
    
    if robot_motion_path.exists():
        try:
            import pickle
            with open(robot_motion_path, "rb") as f:
                motion = pickle.load(f)
            detail.robot_type = motion.get("robot_type")
        except Exception:
            pass
    
    return detail


@router.get("")
async def list_projects() -> list[ProjectInfo]:
    """List all projects."""
    projects = []
    
    if not DATA_DIR.exists():
        return projects
    
    for project_dir in sorted(DATA_DIR.iterdir(), reverse=True):
        if not project_dir.is_dir():
            continue
        # Skip hidden directories
        if project_dir.name.startswith("."):
            continue
        
        try:
            info = _get_project_info(project_dir)
            projects.append(info)
        except Exception:
            continue
    
    return projects


@router.get("/{name}")
async def get_project(name: str) -> ProjectDetail:
    """Get project details."""
    project_dir = _project_dir_or_400(name)
    
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project not found: {name}")
    
    return _get_project_detail(project_dir)


@router.post("")
async def create_project(request: CreateProjectRequest) -> ProjectInfo:
    """Create a new project."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if request.name:
        project_dir = _project_dir_or_400(request.name)
        if project_dir.exists():
            raise HTTPException(status_code=400, detail=f"Project already exists: {request.name}")
    else:
        # Auto-generate name
        existing = list(DATA_DIR.glob("video_*"))
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
        
        project_dir = DATA_DIR / f"video_{num:03d}"
    
    project_dir.mkdir(parents=True, exist_ok=True)
    
    return _get_project_info(project_dir)


@router.delete("/{name}")
async def delete_project(name: str):
    """Delete a project."""
    project_dir = _project_dir_or_400(name)
    
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project not found: {name}")
    
    import shutil
    shutil.rmtree(project_dir)
    
    return {"status": "deleted", "name": name}
