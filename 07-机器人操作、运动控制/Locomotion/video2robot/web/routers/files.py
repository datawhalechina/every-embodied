"""File upload and download API."""

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video2robot.utils import resolve_project_dir, resolve_project_file

router = APIRouter()


def _project_dir_or_400(project_name: str, *, create: bool = False) -> Path:
    try:
        return resolve_project_dir(project_name, create=create)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _project_file_or_400(project_name: str, filename: str) -> Path:
    try:
        return resolve_project_file(project_name, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload/{project_name}")
async def upload_video(project_name: str, file: UploadFile = File(...)):
    """Upload a video file to a project."""
    project_dir = _project_dir_or_400(project_name, create=True)
    
    # Validate file type
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: mp4, mov, avi, webm")
    
    video_path = project_dir / "original.mp4"
    
    # Save file
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    size_mb = video_path.stat().st_size / (1024 * 1024)
    
    return {
        "status": "uploaded",
        "project": project_dir.name,
        "filename": "original.mp4",
        "size_mb": round(size_mb, 2),
    }


@router.get("/video/{project_name}")
async def get_video(project_name: str):
    """Get the video file for a project."""
    project_dir = _project_dir_or_400(project_name)
    video_path = project_dir / "original.mp4"
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{project_dir.name}.mp4",
    )


@router.get("/robot-motion/{project_name}")
async def get_robot_motion(project_name: str, track: int = 1, twist: bool = False):
    """Get robot motion data as JSON."""
    project_dir = _project_dir_or_400(project_name)
    
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Find motion file
    suffix = "_twist" if twist else ""
    if track == 1:
        motion_path = project_dir / f"robot_motion{suffix}.pkl"
        if not motion_path.exists():
            motion_path = project_dir / f"robot_motion_track_1{suffix}.pkl"
    else:
        motion_path = project_dir / f"robot_motion_track_{track}{suffix}.pkl"
    
    if not motion_path.exists():
        raise HTTPException(status_code=404, detail="Robot motion not found")
    
    import pickle
    import numpy as np
    
    with open(motion_path, "rb") as f:
        motion = pickle.load(f)
    
    # Convert numpy arrays to lists for JSON
    def to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(v) for v in obj]
        else:
            return obj
    
    return to_json_serializable(motion)


@router.get("/download/{project_name}/{filename}")
async def download_file(project_name: str, filename: str):
    """Download any file from a project."""
    file_path = _project_file_or_400(project_name, filename)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)
