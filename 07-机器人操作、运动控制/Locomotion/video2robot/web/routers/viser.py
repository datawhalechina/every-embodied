"""Viser visualization server management API."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import os
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..viser_manager import viser_manager

router = APIRouter()
PUBLIC_HOST = os.environ.get("VISER_PUBLIC_HOST")


def _resolve_host(req: Request) -> str:
    """Return a host suitable for iframe connections."""
    if PUBLIC_HOST:
        return PUBLIC_HOST
    host_header = req.headers.get("host")
    if host_header:
        host = host_header.split(":")[0]
        if host:
            return host
    if req.url.hostname:
        return req.url.hostname
    return "localhost"


class StartViserRequest(BaseModel):
    """Request to start viser."""
    project: str
    all_tracks: bool = True
    twist: bool = False
    material_mode: str = "color"


class StopViserRequest(BaseModel):
    """Request to stop viser."""
    project: Optional[str] = None


@router.post("/start")
async def start_viser(request: StartViserRequest, req: Request):
    """Start viser visualization for a project."""
    try:
        session = await viser_manager.start(
            request.project,
            all_tracks=request.all_tracks,
            twist=request.twist,
            material_mode=request.material_mode,
        )

        host = _resolve_host(req)
        return {
            "status": "started",
            "session": session.to_dict(host=host),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_viser(payload: Optional[StopViserRequest] = None):
    """Stop viser visualization."""
    project = payload.project if payload else None

    if project:
        stopped = await viser_manager.stop(project)
        if not stopped:
            raise HTTPException(status_code=404, detail="No active session for project")
        return {"status": "stopping", "project": project}

    await viser_manager.stop_all()
    return {"status": "stopping_all"}


@router.get("/status")
async def viser_status(req: Request):
    """Get viser server status."""
    host = _resolve_host(req)
    return await viser_manager.status(host=host)
