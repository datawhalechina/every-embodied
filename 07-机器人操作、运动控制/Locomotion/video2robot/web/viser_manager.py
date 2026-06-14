"""Viser process manager for 3D visualization."""

import asyncio
import os
import shutil
import socket
import signal
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from video2robot.config import PROJECT_ROOT
from video2robot.utils import resolve_project_dir


@dataclass
class ViserSession:
    """Runtime information for an active viser session."""

    id: str
    project: str
    port: int
    process: asyncio.subprocess.Process
    started_at: datetime
    all_tracks: bool = False
    twist: bool = False
    material_mode: str = "color"
    state: str = "starting"
    last_error: Optional[str] = None
    stop_requested_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    monitor_task: Optional[asyncio.Task] = field(default=None, repr=False)
    stop_task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_dict(self, host: Optional[str] = None) -> dict:
        """Convert the session to a JSON-serializable dict."""
        url = None
        if host:
            url = f"http://{host}:{self.port}"
        return {
            "id": self.id,
            "project": self.project,
            "port": self.port,
            "url": url,
            "state": self.state,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "all_tracks": self.all_tracks,
            "twist": self.twist,
            "material_mode": self.material_mode,
            "last_error": self.last_error,
        }


class ViserManager:
    """Manage viser visualization processes."""

    DEFAULT_PORT = int(os.environ.get("VISER_FIXED_PORT", "8789"))

    def __init__(self):
        self._sessions: dict[str, ViserSession] = {}
        self._lock = asyncio.Lock()

    def _cleanup_finished_sessions(self) -> None:
        """Drop finished sessions from the registry."""
        to_remove = []
        for project, session in self._sessions.items():
            if session.process.returncode is not None and session.state not in ("stopping", "starting"):
                to_remove.append(project)
        for project in to_remove:
            self._sessions.pop(project, None)

    def _cleanup_orphan_robot_viser_processes(self) -> None:
        """Best-effort cleanup for orphan robot_viser.py processes."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "video2robot/visualization/robot_viser.py"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return

        if result.returncode != 0 or not result.stdout.strip():
            return

        active_pids = {
            int(session.process.pid)
            for session in self._sessions.values()
            if session.process and session.process.pid
        }
        for token in result.stdout.split():
            try:
                pid = int(token)
            except ValueError:
                continue
            if pid in active_pids:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                continue

    def _get_available_port(self) -> int:
        """Use a fixed port for browser stability."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", self.DEFAULT_PORT))
                return self.DEFAULT_PORT
        except OSError:
            raise RuntimeError(
                f"Viser fixed port {self.DEFAULT_PORT} is busy. "
                "Please stop other robot_viser processes or set VISER_FIXED_PORT."
            )

    def _resolve_conda_executable(self) -> str:
        """Resolve conda executable robustly for web subprocesses."""
        conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not conda_exe and Path("/opt/conda/bin/conda").exists():
            conda_exe = "/opt/conda/bin/conda"
        if not conda_exe:
            raise FileNotFoundError(
                "Could not locate `conda` executable. "
                "Set CONDA_EXE or ensure conda is in PATH."
            )
        return conda_exe

    async def start(
        self,
        project: str,
        *,
        all_tracks: bool = True,
        twist: bool = False,
        material_mode: str = "color",
        phmr_env: str = "phmr",
    ) -> ViserSession:
        """Start viser for a project (non-blocking)."""
        async with self._lock:
            self._cleanup_finished_sessions()
            self._cleanup_orphan_robot_viser_processes()
            waiting_for: list[str] = []
            existing = self._sessions.get(project)
            if existing and existing.process.returncode is None and existing.state in {"starting", "running"}:
                if (
                    existing.all_tracks == all_tracks
                    and existing.twist == twist
                    and existing.material_mode == material_mode
                ):
                    return existing
                await self._stop_locked(project)
                waiting_for = [project]
            elif project in self._sessions:
                await self._stop_locked(project)
                waiting_for.append(project)

            for active_project in list(self._sessions.keys()):
                if active_project != project:
                    await self._stop_locked(active_project)
                    waiting_for.append(active_project)

        if waiting_for:
            await self._wait_for_projects(waiting_for)

        project_dir = resolve_project_dir(project)
        if not project_dir.exists():
            raise FileNotFoundError(f"Project not found: {project}")

        if not (project_dir / "results.pkl").exists():
            raise FileNotFoundError(f"results.pkl not found in {project}")
        if not (project_dir / "original.mp4").exists():
            raise FileNotFoundError(f"original.mp4 not found in {project}")

        has_robot = (project_dir / "robot_motion.pkl").exists() or (
            project_dir / "robot_motion_track_1.pkl"
        ).exists()
        if not has_robot:
            raise FileNotFoundError(f"Robot motion not found in {project}")

        conda_exe = self._resolve_conda_executable()
        port = self._get_available_port()
        script = PROJECT_ROOT / "video2robot" / "visualization" / "robot_viser.py"
        cmd = [
            conda_exe,
            "run",
            "-n",
            phmr_env,
            "--no-capture-output",
            "python",
            str(script),
            "--project",
            str(project_dir),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]

        if all_tracks:
            cmd.append("--all-tracks")
        if twist:
            cmd.append("--twist")
        cmd.extend(["--material-mode", material_mode])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )

        session = ViserSession(
            id=str(uuid.uuid4()),
            project=project,
            port=port,
            process=process,
            started_at=datetime.now(),
            all_tracks=all_tracks,
            twist=twist,
            material_mode=material_mode,
        )

        async with self._lock:
            self._sessions[project] = session
            session.monitor_task = asyncio.create_task(self._monitor_session(session))

        # Wait for viser server to bind to port
        await self._wait_for_port(port, timeout=10.0)

        return session

    async def stop(self, project: str) -> bool:
        """Request viser shutdown for a project."""
        async with self._lock:
            self._cleanup_finished_sessions()
            return await self._stop_locked(project)

    async def _stop_locked(self, project: str) -> bool:
        session = self._sessions.get(project)
        if not session:
            return False

        if session.state == "stopping":
            return True

        session.state = "stopping"
        session.stop_requested_at = datetime.now()

        if session.stop_task is None:
            session.stop_task = asyncio.create_task(self._terminate_process(session))

        return True

    async def stop_all(self):
        """Request shutdown for all active sessions."""
        async with self._lock:
            projects = list(self._sessions.keys())
            for project in projects:
                await self._stop_locked(project)

    async def status(self, host: Optional[str] = None) -> dict:
        """Report current viser sessions."""
        async with self._lock:
            self._cleanup_finished_sessions()
            sessions = [session.to_dict(host=host) for session in self._sessions.values()]
            return {
                "active_sessions": len(sessions),
                "sessions": sessions,
            }

    async def _monitor_session(self, session: ViserSession):
        """Promote session states and cleanup when finished."""
        try:
            await asyncio.sleep(1.0)
            async with self._lock:
                if (
                    self._sessions.get(session.project) is session
                    and session.state == "starting"
                    and session.process.returncode is None
                ):
                    session.state = "running"

            returncode = await session.process.wait()

            async with self._lock:
                if self._sessions.get(session.project) is not session:
                    return
                session.ended_at = datetime.now()
                if session.state == "stopping":
                    session.state = "stopped"
                elif returncode == 0:
                    session.state = "finished"
                else:
                    session.state = "error"
                    session.last_error = f"Exit code: {returncode}"
                self._sessions.pop(session.project, None)
        except Exception as exc:
            async with self._lock:
                if self._sessions.get(session.project) is session:
                    session.state = "error"
                    session.last_error = str(exc)
                    session.ended_at = datetime.now()
                    self._sessions.pop(session.project, None)

    async def _terminate_process(self, session: ViserSession):
        """Terminate the viser subprocess asynchronously."""
        try:
            if session.process.returncode is None:
                session.process.terminate()
                await asyncio.sleep(1.0)
            if session.process.returncode is None:
                session.process.kill()
        except ProcessLookupError:
            pass
        except Exception as exc:
            async with self._lock:
                if self._sessions.get(session.project) is session:
                    session.last_error = str(exc)

    async def _wait_for_projects(self, projects: list[str], timeout: float = 10.0):
        """Wait until the listed projects no longer have active sessions."""
        if not projects:
            return

        remaining = set(projects)
        deadline = asyncio.get_event_loop().time() + timeout

        while remaining and asyncio.get_event_loop().time() < deadline:
            async with self._lock:
                for name in list(remaining):
                    session = self._sessions.get(name)
                    if not session or session.process.returncode is not None:
                        remaining.discard(name)
            if remaining:
                await asyncio.sleep(0.2)

        if remaining:
            raise TimeoutError(
                f"Timed out stopping viser sessions: {', '.join(sorted(remaining))}"
            )

    async def _wait_for_port(self, port: int, timeout: float = 10.0):
        """Wait until the port is accepting connections."""
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    sock.connect(("127.0.0.1", port))
                    return  # Connection success = server ready
            except (ConnectionRefusedError, socket.timeout, OSError):
                await asyncio.sleep(0.3)

        raise TimeoutError(f"Viser server did not start on port {port}")


# Global viser manager instance
viser_manager = ViserManager()
