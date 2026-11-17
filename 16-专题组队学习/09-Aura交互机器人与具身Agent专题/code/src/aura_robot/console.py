"""Local-only, single-job perception console. No motion, audio or cloud endpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import mimetypes
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid

from .readiness import check_sdk, diagnostic, explain, inventory
from .hardware_history import summarize as hardware_history


PROFILES = ('inspection', 'interaction', 'spatial', 'all')
SAMPLES = {'sample_person': 'vision-vendor-person.jpg', 'sample_text': 'vision-extra-fixture-20260908.png'}
SOURCE_LABELS = {'robot': '实机相机', 'sample_person': '离线厂商人物样例（非实机）', 'sample_text': '离线合成文字路标（非实机）'}
RUN_ID = re.compile(r'console-[0-9a-f]{32}\Z')


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


class Jobs:
    def __init__(self, root, models, host, device, timeout=180):
        self.root, self.models = Path(root).resolve(), Path(models).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.host, self.device, self.timeout = host, device, timeout
        self.lock = threading.RLock()
        self.active = None

    def start(self, request):
        if not isinstance(request, dict) or set(request)-{'profile', 'frames', 'fps', 'source'} or not {'profile', 'frames', 'fps'} <= set(request):
            raise ValueError('Expected profile, frames, fps and optional allowlisted source')
        profile, frames, fps = request['profile'], request['frames'], request['fps']
        source = request.get('source', 'robot')
        if not isinstance(source, str) or source not in SOURCE_LABELS:
            raise ValueError('Unknown source')
        sample = None
        if source != 'robot':
            if frames != 1:
                raise ValueError('Offline still samples support exactly one frame')
            sample = (self.root.parent/SAMPLES[source]).resolve()
            if sample.parent != self.root.parent or not sample.is_file():
                raise ValueError('Local sample missing; no automatic download or robot fallback')
        if profile not in PROFILES or type(frames) is not int or not 1 <= frames <= 60:
            raise ValueError('Unknown profile or frames outside 1..60')
        if type(fps) not in (int, float) or not .5 <= fps <= 5:
            raise ValueError('fps must be .5..5')
        with self.lock:
            self.poll()
            if self.active:
                raise RuntimeError('A job is already running')
            run_id = 'console-' + uuid.uuid4().hex
            directory = self.root/run_id
            directory.mkdir()
            metadata = {'id': run_id, 'profile': profile, 'frames': frames, 'fps': fps,
                        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'starting',
                        'robot_host': self.host, 'motion_sent': False, 'timeout_s': self.timeout,
                        'source': source, 'source_label': SOURCE_LABELS[source]}
            self.write_metadata(directory, metadata)
            command = [sys.executable, '-u', '-m', 'aura_robot.vision', 'run', '--profile', profile,
                       '--host', self.host, '--device', self.device, '--models', str(self.models),
                       '--frames', str(frames), '--fps', str(fps), '--output', str(directory/'output')]
            if sample is not None:
                command += ['--image', str(sample)]
            try:
                with (directory/'run.log').open('xb') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, shell=False,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except Exception as exc:
                metadata.update(status='failed', error=str(exc))
                self.write_metadata(directory, metadata)
                raise
            metadata['status'] = 'running'
            self.write_metadata(directory, metadata)
            self.active = {'process': process, 'directory': directory, 'metadata': metadata,
                           'started': time.monotonic(), 'stop_reason': None}
            return metadata

    @staticmethod
    def write_metadata(directory, value):
        pending = directory/'session.pending.json'
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        pending.replace(directory/'session.json')

    def poll(self):
        with self.lock:
            if not self.active:
                return
            job = self.active
            process = job['process']
            if process.poll() is None and time.monotonic()-job['started'] > self.timeout:
                job['stop_reason'] = 'timed_out'
                process.kill()  # Only our own bounded perception subprocess.
            code = process.poll()
            if code is not None:
                job['metadata'].update(status=job['stop_reason'] or ('completed' if code == 0 else 'failed'),
                                       exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
                self.write_metadata(job['directory'], job['metadata'])
                self.active = None

    def stop(self):
        with self.lock:
            self.poll()
            if not self.active:
                return {'status': 'idle'}
            self.active['stop_reason'] = 'stopped'
            self.active['process'].kill()
            return {'status': 'stopping', 'id': self.active['metadata']['id']}

    def state(self):
        with self.lock:
            self.poll()
            directories = sorted((p for p in self.root.iterdir() if RUN_ID.fullmatch(p.name) and p.is_dir()
                                  and not p.is_symlink()), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
            sessions = []
            for directory in directories:
                metadata = read_json(directory/'session.json')
                if not isinstance(metadata, dict):
                    continue
                if metadata.get('status') in ('starting', 'running') and (
                        not self.active or self.active['metadata']['id'] != directory.name):
                    metadata['status'] = 'unknown_after_restart'
                metadata['report'] = (directory/'output/report.html').is_file()
                sessions.append(metadata)
            return {'active': self.active['metadata']['id'] if self.active else None,
                    'sessions': sessions, 'host': self.host, 'device': self.device,
                    'limitations': ['No motion/audio/cloud actions', 'Battery percentage/charging unresolved',
                                    'Metric stereo calibration and real-person interaction not validated']}

    def detail(self, run_id):
        directory = self.directory(run_id)
        row = None
        # A worker may be halfway through writing JSON; show the latest complete frame.
        for path in sorted((directory/'output').glob('[0-9][0-9][0-9][0-9].json'), reverse=True):
            row = read_json(path)
            if row is not None:
                break
        log = directory/'run.log'
        tail = ''
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-12000))
                tail = stream.read(12000).decode('utf-8', errors='replace')
        summary = read_json(directory/'output/events.json')
        metadata = read_json(directory/'session.json') or {}
        if metadata.get('status') in ('starting', 'running') and (
                not self.active or self.active['metadata']['id'] != run_id):
            metadata['status'] = 'unknown_after_restart'
        return {'row': row, 'summary': summary, 'log': tail, 'assessment': explain(row),
                'diagnostic': diagnostic(metadata, row, summary)}

    def directory(self, run_id):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError('Invalid run ID')
        directory = (self.root/run_id).resolve()
        if directory.parent != self.root or not directory.is_dir():
            raise ValueError('Unknown run')
        return directory

    def file(self, run_id, filename):
        directory = self.directory(run_id)
        if Path(filename).name != filename or not re.fullmatch(r'[a-zA-Z0-9_.-]+', filename):
            raise ValueError('Invalid filename')
        if Path(filename).suffix not in ('.jpg', '.png', '.html', '.json'):
            raise ValueError('Unsupported output type')
        path = (directory/'output'/filename).resolve()
        if not path.is_relative_to(directory) or path.parent != directory/'output' or not path.is_file():
            raise ValueError('Unknown output')
        return path


class ConsoleServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, jobs):
        super().__init__(address, Handler)
        self.jobs, self.token = jobs, secrets.token_urlsafe(32)
        self.origin = f'http://127.0.0.1:{self.server_port}'

    def service_actions(self):
        self.jobs.poll()  # Enforce timeout even with no browser connected.


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, data, kind='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

    def do_GET(self):
        if not self.valid_host():
            return self.reply(403, {'error': 'Local host header required'})
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/':
                page = Path(__file__).with_name('console.html').read_text(encoding='utf-8')
                return self.reply(200, page.replace('__TOKEN__', self.server.token).encode('utf-8'), 'text/html; charset=utf-8')
            if path == '/api/state':
                return self.reply(200, self.server.jobs.state())
            if path == '/api/readiness':
                return self.reply(200, inventory(self.server.jobs.models))
            if path == '/api/hardware-history':
                return self.reply(200, hardware_history(self.server.jobs.root.parent))
            if path.startswith('/api/diagnostic/'):
                return self.reply(200, self.server.jobs.detail(path.removeprefix('/api/diagnostic/'))['diagnostic'])
            if path.startswith('/api/run/'):
                return self.reply(200, self.server.jobs.detail(path.removeprefix('/api/run/')))
            parts = path.split('/')
            if len(parts) == 4 and parts[1] == 'files':
                file = self.server.jobs.file(parts[2], parts[3])
                return self.reply(200, file.read_bytes(), (mimetypes.guess_type(file.name)[0] or 'application/octet-stream') + ('; charset=utf-8' if file.suffix in ('.html', '.json') else ''))
            return self.reply(404, {'error': 'Not found'})
        except (ValueError, OSError) as exc:
            return self.reply(404, {'error': str(exc)})

    def do_POST(self):
        if (not self.valid_host() or self.headers.get('Origin') not in (None, self.server.origin)
                or not secrets.compare_digest(self.headers.get('X-Aura-Token', ''), self.server.token)):
            return self.reply(403, {'error': 'Invalid origin or token'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2048 or self.headers.get_content_type() != 'application/json':
                raise ValueError('Expected bounded JSON body')
            request = json.loads(self.rfile.read(length))
            if self.path == '/api/start':
                return self.reply(200, self.server.jobs.start(request))
            if self.path == '/api/stop' and request == {}:
                return self.reply(200, self.server.jobs.stop())
            if self.path == '/api/check-sdk' and request == {}:
                return self.reply(200, check_sdk(self.server.jobs.host))
            return self.reply(404, {'error': 'Unknown operation'})
        except (ValueError, TypeError) as exc:
            return self.reply(400, {'error': str(exc)})
        except RuntimeError as exc:
            return self.reply(409, {'error': str(exc)})
        except OSError as exc:
            return self.reply(500, {'error': str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8788)
    parser.add_argument('--host', default='192.168.4.1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--models', type=Path, default=Path('models/vision'))
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/vision-console'))
    args = parser.parse_args()
    ipaddress.IPv4Address(args.host)
    if args.device != 'cpu' and not args.device.isdigit():
        parser.error('device must be cpu or a CUDA index')
    jobs = Jobs(args.output_root, args.models, args.host, args.device)
    server = ConsoleServer(('127.0.0.1', args.port), jobs)
    print(f'Aura perception console: {server.origin} (local only; no motion)', flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        jobs.stop()
        if jobs.active:
            jobs.active['process'].wait(timeout=5)
            jobs.poll()
        server.server_close()


if __name__ == '__main__':
    main()
