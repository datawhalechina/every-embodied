"""Trim only the silent lead-in of an audited session, preserving its full take.

The preview omits preparation; it is not evidence of faster G0.5 execution.
No speed, pitch, note filtering, soundtrack replacement, or internal cuts.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from piano_context import write_json


def export(session, out, lead_seconds):
    if not math.isfinite(lead_seconds) or not 0 <= lead_seconds <= 3:
        raise ValueError('Lead must be between zero and three seconds')
    report = json.loads((session / 'report.json').read_text(encoding='utf-8'))
    if not report.get('playback_completed') or report.get('safety_stop'):
        raise ValueError('A complete session without a safety stop is required')
    events = json.loads((session / 'events.json').read_text(encoding='utf-8'))
    onsets = [float(e['time_seconds']) for e in events if e['type'] == 'NoteOn']
    if not onsets or not all(math.isfinite(t) and t >= 0 for t in onsets):
        raise ValueError('No valid measured note onsets')
    first = min(onsets)
    cut = max(0., first - lead_seconds)
    source = session / 'demo.mp4'
    if not source.is_file():
        raise FileNotFoundError(source)
    out.mkdir(parents=True, exist_ok=False)
    video = out / 'preview.mp4'
    subprocess.run([
        'ffmpeg', '-nostdin', '-n', '-loglevel', 'error',
        '-ss', f'{cut:.6f}', '-i', str(source),
        '-map', '0:v:0', '-map', '0:a:0',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(video)
    ], check=True)
    metadata = dict(
        source_video=str(source.resolve()),
        source_video_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_report_sha256=hashlib.sha256((session / 'report.json').read_bytes()).hexdigest(),
        source_first_physical_note_seconds=first,
        removed_silent_lead_seconds=cut,
        first_physical_note_in_preview_seconds=first-cut,
        playback_speed=1, changed_audio_events=False,
        preparation_partly_or_fully_omitted=cut > 0,
        warning='Presentation crop only; use the uncut session to inspect G0.5 preparation.',
        source_onset_metrics=report.get('onset_metrics'),
        source_performance_passed=report.get('performance_passed'))
    write_json(out / 'preview-report.json', metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lead-seconds', type=float, default=.8)
    args = parser.parse_args()
    export(args.session, args.out, args.lead_seconds)
