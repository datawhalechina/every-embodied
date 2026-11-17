"""Debounced perception events and offline session replay. Never dispatches actions."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import math
from pathlib import Path


def validate_zone(zone):
    if zone is not None and (len(zone) != 4 or not all(math.isfinite(v) for v in zone)
                            or not 0 <= zone[0] < zone[2] <= 1
                            or not 0 <= zone[1] < zone[3] <= 1):
        raise ValueError('zone requires normalized x1,y1,x2,y2 in 0..1 with positive area')
    return zone


class EventMonitor:
    """Scene-level observations, not identity tracking or physical entry/exit counts."""
    def __init__(self, confirm_frames=2, max_gap_s=3., zone=None):
        if not isinstance(confirm_frames, int) or confirm_frames < 2:
            raise ValueError('confirm_frames must be an integer >= 2')
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('max_gap_s must be finite and positive')
        self.confirm_frames, self.max_gap_s = confirm_frames, max_gap_s
        self.zone = validate_zone(zone)
        self.channels = {}
        self.stamp = None

    def update(self, row, stamp):
        if not math.isfinite(stamp):
            raise ValueError('Non-finite observation timestamp')
        events = []

        def emit(kind, **detail):
            events.append({'frame': row['frame'], 'time_s': stamp, 'kind': kind, **detail})

        if self.stamp is not None and (stamp <= self.stamp or stamp-self.stamp > self.max_gap_s):
            self.channels.clear()
            emit('continuity_reset', reason='nonmonotonic_or_gap', previous_time_s=self.stamp)
        self.stamp = stamp
        observations = {}
        tasks = row.get('tasks', {}) if 'capture_error' not in row else {}
        for name, entry in tasks.items():
            if entry.get('status') != 'pass':
                continue
            data = entry.get('result', {})
            if name == 'detect':
                people = [b for b in data.get('boxes', [])
                          if b.get('label') == 'person' and b.get('confidence', 0) >= .5]
                observations['people'] = {'present'} if people else set()
                if self.zone is not None:
                    width, height = row['image_size']
                    if width <= 0 or height <= 0:
                        raise ValueError('Invalid image_size')
                    x1, y1, x2, y2 = self.zone
                    inside = any(x1 <= (b['xyxy'][0]+b['xyxy'][2])/2/width <= x2
                                 and y1 <= (b['xyxy'][1]+b['xyxy'][3])/2/height <= y2 for b in people)
                    observations['zone'] = {'occupied'} if inside else set()
            elif name == 'pose':
                observations['raised_hand'] = {'candidate'} if any(data.get('raised_hand_candidates', [])) else set()
            elif name == 'markers':
                observations['markers'] = {str(int(m['id'])) for m in data.get('markers', [])}
            elif name == 'quality':
                observations['quality'] = set(data.get('warnings', []))

        # Missing/failed input is UNKNOWN, never a confirmed disappearance.
        for channel in sorted(set(self.channels) - set(observations)):
            del self.channels[channel]
            emit('observation_unavailable', channel=channel)
        for channel, values in observations.items():
            states = self.channels.setdefault(channel, {})
            for key in sorted(set(states) | values):
                state = states.setdefault(key, {'active': False, 'candidate': None, 'count': 0})
                seen = key in values
                if seen == state['active']:
                    state['candidate'], state['count'] = None, 0
                else:
                    state['count'] = state['count']+1 if state['candidate'] == seen else 1
                    state['candidate'] = seen
                    if state['count'] >= self.confirm_frames:
                        state.update(active=seen, candidate=None, count=0)
                        emit('observation_on' if seen else 'observation_off', channel=channel, value=key)
                if not state['active'] and state['candidate'] is None:
                    del states[key]
        return events


def timing(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
    if not values:
        return {'samples': 0}
    def percentile(q):
        index = (len(values)-1)*q
        lo = int(index)
        return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(index-lo)
    return {'samples': len(values), 'p50_ms': percentile(.5), 'p95_ms': percentile(.95), 'max_ms': values[-1]}


def analyze(rows, *, confirm_frames=2, max_gap_s=3., zone=None):
    if not isinstance(rows, list) or not rows or len(rows) > 10000:
        raise ValueError('Expected 1..10000 vision result rows')
    monitor = EventMonitor(confirm_frames, max_gap_s, zone)
    events, stats = [], {}
    source = 'sample_time_s' if all('sample_time_s' in row for row in rows) else 'created_at'
    base = None
    for row in rows:
        if source == 'sample_time_s':
            stamp = float(row['sample_time_s'])
        else:
            date = datetime.fromisoformat(row['created_at'])
            if date.tzinfo is None:
                raise ValueError('Replay requires timezone-aware created_at')
            stamp = date.timestamp()
        if not math.isfinite(stamp):
            raise ValueError('Invalid timestamp')
        if base is None:
            base = stamp
        events.extend(monitor.update(row, stamp-base))
        for task, result in row.get('tasks', {}).items():
            state = stats.setdefault(task, {'pass': 0, 'failed': 0, 'times': []})
            state['pass' if result.get('status') == 'pass' else 'failed'] += 1
            if result.get('status') == 'pass':
                state['times'].append(result.get('elapsed_ms'))
    for state in stats.values():
        state['timing'] = timing(state.pop('times'))
    return {'frames': len(rows), 'capture_failures': sum('capture_error' in row for row in rows),
            'evidence_sources': sorted({row.get('source_kind', 'unknown_legacy') for row in rows}),
            'capture_timing': timing([r.get('capture_ms') for r in rows]), 'tasks': stats,
            'events': events, 'motion_sent': False, 'timestamp_source': source,
            'settings': {'confirm_frames': confirm_frames, 'max_gap_s': max_gap_s, 'zone': zone},
            'note': 'Offline observation timeline only. Times are host sampling times, not exposure sync. '
                    'On/off means detected/not detected, not physical arrival/departure. '
                    'Raised hand is not a wave. Zone uses box center, not ground coordinates. '
                    'No identities, OCR text, QR contents, or commands are dispatched.'}


def save_analysis(output, summary):
    with (output/'events.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    detail = html.escape(json.dumps({k: v for k, v in summary.items() if k != 'events'}, ensure_ascii=False, indent=2))
    rows = ''.join('<tr>' + ''.join('<td>' + html.escape(str(event.get(k, ''))) + '</td>'
                   for k in ('frame', 'time_s', 'kind', 'channel', 'value', 'reason')) + '</tr>'
                   for event in summary['events'])
    with (output/'events.html').open('x', encoding='utf-8') as stream:
        stream.write('<!doctype html><meta charset="utf-8"><title>Aura 事件与性能</title>'
            '<style>body{font:16px system-ui;margin:32px;background:#111923;color:#e6eff9}'
            'td,th{padding:10px;border-bottom:1px solid #526175;text-align:left}pre{white-space:pre-wrap}</style>'
            '<h1>Aura 事件时间线与性能</h1><p>仅感知记录，不驱动机器人；空事件不代表硬件或识别效果通过。</p>'
            '<table><tr><th>帧</th><th>相对秒</th><th>事件</th><th>通道</th><th>值</th><th>原因</th></tr>'
            + rows + '</table><h2>任务状态及耗时（含首帧预热）</h2><pre>' + detail + '</pre>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Existing vision results.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--confirm-frames', type=int, default=2)
    parser.add_argument('--max-gap-s', type=float, default=3.)
    parser.add_argument('--zone', type=float, nargs=4)
    args = parser.parse_args()
    if args.input.stat().st_size > 32*1024*1024:
        raise ValueError('Replay JSON exceeds 32 MiB')
    rows = json.loads(args.input.read_text(encoding='utf-8'))
    summary = analyze(rows, confirm_frames=args.confirm_frames, max_gap_s=args.max_gap_s, zone=args.zone)
    args.output.mkdir(parents=True, exist_ok=False)
    save_analysis(args.output, summary)
    print(json.dumps({'frames': summary['frames'], 'events': len(summary['events']),
                      'capture_failures': summary['capture_failures'], 'motion_sent': False}))


if __name__ == '__main__':
    main()
