"""Create an audition candidate, never an automatically approved melody.

Keep source pitches/onsets. Unlike the old mandatory-per-onset path, allow a
sustained upper voice to continue over softer accompaniment attacks. All rejected
events remain in the review ledger. This heuristic requires musical review.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

from piano_context import write_json
from simplify_performance import groups
from transcription_review import validate_events


def candidate(notes):
    validate_events(notes, [])
    selected, decisions = [], []
    for batch in groups(notes, window=.06):
        # These bounds are source-specific audition settings, not a definition
        # of melody. The ledger exposes excluded high/quiet/brief events.
        viable = [(i, n) for i, n in batch
                  if 60 <= n['midi_note'] <= 88 and n['velocity'] >= 35
                  and n['offset_time'] - n['onset_time'] >= .08]
        if not viable:
            decisions.extend(dict(source_index=i, reason='outside_audition_bounds') for i, _ in batch)
            continue
        loudest = max(n['velocity'] for _, n in viable)
        salient = [(i, n) for i, n in viable if n['velocity'] >= .7*loudest]
        index, note = max(salient, key=lambda pair: (pair[1]['midi_note'], pair[1]['velocity']))
        reason = 'upper_salient_candidate'
        if selected:
            previous = notes[selected[-1]]
            elapsed = note['onset_time'] - previous['onset_time']
            still_sounding = note['onset_time'] < previous['offset_time']
            if (still_sounding and elapsed < .7 and note['midi_note'] < previous['midi_note']
                    and note['velocity'] < .78*previous['velocity']):
                reason = 'possible_accompaniment_under_sustained_voice'
        accepted = reason == 'upper_salient_candidate'
        if accepted:
            selected.append(index)
        decisions.extend(dict(source_index=i, reason=reason if i == index else 'other_voice_candidate',
                              selected=accepted and i == index) for i, _ in batch)
    if not selected:
        raise ValueError('No candidate melody; do not substitute a different song')
    return selected, decisions


def audition_notes(notes, indices):
    """Propose monophonic articulation without deleting/retuning chosen notes."""
    result = []
    ordered = sorted(indices, key=lambda i: notes[i]['onset_time'])
    for position, index in enumerate(ordered):
        n = notes[index]
        end = n['offset_time']
        if position+1 < len(ordered):
            next_start = notes[ordered[position+1]]['onset_time']
            if next_start <= n['onset_time']:
                raise ValueError('Simultaneous melody voices need explicit musical review')
            end = min(end, next_start - min(.055, (next_start-n['onset_time'])*.2))
        result.append(dict(source_index=index, midi=n['midi_note'], velocity=n['velocity'],
                           start=n['onset_time'], end=end,
                           source_acoustic_end=n['offset_time']))
    return result


def render(events, path):
    from robopianist.music import midi_message, synthesizer
    messages = []
    for n in events:
        messages.extend([midi_message.NoteOn(note=n['midi'], velocity=n['velocity'], time=n['start']),
                         midi_message.NoteOff(note=n['midi'], time=n['end'])])
    messages.sort(key=lambda m: m.time)
    synth = synthesizer.Synthesizer()
    try:
        samples = synth.get_samples(messages)
    finally:
        synth.stop()
    with wave.open(str(path.with_suffix('.wav')), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(samples.tobytes())
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-loglevel', 'error', '-i', str(path.with_suffix('.wav')),
                    '-codec:a', 'libmp3lame', '-q:a', '3', str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audio', action='store_true')
    args = parser.parse_args()
    raw = args.notes.read_bytes()
    notes = json.loads(raw)
    indices, decisions = candidate(notes)
    events = audition_notes(notes, indices)
    args.out.mkdir(parents=True, exist_ok=False)
    selection = dict(source_sha256=hashlib.sha256(raw).hexdigest(), melody_source_indices=indices,
                     source_melody_verified=False, method='upper_voice_audition_with_sustain_skip')
    write_json(args.out/'selection.json', selection)
    write_json(args.out/'audition-notes.json', events)
    report = dict(source_notes=len(notes), selected_notes=len(events), pitch_changes=0,
                  onset_changes=0, global_tempo_changes=0, source_melody_verified=False,
                  robot_performance=False, source_pedal_reproduced=False,
                  articulation='Explicit monophonic audition proposal, not recovered physical releases',
                  decisions=decisions)
    write_json(args.out/'review.json', report)
    if args.audio:
        render(events, args.out/'candidate-melody.mp3')
    print(json.dumps({k:v for k,v in report.items() if k != 'decisions'}, indent=2))


if __name__ == '__main__':
    main()
