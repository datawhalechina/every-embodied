"""Build a short source-indexed audition/physical plan with explicit chord edits.

The recipe is private musical input, not generated from a song title. Its melody
selection still needs musical review; preserving selected notes cannot prove it.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from piano_context import write_json
from polyphonic_score import validate_score
from song_score import assign_fingers
from transcription_review import validate_events


def prepare(notes, recipe):
    validate_events(notes, [])
    start, stretch = recipe['source_start'], recipe['time_stretch']
    if (not math.isfinite(start) or start < 0 or not math.isfinite(stretch)
            or not 1 <= stretch <= 2):
        raise ValueError('Invalid source start or uniform stretch')
    octave = recipe['melody_octaves']
    if type(octave) is not int or octave not in (-1, 0, 1):
        raise ValueError('Melody only permits a uniform octave shift')
    indices = recipe['melody_source_indices']
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Empty or repeated melody indices')

    def source(index):
        if type(index) is not int or not 0 <= index < len(notes):
            raise ValueError('Invalid source index')
        return notes[index]

    indices = sorted(indices, key=lambda i: source(i)['onset_time'])
    output, changes = [], []
    for position, index in enumerate(indices):
        n = source(index)
        onset = (n['onset_time']-start)*stretch
        hold = (n['offset_time']-n['onset_time'])*stretch
        if position+1 < len(indices):
            hold = min(hold, (source(indices[position+1])['onset_time']-n['onset_time'])*stretch-.08)
        else:
            hold = min(hold, recipe['last_melody_hold_seconds'])
        pitch = n['midi_note']+12*octave
        output.append(dict(midi=pitch, start=onset, end=onset+hold, hand='right'))
        changes.append(dict(source_index=index, role='melody', original_midi=n['midi_note'],
                            midi=pitch, original_onset=n['onset_time'], onset=onset,
                            hold_seconds=hold, source_acoustic_end=n['offset_time']))
    for group in recipe['accompaniment_chords']:
        onset = (group['source_onset']-start)*stretch
        hold = group['hold_seconds']
        for member in group['members']:
            n = source(member['source_index'])
            shift = member['octaves']
            if type(shift) is not int or not -3 <= shift <= 3:
                raise ValueError('Accompaniment octave change must be explicit')
            pitch = n['midi_note']+12*shift
            output.append(dict(midi=pitch, start=onset, end=onset+hold, hand='left'))
            changes.append(dict(source_index=member['source_index'], role='accompaniment',
                original_midi=n['midi_note'], midi=pitch, original_onset=n['onset_time'],
                onset=onset, hold_seconds=hold, source_acoustic_end=n['offset_time']))
    for n in output:
        if (not 21 <= n['midi'] <= 108 or not math.isfinite(n['start']) or n['start'] < -1e-8
                or not math.isfinite(n['end']) or n['end']-n['start'] < .05):
            raise ValueError('Unplayable note timing/pitch; revise explicitly, never drop melody')
        n['start'] = max(0., n['start'])
    assigned = assign_fingers(sorted(output, key=lambda n: (n['start'], n['midi'])))
    overrides = recipe.get('melody_finger_overrides', [])
    for override in overrides:
        index = override['source_index']
        if index not in indices:
            raise ValueError('Finger override must refer to a selected melody event')
        n = source(index)
        matches = [a for a in assigned if a['hand'] == 'right'
                   and a['midi'] == n['midi_note']+12*octave
                   and abs(a['start']-(n['onset_time']-start)*stretch) < 1e-8]
        if len(matches) != 1:
            raise ValueError('Ambiguous melody finger override')
        matches[0]['finger'] = override['finger']
    score = dict(title='Source-indexed opening excerpt: explicit octave and chord arrangement',
        tempo_bpm=60, notes=[dict(midi=n['midi'], start_beat=n['start'],
            duration_beats=n['end']-n['start'], hand=n['hand'], finger=n['finger']) for n in assigned])
    validate_score(score, song_mode=True)
    used = {n['source_index'] for n in changes}
    report = dict(source_melody_verified=False, robot_performance_verified=False,
        source_start=start, time_stretch=stretch, melody_uniform_octaves=octave,
        melody_count=len(indices), chord_count=len(recipe['accompaniment_chords']),
        melody_finger_overrides=overrides,
        melody_indices_preserved=indices, changes=changes,
        omitted_source_indices=[i for i in range(len(notes)) if i not in used],
        note='Monophonic release proposal and explicitly regrouped accompaniment; no original pedal')
    return score, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audio', action='store_true')
    args = parser.parse_args()
    raw, recipe_bytes = args.notes.read_bytes(), args.recipe.read_bytes()
    recipe = json.loads(recipe_bytes)
    if hashlib.sha256(raw).hexdigest() != recipe['source_sha256']:
        raise ValueError('Recipe belongs to a different transcription')
    score, report = prepare(json.loads(raw), recipe)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'score.plan.json', score)
    report.update(source_sha256=hashlib.sha256(raw).hexdigest(),
                  recipe_sha256=hashlib.sha256(recipe_bytes).hexdigest(),
                  plan_sha256=hashlib.sha256((args.out/'score.plan.json').read_bytes()).hexdigest())
    write_json(args.out/'arrangement-report.json', report)
    if args.audio:
        from melody_review import render
        render([dict(midi=n['midi'], velocity=80, start=n['start_beat'],
                     end=n['start_beat']+n['duration_beats']) for n in score['notes']],
               args.out/'score-reference.mp3')
    print(json.dumps({k:v for k,v in report.items() if k not in ('changes','omitted_source_indices')}, indent=2))


if __name__ == '__main__':
    main()
