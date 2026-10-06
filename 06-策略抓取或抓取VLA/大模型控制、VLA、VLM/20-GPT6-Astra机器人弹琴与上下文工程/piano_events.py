"""Score-independent key/pedal events from measured physical travel."""

import math
from polyphonic_score import KeyCycles


class PianoEventSensor:
    def __init__(self):
        self.keys = KeyCycles(88)
        self.pedal = False
        self.last_time = -math.inf
        self.events = []

    def update(self, time, travel, pedal):
        if not math.isfinite(time) or time < 0 or time <= self.last_time or not math.isfinite(pedal):
            raise ValueError("Expected increasing finite physical timestamps and a finite pedal value")
        changes = self.keys.update(travel)
        self.last_time = time
        down = pedal >= 0.5
        if down != self.pedal:
            self.events.append({"type": "SustainOn" if down else "SustainOff", "time": time, "note": None})
            self.pedal = down
        for key, active in changes:
            self.events.append({"type": "NoteOn" if active else "NoteOff", "time": time, "note": key + 21})


def midi_messages(events):
    from robopianist.music import midi_message

    result = []
    for event in events:
        kind = event["type"]
        if kind == "NoteOn":
            result.append(midi_message.NoteOn(note=event["note"], velocity=80, time=event["time"]))
        elif kind == "NoteOff":
            result.append(midi_message.NoteOff(note=event["note"], time=event["time"]))
        elif kind in ("SustainOn", "SustainOff"):
            result.append(getattr(midi_message, kind)(time=event["time"]))
        else:
            raise ValueError("Unknown MIDI event")
    return result
