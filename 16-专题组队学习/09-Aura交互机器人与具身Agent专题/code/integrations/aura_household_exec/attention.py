"""Audio-intent / visual-feedback attention controller, without simulator truth.

This is the controller layer, not an ASR or person detector. Production adapters
must supply actual ASR receipts and detections from Aura's own camera. Tests use
structured fixtures and do not establish perception or scene integration.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Person:
    track_id: str
    center_x: float  # normalized image coordinates, 0..1
    confidence: float
    waving: bool = False


@dataclass(frozen=True)
class Camera:
    image_id: str
    stamp: float
    people: tuple[Person, ...]
    available: bool = True


@dataclass(frozen=True)
class Speech:
    utterance_id: str
    stamp: float
    text: str
    audio_sha256: str
    confidence: float
    source: str = 'asr'


@dataclass(frozen=True)
class AttentionCommand:
    state: str
    angular_z: float = 0.
    linear_x: float = 0.
    say: str | None = None
    task: str | None = None
    reason: str = ''


class Attention:
    """Bounded in-place search and centering; LLM not called per camera frame."""
    def __init__(self, *, max_angular=.22, frame_age=.5, search_timeout=40., stable_frames=3):
        if not all(math.isfinite(v) and v > 0 for v in (max_angular, frame_age, search_timeout)):
            raise ValueError('Positive finite controller limits required')
        if type(stable_frames) is not int or stable_frames < 2:
            raise ValueError('At least two distinct confirmation frames required')
        self.max_angular, self.frame_age, self.search_timeout = max_angular, frame_age, search_timeout
        self.stable_frames = stable_frames
        self.state = 'idle'
        self.started = None
        self.target = None
        self.target_last_seen = None
        self.last_image = None
        self.last_stamp = None
        self.last_now = None
        self.stable = 0
        self.greeted = False
        self.pending_task = None
        self.seen_speech = set()

    def update(self, now, camera, speech=None):
        if not math.isfinite(now) or self.last_now is not None and now < self.last_now:
            raise ValueError('Monotonic finite controller clock required')
        self.last_now = now
        if speech and speech.utterance_id not in self.seen_speech:
            valid = (speech.source == 'asr' and math.isfinite(speech.stamp)
                     and 0 <= now - speech.stamp <= 3 and math.isfinite(speech.confidence)
                     and .6 <= speech.confidence <= 1 and len(speech.audio_sha256) == 64
                     and all(c in '0123456789abcdef' for c in speech.audio_sha256))
            if valid:
                self.seen_speech.add(speech.utterance_id)
                text = ''.join(speech.text.lower().split())
                if any(word in text for word in ['取消', '停下', '停止']):
                    self.state = 'idle'; self.pending_task = None; self.target = None; self.stable = 0
                    return AttentionCommand(self.state, reason='spoken_cancel')
                task = 'pick_up_trash' if '垃圾' in text and any(w in text for w in ['收拾', '清理', '捡', '撿']) else None
                wake = task or any(w in text for w in ['看向我', '看看我', '看着我', '看我', '转过来', '轉過來', '你好aura', 'aura你好'])
                if wake and self.state in {'idle', 'timed_out', 'delegated'}:
                    self.state = 'searching'; self.started = now; self.target = None
                    self.stable = 0; self.greeted = False
                if task and self.state not in {'idle', 'timed_out'}:
                    self.pending_task = task
        if self.state in {'idle', 'timed_out', 'delegated'}:
            return AttentionCommand(self.state)
        if now - self.started >= self.search_timeout:
            self.state = 'timed_out'; self.pending_task = None
            return AttentionCommand(self.state, say='我还没找到你，请到相机前挥挥手，再叫我一次。', reason='search_budget_exhausted')
        fresh = (camera is not None and camera.available and camera.image_id
                 and math.isfinite(camera.stamp) and 0 <= now - camera.stamp <= self.frame_age)
        if not fresh:
            self.stable = 0
            return AttentionCommand(self.state, reason='camera_unavailable_or_stale')
        if camera.image_id == self.last_image:
            return AttentionCommand(self.state, reason='camera_not_advanced')
        if self.last_stamp is not None and camera.stamp <= self.last_stamp:
            self.stable = 0
            return AttentionCommand(self.state, reason='camera_not_advanced')
        self.last_image, self.last_stamp = camera.image_id, camera.stamp
        people = [p for p in camera.people if p.track_id and math.isfinite(p.center_x)
                  and 0 <= p.center_x <= 1 and math.isfinite(p.confidence) and .6 <= p.confidence <= 1]
        candidates = [p for p in people if p.track_id == self.target] if self.target else people
        if not candidates and self.target and self.target_last_seen is not None and now-self.target_last_seen>1:
            waving=[p for p in people if p.waving]
            if len(waving)==1:candidates=waving;self.stable=0
        if not self.target and len(candidates) > 1:
            candidates = [p for p in candidates if p.waving]
            if len(candidates) != 1:
                self.stable = 0
                return AttentionCommand('selecting_person', reason='multiple_people_need_unique_wave')
        if not candidates:
            self.state = 'searching'; self.stable = 0
            return AttentionCommand(self.state, angular_z=self.max_angular, reason='scan_until_visual_detection')
        person = candidates[0]
        self.target = person.track_id
        self.target_last_seen = now
        error = person.center_x - .5
        if abs(error) > .055:
            self.state = 'aligning'; self.stable = 0
            angular = max(-self.max_angular, min(self.max_angular, -1.2 * error))
            return AttentionCommand(self.state, angular_z=angular, reason='visual_centering')
        self.stable += 1
        if self.stable < self.stable_frames:
            self.state = 'confirming'
            return AttentionCommand(self.state, reason='distinct_frames_required')
        self.state = 'listening'
        # Conversation can continue indefinitely after confirmed alignment.
        self.started = now
        say = None
        if not self.greeted:
            say = '我看到你了，需要我帮你做什么？'; self.greeted = True
        if self.pending_task:
            task = self.pending_task; self.pending_task = None; self.state = 'delegated'
            return AttentionCommand(self.state, say='收到，我请 R1 帮你收拾垃圾。', task=task, reason='visually_aligned_task_handoff')
        return AttentionCommand(self.state, say=say, reason='visually_aligned')
