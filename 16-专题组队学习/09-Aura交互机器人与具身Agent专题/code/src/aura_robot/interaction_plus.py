"""Opt-in multi-step interaction PREVIEW. No SDK, socket, ROS or actuator adapter.

Structured inputs must come from a separate verified perception/ASR adapter.
This module does not recognize people, gestures, speech or OCR by itself.
Track IDs are not face identities. Output velocities are proposals, never sent.
"""
from dataclasses import dataclass
import math

from .person_follow import PersonTarget, decide_follow


@dataclass(frozen=True)
class Target:
    track_id: str
    x_px: float
    confidence: float = 1.0


@dataclass(frozen=True)
class Input:
    t: float
    observed_at: float
    people: tuple[Target, ...] = ()
    gesture: int | None = None
    gesture_confidence: float = 1.0
    actor_id: str | None = None
    intent: str | None = None
    selected_id: str | None = None
    distance_m: float | None = None
    distance_valid: bool = False
    obstacle_clear: bool | None = None
    battery_ok: bool | None = None
    emergency_stop: bool = False
    owner: str | None = None
    card_text: str | None = None
    source: str = 'structured_fixture_NOT_vision_or_ASR'


class InteractionPreview:
    TTL = .5
    SESSION_LIMIT = 30.
    CHALLENGE = (1, 7, 2)  # documented thumb-up, awesome, victory IDs

    def __init__(self):
        self.active = False
        self.target_id = None
        self.state = 'idle'
        self.started = None
        self.last_t = None
        self.last_reason = None
        self.game_step = 0
        self.game_started = None
        self.last_gesture = None
        self.last_gesture_t = -math.inf
        self.previous_palm = False

    def _out(self, e, state, reason, *, v=0., w=0., message='', data=None):
        self.state = state
        speech = message if reason != self.last_reason else None
        self.last_reason = reason
        return {'t': e.t if isinstance(e.t,(int,float)) and math.isfinite(e.t) else None,
                'state': state, 'reason': reason, 'target_id': self.target_id,
                'linear_preview_mps': v, 'angular_preview_rps': w,
                'speech_proposal': speech, 'display_proposal': state,
                'data_only': data, 'challenge_step': self.game_step,
                'motion_sent': False, 'hardware_access': False,
                'observation_source': e.source, 'mode': 'preview_only_no_actuator_adapter'}

    def _pause(self,e,reason,message='先停一下，请重新确认后继续。'):
        self.active=False
        return self._out(e,'paused',reason,message=message)

    def step(self,e: Input):
        # Stops take priority even when the observation clock is invalid.
        if e.emergency_stop:
            return self._pause(e,'emergency_stop','已请求急停；此处仅为无执行器预览。')
        if e.intent in {'stop','cancel'} or (e.gesture==11 and e.gesture_confidence>=.7):
            self.previous_palm=e.gesture==5
            if e.intent=='cancel': self.target_id=None
            self.game_started=None;self.game_step=0
            return self._pause(e,'explicit_stop','收到停止请求。')
        if (not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (e.t,e.observed_at))
                or (self.last_t is not None and e.t<self.last_t)
                or not 0<=e.t-e.observed_at<=self.TTL):
            return self._pause(e,'invalid_or_stale_clock')
        self.last_t=e.t
        if (len(e.people)>16 or len({p.track_id for p in e.people})!=len(e.people)
                or any(not isinstance(p.track_id,str) or not 1<=len(p.track_id)<=64
                       or not math.isfinite(p.x_px) or not 0<=p.x_px<=960
                       or not math.isfinite(p.confidence) or not 0<=p.confidence<=1 for p in e.people)
                or not math.isfinite(e.gesture_confidence) or not 0<=e.gesture_confidence<=1):
            return self._pause(e,'invalid_observation')
        palm=e.gesture==5 and e.gesture_confidence>=.7
        new_palm=palm and not self.previous_palm
        self.previous_palm=palm
        if e.owner not in (None,'interaction_plus'):
            return self._pause(e,'other_controller_owns_motion')
        if self.active and (e.obstacle_clear is not True or e.battery_ok is not True):
            return self._pause(e,'health_or_obstacle_unknown_or_blocked')
        if self.active and e.t-self.started>self.SESSION_LIMIT:
            return self._pause(e,'session_timeout')

        if e.intent=='gesture_game':
            self.active=False;self.game_step=0;self.game_started=e.t
            self.last_gesture=None;self.last_gesture_t=-math.inf
            return self._out(e,'challenge','challenge_started',message='我们试试依次做点赞、六六六、胜利手势。')
        if self.game_started is not None:
            if e.t-self.game_started>15:
                self.game_started=None;self.game_step=0
                return self._out(e,'idle','challenge_timeout',message='挑战超时，可以重新开始。')
            g=e.gesture if e.gesture_confidence>=.7 else None
            if g==self.last_gesture and e.t-self.last_gesture_t<2:
                return self._out(e,'challenge','gesture_debounced')
            if g is None:
                return self._out(e,'challenge','await_gesture')
            self.last_gesture=g;self.last_gesture_t=e.t
            if g!=self.CHALLENGE[self.game_step]:
                return self._out(e,'challenge','unexpected_gesture')
            self.game_step+=1
            self.game_started=e.t  # documented-style per-step timeout, not total game time
            if self.game_step==len(self.CHALLENGE):
                self.game_started=None
                return self._out(e,'challenge_done','challenge_complete',message='挑战完成！',data={'unlocked_preview':True})
            return self._out(e,'challenge','challenge_advanced',message='很好，继续下一个手势。')

        if e.intent=='read_card':
            self.active=False
            if not isinstance(e.card_text,str) or not e.card_text.strip():
                return self._out(e,'reading_card','card_missing',message='请展示卡片，等待文字识别结果。')
            # No text-to-command path. UI exporters must escape this data.
            text=''.join(c for c in e.card_text if c.isprintable())[:256]
            return self._out(e,'reading_card','card_data_only',message='已接收卡片文字，仅作为内容展示。',
                             data={'ocr_text':text,'execute':False})

        request=e.intent in {'come','select_target'} or new_palm
        people={p.track_id:p for p in e.people if p.confidence>=.5}
        if request:
            if e.obstacle_clear is not True or e.battery_ok is not True:
                return self._pause(e,'health_or_obstacle_unknown_or_blocked')
            requested=e.selected_id or e.actor_id
            if self.target_id and requested and requested!=self.target_id:
                return self._pause(e,'target_change_requires_cancel','已有目标锁定，请先取消再更换。')
            if self.target_id is None:
                if requested in people:
                    self.target_id=requested
                elif requested is not None:
                    return self._pause(e,'selected_target_missing')
                elif len(people)==1:
                    self.target_id=next(iter(people))
                else:
                    return self._out(e,'need_selection','ambiguous_or_missing_target',message='请明确选择要跟随哪一位。')
            if self.target_id not in people:
                return self._pause(e,'locked_target_missing')
            if not self.active: self.started=e.t
            self.active=True

        if not self.active:
            return self._out(e,self.state,'await_explicit_request',message='等待明确的新请求，不自动恢复。')
        target=people.get(self.target_id)
        if target is None:
            return self._pause(e,'locked_target_missing')
        decision=decide_follow(PersonTarget(target.x_px,confidence=target.confidence))
        if decision.angular_z:
            return self._out(e,'turning',decision.reason,w=decision.angular_z,message='先转向目标方向。')
        if not e.distance_valid or e.distance_m is None:
            return self._out(e,'need_distance','validated_distance_required',message='距离还不可靠，暂不前进。')
        if not math.isfinite(e.distance_m) or not .1<=e.distance_m<=3:
            return self._pause(e,'invalid_distance')
        if e.distance_m<=1.0:
            self.active=False
            return self._out(e,'greeting','arrival_preview',message='你好，有什么可以帮忙？')
        return self._out(e,'approaching','bounded_approach_preview',v=.02,message='收到，我会慢慢靠近。')
