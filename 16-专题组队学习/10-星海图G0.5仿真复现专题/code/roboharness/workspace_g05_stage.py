"""Plan, enter, finish, abandon or retrieve a measured task-stage transition.

Planning supports an active compatible skill or no active skill; preserve task-specific skill restrictions.
The model request combines the declared subtask with target_label and arm;
the stage record keeps both the original task and the actual model instruction.
Choose G0.5 or geometry, or abandon the stage before other tool calls change a prepared candidate observation.
Plan with parameters {task, target_label, point, arm, next_skill, next_target_label, goal, seeds,
maximum_steps, gripper_policy} and optional max_eef_distance_m (default 0.25 m). arm is left/right; next_skill is a supported workspace skill.
gripper_policy is "policy" for model grip commands or "hold" to retain the
last commanded effort. The other arm holds its measured position and effort.
Pickup goal is {"type":"grasp_stow"}; drawer goal is
{"type":"tracked_displacement","direction":[x,y,z],"target_m":0.1} in the
robot-base frame. Use a current image and the existing skill's safety procedure.
Optionally include both stall_window (2–8 steps) and min_progress_m to stop
and replan when observed directed displacement stays below that threshold.

After move_to_reach_point, a planned stage enters automatically if its point
is tracked, reach evidence passes, and the selected end effector is within the configured distance.
If it is farther away, use confirmed arm geometry to approach, retrack the point, then enter. Otherwise call enter with
{reach_call_id, tracking_call_id} from current evidence. Enter generates the
configured candidates on that observation. Use g05_refresh_batch for pending
candidates and g05_select_chunk to select a candidate or geometric fallback.
If the local model-call budget is exhausted, new candidates are marked
rejected without a physical command; select an available candidate or geometry.
The approach must use the planned arm; if a plan chooses the other arm,
abandon and replan the stage. A drawer-handle close after reach is accepted
only with a successful close-gripper confirmation for the planned arm.
If the torso changes after reach, entry requires a confirmed planned-arm
approach, current tracking and current end-effector proximity; the measured
torso change and revalidation are retained in the stage entry record.
After an interrupted nonphysical proposal, resume with empty parameters on the
same observation; unresolved physical actions still require inspection.
For pickup set the selection's direction and target_m to null; its target is
monitored for visibility and grasp must be confirmed separately. For drawer
selection use exactly the goal direction and target_m.

Finish parameters are {next_tracking_call_id, next_point, visual_reason,
exit_tracking_call_id, geometry_call_id, grasp_call_id, stow_call_id,
reset_call_id, stow_kind}. Normally the current next point must be tracked;
drawer-to-pickup may defer its observation as described below. visual_reason is the planner's current-view assessment of
next-stage accessibility, recorded as an assertion. For a drawer all grasp and
stow fields are null. For G0.5, the selected chunk must reach the measured
goal and exit_tracking_call_id and geometry_call_id are null. For geometry,
geometry_call_id names one confirmed straight adjust_chassis call, and
exit_tracking_call_id identifies the current original handle track; the host
corrects its position using measured base translation. Use only one base
movement between the latest geometry reference and this exit measurement. If
approach or grasp retries require additional base moves, track the same drawer
handle in the current observation and call reanchor with {tracking_call_id,
direction} before a further intended pull. direction is the newly assessed
pull direction in the current robot-base frame; the original displacement
threshold is retained. Reanchoring sends no physical command and does not
count a pull that happened before the new observation. Otherwise abandon and
plan a new stage rather than claiming displacement from a stale reference. For a
pickup, grasp_call_id references a successful close_gripper. For small-object
stow, stow_kind="small", stow_call_id references a successful holding-arm
set_arm_to_grasp_position with gripper="keep", and reset_call_id references
the later reset_body. For large-object stow, stow_kind="large", stow_call_id
is null and reset_call_id must preserve the holding-arm orientation. The
current wrist view must support the grasp; explain it in visual_reason.
For pickup, exit_tracking_call_id and geometry_call_id are null.
For drawer-to-pickup handoff, track the next object as a different point from
the drawer handle and describe the opening and access in visual_reason. If the
next object is outside the current view, pass null for both next_tracking_call_id
and next_point and describe the missing observation; the drawer exit still needs
its normal measured evidence. The stored transition marks the next observation
as deferred. Reobserve and track the target before planning or matching pickup;
deferred matches provide no stale target position or automatic action.

Abandon parameters are {reason}; this returns to geometric tools. Matches
parameters are {tracking_call_id, point, skill, target_label, arm}; returned
prior records are suggestions and do not substitute for current measurements.
The matching results include previous completed stages that recorded this
target as their next target, with available handoff posture and target position.
Matching uses the exact target label. When planning the next stage for the same
tracked object, reuse the prior stage's next_target_label verbatim as its
target_label. Use a new label if the physical target has changed; always
recheck the current tracking and geometry before acting on a suggestion.
"""
NAME = 'g05_stage'
SCHEMA = {'type': 'object', 'properties': {
    'action': {'enum': ['plan', 'enter', 'resume', 'finish', 'reanchor', 'abandon', 'matches']},
    'parameters': {'type': 'object'}},
    'required': ['action', 'parameters'], 'additionalProperties': False}

if __name__ == '__main__':
    raise RuntimeError('g05_stage requires the orchestrated host')
