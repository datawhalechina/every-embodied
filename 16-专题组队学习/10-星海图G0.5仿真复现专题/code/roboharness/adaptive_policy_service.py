"""Adaptive Policy Service for Galaxea G0.5 in RoboHarness.

Provides:
  1. AdaptiveProprioAdapter: Dynamic idle-arm & trunk OOD detection and in-distribution alignment
     on observation input, with rigid freeze on the idle arm in policy action output.
  2. PromptPurifier: Context purification stripping out cross-stage future instructions.
  3. CotBBoxParser: 1024-bin normalized target-matched 2D bounding box parser.
  4. NeuralHandoverGate: Distance-based gating between G0.5 neural approach (15-25cm)
     and geometric force-control close_gripper (< 5cm).
"""
import copy
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Dataset in-distribution means for R1Pro
TRUNK_MEAN = [0.93472075, -1.27984285, -0.5010786, 0.0]
LEFT_ARM_J4_MEAN = -0.62018555
RIGHT_ARM_J4_MEAN = -0.63724398

# Joint thresholds
ARM_STOWED_J4_THRESHOLD = -1.75  # Dataset q01 is -1.745 rad


class AdaptiveProprioAdapter:
    """Adapts raw proprioception to model training distribution and freezes idle arm."""

    @staticmethod
    def inspect_state(proprio: List[float]) -> Dict[str, Any]:
        """Inspects raw proprioception for OOD trunk and stowed arms."""
        trunk_q0 = float(proprio[57])
        left_j4 = float(proprio[6])
        right_j4 = float(proprio[33])

        is_trunk_inverted = bool(trunk_q0 < 0.0)
        idle_arm = None
        if left_j4 < ARM_STOWED_J4_THRESHOLD:
            idle_arm = "left"
        elif right_j4 < ARM_STOWED_J4_THRESHOLD:
            idle_arm = "right"

        active_arm = "right" if idle_arm == "left" else ("left" if idle_arm == "right" else None)

        return {
            "trunk_inverted": is_trunk_inverted,
            "trunk_q0": trunk_q0,
            "idle_arm": idle_arm,
            "active_arm": active_arm,
            "left_j4": left_j4,
            "right_j4": right_j4
        }

    @classmethod
    def adapt_observation(cls, data: Dict[str, Any], proprio: List[float], active_arm: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Applies in-distribution adaptation to neural input data['state']."""
        state_info = cls.inspect_state(proprio)
        if active_arm not in ('left', 'right', None):
            raise ValueError('active_arm must be left, right, or None')
        idle = {'left': 'right', 'right': 'left'}.get(active_arm)
        idle_j4 = state_info.get(idle + '_j4') if idle else None
        state_info['heuristic_idle_arm'] = state_info['idle_arm']
        state_info['active_arm'] = active_arm
        state_info['idle_arm'] = idle if idle_j4 is not None and idle_j4 < ARM_STOWED_J4_THRESHOLD else None
        history_len = data['state']['trunk_qpos'].shape[0]

        # 1. Align trunk if inverted
        if state_info["trunk_inverted"]:
            trunk_t = torch.tensor(TRUNK_MEAN, dtype=torch.float32, device=data['state']['trunk_qpos'].device)
            data['state']['trunk_qpos'] = trunk_t.unsqueeze(0).repeat(history_len, 1)

        # 2. Align stowed idle arm
        if state_info["idle_arm"] == "left":
            data['state']['left_arm'][:, 3] = LEFT_ARM_J4_MEAN
        elif state_info["idle_arm"] == "right":
            data['state']['right_arm'][:, 3] = RIGHT_ARM_J4_MEAN

        return data, state_info

    @staticmethod
    def adapt_action_output(outputs: List[Dict[str, Any]], proprio: List[float], idle_arm: Optional[str]) -> List[Dict[str, Any]]:
        """Overrides idle arm trajectory in action outputs with the measured proprio state."""
        if not outputs or idle_arm not in ("left", "right"):
            return outputs

        item = outputs[0]
        if idle_arm == "left" and "left_arm" in item:
            ref_tensor = item["right_arm"] if "right_arm" in item else item["left_arm"]
            horizon = ref_tensor.shape[1] if ref_tensor.ndim == 3 else ref_tensor.shape[0]
            left_q = torch.tensor(proprio[3:10], dtype=torch.float32, device=ref_tensor.device)
            if ref_tensor.ndim == 3:
                item["left_arm"] = left_q.unsqueeze(0).unsqueeze(0).repeat(1, horizon, 1)
            else:
                item["left_arm"] = left_q.unsqueeze(0).repeat(horizon, 1)

        elif idle_arm == "right" and "right_arm" in item:
            ref_tensor = item["left_arm"] if "left_arm" in item else item["right_arm"]
            horizon = ref_tensor.shape[1] if ref_tensor.ndim == 3 else ref_tensor.shape[0]
            right_q = torch.tensor(proprio[30:37], dtype=torch.float32, device=ref_tensor.device)
            if ref_tensor.ndim == 3:
                item["right_arm"] = right_q.unsqueeze(0).unsqueeze(0).repeat(1, horizon, 1)
            else:
                item["right_arm"] = right_q.unsqueeze(0).repeat(horizon, 1)

        return outputs


class PromptPurifier:
    """Strips future task instructions and scene distractors from prompts."""

    @staticmethod
    def purify(raw_task: str, target_name: Optional[str] = None, op_arm: Optional[str] = None) -> str:
        """Purifies a task instruction to only the current pick/reach subtask."""
        text = raw_task

        # 1. Normalize compound object prefixes to avoid distracting scene nouns
        replacements = [
            (r'\bsofa_chaise_pumpkin\b', 'pumpkin'),
            (r'\bnearer_tan_candle\b', 'tan candle'),
            (r'\bblack_caldron\b', 'black caldron'),
            (r'\bblack_cauldron\b', 'black caldron')
        ]
        for src_pat, dst_str in replacements:
            text = re.sub(src_pat, dst_str, text, flags=re.IGNORECASE)

        # 2. Strip future placement instructions and environment distractors
        patterns = [
            r'Next place.*$',
            r'place this .* inside.*$',
            r'stow it securely for placement.*$',
            r'beside the sofa.*$',
            r'clear of the coffee table.*$'
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.IGNORECASE)

        text = text.strip(' ,;')

        # 3. If subtask is present, isolate current subtask
        if 'Subtask:' in text:
            parts = text.split('Subtask:')
            main_cmd = parts[0].strip(' ,;')
            sub_cmd = parts[1].strip(' ,;')
            # Clean up trailing conjunctions
            sub_cmd = re.sub(r'[,;]\s*(?:stow|then reset|and reset).*$', '', sub_cmd, flags=re.IGNORECASE)
            return f"{main_cmd}. Subtask: {sub_cmd}."

        return f"{text}."


class CotBBoxParser:
    """Parses 2D bounding boxes from G0.5 reasoning tokens normalized to 1024 bins."""

    @staticmethod
    def parse(cot_text: str, target_name: Optional[str] = None, image_size: Tuple[int, int] = (720, 720)) -> Optional[Dict[str, Any]]:
        """Extracts target bounding box from CoT text."""
        if not cot_text:
            return None

        bbox_part = cot_text.split('|')[0] if '|' in cot_text else cot_text
        if 'BBox:' in bbox_part:
            bbox_part = bbox_part.split('BBox:')[1]

        matches = re.findall(r'([a-zA-Z0-9_\-\s]+?)\s*<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>', bbox_part)
        if not matches:
            return None

        img_h, img_w = image_size
        parsed = []
        for label, y1, x1, y2, x2 in matches:
            label = label.strip('; ,')
            y1_val, x1_val, y2_val, x2_val = int(y1), int(x1), int(y2), int(x2)
            px_x1 = x1_val * img_w / 1024.0
            px_y1 = y1_val * img_h / 1024.0
            px_x2 = x2_val * img_w / 1024.0
            px_y2 = y2_val * img_h / 1024.0
            parsed.append({
                "label": label,
                "token_str": f"<loc{y1_val:04d}><loc{x1_val:04d}><loc{y2_val:04d}><loc{x2_val:04d}>",
                "coords_1024": [y1_val, x1_val, y2_val, x2_val],
                "pixel_bbox": [round(px_x1, 1), round(px_y1, 1), round(px_x2, 1), round(px_y2, 1)],
                "pixel_center": [round((px_x1 + px_x2)/2.0, 1), round((px_y1 + px_y2)/2.0, 1)]
            })

        # Match strategy:
        # 1. Target name keyword match
        if target_name:
            target_words = set(re.findall(r'[a-zA-Z0-9]+', target_name.lower()))
            for item in parsed:
                label_words = set(re.findall(r'[a-zA-Z0-9]+', item['label'].lower()))
                if target_words & label_words:
                    return item

        # 2. Subtask match
        if '|' in cot_text:
            subtask_part = cot_text.split('|')[1].lower()
            for item in parsed:
                if item['label'].lower() in subtask_part:
                    return item

        return parsed[-1]


class NeuralHandoverGate:
    """Evaluates whether to trigger G0.5 neural approach or geometric force-control close_gripper."""

    COARSE_WINDOW_MIN_M = 0.12  # 12 cm
    COARSE_WINDOW_MAX_M = 0.30  # 30 cm
    PREGRASP_THRESHOLD_M = 0.05 # 5 cm

    @classmethod
    def evaluate_gate(cls, distance_to_target_m: float) -> Dict[str, Any]:
        """Evaluates distance to decide operational phase."""
        if distance_to_target_m > cls.COARSE_WINDOW_MAX_M:
            return {
                "phase": "far_navigation",
                "trigger_g05_neural_approach": False,
                "handover_to_geometric_close": False,
                "description": f"Distance ({distance_to_target_m*100:.1f} cm) > {cls.COARSE_WINDOW_MAX_M*100:.0f} cm: geometric base navigation active."
            }
        elif cls.COARSE_WINDOW_MIN_M <= distance_to_target_m <= cls.COARSE_WINDOW_MAX_M:
            return {
                "phase": "neural_coarse_approach",
                "trigger_g05_neural_approach": True,
                "handover_to_geometric_close": False,
                "description": f"Distance ({distance_to_target_m*100:.1f} cm) in coarse window [12, 30] cm: G0.5 neural approach active."
            }
        elif distance_to_target_m < cls.PREGRASP_THRESHOLD_M:
            return {
                "phase": "geometric_close_gripper",
                "trigger_g05_neural_approach": False,
                "handover_to_geometric_close": True,
                "description": f"Distance ({distance_to_target_m*100:.1f} cm) < {cls.PREGRASP_THRESHOLD_M*100:.0f} cm: handover to geometric force-control close_gripper."
            }
        else:
            # Between 5 cm and 12 cm: Transition zone
            return {
                "phase": "transition_approach",
                "trigger_g05_neural_approach": True,
                "handover_to_geometric_close": False,
                "description": f"Distance ({distance_to_target_m*100:.1f} cm) in transition zone: continue G0.5 neural alignment."
            }
