"""Saturate normalized gripper commands before strict physical validation.

OmniGibson BaseController._preprocess_command clips command input limits
before scaling. Raw model action arrays remain unchanged in their artifact.
Arm targets and physical effort limits are never clipped or enlarged here.
"""
import math


def gripper_domain(parts):
    result={key:list(value) for key,value in parts.items()};changes={}
    for key in ('left_gripper','right_gripper'):
        values=result[key]
        if len(values)!=1 or not math.isfinite(values[0]):
            raise ValueError('Gripper output requires one finite scalar')
        raw=float(values[0]);value=max(-1.0,min(1.0,raw));result[key]=[value]
        if raw!=value:changes[key]={'raw':raw,'normalized_command':value}
    return result,changes
