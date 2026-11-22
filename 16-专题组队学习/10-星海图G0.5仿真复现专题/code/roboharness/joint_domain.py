"""Apply static robot position bounds to small model target overshoots.

The original 0.10-radian delta test runs on raw targets before normalization.
Only the official URDF position endpoints are used; metadata is recorded in
the policy fingerprint. Native strict action validation remains unchanged.
"""
import os
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np

EXPECTED_URDF_SHA='f617d7c9d4ab8744a5b3664afe7364aacb742d39d5a3f552dcab36dc058a72e1'


def arm_position_domain(parts,proprio):
    urdf = Path(os.environ['G05_R1PRO_URDF'])
    if hashlib.sha256(urdf.read_bytes()).hexdigest()!=EXPECTED_URDF_SHA:
        raise ValueError('Static robot geometry differs from the registered profile')
    nodes={node.get('name'):node for node in ET.parse(urdf).getroot().findall('joint')}
    result={key:list(values) for key,values in parts.items()};changes={}
    for side,start in [('left',3),('right',30)]:
        values=np.asarray(result[side+'_arm'],dtype=float);state=np.asarray(proprio[start:start+7],dtype=float)
        if values.shape!=(7,) or not np.isfinite(values).all() or np.any(np.abs(values-state)>0.10):
            raise ValueError('Raw model target exceeds the frozen single-command joint bound')
        for i,raw in enumerate(values):
            name=f'{side}_arm_joint{i+1}';limit=nodes[name].find('limit')
            low,high=[float(np.float32(limit.get(key))) for key in ('lower','upper')]
            value=float(np.clip(raw,low,high));result[side+'_arm'][i]=value
            if value!=float(raw):changes[name]={'raw':float(raw),'position_command':value,'limits':[low,high]}
    return result,changes
