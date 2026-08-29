---
library_name: lerobot
tags:
  - robotics
  - embodied-ai
  - smolvla
  - rocm
  - amd
---

# Every Embodied SmolVLA ROCm PnP

This is the protected SmolVLA checkpoint used by the Datawhale Every Embodied AMD ROCm tutorial.

## Verified result

- Strict physical success: `57/60`
- Red instruction: `27/30`
- Blue instruction: `30/30`
- Task: MuJoCo mug-to-plate pick and place
- Training recipe: weighted blue sampling, selected checkpoint at step 500
- Hardware path: AMD ROCm teaching environment

The score uses the tutorial's strict physical predicate. Load this fine-tuned checkpoint with the matching environment, action normalization, camera order, and instruction format.

## Tutorial

See the [Every Embodied AMD ROCm topic](https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98) and [`14_smolvla_end_to_end.ipynb`](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/14_smolvla_end_to_end.ipynb).

## Contents

The `weights/` directory contains model and configuration files. Evaluation metadata is kept in `evaluation/`. Dataset preparation and weighted sampling are documented in the tutorial.

## License and attribution

Please follow the licenses of SmolVLA, LeRobot, the base checkpoint, and the Every Embodied project.
