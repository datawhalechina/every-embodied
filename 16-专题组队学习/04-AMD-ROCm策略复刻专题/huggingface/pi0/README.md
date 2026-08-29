---
tags:
  - robotics
  - embodied-ai
  - pi0
  - rocm
  - amd
---

# Every Embodied Pi0 Protected ROCm Checkpoint

This is the protected Pi0 fine-tuning checkpoint used in the Datawhale Every Embodied AMD ROCm tutorial.

## Verified result

- Strict physical success: `12/14`
- Unseen-position subset: `9/10`
- Hard subset: `6/8`
- Task: MuJoCo mug-to-plate pick and place
- Recipe: clean-success-only data, blue-task reweighting, protected continuation to the selected checkpoint

This fine-tuned Pi0 checkpoint uses the matching `eef_abs` action bridge, 8-D state (`6DoF + gripper + timestamp`), action-chunk execution, normalization statistics, camera order, and strict physical-success predicate.

## Tutorial

See the [Every Embodied AMD ROCm topic](https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98) and [`15_pi0_end_to_end.ipynb`](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/15_pi0_end_to_end.ipynb).

## Contents

The `weights/` directory contains model and configuration files. Evaluation metadata is kept in `evaluation/`. Dataset preparation and continuation training are documented in the tutorial.

## License and attribution

Please follow the licenses of Pi0, PaliGemma, LeRobot, the base checkpoint, and the Every Embodied project.
