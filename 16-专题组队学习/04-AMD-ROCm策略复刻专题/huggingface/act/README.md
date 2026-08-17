---
tags:
  - robotics
  - embodied-ai
  - act
  - rocm
  - amd
---

# Every Embodied ACT Protected Diagnostic Checkpoint

This is the current ACT protection candidate for the Datawhale Every Embodied AMD ROCm tutorial.

## Verified result

- Strict physical success: `15/30`
- Three fixed seed groups of 10: `3/10 + 4/10 + 8/10`
- Action dimension: 16-D
- Recipe: reset-aligned data, timestamp/object-init state, chunk size 20, 10 action steps, low-learning-rate continuation, and weighted recovery data
- Model SHA256: `b9b178377995a674a06bc5d1500c8e7e7fc5d02649268855f892b3987bf5bfeb4`

This is a teaching and diagnosis checkpoint, not a claim that the historical `17/30` summary has been fully reproduced. The current score came from an AMD395 wrapper continuation; the matching native training recipe is included in the tutorial Notebook and should be run in a Jupyter-capable environment before calling it a Notebook-native score.

## Tutorial

See the [Every Embodied AMD ROCm topic](https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98) and [`16_act_end_to_end.ipynb`](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/16_act_end_to_end.ipynb).

## Contents

The `weights/` directory contains the ACT model and configuration. Evaluation metadata and the recipe are kept in `evaluation/`. Raw datasets are intentionally omitted.

## License and attribution

Please follow the licenses of ACT, the base implementation, and the Every Embodied project. This model card describes a reproducible tutorial artifact, not a new policy architecture.
