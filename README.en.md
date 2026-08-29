# Every Embodied: Lightweight AMD ROCm Tutorials

The `amd-rocm` branch is a focused checkout for the AMD ROCm embodied-AI course. It contains the tutorial Markdown, notebooks, small Python helpers, required screenshots, and a few compact teaching rollouts. Datasets, model checkpoints, caches, and batch videos remain outside the repository.

- [Online book](https://datawhalechina.github.io/every-embodied/zh-cn/)
- [Main repository](https://github.com/datawhalechina/every-embodied/tree/main)
- [AMD ROCm tutorial source](https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98)

## Shallow clone

```bash
git clone --depth 1 --single-branch --branch amd-rocm \
  https://github.com/datawhalechina/every-embodied.git every-embodied-amd-rocm
cd every-embodied-amd-rocm
```

Readers who only need the Markdown do not need a Python environment, datasets, or model weights.

For a Markdown-only checkout without images, notebooks, or rollout videos:

```bash
git clone --depth 1 --filter=blob:none --no-checkout \
  --single-branch --branch amd-rocm \
  https://github.com/datawhalechina/every-embodied.git every-embodied-amd-markdown
cd every-embodied-amd-markdown
git sparse-checkout init --no-cone
git sparse-checkout set \
  '/README.md' \
  '/README.en.md' \
  '/LICENSE' \
  '/16-专题组队学习/04-AMD-ROCm策略复刻专题/*.md' \
  '/16-专题组队学习/04-AMD-ROCm策略复刻专题/**/*.md'
git checkout amd-rocm
```

## Course entry

Start from [the AMD ROCm tutorial index](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README.md). The course covers environment checks, Every Embodied, RoboCasa365, DexJoCo, DISCOVERSE, RoboWits, Unitree G1 safety control, training, closed-loop evaluation, multi-view video, and result packaging.

## External artifact layout

```bash
export SRC_ROOT=/path/to/sources
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/models
export RUN_ROOT=/path/to/runs
```

Each chapter defines its own subdirectories, download commands, and validation checkpoints. Long videos are streamed from the public showcase, while datasets and checkpoints are downloaded from their upstream repositories only when an experiment needs them.

## Branch roles

| Branch | Purpose |
| --- | --- |
| `main` | Complete project source and online-book build |
| `amd-rocm` | Lightweight AMD tutorial checkout |
