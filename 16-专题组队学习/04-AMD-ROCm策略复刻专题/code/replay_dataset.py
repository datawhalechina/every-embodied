#!/usr/bin/env python
"""数据集回放工具：从 omy_pnp_language 数据集读取示教帧，合成 mp4 视频

用法:
  /opt/venv/bin/python replay_dataset.py -e 0                  # 回放 episode 0（蓝杯，agent 视角）
  /opt/venv/bin/python replay_dataset.py -e 3 -v wrist         # 回放 episode 3（红杯，wrist 视角）
  /opt/venv/bin/python replay_dataset.py -e 10 --fps 20        # 指定 fps
  /opt/venv/bin/python replay_dataset.py --list                # 列出所有 episode（指令/帧数）

参数:
  -e, --episode     episode 编号 (0-19)，默认 0
  -v, --view        视角: agent / wrist，默认 agent
  -o, --output      输出 mp4 路径，默认 /tmp/replay_ep{X}_{view}.mp4
      --fps         视频帧率，默认 15
      --max-frames  最多输出帧数（超出自动抽帧），默认 200
      --list        只列出 episode 清单
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

TOPIC = Path(__file__).resolve().parents[1]  # code/ 的上一级 = 专题根
DATA = TOPIC / "data" / "omy_pnp_language"
sys.path.insert(0, str(TOPIC / "external" / "mujoco_pnp"))

def list_episodes():
    eps = [json.loads(l) for l in (DATA / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    print(f"{'ep':>3} {'task':<40} {'frames':>6} {'时长s':>6}")
    print("-" * 60)
    for e in eps:
        print(f"{e['episode_index']:>3} {e['tasks'][0]:<40} {e['length']:>6} {e['length']/20:>6.1f}")

def replay(ep_idx, view, out_path, fps, max_frames):
    import imageio.v2 as imageio
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # 主相机在 LeRobot v2.1 中命名为 observation.image（非 agent_image）
    if view == "agent":
        feature = "observation.image"
    else:
        feature = f"observation.{view}_image"
    dataset = LeRobotDataset("datawhale_eai_pnp_language", root=DATA)
    ep_start = dataset.episode_data_index["from"][ep_idx].item()
    ep_end = dataset.episode_data_index["to"][ep_idx].item()
    n = ep_end - ep_start
    step = max(1, n // max_frames)
    frames = []
    for i in range(ep_start, ep_end, step):
        img = dataset[i][feature]          # torch (3,256,256) CHW
        if img.dim() == 3 and img.shape[0] == 3:
            img = img.permute(1, 2, 0)     # -> HWC
        frames.append((img.numpy() * 255).astype(np.uint8))
    imageio.mimsave(out_path, frames, fps=fps, quality=8)
    print(f"✅ episode {ep_idx} ({view}): {len(frames)} 帧 -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--episode", type=int, default=0)
    ap.add_argument("-v", "--view", default="agent", choices=["agent", "wrist"])
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_episodes()
    else:
        out = args.output or f"/tmp/replay_ep{args.episode}_{args.view}.mp4"
        replay(args.episode, args.view, out, args.fps, args.max_frames)
