"""用脚本执行 ACT、SmolVLA 或 pi0 的 MuJoCo 闭环评估。

这个入口和 notebooks/11_mujoco_closed_loop_deploy.ipynb 使用同一套
physical_success 判定。它适合批量 seed、保存视频和 JSONL；Notebook 更适合
逐格查看图像、指标和排障过程。

在专题根目录执行示例：

    export PROJECT_ROOT=/path/to/mujoco_pnp
    export DATA_ROOT=/path/to/datasets/every_embodied
    export MODEL_ROOT=/path/to/checkpoints/every_embodied
    export POLICY_TYPE=smolvla
    export MODEL_RUN_DIR="$MODEL_ROOT/smolvla_weighted_000500"
    python code/run_closed_loop.py

ROCm 下 PyTorch 仍使用 cuda 设备名，这是 LeRobot/ROCm 的兼容约定。
默认不要求打开 MuJoCo 3D 窗口；需要实时窗口时设置 RENDER=1，并提前
准备 DISPLAY 或 Xvfb。
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", TOPIC_ROOT / "external" / "mujoco_pnp")
).expanduser()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", TOPIC_ROOT / "data")).expanduser()
DATASET_ROOT = Path(
    os.environ.get("DATASET_ROOT", DATA_ROOT / "omy_pnp_language")
).expanduser()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", TOPIC_ROOT / "outputs")).expanduser()
MODEL_ROOT = Path(
    os.environ.get("MODEL_ROOT", PROJECT_ROOT / "ckpt")
).expanduser()

POLICY_TYPE = os.environ.get("POLICY_TYPE", "act").lower()
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language")
MODEL_RUN_DIR = Path(
    os.environ.get("MODEL_RUN_DIR", MODEL_ROOT / f"{POLICY_TYPE}_rocm_full")
).expanduser()
POLICY_PATH = os.environ.get("POLICY_PATH", "").strip()
DEVICE = os.environ.get("DEVICE", "cuda")
TASK_TEXT = os.environ.get("TASK_TEXT", "Place the blue mug on the plate.")
OUTPUT_DIR = OUTPUT_ROOT / f"{POLICY_TYPE}_closed_loop"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("EVAL_SEEDS 至少需要包含一个整数 seed。")
    return seeds


EVAL_SEEDS = parse_seeds(os.environ.get("EVAL_SEEDS", "1000,1001,1002,1003"))
MAX_ACTION_STEPS = env_int("MAX_ACTION_STEPS", 300)
VIDEO_STRIDE = max(1, env_int("VIDEO_STRIDE", 2))
VIDEO_FPS = max(1, env_int("VIDEO_FPS", 10))
RENDER = env_bool("RENDER", False)

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        f"找不到 PROJECT_ROOT：{PROJECT_ROOT}\n"
        "请把 PROJECT_ROOT 指向上游 mujoco_pnp 工程，或按 external/README.md 放置工程。"
    )
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch


def require_layout() -> None:
    required = [
        PROJECT_ROOT / "asset" / "example_scene_y2.xml",
        PROJECT_ROOT / "mujoco_env" / "y_env2.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    info_path = DATASET_ROOT / "meta" / "info.json"
    if not info_path.exists():
        missing.append(str(info_path))
    if missing:
        raise FileNotFoundError(
            "运行闭环评估前缺少以下文件：\n- " + "\n- ".join(missing)
        )


def find_pretrained_model(run_dir: Path) -> Path:
    """同时接受直接的 pretrained_model 目录和训练输出目录。"""
    run_dir = Path(run_dir).expanduser()
    if not run_dir.exists():
        raise FileNotFoundError(f"模型目录不存在：{run_dir}")

    direct_markers = (
        "config.json",
        "train_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    if run_dir.is_dir() and any((run_dir / marker).exists() for marker in direct_markers):
        return run_dir

    last = run_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last
    candidates = sorted((run_dir / "checkpoints").glob("*/pretrained_model"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"没有在 {run_dir} 下找到 pretrained_model。")


def image_tensor(array: np.ndarray, size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    from PIL import Image

    image = Image.fromarray(array).convert("RGB").resize(size)
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous()


def load_policy(policy_type: str, policy_path: Path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if policy_type == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy as policy_class
    elif policy_type == "smolvla":
        from lerobot.common.policies.smolvla.modeling_smolvla import (
            SmolVLAPolicy as policy_class,
        )
    elif policy_type == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as policy_class
    else:
        raise ValueError("POLICY_TYPE 只能是 act、smolvla 或 pi0。")

    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=str(DATASET_ROOT),
    )
    policy = policy_class.from_pretrained(
        str(policy_path),
        dataset_stats=metadata.stats,
    )
    policy.to(DEVICE)
    policy.eval()
    return policy


def strict_snapshot(
    env,
    initial_plate_pos: np.ndarray,
    max_target_lift: float,
    max_lifted_run: int,
    stable_place_steps: int,
) -> tuple[dict, int]:
    """按照 Notebook 11 的几何、抬升、直立、释放和稳定条件判定成功。"""
    target_pos = np.asarray(env.env.get_p_body(env.obj_target), dtype=np.float64)
    plate_pos = np.asarray(env.env.get_p_body("body_obj_plate_11"), dtype=np.float64)
    target_rotation = np.asarray(
        env.env.get_R_body(env.obj_target), dtype=np.float64
    )

    xy_dist = float(np.linalg.norm(target_pos[:2] - plate_pos[:2]))
    upright_cos = float(target_rotation[2, 2])
    gripper_open = bool(float(env.env.get_qpos_joint("rh_r1")[0]) < 0.1)
    tcp_high = bool(float(env.env.get_p_body("tcp_link")[2]) > 0.9)
    legacy_success = bool(env.check_success())
    plate_xy_displacement = float(
        np.linalg.norm(plate_pos[:2] - initial_plate_pos[:2])
    )

    placement_candidate = bool(
        legacy_success
        and max_target_lift >= 0.03
        and max_lifted_run >= 3
        and upright_cos >= 0.7
        and abs(float(target_pos[2] - plate_pos[2])) < 0.08
        and plate_xy_displacement < 0.05
        and gripper_open
        and tcp_high
    )
    stable_place_steps = stable_place_steps + 1 if placement_candidate else 0
    physical_success = bool(placement_candidate and stable_place_steps >= 5)
    return {
        "legacy_success": legacy_success,
        "physical_success": physical_success,
        "xy_dist": xy_dist,
        "target_z": float(target_pos[2]),
        "plate_z": float(plate_pos[2]),
        "max_target_lift": float(max_target_lift),
        "max_lifted_run": int(max_lifted_run),
        "upright_cos": upright_cos,
        "plate_xy_displacement": plate_xy_displacement,
        "placement_candidate": placement_candidate,
        "stable_place_steps": int(stable_place_steps),
        "gripper_open": gripper_open,
        "tcp_high": tcp_high,
    }, stable_place_steps


def run_one_seed(policy, seed: int) -> dict:
    from video_io import write_video
    from mujoco_env.y_env2 import SimpleEnv2

    xml_path = PROJECT_ROOT / "asset" / "example_scene_y2.xml"
    env = SimpleEnv2(
        str(xml_path),
        action_type="joint_angle",
        state_type="joint_angle",
        seed=seed,
    )
    env.set_instruction(TASK_TEXT)
    if hasattr(policy, "reset"):
        policy.reset()

    initial_plate_pos = np.asarray(
        env.env.get_p_body("body_obj_plate_11"), dtype=np.float64
    ).copy()
    initial_target_z = float(env.env.get_p_body(env.obj_target)[2])
    max_target_lift = 0.0
    lifted_run = 0
    max_lifted_run = 0
    stable_place_steps = 0
    frames: list[np.ndarray] = []
    final = None
    action_step = 0
    started = time.perf_counter()

    try:
        # 不渲染时允许无窗口运行；RENDER=1 才要求 viewer 保持存活。
        while action_step < MAX_ACTION_STEPS and (
            not RENDER or env.env.is_viewer_alive()
        ):
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            state = env.get_joint_state()[:6]
            agent_image, wrist_image = env.grab_image()
            observation = {
                "observation.state": torch.as_tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0),
                "observation.image": image_tensor(agent_image)
                .to(DEVICE)
                .unsqueeze(0),
                "observation.wrist_image": image_tensor(wrist_image)
                .to(DEVICE)
                .unsqueeze(0),
                "task": [TASK_TEXT],
            }
            with torch.inference_mode():
                action = (
                    policy.select_action(observation)[0, :7]
                    .detach()
                    .cpu()
                    .numpy()
                )
            action = action.astype(np.float32)
            action[6] = np.clip(action[6], 0.0, 1.0)
            env.step(action)

            target_z = float(env.env.get_p_body(env.obj_target)[2])
            lift = target_z - initial_target_z
            max_target_lift = max(max_target_lift, lift)
            lifted_run = lifted_run + 1 if lift >= 0.03 else 0
            max_lifted_run = max(max_lifted_run, lifted_run)
            final, stable_place_steps = strict_snapshot(
                env,
                initial_plate_pos,
                max_target_lift,
                max_lifted_run,
                stable_place_steps,
            )

            if action_step % VIDEO_STRIDE == 0:
                frames.append(np.asarray(agent_image).copy())
            if RENDER:
                env.render()
            action_step += 1
            if final["physical_success"]:
                break

        if final is None:
            final, _ = strict_snapshot(
                env,
                initial_plate_pos,
                max_target_lift,
                max_lifted_run,
                stable_place_steps,
            )

        video_path = OUTPUT_DIR / f"{POLICY_TYPE}_seed{seed}.mp4"
        video_saved = False
        if frames:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            try:
                write_video(video_path, frames, VIDEO_FPS)
                video_saved = True
            except Exception as exc:
                print(f"视频保存失败，仍保留 JSONL 结果：{exc!r}", flush=True)

        return {
            "policy_type": POLICY_TYPE,
            "seed": int(seed),
            "instruction": TASK_TEXT,
            "action_steps": int(action_step),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "video": str(video_path) if video_saved else None,
            **final,
        }
    finally:
        close_viewer = getattr(env.env, "close_viewer", None)
        if close_viewer is not None:
            close_viewer()


def main() -> None:
    require_layout()
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "当前 DEVICE=cuda，但 torch.cuda.is_available() 为 False；"
            "请确认 ROCm/PyTorch 环境，或显式设置 DEVICE=cpu 做接口排查。"
        )

    policy_path = (
        Path(POLICY_PATH).expanduser()
        if POLICY_PATH
        else find_pretrained_model(MODEL_RUN_DIR)
    )
    print("TOPIC_ROOT =", TOPIC_ROOT)
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATASET_ROOT =", DATASET_ROOT)
    print("MODEL_RUN_DIR =", MODEL_RUN_DIR)
    print("POLICY_PATH =", policy_path)
    print("POLICY_TYPE =", POLICY_TYPE)
    print("DEVICE =", DEVICE, "RENDER =", RENDER)
    print("EVAL_SEEDS =", EVAL_SEEDS)

    policy = load_policy(POLICY_TYPE, policy_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in EVAL_SEEDS:
        result = run_one_seed(policy, seed)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    result_path = OUTPUT_DIR / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    physical = sum(bool(row.get("physical_success")) for row in results)
    legacy = sum(bool(row.get("legacy_success")) for row in results)
    print(f"完成：legacy_success={legacy}/{len(results)}")
    print(f"完成：physical_success={physical}/{len(results)}")
    print("结果文件：", result_path)


if __name__ == "__main__":
    main()
