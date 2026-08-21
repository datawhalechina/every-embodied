"""pi0 / SmolVLA / ACT 端到端闭环评估脚本（替代 nbconvert，规避 kernel died）

用法（在专题根目录执行）:
    POLICY_TYPE=pi0 \
    MODEL_RUN_DIR=$PWD/external/mujoco_pnp/ckpt/pi0_rocm_full/checkpoints/last/pretrained_model \
    EVAL_SEEDS=1000,1001,1002,1003 \
    /opt/venv/bin/python code/run_closed_loop.py

前置:
    export DISPLAY=:99 PYTHONPATH=$PWD/external/mujoco_pnp HF_ENDPOINT=https://hf-mirror.com
    export HF_TOKEN=$(cat /root/.hf_token)
"""
import os, sys, json, time, traceback
from pathlib import Path

TOPIC = str(Path(__file__).resolve().parents[1])  # code/ 的上一级 = 专题根
os.environ.setdefault("AMD_TOPIC_ROOT", TOPIC)
os.chdir(TOPIC)
sys.path.insert(0, TOPIC)
sys.path.insert(0, TOPIC + "/external/mujoco_pnp")

import numpy as np
import torch
from pathlib import Path

# ---- 环境变量 ----
POLICY_TYPE = os.environ.get("POLICY_TYPE", "act")
MODEL_RUN_DIR = os.environ.get("MODEL_RUN_DIR", "")
EVAL_SEEDS = [int(v) for v in os.environ.get("EVAL_SEEDS", "1000,1001,1002,1003").split(",")]
TASK_TEXT = os.environ.get("TASK_TEXT", "Place the blue mug on the plate.")
OUTPUT_DIR = Path(TOPIC) / "outputs" / f"{POLICY_TYPE}_closed_loop"
DATA_ROOT = Path(TOPIC) / "data" / "omy_pnp_language"

def find_pretrained_model(run_dir):
    run_dir = Path(run_dir)
    last = run_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last
    candidates = sorted((run_dir / "checkpoints").glob("*/pretrained_model"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"在 {run_dir} 下找不到 pretrained_model")

def image_tensor(array, size=(256, 256)):
    from PIL import Image
    image = Image.fromarray(array).convert("RGB").resize(size)
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous()

def load_policy(policy_type, policy_path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    if policy_type == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy as PolicyClass
    elif policy_type == "smolvla":
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PolicyClass
    elif policy_type == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as PolicyClass
    else:
        raise ValueError(f"不支持的 policy_type: {policy_type}")
    metadata = LeRobotDatasetMetadata("datawhale_eai_pnp_language", root=str(DATA_ROOT))
    policy = PolicyClass.from_pretrained(str(policy_path), dataset_stats=metadata.stats)
    policy.to("cuda")
    policy.eval()
    return policy

def run_one_seed(policy, seed, output_dir, max_action_steps=300):
    import imageio.v2 as imageio
    from mujoco_env.y_env2 import SimpleEnv2
    xml_path = TOPIC + "/external/mujoco_pnp/asset/example_scene_y2.xml"
    env = SimpleEnv2(xml_path, action_type="joint_angle", state_type="joint_angle", seed=seed)
    env.set_instruction(TASK_TEXT)

    initial_target_z = float(env.env.get_p_body(env.obj_target)[2])
    initial_plate_pos = env.env.get_p_body(env.obj_plate)
    frames, action_step = [], 0
    while action_step < max_action_steps and env.env.is_viewer_alive():
        env.step_env()
        if not env.env.loop_every(HZ=20):
            continue
        state = env.get_joint_state()[:6]
        agent_image, wrist_image = env.grab_image()
        observation = {
            "observation.state": torch.as_tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0),
            "observation.image": image_tensor(agent_image).to("cuda").unsqueeze(0),
            "observation.wrist_image": image_tensor(wrist_image).to("cuda").unsqueeze(0),
            "task": [TASK_TEXT],
        }
        with torch.inference_mode():
            action = policy.select_action(observation)[0, :7].detach().cpu().numpy()
        action[6] = np.clip(action[6], 0.0, 1.0)
        env.step(action.astype(np.float32))
        frames.append(np.asarray(agent_image))
        action_step += 1
        if action_step % 50 == 0:
            print(f"  seed {seed} 动作步 {action_step}", flush=True)

    legacy_success = bool(env.check_success())
    p_target = env.env.get_p_body(env.obj_target)
    p_plate = env.env.get_p_body("body_obj_plate_11")
    xy_dist = float(np.linalg.norm(p_target[:2] - p_plate[:2]))
    max_target_lift = float(p_target[2] - initial_target_z)
    target_R = np.asarray(env.env.get_R_body(env.obj_target), dtype=np.float64)
    upright_cos = float(target_R[2, 2])

    video_path = output_dir / f"{POLICY_TYPE}_seed{seed}.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=10, quality=8)

    result = {
        "policy_type": POLICY_TYPE, "seed": seed, "instruction": TASK_TEXT,
        "action_steps": action_step, "video": str(video_path),
        "legacy_success": legacy_success, "physical_success": legacy_success,
        "xy_dist": xy_dist, "max_target_lift": max_target_lift,
        "upright_cos": upright_cos,
    }
    print(f"  seed {seed}: success={result['physical_success']} xy_dist={xy_dist:.3f} lift={max_target_lift:.4f}", flush=True)
    return result

if __name__ == "__main__":
    print(f"policy_type={POLICY_TYPE} seeds={EVAL_SEEDS}", flush=True)
    print(f"model_run_dir={MODEL_RUN_DIR}", flush=True)
    if not MODEL_RUN_DIR:
        print("❌ 请设置 MODEL_RUN_DIR", flush=True)
        sys.exit(1)
    try:
        policy_path = find_pretrained_model(MODEL_RUN_DIR)
    except FileNotFoundError:
        policy_path = Path(MODEL_RUN_DIR)
    print(f"加载权重: {policy_path}", flush=True)
    policy = load_policy(POLICY_TYPE, policy_path)
    print("✅ 权重加载成功", flush=True)

    results = []
    for seed in EVAL_SEEDS:
        r = run_one_seed(policy, seed, OUTPUT_DIR)
        results.append(r)
    with open(OUTPUT_DIR / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(1 for r in results if r["physical_success"])
    print(f"\n✅ 完成: physical_success = {n_ok}/{len(results)}", flush=True)
    print(f"产物目录: {OUTPUT_DIR}", flush=True)
