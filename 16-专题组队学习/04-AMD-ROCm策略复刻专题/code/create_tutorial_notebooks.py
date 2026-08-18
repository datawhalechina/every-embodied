#!/usr/bin/env python3
"""Create the companion notebooks for the AMD ROCm tutorial topic."""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"


METADATA = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}


COMMON_SETUP = r'''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


def find_topic_root():
    override = (
        os.environ.get("AMD_TOPIC_ROOT")
        or os.environ.get("NOTEBOOK_TOPIC_ROOT")
        or os.environ.get("TOPIC_ROOT")
    )
    roots = [Path(override).expanduser()] if override else []
    cwd = Path.cwd().resolve()
    roots.extend([cwd, *cwd.parents])

    candidates = []
    for root in roots:
        candidates.extend(
            [
                root,
                root / "16-专题组队学习" / "04-AMD-ROCm策略复刻专题",
                root / "04-AMD-ROCm策略复刻专题",
            ]
        )
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "assets" / "metrics_snapshot.json").exists():
            return candidate
    raise RuntimeError(
        "找不到 AMD ROCm 专题目录。请从仓库根目录、专题目录启动 Jupyter，"
        "或设置 AMD_TOPIC_ROOT。"
    )


TOPIC_ROOT = find_topic_root()
ASSET_DIR = TOPIC_ROOT / "assets"
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", TOPIC_ROOT / "external" / "mujoco_pnp")
).expanduser()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", TOPIC_ROOT / "data")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", TOPIC_ROOT / "outputs"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "ckpt"))

print("TOPIC_ROOT =", TOPIC_ROOT)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("DATA_ROOT =", DATA_ROOT)
print("OUTPUT_ROOT =", OUTPUT_ROOT)
print("MODEL_ROOT =", MODEL_ROOT)
'''


DISPLAY_HELPERS = r'''
try:
    from IPython.display import Image, Markdown, Video, display
except Exception:
    class Markdown(str):
        pass

    def display(obj):
        print(obj)

    def Image(filename=None, width=None):
        return f"[image] {filename}"

    def Video(filename=None, embed=False, width=None, **kwargs):
        return f"[video] {filename}"


def show_video(filename, title=None, width=960):
    path = ASSET_DIR / filename
    if title:
        display(Markdown(f"**{title}**"))
    if not path.exists():
        print(f"缺少视频素材：{path}")
        return
    try:
        display(Video(filename=str(path), embed=True, width=width, html_attributes="controls muted"))
    except TypeError:
        display(Video(filename=str(path), embed=True, width=width))


def show_asset(filename, width=960):
    path = ASSET_DIR / filename
    if path.exists():
        if path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}:
            show_video(filename, width=width)
            return
        display(Image(filename=str(path), width=width))
    else:
        print(f"缺少素材：{path}")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    display(Markdown("\n".join(lines)))
'''


TRAINING_HELPERS = r'''
import shlex

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("当前环境缺少 PyYAML，请先执行 pip install pyyaml。") from exc


def require_project_layout():
    required = [
        PROJECT_ROOT / "train_model.py",
        PROJECT_ROOT / "env_config.py",
        PROJECT_ROOT / "asset" / "example_scene_y2.xml",
        PROJECT_ROOT / "mujoco_env" / "y_env2.py",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "PROJECT_ROOT 不是 04mujoco 教程目录，缺少：\n"
            + "\n".join(str(path) for path in missing)
        )
    return True


def dataset_report(dataset_root):
    dataset_root = Path(dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not info_path.exists():
        raise FileNotFoundError(f"找不到 LeRobot 元数据：{info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    tasks = []
    if tasks_path.exists():
        tasks = [
            json.loads(line)["task"]
            for line in tasks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    features = info.get("features", {})
    rows = [
        ("repo_id", info.get("repo_id", "")),
        ("episodes", info.get("total_episodes", 0)),
        ("frames", info.get("total_frames", 0)),
        ("fps", info.get("fps", "")),
        ("state shape", features.get("observation.state", {}).get("shape")),
        ("action shape", features.get("action", {}).get("shape")),
        ("tasks", " / ".join(tasks)),
    ]
    md_table(["数据项", "读取结果"], rows)
    return info, tasks


def make_training_config(
    policy_type,
    dataset_repo_id,
    dataset_root,
    output_dir,
    steps,
    batch_size,
    chunk_size,
    n_action_steps,
    save_freq,
    seed=42,
):
    return {
        "dataset": {"repo_id": dataset_repo_id, "root": str(Path(dataset_root))},
        "policy": {
            "type": policy_type,
            "chunk_size": int(chunk_size),
            "n_action_steps": int(n_action_steps),
            "device": "cuda",
        },
        "save_checkpoint": True,
        "output_dir": str(Path(output_dir)),
        "batch_size": int(batch_size),
        "job_name": Path(output_dir).name,
        "resume": False,
        "seed": int(seed),
        "num_workers": 4,
        "steps": int(steps),
        "eval_freq": 0,
        "log_freq": max(1, min(50, int(steps))),
        "save_freq": int(save_freq),
        "use_policy_training_preset": True,
        "wandb": {
            "enable": False,
            "project": f"every_embodied_{policy_type}",
            "entity": None,
            "disable_artifact": True,
        },
    }


def write_training_config(name, config):
    config_dir = OUTPUT_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print("已写出配置：", path)
    return path


def training_command(config_path):
    return [
        sys.executable,
        str(PROJECT_ROOT / "train_model.py"),
        "--config_path",
        str(Path(config_path)),
    ]


def run_training(config_path, enabled=False):
    require_project_layout()
    command = training_command(config_path)
    print("$", shlex.join(command))
    if not enabled:
        print("当前只预览命令。确认配置后，把对应 RUN_* 开关改为 True。")
        return None
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def show_rocm_resources():
    command = ["rocm-smi", "--showuse", "--showmemuse", "--showtemp"]
    print("$", shlex.join(command))
    if shutil.which(command[0]) is None:
        print("未找到 rocm-smi。确认当前机器是否安装 ROCm。")
        return
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
'''


ROLLOUT_HELPERS = r'''
import random
import time

import numpy as np
from PIL import Image

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def find_pretrained_model(run_dir):
    run_dir = Path(run_dir)
    last = run_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last
    candidates = sorted((run_dir / "checkpoints").glob("*/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(f"没有在 {run_dir} 下找到 pretrained_model")
    return candidates[-1]


def image_tensor(array, size=(256, 256)):
    import torch

    image = Image.fromarray(array).convert("RGB").resize(size)
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous()


def load_policy(policy_type, policy_path, dataset_repo_id, dataset_root, device="cuda"):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if policy_type == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy as PolicyClass
    elif policy_type == "smolvla":
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PolicyClass
    elif policy_type == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as PolicyClass
    else:
        raise ValueError(f"不支持的 policy_type：{policy_type}")

    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    policy = PolicyClass.from_pretrained(str(policy_path), dataset_stats=metadata.stats)
    policy.to(device)
    policy.eval()
    return policy


def strict_snapshot(
    env,
    initial_target_z,
    initial_plate_pos,
    max_target_lift,
    max_lifted_run,
    stable_place_steps,
):
    target_pos = np.asarray(env.env.get_p_body(env.obj_target), dtype=np.float64)
    plate_pos = np.asarray(env.env.get_p_body("body_obj_plate_11"), dtype=np.float64)
    target_R = np.asarray(env.env.get_R_body(env.obj_target), dtype=np.float64)
    xy_dist = float(np.linalg.norm(target_pos[:2] - plate_pos[:2]))
    upright_cos = float(target_R[2, 2])
    gripper_open = bool(float(env.env.get_qpos_joint("rh_r1")[0]) < 0.1)
    tcp_high = bool(float(env.env.get_p_body("tcp_link")[2]) > 0.9)
    legacy_success = bool(env.check_success())
    plate_xy_displacement = float(np.linalg.norm(plate_pos[:2] - initial_plate_pos[:2]))
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


def run_closed_loop(
    policy,
    policy_type,
    instruction,
    seeds,
    output_dir,
    device="cuda",
    max_action_steps=300,
    render=True,
):
    from mujoco_env.y_env2 import SimpleEnv2
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = PROJECT_ROOT / "asset" / "example_scene_y2.xml"
    results = []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        env = SimpleEnv2(str(xml_path), action_type="joint_angle", state_type="joint_angle", seed=seed)
        env.set_instruction(instruction)
        policy.reset()

        initial_target_z = float(env.env.get_p_body(env.obj_target)[2])
        initial_plate_pos = np.asarray(
            env.env.get_p_body("body_obj_plate_11"), dtype=np.float64
        ).copy()
        max_target_lift = 0.0
        lifted_run = 0
        max_lifted_run = 0
        stable_place_steps = 0
        frames = []
        started = time.perf_counter()
        final = None

        action_step = 0
        while action_step < max_action_steps and env.env.is_viewer_alive():
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            state = env.get_joint_state()[:6]
            agent_image, wrist_image = env.grab_image()
            observation = {
                "observation.state": torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                "observation.image": image_tensor(agent_image).to(device).unsqueeze(0),
                "observation.wrist_image": image_tensor(wrist_image).to(device).unsqueeze(0),
                "task": [instruction],
            }
            with torch.inference_mode():
                action = policy.select_action(observation)[0, :7].detach().cpu().numpy()
            action[6] = np.clip(action[6], 0.0, 1.0)
            env.step(action.astype(np.float32))

            target_z = float(env.env.get_p_body(env.obj_target)[2])
            lift = target_z - initial_target_z
            max_target_lift = max(max_target_lift, lift)
            lifted_run = lifted_run + 1 if lift >= 0.03 else 0
            max_lifted_run = max(max_lifted_run, lifted_run)
            final, stable_place_steps = strict_snapshot(
                env,
                initial_target_z,
                initial_plate_pos,
                max_target_lift,
                max_lifted_run,
                stable_place_steps,
            )

            if action_step % 2 == 0:
                frames.append(np.asarray(agent_image))
            if render:
                env.render()
            action_step += 1
            if final["physical_success"]:
                break

        video_path = output_dir / f"{policy_type}_seed{seed}.mp4"
        video_saved = False
        if frames and imageio is not None:
            try:
                imageio.mimsave(video_path, frames, fps=10, quality=8)
                video_saved = True
            except Exception as exc:
                print("视频保存失败：", repr(exc))
        elif frames:
            print("未安装 imageio，跳过视频保存。可执行 pip install imageio imageio-ffmpeg 后重跑。")
        row = {
            "policy_type": policy_type,
            "seed": int(seed),
            "instruction": instruction,
            "action_steps": int(action_step),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "video": str(video_path) if video_saved else None,
            **(final or {}),
        }
        print(json.dumps(row, ensure_ascii=False))
        results.append(row)
        env.env.close_viewer()

    result_path = output_dir / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    physical = sum(row.get("physical_success", False) for row in results)
    print(f"physical_success = {physical}/{len(results)}")
    print("结果文件：", result_path)
    return results
'''


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n") + "\n",
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOK_DIR / filename
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


NOTEBOOKS: dict[str, list[dict]] = {}


def make_training_notebook(
    filename: str,
    number: str,
    policy_type: str,
    title: str,
    summary: str,
    full_steps: int,
    batch_size: int,
    chunk_size: int,
    n_action_steps: int,
    model_note: str,
) -> None:
    gate_cells: list[dict] = []
    if policy_type == "pi0":
        gate_cells = [
            md("## Checkpoint 2：检查 Hugging Face gated 权限"),
            code(
                r'''
token_present = bool(os.environ.get("HF_TOKEN"))
print("HF_TOKEN 已注入当前进程：", token_present)
if not token_present:
    print("请在终端私密设置 HF_TOKEN，再重新启动 Jupyter kernel。不要把 token 写进 Notebook。")
'''
            ),
            md("这里只检查环境变量是否存在，不打印 token。pi_0 训练还需要账号已经获得 PaliGemma 等 gated 权重的访问权限。"),
        ]

    config_cell = f'''
MODEL_TYPE = {policy_type!r}
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language")
TRAIN_DATA_ROOT = Path(
    os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "omy_pnp_language")
).expanduser()

RUN_SMOKE = False
RUN_FULL_TRAIN = False

SMOKE_OUTPUT = MODEL_ROOT / f"{{MODEL_TYPE}}_rocm_smoke"
FULL_OUTPUT = MODEL_ROOT / f"{{MODEL_TYPE}}_rocm_full"

smoke_config = make_training_config(
    policy_type=MODEL_TYPE,
    dataset_repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
    output_dir=SMOKE_OUTPUT,
    steps=2,
    batch_size=min(4, {batch_size}),
    chunk_size={chunk_size},
    n_action_steps={n_action_steps},
    save_freq=2,
)
full_config = make_training_config(
    policy_type=MODEL_TYPE,
    dataset_repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
    output_dir=FULL_OUTPUT,
    steps={full_steps},
    batch_size={batch_size},
    chunk_size={chunk_size},
    n_action_steps={n_action_steps},
    save_freq=max(1, {full_steps} // 2),
)

smoke_config_path = write_training_config(f"{{MODEL_TYPE}}_smoke", smoke_config)
full_config_path = write_training_config(f"{{MODEL_TYPE}}_full", full_config)
print("smoke output =", SMOKE_OUTPUT)
print("full output =", FULL_OUTPUT)
'''

    cells = [
        md(
            f"""
            # {number} {title}

            {summary}

            这一节从 LeRobot 数据检查开始，依次完成 2-step smoke、正式训练、ROCm 资源观察和 checkpoint 定位。所有长任务默认关闭，先确认生成的配置，再显式打开运行开关。
            """
        ),
        code(COMMON_SETUP),
        code(DISPLAY_HELPERS),
        code(TRAINING_HELPERS),
        md("## Checkpoint 1：确认项目和训练数据"),
        code(
            r'''
require_project_layout()
print("train_model.py =", PROJECT_ROOT / "train_model.py")
print("训练数据 =", Path(os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "omy_pnp_language")))
'''
        ),
        code(
            r'''
candidate = Path(os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "omy_pnp_language"))
if (candidate / "meta" / "info.json").exists():
    dataset_report(candidate)
else:
    print("训练数据还没准备好：", candidate)
    print("先完成 07_data_collection_and_audit.ipynb，或设置 TRAIN_DATA_ROOT 指向已有 LeRobot 数据集。")
'''
        ),
    ]
    cells.extend(gate_cells)
    checkpoint_number = 3 if policy_type == "pi0" else 2
    cells.extend(
        [
            md(f"## Checkpoint {checkpoint_number}：生成 smoke 与正式训练配置"),
            code(config_cell),
            md(
                "配置文件会写到 `$OUTPUT_ROOT/configs`，checkpoint 写到 `$MODEL_ROOT`。这两个目录应放在容量充足的磁盘，不要写进 Git 仓库。"
            ),
            md(f"## Checkpoint {checkpoint_number + 1}：先跑 2-step smoke"),
            code(
                r'''
show_rocm_resources()
run_training(smoke_config_path, enabled=RUN_SMOKE)
'''
            ),
            md(
                "2-step smoke 只证明数据加载、模型构造、forward/backward、optimizer step 和 checkpoint 写出可用；它不证明模型收敛，更不能替代 MuJoCo closed-loop 成功率。"
            ),
            md(f"## Checkpoint {checkpoint_number + 2}：启动正式训练"),
            code(
                r'''
run_training(full_config_path, enabled=RUN_FULL_TRAIN)
'''
            ),
            md(model_note),
            md(
                """
                常见恢复顺序：显存不足先减小 `batch_size`；出现 DataLoader worker / pickle 错误时把 YAML 中的 `num_workers` 改为 `0`；下载报 `401/403` 时检查账号权限和私密 `HF_TOKEN`；写 checkpoint 失败先检查 `$MODEL_ROOT` 所在磁盘。ROCm 的 MIOpen 数据库 warning 如果没有伴随训练退出可以记录后继续观察，出现 kernel 退出、NaN 或进程消失则必须停止长训练并回到 smoke。
                """
            ),
            md(f"## Checkpoint {checkpoint_number + 3}：定位 checkpoint 并检查资源"),
            code(
                r'''
show_rocm_resources()
if (FULL_OUTPUT / "checkpoints").exists():
    checkpoints = sorted((FULL_OUTPUT / "checkpoints").glob("*/pretrained_model"))
    print("可用 checkpoint：")
    for path in checkpoints:
        print(" -", path)
else:
    print("正式训练尚未运行：", FULL_OUTPUT)
'''
            ),
            md(
                "训练完成后不要只挑 loss 最低的节点。下一步进入 `11_mujoco_closed_loop_deploy.ipynb`，用固定 seed、严格 `physical_success` 和视频复核比较中间 checkpoint 与最终 checkpoint。"
            ),
        ]
    )
    NOTEBOOKS[filename] = cells


NOTEBOOKS["01_device_env_check.ipynb"] = [
    md(
        """
        # 01 AMD ROCm 设备与环境确认

        这一节的目标是先证明设备、环境、缓存和权限链路是可用的。很多训练失败看起来像模型问题，实际可能是 ROCm 没识别 GPU、缓存放错磁盘、网络无法下载权重，或者统一内存被其它进程挤占。

        Notebook 里主要做三件事：检查硬件和 PyTorch、规划大文件目录、形成一张设备资源表。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：ROCm 和 PyTorch 是否能看到 GPU"),
    code(
        r'''
def run_cmd(cmd):
    print("$", " ".join(cmd))
    if shutil.which(cmd[0]) is None:
        print(f"未找到命令：{cmd[0]}")
        return
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)


run_cmd(["rocm-smi", "--showuse", "--showtemp", "--showmemuse"])

try:
    import torch
    print("torch =", torch.__version__)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device =", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch 检查失败：", repr(exc))
'''
    ),
    md("如果 `rocm-smi` 正常、`torch.cuda.is_available()` 为 `True`，说明 ROCm 与 PyTorch 的基础链路已经打通。若前者正常但后者为 `False`，优先检查 PyTorch 是否为 ROCm build。"),
    md("## Checkpoint 2：统一内存、显存和磁盘目录"),
    code(
        r'''
paths = [
    ("项目源码", PROJECT_ROOT),
    ("数据目录", DATA_ROOT),
    ("模型与 checkpoint", MODEL_ROOT),
    ("Notebook 输出", OUTPUT_ROOT),
    ("Hugging Face cache", Path(os.environ.get("HF_HOME", TOPIC_ROOT / "cache" / "huggingface"))),
]
md_table(["项目", "建议路径", "是否存在"], [(name, f"`{path}`", path.exists()) for name, path in paths])
'''
    ),
    md("大模型缓存、数据集、checkpoint 和批量视频都可能很大。实验报告里说明目录规划和磁盘类型即可，不需要把这些文件放进教程目录。"),
    md("## Checkpoint 3：设备资源表模板"),
    code(
        r'''
rows = [
    ("设备型号", "AMD Ryzen AI MAX+ / Radeon GPU"),
    ("系统版本", "Ubuntu / WSL / 其它"),
    ("ROCm 版本", "填写 rocm-smi 或包版本"),
    ("PyTorch 版本", "填写 torch.__version__"),
    ("空闲温度", "例如 30-45 C"),
    ("训练温度", "例如 75-85 C"),
    ("ACT 训练 VRAM", "填写 rocm-smi 观察值"),
    ("SmolVLA 训练 VRAM", "填写 rocm-smi 观察值"),
    ("pi0 smoke 状态", "未跑 / 通过 / 失败原因"),
]
md_table(["项目", "记录"], rows)
'''
    ),
    md("完成本节后，应当能判断这台 AMD 设备是否已经具备继续训练和评估的条件。"),
]


NOTEBOOKS["02_physical_success_review.ipynb"] = [
    md(
        """
        # 02 物理成功评估与视频复核

        这一节解决一个关键问题：日志里的 success 是否真的代表机器人夹起杯子并放到盘子上。这里把旧的几何成功条件和 `physical_success` 分开统计，并用视频关键帧复核成功与失败。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：读取本专题的示例指标"),
    code(
        r'''
snapshot_path = ASSET_DIR / "metrics_snapshot.json"
snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
print(json.dumps(snapshot["best_summary"], ensure_ascii=False, indent=2))
'''
    ),
    md("## Checkpoint 2：理解 `physical_success`"),
    code(
        r'''
def physical_success(row, min_lift=0.03, min_lift_steps=3, upright_threshold=0.7):
    """示例判定逻辑。实际项目中应以 eval 脚本记录的字段为准。"""
    legacy = bool(row.get("legacy_success", row.get("success", False)))
    max_lift = float(row.get("max_target_lift", row.get("max_mug_lift", 0.0)))
    lifted_steps = int(row.get("lifted_steps", 0))
    upright = float(row.get("final_target_upright_cos", row.get("final_mug_upright_cos", 1.0)))
    return legacy and max_lift >= min_lift and lifted_steps >= min_lift_steps and upright >= upright_threshold


examples = [
    {"name": "几何成功但倒杯", "success": True, "max_mug_lift": 0.08, "lifted_steps": 80, "final_mug_upright_cos": 0.2},
    {"name": "推到盘边但没夹起", "success": True, "max_mug_lift": 0.005, "lifted_steps": 0, "final_mug_upright_cos": 0.99},
    {"name": "真抓取并直立放置", "success": True, "max_mug_lift": 0.09, "lifted_steps": 120, "final_mug_upright_cos": 0.96},
]
md_table(["样例", "physical_success"], [(e["name"], physical_success(e)) for e in examples])
'''
    ),
    md("旧 success 可以作为辅助指标，但不能替代视频和物理状态复核。特别是抓杯任务里，推、挤、倒杯都可能让终态几何看起来接近成功。"),
    md("## Checkpoint 3：复核成功与失败关键帧"),
    code(
        r'''
show_video(
    "smolvla_weighted500_red_failure_seed8.mp4",
    "SmolVLA 真实失败回放：红杯 seed8",
    width=1100,
)
show_video(
    "smolvla_weighted500_blue_success_seed0.mp4",
    "SmolVLA 真实成功回放：蓝杯 seed0",
    width=1100,
)
show_video(
    "pnp_four_view_strict_success.mp4",
    "四视角严格成功回放：用于检查接触、抬升、搬运和释放",
    width=1100,
)
show_asset("act_dagger_progress_curve.png", width=1100)
'''
    ),
    md("看视频时，沿时间轴观察四件事：是否接触目标杯、是否稳定夹起、是否搬运到盘子上、终态是否直立。"),
    md("## Checkpoint 4：批量 JSONL 统计模板"),
    code(
        r'''
def summarize_jsonl(path):
    path = Path(path)
    if not path.exists():
        print(f"文件不存在：{path}")
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    legacy = sum(bool(r.get("success") or r.get("legacy_success")) for r in rows)
    physical = sum(bool(r.get("physical_success")) for r in rows)
    return {"total": total, "legacy_success": legacy, "physical_success": physical}


example_jsonl = OUTPUT_ROOT / "eval_result.jsonl"
print("把自己的 JSONL 放到这里后运行：", example_jsonl)
print(summarize_jsonl(example_jsonl))
'''
    ),
]


NOTEBOOKS["03_act_dagger_diagnostics.ipynb"] = [
    md(
        """
        # 03 ACT 在 ROCm 上的迁移与 DAgger 诊断

        这一节用 ACT 说明：训练 loss 下降不等于闭环 rollout 成功。先看诊断曲线，再把数据回放、open-loop、closed-loop、失败视频和 DAgger 串成一条排障链路。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：查看 ACT 诊断曲线"),
    code(
        r'''
snapshot = json.loads((ASSET_DIR / "metrics_snapshot.json").read_text(encoding="utf-8"))
rows = []
for item in snapshot["act_progress"]:
    rows.append((item["label"], f'{item["success"]}/{item["total"]}', f'{item["rate"]:.1%}'))
md_table(["阶段", "physical_success", "成功率"], rows)
show_asset("act_dagger_progress_curve.png", width=960)
'''
    ),
    md("这条曲线不是为了证明 ACT 已经完美泛化，而是为了展示排障方向：先确认数据和闭环状态，再决定是否需要 DAgger。"),
    md("## Checkpoint 2：ACT 严格评估命令模板"),
    code(
        r'''
act_policy = MODEL_ROOT / "act_best" / "step_5000"
cmd = f"""
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
./.venv/bin/python eval_policy_success.py \\
  --policy act \\
  --act-policy-path "{act_policy}" \\
  --physical-success \\
  --seed-start 1000 \\
  --episodes 10 \\
  --max-action-steps 1200 \\
  --output-jsonl "$OUTPUT_ROOT/act_seen_1000_1009.jsonl"
"""
print(cmd)
'''
    ),
    md("正式训练或批量评估更适合在终端或后台脚本里跑。Notebook 这里负责保存命令模板、解释参数含义和读取结果。"),
    md("## Checkpoint 3：成功与失败视频对照"),
    code(
        r'''
show_asset("act_dagger_progress_curve.png", width=1100)
display(Markdown("本轻量分支没有提交原始 ACT rollout 视频，因此这里不使用占位关键帧。拿到自己的 OUTPUT_ROOT 后，可按 README 说明重新生成并复核 ACT 视频。"))
'''
    ),
    md("ACT 的失败常常不是完全不动，而是接近、夹取、搬运或释放中的某一段出问题。复核视频时要写出失败发生在哪个阶段。"),
    md("## Checkpoint 4：记录 DAgger 实验"),
    code(
        r'''
rows = [
    ("prefix 长度", "例如 40 个 control tick"),
    ("oracle 类型", "scripted policy / human correction"),
    ("timestamp offset", "例如 correction episode 使用 2.0"),
    ("采样权重", "例如 correction episode weight=0.25"),
    ("评估 seed", "seen / mixed / heldout 分开写"),
    ("替换基线条件", "必须同一 physical_success 口径超过旧基线"),
]
md_table(["记录项", "示例"], rows)
'''
    ),
]


NOTEBOOKS["04_smolvla_weighted_sampling.ipynb"] = [
    md(
        """
        # 04 SmolVLA 在 ROCm 上的迁移与采样加权

        这一节用 SmolVLA 说明：语言策略不能只看总体成功率。红杯和蓝杯指令要拆开评估，再比较复制 episode 与 Weighted sampler 的效果。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：读取红杯/蓝杯固定指令结果"),
    code(
        r'''
snapshot = json.loads((ASSET_DIR / "metrics_snapshot.json").read_text(encoding="utf-8"))
rows = []
for item in snapshot["smolvla_forced_instruction_n10"]:
    rows.append((item["label"], item["red_text"], item["blue_text"]))
md_table(["策略", "红杯 physical_success", "蓝杯 physical_success"], rows)
show_asset("smolvla_red_blue_success.png", width=1100)
'''
    ),
    md("baseline 红杯好、蓝杯差，说明模型不是完全不会抓，而是任务条件和颜色相关分布没有学稳。"),
    md("## Checkpoint 2：复制 episode 与 Weighted sampler 的区别"),
    code(
        r'''
rows = [
    ("复制 episode", "直接改变数据集 episode 分布", "实现简单，但容易让模型偏向某个颜色"),
    ("Weighted sampler", "不改原始 parquet，只改采样概率", "更容易回滚，也更适合小数据平衡"),
    ("中间 checkpoint 评估", "不要只看 final step", "本示例里 step500 比 step1000 更均衡"),
]
md_table(["方法", "做法", "为什么要注意"], rows)
'''
    ),
    md("## Checkpoint 3：重新生成图表"),
    code(
        r'''
cmd = [
    sys.executable,
    str(TOPIC_ROOT / "code" / "generate_tutorial_assets.py"),
    "--source-root",
    str(OUTPUT_ROOT),
]
print("如果 OUTPUT_ROOT 中有本章需要的 TSV/JSONL/MP4，可以运行：")
print(" ".join(cmd))
# subprocess.run(cmd, check=True)
'''
    ),
    md("上面的单元默认不直接执行，避免输出目录还没准备好时覆盖教学素材。确认 `$OUTPUT_ROOT` 正确后，取消最后一行注释即可；缺失源视频会被跳过，不会生成不可播放的占位图。"),
    md("## Checkpoint 4：蓝杯失败和修复后成功对照"),
    code(
        r'''
show_video(
    "smolvla_weighted500_red_failure_seed8.mp4",
    "已提交的 SmolVLA 失败参考视频：红杯 seed8",
    width=1100,
)
show_video(
    "smolvla_weighted500_blue_success_seed0.mp4",
    "已提交的 SmolVLA 成功参考视频：蓝杯 seed0",
    width=1100,
)
'''
    ),
]


NOTEBOOKS["05_pi0_smoke_gate.ipynb"] = [
    md(
        """
        # 05 pi_0 权限 smoke 与训练门控

        这一节不急着启动长训练，而是先检查 PaliGemma、Hugging Face gated 权限、权重下载、数据加载和 1-step smoke。只有 smoke 通过后，正式训练才有意义。
        """
    ),
    md(
        """
        ## 重要更正：旧 30-seed 面板不是位置泛化

        `y_env2.py` 曾把任意环境 seed 都写成 `np.random.seed(seed=0)`。旧 `1010-1039` 面板只改变了 Pi0 策略采样，杯子和盘子位置始终固定。那些结果仍能分析 scaffold 的动作时序，却不能作为空间泛化证据。修复后，固定环境 raw/head 对照是 `0/12 -> 6/12`，真正随机环境 gate 只有 `1/4`；下一步必须重采多位置数据。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：权限检查命令模板"),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"
HF_TOKEN_STDIN=1 REQUIRE_PROXY=1 ./install_hf_token_for_pi0.sh
""")
'''
    ),
    md("这个命令形态强调两件事：token 不写进 Notebook，脚本先验证 `whoami`、PaliGemma 和 `lerobot/pi0`，全部通过后再保存。"),
    md("## Checkpoint 2：1-step smoke 证明什么"),
    code(
        r'''
rows = [
    ("gated model 权限", "能读取 PaliGemma / pi0 所需配置"),
    ("数据加载", "能读取 LeRobot 数据集和 observation/action key"),
    ("policy 构造", "pi0 policy 能初始化"),
    ("一次训练步", "forward/backward/optimizer step 能跑通"),
    ("checkpoint 写出", "输出目录出现 smoke checkpoint"),
]
md_table(["检查项", "通过后说明什么"], rows)
'''
    ),
    md("1-step smoke 不证明策略收敛，也不代表成功率。它只是正式训练前的门槛。"),
    md("## Checkpoint 3：训练命令模板"),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"
RUN_SMOKE=1 RUN_FULL_TRAIN=1 PI0_STEPS=20000 PI0_BATCH_SIZE=4 \\
  ./run_pi0_train_eval_after_hf_ready.sh
""")
'''
    ),
    md("正式训练建议放到终端、tmux 或后台脚本中执行。Notebook 负责记录参数、解释门控和读取训练后的 summary。"),
    md("## Checkpoint 4：阶段性状态图"),
    code('show_asset("model_status_summary.png", width=900)'),
    md("## Checkpoint 5：`tcp_to_plate` 后段 finisher 诊断"),
    md(
        """
        raw pi_0 目前还不能写成复刻成功。更有价值的一轮诊断是把后段 finisher 的 state 从 19 维扩到 22 维，额外加入 `tcp_to_plate = tcp_link_xyz - plate_xyz`。这一步让模型显式知道盘心相对 TCP 的方向，避免后段只靠图像和 phase 隐式猜目标。

        这个实验仍然是诊断 scaffold：前缀由 oracle 提供，schedule 从 `move_preplace` 对齐，finisher 使用 22D state。后续 30-seed 排查发现，短 `move_preplace` schedule 会提前 lower / release；强制使用长 schedule 模板后，改好版 scaffold 能稳定通过 30 个策略采样 seed（环境实际固定为 seed 0）。这个结果证明瓶颈被定位和修复，但不代表 raw pi_0 已经端到端成功。

        下一步开始拆 scaffold。第一个被替换的是手写 gripper 规则：用 phase-only logistic head 预测夹爪开闭，EEF/arm 仍由 22D finisher 输出。full-state gripper head 虽然训练集准确率 `100%`，但闭环上线失败；phase-only head 只看 `timestamp + phase_index_norm + phase_onehot11`，在完整策略采样 seeds（环境实际固定为 seed 0） `1010-1039` 上仍保持 strict `30/30`。这个对照说明，小数据短事件 head 的输入要克制，不要把闭环会漂移的机械臂状态全塞进去。

        第二块脚手架先用固定阈值 adaptive gate 替代强制长 `move_preplace` 模板，完整策略采样 seeds（环境实际固定为 seed 0） `1010-1039` 仍是 strict `30/30`，但其中一部分依赖 `max_steps=180` 安全兜底。继续往前走，我们训练了 logistic transition head。第一版 full head 训练集准确率约 `99.27%`，但 seed `1010` 闭环早放失败；第二版只保留 `tcp_to_plate_xy + local_step_norm`，训练集准确率约 `95.01%`，却在完整策略采样 seeds（环境实际固定为 seed 0） `1010-1039` 上达到 strict `30/30`、legacy `30/30`。30 条里 `29/30` 是 transition head 主动触发，只有 seed `1021` 走到 max-step 安全兜底。这个对照说明，小数据 phase head 的输入也要克制，稳定几何线索比训练集高准确率更重要。

        第三层继续拆前段 contact primitive。naive all-head 把所有 transition 都按 phase tail 学，训练集指标不差，但 smoke `1010/1011` 变成 `0/2`，因为 `descend_to_close` 会在 TCP 还高出抓取 floor 约 `0.08 m` 时提前触发。修复版把 `pregrasp_to_descend` 改成几何标签，特征加入 TCP 到 grasp / pregrasp / floor 的相对量，并在部署时加 `descend_floor_guard`。完整策略采样 seeds（环境实际固定为 seed 0） `1010-1039` 为 strict `30/30`、legacy `30/30`；核心切换 `pregrasp->descend`、`descend->close`、`close->lift` 都是 `30/30` 由 head 触发，floor guard 共阻挡 `342` 次高空 close。这仍是 contact scaffold，不是 raw pi_0 端到端成功。

        第四层拆掉 `dataset_schedule` 尾段，改成 `dynamic_timed` finisher。第一次完整 30 seed 只有 strict `27/30`，失败 seeds `1021/1031/1036` 看起来像固定 dwell 不合适；复盘代码后发现根因是 `--tcpplate-prefix-target-state` 这个全局开关越界了：prefix 需要 `tcp_to_target`，但 dynamic finisher 也误用了它，而 finisher 训练时应该吃 `tcp_to_plate`。把 state builder 改成 stage-aware 后，prefix 阶段用 target-relative，finisher 阶段用 plate-relative；三个失败 seed 复测 `3/3`，旧版同一环境连续跑完整策略采样 seeds（环境实际固定为 seed 0） `1010-1039` 回到 strict `30/30`、legacy `30/30`。但继续 trace 又发现，连续评估时前一个 episode 的 MuJoCo qvel / ctrl / free-joint 动态残留可能污染下一个 seed，seed `1035` 曾出现采样范围外的初始位置。于是 evaluator 增加 `--fresh-env-per-episode` 和 `--hard-reset-sim-data`。最终用 `--hard-reset-sim-data` clean protocol 重跑完整策略采样 seeds（环境实际固定为 seed 0） `1010-1039`，仍是 strict `30/30`、legacy `30/30`，红杯 `18/18`、蓝杯 `12/12`；mean `xy=0.0219 m`，max `xy=0.0450 m`，最小 lift `0.1093 m`，最小 upright cos `0.9504`。之前 seed `1036` 的 `0.0993 m` 更像 reset 残留造成的评估伪影，不再作为边界样本处理。
        """
    ),
    code(
        r'''
rows = [
    ("raw full20 open-loop", "1/20 final，3/20 ever", "raw policy 仍未过关"),
    ("raw closed-loop strict", "0/20", "最接近部署口径"),
    ("template-tail full20", "4/20", "固定尾段只能救部分样本"),
    ("旧 19D finisher handoff", "prefix 120/180/240/300 均为 0/2", "后段目标仍偏在杯子侧"),
    ("22D tcp_to_plate finisher", "strict 5/10，legacy 6/10", "plate-relative state 是有效线索，但仍是 scaffold"),
    ("22D + phase-scripted gripper", "strict 7/10，legacy 7/10", "夹爪时序能救一批失败，剩余主要是红杯落点"),
    ("固定环境 30 个 policy seeds，schedule 0..9", "strict 21/30，legacy 23/30", "失败全部来自短 move_preplace 模板"),
    ("固定环境 30 个 policy seeds，强制长 schedule", "strict 30/30，legacy 30/30", "仅证明固定场景 scaffold 稳定"),
    ("长 schedule + phase-only learned gripper head", "strict 30/30，legacy 30/30", "第一块脚手架被可学习 head 替代"),
    ("schedule 0..9 + adaptive move gate + learned gripper", "strict 30/30，legacy 30/30", "第二块脚手架工程版：不再强制长 episode0"),
    ("schedule 0..9 + xy-step transition head + learned gripper", "strict 30/30，legacy 30/30", "第二块脚手架可学习版：29/30 主动触发"),
    ("policy prefix + contact transition heads + floor guard", "strict 30/30，legacy 30/30", "第三层去脚手架：核心接触切换由 head 触发，但仍保留 scaffold"),
    ("stage-aware dynamic_timed finisher + hard-reset clean protocol", "strict 30/30，legacy 30/30", "第四层去脚手架：prefix 用 target，finisher 用 plate；mean xy=0.0219 m，max xy=0.0450 m"),
]
md_table(["口径", "结果", "怎么理解"], rows)
'''
    ),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"

# 1. 把 19D phase-state 数据转换成 22D tcp_to_plate 数据。
python code/pi0/build_lerobot_state_phase_tcpplate.py \\
  --src-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_eef_abs_10eps_v1 \\
  --src-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_eef_abs_10eps_v1 \\
  --dst-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dst-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --overwrite

# 2. 评估 22D finisher。注意：这个命令需要已有 22D checkpoint。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval/summary.json"

# 3. 更大样本量复核时，可以强制使用长 move_preplace 模板。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-force-schedule-episode 0 \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_long_schedule/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_long_schedule/summary.json"

# 4. 训练 phase-only gripper head。这个 head 只替代 gripper 规则，不改 EEF/arm。
python code/pi0/train_tcpplate_gripper_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --feature-mode phase \\
  --label-source action \\
  --output "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg_summary.json"

# 5. 用 learned gripper head 复核 30 个策略采样 seed（环境实际固定为 seed 0）。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-force-schedule-episode 0 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_gripper_head/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_gripper_head/summary.json"

# 6. 去掉强制长 episode0，改用 adaptive move_preplace gate。
# 这个版本使用 schedule 0..9 正常重复；如果 live TCP-to-plate 还太远，
# 就 hold move_preplace，直到 xy 足够小或达到保守 max-step 兜底。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-adaptive-move-preplace \\
  --tcpplate-adaptive-move-preplace-xy-threshold 0.05 \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_adaptive_gate/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_adaptive_gate/summary.json"

# 7. 训练 xy-step transition head。full-state transition head 训练集更准，
# 但会被高度/方向相关性误导；稳定版只用 tcp_to_plate_xy + local_step_norm。
python code/pi0/train_tcpplate_transition_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --feature-mode xy_step \\
  --step-scale 180 \\
  --output "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg_summary.json"

# 8. 用 learned transition head 替代固定阈值 gate，schedule 0..9 正常重复。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --tcpplate-transition-head-threshold 0.5 \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_transition_head/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_transition_head/summary.json"

# 9. 训练 target-relative contact transition heads。
# naive all-head 会学到“阶段快结束了”，不等于“现在可以安全闭爪”；
# 这里用 pregrasp geometry label + grasp/floor geometry features。
python code/pi0/train_tcptarget_contact_transition_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --task-set all \\
  --early-label-mode pregrasp_geometry \\
  --feature-mode grasp_geom_count_step \\
  --step-scale 180 \\
  --output-npz "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step_summary.json"

# 10. 用 contact transition heads 替代前段 pregrasp/descend/close 的部分手写切换。
# descend_floor_guard 仍保留，防止 TCP 还明显高于抓取 floor 时提前闭爪。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-prefix-target-state \\
  --tcpplate-contact-primitive guided_contact_lift \\
  --tcpplate-contact-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --tcpplate-contact-transition-head-threshold 0.5 \\
  --tcpplate-contact-pregrasp-hold 999 \\
  --tcpplate-contact-descend-hold 999 \\
  --tcpplate-contact-transition-head-strict \\
  --tcpplate-contact-descend-floor-guard \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --tcpplate-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --prefix-source policy \\
  --prefix-policy-path "$MODEL_ROOT/pi0_prefix_tcptarget/checkpoints/001000/pretrained_model" \\
  --prefix-dataset-repo-id datawhale_eai_pnp_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-dataset-root ./demo_data_pi0_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_contact_transition_head_unseen30/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_contact_transition_head_unseen30/summary.json"

# 11. 拆掉 dataset_schedule 尾段，改用 stage-aware dynamic_timed finisher。
# 关键点：--tcpplate-prefix-target-state 只让 prefix 使用 tcp_to_target；
# stage-aware evaluator 会让 finisher 回到 tcp_to_plate。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-prefix-target-state \\
  --tcpplate-contact-primitive guided_contact_lift \\
  --tcpplate-contact-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --tcpplate-contact-transition-head-threshold 0.5 \\
  --tcpplate-contact-pregrasp-hold 999 \\
  --tcpplate-contact-descend-hold 999 \\
  --tcpplate-contact-transition-head-strict \\
  --tcpplate-contact-descend-floor-guard \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source policy \\
  --prefix-policy-path "$MODEL_ROOT/pi0_prefix_tcptarget/checkpoints/001000/pretrained_model" \\
  --prefix-dataset-repo-id datawhale_eai_pnp_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-dataset-root ./demo_data_pi0_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-steps 180 \\
  --finisher-phase-state dynamic_timed \\
  --finisher-start-phase move_preplace \\
  --timed-phase-dwell-spec move_preplace:260,lower_to_plate:40,retreat:40 \\
  --hard-reset-sim-data \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_dynamic_timed_stageaware_unseen30/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_dynamic_timed_stageaware_unseen30/summary.json"
""")
'''
    ),
    md("如果要跑夹爪脚本化对照，在评估命令里额外加入 `--finisher-scripted-gripper`。这一项只改 gripper，不改 EEF/arm，用来判断开闭爪时序是不是主要瓶颈。`--tcpplate-force-schedule-episode 0` 用于复核长 `move_preplace` 模板是否能消掉提前释放问题。`--tcpplate-gripper-head-path` 会在 `--finisher-scripted-gripper` 的钩子里加载 learned head，把手写 phase 规则替换成可训练的小模型。`--tcpplate-adaptive-move-preplace` 把强制长模板换成 live progress gate；`--tcpplate-transition-head-path` 则把固定阈值 gate 换成 logistic transition head；`--tcpplate-contact-transition-head-path` 进一步替代前段 contact primitive 的部分 phase 切换。`dynamic_timed` 版本要特别注意 stage-aware state：prefix 可以用 `tcp_to_target`，finisher 必须用 `tcp_to_plate`。批量评估建议加 `--hard-reset-sim-data`，先清掉跨 episode 的 MuJoCo 动力学残留；`--fresh-env-per-episode` 更干净但反复创建环境时可能遇到资产 provider 偶发报错。当前 stage-aware dynamic finisher 在 hard-reset clean protocol 的 30 个策略采样 seed（环境实际固定为 seed 0） 上是 strict `30/30`，但仍保留 target-relative contact scaffold、floor guard、learned gripper head 和后段 finisher。"),
    md("如果旧 checkpoint 的 state normalizer 是 19 维，而新数据是 22 维，会出现 `mean/std` shape mismatch。诊断实验里可以复制 checkpoint 并删除 `normalize_inputs.buffer_observation_state.mean/std`，让 22D 数据集统计重新初始化；不要把这个临时处理误写成模型结构创新。"),
]


NOTEBOOKS["06_rocm_debug_playbook.ipynb"] = [
    md(
        """
        # 06 ROCm 调试复盘与排障案例

        这一节把复刻中的折腾整理成排障工作簿。不需要记住所有命令，关键是把一个失败现象拆成证据、根因、修复和验证。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：排障记录模板"),
    code(
        r'''
template = """
### 问题名

- 现象：
- 复现命令：
- 关键证据：
- 排除项：
- 根因：
- 修复：
- 修复后验证：
- 学习结论：
"""
print(template)
'''
    ),
    md("## Checkpoint 2：九个典型案例索引"),
    code(
        r'''
rows = [
    ("旧 success 误判", "必须引入 physical_success"),
    ("示教数据 gripper 标签问题", "先审计数据，再怀疑模型"),
    ("ACT loss 下降但闭环失败", "open-loop 与 closed-loop 分开看"),
    ("DAgger 数据混入后退化", "新增数据要说明补的是哪个状态分布"),
    ("SmolVLA 红蓝杯偏置", "按 instruction 拆开评估"),
    ("复制数据伤害另一类任务", "优先试 sampler/loss 层加权"),
    ("视频与 batch 统计不一致", "视频解释行为，成功率看 batch JSONL"),
    ("ROCm 长训练资源问题", "脚本化、分批评估、检查进程和温度"),
    ("pi0 训练前门控", "权限和 smoke 通过后再谈长训练"),
    ("pi0 reset 残留", "同一环境连续评估前先清掉 MuJoCo 动力学状态"),
]
md_table(["案例", "学习结论"], rows)
'''
    ),
]


NOTEBOOKS["07_data_collection_and_audit.ipynb"] = [
    md(
        """
        # 07 从零采集 MuJoCo LeRobot 示教数据

        这一节把 `5.language_env.ipynb` 的交互采集流程整理成可配置、可复核的版本。最终产物是一个包含双相机图像、6 维关节状态、7 维动作和红/蓝杯语言指令的 LeRobot 数据集。

        采集不是纯 GPU 任务，NVIDIA 或 AMD 设备都可以完成。真正需要保持一致的是 MuJoCo 场景、LeRobot 数据格式、控制频率、state/action 定义和成功判定。远端 Jupyter 无法稳定接收 MuJoCo viewer 键盘事件时，可以在有桌面的机器采集，再把数据目录同步到 AMD 训练机。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    code(TRAINING_HELPERS),
    md("## Checkpoint 1：设置采集目录与安全开关"),
    code(
        r'''
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language_local")
COLLECTION_ROOT = Path(
    os.environ.get("COLLECTION_ROOT", DATA_ROOT / "omy_pnp_language")
).expanduser()
NUM_DEMOS = int(os.environ.get("NUM_DEMOS", "20"))
SEED_START = int(os.environ.get("SEED_START", "0"))
INSTRUCTIONS = [
    "Place the red mug on the plate.",
    "Place the blue mug on the plate.",
]
RUN_INTERACTIVE_COLLECTION = False
OVERWRITE_DATASET = False
RECORD_FOUR_VIEW_VIDEO = True
COLLECTION_VIDEO_DIR = Path(
    os.environ.get("COLLECTION_VIDEO_DIR", OUTPUT_ROOT / "collection_four_view")
).expanduser()

print("dataset repo_id =", DATASET_REPO_ID)
print("collection root =", COLLECTION_ROOT)
print("num demos =", NUM_DEMOS)
print("instructions =", INSTRUCTIONS)
print("four-view videos =", COLLECTION_VIDEO_DIR)
'''
    ),
    md(
        "`COLLECTION_ROOT` 应放在大容量数据盘。默认不开启采集，也不会删除已有目录；只有明确把 `OVERWRITE_DATASET=True` 后才允许覆盖。20 条可以跑通语言条件小实验，想做位置泛化时建议采 30–50 条高质量轨迹，并让红杯、蓝杯和初始位置都得到覆盖。"
    ),
    md("## Checkpoint 2：检查项目、显示和数据 schema"),
    code(
        r'''
require_project_layout()
print("MuJoCo scene =", PROJECT_ROOT / "asset" / "example_scene_y2.xml")
print("DISPLAY =", os.environ.get("DISPLAY"))
if not os.environ.get("DISPLAY"):
    print("当前没有 DISPLAY。交互采集需要本地桌面、远程桌面或可用的 X11 会话。")

FEATURES = {
    "observation.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"],
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (9,),
        "names": [
            "red_x", "red_y", "red_z",
            "blue_x", "blue_y", "blue_z",
            "plate_x", "plate_y", "plate_z",
        ],
    },
}
md_table(
    ["字段", "shape", "作用"],
    [
        ("observation.image", "256x256x3", "固定相机"),
        ("observation.wrist_image", "256x256x3", "腕部相机"),
        ("observation.state", "6", "当前 6 关节状态"),
        ("action", "7", "下一步 6 关节目标 + 夹爪"),
        ("obj_init", "9", "三件物体初始 xyz，仅用于回放/审计"),
    ],
)
'''
    ),
    md("## Checkpoint 3：定义严格成功判定"),
    code(
        r'''
import numpy as np


def collection_snapshot(env, initial_target_z, initial_plate_pos, max_target_lift, max_lifted_run):
    target_pos = np.asarray(env.env.get_p_body(env.obj_target), dtype=np.float64)
    plate_pos = np.asarray(env.env.get_p_body("body_obj_plate_11"), dtype=np.float64)
    target_R = np.asarray(env.env.get_R_body(env.obj_target), dtype=np.float64)
    legacy_success = bool(env.check_success())
    upright_cos = float(target_R[2, 2])
    z_gap = abs(float(target_pos[2] - plate_pos[2]))
    plate_xy_displacement = float(
        np.linalg.norm(plate_pos[:2] - np.asarray(initial_plate_pos)[:2])
    )
    placement_candidate = bool(
        legacy_success
        and max_target_lift >= 0.03
        and max_lifted_run >= 3
        and upright_cos >= 0.7
        and z_gap < 0.08
        and plate_xy_displacement < 0.05
    )
    return {
        "legacy_success": legacy_success,
        "placement_candidate": placement_candidate,
        "xy_dist": float(np.linalg.norm(target_pos[:2] - plate_pos[:2])),
        "plate_z_gap": z_gap,
        "plate_xy_displacement": plate_xy_displacement,
        "max_target_lift": float(max_target_lift),
        "max_lifted_run": int(max_lifted_run),
        "upright_cos": upright_cos,
    }
'''
    ),
    md(
        "旧 `check_success()` 只看终态几何关系，杯子被推到盘子附近也可能触发。这里额外要求杯子至少抬升 3 cm、连续保持 3 个控制步、终态保持直立，并且杯子高度与盘子相符。只有 `physical_success=True` 才写入 episode。"
    ),
    md("## Checkpoint 4：创建或加载 LeRobot 数据集"),
    code(
        r'''
def create_or_load_dataset():
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    COLLECTION_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if COLLECTION_ROOT.exists() and OVERWRITE_DATASET:
        shutil.rmtree(COLLECTION_ROOT)
    if COLLECTION_ROOT.exists():
        print("继续写入已有数据集：", COLLECTION_ROOT)
        return LeRobotDataset(DATASET_REPO_ID, root=COLLECTION_ROOT)
    print("创建新数据集：", COLLECTION_ROOT)
    return LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        root=COLLECTION_ROOT,
        robot_type="omy",
        fps=20,
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=0,
    )
'''
    ),
    md("## Checkpoint 5：键盘采集完整循环"),
    code(
        r'''
def collect_demonstrations():
    import random
    import numpy as np
    from PIL import Image
    from mujoco_env.y_env2 import SimpleEnv2

    dataset = create_or_load_dataset()
    xml_path = PROJECT_ROOT / "asset" / "example_scene_y2.xml"
    COLLECTION_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError:
        cv2 = None
        if RECORD_FOUR_VIEW_VIDEO:
            print("未安装 OpenCV，数据仍会采集，但跳过四视角 MP4。")

    video_writer = None
    video_tmp_path = None

    def open_episode_video(episode_id):
        nonlocal video_writer, video_tmp_path
        if not RECORD_FOUR_VIEW_VIDEO or cv2 is None:
            return
        video_tmp_path = COLLECTION_VIDEO_DIR / f"episode_{episode_id:03d}.recording.mp4"
        video_writer = cv2.VideoWriter(
            str(video_tmp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (640, 480),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"无法创建四视角视频：{video_tmp_path}")

    def write_episode_video(env, agent_image, wrist_image):
        if video_writer is None:
            return
        views = [
            ("Agent", agent_image),
            ("Egocentric", wrist_image),
            ("Top", env.env.get_fixed_cam_rgb(cam_name="topview")),
            ("Side", env.env.get_fixed_cam_rgb(cam_name="sideview")),
        ]
        panels = []
        for label, image in views:
            panel = cv2.resize(np.asarray(image), (320, 240), interpolation=cv2.INTER_AREA)
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            cv2.rectangle(panel, (0, 0), (150, 28), (20, 20, 20), thickness=-1)
            cv2.putText(
                panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            panels.append(panel)
        video_writer.write(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))

    def close_episode_video(episode_id, keep):
        nonlocal video_writer, video_tmp_path
        if video_writer is not None:
            video_writer.release()
        if video_tmp_path is not None and video_tmp_path.exists():
            if keep:
                final_path = COLLECTION_VIDEO_DIR / f"episode_{episode_id:03d}_success.mp4"
                video_tmp_path.replace(final_path)
                print("saved four-view video:", final_path)
            else:
                video_tmp_path.unlink()
        video_writer = None
        video_tmp_path = None

    def reset_episode(env, episode_id):
        seed = SEED_START + episode_id
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)
        instruction = INSTRUCTIONS[episode_id % len(INSTRUCTIONS)]
        env.set_instruction(instruction)
        initial_z = float(env.env.get_p_body(env.obj_target)[2])
        initial_plate_pos = np.asarray(
            env.env.get_p_body("body_obj_plate_11"), dtype=np.float64
        ).copy()
        print(f"episode={episode_id} seed={seed} task={instruction}")
        return initial_z, initial_plate_pos

    np.random.seed(SEED_START)
    random.seed(SEED_START)
    env = SimpleEnv2(str(xml_path), seed=SEED_START, state_type="joint_angle")
    episode_id = 0
    record_flag = False
    initial_target_z, initial_plate_pos = reset_episode(env, episode_id)
    max_target_lift = 0.0
    lifted_run = 0
    max_lifted_run = 0
    stable_place_steps = 0

    try:
        while env.env.is_viewer_alive() and episode_id < NUM_DEMOS:
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            target_z = float(env.env.get_p_body(env.obj_target)[2])
            lift = target_z - initial_target_z
            max_target_lift = max(max_target_lift, lift)
            lifted_run = lifted_run + 1 if lift >= 0.03 else 0
            max_lifted_run = max(max_lifted_run, lifted_run)
            status = collection_snapshot(
                env,
                initial_target_z,
                initial_plate_pos,
                max_target_lift,
                max_lifted_run,
            )
            stable_place_steps = stable_place_steps + 1 if status["placement_candidate"] else 0
            status["stable_place_steps"] = stable_place_steps
            status["physical_success"] = bool(stable_place_steps >= 5)

            if record_flag and status["physical_success"]:
                dataset.save_episode()
                close_episode_video(episode_id, keep=True)
                print("saved:", json.dumps(status, ensure_ascii=False))
                episode_id += 1
                record_flag = False
                if episode_id >= NUM_DEMOS:
                    break
                initial_target_z, initial_plate_pos = reset_episode(env, episode_id)
                max_target_lift = 0.0
                lifted_run = 0
                max_lifted_run = 0
                stable_place_steps = 0
                continue

            teleop_delta, reset = env.teleop_robot()
            if reset:
                dataset.clear_episode_buffer()
                close_episode_video(episode_id, keep=False)
                record_flag = False
                initial_target_z, initial_plate_pos = reset_episode(env, episode_id)
                max_target_lift = 0.0
                lifted_run = 0
                max_lifted_run = 0
                stable_place_steps = 0
                continue

            if not record_flag and np.any(np.abs(teleop_delta) > 1e-8):
                record_flag = True
                open_episode_video(episode_id)
                print("Start recording")

            agent_image, wrist_image = env.grab_image()
            if record_flag:
                write_episode_video(env, agent_image, wrist_image)
            state = env.get_joint_state()[:6].astype(np.float32)
            env.step(teleop_delta)
            target_action = env.q[:7].astype(np.float32)

            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": np.asarray(Image.fromarray(agent_image).resize((256, 256))),
                        "observation.wrist_image": np.asarray(Image.fromarray(wrist_image).resize((256, 256))),
                        "observation.state": state,
                        "action": target_action,
                        "obj_init": np.asarray(env.obj_init_pose, dtype=np.float32),
                    },
                    task=env.instruction,
                )
            env.render(teleop=True, idx=episode_id)
    finally:
        close_episode_video(episode_id, keep=False)
        env.env.close_viewer()
        shutil.rmtree(dataset.root / "images", ignore_errors=True)
    print(f"采集完成：{episode_id}/{NUM_DEMOS}")
    return dataset


if RUN_INTERACTIVE_COLLECTION:
    dataset = collect_demonstrations()
else:
    print("采集默认关闭。确认 DISPLAY、路径和覆盖开关后，将 RUN_INTERACTIVE_COLLECTION 改为 True。")
'''
    ),
    md(
        "键位与上游教程一致：`W/A/S/D` 控制平面移动，`R/F` 控制高度，`Q/E` 与方向键控制姿态，空格切换夹爪，`Z` 丢弃当前失败回合。每次成功保存后会重新关闭记录开关，避免把 reset 过程混进下一条 episode。模型训练仍只使用固定相机和腕部相机；Top/Side 视角写入独立 MP4，专门用于人工复核，不偷偷扩充模型输入。"
    ),
    md("## Checkpoint 6：采集后审计"),
    code(
        r'''
info_path = COLLECTION_ROOT / "meta" / "info.json"
if info_path.exists():
    info, tasks = dataset_report(COLLECTION_ROOT)
    episodes_path = COLLECTION_ROOT / "meta" / "episodes.jsonl"
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_counts = {}
    for episode in episodes:
        task = episode.get("tasks", [""])[0]
        task_counts[task] = task_counts.get(task, 0) + 1
    md_table(["指令", "episode 数"], sorted(task_counts.items()))
else:
    print("还没有可审计的数据：", COLLECTION_ROOT)
'''
    ),
    md(
        "数据表通过后，再打开上游 `6.visualize_data.ipynb` 随机回放若干 episode。至少检查图像、state/action shape、夹爪开闭时序、是否真的抬起，以及红蓝杯指令是否平衡。完成这些检查后再进入 08–10 的训练 Notebook。"
    ),
]


NOTEBOOKS["07_data_collection_and_audit.ipynb"].extend(
    [
        md("## Checkpoint 7：查看已经跑过的数据集实测结果"),
        code(
            r'''
snapshot = json.loads(
    (ASSET_DIR / "collection_dataset_snapshot.json").read_text(encoding="utf-8")
)
rows = [
    ("repo_id", snapshot["dataset_repo_id"]),
    ("episodes", snapshot["total_episodes"]),
    ("frames", snapshot["total_frames"]),
    ("fps", snapshot["fps"]),
    ("state / action", f'{snapshot["state_shape"]} / {snapshot["action_shape"]}'),
    ("模型输入相机", " / ".join(snapshot["stored_cameras"])),
    ("复核视频视角", " / ".join(snapshot["recorder_views"])),
]
md_table(["数据项", "AMD 实测值"], rows)
print("平均每条轨迹帧数 =", round(snapshot["total_frames"] / snapshot["total_episodes"], 1))
'''
        ),
        md("## Checkpoint 8：四视角采集/回放样例"),
        md(
            """
            下面的视频是在 AMD 设备上用与键盘采集相同的四视角 recorder 实际录制的严格成功策略回放。它用于先检查画面布局、接触、抬升、搬运和释放是否清楚；它不是人工键盘示教。真正采集时，将 `RUN_INTERACTIVE_COLLECTION=True`，每条成功示教会在 `COLLECTION_VIDEO_DIR` 生成同样布局的 MP4。

            ![MuJoCo 四视角严格成功序列](../assets/pnp_four_view_strict_success_sequence.jpg)

            <video controls width="960" src="../assets/pnp_four_view_strict_success.mp4"></video>
            """
        ),
        md("## Checkpoint 9：确认环境 seed 真的改变物体位置"),
        code(
            r'''
source_path = PROJECT_ROOT / "mujoco_env" / "y_env2.py"
source = source_path.read_text(encoding="utf-8")
bad_pattern = "np.random.seed(seed=0)"
good_pattern = "np.random.seed(seed=seed)"
print("seed source =", source_path)
print("发现旧的固定 seed 写法：", bad_pattern in source)
print("发现修正后的 seed 写法：", good_pattern in source)
if bad_pattern in source:
    raise RuntimeError(
        "当前 y_env2.py 会把所有环境固定成 seed 0。先修复它，再采集位置随机化数据。"
    )
'''
        ),
        md(
            "这个检查很重要。旧实现曾把任意传入 seed 都改成 `0`，导致多 seed 评估只改变了策略采样，杯子和盘子位置并没有真正变化。修复后要重新采 30–50 条覆盖不同位置的轨迹；旧固定场景数据仍可用于链路 smoke，但不能作为空间泛化训练集。"
        ),
    ]
)


make_training_notebook(
    filename="08_act_training_rocm.ipynb",
    number="08",
    policy_type="act",
    title="ACT 从 smoke 到正式训练",
    summary="ACT 是最适合先跑通完整闭环的基线。它不依赖语言主干，模型更小，能较快暴露数据、state/action 对齐和夹爪标签问题。",
    full_steps=5000,
    batch_size=16,
    chunk_size=10,
    n_action_steps=10,
    model_note="ACT 示例先以 5,000 步作为基线。单条轨迹很容易过拟合，位置泛化实验应使用多条高质量轨迹，并固定 held-out seeds。训练时同时看 loss、动作 MAE 和显存，但最终以 closed-loop physical_success 为准。",
)


make_training_notebook(
    filename="09_smolvla_training_rocm.ipynb",
    number="09",
    policy_type="smolvla",
    title="SmolVLA 从 smoke 到正式训练",
    summary="SmolVLA 同时读取图像、语言和机器人状态。这里使用与 ACT 相同的数据根目录，但必须检查红杯、蓝杯指令是否都存在，不能只看总体成功率。",
    full_steps=20000,
    batch_size=4,
    chunk_size=5,
    n_action_steps=5,
    model_note="SmolVLA 首次运行会下载基础权重。正式训练建议保存中间 checkpoint，并分别统计红杯和蓝杯的 physical_success；如果一类明显落后，先检查任务分布和 gripper 事件，再考虑 Weighted sampler。",
)


make_training_notebook(
    filename="10_pi0_training_rocm.ipynb",
    number="10",
    policy_type="pi0",
    title="pi_0 从权限门控到正式训练",
    summary="pi_0 的模型加载、gated 权限和显存压力都更高。这里先把权限、数据、模型构造和 2-step 反向传播逐项过门，再启动正式训练。",
    full_steps=20000,
    batch_size=4,
    chunk_size=5,
    n_action_steps=5,
    model_note="pi_0 的训练 loss 下降并不保证闭环抓取成功。小数据接触任务尤其容易出现 gripper 时序和误差累积问题。raw policy、learned auxiliary head 和带脚手架的 hybrid 结果必须分开报告。",
)


for notebook_name, model_name in [
    ("08_act_training_rocm.ipynb", "ACT"),
    ("09_smolvla_training_rocm.ipynb", "SmolVLA"),
    ("10_pi0_training_rocm.ipynb", "pi0"),
]:
    NOTEBOOKS[notebook_name].extend(
        [
            md("## 已完成训练的日志快照"),
            code(
                f'''
progress = json.loads(
    (ASSET_DIR / "training_progress_snapshot.json").read_text(encoding="utf-8")
)
records = [row for row in progress["records"] if row["model"] == {model_name!r}]
for row in records:
    ratio = row["current_step"] / max(1, row["total_steps"])
    filled = round(36 * ratio)
    bar = "#" * filled + "." * (36 - filled)
    print(f'{{row["model"]:>18}} [{{bar}}] {{row["current_step"]}}/{{row["total_steps"]}}')
    print("stage:", row["stage"])
    print("closed-loop physical_success:", row["physical_success"])
    print("note:", row["note"])
'''
            ),
            md(
                """
                ![AMD ROCm 历史训练进度与闭环结果](../assets/training_progress_overview.png)

                这张图来自已经完成的 AMD 实验日志，不是假装本次打开 Notebook 后重新跑完的训练。上面的 smoke/full 单元提供真实启动代码；运行长训练时，终端进度会继续写入输出目录，完成后再用 11 的闭环评估决定 checkpoint 是否可用。
                """
            ),
            md("## 在 Notebook 中跟踪后台训练日志"),
            code(
                r'''
def tail_training_log(log_path, lines=25):
    log_path = Path(log_path)
    if not log_path.exists():
        print("训练日志尚不存在：", log_path)
        return
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(content[-lines:]))


TRAIN_LOG = Path(os.environ.get("TRAIN_LOG", FULL_OUTPUT / "train.log"))
print("训练运行后可重复执行：tail_training_log(TRAIN_LOG)")
tail_training_log(TRAIN_LOG)
'''
            ),
        ]
    )


NOTEBOOKS["11_mujoco_closed_loop_deploy.ipynb"] = [
    md(
        """
        # 11 把 checkpoint 部署到 MuJoCo 闭环

        这一节把 ACT、SmolVLA 或 pi_0 checkpoint 放回 `SimpleEnv2`，让策略按 20 Hz 读取双相机图像和 6 维关节状态，并输出 7 维动作。评估会保存视频和 JSONL，同时报告旧几何成功与严格 `physical_success`。

        这是仿真闭环推理，不是把模型导出成服务。模型每次动作后都会重新读取环境观测；如果只在数据集上预测 GT 动作，那属于 open-loop replay，不能替代这里的成功率。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md(
        """
        ## 零训练成功预览：先看正确行为

        这一格不加载模型、不启动 MuJoCo，也不会消耗训练额度。它直接显示仓库内置的四视角严格成功回放，先确认接近、夹取、抬升、搬运、释放和稳定放置的正确顺序。视频来自固定环境上的 `pi0 + visual/history learned head`，不是 raw pi0，也不能代表随机位置泛化。
        """
    ),
    code(
        r'''
SUCCESS_PREVIEW = ASSET_DIR / "pnp_four_view_strict_success.mp4"
if not SUCCESS_PREVIEW.exists():
    raise FileNotFoundError(f"缺少成功预览视频：{SUCCESS_PREVIEW}")

try:
    from IPython.display import Video
except ImportError:
    print("当前是纯 Python 执行器；在 Jupyter 中运行本单元格即可嵌入播放视频。")
else:
    display(Video(filename=str(SUCCESS_PREVIEW), embed=True, html_attributes="controls muted"))

print("严格成功预览：", SUCCESS_PREVIEW)
'''
    ),
    code(TRAINING_HELPERS),
    code(ROLLOUT_HELPERS),
    md("## Checkpoint 1：选择模型、数据统计和评估 seed"),
    code(
        r'''
POLICY_TYPE = os.environ.get("POLICY_TYPE", "act")  # act / smolvla / pi0
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language")
EVAL_DATA_ROOT = Path(
    os.environ.get("EVAL_DATA_ROOT", DATA_ROOT / "omy_pnp_language")
).expanduser()
MODEL_RUN_DIR = Path(
    os.environ.get("MODEL_RUN_DIR", MODEL_ROOT / f"{POLICY_TYPE}_rocm_full")
).expanduser()
TASK_TEXT = os.environ.get("TASK_TEXT", "Place the blue mug on the plate.")
EVAL_SEEDS = [int(value) for value in os.environ.get("EVAL_SEEDS", "1000,1001,1002,1003").split(",")]
MAX_ACTION_STEPS = int(os.environ.get("MAX_ACTION_STEPS", "300"))
RUN_CLOSED_LOOP = False

print("policy type =", POLICY_TYPE)
print("model run =", MODEL_RUN_DIR)
print("dataset =", EVAL_DATA_ROOT)
print("task =", TASK_TEXT)
print("seeds =", EVAL_SEEDS)
'''
    ),
    md(
        "先用 4 个固定 seed 做 smoke，但 4 条不具有统计代表性。模型通过小面板后，再扩到至少 20–30 个 held-out seeds，并把红杯、蓝杯分别统计。评估时不要读取 target/plate 坐标、数据集 phase 或 GT 动作作为策略输入。"
    ),
    md("## Checkpoint 2：检查数据与 checkpoint"),
    code(
        r'''
require_project_layout()
if (EVAL_DATA_ROOT / "meta" / "info.json").exists():
    dataset_report(EVAL_DATA_ROOT)
else:
    print("缺少评估数据统计：", EVAL_DATA_ROOT)

try:
    POLICY_PATH = Path(os.environ.get("POLICY_PATH", ""))
    if not str(POLICY_PATH) or str(POLICY_PATH) == ".":
        POLICY_PATH = find_pretrained_model(MODEL_RUN_DIR)
    print("policy path =", POLICY_PATH)
except FileNotFoundError as exc:
    POLICY_PATH = None
    print(exc)
'''
    ),
    md("## Checkpoint 3：加载策略并执行闭环"),
    code(
        r'''
if RUN_CLOSED_LOOP:
    if POLICY_PATH is None:
        raise FileNotFoundError("请先完成训练，或通过 POLICY_PATH 指定 pretrained_model。")
    if POLICY_TYPE == "pi0" and not os.environ.get("HF_TOKEN"):
        print("本地 checkpoint 通常可离线加载；若仍需读取 gated 配置，请先私密设置 HF_TOKEN。")
    show_rocm_resources()
    policy = load_policy(
        policy_type=POLICY_TYPE,
        policy_path=POLICY_PATH,
        dataset_repo_id=DATASET_REPO_ID,
        dataset_root=EVAL_DATA_ROOT,
        device="cuda",
    )
    results = run_closed_loop(
        policy=policy,
        policy_type=POLICY_TYPE,
        instruction=TASK_TEXT,
        seeds=EVAL_SEEDS,
        output_dir=OUTPUT_ROOT / f"{POLICY_TYPE}_closed_loop",
        device="cuda",
        max_action_steps=MAX_ACTION_STEPS,
        render=True,
    )
else:
    print("闭环评估默认关闭。确认 DISPLAY、checkpoint、数据统计和 seed 后，将 RUN_CLOSED_LOOP 改为 True。")
'''
    ),
    md("## Checkpoint 4：读取成功率和失败类型"),
    code(
        r'''
result_path = OUTPUT_ROOT / f"{POLICY_TYPE}_closed_loop" / "results.jsonl"
if result_path.exists():
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    physical = sum(row.get("physical_success", False) for row in rows)
    legacy = sum(row.get("legacy_success", False) for row in rows)
    md_table(
        ["口径", "结果"],
        [
            ("legacy_success", f"{legacy}/{len(rows)}"),
            ("physical_success", f"{physical}/{len(rows)}"),
        ],
    )
    for row in rows:
        if not row.get("physical_success"):
            print(
                "失败 seed",
                row["seed"],
                "xy=", round(row.get("xy_dist", float("nan")), 4),
                "lift=", round(row.get("max_target_lift", float("nan")), 4),
                "upright=", round(row.get("upright_cos", float("nan")), 4),
            )
else:
    print("尚未生成闭环结果：", result_path)
'''
    ),
    md(
        "如果 `legacy_success > physical_success`，优先查看失败视频，通常是推杯、未真正夹住、运输中掉落或只满足终态几何条件。下一步回到 02–06 的诊断 Notebook，把失败定位到 approach、contact、lift、transport、release 中的具体阶段。"
    ),
]


NOTEBOOKS["11_mujoco_closed_loop_deploy.ipynb"].extend(
    [
        md("## Checkpoint 5：严格判定与四视角实测结果"),
        code(
            r'''
strict_results = json.loads(
    (ASSET_DIR / "pi0_strict_input_results.json").read_text(encoding="utf-8")
)
predicate = strict_results["strict_predicate"]
md_table(
    ["严格条件", "阈值"],
    [
        ("真实抬升", f'>= {predicate["min_lift_m"]:.2f} m，连续 {predicate["min_lift_steps"]} 步'),
        ("终态直立", f'>= {predicate["min_upright_cos"]:.1f}'),
        ("杯盘高度差", f'<= {predicate["max_plate_z_gap_m"]:.2f} m'),
        ("盘子位移", f'<= {predicate["max_plate_xy_displacement_m"]:.2f} m'),
        ("稳定放置", f'连续 {predicate["stable_place_steps"]} 步'),
    ],
)
'''
        ),
        md(
            """
            ![pi0 strict-input 闭环结果](../assets/pi0_strict_input_progress.png)

            <video controls width="960" src="../assets/pnp_four_view_strict_success.mp4"></video>

            视频对应环境 seed 0、策略采样 seed 3 的严格成功回放。注意它属于 `pi0 + visual/history learned head`，不是 raw pi0；模型输入只有双相机图像、语言、robot proprio 和历史执行动作，没有读取杯子坐标、盘子坐标、GT phase 或 oracle action。
            """
        ),
        md(
            "随机环境 gate 只有 `1/4`，所以本章没有把固定场景 `6/12` 写成空间泛化成功。下一步应先用修正后的环境 seed 重新采集多位置示教，再扩大到至少 20–30 个真正随机环境评估。"
        ),
    ]
)


NOTEBOOKS["12_pi0_strict_input_end_to_end.ipynb"] = [
    md(
        """
        # 12 pi0 strict-input 端到端诊断

        这一节把 pi0 调试中最容易混淆的四件事拆开：raw policy、只用 VLA 合法输入的 learned head、固定场景策略采样稳定性、真正随机物体位置泛化。所有表格都来自 AMD 设备实测结果。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：读取最终严格评估结果"),
    code(
        r'''
results = json.loads(
    (ASSET_DIR / "pi0_strict_input_results.json").read_text(encoding="utf-8")
)
fixed = results["fixed_scene"]
randomized = results["randomized_environment_gate"]
rows = [
    (
        "raw pi0",
        "env seed 0；12 个 policy sampling seeds",
        f'{fixed["raw_pi0"]["success"]}/{fixed["raw_pi0"]["total"]}',
    ),
    (
        "pi0 + visual/history GRU head",
        "env seed 0；同一批 12 个 policy sampling seeds",
        f'{fixed["pi0_visual_history_gru"]["success"]}/{fixed["pi0_visual_history_gru"]["total"]}',
    ),
    (
        "pi0 + visual/history GRU head",
        "修正 seed 后的随机环境 41–44",
        f'{randomized["pi0_visual_history_gru"]["success"]}/{randomized["pi0_visual_history_gru"]["total"]}',
    ),
]
md_table(["策略", "协议", "final strict"], rows)
'''
    ),
    md(
        """
        ![pi0 strict-input 对照](../assets/pi0_strict_input_progress.png)

        固定场景从 `0/12` 提到 `6/12` 是实质进步，但不能跳过随机环境 `1/4`。这也是为什么教程把“固定场景拟合”和“位置泛化”分开写。
        """
    ),
    md("## Checkpoint 2：learned head 到底看了什么"),
    code(
        r'''
rows = [
    ("允许", "agent/wrist 图像、语言、6D robot proprio、上一条策略自己执行过的 7D EEF/gripper 命令"),
    ("禁止", "target xyz、plate xyz、GT phase、oracle action、当前或未来 GT gripper 标签"),
    ("输出", "EEF keyframe/历史动作与 gripper close probability"),
    ("部署", "gripper probability threshold=0.5，再转成二值执行命令"),
]
md_table(["类别", "内容"], rows)
'''
    ),
    md(
        "这个 head 具备 VLA-compatible 的输入边界，但它是额外训练的视觉/历史控制头，因此报告时写 `pi0 + learned head`，不能写成 raw pi0。它学的是图像与机器人历史中的阶段线索，不依赖鲜艳标记或手工目标坐标。"
    ),
    md("## Checkpoint 3：visual/history head 的实际训练命令"),
    code(
        r'''
import shlex

HEAD_SOURCE_REPO = os.environ.get("HEAD_SOURCE_REPO", "your_multiseed_fulltask_dataset")
HEAD_SOURCE_ROOT = Path(os.environ.get("HEAD_SOURCE_ROOT", DATA_ROOT / HEAD_SOURCE_REPO))
HEAD_SOURCE_JSONL = Path(os.environ.get("HEAD_SOURCE_JSONL", OUTPUT_ROOT / "collection_results.jsonl"))
HEAD_POLICY_PATH = Path(os.environ.get("HEAD_POLICY_PATH", MODEL_ROOT / "pi0_base" / "pretrained_model"))
HEAD_DIR = Path(os.environ.get("HEAD_DIR", MODEL_ROOT / "pi0_visual_history_head"))
FEATURE_CACHE = HEAD_DIR / "visual_features.npz"
DIRECT_HEAD = HEAD_DIR / "visual_keyframe_head.npz"
GRU_HEAD = HEAD_DIR / "visual_keyframe_gru_head.npz"
RUN_AUXILIARY_TRAIN = False

commands = [
    [
        sys.executable,
        str(TOPIC_ROOT / "code" / "pi0" / "train_pi0_visual_contact_head.py"),
        "--source", HEAD_SOURCE_REPO, str(HEAD_SOURCE_ROOT), str(HEAD_SOURCE_JSONL),
        "--policy-path", str(HEAD_POLICY_PATH),
        "--stats-repo-id", HEAD_SOURCE_REPO,
        "--stats-dataset-root", str(HEAD_SOURCE_ROOT),
        "--train-seeds", "21-32",
        "--val-seeds", "33-36",
        "--append-prev-action",
        "--feature-cache", str(FEATURE_CACHE),
        "--output", str(DIRECT_HEAD),
        "--summary-json", str(HEAD_DIR / "visual_keyframe_head_summary.json"),
    ],
    [
        sys.executable,
        str(TOPIC_ROOT / "code" / "pi0" / "train_pi0_visual_gripper_gru.py"),
        "--source", HEAD_SOURCE_REPO, str(HEAD_SOURCE_ROOT), str(HEAD_SOURCE_JSONL),
        "--feature-cache", str(FEATURE_CACHE),
        "--direct-head", str(DIRECT_HEAD),
        "--train-seeds", "21-32",
        "--val-seeds", "33-36",
        "--epochs", "300",
        "--transition-weight", "20",
        "--release-weight", "40",
        "--output", str(GRU_HEAD),
        "--summary-json", str(HEAD_DIR / "visual_keyframe_gru_summary.json"),
    ],
    [
        sys.executable,
        str(TOPIC_ROOT / "code" / "pi0" / "test_pi0_visual_gripper_gru_parity.py"),
        "--head", str(GRU_HEAD),
        "--steps", "32",
    ],
]

for command in commands:
    print("$", shlex.join(command))
if RUN_AUXILIARY_TRAIN:
    HEAD_DIR.mkdir(parents=True, exist_ok=True)
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
else:
    print("默认只预览。先替换多位置数据、collector JSONL 和 base policy 路径，再打开 RUN_AUXILIARY_TRAIN。")
'''
    ),
    md("## Checkpoint 4：语言表述也属于部署协议"),
    code(
        r'''
rows = [
    (
        "Place the blue mug on the plate.",
        "0 active steps",
        "0/1",
        "短 prompt 超出小头训练分布，prototype gate 一直拒绝接管",
    ),
    (
        "Pick up the blue mug and place it on the plate.",
        "269 active steps",
        "1/1",
        "与训练指令一致；seed3 在 270 步通过 final strict",
    ),
]
md_table(["instruction", "learned head", "strict", "现象"], rows)
'''
    ),
    md(
        "这不是说 VLA 永远不能接受同义指令，而是当前 4 条小数据训练出来的 auxiliary head 还没有语言改写泛化。部署时要先固定训练/评估 prompt；下一轮再加入同义指令增强，并单独测试 language paraphrase gate。"
    ),
    md("## Checkpoint 5：为什么 overall accuracy 会骗人"),
    code(
        r'''
rows = [
    ("旧线性 gripper head", "99.66%", "0/4", "release 只有极少帧，总体准确率掩盖了全漏"),
    ("GRU sequence head", "96.8% balanced", "5/5", "按 close/release transition 单独选阈值"),
]
md_table(["模型", "验证指标", "held-out release", "解读"], rows)
'''
    ),
    md("## Checkpoint 6：用代码复现最终成功谓词"),
    code(
        r'''
def final_strict_success(row):
    p = results["strict_predicate"]
    return bool(
        row["legacy_success"]
        and row["max_target_lift"] >= p["min_lift_m"]
        and row["max_lifted_run"] >= p["min_lift_steps"]
        and row["upright_cos"] >= p["min_upright_cos"]
        and row["plate_z_gap"] <= p["max_plate_z_gap_m"]
        and row["plate_xy_displacement"] <= p["max_plate_xy_displacement_m"]
        and row["stable_place_steps"] >= p["stable_place_steps"]
    )


known_success = {
    "legacy_success": True,
    "max_target_lift": 0.11,
    "max_lifted_run": 8,
    "upright_cos": 0.95,
    "plate_z_gap": 0.0448,
    "plate_xy_displacement": 0.0105,
    "stable_place_steps": 5,
}
print("示例是否通过最终严格判定：", final_strict_success(known_success))
'''
    ),
    md("## Checkpoint 7：seed bug 对结论的影响"),
    code(
        r'''
seed_bug = results["seed_bug"]
print("旧代码：", seed_bug["old_code"])
print("修复后：", seed_bug["fixed_code"])
print("影响：", seed_bug["impact"])
'''
    ),
    md(
        """
        ## Checkpoint 8：四视角成功回放

        ![四视角成功回放关键帧](../assets/pnp_four_view_strict_success_sequence.jpg)

        <video controls width="960" src="../assets/pnp_four_view_strict_success.mp4"></video>
        """
    ),
    md("## Checkpoint 9：下一轮数据计划"),
    code(
        r'''
plan = [
    ("采集", "修正 seed 后采 30–50 条多位置完整成功轨迹"),
    ("分层", "按红/蓝杯、起点区域、contact miss、transport tip、release fail 分桶"),
    ("训练", "先保护 fixed-scene 6/12，再训练 visual/history head；不要继续盲加 raw pi0 steps"),
    ("评估", "fixed scene 与 random env 分开；随机环境至少 20–30 条"),
    ("替换门槛", "新模型同协议超过保护基线，且视频无推杯/悬空误判"),
]
md_table(["阶段", "动作"], plan)
'''
    ),
]


NOTEBOOKS["06_rocm_debug_playbook.ipynb"].extend([
    md("## Checkpoint 3：pi0 hard-reset 评估协议案例"),
    md(
        """
        `dynamic_timed` finisher 的一次排障很典型：stage-aware state 修复后，旧版连续环境评估已经能跑到 strict `30/30`，但 seed `1036` 的终态 `xy` 一度贴近阈值。继续 trace 时发现，后一个 seed 的初始杯子位置偶尔跑出该 seed 的采样范围，说明前一个 episode 的 qvel / ctrl / free-joint 动态状态污染了下一个 episode。

        这里不要急着把它归因成模型泛化问题。更干净的做法是先修评估协议：小面板可以用 `--fresh-env-per-episode` 每个 seed 新建环境；完整批量评估推荐 `--hard-reset-sim-data`，在每次 `env.reset(seed)` 前先清掉底层 MuJoCo sim data。用 hard-reset clean protocol 重跑 策略采样 seeds（环境实际固定为 seed 0） `1010-1039` 后，结果仍是 strict `30/30`、legacy `30/30`，mean `xy=0.0219 m`，max `xy=0.0450 m`，seed `1036` 不再贴边。
        """
    ),
    code(
        r'''
rows = [
    ("现象", "旧连续环境评估 30/30，但 seed 1036 一度 max xy=0.0993 m"),
    ("异常证据", "seed 1035 初始杯子位置出现采样范围外的值"),
    ("根因", "env.reset(seed) 重设物体位置，但没有先清掉底层 MuJoCo 动力学残留"),
    ("修复", "新增 --fresh-env-per-episode 和 --hard-reset-sim-data"),
    ("clean 结果", "hard-reset 后 1010-1039 为 strict 30/30，mean xy=0.0219 m，max xy=0.0450 m"),
]
md_table(["项", "记录"], rows)
'''
    ),
    md("## Checkpoint 4：把指标和视频放在一起看"),
    code(
        r'''
show_asset("act_dagger_progress_curve.png", width=900)
show_asset("smolvla_red_blue_success.png", width=1000)
'''
    ),
    md("图表回答“整体表现如何”，视频回答“行为到底像不像成功”。这两类证据最好同时放进实验报告。"),
    md("## Checkpoint 5：结果摘要模板"),
    code(
        r'''
rows = [
    ("设备", "GPU/APU 型号、ROCm、PyTorch、温度和显存"),
    ("数据", "episode 数量、任务类型、是否通过物理回放审计"),
    ("ACT", "best checkpoint、physical_success、主要失败类型"),
    ("SmolVLA", "红杯/蓝杯成功率、采样策略、保护基线"),
    ("pi0", "gated 权限、1-step smoke、正式训练状态"),
    ("视频", "一个真实成功、一个典型失败"),
    ("复盘", "最关键的一个坑和修复证据"),
]
md_table(["项目", "应该写什么"], rows)
'''
    ),
])


def main() -> None:
    for filename, cells in NOTEBOOKS.items():
        write_nb(filename, cells)


if __name__ == "__main__":
    main()
