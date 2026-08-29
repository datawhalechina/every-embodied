#!/usr/bin/env python3
"""Create clean per-model end-to-end notebooks for the AMD ROCm topic."""

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


SETUP = r'''
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import sys

try:
    from IPython.display import HTML, Markdown, display
except Exception:
    class Markdown(str):
        pass

    class HTML(str):
        pass

    def display(obj):
        print(obj)


def find_topic_root():
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "assets" / "metrics_snapshot.json").exists():
            return candidate
    raise RuntimeError("请从 AMD ROCm 专题目录或 notebooks 子目录启动 Jupyter。")


TOPIC_ROOT = find_topic_root()
ASSET_DIR = TOPIC_ROOT / "assets"
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/path/to/04mujoco复现ACT、Pi0、SmolVLA"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/path/to/datasets/every_embodied"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/path/to/model/checkpoints"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", TOPIC_ROOT / "outputs"))

# The AMD teaching workflow should be runnable from local datasets/checkpoints.
# Avoid surprising network calls during class or when AUP/Radeon Cloud cannot
# reach Hugging Face.
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("NOTEBOOK_HF_OFFLINE", "1"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.environ.get("NOTEBOOK_HF_OFFLINE", "1"))
os.environ.setdefault("HF_DATASETS_OFFLINE", os.environ.get("NOTEBOOK_HF_OFFLINE", "1"))
os.environ.setdefault("HF_HOME", str(Path(os.environ.get("CACHE_ROOT", OUTPUT_ROOT / "cache")) / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))

def public_path(path):
    path = Path(path)
    replacements = [
        (TOPIC_ROOT, "$TOPIC_ROOT"),
        (PROJECT_ROOT, "$PROJECT_ROOT"),
        (DATA_ROOT, "$DATA_ROOT"),
        (MODEL_ROOT, "$MODEL_ROOT"),
        (OUTPUT_ROOT, "$OUTPUT_ROOT"),
    ]
    value = str(path)
    for root, label in sorted(replacements, key=lambda item: len(str(item[0])), reverse=True):
        root_value = str(root)
        if root_value and value.startswith(root_value):
            return label + value[len(root_value):]
    return value


print("TOPIC_ROOT = $TOPIC_ROOT")
print("PROJECT_ROOT =", public_path(PROJECT_ROOT))
print("DATA_ROOT =", public_path(DATA_ROOT))
print("MODEL_ROOT =", public_path(MODEL_ROOT))
print("OUTPUT_ROOT =", public_path(OUTPUT_ROOT))
'''


HELPERS = r'''
def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(public_path(x) if isinstance(x, (str, Path)) else str(x) for x in row) + " |")
    display(Markdown("\n".join(lines)))


def show_json(path, max_chars=5000):
    path = Path(path)
    if not path.exists():
        print("文件不存在：", public_path(path))
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(data, ensure_ascii=False, indent=2)
    print(text[:max_chars] + ("\n..." if len(text) > max_chars else ""))
    return data


def show_video(filename, title):
    path = ASSET_DIR / filename
    display(Markdown(f"**{title}**"))
    if path.exists():
        rel = f"../assets/{filename}"
        display(HTML(f"<video controls muted preload='metadata' width='960'><source src='{rel}' type='video/mp4'></video>"))
    else:
        print("缺少视频素材：", public_path(path))


def show_image(filename, title, width=960):
    path = ASSET_DIR / filename
    display(Markdown(f"**{title}**"))
    if path.exists():
        rel = f"../assets/{filename}"
        display(HTML(f"<img src='{rel}' width='{width}'>"))
    else:
        print("缺少图片素材：", public_path(path))


def run_cmd_preview(command, cwd=None):
    shown = [public_path(x) if isinstance(x, (str, Path)) else x for x in command]
    print("$", shlex.join([str(x) for x in shown]))
    if cwd:
        print("cwd =", public_path(cwd))


def tail_log(log_path, lines=40):
    path = Path(log_path)
    if not path.exists():
        print("日志不存在：", public_path(path))
        return
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(content[-lines:]))


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


RUN_SMOKE = env_flag("RUN_SMOKE")
RUN_LONG_TRAIN = env_flag("RUN_LONG_TRAIN")
RUN_EVAL = env_flag("RUN_EVAL")
EVAL_SCRIPT = Path(os.environ.get("EVAL_SCRIPT", PROJECT_ROOT / "eval_policy_success.py"))


_XVFB_PROCESS = None


def ensure_xvfb_display():
    """Start a lightweight virtual display for headless MuJoCo evaluation."""
    global _XVFB_PROCESS
    if os.environ.get("DISPLAY"):
        print("DISPLAY =", os.environ["DISPLAY"])
        return None
    xvfb_bin = shutil.which("Xvfb")
    if not xvfb_bin:
        print("没有发现 Xvfb；如遇 GLFW DISPLAY 报错，请先安装 xvfb。")
        return None
    display_id = os.environ.get("NOTEBOOK_XVFB_DISPLAY", ":99")
    _XVFB_PROCESS = subprocess.Popen(
        [xvfb_bin, display_id, "-screen", "0", "1280x1024x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = display_id
    print("已启动 Notebook 内部 Xvfb：DISPLAY =", display_id)
    return _XVFB_PROCESS


def ensure_project_layout():
    required = [PROJECT_ROOT / "asset" / "example_scene_y2.xml", PROJECT_ROOT / "mujoco_env"]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("当前 PROJECT_ROOT 还不是可运行工程，缺少：")
        for path in missing:
            print(" -", public_path(path))
        print("请先设置 PROJECT_ROOT，再运行训练或评估单元。")
        return False
    return True


def write_json_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    except Exception:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    print("写出配置：", public_path(path))
    return path


def make_lerobot_train_config(policy_type, dataset_repo_id, dataset_root, output_dir, steps, batch_size, chunk_size, n_action_steps, seed=42):
    save_freq = int(os.environ.get(f"{policy_type.upper()}_SAVE_FREQ", os.environ.get("SAVE_FREQ", str(steps))))
    return {
        "dataset": {
            "repo_id": dataset_repo_id,
            "root": str(dataset_root),
            "use_imagenet_stats": True,
        },
        "policy": {
            "type": policy_type,
            "chunk_size": int(chunk_size),
            "n_action_steps": int(n_action_steps),
            "device": "cuda",
        },
        "output_dir": str(output_dir),
        "job_name": Path(output_dir).name,
        "batch_size": int(batch_size),
        "steps": int(steps),
        "save_freq": max(1, save_freq),
        "log_freq": 20,
        "num_workers": 4,
        "seed": int(seed),
        "resume": False,
        "eval_freq": -1,
        "save_checkpoint": True,
        "use_policy_training_preset": True,
        "wandb": {"enable": False, "disable_artifact": True},
    }


def train_lerobot_config_in_notebook(config_path, enabled=False, progress_name="train"):
    """Run LeRobot offline training directly inside the notebook kernel.

    The notebook cell owns dataset creation, policy creation, optimizer steps,
    checkpoint saving, tqdm progress, and metric JSONL writing.
    """
    config_path = Path(config_path)
    print("config =", public_path(config_path))
    if not enabled:
        print("未启动。设置 RUN_SMOKE=1 或 RUN_LONG_TRAIN=1 后，本单元会直接在 Notebook 内训练。")
        return None
    if not ensure_project_layout():
        return None

    import time
    from contextlib import nullcontext

    import draccus
    import torch
    from torch.amp import GradScaler
    from tqdm.auto import tqdm

    from lerobot.common.datasets.factory import make_dataset
    from lerobot.common.datasets.sampler import EpisodeAwareSampler
    from lerobot.common.optim.factory import make_optimizer_and_scheduler
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.policies.utils import get_device_from_parameters
    from lerobot.common.utils.random_utils import set_seed
    from lerobot.common.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint
    from lerobot.common.utils.utils import get_safe_torch_device
    from lerobot.configs.train import TrainPipelineConfig

    cfg = draccus.parse(TrainPipelineConfig, config_path=config_path, args=[])
    cfg.validate()
    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Creating dataset...")
    dataset = make_dataset(cfg)
    print("Creating policy...")
    pretrained_override = os.environ.get(f"{cfg.policy.type.upper()}_PRETRAINED_PATH_OVERRIDE") or os.environ.get("POLICY_PRETRAINED_PATH_OVERRIDE")
    if pretrained_override and not cfg.resume:
        cfg.policy.pretrained_path = str(Path(pretrained_override))
        print("pretrained override =", public_path(cfg.policy.pretrained_path))
    elif cfg.policy.type == "pi0" and not cfg.resume:
        cfg.policy.pretrained_path = "lerobot/pi0"
    elif cfg.policy.type == "smolvla" and not cfg.resume:
        smolvla_base_candidates = [
            os.environ.get("SMOLVLA_BASE_PATH"),
            os.environ.get("SMOLVLA_PRETRAINED_BASE_PATH"),
            str(MODEL_ROOT / "smolvla_base" / "pretrained_model"),
            str(MODEL_ROOT / "lerobot_smolvla_base_legacy"),
            str(MODEL_ROOT / "lerobot_smolvla_base"),
        ]
        local_smolvla_base = next((Path(p) for p in smolvla_base_candidates if p and Path(p).exists()), None)
        if local_smolvla_base is not None:
            cfg.policy.pretrained_path = str(local_smolvla_base)
            print("local smolvla base =", public_path(cfg.policy.pretrained_path))
        else:
            cfg.policy.pretrained_path = "lerobot/smolvla_base"
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    # Compatibility for newer Transformers: PaliGemmaForConditionalGeneration may expose
    # language_model as GemmaModel directly, while this LeRobot Pi0 code expects
    # language_model.model.  Use a non-Module proxy so checkpoints/state_dict stay clean.
    if cfg.policy.type == "pi0":
        try:
            lm = policy.model.paligemma_with_expert.paligemma.language_model
            if not hasattr(lm, "model"):
                class _LanguageModelCoreProxy:
                    def __init__(self, core):
                        self._core = core

                    def __getattr__(self, name):
                        return getattr(self._core, name)

                object.__setattr__(lm, "model", _LanguageModelCoreProxy(lm))
                print("patched Pi0 PaliGemma language_model.model compatibility proxy")
        except Exception as exc:
            print(f"Pi0 PaliGemma compatibility patch skipped: {exc}")

    policy.to(device)
    policy.train()

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    def _dataset_column_values(name):
        hf_dataset = getattr(dataset, "hf_dataset", None)
        if hf_dataset is None or name not in getattr(hf_dataset, "column_names", []):
            return None
        values = hf_dataset[name]
        try:
            return list(values)
        except TypeError:
            return [values[i] for i in range(len(values))]

    def _task_name_map():
        meta = getattr(dataset, "meta", None)
        tasks = getattr(meta, "tasks", None)
        if tasks is None:
            return {}
        if isinstance(tasks, dict):
            return {int(k): str(v) for k, v in tasks.items()}
        try:
            return {int(k): str(v) for k, v in dict(tasks).items()}
        except Exception:
            return {}

    def _make_weighted_sampler(generator):
        mode = os.environ.get("NOTEBOOK_FRAME_WEIGHT_MODE", "").strip().lower()
        if not mode or mode in {"0", "none", "off", "false"}:
            return None, {"mode": "none"}
        weights = torch.ones(len(dataset), dtype=torch.double)
        info = {"mode": mode, "num_frames": len(dataset)}

        if "blue" in mode:
            blue_weight = float(os.environ.get("NOTEBOOK_BLUE_WEIGHT", "2.0"))
            mask = [False] * len(dataset)
            task_indices = _dataset_column_values("task_index")
            task_names = _task_name_map()
            if task_indices is not None and task_names:
                for idx, task_index in enumerate(task_indices):
                    task_text = task_names.get(int(task_index), "").lower()
                    mask[idx] = ("blue" in task_text) or ("蓝" in task_text)
            else:
                for column in ["task", "language_instruction", "instruction"]:
                    values = _dataset_column_values(column)
                    if values is None:
                        continue
                    for idx, value in enumerate(values):
                        text = str(value).lower()
                        mask[idx] = ("blue" in text) or ("蓝" in text)
                    break
            blue_count = int(sum(mask))
            if blue_count == 0:
                print("警告：NOTEBOOK_FRAME_WEIGHT_MODE=blue 但没有识别到 blue/蓝 指令帧，采样退回均匀。")
            else:
                for idx, is_blue in enumerate(mask):
                    if is_blue:
                        weights[idx] *= blue_weight
            info.update({"blue_weight": blue_weight, "blue_frames": blue_count})

        weight_file = os.environ.get("NOTEBOOK_FRAME_WEIGHT_JSON")
        if weight_file:
            payload = json.loads(Path(weight_file).read_text(encoding="utf-8"))
            for key, value in payload.items():
                weights[int(key)] *= float(value)
            info.update({"weight_json": public_path(weight_file), "json_entries": len(payload)})

        if float(weights.sum()) <= 0:
            raise ValueError("采样权重总和为 0。")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        info.update(
            {
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
                "weight_mean": float(weights.mean()),
            }
        )
        return sampler, info

    generator = torch.Generator()
    if cfg.seed is not None:
        generator.manual_seed(int(cfg.seed))

    weighted_sampler, sampler_info = _make_weighted_sampler(generator)
    if weighted_sampler is not None:
        shuffle = False
        sampler = weighted_sampler
        print("Notebook weighted sampler =", json.dumps(sampler_info, ensure_ascii=False))
    elif hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator if sampler is None else None,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    # Do not use itertools.cycle here: it caches every batch and can exhaust
    # host RAM during a long Notebook training run. Recreate the iterator only
    # when the finite DataLoader is exhausted.
    dl_iter = iter(dataloader)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "notebook_train_metrics.jsonl"
    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    print(f"output_dir = {public_path(output_dir)}")
    print(f"steps = {cfg.steps}, batch_size = {cfg.batch_size}, frames = {dataset.num_frames}, episodes = {dataset.num_episodes}")
    print(f"learnable_params = {num_learnable:,}, total_params = {num_total:,}")

    last_metrics = None
    progress = tqdm(range(1, cfg.steps + 1), desc=progress_name, dynamic_ncols=True)
    start_all = time.perf_counter()
    for step in progress:
        load_start = time.perf_counter()
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)
        data_s = time.perf_counter() - load_start
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)

        update_start = time.perf_counter()
        device_from_policy = get_device_from_parameters(policy)
        with torch.autocast(device_type=device_from_policy.type) if cfg.policy.use_amp else nullcontext():
            loss, output_dict = policy.forward(batch)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            cfg.optimizer.grad_clip_norm,
            error_if_nonfinite=False,
        )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if hasattr(policy, "update"):
            policy.update()
        update_s = time.perf_counter() - update_start

        is_log_step = cfg.log_freq > 0 and (step % cfg.log_freq == 0 or step == 1 or step == cfg.steps)
        is_saving_step = cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps)
        if is_log_step:
            last_metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "update_s": float(update_s),
                "data_s": float(data_s),
                "elapsed_s": float(time.perf_counter() - start_all),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last_metrics, ensure_ascii=False) + "\n")
            progress.set_postfix(
                loss=f"{last_metrics['loss']:.4f}",
                lr=f"{last_metrics['lr']:.1e}",
                updt_s=f"{last_metrics['update_s']:.3f}",
            )
        if is_saving_step:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            print(f"\nSaving checkpoint at step {step}: {public_path(checkpoint_dir)}")
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)

    print("训练完成。metrics =", public_path(metrics_path))
    if last_metrics is not None:
        print(json.dumps(last_metrics, ensure_ascii=False, indent=2))
    return {"output_dir": output_dir, "metrics_path": metrics_path, "last_metrics": last_metrics}


def load_eval_module():
    import importlib.util

    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"评估脚本不存在：{public_path(EVAL_SCRIPT)}")
    spec = importlib.util.spec_from_file_location("notebook_eval_policy_success", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_eval_policy_in_notebook(
    policy_name,
    policy_path,
    result_path,
    episodes,
    seed_start,
    render=False,
    enabled=False,
    repo_id=None,
    dataset_root=None,
):
    print("policy =", policy_name)
    print("policy_path =", public_path(policy_path))
    print("result =", public_path(result_path))
    if not enabled:
        print("未启动。设置 RUN_EVAL=1 后，本单元会在 Notebook 内直接加载策略并闭环评估。")
        return None
    if not ensure_project_layout():
        return None

    import argparse
    from contextlib import contextmanager
    from tqdm.auto import tqdm

    @contextmanager
    def pushd(path):
        old = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old)

    ensure_xvfb_display()
    module = load_eval_module()
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    args = argparse.Namespace(
        policy=policy_name,
        episodes=int(episodes),
        seed_start=int(seed_start),
        max_action_steps=int(os.environ.get("EVAL_MAX_ACTION_STEPS", "400")),
        hz=float(os.environ.get("EVAL_HZ", "20")),
        render=bool(render),
        output_jsonl=result_path,
        device=os.environ.get("EVAL_DEVICE", "cuda"),
        reset_policy_each_action=env_flag("EVAL_RESET_POLICY_EACH_ACTION", False),
        act_n_action_steps=None,
        act_force_dataset_gripper=False,
        act_clamp_timestamp=False,
        act_policy_path=Path(policy_path),
        act_repo_id=repo_id or "datawhale_eai_pnp",
        act_dataset_root=Path(dataset_root or "./demo_data"),
        act_episode_timestamp_offsets="",
        act_episode_source_flags="",
        physical_success=env_flag("EVAL_PHYSICAL_SUCCESS", True),
        physical_min_lift=float(os.environ.get("EVAL_PHYSICAL_MIN_LIFT", "0.06")),
        physical_min_lift_steps=int(os.environ.get("EVAL_PHYSICAL_MIN_LIFT_STEPS", "3")),
        physical_final_upright_cos=float(os.environ.get("EVAL_PHYSICAL_FINAL_UPRIGHT_COS", "0.85")),
        smolvla_policy_path=Path(policy_path),
        pi0_policy_path=Path(policy_path),
        pi0_repo_id=repo_id or os.environ.get("PI0_DATASET_REPO_ID", "datawhale_eai_pnp_language"),
        pi0_dataset_root=Path(dataset_root or os.environ.get("PI0_DATASET_ROOT", "./demo_data_language")),
    )

    with pushd(PROJECT_ROOT):
        if policy_name == "act":
            policy = module.make_act_policy(
                args.device,
                args.act_policy_path,
                args.act_repo_id,
                args.act_dataset_root,
                n_action_steps=args.act_n_action_steps,
                episode_timestamp_offsets=args.act_episode_timestamp_offsets,
                episode_source_flags=args.act_episode_source_flags,
            )
            rollout = module.rollout_act
        elif policy_name == "smolvla":
            policy = module.make_smolvla_policy(args.device, args.smolvla_policy_path)
            rollout = module.rollout_language_policy
        elif policy_name == "pi0":
            policy = module.make_pi0_policy(args.device, args.pi0_policy_path, args.pi0_repo_id, args.pi0_dataset_root)
            rollout = module.rollout_language_policy
        else:
            raise ValueError(policy_name)

        rows = []
        for offset in tqdm(range(args.episodes), desc=f"{policy_name} eval", dynamic_ncols=True):
            seed = args.seed_start + offset
            row = rollout(args, policy, seed)
            rows.append(row)
            with result_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))
    summarize_jsonl(result_path)
    return rows


def list_checkpoints(run_dir):
    run_dir = Path(run_dir)
    candidates = []
    for pattern in ["checkpoints/*/pretrained_model", "checkpoint*/pretrained_model", "*/pretrained_model", "pretrained_model"]:
        candidates.extend(run_dir.glob(pattern))
    unique = sorted(set(candidates))
    if not unique:
        print("尚未发现 checkpoint：", public_path(run_dir))
        return []
    for path in unique:
        print(" -", public_path(path))
    return unique


def resolve_eval_policy(default_path, trained_run_dir=None, env_name=None):
    if env_name and os.environ.get(env_name):
        path = Path(os.environ[env_name])
        print("评估使用环境变量指定权重：", public_path(path))
        return path
    if trained_run_dir is not None and env_flag("EVAL_USE_LONG_TRAIN"):
        checkpoints = list_checkpoints(trained_run_dir)
        if checkpoints:
            path = checkpoints[-1]
            print("评估使用本次长训最新 checkpoint：", public_path(path))
            return path
        print("未找到本次长训 checkpoint，回退到保护权重。")
    path = Path(default_path)
    print("评估使用保护/预训练权重：", public_path(path))
    return path


def summarize_jsonl(path):
    path = Path(path)
    if not path.exists():
        print("结果 JSONL 尚不存在：", public_path(path))
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    legacy = sum(bool(row.get("success") or row.get("legacy_success")) for row in rows)
    if rows and all("physical_success" in row for row in rows):
        physical_count = sum(bool(row.get("physical_success")) for row in rows)
        physical_text = str(physical_count) + "/" + str(total)
    else:
        physical_text = "未记录"
    md_table(
        ["结果文件", "episodes", "legacy_success", "physical_success"],
        [(public_path(path), total, f"{legacy}/{total}", physical_text)],
    )
    return rows
'''


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str, output: str | None = None) -> dict:
    outputs = []
    execution_count = None
    if output is not None:
        execution_count = 1
        outputs = [{"name": "stdout", "output_type": "stream", "text": output}]
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source.strip("\n") + "\n",
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / filename
    nb = {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


def smolvla_cells() -> list[dict]:
    return [
        md(
            """
            # 14 SmolVLA 端到端：训练、评估、视频与诊断

            这个 Notebook 把 SmolVLA 的完整学习路径放在一个文件里：先看已经复现成功的权重和视频，再检查数据，最后给出 smoke、长训、严格评估和日志追踪命令。

            运行方式建议：第一次教学演示只开 `RUN_SMOKE=1 RUN_EVAL=1`；完整复现实验再开 `RUN_LONG_TRAIN=1`。长训输出会真实写回 Notebook 单元格，而不是粘贴静态日志。
            """
        ),
        code(SETUP),
        code(HELPERS),
        md("## Checkpoint 1：先确认这一版为什么作为主线"),
        code(
            r'''
rows = [
    ("历史教程记录", "53/60", "SmolVLA weighted step500，红蓝杯平衡较好"),
    ("当前重建结果", "57/60", "red 27/30，blue 30/30"),
    ("发布建议", "主推权重", "适合作为零训练预览和默认 Notebook 案例"),
]
md_table(["项目", "结果", "说明"], rows)
''',
            "项目 | 结果 | 说明\n历史教程记录 | 53/60 | SmolVLA weighted step500，红蓝杯平衡较好\n当前重建结果 | 57/60 | red 27/30，blue 30/30\n发布建议 | 主推权重 | 适合作为零训练预览和默认 Notebook 案例\n",
        ),
        md("## Checkpoint 2：显示严格成功与失败视频"),
        code(
            r'''
show_video("smolvla_weighted500_red_success_seed0.mp4", "红杯成功回放：weighted500 seed0")
show_video("smolvla_weighted500_blue_success_seed0.mp4", "蓝杯成功回放：weighted500 seed0")
show_video("smolvla_weighted500_red_failure_seed8.mp4", "红杯失败回放：用于观察 upright/release 问题")
''',
        ),
        md(
            """
            ## Checkpoint 3：检查数据和权重路径

            预计耗时：几秒。这里不会训练，只确认数据、权重和输出目录是否指向正确位置。
            """
        ),
        code(
            r'''
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language")
TRAIN_DATA_ROOT = Path(os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "demo_data_language"))
PRETRAINED_POLICY = Path(os.environ.get("SMOLVLA_POLICY_PATH", MODEL_ROOT / "smolvla_weighted_000500" / "pretrained_model"))

rows = [
    ("DATASET_REPO_ID", DATASET_REPO_ID),
    ("TRAIN_DATA_ROOT", TRAIN_DATA_ROOT),
    ("SMOLVLA_POLICY_PATH", PRETRAINED_POLICY),
    ("数据 meta", TRAIN_DATA_ROOT / "meta" / "info.json"),
]
md_table(["变量", "当前值"], rows)
''',
        ),
        md(
            """
            ## Checkpoint 4：生成配置并真实启动 smoke / 长训

            预计耗时：smoke 约 1-5 分钟；`SMOLVLA_STEPS=5000` 的长训通常需要几十分钟到数小时，取决于 ROCm、batch size 和数据盘速度。
            这一格是真实训练入口：设置 `RUN_SMOKE=1` 或 `RUN_LONG_TRAIN=1` 后执行，会在 Notebook kernel 内直接创建 dataset、policy、optimizer 和训练循环，并显示 tqdm 进度。
            """
        ),
        code(
            r'''
CONFIG_DIR = OUTPUT_ROOT / "configs"
LOG_DIR = OUTPUT_ROOT / "logs"
RUN_ROOT = OUTPUT_ROOT / "runs" / "smolvla_weighted_repro"
SMOKE_OUTPUT = RUN_ROOT / "smoke"
LONG_OUTPUT = RUN_ROOT / "weighted_full"

smoke_config = make_lerobot_train_config(
    "smolvla", DATASET_REPO_ID, TRAIN_DATA_ROOT, SMOKE_OUTPUT,
    steps=2, batch_size=2, chunk_size=50, n_action_steps=50,
)
long_config = make_lerobot_train_config(
    "smolvla", DATASET_REPO_ID, TRAIN_DATA_ROOT, LONG_OUTPUT,
    steps=int(os.environ.get("SMOLVLA_STEPS", "5000")),
    batch_size=int(os.environ.get("SMOLVLA_BATCH_SIZE", "4")),
    chunk_size=50,
    n_action_steps=50,
)
smoke_config_path = write_json_yaml(CONFIG_DIR / "smolvla_smoke.yaml", smoke_config)
long_config_path = write_json_yaml(CONFIG_DIR / "smolvla_weighted_full.yaml", long_config)

train_lerobot_config_in_notebook(smoke_config_path, enabled=RUN_SMOKE, progress_name="SmolVLA smoke")
train_lerobot_config_in_notebook(long_config_path, enabled=RUN_LONG_TRAIN, progress_name="SmolVLA long train")
''',
            "Notebook 原生训练入口。RUN_SMOKE/RUN_LONG_TRAIN 默认关闭；确认路径后再打开。\n",
        ),
        md(
            """
            ## Checkpoint 5：实时查看训练日志和 checkpoint

            预计耗时：几秒。长训进行中可以反复执行本格，查看 Notebook 训练写出的 metrics 和已经落盘的 checkpoint。
            """
        ),
        code(
            r'''
print("smoke metrics:")
tail_log(SMOKE_OUTPUT / "notebook_train_metrics.jsonl", lines=20)
print("\nlong train metrics:")
tail_log(LONG_OUTPUT / "notebook_train_metrics.jsonl", lines=40)
print("\ncheckpoints:")
list_checkpoints(LONG_OUTPUT)
''',
        ),
        md(
            """
            ## Checkpoint 6：已完成长训的实测对照

            这是我们已经在 AMD 设备上跑完并复核过的视频/指标对照，用来帮助学习者先看到“正确跑起来是什么样”。它不替代本次 Notebook 的训练输出。
            """
        ),
        code(
            r'''
rows = [
    ("parent", "5000/5000 steps", "基础 SmolVLA 收敛到可用 checkpoint"),
    ("weighted-blue", "selected step 500/1000", "step500 比 step1000 更平衡"),
    ("strict gate", "57/60", "legacy 60/60，physical 57/60"),
]
md_table(["阶段", "训练/评估进度", "结论"], rows)
''',
            "阶段 | 训练/评估进度 | 结论\nparent | 5000/5000 steps | 基础 SmolVLA 收敛到可用 checkpoint\nweighted-blue | selected step 500/1000 | step500 比 step1000 更平衡\nstrict gate | 57/60 | legacy 60/60，physical 57/60\n",
        ),
        code(
            r'''
show_image("training_progress_overview.png", "历史训练进度与闭环结果")
show_image("smolvla_red_blue_success.png", "红杯/蓝杯分指令对比")
''',
        ),
        md(
            """
            ## Checkpoint 7：Notebook 内严格评估

            预计耗时：10 个 episode 通常 10-30 分钟；`SMOLVLA_EVAL_EPISODES=60` 会更久。
            本单元会在 Notebook kernel 内加载策略并逐个 seed 闭环 rollout；如无显示器，会自动尝试启动 Xvfb。
            """
        ),
        code(
            r'''
eval_episodes = os.environ.get("SMOLVLA_EVAL_EPISODES", "10")
eval_policy = resolve_eval_policy(PRETRAINED_POLICY, LONG_OUTPUT, "SMOLVLA_EVAL_POLICY_PATH")
run_eval_policy_in_notebook(
    "smolvla",
    eval_policy,
    OUTPUT_ROOT / f"smolvla_eval_seed0_{int(eval_episodes)-1}.jsonl",
    episodes=eval_episodes,
    seed_start=0,
    render=env_flag("RENDER_EVAL"),
    enabled=RUN_EVAL,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
summarize_jsonl(OUTPUT_ROOT / f"smolvla_eval_seed0_{int(eval_episodes)-1}.jsonl")
'''
        ),
        md("## Checkpoint 8：怎么读结果"),
        code(
            r'''
summary = {
    "candidate": "weighted_000500",
    "episodes": 60,
    "physical_success_count": 57,
    "legacy_success_count": 60,
    "by_color": {"red": "27/30", "blue": "30/30"},
    "release_decision": "作为教程默认预训练权重",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
''',
            '{\n  "candidate": "weighted_000500",\n  "episodes": 60,\n  "physical_success_count": 57,\n  "legacy_success_count": 60,\n  "by_color": {\n    "red": "27/30",\n    "blue": "30/30"\n  },\n  "release_decision": "作为教程默认预训练权重"\n}\n',
        ),
    ]


def pi0_cells() -> list[dict]:
    return [
        md(
            """
            # 15 Pi0 端到端：权限、长训、严格输入与失败桶

            Pi0 这一版不是“完全复现历史 hard8”的主线，而是进阶诊断模型。Notebook 重点展示 gated 权限、长训配置、S8500 checkpoint、median reducer、严格评估和失败桶分析。
            """
        ),
        code(SETUP),
        code(HELPERS),
        md("## Checkpoint 1：当前恢复边界"),
        code(
            r'''
rows = [
    ("unseen10", "9/10", "超过早期 7/10"),
    ("full14", "12/14", "红 6/7，蓝 6/7"),
    ("hard8", "6/8", "未恢复历史 8/8，不能写成完全复现"),
]
md_table(["面板", "当前结果", "结论"], rows)
''',
            "面板 | 当前结果 | 结论\nunseen10 | 9/10 | 超过早期 7/10\nfull14 | 12/14 | 红 6/7，蓝 6/7\nhard8 | 6/8 | 未恢复历史 8/8，不能写成完全复现\n",
        ),
        md(
            """
            ## Checkpoint 2：gated 权限和缓存检查

            预计耗时：几秒。Pi0 需要确认模型权限、Hugging Face 缓存和训练数据路径，Notebook 只检查状态，不打印任何 token。
            """
        ),
        code(
            r'''
print("HF_TOKEN 已注入：", bool(os.environ.get("HF_TOKEN")))
rows = [
    ("HF_TOKEN", "只检查是否存在，不在 Notebook 打印 token"),
    ("HF_HOME", "已设置" if os.environ.get("HF_HOME") else "未设置"),
    ("模型缓存", "建议放到 $CACHE_ROOT/huggingface 或云平台持久盘"),
]
md_table(["项目", "说明"], rows)
''',
        ),
        md(
            """
            ## Checkpoint 3：生成配置并真实启动训练/续训

            预计耗时：smoke 约 1-5 分钟；`PI0_STEPS=8500` 属于长训，可能需要数小时。
            这一步是真实训练入口：开 `RUN_SMOKE=1` 做环境验证，开 `RUN_LONG_TRAIN=1` 后会在 Notebook kernel 内直接跑 Pi0 训练循环，并显示 tqdm 进度。
            """
        ),
        code(
            r'''
DATASET_REPO_ID = "datawhale_eai_pnp_pi0_clean_oracle_y060z000_3100_3139_g8_rebuild_v1"
TRAIN_DATA_ROOT = Path(os.environ.get("PI0_TRAIN_DATA_ROOT", DATA_ROOT / "pi0_clean40_successonly"))
PI0_POLICY_PATH = Path(os.environ.get("PI0_POLICY_PATH", MODEL_ROOT / "pi0_clean40_successonly_blue2x_s8500" / "pretrained_model"))
CONFIG_DIR = OUTPUT_ROOT / "configs"
LOG_DIR = OUTPUT_ROOT / "logs"
RUN_ROOT = OUTPUT_ROOT / "runs" / "pi0_s8500_repro"
SMOKE_OUTPUT = RUN_ROOT / "smoke"
LONG_OUTPUT = RUN_ROOT / "clean40_blue2x_s8500"

smoke_config = make_lerobot_train_config(
    "pi0", DATASET_REPO_ID, TRAIN_DATA_ROOT, SMOKE_OUTPUT,
    steps=2, batch_size=1, chunk_size=50, n_action_steps=50,
)
long_config = make_lerobot_train_config(
    "pi0", DATASET_REPO_ID, TRAIN_DATA_ROOT, LONG_OUTPUT,
    steps=int(os.environ.get("PI0_STEPS", "8500")),
    batch_size=int(os.environ.get("PI0_BATCH_SIZE", "2")),
    chunk_size=50,
    n_action_steps=50,
)
smoke_config_path = write_json_yaml(CONFIG_DIR / "pi0_smoke.yaml", smoke_config)
long_config_path = write_json_yaml(CONFIG_DIR / "pi0_clean40_blue2x_s8500.yaml", long_config)

train_lerobot_config_in_notebook(smoke_config_path, enabled=RUN_SMOKE, progress_name="Pi0 smoke")
train_lerobot_config_in_notebook(long_config_path, enabled=RUN_LONG_TRAIN, progress_name="Pi0 long train")
''',
            "Notebook 原生训练入口。RUN_SMOKE/RUN_LONG_TRAIN 默认关闭；Pi0 训练前必须确认 gated 权限。\n",
        ),
        md(
            """
            ## Checkpoint 4：实时查看训练日志和 checkpoint

            预计耗时：几秒。训练中可以反复执行这一格，观察 loss、step、保存点和是否有 OOM/NaN。
            """
        ),
        code(
            r'''
print("smoke metrics:")
tail_log(SMOKE_OUTPUT / "notebook_train_metrics.jsonl", lines=20)
print("\nlong train metrics:")
tail_log(LONG_OUTPUT / "notebook_train_metrics.jsonl", lines=40)
print("\ncheckpoints:")
list_checkpoints(LONG_OUTPUT)
''',
        ),
        md(
            """
            ## Checkpoint 5：已完成长训的实测对照

            这是此前已在 AMD 设备上完成的结果对照，帮助学习者知道 Pi0 当前恢复到了哪里；它不替代本次 Notebook 的真实运行输出。
            """
        ),
        code(
            r'''
rows = [
    ("S7500", "canary2/4", "继续训练的父 checkpoint"),
    ("S8500", "canary3/4", "进入 full14 扩展评估"),
    ("full14 median", "12/14", "作为当前保护版本"),
]
md_table(["阶段", "门禁", "结论"], rows)
''',
            "阶段 | 门禁 | 结论\nS7500 | canary2/4 | 继续训练的父 checkpoint\nS8500 | canary3/4 | 进入 full14 扩展评估\nfull14 median | 12/14 | 作为当前保护版本\n",
        ),
        md(
            """
            ## Checkpoint 6：Notebook 内严格闭环评估

            预计耗时：14 个 episode 常见为几十分钟；可以用 `PI0_EVAL_EPISODES=4` 做快速门禁。
            本单元会在 Notebook kernel 内直接加载 Pi0，并使用 checkpoint 对应的 8 维 `observation.state` 协议闭环评估。
            """
        ),
        code(
            r'''
eval_episodes = os.environ.get("PI0_EVAL_EPISODES", "14")
eval_policy = resolve_eval_policy(PI0_POLICY_PATH, LONG_OUTPUT, "PI0_EVAL_POLICY_PATH")
result_path = OUTPUT_ROOT / f"pi0_s8500_seed{os.environ.get('PI0_EVAL_SEED_START', '3000')}_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "pi0",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=os.environ.get("PI0_EVAL_SEED_START", "3000"),
    render=env_flag("RENDER_EVAL"),
    enabled=RUN_EVAL,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
summarize_jsonl(result_path)
''',
        ),
        md("## Checkpoint 7：失败桶和视频复核"),
        code(
            r'''
summary = {
    "episodes": 14,
    "physical_success_count": 12,
    "by_color": {"blue": "6/7", "red": "6/7"},
    "failures": [
        "blue seed3007: lifted but upright unstable",
        "red seed3012: no enough lift/contact",
    ],
    "do_not_claim": "hard8 8/8 has not been recovered",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
show_video("pi0_ep2_raw_vs_finisher_side_by_side.mp4", "Pi0 raw 与诊断 finisher 对照视频")
show_image("pi0_raw_vs_finisher_diagnostic.png", "Pi0 诊断图：raw、scaffold 与严格判定")
''',
            '{\n  "episodes": 14,\n  "physical_success_count": 12,\n  "by_color": {\n    "blue": "6/7",\n    "red": "6/7"\n  },\n  "failures": [\n    "blue seed3007: lifted but upright unstable",\n    "red seed3012: no enough lift/contact"\n  ],\n  "do_not_claim": "hard8 8/8 has not been recovered"\n}\n',
        ),
    ]


def act_cells() -> list[dict]:
    return [
        md(
            """
            # 16 ACT 端到端：训练链路与失败诊断

            ACT 的训练、推理、评估链路已经跑通。当前 AMD395 修复续训的最佳保护候选在严格 30 条上为 `15/30`，已经超过 stable61 fallback 的 `7/30`；W7900 普通 ACT 为 `0/30`，旧 DAgger artifact 为 `2/30`。旧教程摘要中的 `17/30` 仍保留为历史目标，不能冒充当前权重结果。
            """
        ),
        code(SETUP),
        code(HELPERS),
        md("## Checkpoint 1：当前结果与历史结果"),
        code(
            r'''
rows = [
    ("历史教程记录", "17/30（历史目标）", "旧 DAgger 摘要；当前未由保护权重完全恢复"),
    ("当前 repair15 保护候选", "15/30", "AMD395；step1500；三面板 3/10 + 4/10 + 8/10"),
    ("旧 stable61 fallback", "7/30", "AMD395；step2500；基线对照"),
    ("发布建议", "当前可保护/可教学", "不宣称已完全恢复历史 17/30"),
]
md_table(["项目", "结果", "说明"], rows)
''',
            "项目 | 结果 | 说明\n历史教程记录 | 17/30 | 旧 DAgger 最好结果\n当前重建最好 | 7/30 | stable61 step2500 strict30\n发布建议 | 负例/诊断 | 不要作为主推成功权重\n",
        ),
        md(
            """
            ## Checkpoint 2：生成配置并真实启动训练

            预计耗时：ACT smoke 约 1-3 分钟；`ACT_STEPS=5000` 长训通常几十分钟到数小时。
            ACT 这里主要用于诊断：训练能跑通并不等于闭环成功率高，后面必须接 strict eval 和视频复核。
            """
        ),
        code(
            r'''
DATASET_REPO_ID = os.environ.get("ACT_DATASET_REPO_ID", "datawhale_eai_pnp_language")
TRAIN_DATA_ROOT = Path(os.environ.get("ACT_TRAIN_DATA_ROOT", DATA_ROOT / "demo_data_language"))
ACT_POLICY_PATH = Path(os.environ.get("ACT_POLICY_PATH", MODEL_ROOT / "act_stable61_step2500" / "pretrained_model"))
CONFIG_DIR = OUTPUT_ROOT / "configs"
LOG_DIR = OUTPUT_ROOT / "logs"
RUN_ROOT = OUTPUT_ROOT / "runs" / "act_stable61_repro"
SMOKE_OUTPUT = RUN_ROOT / "smoke"
LONG_OUTPUT = RUN_ROOT / "stable61_full"

smoke_config = make_lerobot_train_config(
    "act", DATASET_REPO_ID, TRAIN_DATA_ROOT, SMOKE_OUTPUT,
    steps=2, batch_size=4, chunk_size=50, n_action_steps=50,
)
long_config = make_lerobot_train_config(
    "act", DATASET_REPO_ID, TRAIN_DATA_ROOT, LONG_OUTPUT,
    steps=int(os.environ.get("ACT_STEPS", "5000")),
    batch_size=int(os.environ.get("ACT_BATCH_SIZE", "8")),
    chunk_size=50,
    n_action_steps=50,
)
smoke_config_path = write_json_yaml(CONFIG_DIR / "act_smoke.yaml", smoke_config)
long_config_path = write_json_yaml(CONFIG_DIR / "act_stable61_full.yaml", long_config)

train_lerobot_config_in_notebook(smoke_config_path, enabled=RUN_SMOKE, progress_name="ACT smoke")
train_lerobot_config_in_notebook(long_config_path, enabled=RUN_LONG_TRAIN, progress_name="ACT long train")
''',
            "Notebook 原生训练入口。RUN_SMOKE/RUN_LONG_TRAIN 默认关闭；训练完成后必须做 strict closed-loop。\n",
        ),
        md(
            """
            ## Checkpoint 3：实时查看训练日志和 checkpoint

            预计耗时：几秒。长训中可以重复运行本格，确认 step 是否推进、checkpoint 是否落盘。
            """
        ),
        code(
            r'''
print("smoke metrics:")
tail_log(SMOKE_OUTPUT / "notebook_train_metrics.jsonl", lines=20)
print("\nlong train metrics:")
tail_log(LONG_OUTPUT / "notebook_train_metrics.jsonl", lines=40)
print("\ncheckpoints:")
list_checkpoints(LONG_OUTPUT)
''',
        ),
        md(
            """
            ## Checkpoint 4：已完成长训的实测对照

            这是此前已在 AMD 设备上完成的 ACT 诊断对照。当前 Notebook 的真实长训结果需要以后续 strict eval 为准。
            """
        ),
        code(
            r'''
rows = [
    ("stage1 stable42", "8000 steps", "可训练但闭环仍不稳"),
    ("stable61", "5000 steps", "step2500 优于 step5000"),
    ("strict30", "15/30", "当前 repair15 保护候选；三面板 3/10 + 4/10 + 8/10"),
]
md_table(["阶段", "训练进度", "结论"], rows)
''',
            "阶段 | 训练进度 | 结论\nstage1 stable42 | 8000 steps | 可训练但闭环仍不稳\nstable61 | 5000 steps | step2500 优于 step5000\nstrict30 | 7/30 | 当前保护的 fallback best\n",
        ),
        md(
            """
            ## Checkpoint 5：Notebook 内严格评估

            预计耗时：30 个 episode 往往需要较久；可以先用 `ACT_EVAL_EPISODES=6` 做快速门禁，再扩到 30。
            本单元会在 Notebook kernel 内直接加载 ACT 并逐 seed rollout；不再 shell 到外部训练/评估脚本。
            """
        ),
        code(
            r'''
eval_episodes = os.environ.get("ACT_EVAL_EPISODES", "30")
eval_seed_start = os.environ.get("ACT_EVAL_SEED_START", "1030")
eval_policy = resolve_eval_policy(ACT_POLICY_PATH, LONG_OUTPUT, "ACT_EVAL_POLICY_PATH")
result_path = OUTPUT_ROOT / f"act_stable61_seed{eval_seed_start}_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "act",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=eval_seed_start,
    render=env_flag("RENDER_EVAL"),
    enabled=RUN_EVAL,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
summarize_jsonl(result_path)
''',
        ),
        md("## Checkpoint 6：为什么它是诊断反例"),
        code(
            r'''
rows = [
    ("接近/接触", "经常能做到", "说明不是完全没有视觉或动作能力"),
    ("搬运/对准/释放", "最容易失败", "长时序行为克隆会累积误差"),
    ("loss", "不能单独决定 checkpoint", "必须用 strict physical_success + 视频"),
    ("继续方向", "更高质量 success-only / recovery 数据", "不建议盲目扩大模型参数"),
]
md_table(["观察", "现象", "处理方式"], rows)
show_image("act_success_sequence.jpg", "ACT 成功关键帧")
show_image("act_failure_sequence.jpg", "ACT 失败关键帧")
''',
        ),
        md("## Checkpoint 7：写入教程时的结论"),
        code(
            r'''
print("""
ACT 这条线适合放在教程的诊断章节：
1. 训练和评估链路可复现；
2. 当前 repair15 保护候选为严格 15/30，历史目标 17/30 尚未完全恢复；
3. 它很好地说明 strict physical_success、视频复核和分阶段失败桶为什么必要。
""")
''',
        ),
    ]


def main() -> None:
    write_nb("14_smolvla_end_to_end.ipynb", smolvla_cells())
    write_nb("15_pi0_end_to_end.ipynb", pi0_cells())
    write_nb("16_act_end_to_end.ipynb", act_cells())


if __name__ == "__main__":
    main()
