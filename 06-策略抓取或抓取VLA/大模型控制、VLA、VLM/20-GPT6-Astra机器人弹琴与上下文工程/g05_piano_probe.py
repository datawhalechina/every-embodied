"""Bounded offline G0.5 AR inference on saved native R1 Pro observations."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shared-models", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    repo = args.repo.resolve()
    for path in (repo, repo / "src"):
        sys.path.insert(0, str(path))
    os.chdir(repo)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from g05.models.g05.inferencer import PolicyInferencer
    from g05.utils.checkpoint.ckpt_utils import load_config_from_task_yaml
    from scripts.serve_policy_mem import build_obs_dict, setup

    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.25)
    shared = args.shared_models.resolve()
    overrides = ["eval_embodiment=galaxea_r1pro", "model.model_arch.discrete_action=true", "model.model_arch.continuous_action=false", "model.model_weights_to_bf16=true", "model.use_torch_compile=false", "model.model_arch.attn_implementation=sdpa", "model.model_arch.hf_processor_path=" + str(shared / "qwen3_5_2b_base_processor"), "model.processor.tokenizer_params.pretrained_model_name_or_path=" + str(shared / "qwen3_5_2b_base_processor"), "tokenizer.vq_config.ckpt_dir=" + str(shared / "action_tokenizer.pt"), "model.tokenizer.vq_config.ckpt_dir=" + str(shared / "action_tokenizer.pt")]
    cfg = load_config_from_task_yaml(str(repo / "configs/task/r1pro.yaml"), str(args.checkpoint.resolve()), overrides)
    started = time.monotonic()
    policy, processor = setup(cfg, device="cuda")
    loaded = time.monotonic() - started
    inferencer = PolicyInferencer(policy, processor, device="cuda")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    with np.load(args.observation, allow_pickle=False) as archive:
        observation = {k: archive[k] for k in archive.files}
    raw = {"images": {k: observation[k] for k in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")}, "state": {k: observation[k] for k in ("left_arm", "left_gripper", "right_arm", "right_gripper")}, "frequency": 15, "embodiment_type": "galaxea_r1pro"}
    instructions = ["Play Twinkle Twinkle Little Star on the piano with both hands.", "Press and release a white key near the center of the piano with the right gripper. Keep the left arm still."]
    calls = []
    for index, instruction in enumerate(instructions):
        raw["task"] = instruction
        raw["plan"] = "Tempo: " + str(plan["tempo_bpm"]) + " BPM. Right hand MIDI notes: " + str([n["midi"] for n in plan["notes"] if n["hand"] == "right"])
        started = time.monotonic()
        actions = inferencer.infer_one(build_obs_dict(raw, processor))
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        absent = sorted(actions.pop("_absent_keys", set()))
        public_text = actions.pop("_cot_text", None)
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in actions.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Non-finite model action")
        np.savez_compressed(args.out / f"actions-{index}.npz", **arrays)
        calls.append({"instruction": instruction, "plan_context": raw["plan"], "inference_seconds": seconds, "action_shapes": {k: list(v.shape) for k, v in arrays.items()}, "absent_action_groups": absent, "public_cot": public_text, "actions_file": f"actions-{index}.npz", "actions_sha256": sha256(args.out / f"actions-{index}.npz"), "model_output_executed": False})
        print(json.dumps(calls[-1]), flush=True)
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_bytes": args.checkpoint.stat().st_size, "repo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(), "action_tokenizer_sha256": sha256(args.shared_models / "action_tokenizer.pt"), "dataset_stats_sha256": sha256(args.checkpoint.parent.parent / "dataset_stats.json"), "observation_sha256": sha256(args.observation), "gpt_plan_sha256": sha256(args.plan), "backend": "G0.5_Qwen35_autoregressive", "embodiment": "galaxea_r1pro", "raw_arm_state_units": "radians", "raw_gripper_units": "0_to_100", "dexterous_finger_outputs": False, "native_action_layout": "left_arm7_gripper1_right_arm7_gripper1", "load_seconds": loaded, "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2, "torch": torch.__version__, "calls": calls, "piano_success_verified": False, "note": "Inference smoke test only. GPT plan is context, not evidence that G0.5 can play. Outputs require simulation evaluation."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
