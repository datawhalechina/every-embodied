"""Persistent G0.5 worker: native arm proposals from each fresh simulated observation."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from g05_piano_probe import sha256
from piano_context import write_json


def serve(args):
    import numpy as np

    repo, shared, checkpoint = args.repo.resolve(), args.shared_models.resolve(), args.checkpoint.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for path in (repo, repo / "src"):
        sys.path.insert(0, str(path))
    os.chdir(repo)
    # Reserve stdout for protocol messages; upstream libraries may print during setup.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from g05.models.g05.inferencer import PolicyInferencer
        from g05.utils.checkpoint.ckpt_utils import load_config_from_task_yaml, load_config_from_run_dir, find_run_dir
        from g05.utils.eval.eval_utils import filter_embodiment
        from omegaconf import OmegaConf
        from scripts.serve_policy_mem import build_obs_dict, setup

        torch.manual_seed(0)
        torch.set_num_threads(2)
        memory_fraction = float(os.environ.get("G05_CUDA_MEMORY_FRACTION", "1.0"))
        if not 0 < memory_fraction <= 1:
            raise ValueError("G05_CUDA_MEMORY_FRACTION must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        overrides = ["eval_embodiment=galaxea_r1pro", "model.model_weights_to_bf16=true",
            "model.use_torch_compile=false", "model.model_arch.attn_implementation=sdpa",
            "model.model_arch.hf_processor_path=" + str(shared / "qwen3_5_2b_base_processor"),
            "model.processor.tokenizer_params.pretrained_model_name_or_path=" + str(shared / "qwen3_5_2b_base_processor"),
            "tokenizer.vq_config.ckpt_dir=" + str(shared / "action_tokenizer.pt"),
            "model.model_arch.AT_CONFIG.ckpt_dir=" + str(shared / "action_tokenizer.pt"),
            "model.tokenizer.vq_config.ckpt_dir=" + str(shared / "action_tokenizer.pt")]
        if args.config_mode == "checkpoint":
            cfg = load_config_from_run_dir(find_run_dir(str(checkpoint)), str(checkpoint), overrides)
            filter_embodiment(cfg, "galaxea_r1pro")
        else:
            cfg = load_config_from_task_yaml(str(repo / "configs/task/r1pro.yaml"), str(checkpoint),
                overrides + ["model.model_arch.discrete_action=true", "model.model_arch.continuous_action=false"])
        (out / "resolved-config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        started = time.monotonic()
        policy, processor = setup(cfg, device="cuda")
        inferencer = PolicyInferencer(policy, processor, device="cuda")
        load_seconds = time.monotonic() - started
    metadata = {"backend": "G0.5_Qwen35_policy", "embodiment": "galaxea_r1pro",
        "config_mode": args.config_mode, "resolved_config_sha256": sha256(out / "resolved-config.yaml"),
        "inference_paths": {key: bool(getattr(policy, key)) for key in ("discrete_action", "continuous_action", "predict_cot", "return_continuous_action")},
        "checkpoint_sha256": sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "repo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "load_seconds": load_seconds, "calls": [], "finger_action_outputs": False,
        "cuda_memory_fraction": memory_fraction,
        "gripper_state": "thumb_index_distance_proxy_0_to_100; non-native Shadow Hand observation",
        "simulation_clock_pauses_for_inference": True}
    write_json(out / "report.json", metadata)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if "seed" in request:
            seed = request["seed"]
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("Invalid inference seed")
            torch.manual_seed(seed)
            np.random.seed(seed)
        index = len(metadata["calls"])
        observation_file = Path(request["observation"]).resolve()
        with np.load(observation_file, allow_pickle=False) as archive:
            observation = {k: archive[k] for k in archive.files}
        raw = {"images": {k: observation[k] for k in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")},
               "state": {k: observation[k] for k in ("left_arm", "left_gripper", "right_arm", "right_gripper")},
               "frequency": 15, "embodiment_type": "galaxea_r1pro",
               "task": request["instruction"], "plan": request["plan"]}
        started = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            from g05.models.g05.inferencer import resolve_processor
            obs_dict = build_obs_dict(raw, processor)
            sub = resolve_processor(processor, obs_dict)
            sample = sub.preprocess(obs_dict)["samples"]
            template = sample["template"]
            # The legacy BaseSamplesBuilder ignores the separate plan field.
            # Audit actual model input instead of assuming all request fields survive.
            model_input = {"template": template, "command": sample.get("command"),
                           "plan": sample.get("plan"), "plan_field_consumed": "<plan_text" in template}
            write_json(out / f"model-input-{index}.json", model_input)
            result = inferencer.infer_one(obs_dict)
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        absent = sorted(result.pop("_absent_keys", set()))
        result.pop("_cot_text", None)
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in result.items()}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Non-finite G0.5 proposal")
        actions_file = out / f"actions-{index}.npz"
        np.savez_compressed(actions_file, **arrays)
        call = {"simulated_time_seconds": request["time_seconds"], "instruction": raw["task"],
                "seed": request.get("seed"),
                "model_input_file": f"model-input-{index}.json", "plan_field_consumed": model_input["plan_field_consumed"],
                "plan_context": raw["plan"], "observation_sha256": sha256(observation_file),
                "observation_file": str(observation_file), "actions_file": actions_file.name,
                "actions_sha256": sha256(actions_file), "absent_action_groups": absent,
                "inference_seconds": elapsed, "action_shapes": {k: list(v.shape) for k, v in arrays.items()}}
        metadata["calls"].append(call)
        metadata["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        write_json(out / "report.json", metadata)
        print(json.dumps({"call_index": index, "actions": str(actions_file), "report": str(out / "report.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--shared-models", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-mode", choices=("legacy-ar", "checkpoint"), default="legacy-ar")
    serve(parser.parse_args())
