"""Task 11 SmolVLA 4 seeds smoke — 复刻 notebook 核心流程（脚本方式，规避 kernel 崩溃）"""
import os, sys, json
os.environ["DISPLAY"] = ":99"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.system("touch /root/.Xauthority")
from pathlib import Path

TOPIC = Path(__file__).resolve().parents[1]  # code/ 的上一级 = 专题根
os.environ.setdefault("AMD_TOPIC_ROOT", str(TOPIC))
PROJ = TOPIC / "external" / "mujoco_pnp"
sys.path.insert(0, str(PROJ))

# 从 notebook 提取并执行 cell 1(路径) + cell 5(检查) + cell 6(核心函数)
nb = json.load(open(TOPIC / "notebooks" / "11_mujoco_closed_loop_deploy.ipynb"))
exec("".join(nb["cells"][1]["source"]), globals())   # TOPIC_ROOT / PROJECT_ROOT / DATA_ROOT / OUTPUT_ROOT
exec("".join(nb["cells"][2]["source"]), globals())   # md_table / show_image / show_video 辅助函数
exec("".join(nb["cells"][5]["source"]), globals())   # require_project_layout / dataset_report / ...
exec("".join(nb["cells"][6]["source"]), globals())   # find_pretrained_model / load_policy / strict_snapshot / run_closed_loop

require_project_layout()
print("PROJECT_ROOT =", PROJECT_ROOT)

# --- cell 8 配置（smolvla）---
POLICY_TYPE = "smolvla"
MODEL_RUN_DIR = TOPIC / "huggingface" / "smolvla" / "weights"
TASK_TEXT = "Place the blue mug on the plate."
EVAL_SEEDS = [1000, 1001, 1002, 1003]
MAX_ACTION_STEPS = 300
print("policy type =", POLICY_TYPE)
print("model run =", MODEL_RUN_DIR)
print("dataset =", DATA_ROOT / "omy_pnp_language")
print("seeds =", EVAL_SEEDS)

# --- cell 11 数据审计 + policy path ---
if (DATA_ROOT / "omy_pnp_language" / "meta" / "info.json").exists():
    dataset_report(DATA_ROOT / "omy_pnp_language")
POLICY_PATH = MODEL_RUN_DIR
print("policy path =", POLICY_PATH)

# --- cell 13 加载 + 闭环 ---
show_rocm_resources()
policy = load_policy(
    policy_type=POLICY_TYPE,
    policy_path=POLICY_PATH,
    dataset_repo_id="datawhale_eai_pnp_language",
    dataset_root=DATA_ROOT / "omy_pnp_language",
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
    render=False,  # 离屏渲染出视频帧；render=True 的 3D 窗口在 Xvfb 下第 3 次初始化会 segfault
)
physical = sum(r.get("physical_success", False) for r in results)
legacy = sum(r.get("legacy_success", False) for r in results)
print(f"\n=== FINAL: legacy={legacy}/{len(results)} physical={physical}/{len(results)} ===")
print("结果文件:", OUTPUT_ROOT / f"{POLICY_TYPE}_closed_loop" / "results.jsonl")
