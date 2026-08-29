#!/usr/bin/env python3
"""Generate small tutorial figures from ROCm reproduction outputs.

Pass the local experiment output root with --source-root when regenerating
contact sheets from videos or refreshing metrics from JSONL/TSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency for keyframes
    cv2 = None


HERE = Path(__file__).resolve().parent
TOPIC_ROOT = HERE.parent
ASSET_DIR = TOPIC_ROOT / "assets"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SMOLVLA_COMPARE_REL = (
    "smolvla_weighted_blue2_step500_forced_red_blue_20260629_1550/"
    "compare_baseline_weightedblue2_step500_step1000_blue15x_blue2x_blue3x.tsv"
)

SMOLVLA_FALLBACK = [
    ("baseline", 0.80, 0.00, "8/10", "0/10"),
    ("copy 1.5x", 0.30, 0.70, "3/10", "7/10"),
    ("copy 2x", 0.40, 0.80, "4/10", "8/10"),
    ("copy 3x", 0.20, 0.80, "2/10", "8/10"),
    ("weighted 1k", 0.60, 0.90, "6/10", "9/10"),
    ("weighted 500", 0.80, 1.00, "8/10", "10/10"),
]

ACT_STAGE_FILES = [
    (
        "clean closed-loop",
        [
            "act_clean_v2_gripper_bce05cont_eval_step5000/"
            "closedloop_trainish_seeds_1000_1004_physical.jsonl",
            "act_clean_v2_gripper_bce05cont_eval_step5000/"
            "closedloop_general_seeds_1030_1034_physical.jsonl",
        ],
        0,
        10,
    ),
    (
        "time offset",
        [
            "act_reset_oracle_v1/plus_prefix40_stable61_toffset2_eval_step5000/"
            "eval_seen_1000_1004_clampts.jsonl",
            "act_reset_oracle_v1/plus_prefix40_stable61_toffset2_eval_step5000/"
            "eval_mixed_1030_1034_clampts.jsonl",
            "act_reset_oracle_v1/plus_prefix40_stable61_toffset2_eval_step5000/"
            "eval_heldout_1080_1084_clampts.jsonl",
        ],
        3,
        15,
    ),
    (
        "downweight DAgger",
        [
            "act_reset_oracle_v1/downweight025_toffset2_stable61_eval/"
            "eval_step5000_expand10/eval_seen_1000_1009_clampts.jsonl",
            "act_reset_oracle_v1/downweight025_toffset2_stable61_eval/"
            "eval_step5000_expand10/eval_mixed_1030_1039_clampts.jsonl",
            "act_reset_oracle_v1/downweight025_toffset2_stable61_eval/"
            "eval_step5000_expand10/eval_heldout_1080_1089_clampts.jsonl",
        ],
        13,
        30,
    ),
    (
        "best DAgger",
        [
            "act_reset_oracle_v1/dagger_best025_eval_step5000_expand10/"
            "seen_1000_1009.jsonl",
            "act_reset_oracle_v1/dagger_best025_eval_step5000_expand10/"
            "mixed_1030_1039.jsonl",
            "act_reset_oracle_v1/dagger_best025_eval_step5000_expand10/"
            "heldout_1080_1089.jsonl",
        ],
        17,
        30,
    ),
]

VIDEO_SPECS = [
    (
        "smolvla_blue_failure_sequence.jpg",
        "SmolVLA baseline blue failure",
        "smolvla_step5000_blue_failure_video_20260629_101547/"
        "seed0_forced_blue_failure.mp4",
    ),
    (
        "smolvla_blue_success_sequence.jpg",
        "SmolVLA weighted blue success",
        "smolvla_weighted_blue2_step500_representative_videos_20260629_1626/"
        "videos/seed29_blue_success.mp4",
    ),
    (
        "act_failure_sequence.jpg",
        "ACT DAgger typical failure",
        "act_reset_oracle_v1/dagger_best025_representative_videos/seed1088.mp4",
    ),
    (
        "act_success_sequence.jpg",
        "ACT DAgger physical success",
        "act_reset_oracle_v1/dagger_best025_representative_videos/seed1089.mp4",
    ),
]

PI0_DIAGNOSTIC = {
    "raw": {
        "label": "pi0 raw",
        "success": 0,
        "total": 4,
        "xy_dist": 0.0087,
        "tcp_z": 0.8953,
        "max_lift": 0.0682,
        "upright": 1.0,
        "gripper": 0.0535,
    },
    "hybrid": {
        "label": "pi0 + finisher",
        "success": 4,
        "total": 4,
        "xy_dist": 0.0409,
        "tcp_z": 0.9457,
        "max_lift": 0.0948,
        "upright": 0.9741,
        "gripper": 0.0871,
    },
}

PI0_FULL20_RAW = {
    "success": 1,
    "total": 20,
    "ever_success": 3,
    "ever_total": 20,
    "success_episodes": [18],
    "ever_success_episodes": [4, 18, 19],
    "mean_action_mae": 0.0423,
    "mean_gripper_abs": 0.0991,
}

PI0_FULL20_FINISHER = {
    "success": 4,
    "total": 20,
    "ever_success": 4,
    "ever_total": 20,
    "success_episodes": [7, 9, 13, 14],
    "red_success": 2,
    "red_total": 10,
    "blue_success": 2,
    "blue_total": 10,
    "mean_action_mae": 0.0511,
    "mean_gripper_abs": 0.1145,
}

PI0_CLOSEDLOOP_RAW = {
    "success": 0,
    "total": 20,
    "legacy_success": 2,
    "legacy_total": 20,
    "legacy_success_episodes": [4, 18],
    "failure_buckets": {
        "not_upright_after_lift": 10,
        "no_enough_lift": 9,
        "strict_components_ok_but_legacy_false": 1,
    },
    "mean_xy_dist": 0.3382,
    "mean_max_lift": 0.0444,
}

PI0_TCPPLATE_SCAFFOLD = {
    "no_scripted_gripper_success": 5,
    "no_scripted_gripper_total": 10,
    "policy_prefix_success": 3,
    "policy_prefix_total": 10,
    "repeated_schedule_success": 21,
    "repeated_schedule_legacy_success": 23,
    "repeated_schedule_total": 30,
    "long_schedule_success": 30,
    "long_schedule_legacy_success": 30,
    "long_schedule_total": 30,
    "long_schedule_red_success": 18,
    "long_schedule_red_total": 18,
    "long_schedule_blue_success": 12,
    "long_schedule_blue_total": 12,
    "long_schedule_mean_xy": 0.0281,
    "long_schedule_max_xy": 0.0739,
    "phase_head_success": 30,
    "phase_head_legacy_success": 30,
    "phase_head_total": 30,
    "phase_head_red_success": 18,
    "phase_head_red_total": 18,
    "phase_head_blue_success": 12,
    "phase_head_blue_total": 12,
    "phase_head_mean_xy": 0.027151,
    "phase_head_max_xy": 0.073472,
    "adaptive_gate_success": 30,
    "adaptive_gate_legacy_success": 30,
    "adaptive_gate_total": 30,
    "adaptive_gate_red_success": 18,
    "adaptive_gate_blue_success": 12,
    "adaptive_gate_mean_xy": 0.028102,
    "adaptive_gate_max_xy": 0.061731,
    "transition_head_success": 30,
    "transition_head_legacy_success": 30,
    "transition_head_total": 30,
    "transition_head_red_success": 18,
    "transition_head_blue_success": 12,
    "transition_head_mean_xy": 0.027873,
    "transition_head_max_xy": 0.074125,
    "transition_head_release_mean": 146.633333,
    "transition_head_release_min": 126,
    "transition_head_release_max": 180,
    "transition_head_active_releases": 29,
    "transition_head_fallback_releases": 1,
    "short_schedule_episodes": [1, 7, 9],
    "short_move_preplace_frames": "72-75",
    "long_move_preplace_frames": "120-123",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size)


def percent_text(value: float) -> str:
    return f"{round(value * 100):.0f}%"


def count_physical_success(paths: Iterable[Path]) -> tuple[int, int] | None:
    success = 0
    total = 0
    found_any = False
    for path in paths:
        if not path.exists():
            continue
        found_any = True
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                if bool(row.get("physical_success")):
                    success += 1
    if not found_any or total == 0:
        return None
    return success, total


def load_act_stages(source_root: Path | None) -> list[dict]:
    stages = []
    for label, rels, fallback_success, fallback_total in ACT_STAGE_FILES:
        measured = None
        if source_root is not None:
            measured = count_physical_success(source_root / rel for rel in rels)
        success, total = measured or (fallback_success, fallback_total)
        stages.append(
            {
                "label": label,
                "success": success,
                "total": total,
                "rate": success / total if total else 0.0,
            }
        )
    return stages


def load_smolvla_compare(source_root: Path | None) -> list[dict]:
    compare_path = source_root / SMOLVLA_COMPARE_REL if source_root else None
    if compare_path is None or not compare_path.exists():
        return [
            {
                "label": label,
                "red_rate": red,
                "blue_rate": blue,
                "red_text": red_text,
                "blue_text": blue_text,
            }
            for label, red, blue, red_text, blue_text in SMOLVLA_FALLBACK
        ]

    label_map = {
        "baseline_step5000": "baseline",
        "blue15x_copy": "copy 1.5x",
        "blue2x_copy": "copy 2x",
        "blue3x_copy": "copy 3x",
        "weighted_blue2_step1000": "weighted 1k",
        "weighted_blue2_step500": "weighted 500",
    }
    grouped: dict[str, dict] = {}
    with compare_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            variant = row["variant"]
            label = label_map.get(variant, variant)
            entry = grouped.setdefault("label:" + label, {"label": label})
            color = row["color"]
            entry[f"{color}_rate"] = float(row["physical_success_rate"])
            entry[f"{color}_text"] = row["physical_success"]

    order = [item[0] for item in SMOLVLA_FALLBACK]
    return sorted(grouped.values(), key=lambda x: order.index(x["label"]))


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    axis = "#39434d"
    grid = "#d8dee6"
    font = load_font(18)
    for i in range(6):
        value = i / 5
        y = y1 - int((y1 - y0) * value)
        draw.line((x0, y, x1, y), fill=grid, width=1)
        draw.text((x0 - 62, y - 11), percent_text(value), fill=axis, font=font)
    draw.line((x0, y0, x0, y1), fill=axis, width=2)
    draw.line((x0, y1, x1, y1), fill=axis, width=2)


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((52, 30), title, fill="#18202a", font=load_font(34, bold=True))
    draw.text((52, 76), subtitle, fill="#55606d", font=load_font(20))


def save_grouped_bars(data: list[dict], out_path: Path) -> None:
    width, height = 1280, 760
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "SmolVLA red/blue forced-instruction success",
        "physical_success; n=10 per color for each variant",
    )

    chart = (118, 150, 1220, 600)
    draw_axes(draw, chart)
    x0, y0, x1, y1 = chart
    n = len(data)
    group_w = (x1 - x0) / n
    bar_w = 46
    red = "#d84a4a"
    blue = "#2878d7"
    text_font = load_font(18)
    label_font = load_font(17)

    for i, row in enumerate(data):
        center = x0 + group_w * (i + 0.5)
        for offset, color, key, text_key in [
            (-bar_w * 0.62, red, "red_rate", "red_text"),
            (bar_w * 0.62, blue, "blue_rate", "blue_text"),
        ]:
            value = row[key]
            bx0 = int(center + offset - bar_w / 2)
            bx1 = int(center + offset + bar_w / 2)
            by1 = y1
            by0 = y1 - int((y1 - y0) * value)
            draw.rounded_rectangle((bx0, by0, bx1, by1), radius=4, fill=color)
            draw.text(
                (bx0 - 8, by0 - 28),
                row.get(text_key, percent_text(value)),
                fill="#18202a",
                font=text_font,
            )

        label = row["label"]
        label_w = draw.textlength(label, font=label_font)
        draw.text((center - label_w / 2, y1 + 20), label, fill="#26313d", font=label_font)

    legend_y = 646
    draw.rounded_rectangle((118, legend_y, 145, legend_y + 18), radius=4, fill=red)
    draw.text((154, legend_y - 3), "red mug", fill="#26313d", font=load_font(18))
    draw.rounded_rectangle((260, legend_y, 287, legend_y + 18), radius=4, fill=blue)
    draw.text((296, legend_y - 3), "blue mug", fill="#26313d", font=load_font(18))
    draw.text(
        (52, 705),
        "Figure: weighted sampling recovers blue-mug performance without fully sacrificing red-mug performance.",
        fill="#55606d",
        font=load_font(18),
    )
    img.save(out_path)


def save_act_progress(stages: list[dict], out_path: Path) -> None:
    width, height = 1180, 720
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "ACT closed-loop physical-success progress",
        "same physical_success criterion; stages summarize iterative fixes",
    )
    chart = (120, 145, 1080, 560)
    draw_axes(draw, chart)
    x0, y0, x1, y1 = chart
    max_rate = 1.0
    color = "#20866d"
    point_fill = "#f4a340"
    line_pts = []
    font = load_font(18)
    label_font = load_font(17)

    for i, row in enumerate(stages):
        x = x0 + int((x1 - x0) * (i / max(1, len(stages) - 1)))
        y = y1 - int((y1 - y0) * (row["rate"] / max_rate))
        line_pts.append((x, y))
    draw.line(line_pts, fill=color, width=5, joint="curve")

    for i, (x, y) in enumerate(line_pts):
        row = stages[i]
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=point_fill, outline="#18202a", width=2)
        label = f'{row["success"]}/{row["total"]} ({percent_text(row["rate"])})'
        draw.text((x - 44, y - 44), label, fill="#18202a", font=font)
        parts = row["label"].split()
        y_text = y1 + 22
        for part in parts:
            part_w = draw.textlength(part, font=label_font)
            draw.text((x - part_w / 2, y_text), part, fill="#26313d", font=label_font)
            y_text += 22

    draw.text(
        (52, 668),
        "Figure: DAgger-style correction improves ACT, but the task still needs video-level physical review.",
        fill="#55606d",
        font=load_font(18),
    )
    img.save(out_path)


def save_best_summary(act_stages: list[dict], out_path: Path) -> None:
    rows = [
        ("ACT best DAgger", act_stages[-1]["success"], act_stages[-1]["total"], "#20866d"),
        ("SmolVLA weighted 500", 53, 60, "#2878d7"),
        ("pi0 raw closed20", PI0_CLOSEDLOOP_RAW["success"], PI0_CLOSEDLOOP_RAW["total"], "#d76445"),
        ("pi0 raw fixed scene", 0, 12, "#b54a5e"),
        ("pi0 learned head fixed", 6, 12, "#7b5fb5"),
        ("pi0 learned head random", 1, 4, "#d08a28"),
        ("pi0 scaffold fixed env", 30, 30, "#7b858f"),
    ]
    width, height = 1500, 780
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "Current reproducibility status",
        "strict physical_success; raw policies and scaffolded diagnostics are separated",
    )
    chart = (120, 180, 1380, 570)
    draw_axes(draw, chart)
    x0, y0, x1, y1 = chart
    bar_w = 110
    font = load_font(20)
    label_font = load_font(18)

    for i, (label, success, total, color) in enumerate(rows):
        rate = success / total
        cx = x0 + int((x1 - x0) * (i + 1) / (len(rows) + 1))
        by0 = y1 - int((y1 - y0) * rate)
        draw.rounded_rectangle((cx - bar_w // 2, by0, cx + bar_w // 2, y1), radius=5, fill=color)
        draw.text((cx - 44, by0 - 34), f"{success}/{total}", fill="#18202a", font=font)
        draw.text((cx - 40, by0 - 62), percent_text(rate), fill="#18202a", font=font)
        for j, part in enumerate(label.split()):
            part_w = draw.textlength(part, font=label_font)
            draw.text((cx - part_w / 2, y1 + 24 + j * 24), part, fill="#26313d", font=label_font)

    draw.text(
        (52, 705),
        "Figure: old scaffold 30/30 used a fixed environment; only the 1/4 random-env gate tests position changes.",
        fill="#55606d",
        font=load_font(18),
    )
    img.save(out_path)


def save_pi0_diagnostic(out_path: Path) -> None:
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "pi0 terminal-stage diagnostic",
        "episode2 red mug; raw policy is close but misses strict physical_success",
    )

    chart = (120, 160, 1160, 500)
    draw_axes(draw, chart)
    x0, y0, x1, y1 = chart
    metrics = [
        ("xy dist", "xy_dist", 0.10, "< 0.10"),
        ("tcp z", "tcp_z", 0.95, "> 0.90"),
        ("max lift", "max_lift", 0.12, "lifted"),
        ("upright", "upright", 1.0, ">= 0.7"),
        ("gripper", "gripper", 0.10, "< 0.10"),
    ]
    colors = {"raw": "#d76445", "hybrid": "#4f9f2e"}
    bar_w = 48
    label_font = load_font(17)
    value_font = load_font(17)
    group_w = (x1 - x0) / len(metrics)

    for i, (label, key, scale_max, hint) in enumerate(metrics):
        center = x0 + group_w * (i + 0.5)
        for offset, variant in [(-32, "raw"), (32, "hybrid")]:
            row = PI0_DIAGNOSTIC[variant]
            value = float(row[key])
            h = int(min(value / scale_max, 1.0) * (y1 - y0))
            bx0 = int(center + offset - bar_w / 2)
            bx1 = int(center + offset + bar_w / 2)
            by0 = y1 - h
            draw.rounded_rectangle((bx0, by0, bx1, y1), radius=4, fill=colors[variant])
            draw.text((bx0 - 9, by0 - 26), f"{value:.3f}", fill="#18202a", font=value_font)
        label_w = draw.textlength(label, font=label_font)
        draw.text((center - label_w / 2, y1 + 20), label, fill="#26313d", font=label_font)
        hint_w = draw.textlength(hint, font=label_font)
        draw.text((center - hint_w / 2, y1 + 44), hint, fill="#66717f", font=label_font)

    legend_y = 592
    draw.rounded_rectangle((120, legend_y, 150, legend_y + 20), radius=4, fill=colors["raw"])
    draw.text((162, legend_y - 2), "small-set raw reference: 0/4 final physical_success", fill="#26313d", font=load_font(18))
    draw.rounded_rectangle((120, legend_y + 34, 150, legend_y + 54), radius=4, fill=colors["hybrid"])
    draw.text((162, legend_y + 32), "small-set finisher: 4/4; full20 template-tail: 4/20", fill="#26313d", font=load_font(18))
    draw.text(
        (52, 678),
        "Figure: the finisher validates a release/raise/stabilize bottleneck; it should not be reported as raw pi0 success.",
        fill="#55606d",
        font=load_font(18),
    )
    img.save(out_path)


def placeholder_sheet(title: str, out_path: Path, reason: str) -> None:
    width, height = 1280, 350
    img = Image.new("RGB", (width, height), "#eef1f5")
    draw = ImageDraw.Draw(img)
    draw.text((40, 42), title, fill="#18202a", font=load_font(32, bold=True))
    draw.text((40, 96), reason, fill="#55606d", font=load_font(22))
    draw.text(
        (40, 255),
        "Regenerate with: python code/generate_tutorial_assets.py --source-root /path/to/outputs",
        fill="#55606d",
        font=load_font(20),
    )
    img.save(out_path)


def sample_video_frames(video_path: Path, title: str, out_path: Path) -> None:
    if cv2 is None:
        placeholder_sheet(title, out_path, "OpenCV is not installed; video keyframes were not extracted.")
        return
    if not video_path.exists():
        placeholder_sheet(title, out_path, "Source video is unavailable in this checkout.")
        return

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if total <= 1:
        cap.release()
        placeholder_sheet(title, out_path, "Source video did not expose readable frames.")
        return

    ratios = [0.05, 0.25, 0.50, 0.75, 0.95]
    thumbs = []
    for ratio in ratios:
        idx = min(total - 1, max(0, int((total - 1) * ratio)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        thumb_h = 210
        thumb_w = int(pil.width * thumb_h / pil.height)
        pil = pil.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        thumbs.append((idx / fps, pil))
    cap.release()

    if not thumbs:
        placeholder_sheet(title, out_path, "No keyframes could be decoded from the source video.")
        return

    gap = 18
    margin = 36
    title_h = 78
    label_h = 42
    content_w = sum(t.width for _, t in thumbs) + gap * (len(thumbs) - 1)
    width = max(1180, content_w + margin * 2)
    height = title_h + 210 + label_h + 28
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 24), title, fill="#18202a", font=load_font(28, bold=True))

    x = margin
    y = title_h
    for seconds, thumb in thumbs:
        img.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb.width, y + thumb.height), outline="#c5ccd6", width=2)
        label = f"t={seconds:.1f}s"
        label_w = draw.textlength(label, font=load_font(18))
        draw.text((x + thumb.width / 2 - label_w / 2, y + thumb.height + 10), label, fill="#26313d", font=load_font(18))
        x += thumb.width + gap
    img.save(out_path)


def save_training_progress(snapshot_path: Path, out_path: Path) -> None:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    records = snapshot["records"]
    width, height = 1600, 720
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "Measured ROCm training checkpoints",
        "Historical runs recorded on the AMD device; progress bars show completed training steps",
    )

    colors = ["#20866d", "#2878d7", "#d76445", "#7b5fb5"]
    label_font = load_font(21, bold=True)
    text_font = load_font(19)
    y = 155
    for index, row in enumerate(records):
        ratio = min(1.0, row["current_step"] / max(1, row["total_steps"]))
        draw.text((70, y), row["model"], fill="#18202a", font=label_font)
        draw.text((285, y), row["stage"], fill="#55606d", font=text_font)
        bar = (70, y + 39, 900, y + 67)
        draw.rounded_rectangle(bar, radius=5, fill="#dfe4ea")
        draw.rounded_rectangle(
            (bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * ratio), bar[3]),
            radius=5,
            fill=colors[index % len(colors)],
        )
        step_text = row.get("progress", f'{row["current_step"]}/{row["total_steps"]} steps')
        draw.text((930, y + 36), step_text, fill="#26313d", font=text_font)
        draw.text(
            (1320, y + 36),
            row["physical_success"],
            fill=colors[index % len(colors)],
            font=label_font,
        )
        y += 122

    draw.text(
        (70, 655),
        "Training completion is not task success: every checkpoint is evaluated again in MuJoCo closed loop.",
        fill="#55606d",
        font=load_font(19),
    )
    img.save(out_path)


def save_pi0_strict_input_progress(snapshot_path: Path, out_path: Path) -> None:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    fixed = snapshot["fixed_scene"]
    randomized = snapshot["randomized_environment_gate"]
    rows = [
        (
            "raw pi0 / fixed env",
            fixed["raw_pi0"]["success"],
            fixed["raw_pi0"]["total"],
            "#d76445",
        ),
        (
            "visual-history head / fixed env",
            fixed["pi0_visual_history_gru"]["success"],
            fixed["pi0_visual_history_gru"]["total"],
            "#20866d",
        ),
        (
            "visual-history head / random env",
            randomized["pi0_visual_history_gru"]["success"],
            randomized["pi0_visual_history_gru"]["total"],
            "#2878d7",
        ),
    ]
    width, height = 1260, 740
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "pi0 strict-input closed-loop audit",
        "Final strict predicate includes lift, upright, placement height, plate motion and 5-step stability",
    )
    chart = (120, 165, 1140, 560)
    draw_axes(draw, chart)
    x0, y0, x1, y1 = chart
    label_font = load_font(18)
    value_font = load_font(22, bold=True)
    for index, (label, success, total, color) in enumerate(rows):
        rate = success / total
        center = x0 + int((x1 - x0) * (index + 1) / (len(rows) + 1))
        bar_w = 155
        top = y1 - int((y1 - y0) * rate)
        draw.rounded_rectangle((center - bar_w // 2, top, center + bar_w // 2, y1), radius=5, fill=color)
        value = f"{success}/{total} ({percent_text(rate)})"
        value_w = draw.textlength(value, font=value_font)
        draw.text((center - value_w / 2, top - 38), value, fill="#18202a", font=value_font)
        parts = label.split(" / ")
        for line_index, part in enumerate(parts):
            part_w = draw.textlength(part, font=label_font)
            draw.text((center - part_w / 2, y1 + 22 + line_index * 25), part, fill="#26313d", font=label_font)

    draw.text(
        (58, 678),
        "Fixed-scene gains are real, but 1/4 on corrected random environments is not spatial generalization.",
        fill="#55606d",
        font=load_font(19),
    )
    img.save(out_path)


def write_metrics_snapshot(smolvla: list[dict], act_stages: list[dict], out_path: Path) -> None:
    snapshot = {
        "smolvla_forced_instruction_n10": smolvla,
        "act_progress": act_stages,
        "pi0_diagnostic_balanced_2b2r": {
            "raw_final_physical_success": "0/4",
            "raw_batch_rerun_final_physical_success": "1/4",
            "scripted_finisher_final_physical_success": "4/4",
            "full20_template_tail_final_physical_success": "4/20",
            "full20_template_tail_physical_success_ever": "4/20",
            "full20_template_tail_success_episodes": PI0_FULL20_FINISHER["success_episodes"],
            "full20_template_tail_red_success": "2/10",
            "full20_template_tail_blue_success": "2/10",
            "full20_template_tail_mean_action_mae": PI0_FULL20_FINISHER["mean_action_mae"],
            "full20_template_tail_mean_gripper_abs": PI0_FULL20_FINISHER["mean_gripper_abs"],
            "full20_openloop_raw_final_physical_success": "1/20",
            "full20_openloop_raw_physical_success_ever": "3/20",
            "full20_openloop_raw_success_episodes": [18],
            "full20_openloop_raw_ever_success_episodes": [4, 18, 19],
            "full20_openloop_raw_mean_action_mae": PI0_FULL20_RAW["mean_action_mae"],
            "full20_openloop_raw_mean_gripper_abs": PI0_FULL20_RAW["mean_gripper_abs"],
            "full20_closedloop_raw_physical_success": "0/20",
            "full20_closedloop_raw_legacy_success": "2/20",
            "full20_closedloop_raw_legacy_success_episodes": PI0_CLOSEDLOOP_RAW["legacy_success_episodes"],
            "full20_closedloop_raw_failure_buckets": PI0_CLOSEDLOOP_RAW["failure_buckets"],
            "full20_closedloop_raw_mean_xy_dist": PI0_CLOSEDLOOP_RAW["mean_xy_dist"],
            "full20_closedloop_raw_mean_max_lift": PI0_CLOSEDLOOP_RAW["mean_max_lift"],
            "tcpplate_scaffold_no_scripted_gripper_physical_success": "5/10",
            "tcpplate_scaffold_policy_prefix_scripted_gripper_physical_success": "3/10",
            "tcpplate_scaffold_repeated_schedule_physical_success": "21/30",
            "tcpplate_scaffold_repeated_schedule_legacy_success": "23/30",
            "tcpplate_scaffold_long_schedule_physical_success": "30/30",
            "tcpplate_scaffold_long_schedule_legacy_success": "30/30",
            "tcpplate_scaffold_long_schedule_red_success": "18/18",
            "tcpplate_scaffold_long_schedule_blue_success": "12/12",
            "tcpplate_scaffold_long_schedule_mean_xy": PI0_TCPPLATE_SCAFFOLD["long_schedule_mean_xy"],
            "tcpplate_scaffold_long_schedule_max_xy": PI0_TCPPLATE_SCAFFOLD["long_schedule_max_xy"],
            "tcpplate_scaffold_phase_head_physical_success": "30/30",
            "tcpplate_scaffold_phase_head_legacy_success": "30/30",
            "tcpplate_scaffold_phase_head_red_success": "18/18",
            "tcpplate_scaffold_phase_head_blue_success": "12/12",
            "tcpplate_scaffold_phase_head_mean_xy": PI0_TCPPLATE_SCAFFOLD["phase_head_mean_xy"],
            "tcpplate_scaffold_phase_head_max_xy": PI0_TCPPLATE_SCAFFOLD["phase_head_max_xy"],
            "tcpplate_scaffold_adaptive_gate_physical_success": "30/30",
            "tcpplate_scaffold_adaptive_gate_legacy_success": "30/30",
            "tcpplate_scaffold_adaptive_gate_red_success": "18/18",
            "tcpplate_scaffold_adaptive_gate_blue_success": "12/12",
            "tcpplate_scaffold_adaptive_gate_mean_xy": PI0_TCPPLATE_SCAFFOLD["adaptive_gate_mean_xy"],
            "tcpplate_scaffold_adaptive_gate_max_xy": PI0_TCPPLATE_SCAFFOLD["adaptive_gate_max_xy"],
            "tcpplate_scaffold_adaptive_gate_threshold": "xy=0.05m,min_steps=20,max_steps=180",
            "tcpplate_scaffold_adaptive_gate_counterexamples": {
                "xy0p09_full30": "29/30, fail seed1034, max xy 3.161459m",
                "xy0p08_partial": "8/9 strict before stopped, fail seed1012",
            },
            "tcpplate_scaffold_transition_head_physical_success": "30/30",
            "tcpplate_scaffold_transition_head_legacy_success": "30/30",
            "tcpplate_scaffold_transition_head_red_success": "18/18",
            "tcpplate_scaffold_transition_head_blue_success": "12/12",
            "tcpplate_scaffold_transition_head_mean_xy": PI0_TCPPLATE_SCAFFOLD["transition_head_mean_xy"],
            "tcpplate_scaffold_transition_head_max_xy": PI0_TCPPLATE_SCAFFOLD["transition_head_max_xy"],
            "tcpplate_scaffold_transition_head_release_mean": PI0_TCPPLATE_SCAFFOLD["transition_head_release_mean"],
            "tcpplate_scaffold_transition_head_release_range": [
                PI0_TCPPLATE_SCAFFOLD["transition_head_release_min"],
                PI0_TCPPLATE_SCAFFOLD["transition_head_release_max"],
            ],
            "tcpplate_scaffold_transition_head_active_releases": "29/30",
            "tcpplate_scaffold_transition_head_fallback_releases": "1/30",
            "tcpplate_scaffold_transition_head_feature_mode": "xy_step",
            "tcpplate_scaffold_short_schedule_episodes": PI0_TCPPLATE_SCAFFOLD["short_schedule_episodes"],
            "tcpplate_scaffold_short_move_preplace_frames": PI0_TCPPLATE_SCAFFOLD["short_move_preplace_frames"],
            "tcpplate_scaffold_long_move_preplace_frames": PI0_TCPPLATE_SCAFFOLD["long_move_preplace_frames"],
            "note": "pi0 + finisher is a diagnostic hybrid result, not pure pi0 raw success; the old 30-seed panels changed policy sampling but the environment stayed at seed 0, so they are fixed-scene stability evidence rather than spatial generalization",
        },
        "best_summary": {
            "act_best_dagger": {"physical_success": "17/30"},
            "smolvla_weighted_blue2_step500_expand30": {"physical_success": "53/60"},
            "pi0_raw_full20_closedloop": {"physical_success": "0/20"},
            "pi0_raw_full20_openloop": {"physical_success": "1/20"},
            "pi0_raw_balanced_2b2r_reference": {"physical_success": "0/4"},
            "pi0_raw_balanced_2b2r_batch_rerun": {"physical_success": "1/4"},
            "pi0_scripted_finisher_balanced_2b2r": {"physical_success": "4/4"},
            "pi0_template_tail_full20": {"physical_success": "4/20"},
            "pi0_tcpplate_scaffold_fixed_env_policy_seeds30_long_schedule": {"physical_success": "30/30"},
            "pi0_tcpplate_scaffold_fixed_env_phase_head_seeds30": {"physical_success": "30/30"},
            "pi0_tcpplate_scaffold_fixed_env_adaptive_gate_seeds30": {"physical_success": "30/30"},
            "pi0_tcpplate_scaffold_fixed_env_transition_head_seeds30": {"physical_success": "30/30"},
            "pi0_visual_history_gru_fixed_env": {"physical_success": "6/12"},
            "pi0_visual_history_gru_random_env_gate": {"physical_success": "1/4"},
        },
    }
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Local experiment outputs root. Omit it to regenerate charts from the embedded redacted metrics.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ASSET_DIR,
        help="Directory for generated tutorial assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_dir = args.asset_dir
    asset_dir.mkdir(parents=True, exist_ok=True)

    source_root = args.source_root.resolve() if args.source_root else None
    smolvla = load_smolvla_compare(source_root)
    act_stages = load_act_stages(source_root)

    save_grouped_bars(smolvla, asset_dir / "smolvla_red_blue_success.png")
    save_act_progress(act_stages, asset_dir / "act_dagger_progress_curve.png")
    save_best_summary(act_stages, asset_dir / "model_status_summary.png")
    save_pi0_diagnostic(asset_dir / "pi0_raw_vs_finisher_diagnostic.png")
    save_training_progress(
        asset_dir / "training_progress_snapshot.json",
        asset_dir / "training_progress_overview.png",
    )
    save_pi0_strict_input_progress(
        asset_dir / "pi0_strict_input_results.json",
        asset_dir / "pi0_strict_input_progress.png",
    )
    write_metrics_snapshot(smolvla, act_stages, asset_dir / "metrics_snapshot.json")

    for filename, title, rel_path in VIDEO_SPECS:
        video_path = source_root / rel_path if source_root else Path(rel_path)
        sample_video_frames(video_path, title, asset_dir / filename)

    sample_video_frames(
        asset_dir / "pnp_four_view_strict_success.mp4",
        "MuJoCo four-view strict success (pi0 + learned visual/history head)",
        asset_dir / "pnp_four_view_strict_success_sequence.jpg",
    )


if __name__ == "__main__":
    main()
