"""Audit published traces and generate a local gallery. Optionally run MuJoCo."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
from urllib.parse import quote
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
PIANO = ROOT.parents[1] / (
    "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/20-GPT6-Astra机器人弹琴与上下文工程"
)
METHODS = ("g05_only", "gpt_only", "g05_plus_gpt")
TITLES = {"piano": "双手弹琴", "fold": "叠衣服与桌面整理", "halloween": "万圣节收纳"}
LESSONS = {
    "piano": "01-双手弹琴.md",
    "fold": "02-叠衣服与桌面整理.md",
    "halloween": "03-万圣节收纳.md",
}
FULL_VIDEOS = {
    "piano": (
        ("小红书完整版", "https://www.xiaohongshu.com/explore/6ac373a8000000000a02209c?xsec_token=AB1VkPJCiZ9JR012UmZ5XPREYBX5v5UZafYfhVPVyAFQM=&xsec_source=pc_user"),
        ("B 站完整版", "https://www.bilibili.com/video/BV1FhHW6dERA/"),
    ),
    "fold": (
        ("小红书完整版", "https://www.xiaohongshu.com/explore/6ad2e41a0000000015014966?xsec_token=ABzE4Vr7EXL0p7MRPAA9JzcUswNHzKqMLni0Y0fK8A5L0=&xsec_source=pc_user"),
        ("B 站完整版", "https://www.bilibili.com/video/BV1HMJY61ENP/"),
    ),
    "halloween": (
        ("小红书完整版", "https://www.xiaohongshu.com/explore/6b019d8c000000001802e0b7?xsec_token=ABl4vfrbBHNES09U31x3CEc-ve4pQ3So1019yNsyF06oQ=&xsec_source=pc_user"),
        ("B 站完整版", "https://www.bilibili.com/video/BV1huyezgEAH/"),
    ),
}

def require(condition, description):
    if not condition:
        raise ValueError(description)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite_vector(values, length):
    return len(values) == length and all(
        type(value) in (int, float) and math.isfinite(value) for value in values
    )


def audit_fold():
    data = ROOT / "data/fold"
    cases = defaultdict(dict)
    with (data / "episodes.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row["phase"] != "formal_3x2x5":
            continue
        task, case, method = row["task"], row["case_id"], row["method"]
        score = float(row["native_score_0_1"])
        steps = int(row["native_control_steps"])
        require(0 <= score <= 1 and 0 < steps <= int(row["native_step_limit"]), case)
        require(row["native_success"] in ("True", "False"), case)
        require(method in METHODS and method not in cases[task, case], case)
        cases[task, case][method] = (score, row["native_success"] == "True")
    require(len(cases) == 10 and all(set(row) == set(METHODS) for row in cases.values()),
            "Expected ten complete matched cases")
    results = []
    for task in ("fold_clothes", "organize_table"):
        task_cases = [row for (name, _), row in cases.items() if name == task]
        require(len(task_cases) == 5, task)
        for method in METHODS:
            values = [row[method] for row in task_cases]
            results.append(dict(task=task, method=method, cases=len(values),
                                success=sum(v[1] for v in values),
                                mean_score=round(mean(v[0] for v in values), 6)))
    improvements = sum(row[METHODS[2]][0] > max(row[METHODS[0]][0], row[METHODS[1]][0])
                       for row in cases.values())

    with gzip.open(data / "fold-mixed-trace.json.gz", "rt", encoding="utf-8") as file:
        trace = json.load(file)
    steps = 0
    for chunk in trace["chunks"]:
        actions, states = chunk["actions"], chunk["states"]
        require(len(actions) == len(states) and 1 <= len(actions) <= 16, "Invalid trace chunk")
        require(all(finite_vector(row, 14) for row in actions + states), "Invalid trace row")
        steps += len(actions)
    require(steps == trace["control_steps"], "Trace length differs from recorded steps")
    row = cases["fold_clothes", trace["case_id"]]["g05_plus_gpt"]
    require(trace["native_score"] == row[0], "Trace case score differs from episode table")
    return dict(formal_episodes=30, debug_episodes=len(rows) - 30, matched_cases=10,
                mixed_strictly_higher=improvements, results=results,
                action_trace=dict(case_id=trace["case_id"], commands=steps,
                                  chunks=len(trace["chunks"])))


def audit_halloween():
    data = ROOT / "data/halloween"
    runs = read_json(data / "runs.json")
    outcomes = {row["method"]: row for row in read_json(data / "native-outcomes.json")}
    require({r["method"] for r in runs} == {"astra", "purple", "pink"} and len(runs) == 3,
            "Expected three Halloween trajectories")
    for row in runs:
        folder = data / row["method"]
        outcome = outcomes[row["method"]]
        require((outcome["instance_id"], outcome["steps"], outcome["q_score"]["final"])
                == (row["instance"], row["steps"], row["native_score"]) and outcome["success"],
                "Native Halloween outcome differs from the published summary")
        require(row["instance"] == 301 and 0 <= row["native_score"] <= 1, "Invalid episode summary")
        sequence = row["initial_sequence"]
        check = read_json(folder / "checkpoint.json")
        joints = read_json(folder / check["joint_state"])
        require(joints["sequence"] == check["sequence"] == sequence and
                finite_vector(joints["proprioception"], 65) and (folder / check["head_image"]).is_file(),
                "Missing initial replay reference")
        digest = hashlib.sha256()
        with gzip.open(folder / "actions.jsonl.gz", "rb") as file:
            for line in file:
                digest.update(line)
                step = json.loads(line)
                require(step["sequence_before"] == sequence and step["sequence_after"] == sequence + 1
                        and finite_vector(step["action"], 27),
                        f"Invalid {row['method']} command at sequence {sequence}")
                sequence += 1
        require(sequence == row["steps"], f"Unexpected final step for {row['method']}")
        require(digest.hexdigest() == row["action_stream_sha256"], "Action archive hash mismatch")
        row["recorded_commands"] = sequence - row["initial_sequence"]
    return runs


def piano_replay(output, interpreter, viewer):
    with ZipFile(PIANO / "assets/galaxea-replay.zip") as archive:
        archive.extractall(output)
    subprocess.run([interpreter, str(PIANO / "replay_windows.py"), "--run", str(output),
                    *(["--viewer"] if viewer else [])], check=True, stdout=subprocess.DEVNULL)
    result = read_json(output / "native_replay_report.json")
    return dict(measured_onsets=len(result["measured_onsets"]), model_calls=result["model_calls"],
                onset_sequence_matches=result["onset_sequence_matches"])


def asset_url(output, relative):
    path = (ROOT / relative).resolve()
    require(path.is_file(), f"Missing gallery asset {relative}")
    return quote(Path(os.path.relpath(path, output)).as_posix())


def render_gallery(output, report):
    media = read_json(ROOT / "data/media.json")
    blocks = []
    for project, title in TITLES.items():
        lesson = asset_url(output, LESSONS[project])
        published_links = " · ".join(
            f'<a href="{escape(url)}">{label}</a>' for label, url in FULL_VIDEOS[project]
        )
        blocks.append(f'<section id="{project}"><h2>{title}</h2>'
                      f'<p><a href="{lesson}">查看复现教程</a> · '
                      f'{published_links}</p>')
        for clip in [row for row in media if row["project"] == project]:
            source = asset_url(output, clip["video"])
            frames = []
            for frame in clip["frames"]:
                image = asset_url(output, frame["path"])
                time = f'{frame["time_seconds"]:g}'
                frames.append(f'<figure><a href="{image}"><img loading="lazy" src="{image}" '
                              f'alt="{escape(clip["title"])} {time} 秒"></a><figcaption>{time} 秒</figcaption></figure>')
            poster = asset_url(output, clip["frames"][0]["path"])
            blocks.append(f'<article><h3>{escape(clip["title"])}</h3>'
                          f'<video controls playsinline preload="none" src="{source}" poster="{poster}"></video>'
                          '<div class="frames">' + "".join(frames) + "</div></article>")
        blocks.append("</section>")
    blocks.append('<section id="metrics"><h2>复核结果</h2><p><a href="report.json">下载复核结果</a></p>')
    blocks.append('<table><caption>RoboDojo 同场景比较</caption><thead><tr><th>任务</th><th>方法</th>'
                  '<th>成功数</th><th>平均原生分</th></tr></thead><tbody>')
    for row in report["fold"]["results"]:
        blocks.append(f'<tr><td>{row["task"]}</td><td>{row["method"]}</td>'
                      f'<td>{row["success"]}/{row["cases"]}</td><td>{row["mean_score"]:.2f}</td></tr>')
    blocks.append('</tbody></table><table><caption>万圣节收纳实例 301</caption><thead><tr><th>方法</th>'
                  '<th>终局分</th><th>步数</th><th>模型调用</th></tr></thead><tbody>')
    for row in report["halloween"]:
        blocks.append(f'<tr><td>{row["method"]}</td><td>{row["native_score"]:.1f}</td>'
                      f'<td>{row["steps"]:,}</td><td>{row["g05_calls"]}</td></tr>')
    blocks.append('</tbody></table></section>')
    (output / "index.html").write_text("""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>星海图 G0.5 仿真复现</title>
<style>
body{max-width:1050px;margin:auto;padding:28px;background:#101925;color:#edf2f8;font:16px/1.65 system-ui,sans-serif}
a{color:#8bd3ff} nav{display:flex;gap:1rem;flex-wrap:wrap} section{margin:2.5rem 0} article{padding:1rem 0;border-top:1px solid #334155}
video{width:min(100%,720px);aspect-ratio:16/9;background:#050b12}.frames{display:grid;gap:10px;margin-top:1rem;grid-template-columns:repeat(auto-fit,minmax(145px,1fr))}
figure{margin:0}img{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#050b12}
figcaption{font-size:.85rem;color:#bdcbe0}table{border-collapse:collapse;margin:1rem 0;width:100%}caption{text-align:left;font-weight:bold}
td,th{text-align:left;padding:.35rem .5rem;border-bottom:1px solid #334155}
</style>
<header><h1>星海图 G0.5 仿真复现</h1>
<nav aria-label="专题章节"><a href="#piano">双手弹琴</a><a href="#fold">叠衣服与桌面整理</a>
<a href="#halloween">万圣节收纳</a><a href="#metrics">复核结果</a></nav></header><main>
""" + "\n".join(blocks) + "\n</main>\n", encoding="utf-8")
    return len(media), sum(len(clip["frames"]) for clip in media)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction-output")
    parser.add_argument("--piano", action="store_true", help="run the native MuJoCo replay")
    parser.add_argument("--piano-python", default=sys.executable, help="Python with MuJoCo 3.1.6")
    parser.add_argument("--viewer", action="store_true", help="show the native piano viewer")
    args = parser.parse_args()
    if args.viewer and not args.piano:
        parser.error("--viewer requires --piano")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(fold=audit_fold(), halloween=audit_halloween())
    if args.piano:
        (output / "piano").mkdir(exist_ok=True)
        report["piano"] = piano_replay(output / "piano", args.piano_python, args.viewer)
    videos, frames = render_gallery(output, report)
    report["media"] = {"playable_videos": videos, "screenshots": frames}
    write_json(output / "report.json", report)
    print(f"{output / 'index.html'}\n{output / 'report.json'}")


if __name__ == "__main__":
    main()
