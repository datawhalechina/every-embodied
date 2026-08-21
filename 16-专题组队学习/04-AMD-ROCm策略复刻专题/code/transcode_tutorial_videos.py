#!/usr/bin/env python3
"""Normalize tutorial MP4 files to a browser-friendly H.264 profile."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_ASSET_DIR = HERE.parent / "assets"


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} was not found on PATH")
    return path


def video_codec(ffprobe: str, path: Path) -> str:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def transcode(ffmpeg: str, ffprobe: str, source: Path) -> None:
    codec = video_codec(ffprobe, source)
    if codec == "h264":
        print(f"OK       {source.name} (h264)")
        return

    temporary = source.with_name(source.stem + ".h264.tmp.mp4")
    if temporary.exists():
        temporary.unlink()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        if video_codec(ffprobe, temporary) != "h264":
            raise RuntimeError(f"transcoded file is not H.264: {temporary}")
        temporary.replace(source)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"CONVERT  {source.name} ({codec} -> h264)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tutorial MP4 files to H.264 for Jupyter/browser playback."
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Directory containing tutorial MP4 files.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report codecs without changing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_dir = args.asset_dir.expanduser().resolve()
    if not asset_dir.is_dir():
        raise SystemExit(f"asset directory does not exist: {asset_dir}")

    ffprobe = executable("ffprobe")
    ffmpeg = executable("ffmpeg")
    videos = sorted(asset_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no MP4 files found in {asset_dir}")

    for video in videos:
        codec = video_codec(ffprobe, video)
        if args.check_only:
            status = "OK" if codec == "h264" else "NEEDS TRANSCODE"
            print(f"{status:15} {video.name} ({codec})")
        else:
            transcode(ffmpeg, ffprobe, video)


if __name__ == "__main__":
    main()
