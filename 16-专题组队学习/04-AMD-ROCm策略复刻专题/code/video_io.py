from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> str:
    """优先写 H.264；没有 imageio/ffmpeg 时回退到 OpenCV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("没有可写入的视频帧。")

    try:
        import imageio.v2 as imageio

        imageio.mimsave(
            path,
            list(frames),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )
        return "imageio/libx264"
    except Exception:
       pass

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "视频写出需要 imageio+imageio-ffmpeg 或 opencv-python。"
        ) from exc

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"视频帧必须是 HWC RGB 图像，实际为 {first.shape}")
    height, width = first.shape[:2]
    writer = None
    selected_codec = None
    for codec in ("avc1", "H264", "mp4v"):
        candidate = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            selected_codec = codec
            break
        candidate.release()
    if writer is None:
        raise RuntimeError("OpenCV 没有可用的 MP4 编码器。")

    try:
        for frame in frames:
            frame = np.asarray(frame)
            if frame.shape != first.shape:
                raise ValueError(
                    f"视频帧尺寸不一致：首帧 {first.shape}，当前 {frame.shape}"
                )
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return f"opencv/{selected_codec}"
