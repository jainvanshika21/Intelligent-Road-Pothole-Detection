from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoMeta:
    fps: float
    total_frames: int
    width: int
    height: int


def get_video_meta(video_path: Path) -> VideoMeta:

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return VideoMeta(
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height
    )


def open_video(video_path: Path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    # Reduce internal buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def read_frame(cap):

    if cap is None or not cap.isOpened():
        return False, None

    return cap.read()