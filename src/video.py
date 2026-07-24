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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return VideoMeta(
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height
    )


# NEW FUNCTION
def open_video(video_path: Path):
    return cv2.VideoCapture(str(video_path))


def read_frame(cap):
    return cap.read()