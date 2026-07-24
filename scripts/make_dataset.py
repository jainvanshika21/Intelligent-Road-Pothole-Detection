from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import PotholeDetector


def yolo_bbox(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=10, help="sample every Nth frame")
    ap.add_argument("--max", type=int, default=400, help="max frames to export")
    ap.add_argument("--val", type=float, default=0.2, help="val split")
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--max-area-ratio", type=float, default=0.5)
    args = ap.parse_args()

    video_path = Path(args.video)
    out = Path(args.out)
    img_train = out / "images" / "train"
    img_val = out / "images" / "val"
    lbl_train = out / "labels" / "train"
    lbl_val = out / "labels" / "val"
    for p in (img_train, img_val, lbl_train, lbl_val):
        p.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detector = PotholeDetector(
        model_path=None,
        conf=0.25,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        blur_ksize=5,
    )

    exported = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.every != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        dets = detector.detect(frame)
        if not dets:
            frame_idx += 1
            continue

        split = "val" if random.random() < args.val else "train"
        img_out = (img_val if split == "val" else img_train) / f"frame_{frame_idx:06d}.jpg"
        lbl_out = (lbl_val if split == "val" else lbl_train) / f"frame_{frame_idx:06d}.txt"

        cv2.imwrite(str(img_out), frame)
        with open(lbl_out, "w", encoding="utf-8") as f:
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                cx, cy, bw, bh = yolo_bbox(x1, y1, x2, y2, w, h)
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        exported += 1
        frame_idx += 1
        if exported >= args.max:
            break

    cap.release()
    print(f"exported={exported} total_frames={total}")


if __name__ == "__main__":
    main()
