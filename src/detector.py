from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class PotholeDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.25,
        min_area: int = 300,
        max_area_ratio: float = 0.5,
        blur_ksize: int = 5,
        use_reflective: bool = True,
        sat_thresh: int = 40,
        val_thresh: int = 200,
        texture_thresh: float = 8.0,
    ):
        self.model_path = model_path
        self.conf = conf
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.blur_ksize = blur_ksize
        self.use_reflective = use_reflective
        self.sat_thresh = sat_thresh
        self.val_thresh = val_thresh
        self.texture_thresh = texture_thresh

        self._model = None

        if model_path:
            try:
                from ultralytics import YOLO
                self._model = YOLO(model_path)
            except Exception:
                self._model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._model is not None:
            return self._detect_with_yolo(frame)
        return self._detect_with_heuristic(frame)

    def _detect_with_yolo(self, frame: np.ndarray) -> List[Detection]:

        results = self._model.predict(
            source=frame,
            conf=self.conf,
            imgsz=320,
            device="cpu",
            verbose=False,
            stream=False
        )

        detections: List[Detection] = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = self._model.names.get(cls_id, "pothole")

                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections

    def _detect_with_heuristic(self, frame: np.ndarray) -> List[Detection]:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        k = max(3, int(self.blur_ksize) | 1)

        blur = cv2.GaussianBlur(gray, (k, k), 0)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(blur)

        dark = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            5,
        )

        mask = dark

        if self.use_reflective:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            h, s, v = cv2.split(hsv)

            reflective = cv2.inRange(
                s,
                0,
                int(self.sat_thresh)
            ) & cv2.inRange(
                v,
                int(self.val_thresh),
                255,
            )

            mask = cv2.bitwise_or(mask, reflective)

        lap = cv2.Laplacian(
            enhanced,
            cv2.CV_64F
        )

        tex = np.uint8(
            np.clip(np.abs(lap) * 2, 0, 255)
        )

        _, tex_mask = cv2.threshold(
            tex,
            self.texture_thresh,
            255,
            cv2.THRESH_BINARY,
        )

        mask = cv2.bitwise_and(mask, tex_mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        h, w = gray.shape[:2]

        frame_area = float(h * w)

        detections: List[Detection] = []

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if (
                area < self.min_area
                or area > frame_area * self.max_area_ratio
            ):
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            aspect = cw / max(1, ch)

            if aspect < 0.15 or aspect > 6.0:
                continue

            confidence = min(
                0.95,
                area / (frame_area * 0.05),
            )

            detections.append(
                Detection(
                    label="pothole",
                    confidence=confidence,
                    bbox=(x, y, x + cw, y + ch),
                )
            )

        return detections


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:

    output = frame.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        label = (
            f"{detection.label} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return output