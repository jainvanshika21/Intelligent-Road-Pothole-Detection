def _detect_with_yolo(self, frame: np.ndarray) -> List[Detection]:
    results = self._model.predict(
        source=frame,
        conf=self.conf,
        imgsz=320,          # Keep 320 for faster inference
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
                    bbox=(x1, y1, x2, y2)
                )
            )

    return detections