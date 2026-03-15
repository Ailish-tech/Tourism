"""
YOLOv11 Person Detector Module
Uses Ultralytics YOLOv11 to detect persons with bounding boxes.
"""
import numpy as np
from ultralytics import YOLO
import config


class YOLODetector:
    """YOLOv11-based person detector."""

    def __init__(self):
        print("[YOLO] Loading YOLOv11 model...")
        self.model = YOLO(config.YOLO_MODEL)
        self.confidence = config.YOLO_CONFIDENCE
        self.iou_threshold = config.YOLO_IOU_THRESHOLD
        self.person_class = config.YOLO_PERSON_CLASS
        self.img_size = config.YOLO_IMG_SIZE

        # Determine device
        if config.USE_GPU:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        print(f"[YOLO] Model loaded on {self.device}")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect persons in a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict with keys:
                - boxes: list of [x1, y1, x2, y2] bounding boxes
                - scores: list of confidence scores
                - centers: list of [cx, cy] center points
                - count: number of detected persons
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[self.person_class],
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        boxes = []
        scores = []
        centers = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())

                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    scores.append(conf)
                    centers.append([
                        int((x1 + x2) / 2),
                        int((y1 + y2) / 2)
                    ])

        return {
            "boxes": boxes,
            "scores": scores,
            "centers": centers,
            "count": len(boxes),
        }
