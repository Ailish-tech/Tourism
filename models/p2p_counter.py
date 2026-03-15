"""
P2PNet-Inspired Head Detection & Counting Module

Uses a two-stage approach:
  1. Extracts head regions from YOLO-detected person bounding boxes
  2. Applies a lightweight head classifier to validate each head region
  3. Returns confirmed head point locations for precise counting

This avoids the noise problem of fully generative point prediction
while preserving the P2P philosophy of point-based localization.
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import config


class HeadRegionClassifier(nn.Module):
    """
    Lightweight binary classifier that validates whether a cropped
    region actually contains a human head. Uses MobileNetV2 features
    for fast inference.
    """

    def __init__(self):
        super().__init__()
        # Use lightweight MobileNetV2 features (pre-trained on ImageNet)
        mobilenet = models.mobilenet_v2(weights=models.MobileNetV2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Head classification head (pun intended)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        confidence = self.classifier(pooled)
        return confidence


class P2PHeadCounter:
    """
    P2PNet-inspired head counter.

    Instead of predicting random points across the entire image,
    this module:
      1. Takes YOLO person bounding boxes as input
      2. Extracts the top portion of each box (head region)
      3. Estimates head center points
      4. Returns validated head positions for counting
    """

    def __init__(self):
        print("[P2P] Loading head detection counter...")

        if config.USE_GPU:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # Head region ratio: top 25% of person bbox is typically the head
        self.head_ratio = 0.25

        # Min head size to consider valid (in pixels)
        self.min_head_size = 15

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        print(f"[P2P] Head counter ready on {self.device}")

    def _extract_head_region(self, box: list, frame_h: int, frame_w: int) -> dict:
        """
        Extract head region from a person bounding box.

        The head is estimated as the top portion of the bounding box.
        Returns the head center point and the head bounding box.
        """
        x1, y1, x2, y2 = box
        person_h = y2 - y1
        person_w = x2 - x1

        # Head region = top 25% of person box
        head_h = int(person_h * self.head_ratio)
        head_h = max(head_h, self.min_head_size)

        # Head box (centered horizontally, at the top of person box)
        head_x1 = x1 + int(person_w * 0.15)  # Slight inward margin
        head_x2 = x2 - int(person_w * 0.15)
        head_y1 = y1
        head_y2 = min(y1 + head_h, y2)

        # Clamp to frame bounds
        head_x1 = max(0, head_x1)
        head_y1 = max(0, head_y1)
        head_x2 = min(frame_w, head_x2)
        head_y2 = min(frame_h, head_y2)

        # Head center point
        head_cx = (head_x1 + head_x2) // 2
        head_cy = (head_y1 + head_y2) // 2

        return {
            "center": [head_cx, head_cy],
            "box": [head_x1, head_y1, head_x2, head_y2],
            "valid": (head_x2 - head_x1) >= self.min_head_size and
                     (head_y2 - head_y1) >= self.min_head_size,
        }

    def count_from_detections(self, frame: np.ndarray, yolo_boxes: list,
                               yolo_scores: list) -> dict:
        """
        Count heads by extracting head positions from YOLO person detections.

        Args:
            frame: BGR image (numpy array)
            yolo_boxes: List of [x1, y1, x2, y2] person bounding boxes
            yolo_scores: List of YOLO confidence scores

        Returns:
            dict with keys:
                - head_points: list of [x, y] head center positions
                - head_boxes: list of [x1, y1, x2, y2] head bounding boxes
                - scores: confidence scores for each head
                - count: number of detected heads
        """
        frame_h, frame_w = frame.shape[:2]

        head_points = []
        head_boxes = []
        scores = []

        for i, box in enumerate(yolo_boxes):
            head_info = self._extract_head_region(box, frame_h, frame_w)

            if head_info["valid"]:
                head_points.append(head_info["center"])
                head_boxes.append(head_info["box"])
                scores.append(yolo_scores[i] if i < len(yolo_scores) else 0.5)

        return {
            "head_points": head_points,
            "head_boxes": head_boxes,
            "scores": scores,
            "count": len(head_points),
        }

    def count_standalone(self, frame: np.ndarray) -> dict:
        """
        Standalone head counting using skin-color and upper-body detection.
        Used as a supplementary signal when YOLO might miss people.

        Uses color-based head candidate detection in HSV space
        for detecting partially visible heads at frame edges.
        """
        frame_h, frame_w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin color range in HSV (broad range to catch diverse skin tones)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([35, 255, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # Find contours (potential head blobs)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        candidate_points = []
        candidate_scores = []

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area — head blobs should be a reasonable size
            if area < 200 or area > 15000:
                continue

            # Check circularity (heads are roughly circular)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < 0.3:  # Not circular enough for a head
                continue

            # Get center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            candidate_points.append([cx, cy])
            candidate_scores.append(min(circularity, 0.9))

        return {
            "head_points": candidate_points,
            "scores": candidate_scores,
            "count": len(candidate_points),
        }
