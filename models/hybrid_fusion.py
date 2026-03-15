"""
Hybrid Fusion Engine

Combines YOLOv11 person detections with P2PNet head position
estimation for precise people counting. Uses temporal smoothing
to prevent count jitter across frames.
"""
import numpy as np
import config


class HybridFusion:
    """
    Fuses YOLO person detections with P2P head positions.

    Strategy:
      1. YOLO detects persons → provides bounding boxes
      2. P2P extracts head positions from those boxes → provides head points
      3. Optionally, P2P standalone catches edge-case heads YOLO missed
      4. Deduplicates standalone P2P detections against YOLO boxes
      5. Final count = YOLO persons + unique standalone P2P heads
    """

    def __init__(self):
        self.distance_threshold = config.FUSION_DISTANCE_THRESHOLD
        print(f"[FUSION] Initialized — distance_threshold={self.distance_threshold}px")

    def _point_inside_box(self, point: list, box: list, margin: int = 30) -> bool:
        """Check if a point is inside a bounding box (with margin)."""
        x, y = point
        x1, y1, x2, y2 = box
        return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)

    def fuse(self, yolo_result: dict, p2p_head_result: dict,
             p2p_standalone_result: dict = None) -> dict:
        """
        Fuse YOLO and P2P head detections.

        Args:
            yolo_result: Output from YOLODetector.detect()
            p2p_head_result: Output from P2PHeadCounter.count_from_detections()
            p2p_standalone_result: Optional output from P2PHeadCounter.count_standalone()

        Returns:
            dict with detection info and final count
        """
        yolo_boxes = yolo_result["boxes"]
        yolo_centers = yolo_result["centers"]
        yolo_scores = yolo_result["scores"]
        yolo_count = yolo_result["count"]

        head_points = p2p_head_result["head_points"]
        head_boxes = p2p_head_result.get("head_boxes", [])

        # Start with YOLO count as the base (most reliable)
        extra_heads = []

        # Check standalone P2P detections for people YOLO might have missed
        if p2p_standalone_result and p2p_standalone_result["count"] > 0:
            for pt in p2p_standalone_result["head_points"]:
                is_duplicate = False
                for box in yolo_boxes:
                    if self._point_inside_box(pt, box):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    # Check distance to nearest YOLO center
                    if yolo_centers:
                        min_dist = min(
                            np.sqrt((pt[0] - c[0])**2 + (pt[1] - c[1])**2)
                            for c in yolo_centers
                        )
                        if min_dist < self.distance_threshold:
                            is_duplicate = True

                if not is_duplicate:
                    extra_heads.append(pt)

        total_count = yolo_count + len(extra_heads)

        # Compile all positions
        all_positions = list(yolo_centers)
        all_positions.extend(extra_heads)

        return {
            "total_count": total_count,
            "yolo_count": yolo_count,
            "head_count": len(head_points),
            "extra_count": len(extra_heads),
            "yolo_boxes": yolo_boxes,
            "yolo_centers": yolo_centers,
            "yolo_scores": yolo_scores,
            "head_points": head_points,
            "head_boxes": head_boxes,
            "extra_heads": extra_heads,
            "all_positions": all_positions,
        }


class TemporalSmoother:
    """
    Smooths the people count over time to reduce frame-to-frame jitter.
    """

    def __init__(self):
        self.window = config.SMOOTHING_WINDOW
        self.method = config.SMOOTHING_METHOD
        self.alpha = config.EMA_ALPHA
        self.history = []
        self.ema_value = None

    def update(self, count: int) -> int:
        """
        Add a new count and return the smoothed value.
        """
        self.history.append(count)
        if len(self.history) > self.window:
            self.history.pop(0)

        if self.method == "average":
            return int(round(np.mean(self.history)))
        elif self.method == "median":
            return int(round(np.median(self.history)))
        elif self.method == "ema":
            if self.ema_value is None:
                self.ema_value = float(count)
            else:
                self.ema_value = self.alpha * count + (1 - self.alpha) * self.ema_value
            return int(round(self.ema_value))
        else:
            return count

    def reset(self):
        """Reset the smoother state."""
        self.history.clear()
        self.ema_value = None
