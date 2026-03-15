"""
DBSCAN Cluster Detector

Groups detected person positions into spatial clusters using DBSCAN.
Draws colored convex hulls around each cluster on the video frame.
"""
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import config


# Distinct colors for up to 10 clusters (BGR)
CLUSTER_COLORS = [
    (0, 200, 255),    # Orange-yellow
    (255, 100, 100),  # Light blue
    (100, 255, 100),  # Light green
    (200, 100, 255),  # Pink
    (255, 255, 100),  # Cyan
    (100, 200, 255),  # Peach
    (200, 255, 100),  # Lime
    (150, 100, 255),  # Magenta
    (255, 200, 150),  # Sky blue
    (100, 255, 255),  # Yellow
]


class ClusterDetector:
    """Detects and visualizes spatial clusters of people using DBSCAN."""

    def __init__(self):
        self.eps = config.CLUSTER_DISTANCE
        self.min_samples = config.CLUSTER_MIN_PEOPLE
        self.hull_opacity = config.CLUSTER_HULL_OPACITY
        self.last_result = None
        print(f"[CLUSTER] Initialized — eps={self.eps}px, "
              f"min_samples={self.min_samples}")

    def detect(self, positions: list) -> dict:
        """
        Run DBSCAN on detected person positions.

        Args:
            positions: List of (x, y) tuples.

        Returns:
            dict with cluster_count, labels, and per-cluster groups.
        """
        if len(positions) < self.min_samples:
            self.last_result = {
                "cluster_count": 0,
                "labels": [],
                "clusters": {},
                "positions": positions,
            }
            return self.last_result

        points = np.array(positions)
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(points)

        labels = clustering.labels_

        # Group points by cluster label (ignore noise = -1)
        clusters = {}
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in unique_labels:
            mask = labels == label
            cluster_points = points[mask].tolist()
            clusters[int(label)] = cluster_points

        self.last_result = {
            "cluster_count": len(clusters),
            "labels": labels.tolist(),
            "clusters": clusters,
            "positions": positions,
        }
        return self.last_result

    def draw_clusters(self, frame: np.ndarray, result: dict = None) -> np.ndarray:
        """
        Draw colored convex hulls and labels around each cluster.

        Args:
            frame: The video frame to draw on (modified in-place).
            result: Cluster detection result (uses last_result if None).

        Returns:
            The frame with cluster overlays drawn.
        """
        if result is None:
            result = self.last_result
        if result is None or result["cluster_count"] == 0:
            return frame

        overlay = frame.copy()

        for label, points in result["clusters"].items():
            if len(points) < 2:
                continue

            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            pts = np.array(points, dtype=np.int32)

            if len(points) == 2:
                # Just two points — draw a thick line between them
                cv2.line(overlay, tuple(pts[0]), tuple(pts[1]), color, 3)
                # Draw a translucent capsule around the line
                for p in pts:
                    cv2.circle(overlay, tuple(p), 25, color, -1)
            else:
                # 3+ points — draw convex hull
                hull = cv2.convexHull(pts)
                # Expand hull slightly for visual padding
                M = cv2.moments(hull)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Scale hull outward by ~15% for padding
                    expanded = []
                    for p in hull:
                        px, py = p[0]
                        dx = px - cx
                        dy = py - cy
                        expanded.append([int(px + dx * 0.15), int(py + dy * 0.15)])
                    expanded = np.array(expanded, dtype=np.int32)
                    cv2.fillConvexPoly(overlay, expanded, color)
                    cv2.polylines(frame, [expanded], True, color, 2)
                else:
                    cv2.fillConvexPoly(overlay, hull, color)
                    cv2.polylines(frame, [hull], True, color, 2)

            # Cluster label at centroid
            centroid_x = int(np.mean([p[0] for p in points]))
            centroid_y = int(np.mean([p[1] for p in points]))
            label_text = f"Cluster {label + 1} ({len(points)})"
            cv2.putText(frame, label_text,
                        (centroid_x - 40, centroid_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(frame, label_text,
                        (centroid_x - 40, centroid_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Blend the hull fills
        cv2.addWeighted(overlay, self.hull_opacity, frame,
                        1 - self.hull_opacity, 0, frame)

        return frame
