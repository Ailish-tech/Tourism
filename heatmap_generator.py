"""
Crowd Density Heatmap Generator

Accumulates detected person positions over time and produces
a color-gradient overlay (blue → green → yellow → red) showing
where people concentrate in the video frame.
"""
import cv2
import numpy as np
import config


class HeatmapGenerator:
    """Generates a real-time crowd density heatmap from detection positions."""

    def __init__(self):
        self.accumulator = None
        self.radius = config.HEATMAP_RADIUS
        self.decay = config.HEATMAP_DECAY
        self.colormap = config.HEATMAP_COLORMAP
        self.opacity = config.HEATMAP_OPACITY
        print(f"[HEATMAP] Initialized — radius={self.radius}, "
              f"decay={self.decay}, opacity={self.opacity}")

    def update(self, positions: list, frame_shape: tuple) -> np.ndarray:
        """
        Stamp new detections onto the accumulator and return a colored heatmap.

        Args:
            positions: List of (x, y) tuples for detected person positions.
            frame_shape: Shape of the video frame (h, w, c).

        Returns:
            BGR heatmap image ready for blending onto the original frame.
        """
        h, w = frame_shape[:2]

        # Lazy-init the accumulator to match the video resolution
        if self.accumulator is None or self.accumulator.shape[:2] != (h, w):
            self.accumulator = np.zeros((h, w), dtype=np.float32)

        # Decay old detections so the heatmap fades over time
        self.accumulator *= self.decay

        # Stamp a Gaussian blob at each detected position
        for (x, y) in positions:
            x, y = int(x), int(y)
            if 0 <= x < w and 0 <= y < h:
                # Define the region around the point
                r = self.radius
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(w, x + r + 1)
                y2 = min(h, y + r + 1)

                # Build a small Gaussian kernel for this region
                gx = np.arange(x1, x2) - x
                gy = np.arange(y1, y2) - y
                gx, gy = np.meshgrid(gx, gy)
                sigma = r / 2.5
                gaussian = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))

                # Add to accumulator
                self.accumulator[y1:y2, x1:x2] += gaussian.astype(np.float32)

        # Normalize to 0-255 for colormap application
        if self.accumulator.max() > 0:
            norm = self.accumulator / self.accumulator.max()
        else:
            norm = self.accumulator

        heatmap_gray = (norm * 255).astype(np.uint8)

        # Apply a slight extra Gaussian blur for smoother gradients
        heatmap_gray = cv2.GaussianBlur(heatmap_gray, (0, 0), sigmaX=self.radius // 2)

        # Apply colormap  (JET: blue → green → yellow → red)
        heatmap_color = cv2.applyColorMap(heatmap_gray, self.colormap)

        # Make areas with no detections fully transparent (black)
        # by zeroing out pixels where the accumulator is near zero
        mask = heatmap_gray < 5
        heatmap_color[mask] = 0

        return heatmap_color

    def reset(self):
        """Reset the accumulator (e.g. when switching videos)."""
        self.accumulator = None
        print("[HEATMAP] Accumulator reset.")
