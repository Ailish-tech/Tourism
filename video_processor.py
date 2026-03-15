"""
Video Processing Pipeline

Processes video files from the video/ folder, runs hybrid detection
(YOLOv11 person detection + P2P head localization) on each frame,
and displays live people count with visual overlays.
"""
import os
import cv2
import time
import numpy as np

import config
from models.yolo_detector import YOLODetector
from models.p2p_counter import P2PHeadCounter
from models.hybrid_fusion import HybridFusion, TemporalSmoother
from heatmap_generator import HeatmapGenerator
from cluster_detector import ClusterDetector


class VideoProcessor:
    """Processes video with hybrid people detection and live count overlay."""

    def __init__(self):
        print("=" * 60)
        print("  HYBRID PEOPLE DETECTION SYSTEM")
        print("  YOLOv11 + P2PNet Head Detection Fusion")
        print("=" * 60)
        print()

        # Initialize models
        self.yolo = YOLODetector()
        self.p2p = P2PHeadCounter()
        self.fusion = HybridFusion()
        self.smoother = TemporalSmoother()
        self.heatmap = HeatmapGenerator()
        self.cluster = ClusterDetector()
        self.show_heatmap = config.SHOW_HEATMAP
        self.show_clusters = config.SHOW_CLUSTERS

        # State
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.fps_history = []

    def get_video_files(self) -> list:
        """Scan video folder for video files."""
        video_dir = config.VIDEO_FOLDER
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            print(f"[VIDEO] Created video folder: {video_dir}")

        files = []
        for f in sorted(os.listdir(video_dir)):
            if f.lower().endswith(config.VIDEO_EXTENSIONS):
                files.append(os.path.join(video_dir, f))

        # Also check root directory for any video files
        root_dir = config.BASE_DIR
        for f in os.listdir(root_dir):
            full_path = os.path.join(root_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(config.VIDEO_EXTENSIONS):
                if full_path not in files:
                    files.append(full_path)

        return files

    def draw_overlay(self, frame: np.ndarray, fused: dict, smoothed_count: int,
                     cluster_result: dict = None) -> np.ndarray:
        """
        Draw detection overlay and HUD on frame.

        Shows:
        - Heatmap overlay (toggle H)
        - Cluster convex hulls (toggle C)
        - YOLO bounding boxes (green)
        - Head position dots (cyan) at top of each person
        - Extra standalone detections (orange)
        - Live count HUD with cluster info
        - Crowd alert banner
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # ── Draw heatmap overlay ──
        if self.show_heatmap:
            heatmap_img = self.heatmap.update(fused["all_positions"], frame.shape)
            # Blend heatmap onto overlay (only where heatmap is non-black)
            mask = np.any(heatmap_img > 0, axis=2)
            blended = cv2.addWeighted(
                overlay, 1 - config.HEATMAP_OPACITY,
                heatmap_img, config.HEATMAP_OPACITY, 0
            )
            overlay[mask] = blended[mask]

        # ── Draw YOLO bounding boxes ──
        if config.SHOW_YOLO_BOXES:
            for i, box in enumerate(fused["yolo_boxes"]):
                x1, y1, x2, y2 = box
                score = fused["yolo_scores"][i] if i < len(fused["yolo_scores"]) else 0

                # Person bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), config.BOX_COLOR, 2)

                # Confidence label at top of box
                label = f"Person {score:.0%}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(overlay,
                              (x1, y1 - label_size[1] - 8),
                              (x1 + label_size[0] + 6, y1),
                              config.BOX_COLOR, -1)
                cv2.putText(overlay, label,
                            (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # ── Draw head position markers ──
        if config.SHOW_P2P_POINTS:
            for pt in fused.get("head_points", []):
                x, y = pt
                # Solid circle at head position
                cv2.circle(overlay, (x, y), 6, (0, 255, 255), -1)  # Cyan filled
                cv2.circle(overlay, (x, y), 8, (255, 255, 255), 1)  # White outline

            # Draw head bounding boxes (small boxes around head region)
            for hbox in fused.get("head_boxes", []):
                hx1, hy1, hx2, hy2 = hbox
                cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (0, 255, 255), 1)

        # ── Draw extra standalone detections (people YOLO missed) ──
        for pt in fused.get("extra_heads", []):
            x, y = pt
            cv2.drawMarker(overlay, (x, y), (0, 140, 255),
                           cv2.MARKER_DIAMOND, 12, 2)
            cv2.circle(overlay, (x, y), 10, (0, 140, 255), 1)

        # ── Draw cluster hulls ──
        if self.show_clusters and cluster_result is not None:
            overlay = self.cluster.draw_clusters(overlay, cluster_result)

        # ── Draw HUD panel ──
        hud_h = 155
        hud_w = 300
        hud_x = w - hud_w - 15
        hud_y = 15

        # Semi-transparent background
        hud_bg = overlay.copy()
        cv2.rectangle(hud_bg, (hud_x, hud_y),
                      (hud_x + hud_w, hud_y + hud_h),
                      config.HUD_BG_COLOR, -1)
        cv2.addWeighted(hud_bg, config.HUD_OPACITY,
                        overlay, 1 - config.HUD_OPACITY, 0, overlay)

        # Border
        cv2.rectangle(overlay, (hud_x, hud_y),
                      (hud_x + hud_w, hud_y + hud_h),
                      config.COUNT_COLOR, 2)

        # Title
        cv2.putText(overlay, "PEOPLE COUNT",
                    (hud_x + 15, hud_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COUNT_COLOR, 2)

        # Main count (large)
        count_text = str(smoothed_count)
        count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3)[0]
        count_x = hud_x + (hud_w - count_size[0]) // 2
        cv2.putText(overlay, count_text,
                    (count_x, hud_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)

        # Detection breakdown
        if config.SHOW_FUSION_INFO:
            info_y = hud_y + 100
            breakdown = f"YOLO: {fused['yolo_count']}  Heads: {fused['head_count']}"
            if fused['extra_count'] > 0:
                breakdown += f"  +{fused['extra_count']} extra"
            cv2.putText(overlay, breakdown,
                        (hud_x + 12, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            # Cluster count line
            cluster_count = cluster_result["cluster_count"] if cluster_result else 0
            cluster_text = f"Clusters: {cluster_count}"
            cluster_color = (0, 200, 255) if cluster_count > 0 else (120, 120, 120)
            cv2.putText(overlay, cluster_text,
                        (hud_x + 12, info_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cluster_color, 1)

            cv2.putText(overlay,
                        f"FPS: {self.fps:.1f}  Frame: {self.frame_count}",
                        (hud_x + 12, info_y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # ── Top-left badge ──
        cv2.putText(overlay, "HYBRID AI - YOLOv11 + P2PNet",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        # Heatmap status indicator
        hm_status = "HEATMAP: ON" if self.show_heatmap else "HEATMAP: OFF"
        hm_color = (0, 255, 100) if self.show_heatmap else (100, 100, 100)
        cv2.putText(overlay, hm_status,
                    (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, hm_color, 1)

        # Cluster status indicator
        cl_status = "CLUSTERS: ON" if self.show_clusters else "CLUSTERS: OFF"
        cl_color = (0, 200, 255) if self.show_clusters else (100, 100, 100)
        cv2.putText(overlay, cl_status,
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cl_color, 1)

        # ── Crowd Alert Banner ──
        if smoothed_count >= config.CROWD_ALERT_THRESHOLD:
            # Flashing effect — visible ~75% of the time
            if int(time.time() * 3) % 4 != 0:
                alert_text = "!! CROWD FORMING !!"
                alert_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                alert_x = (w - alert_size[0]) // 2
                alert_y = h - 40
                # Dark background bar
                cv2.rectangle(overlay,
                              (alert_x - 15, alert_y - alert_size[1] - 12),
                              (alert_x + alert_size[0] + 15, alert_y + 10),
                              (0, 0, 80), -1)
                # Red border
                cv2.rectangle(overlay,
                              (alert_x - 15, alert_y - alert_size[1] - 12),
                              (alert_x + alert_size[0] + 15, alert_y + 10),
                              (0, 0, 255), 2)
                # Alert text
                cv2.putText(overlay, alert_text,
                            (alert_x, alert_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Recording indicator (blinking red dot)
        if int(time.time() * 2) % 2:
            cv2.circle(overlay, (w - 30, 45 + hud_h), 6, (0, 0, 255), -1)

        return overlay

    def calculate_fps(self):
        """Calculate and smooth FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self.fps_history.append(instant_fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            self.fps = np.mean(self.fps_history)
        self.last_time = current_time

    def process_video(self, video_path: str):
        """
        Process a single video file with hybrid detection.
        """
        print(f"\n[VIDEO] Processing: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[VIDEO] Resolution: {video_w}x{video_h} | "
              f"FPS: {video_fps:.1f} | Frames: {total_frames}")

        # Create display window
        window_name = "Hybrid People Detection - YOLOv11 + P2PNet"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

        self.frame_count = 0
        self.smoother.reset()
        self.heatmap.reset()
        paused = False

        # Cache last results for skipped frames
        last_fused = None
        last_smoothed = 0
        last_cluster = None

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[VIDEO] End of video reached.")
                    break

                self.frame_count += 1

                # Process every Nth frame
                if self.frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                    # Step 1: YOLO person detection
                    yolo_result = self.yolo.detect(frame)

                    # Step 2: P2P head position estimation from YOLO boxes
                    p2p_head_result = self.p2p.count_from_detections(
                        frame, yolo_result["boxes"], yolo_result["scores"]
                    )

                    # Step 3: Standalone P2P for edge-case heads
                    # (only when YOLO detects few people — performance optimization)
                    p2p_standalone = None
                    if yolo_result["count"] < 3:
                        p2p_standalone = self.p2p.count_standalone(frame)

                    # Step 4: Fuse detections
                    fused = self.fusion.fuse(yolo_result, p2p_head_result, p2p_standalone)

                    # Step 5: Cluster detection
                    cluster_result = self.cluster.detect(fused["all_positions"])

                    # Step 6: Smooth the count
                    smoothed_count = self.smoother.update(fused["total_count"])

                    # Calculate FPS
                    self.calculate_fps()

                    # Cache results
                    last_fused = fused
                    last_smoothed = smoothed_count
                    last_cluster = cluster_result

                    # Draw overlay
                    display_frame = self.draw_overlay(frame, fused, smoothed_count, cluster_result)
                elif last_fused is not None:
                    display_frame = self.draw_overlay(frame, last_fused, last_smoothed, last_cluster)
                else:
                    display_frame = frame

                cv2.imshow(window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC to quit
                print("[VIDEO] Quit requested.")
                cap.release()
                cv2.destroyAllWindows()
                return "quit"
            elif key == ord('p') or key == 32:  # P or SPACE to pause/resume
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"[VIDEO] {status}")
            elif key == ord('n'):  # N to skip to next video
                print("[VIDEO] Skipping to next video...")
                break
            elif key == ord('h'):  # H to toggle heatmap
                self.show_heatmap = not self.show_heatmap
                status = "ON" if self.show_heatmap else "OFF"
                print(f"[HEATMAP] {status}")
            elif key == ord('c'):  # C to toggle clusters
                self.show_clusters = not self.show_clusters
                status = "ON" if self.show_clusters else "OFF"
                print(f"[CLUSTERS] {status}")

        cap.release()
        return "next"

    def run(self):
        """
        Main processing loop.
        Scans for video files and processes them in sequence.
        """
        video_files = self.get_video_files()

        if not video_files:
            print("\n[ERROR] No video files found!")
            print(f"  Place video files in: {config.VIDEO_FOLDER}")
            print(f"  Supported formats: {', '.join(config.VIDEO_EXTENSIONS)}")
            return

        print(f"\n[VIDEO] Found {len(video_files)} video(s):")
        for i, vf in enumerate(video_files, 1):
            print(f"  {i}. {os.path.basename(vf)}")

        for video_path in video_files:
            result = self.process_video(video_path)
            if result == "quit":
                break

        cv2.destroyAllWindows()
        print("\n[DONE] Processing complete.")
