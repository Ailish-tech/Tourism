"""
Hybrid People Detection System
Entry point: YOLOv11 + P2PNet Fusion

Usage:
    python run.py                    # Process all videos in video/ folder
    python run.py --video path.mp4   # Process a specific video file
    python run.py --camera 0         # Use webcam (device 0)

Controls:
    Q / ESC  - Quit
    P / SPACE - Pause/Resume
    N        - Next video
    H        - Toggle heatmap overlay
    C        - Toggle cluster overlays
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid People Detection: YOLOv11 + P2PNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Q / ESC    Quit the application
  P / SPACE  Pause / Resume video
  N          Skip to next video
  H          Toggle heatmap overlay
  C          Toggle cluster overlays
        """
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a specific video file (overrides video/ folder scan)"
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Webcam device index (e.g., 0 for default camera)"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="YOLO confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU-only mode"
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Disable crowd density heatmap overlay"
    )

    args = parser.parse_args()

    # Apply overrides
    import config
    if args.conf is not None:
        config.YOLO_CONFIDENCE = args.conf
    if args.no_gpu:
        config.USE_GPU = False
    if args.no_heatmap:
        config.SHOW_HEATMAP = False

    from video_processor import VideoProcessor

    processor = VideoProcessor()

    if args.camera is not None:
        # Webcam mode
        print(f"\n[MODE] Webcam (device {args.camera})")
        import cv2
        import numpy as np
        from models.hybrid_fusion import TemporalSmoother

        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {args.camera}")
            sys.exit(1)

        window_name = "Hybrid People Detection - LIVE"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

        smoother = TemporalSmoother()

        print("[CAMERA] Press Q/ESC to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processor.frame_count += 1

            yolo_result = processor.yolo.detect(frame)
            p2p_result = processor.p2p.count(frame)
            fused = processor.fusion.fuse(yolo_result, p2p_result)
            smoothed = smoother.update(fused["total_count"])
            processor.calculate_fps()

            display = processor.draw_overlay(frame, fused, smoothed)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    elif args.video:
        # Single video mode
        if not os.path.exists(args.video):
            print(f"[ERROR] Video not found: {args.video}")
            sys.exit(1)
        processor.process_video(args.video)
        import cv2
        cv2.destroyAllWindows()

    else:
        # Default: scan video/ folder
        processor.run()


if __name__ == "__main__":
    main()
