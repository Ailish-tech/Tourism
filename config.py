"""
Centralized configuration for the Hybrid People Detection System.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(BASE_DIR, "video")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")

# ─── YOLO Settings ────────────────────────────────────────────────────────────
YOLO_MODEL = "yolo11s.pt"  # YOLOv11 small — better accuracy than nano
YOLO_CONFIDENCE = 0.20     # Lower threshold to catch more people
YOLO_IOU_THRESHOLD = 0.50  # NMS IoU threshold
YOLO_PERSON_CLASS = 0      # COCO class index for 'person'
YOLO_IMG_SIZE = 960        # Larger input for detecting smaller/distant people

# ─── P2PNet Settings ─────────────────────────────────────────────────────────
P2P_CONFIDENCE = 0.5       # Confidence threshold for P2P point predictions
P2P_INPUT_SIZE = (512, 512)  # Input size for P2P feature extractor
P2P_NUM_ANCHOR_POINTS = 4  # Anchor grid multiplier (grid = H/8 * W/8 * N)

# ─── Hybrid Fusion Settings ──────────────────────────────────────────────────
FUSION_DISTANCE_THRESHOLD = 50   # Max pixel distance to consider P2P point inside YOLO detection
FUSION_MODE = "weighted"         # "yolo_priority" | "p2p_priority" | "weighted"
FUSION_YOLO_WEIGHT = 0.6        # Weight for YOLO count in weighted mode
FUSION_P2P_WEIGHT = 0.4         # Weight for P2P count in weighted mode

# ─── Temporal Smoothing ──────────────────────────────────────────────────────
SMOOTHING_WINDOW = 5       # Number of past frames for temporal smoothing
SMOOTHING_METHOD = "ema"   # "average" | "median" | "ema" (exponential moving average)
EMA_ALPHA = 0.3            # EMA smoothing factor (higher = more responsive)

# ─── Display Settings ────────────────────────────────────────────────────────
DISPLAY_WIDTH = 1280       # Display window width
DISPLAY_HEIGHT = 720       # Display window height
SHOW_YOLO_BOXES = True     # Show YOLO bounding boxes
SHOW_P2P_POINTS = True     # Show P2P head points
SHOW_FUSION_INFO = True    # Show fusion debug info
BOX_COLOR = (0, 255, 0)    # Green for YOLO boxes
POINT_COLOR = (255, 100, 0)  # Blue-orange for P2P points
COUNT_COLOR = (0, 255, 255)  # Yellow for count text
HUD_BG_COLOR = (20, 20, 20)  # Dark background for HUD
HUD_OPACITY = 0.75         # HUD background opacity

# ─── Heatmap Settings ────────────────────────────────────────────────────────
SHOW_HEATMAP = True        # Show crowd density heatmap overlay
HEATMAP_OPACITY = 0.45     # Heatmap blend strength over original frame
HEATMAP_RADIUS = 40        # Gaussian blob radius per detection point (px)
HEATMAP_DECAY = 0.92       # Exponential decay factor (0-1); lower = faster fade
HEATMAP_COLORMAP = 11      # cv2.COLORMAP_HOT = 11 (black → red → orange → white heat texture)

# ─── Clustering Settings (DBSCAN) ────────────────────────────────────────────
SHOW_CLUSTERS = True       # Show cluster convex hull overlays
CLUSTER_DISTANCE = 100     # DBSCAN eps: max pixel distance between cluster members
CLUSTER_MIN_PEOPLE = 2     # DBSCAN min_samples: minimum people to form a cluster
CLUSTER_HULL_OPACITY = 0.3 # Opacity of cluster hull fill

# ─── Crowd Alert Settings ────────────────────────────────────────────────────
CROWD_ALERT_THRESHOLD = 5  # Show "CROWD FORMING" alert when >= N people detected

# ─── Processing Settings ─────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 1  # Process every Nth frame (1 = every frame)
USE_GPU = True              # Use GPU if available
