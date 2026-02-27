"""Analysis service configuration constants."""

# --- Confidence calibration ---
REAL_MEAN = 15.0
REAL_STD = 8.0

# --- Video sampling ---
VIDEO_MAX_SIDE = 640
VIDEO_MIN_FRAMES = 12
VIDEO_MAX_FRAMES_CAP = 48
VIDEO_FRAMES_PER_MINUTE = 24

# --- Aggregation ---
AGG_MODE_VIDEO = "mean"
TOPK = 5
VIDEO_TRIM_LOW_RATIO = 0.10
VIDEO_TRIM_HIGH_RATIO = 0.30
VIDEO_AGG_MODE_LABEL = "Trimmed Mean (Low 10 Percent, High 30 Percent)"

# --- Redis TTL ---
RESULT_TTL_SEC = 3600
CACHE_TTL_SEC = 24 * 3600

