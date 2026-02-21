import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import scipy.stats as stats


# =========================
# Stats / Reliability
# =========================

def calculate_p_value(score: float, real_mean: float, real_std: float) -> float:
    """score가 real 분포에서 얼마나 드문지(우측 꼬리) p-value로 계산."""
    z_score = (score - real_mean) / real_std
    p_value = 1 - stats.norm.cdf(z_score)
    return round(max(float(p_value), 0.0001), 4)

def make_reliability_label(p_val: float) -> str:
    if p_val < 0.01:
        return "매우 높음"
    if p_val < 0.05:
        return "높음"
    return "보통"


# =========================
# Hash / Redis JSON helpers
# =========================

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def redis_set_json(redis_db, key: str, payload: Dict[str, Any], ex: int) -> None:
    redis_db.set(key, json.dumps(payload), ex=ex)

def redis_get_json(redis_db, key: str) -> Optional[Dict[str, Any]]:
    v = redis_db.get(key)
    if v is None:
        return None
    return json.loads(v)


# =========================
# Resize / Encode helpers
# =========================

def resize_with_aspect_ratio(frame_bgr: np.ndarray, max_side: int = 640) -> np.ndarray:
    """
    긴 변 기준으로 비율 유지하며 축소.
    max_side보다 작으면 그대로 반환.
    """
    h, w = frame_bgr.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_side:
        return frame_bgr

    scale = max_side / max_dim
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def frame_to_jpeg_bytes(frame_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("프레임 JPEG 인코딩 실패")
    return buf.tobytes()


# =========================
# Video / Frame helpers
# =========================

def video_to_sampled_frames_per_second(
    video_path: str,
    seconds_step: float = 1.0,
    max_side: int = 640,
    max_frames: Optional[int] = 60,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    ✅ 순차 샘플링 기반 (cap.set 없이 cap.read로 쭉 읽음)
    - seconds_step=1.0이면 1초마다 1프레임
    - max_frames로 상한을 두어 긴 영상 폭주 방지
    반환:
      frames: 샘플 프레임 리스트(BGR)
      meta: fps, frame_interval, sampled_frames 등 메타
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("비디오를 열 수 없습니다. 파일이 손상되었거나 코덱을 지원하지 않을 수 있습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0

    seconds_step = float(seconds_step) if seconds_step and seconds_step > 0 else 1.0
    frame_interval = max(int(round(fps * seconds_step)), 1)

    frames: List[np.ndarray] = []
    frame_idx = 0
    picked = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx % frame_interval == 0:
            frame = resize_with_aspect_ratio(frame, max_side=max_side)
            frames.append(frame)
            picked += 1

            if max_frames is not None and picked >= int(max_frames):
                break

        frame_idx += 1

        # 안전장치 (깨진 파일/초장시간 영상)
        if frame_idx > 500000:
            break

    cap.release()

    meta = {
        "fps": fps,
        "seconds_step": seconds_step,
        "frame_interval": frame_interval,
        "sampled_frames": len(frames),
        "max_frames": max_frames,
        "max_side": max_side,
        "sampling": "sequential_per_second"
    }
    return frames, meta


# =========================
# Aggregation
# =========================

def aggregate_scores(values: List[float], mode: str = "mean", topk: int = 5) -> Optional[float]:
    if not values:
        return None

    arr = np.array(values, dtype=np.float32)

    if mode == "median":
        return float(np.median(arr))

    if mode == "topk_mean":
        k = min(int(topk), len(arr))
        topk_vals = np.sort(arr)[-k:]
        return float(np.mean(topk_vals))

    return float(np.mean(arr))


# =========================
# Result builder
# =========================

def build_analysis_result(
    score: float,
    pixel: float,
    freq: float,
    real_mean: float,
    real_std: float
) -> Dict[str, Any]:
    p_val = calculate_p_value(score, real_mean=real_mean, real_std=real_std)
    return {
        "confidence": float(score),
        "pixel_score": float(pixel) if pixel is not None else None,
        "freq_score": float(freq) if freq is not None else None,
        "is_fake": float(score) < 50,
        "p_value": p_val,
        "reliability": make_reliability_label(p_val),
    }