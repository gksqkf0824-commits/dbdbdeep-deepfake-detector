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


def _compute_target_frame_budget(
    duration_sec: float,
    min_frames: int,
    max_frames: int,
    frames_per_minute: int,
) -> int:
    if min_frames <= 0:
        min_frames = 1
    if max_frames < min_frames:
        max_frames = min_frames

    if duration_sec <= 0:
        return min_frames

    adaptive = int(round((duration_sec / 60.0) * float(frames_per_minute)))
    return max(min_frames, min(max_frames, adaptive))


def _uniform_indices(total_frames: int, target_frames: int) -> np.ndarray:
    if total_frames <= 0 or target_frames <= 0:
        return np.array([], dtype=np.int64)
    if total_frames <= target_frames:
        return np.arange(total_frames, dtype=np.int64)

    idx = np.linspace(0, total_frames - 1, num=target_frames, dtype=np.int64)
    return np.unique(idx)


def _read_selected_frames_with_grab(
    cap: cv2.VideoCapture,
    frame_indices: np.ndarray,
    max_side: int,
) -> Tuple[List[np.ndarray], int]:
    frames: List[np.ndarray] = []
    current_idx = -1

    for target in frame_indices:
        target_idx = int(target)
        while current_idx < target_idx:
            if not cap.grab():
                return frames, current_idx + 1
            current_idx += 1

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        frames.append(resize_with_aspect_ratio(frame, max_side=max_side))

    return frames, current_idx + 1


def _reservoir_sample_frames(
    cap: cv2.VideoCapture,
    target_frames: int,
    max_side: int,
) -> Tuple[List[np.ndarray], int]:
    """
    frame_count를 알 수 없을 때 전체 구간을 고르게 반영하기 위한 fallback.
    """
    target = max(int(target_frames), 1)
    rng = np.random.default_rng(7)

    frames: List[np.ndarray] = []
    seen = 0

    while True:
        ok = cap.grab()
        if not ok:
            break

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        seen += 1
        frame = resize_with_aspect_ratio(frame, max_side=max_side)

        if len(frames) < target:
            frames.append(frame)
        else:
            j = int(rng.integers(0, seen))
            if j < target:
                frames[j] = frame

        if seen > 500000:
            break

    return frames, seen


def video_to_uniform_sampled_frames(
    video_path: str,
    max_side: int = 640,
    min_frames: int = 12,
    max_frames: int = 36,
    frames_per_minute: int = 18,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    ✅ 전체 길이 기반 균등 샘플링.
    - 긴 영상도 앞부분에 치우치지 않고 전 구간에서 프레임을 선택
    - 샘플 프레임만 decode/retrieve 해서 속도 개선
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("비디오를 열 수 없습니다. 파일이 손상되었거나 코덱을 지원하지 않을 수 있습니다.")

    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_raw) if fps_raw and fps_raw > 0 else 30.0

    total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(total_frames_raw) if total_frames_raw and total_frames_raw > 0 else 0
    duration_sec = (float(total_frames) / fps) if total_frames > 0 else 0.0

    target_frames = _compute_target_frame_budget(
        duration_sec=duration_sec,
        min_frames=min_frames,
        max_frames=max_frames,
        frames_per_minute=frames_per_minute,
    )

    sampled_mode = "uniform_full_span_indexed"
    decoded_cursor = 0

    if total_frames > 0:
        idx = _uniform_indices(total_frames=total_frames, target_frames=target_frames)
        frames, decoded_cursor = _read_selected_frames_with_grab(cap, idx, max_side=max_side)
    else:
        sampled_mode = "uniform_reservoir_fallback"
        frames, decoded_cursor = _reservoir_sample_frames(
            cap,
            target_frames=target_frames,
            max_side=max_side,
        )

    cap.release()

    meta = {
        "fps": fps,
        "duration_sec": duration_sec,
        "total_frames": total_frames if total_frames > 0 else None,
        "target_frames": target_frames,
        "sampled_frames": len(frames),
        "decoded_cursor": int(decoded_cursor),
        "max_side": max_side,
        "frames_per_minute": frames_per_minute,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "sampling": sampled_mode,
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


def trimmed_mean_confidence(
    values: List[float],
    trim_ratio: float = 0.10,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    상/하위 trim_ratio 비율을 제외한 값들의 평균을 계산.
    예: trim_ratio=0.10 이면 하위 10%, 상위 10%를 제외.
    """
    if not values:
        return None, {
            "trim_ratio": float(trim_ratio),
            "raw_count": 0,
            "used_count": 0,
            "excluded_low_count": 0,
            "excluded_high_count": 0,
        }

    arr = np.sort(np.array(values, dtype=np.float32))
    n = len(arr)

    ratio = float(trim_ratio)
    if ratio < 0:
        ratio = 0.0
    if ratio > 0.49:
        ratio = 0.49

    trim_count = int(np.floor(n * ratio))
    max_trim = (n - 1) // 2
    trim_count = min(trim_count, max_trim)

    if trim_count > 0:
        core = arr[trim_count : n - trim_count]
    else:
        core = arr

    if core.size == 0:
        core = arr
        trim_count = 0

    return float(np.mean(core)), {
        "trim_ratio": ratio,
        "raw_count": n,
        "used_count": int(core.size),
        "excluded_low_count": int(trim_count),
        "excluded_high_count": int(trim_count),
    }


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
