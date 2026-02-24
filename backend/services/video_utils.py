"""Video/frame/wavelet/media utilities."""

import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def resize_with_aspect_ratio(frame_bgr: np.ndarray, max_side: int = 640) -> np.ndarray:
    """긴 변 기준으로 비율 유지하며 축소."""
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


def video_to_sampled_frames_per_second(
    video_path: str,
    seconds_step: float = 1.0,
    max_side: int = 640,
    max_frames: Optional[int] = 60,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    순차 샘플링 기반(cap.read).
    - seconds_step=1.0이면 1초마다 1프레임
    - max_frames로 상한을 두어 긴 영상 폭주 방지
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
        "sampling": "sequential_per_second",
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
    """frame_count를 알 수 없을 때 전체 구간을 고르게 반영하기 위한 fallback."""
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
    전체 길이 기반 균등 샘플링.
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


def to_gray(img_rgb_uint8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def gray01_to_rgb_uint8(img_gray01: np.ndarray) -> np.ndarray:
    g = (np.clip(img_gray01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=2)


def _prepare_for_wavelet(gray01: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = gray01.shape[:2]
    h4 = h - (h % 4)
    w4 = w - (w % 4)
    if h4 < 4 or w4 < 4:
        raise ValueError("Wavelet 분해를 위한 최소 해상도(4x4)가 부족합니다.")
    return gray01[:h4, :w4], (h, w)


def _haar_dwt2(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]

    ll = (a + b + c + d) / 2.0
    lh = (a + b - c - d) / 2.0
    hl = (a - b + c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    return ll.astype(np.float32), lh.astype(np.float32), hl.astype(np.float32), hh.astype(np.float32)


def _haar_idwt2(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
) -> np.ndarray:
    h, w = ll.shape
    out = np.zeros((h * 2, w * 2), dtype=np.float32)

    out[0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    out[0::2, 1::2] = (ll + lh - hl - hh) / 2.0
    out[1::2, 0::2] = (ll - lh + hl - hh) / 2.0
    out[1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return out


def _decompose_l2(gray01: np.ndarray) -> Dict[str, np.ndarray]:
    ll1, lh1, hl1, hh1 = _haar_dwt2(gray01)
    ll2, lh2, hl2, hh2 = _haar_dwt2(ll1)
    return {
        "ll2": ll2,
        "lh2": lh2,
        "hl2": hl2,
        "hh2": hh2,
        "lh1": lh1,
        "hl1": hl1,
        "hh1": hh1,
    }


def _reconstruct_l2(coeffs: Dict[str, np.ndarray]) -> np.ndarray:
    ll1 = _haar_idwt2(coeffs["ll2"], coeffs["lh2"], coeffs["hl2"], coeffs["hh2"])
    x = _haar_idwt2(ll1, coeffs["lh1"], coeffs["hl1"], coeffs["hh1"])
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _copy_coeffs(coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in coeffs.items()}


def _norm01(x: np.ndarray) -> np.ndarray:
    y = np.abs(x).astype(np.float32)
    y = y - y.min()
    y = y / (y.max() + 1e-6)
    return y


def wavelet_band_energy_ratio(gray_img01: np.ndarray) -> Dict[str, float]:
    x, _ = _prepare_for_wavelet(gray_img01)
    c = _decompose_l2(x)

    e_low = float(np.sum(c["ll2"] ** 2))
    e_mid = float(np.sum(c["lh2"] ** 2) + np.sum(c["hl2"] ** 2) + np.sum(c["hh2"] ** 2))
    e_high = float(np.sum(c["lh1"] ** 2) + np.sum(c["hl1"] ** 2) + np.sum(c["hh1"] ** 2))
    total = e_low + e_mid + e_high + 1e-6

    return {
        "low": e_low / total,
        "mid": e_mid / total,
        "high": e_high / total,
    }


def wavelet_signature_rgb(gray_img01: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    x, _ = _prepare_for_wavelet(gray_img01)
    c = _decompose_l2(x)

    low_map = cv2.resize(_norm01(c["ll2"]), target_size, interpolation=cv2.INTER_LINEAR)
    mid_raw = np.abs(c["lh2"]) + np.abs(c["hl2"]) + np.abs(c["hh2"])
    mid_map = cv2.resize(_norm01(mid_raw), target_size, interpolation=cv2.INTER_LINEAR)
    high_raw = np.abs(c["lh1"]) + np.abs(c["hl1"]) + np.abs(c["hh1"])
    high_map = cv2.resize(_norm01(high_raw), target_size, interpolation=cv2.INTER_LINEAR)

    rgb = np.stack([high_map, mid_map, low_map], axis=2)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def band_ablation_wavelet(
    infer_fn,
    preprocess_fn,
    img_rgb_uint8: np.ndarray,
) -> Tuple[Dict[str, float], str, np.ndarray]:
    gray = to_gray(img_rgb_uint8)
    x, (orig_h, orig_w) = _prepare_for_wavelet(gray)
    coeffs = _decompose_l2(x)

    p0 = infer_fn(preprocess_fn(img_rgb_uint8))
    deltas: Dict[str, float] = {}

    for band in ("low", "mid", "high"):
        c2 = _copy_coeffs(coeffs)
        if band == "low":
            c2["ll2"].fill(0.0)
        elif band == "mid":
            c2["lh2"].fill(0.0)
            c2["hl2"].fill(0.0)
            c2["hh2"].fill(0.0)
        else:
            c2["lh1"].fill(0.0)
            c2["hl1"].fill(0.0)
            c2["hh1"].fill(0.0)

        restored = _reconstruct_l2(c2)
        if restored.shape != (orig_h, orig_w):
            restored = cv2.resize(restored, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        rgb2 = gray01_to_rgb_uint8(restored)
        pb = infer_fn(preprocess_fn(rgb2))
        deltas[band] = float(pb - p0)

    dominant = max(deltas.keys(), key=lambda k: abs(deltas[k])) if deltas else "unknown"
    wavelet_rgb = wavelet_signature_rgb(gray, (orig_w, orig_h))
    return deltas, dominant, wavelet_rgb


def to_png_data_url(img_rgb_uint8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


__all__ = [
    "resize_with_aspect_ratio",
    "frame_to_jpeg_bytes",
    "video_to_sampled_frames_per_second",
    "video_to_uniform_sampled_frames",
    "to_gray",
    "gray01_to_rgb_uint8",
    "wavelet_band_energy_ratio",
    "wavelet_signature_rgb",
    "band_ablation_wavelet",
    "to_png_data_url",
]
