import json
import hashlib
import base64
import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import scipy.stats as stats
import torch
import torch.nn.functional as F
import requests

from model import (
    detector,
    make_square_bbox_with_margin,
    resize_with_padding,
    build_4ch_srm_y,
)


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


# =========================
# Evidence / Explain helpers
# =========================

def infer_prob_binary(model: torch.nn.Module, x: torch.Tensor) -> float:
    if model is None:
        raise RuntimeError("추론 실패: 모델 인스턴스가 없습니다.")

    model.eval()
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]

    if y.ndim == 2 and y.shape[1] == 2:
        prob_fake = F.softmax(y, dim=1)[:, 1]
        return float(prob_fake.item())

    prob_fake = torch.sigmoid(y.view(-1))
    return float(prob_fake.item())


def fuse_probs(p_rgb: float, p_freq: float, w: float = 0.5) -> float:
    w = float(max(0.0, min(1.0, w)))
    p = (w * float(p_rgb)) + ((1.0 - w) * float(p_freq))
    return float(max(0.0, min(1.0, p)))


def _ensure_224_rgb(img_rgb_uint8: np.ndarray) -> np.ndarray:
    if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
        raise ValueError("RGB uint8 이미지(3채널) 입력이 필요합니다.")
    if img_rgb_uint8.dtype != np.uint8:
        img_rgb_uint8 = np.clip(img_rgb_uint8, 0, 255).astype(np.uint8)
    if img_rgb_uint8.shape[0] == 224 and img_rgb_uint8.shape[1] == 224:
        return img_rgb_uint8
    return cv2.resize(img_rgb_uint8, (224, 224), interpolation=cv2.INTER_LINEAR)


def rgb_preprocess_tensor(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    from PIL import Image

    img = _ensure_224_rgb(img_rgb_uint8)
    pil = Image.fromarray(img)
    return detector.pixel_transform(pil).unsqueeze(0).to(detector.device, non_blocking=True)


def freq_preprocess_tensor(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    img = _ensure_224_rgb(img_rgb_uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    combined_4ch = build_4ch_srm_y(bgr, detector.srm_filters)
    return (
        torch.from_numpy(combined_4ch.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(detector.device, non_blocking=True)
    )


def get_cam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "model") and hasattr(model.model, "features"):
        return model.model.features[-1]
    raise RuntimeError("CAM target layer를 자동 선택할 수 없습니다. 모델 구조를 확인하세요.")


def _extract_landmarks_5pt(face_obj) -> Optional[np.ndarray]:
    kps = getattr(face_obj, "kps", None)
    if kps is not None:
        arr = np.asarray(kps, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 5 and arr.shape[1] >= 2:
            return arr[:5, :2]

    kps106 = getattr(face_obj, "landmark_2d_106", None)
    if kps106 is not None:
        arr = np.asarray(kps106, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 5 and arr.shape[1] >= 2:
            return arr[:5, :2]

    return None


def _default_landmarks_5pt(target_size: int = 224) -> np.ndarray:
    return np.asarray(
        [
            [0.31 * target_size, 0.40 * target_size],
            [0.69 * target_size, 0.40 * target_size],
            [0.50 * target_size, 0.56 * target_size],
            [0.39 * target_size, 0.73 * target_size],
            [0.61 * target_size, 0.73 * target_size],
        ],
        dtype=np.float32,
    )


def _map_landmarks_to_crop(
    landmarks_xy: np.ndarray,
    square_bbox: List[int],
    crop_h: int,
    crop_w: int,
    target_size: int,
) -> np.ndarray:
    x1, y1, _, _ = square_bbox

    scale = float(target_size) / float(max(crop_h, crop_w))
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    pad_t = (target_size - new_h) // 2
    pad_l = (target_size - new_w) // 2

    local = landmarks_xy.astype(np.float32).copy()
    local[:, 0] = (local[:, 0] - float(x1)) * scale + float(pad_l)
    local[:, 1] = (local[:, 1] - float(y1)) * scale + float(pad_t)

    local[:, 0] = np.clip(local[:, 0], 0, target_size - 1)
    local[:, 1] = np.clip(local[:, 1], 0, target_size - 1)
    return local.astype(np.float32)


def _face_area_ratio(face_obj, img_w: int, img_h: int) -> float:
    bbox = np.asarray(getattr(face_obj, "bbox", [0, 0, 0, 0]), dtype=np.float32)
    bw = max(1.0, float(bbox[2] - bbox[0]))
    bh = max(1.0, float(bbox[3] - bbox[1]))
    img_area = max(1.0, float(img_w * img_h))
    return float((bw * bh) / img_area)


def _pose_frontal_score(face_obj) -> Optional[float]:
    pose = getattr(face_obj, "pose", None)
    if pose is None:
        return None

    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None

    yaw = abs(float(arr[0]))
    pitch = abs(float(arr[1]))
    roll = abs(float(arr[2])) if arr.size >= 3 else 0.0

    yaw_score = max(0.0, 1.0 - (yaw / 45.0))
    pitch_score = max(0.0, 1.0 - (pitch / 30.0))
    roll_score = max(0.0, 1.0 - (roll / 40.0))

    return float((0.6 * yaw_score) + (0.3 * pitch_score) + (0.1 * roll_score))


def _landmark_frontal_score(face_obj) -> float:
    lm = _extract_landmarks_5pt(face_obj)
    if lm is None:
        return 0.5

    le, re, nose, ml, mr = lm[0], lm[1], lm[2], lm[3], lm[4]
    eye_dist = float(max(np.linalg.norm(le - re), 1e-6))

    eye_center_x = float((le[0] + re[0]) * 0.5)
    mouth_center_x = float((ml[0] + mr[0]) * 0.5)
    center_x = (eye_center_x + mouth_center_x) * 0.5
    nose_center_dev = abs(float(nose[0]) - float(center_x))
    center_score = max(0.0, 1.0 - (nose_center_dev / (0.35 * eye_dist)))

    nose_to_ml = abs(float(nose[0]) - float(ml[0]))
    mr_to_nose = abs(float(mr[0]) - float(nose[0]))
    den = max(nose_to_ml, mr_to_nose, 1e-6)
    lr_balance = abs(nose_to_ml - mr_to_nose) / den
    symmetry_score = max(0.0, 1.0 - lr_balance)

    eye_center_y = float((le[1] + re[1]) * 0.5)
    mouth_center_y = float((ml[1] + mr[1]) * 0.5)
    y_order_score = 1.0 if (eye_center_y < float(nose[1]) < mouth_center_y) else 0.0

    return float((0.5 * center_score) + (0.4 * symmetry_score) + (0.1 * y_order_score))


def _face_frontal_score(face_obj) -> float:
    pose_score = _pose_frontal_score(face_obj)
    if pose_score is not None:
        return float(max(0.0, min(1.0, pose_score)))
    return float(max(0.0, min(1.0, _landmark_frontal_score(face_obj))))


def _rank_faces_by_primary_priority(
    faces: List[Any],
    img_w: int,
    img_h: int,
) -> List[Any]:
    if not faces:
        return []

    scored = []
    area_ratios = []
    for f in faces:
        ar = _face_area_ratio(f, img_w=img_w, img_h=img_h)
        area_ratios.append(ar)
        scored.append({"face": f, "area_ratio": ar, "frontal": _face_frontal_score(f)})

    max_area = max(area_ratios) if area_ratios else 1.0
    max_area = max(max_area, 1e-6)

    for item in scored:
        area_norm = float(item["area_ratio"] / max_area)
        frontal = float(item["frontal"])
        # 큰 얼굴을 우선하되, 비슷한 크기라면 정면성 높은 얼굴 선택.
        item["priority"] = float((0.8 * area_norm) + (0.2 * frontal))

    scored.sort(key=lambda x: (x["priority"], x["area_ratio"], x["frontal"]), reverse=True)
    return [x["face"] for x in scored]


def detect_faces_with_aligned_crops(
    image_bgr: np.ndarray,
    margin: float = 0.15,
    target_size: int = 224,
    max_faces: int = 8,
    prioritize_frontal: bool = False,
) -> List[Dict[str, np.ndarray]]:
    face_app = getattr(getattr(detector, "face_cropper", None), "app", None)
    if face_app is None:
        raise RuntimeError("InsightFace 초기화 실패: detector.face_cropper.app를 찾을 수 없습니다.")

    faces = face_app.get(image_bgr)
    if not faces:
        return []

    img_h, img_w = image_bgr.shape[:2]
    if prioritize_frontal:
        faces_ranked = _rank_faces_by_primary_priority(faces, img_w=img_w, img_h=img_h)
    else:
        faces_ranked = sorted(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            reverse=True,
        )
    limit = max(1, int(max_faces))
    out: List[Dict[str, np.ndarray]] = []

    for face in faces_ranked[:limit]:
        bbox = np.asarray(face.bbox, dtype=np.float32)
        square_bbox = make_square_bbox_with_margin(
            bbox.tolist(),
            margin=margin,
            img_width=img_w,
            img_height=img_h,
        )

        x1, y1, x2, y2 = [int(v) for v in square_bbox]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_224_bgr = resize_with_padding(crop, target_size=target_size)
        crop_224_rgb = cv2.cvtColor(crop_224_bgr, cv2.COLOR_BGR2RGB)

        landmarks = _extract_landmarks_5pt(face)
        if landmarks is None:
            lm_crop = _default_landmarks_5pt(target_size=target_size)
        else:
            lm_crop = _map_landmarks_to_crop(
                landmarks_xy=landmarks,
                square_bbox=square_bbox,
                crop_h=crop.shape[0],
                crop_w=crop.shape[1],
                target_size=target_size,
            )

        out.append(
            {
                "crop_rgb": crop_224_rgb,
                "landmarks": lm_crop,
                "bbox": bbox,
            }
        )

    return out


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self._features: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(_, __, output):
            self._features = output

        def bwd_hook(_, grad_in, grad_out):
            _ = grad_in
            self._grads = grad_out[0]

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def close(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)

        y = self.model(x)
        if isinstance(y, (tuple, list)) and y:
            y = y[0]

        if y.ndim == 2 and y.shape[1] == 2:
            idx = 1 if class_idx is None else int(class_idx)
            score = y[:, idx].sum()
        else:
            score = y.reshape(-1).sum()

        score.backward(retain_graph=False)

        feats = self._features
        grads = self._grads
        if feats is None or grads is None:
            raise RuntimeError("GradCAM hook에서 feature/gradient를 수집하지 못했습니다.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feats).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()[0, 0]
        cam_np = cv2.resize(cam_np, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_LINEAR)
        cam_np = cam_np - cam_np.min()
        cam_np = cam_np / (cam_np.max() + 1e-6)
        return cam_np.astype(np.float32)


def overlay_cam(rgb_img_uint8: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    hmap = (np.clip(heatmap01, 0.0, 1.0) * 255).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    out = (rgb_img_uint8 * (1.0 - alpha) + hmap * alpha).astype(np.uint8)
    return out


def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h, y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _box_mask(h: int, w: int, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def build_region_masks_from_5pt(landmarks: np.ndarray, h: int, w: int) -> Dict[str, np.ndarray]:
    lm = landmarks.astype(np.float32)
    le, re, nose, ml, mr = lm[0], lm[1], lm[2], lm[3], lm[4]

    eye_pad_x = 0.12 * w
    eye_pad_y = 0.08 * h
    le_box = _clamp_box(le[0] - eye_pad_x, le[1] - eye_pad_y, le[0] + eye_pad_x, le[1] + eye_pad_y, w, h)
    re_box = _clamp_box(re[0] - eye_pad_x, re[1] - eye_pad_y, re[0] + eye_pad_x, re[1] + eye_pad_y, w, h)
    eyes_mask = np.maximum(_box_mask(h, w, le_box), _box_mask(h, w, re_box))

    nose_pad_x = 0.10 * w
    nose_pad_y = 0.12 * h
    nose_box = _clamp_box(
        nose[0] - nose_pad_x,
        nose[1] - nose_pad_y,
        nose[0] + nose_pad_x,
        nose[1] + nose_pad_y,
        w,
        h,
    )
    nose_mask = _box_mask(h, w, nose_box)

    mx1, mx2 = min(ml[0], mr[0]), max(ml[0], mr[0])
    my = (ml[1] + mr[1]) / 2.0
    mouth_pad_x = 0.08 * w
    mouth_pad_y = 0.14 * h
    mouth_box = _clamp_box(mx1 - mouth_pad_x, my - mouth_pad_y, mx2 + mouth_pad_x, my + mouth_pad_y, w, h)
    mouth_mask = _box_mask(h, w, mouth_box)

    forehead_mask = _box_mask(h, w, _clamp_box(0, 0, w, int(0.35 * h), w, h))
    jawline_mask = _box_mask(h, w, _clamp_box(0, int(0.65 * h), w, h, w, h))

    union = np.clip(eyes_mask + nose_mask + mouth_mask + forehead_mask + jawline_mask, 0, 1).astype(np.uint8)
    cheeks_mask = (1 - union).astype(np.uint8)

    return {
        "eyes": eyes_mask,
        "nose": nose_mask,
        "mouth": mouth_mask,
        "forehead": forehead_mask,
        "jawline": jawline_mask,
        "cheeks": cheeks_mask,
    }


def region_importance_from_heatmap(heatmap01: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    h = heatmap01.astype(np.float32)
    denom = float(h.sum() + 1e-6)
    out: Dict[str, float] = {}
    for k, m in masks.items():
        out[k] = float((h * m.astype(np.float32)).sum() / denom)
    return out


def _blur_region(img_rgb_uint8: np.ndarray, mask01: np.ndarray, ksize: int = 31) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img_rgb_uint8, (ksize, ksize), 0)
    mask3 = np.repeat(mask01[..., None], 3, axis=2).astype(np.uint8)
    out = img_rgb_uint8.copy()
    out[mask3 == 1] = blurred[mask3 == 1]
    return out


def occlusion_validate_topk(
    infer_fn,
    preprocess_fn,
    img_rgb_uint8: np.ndarray,
    region_masks: Dict[str, np.ndarray],
    ranked_regions: List[str],
    k: int = 2,
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    p0 = infer_fn(preprocess_fn(img_rgb_uint8))
    for r in ranked_regions[: max(1, int(k))]:
        occ = _blur_region(img_rgb_uint8, region_masks[r], ksize=31)
        pr = infer_fn(preprocess_fn(occ))
        deltas[r] = float(pr - p0)
    return deltas


def estimate_outside_face_ratio(heatmap01: np.ndarray, landmarks: np.ndarray) -> float:
    h, w = heatmap01.shape
    lm = landmarks.astype(np.float32)
    face_mask = np.zeros((h, w), dtype=np.uint8)

    x_min = float(np.clip(np.min(lm[:, 0]) - (0.20 * w), 0, w - 1))
    x_max = float(np.clip(np.max(lm[:, 0]) + (0.20 * w), 1, w))
    y_min = float(np.clip(np.min(lm[:, 1]) - (0.30 * h), 0, h - 1))
    y_max = float(np.clip(np.max(lm[:, 1]) + (0.35 * h), 1, h))

    cx = int((x_min + x_max) * 0.5)
    cy = int((y_min + y_max) * 0.5)
    ax = max(2, int((x_max - x_min) * 0.5))
    ay = max(2, int((y_max - y_min) * 0.6))
    cv2.ellipse(face_mask, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)

    hmap = np.clip(heatmap01.astype(np.float32), 0.0, 1.0)
    denom = float(hmap.sum() + 1e-6)
    outside = float((hmap * (1 - face_mask).astype(np.float32)).sum() / denom)
    return float(max(0.0, min(1.0, outside)))


def estimate_localization_confidence(top_importance: float, outside_face_ratio: float) -> str:
    if top_importance >= 0.25 and outside_face_ratio <= 0.20:
        return "high"
    if top_importance >= 0.16 and outside_face_ratio <= 0.35:
        return "med"
    return "low"


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


def _round6(value: float) -> float:
    return float(round(float(value), 6))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _extract_responses_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return _normalize_text(output_text)
    if isinstance(output_text, list):
        parts = [str(x).strip() for x in output_text if isinstance(x, str)]
        if parts:
            return _normalize_text(" ".join(parts))

    parts: List[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return _normalize_text(" ".join(parts))


def _extract_chat_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return _normalize_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return _normalize_text(" ".join(parts))
    return ""


def _call_openai_comment(system_prompt: str, user_prompt: str, max_output_tokens: int = 200) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1").rstrip("/")
    timeout_sec = _env_float("OPENAI_TIMEOUT_SEC", 20.0)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        # 우선 Responses API 시도
        resp = requests.post(
            f"{base_url}/responses",
            headers=headers,
            json={
                "model": model,
                "temperature": 0.3,
                "max_output_tokens": int(max_output_tokens),
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=timeout_sec,
        )
        if resp.ok:
            text = _extract_responses_text(resp.json())
            if text:
                return text
    except Exception:
        pass

    try:
        # 구버전/호환 경로 fallback
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "temperature": 0.3,
                "max_tokens": int(max_output_tokens),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=timeout_sec,
        )
        if resp.ok:
            text = _extract_chat_text(resp.json())
            if text:
                return text
    except Exception:
        pass

    return None


def generate_image_ai_comment(
    fake_prob: float,
    real_prob: float,
    top_regions: List[str],
    dominant_band_label: str,
    energy_low_pct: float,
    energy_mid_pct: float,
    energy_high_pct: float,
) -> Optional[str]:
    system_prompt = (
        "너는 딥페이크 판독 결과를 사용자에게 전달하는 한국어 리포터다. "
        "출력은 1~2문장으로 짧고 자연스럽게 작성하고, 어색한 비유/은유 표현은 금지한다. "
        "확정 단정 대신 가능성 중심으로 표현한다."
    )

    region_text = ", ".join(top_regions) if top_regions else "얼굴 핵심 부위"
    user_prompt = (
        f"최종 fake 확률 {fake_prob*100:.1f}%, real 확률 {real_prob*100:.1f}%.\n"
        f"주요 부위: {region_text}\n"
        f"우세 대역: {dominant_band_label}\n"
        f"밴드 에너지: low {energy_low_pct:.1f}%, mid {energy_mid_pct:.1f}%, high {energy_high_pct:.1f}%\n"
        "사용자용 AI 코멘트를 작성해줘. 전문적이되 딱딱하지 않게 작성하고, 의미 없는 수식어는 생략해."
    )
    return _call_openai_comment(system_prompt=system_prompt, user_prompt=user_prompt, max_output_tokens=180)


def _series_stats(values: List[float]) -> Optional[Dict[str, float]]:
    arr = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    if not arr:
        return None

    start = arr[0]
    mid = arr[(len(arr) - 1) // 2]
    end = arr[-1]
    swing = max(arr) - min(arr)
    drift = end - start
    trend = "상승" if drift > 3 else ("하강" if drift < -3 else "유지")

    return {
        "start": float(start),
        "mid": float(mid),
        "end": float(end),
        "swing": float(swing),
        "drift": float(drift),
        "trend": trend,
    }


def generate_video_ai_comment(
    final_scores: List[float],
    pixel_scores: List[float],
    freq_scores: List[float],
    is_fake: Optional[bool],
) -> Optional[str]:
    final_stats = _series_stats(final_scores)
    pixel_stats = _series_stats(pixel_scores)
    freq_stats = _series_stats(freq_scores)
    if final_stats is None:
        return None

    system_prompt = (
        "너는 딥페이크 영상 판독 결과를 사용자에게 전달하는 한국어 리포터다. "
        "출력은 1~2문장으로 짧고 자연스럽게 작성한다. "
        "어색한 비유/은유, 과장, 단정적 표현은 금지한다."
    )

    verdict = (
        "조작 가능성 쪽으로 기울었습니다."
        if is_fake is True
        else "원본 가능성 쪽으로 기울었습니다."
        if is_fake is False
        else "추가 검증이 필요합니다."
    )

    def _fmt(stats_obj: Optional[Dict[str, float]], label: str) -> str:
        if not stats_obj:
            return f"{label}: 데이터 부족"
        return (
            f"{label}: 시작 {stats_obj['start']:.1f}%, 중간 {stats_obj['mid']:.1f}%, "
            f"종료 {stats_obj['end']:.1f}%, 추세 {stats_obj['trend']}, 변동폭 {stats_obj['swing']:.1f}%"
        )

    user_prompt = (
        f"{_fmt(final_stats, '최종')}\n"
        f"{_fmt(pixel_stats, '픽셀')}\n"
        f"{_fmt(freq_stats, '주파수')}\n"
        f"판정 방향: {verdict}\n"
        "사용자에게 보여줄 AI 코멘트를 작성해줘. 이미지 코멘트 톤과 동일하게 간결하고 자연스럽게 작성해."
    )
    return _call_openai_comment(system_prompt=system_prompt, user_prompt=user_prompt, max_output_tokens=180)


def build_evidence_for_face(
    face_rgb_uint8: np.ndarray,
    landmarks: np.ndarray,
    rgb_model: torch.nn.Module,
    freq_model: torch.nn.Module,
    cam: GradCAM,
    fusion_w: float = 0.5,
    evidence_level: str = "mvp",
) -> Dict[str, Any]:
    x_rgb = rgb_preprocess_tensor(face_rgb_uint8)
    x_freq = freq_preprocess_tensor(face_rgb_uint8)

    p_rgb = infer_prob_binary(rgb_model, x_rgb)
    p_freq = infer_prob_binary(freq_model, x_freq)
    p_final = fuse_probs(p_rgb, p_freq, w=fusion_w)

    heat = cam(x_rgb, class_idx=1)

    h, w, _ = face_rgb_uint8.shape
    masks = build_region_masks_from_5pt(landmarks, h, w)
    region_imp = region_importance_from_heatmap(heat, masks)
    ranked = sorted(region_imp.keys(), key=lambda k: region_imp[k], reverse=True)

    spatial_notes = ["insightface_aligned_crop"]
    occ_deltas: Dict[str, float] = {}
    if evidence_level != "off" and p_final >= 0.60:
        occ_deltas = occlusion_validate_topk(
            infer_fn=lambda t: infer_prob_binary(rgb_model, t),
            preprocess_fn=rgb_preprocess_tensor,
            img_rgb_uint8=face_rgb_uint8,
            region_masks=masks,
            ranked_regions=ranked,
            k=2,
        )
    elif evidence_level == "off":
        spatial_notes.append("occlusion_skipped:evidence_off")
    else:
        spatial_notes.append("occlusion_skipped:low_fake_prob")

    regions_topk = []
    for r in ranked[:3]:
        regions_topk.append(
            {
                "region": r,
                "importance_cam": _round6(region_imp[r]),
                "delta_occlusion": _round6(occ_deltas[r]) if r in occ_deltas else None,
            }
        )

    outside_face_ratio = estimate_outside_face_ratio(heat, landmarks)
    top_importance = regions_topk[0]["importance_cam"] if regions_topk else 0.0
    localization_conf = estimate_localization_confidence(float(top_importance), float(outside_face_ratio))

    band_deltas: Dict[str, float] = {}
    dominant_band = "unknown"
    gray = to_gray(face_rgb_uint8)
    energy_ratio_map = wavelet_band_energy_ratio(gray)
    dominant_energy_band = (
        max(energy_ratio_map.keys(), key=lambda k: energy_ratio_map[k]) if energy_ratio_map else "unknown"
    )

    freq_notes = ["wavelet_haar_l2_ablation"]
    if evidence_level != "off" and p_final >= 0.60:
        band_deltas, dominant_band, _ = band_ablation_wavelet(
            infer_fn=lambda t: infer_prob_binary(freq_model, t),
            preprocess_fn=freq_preprocess_tensor,
            img_rgb_uint8=face_rgb_uint8,
        )
    elif evidence_level == "off":
        freq_notes.append("ablation_skipped:evidence_off")
    else:
        freq_notes.append("ablation_skipped:low_fake_prob")

    band_order = ["low", "mid", "high"]
    band_list = []
    for b in band_order:
        if b in band_deltas:
            band_list.append({"band": b, "delta_fake_prob": _round6(band_deltas[b])})
    band_energy = []
    for b in band_order:
        if b in energy_ratio_map:
            band_energy.append({"band": b, "energy_ratio": _round6(energy_ratio_map[b])})

    assets = {
        "face_crop_url": to_png_data_url(face_rgb_uint8),
    }

    evidence = {
        "spatial": {
            "regions_topk": regions_topk,
            "outside_face_ratio": _round6(outside_face_ratio),
            "localization_confidence": localization_conf,
            "notes": spatial_notes,
        },
        "frequency": {
            "band_ablation": band_list,
            "dominant_band": dominant_band,
            "band_energy": band_energy,
            "dominant_energy_band": dominant_energy_band,
            "method": "wavelet_haar_l2",
            "notes": freq_notes,
        },
    }

    return {
        "score": {"p_rgb": _round6(p_rgb), "p_freq": _round6(p_freq), "p_final": _round6(p_final)},
        "assets": assets,
        "evidence": evidence,
    }


def explain_from_evidence(
    evidence: Dict[str, Any],
    score: Dict[str, float],
    use_openai: bool = True,
) -> Dict[str, Any]:
    spatial = evidence.get("spatial", {})
    freq = evidence.get("frequency", {})

    region_label = {
        "eyes": "눈 주변",
        "nose": "코 주변",
        "mouth": "입 주변",
        "forehead": "이마",
        "jawline": "턱선",
        "cheeks": "볼",
    }
    band_label = {"low": "저주파", "mid": "중주파", "high": "고주파", "unknown": "미확정"}

    def _region(r: str) -> str:
        rr = str(r or "미확정")
        return region_label.get(rr, rr)

    def _band(b: str) -> str:
        bb = str(b or "unknown")
        return band_label.get(bb, bb)

    top = spatial.get("regions_topk", [])[:2]
    dom = str(freq.get("dominant_band", "unknown"))
    band_map = {x["band"]: x["delta_fake_prob"] for x in freq.get("band_ablation", []) if "band" in x}
    energy_map = {x["band"]: x["energy_ratio"] for x in freq.get("band_energy", []) if "band" in x}
    energy_dom = str(freq.get("dominant_energy_band", "unknown"))

    fake_prob = float(score.get("p_final", 0.0))
    fake_prob = max(0.0, min(1.0, fake_prob))
    real_prob = 1.0 - fake_prob
    is_fake_mode = fake_prob >= 0.5
    low = float(energy_map.get("low", 0.0)) * 100.0
    mid = float(energy_map.get("mid", 0.0)) * 100.0
    high = float(energy_map.get("high", 0.0)) * 100.0

    top_regions_kor = [_region(item.get("region", "")) for item in top if item.get("region")]
    region_hint = "얼굴 핵심 부위"
    if top_regions_kor:
        region_hint = ", ".join(top_regions_kor)

    band_hint = _band(dom if dom != "unknown" else energy_dom)
    if is_fake_mode:
        summary = (
            f"{region_hint}에서 미세 경계와 질감의 불연속이 관측되고 "
            f"{band_hint} 대역 신호 편차도 함께 나타나, 이번 샘플은 조작 가능성이 높게 관측됩니다."
        )
    else:
        summary = (
            f"{region_hint}의 질감 흐름과 {band_hint} 대역 분포가 전반적으로 일관되어, "
            "이번 샘플은 원본 가능성이 우세합니다."
        )

    if use_openai:
        llm_summary = generate_image_ai_comment(
            fake_prob=fake_prob,
            real_prob=real_prob,
            top_regions=top_regions_kor,
            dominant_band_label=band_hint,
            energy_low_pct=low,
            energy_mid_pct=mid,
            energy_high_pct=high,
        )
        if llm_summary:
            summary = llm_summary

    spatial_findings = []
    for item in top:
        region = _region(item.get("region", "face"))
        importance = float(item.get("importance_cam", 0.0))
        claim = f"{region} 부위가 판별의 핵심 단서로 반영되었습니다."
        evidence_txt = f"CAM {importance:.2f}"
        delta = item.get("delta_occlusion")
        if delta is not None:
            delta_f = float(delta) * 100.0
            direction = "증가" if delta_f > 0 else ("감소" if delta_f < 0 else "변화 거의 없음")
            evidence_txt += f", occlusion 시 fake 확률 {abs(delta_f):.1f}% {direction}"
        spatial_findings.append({"claim": claim, "evidence": evidence_txt})

    outside_face_ratio = spatial.get("outside_face_ratio", None)
    localization_conf = str(spatial.get("localization_confidence", "unknown"))
    if outside_face_ratio is not None:
        try:
            outside_pct = float(outside_face_ratio) * 100.0
            if outside_pct <= 25.0:
                claim = "근거가 얼굴 중심에 비교적 잘 모여 있습니다."
            else:
                claim = "근거가 얼굴 외곽에도 일부 분산되어 해석 시 주의가 필요합니다."
            evidence_txt = f"outside-face ratio {outside_pct:.1f}%, localization {localization_conf}"
            spatial_findings.append({"claim": claim, "evidence": evidence_txt})
        except Exception:
            pass

    if not spatial_findings:
        spatial_findings.append(
            {
                "claim": "얼굴 전반 패턴을 기반으로 판별했습니다.",
                "evidence": "부위별 상위 근거가 제한되어 전체 정보를 함께 활용했습니다.",
            }
        )

    frequency_findings = []
    if dom in band_map:
        delta_f = float(band_map[dom]) * 100.0
        direction = "증가" if delta_f > 0 else ("감소" if delta_f < 0 else "변화 거의 없음")
        frequency_findings.append(
            {
                "claim": f"{_band(dom)} 대역이 예측 민감도에 크게 작용했습니다.",
                "evidence": f"{_band(dom)} 제거 시 fake 확률 {abs(delta_f):.1f}% {direction}",
            }
        )
    else:
        frequency_findings.append(
            {
                "claim": "대역 제거 실험의 변화가 제한적이었습니다.",
                "evidence": "band ablation 변화량이 작거나 계산되지 않았습니다.",
            }
        )

    frequency_findings.append(
        {
            "claim": f"에너지 우세 대역은 {_band(energy_dom)}입니다.",
            "evidence": f"low {low:.1f}%, mid {mid:.1f}%, high {high:.1f}%",
        }
    )

    if dom != "unknown" and energy_dom != "unknown":
        consistency = "일관" if dom == energy_dom else "부분 불일치"
        frequency_findings.append(
            {
                "claim": "주파수 민감도와 에너지 우세 대역의 합치도를 확인했습니다.",
                "evidence": f"dominant {_band(dom)} / energy-dominant {_band(energy_dom)} ({consistency})",
            }
        )

    frequency_findings.append(
        {
            "claim": "최종 확률 축에서도 같은 방향의 결론이 확인됩니다.",
            "evidence": f"fake {fake_prob*100.0:.1f}%, real {real_prob*100.0:.1f}%",
        }
    )

    freq_notes = freq.get("notes", [])
    spatial_notes = spatial.get("notes", [])
    caveats = [
        "강한 압축이나 저해상도는 주파수 패턴을 왜곡해 오탐/미탐을 늘릴 수 있습니다.",
        "자동 판별은 보조 근거입니다. 중요한 의사결정은 추가 검증과 함께 진행하세요.",
    ]
    if any("skipped" in str(note) for note in (freq_notes or [])) or any(
        "skipped" in str(note) for note in (spatial_notes or [])
    ):
        caveats.insert(0, "일부 근거 실험이 생략되어, 이번 결과는 보수적으로 해석하는 편이 안전합니다.")

    return {
        "summary": summary,
        "spatial_findings": spatial_findings[:4],
        "frequency_findings": frequency_findings[:4],
        "next_steps": [
            "원본에 가까운 고해상도 파일(재인코딩 전)로 한 번 더 교차 검증하세요.",
            "가능하면 다른 각도/조명 샘플을 추가해 같은 결론이 반복되는지 확인하세요.",
        ],
        "caveats": caveats[:3],
    }
