"""Evidence construction utilities built on top of inference/media outputs."""

from typing import Any, Callable, Dict, List

import cv2
import numpy as np
import torch

from .inference import (
    GradCAM,
    freq_preprocess_tensor,
    fuse_probs,
    infer_prob_binary,
    overlay_cam,
    rgb_preprocess_tensor,
)
from .video_utils import band_ablation_wavelet, to_gray, to_png_data_url, wavelet_band_energy_ratio


def _round6(value: float) -> float:
    return float(round(float(value), 6))


def _encode_png_data_url_or_raise(img_rgb_uint8: np.ndarray, asset_name: str) -> str:
    data_url = to_png_data_url(img_rgb_uint8)
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
        raise RuntimeError(f"{asset_name} PNG 인코딩 실패")
    return data_url


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
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


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
    heatmap = heatmap01.astype(np.float32)
    denom = float(heatmap.sum() + 1e-6)
    out: Dict[str, float] = {}
    for key, mask in masks.items():
        out[key] = float((heatmap * mask.astype(np.float32)).sum() / denom)
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
    infer_fn: Callable[[torch.Tensor], float],
    preprocess_fn: Callable[[np.ndarray], torch.Tensor],
    img_rgb_uint8: np.ndarray,
    region_masks: Dict[str, np.ndarray],
    ranked_regions: List[str],
    k: int = 2,
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    p0 = infer_fn(preprocess_fn(img_rgb_uint8))
    for region in ranked_regions[: max(1, int(k))]:
        occ = _blur_region(img_rgb_uint8, region_masks[region], ksize=31)
        pr = infer_fn(preprocess_fn(occ))
        deltas[region] = float(pr - p0)
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


def build_evidence_for_face(
    face_rgb_uint8: np.ndarray,
    landmarks: np.ndarray,
    rgb_model: torch.nn.Module,
    freq_model: torch.nn.Module,
    cam: GradCAM,
    fusion_w: float = 0.5,
    evidence_level: str = "mvp",
) -> Dict[str, Any]:
    if face_rgb_uint8 is None or not isinstance(face_rgb_uint8, np.ndarray):
        raise ValueError("face_rgb_uint8 입력이 올바르지 않습니다.")
    if face_rgb_uint8.ndim != 3 or face_rgb_uint8.shape[2] != 3:
        raise ValueError("face_rgb_uint8은 (H,W,3) RGB uint8 형태여야 합니다.")

    x_rgb = rgb_preprocess_tensor(face_rgb_uint8)
    x_freq = freq_preprocess_tensor(face_rgb_uint8)

    p_rgb = infer_prob_binary(rgb_model, x_rgb)
    p_freq = infer_prob_binary(freq_model, x_freq)
    p_final = fuse_probs(p_rgb, p_freq, w=fusion_w)

    try:
        heat = cam(x_rgb, class_idx=1)
    except Exception as exc:
        raise RuntimeError(f"Grad-CAM 계산 실패: {exc}") from exc

    heat = np.asarray(heat, dtype=np.float32)
    if heat.ndim != 2 or heat.size == 0:
        raise RuntimeError(f"Grad-CAM 출력 형식 오류: shape={getattr(heat, 'shape', None)}")
    if not np.isfinite(heat).all():
        heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)
    heat = np.clip(heat, 0.0, 1.0)

    gradcam_overlay_rgb = overlay_cam(face_rgb_uint8, heat, alpha=0.45)

    h, w, _ = face_rgb_uint8.shape
    masks = build_region_masks_from_5pt(landmarks, h, w)
    region_imp = region_importance_from_heatmap(heat, masks)
    ranked = sorted(region_imp.keys(), key=lambda k: region_imp[k], reverse=True)

    spatial_notes = ["insightface_aligned_crop"]
    occ_deltas: Dict[str, float] = {}
    if evidence_level != "off" and p_final >= 0.60:
        try:
            occ_deltas = occlusion_validate_topk(
                infer_fn=lambda tensor: infer_prob_binary(rgb_model, tensor),
                preprocess_fn=rgb_preprocess_tensor,
                img_rgb_uint8=face_rgb_uint8,
                region_masks=masks,
                ranked_regions=ranked,
                k=2,
            )
        except Exception as exc:
            spatial_notes.append(f"occlusion_failed:{type(exc).__name__}")
    elif evidence_level == "off":
        spatial_notes.append("occlusion_skipped:evidence_off")
    else:
        spatial_notes.append("occlusion_skipped:low_fake_prob")

    regions_topk = []
    for region in ranked[:3]:
        regions_topk.append(
            {
                "region": region,
                "importance_cam": _round6(region_imp[region]),
                "delta_occlusion": _round6(occ_deltas[region]) if region in occ_deltas else None,
            }
        )

    outside_face_ratio = estimate_outside_face_ratio(heat, landmarks)
    top_importance = regions_topk[0]["importance_cam"] if regions_topk else 0.0
    localization_conf = estimate_localization_confidence(float(top_importance), float(outside_face_ratio))

    band_deltas: Dict[str, float] = {}
    dominant_band = "unknown"
    energy_ratio_map: Dict[str, float] = {}
    dominant_energy_band = "unknown"

    freq_notes = ["wavelet_haar_l2_ablation"]
    try:
        gray = to_gray(face_rgb_uint8)
        energy_ratio_map = wavelet_band_energy_ratio(gray)
        dominant_energy_band = (
            max(energy_ratio_map.keys(), key=lambda k: energy_ratio_map[k]) if energy_ratio_map else "unknown"
        )
    except Exception as exc:
        freq_notes.append(f"wavelet_energy_failed:{type(exc).__name__}")

    if evidence_level != "off" and p_final >= 0.60:
        try:
            band_deltas, dominant_band, _ = band_ablation_wavelet(
                infer_fn=lambda tensor: infer_prob_binary(freq_model, tensor),
                preprocess_fn=freq_preprocess_tensor,
                img_rgb_uint8=face_rgb_uint8,
            )
        except Exception as exc:
            freq_notes.append(f"ablation_failed:{type(exc).__name__}")
    elif evidence_level == "off":
        freq_notes.append("ablation_skipped:evidence_off")
    else:
        freq_notes.append("ablation_skipped:low_fake_prob")

    band_order = ["low", "mid", "high"]
    band_list = []
    for band in band_order:
        if band in band_deltas:
            band_list.append({"band": band, "delta_fake_prob": _round6(band_deltas[band])})

    band_energy = []
    for band in band_order:
        if band in energy_ratio_map:
            band_energy.append({"band": band, "energy_ratio": _round6(energy_ratio_map[band])})

    assets = {
        "face_crop_url": _encode_png_data_url_or_raise(face_rgb_uint8, "face_crop"),
        "gradcam_overlay_url": _encode_png_data_url_or_raise(gradcam_overlay_rgb, "gradcam_overlay"),
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


__all__ = [
    "build_region_masks_from_5pt",
    "region_importance_from_heatmap",
    "occlusion_validate_topk",
    "estimate_outside_face_ratio",
    "estimate_localization_confidence",
    "build_evidence_for_face",
]
