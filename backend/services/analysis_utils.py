"""Reusable helper utilities for analysis workflows."""

from fastapi import HTTPException

from .analysis_config import AGG_MODE_VIDEO, TOPK, VIDEO_AGG_MODE_LABEL


def validate_evidence_level(level: str) -> str:
    lv = (level or "mvp").strip().lower()
    if lv not in {"off", "mvp", "full"}:
        raise HTTPException(status_code=400, detail="evidence_level은 off/mvp/full 중 하나여야 합니다.")
    return lv


def safe_score_agg(values):
    return float(max(values)) if values else 0.0


def fake_prob_to_real_percent(p_fake: float) -> float:
    p = float(p_fake)
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return float((1.0 - p) * 100.0)


def normalize_pixel_weight(pixel_weight: float) -> float:
    w = float(pixel_weight)
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    return w


def model_weighted_confidence(pixel_real: float, freq_real: float, pixel_weight: float) -> float:
    # model.py의 앙상블 방식과 동일하게 계산한다.
    w = normalize_pixel_weight(pixel_weight)
    return float((float(pixel_real) * w) + (float(freq_real) * (1.0 - w)))


def build_video_face_not_detected_result(
    sampled_frames: int,
    failed_frames: int,
    sampling_meta: dict | None = None,
    frame_failure_reasons: dict | None = None,
) -> dict:
    result = {
        "confidence": None,
        "pixel_score": None,
        "freq_score": None,
        "is_fake": None,
        "p_value": None,
        "reliability": "",
        "video_representative_confidence": None,
        "video_frame_confidences": [],
        "video_frame_pixel_scores": [],
        "video_frame_freq_scores": [],
        "video_meta": {
            "sampled_frames": int(sampled_frames),
            "used_frames": 0,
            "failed_frames": int(failed_frames),
            "agg_mode": VIDEO_AGG_MODE_LABEL,
            "pixel_freq_agg_mode": AGG_MODE_VIDEO,
            "topk": TOPK,
        },
        "ai_comment": "얼굴이 감지되지 않아 추론을 완료하지 못했습니다. 다른 구도/해상도의 영상을 사용해 다시 시도해 주세요.",
        "ai_comment_source": "rule_based",
        "input_media_type": "video",
        "inference_failed": True,
        "failure_reason": "face_not_detected",
    }
    if isinstance(sampling_meta, dict):
        result["video_meta"].update(sampling_meta)
    if isinstance(frame_failure_reasons, dict) and frame_failure_reasons:
        result["video_meta"]["frame_failure_reasons"] = dict(frame_failure_reasons)
    return result

