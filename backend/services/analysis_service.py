import os
import secrets
import tempfile
import uuid
import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException

from model import detector
from .redis_client import redis_db
from .evidence import (
    build_evidence_for_face,
)
from .explain import (
    explain_from_evidence,
    generate_video_ai_comment,
    sanitize_ai_comment,
)
from .inference import (
    GradCAM,
    detect_faces_with_aligned_crops,
    get_cam_target_layer,
)
from .stats import (
    aggregate_scores,
    build_analysis_result,
    trimmed_mean_confidence,
)
from .storage import (
    redis_get_json,
    redis_set_json,
    sha256_bytes,
)
from .url_media_utils import download_media_from_url
from .video_utils import (
    video_to_uniform_sampled_frames,
)


# --- Configuration ---
REAL_MEAN = 15.0
REAL_STD = 8.0

# Video sampling
VIDEO_MAX_SIDE = 640
VIDEO_MIN_FRAMES = 12
VIDEO_MAX_FRAMES_CAP = 48
VIDEO_FRAMES_PER_MINUTE = 24

# Aggregation
AGG_MODE_VIDEO = "mean"
TOPK = 5
VIDEO_TRIM_LOW_RATIO = 0.10
VIDEO_TRIM_HIGH_RATIO = 0.30
VIDEO_AGG_MODE_LABEL = "Trimmed Mean (Low 10 Percent, High 30 Percent)"

# Redis TTL
RESULT_TTL_SEC = 3600
CACHE_TTL_SEC = 24 * 3600


# =========================
# Shared helpers
# =========================

def ping_redis() -> None:
    redis_db.ping()


def store_result_and_make_response(analysis_result: dict, stored_result: dict = None) -> dict:
    token = secrets.token_urlsafe(16)
    payload_for_store = stored_result if stored_result is not None else analysis_result
    redis_set_json(redis_db, f"res:{token}", payload_for_store, ex=RESULT_TTL_SEC)
    return {
        "result_url": f"http://127.0.0.1:8000/get-result/{token}",
        "data": analysis_result,
    }


def has_frame_series(payload: dict, key: str) -> bool:
    values = payload.get(key)
    return isinstance(values, list) and len(values) >= 2


def delete_keys_by_patterns(patterns, batch_size: int = 500) -> int:
    deleted_total = 0

    for pattern in patterns:
        batch = []
        for key in redis_db.scan_iter(match=pattern, count=1000):
            batch.append(key)
            if len(batch) >= batch_size:
                deleted_total += int(redis_db.delete(*batch))
                batch.clear()

        if batch:
            deleted_total += int(redis_db.delete(*batch))

    return deleted_total


def clear_cache_entries() -> dict:
    patterns = ["cache:*", "res:*"]
    deleted_count = delete_keys_by_patterns(patterns)
    return {
        "message": "Redis cache cleared",
        "deleted_keys": deleted_count,
        "patterns": patterns,
    }


def get_result_by_token(token: str) -> dict:
    data = redis_get_json(redis_db, f"res:{token}")
    if data is None:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    return data


def _validate_evidence_level(level: str) -> str:
    lv = (level or "mvp").strip().lower()
    if lv not in {"off", "mvp", "full"}:
        raise HTTPException(status_code=400, detail="evidence_level은 off/mvp/full 중 하나여야 합니다.")
    return lv


def _safe_score_agg(values):
    return float(max(values)) if values else 0.0


def _fake_prob_to_real_percent(p_fake: float) -> float:
    p = float(p_fake)
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return float((1.0 - p) * 100.0)


def _normalize_pixel_weight(pixel_weight: float) -> float:
    w = float(pixel_weight)
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    return w


def _model_weighted_confidence(pixel_real: float, freq_real: float, pixel_weight: float) -> float:
    # model.py의 앙상블 방식과 동일하게 계산한다.
    w = _normalize_pixel_weight(pixel_weight)
    return float((float(pixel_real) * w) + (float(freq_real) * (1.0 - w)))


def _build_source_preview(source_url: str, media_type: str) -> dict:
    kind = "video" if str(media_type).lower() == "video" else "image"
    return {
        "kind": kind,
        "url": source_url,
        "page_url": source_url,
        "pageUrl": source_url,
    }


def _to_data_url_jpeg(image_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("미리보기 JPEG 인코딩 실패")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _resize_keep_ratio(image_bgr: np.ndarray, max_side: int = 720) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_bgr
    scale = float(max_side) / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _image_preview_from_bytes(content: bytes) -> str | None:
    try:
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = _resize_keep_ratio(img, max_side=720)
        return _to_data_url_jpeg(img, quality=88)
    except Exception:
        return None


def _video_thumbnail_from_bytes(content: bytes, filename: str) -> str | None:
    suffix = os.path.splitext(filename or "")[1] or ".mp4"
    tmp_path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None

        frame = _resize_keep_ratio(frame, max_side=720)
        return _to_data_url_jpeg(frame, quality=86)
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _build_source_preview_from_downloaded(source_url: str, media_type: str, filename: str, content: bytes, title: str) -> dict:
    preview = _build_source_preview(source_url=source_url, media_type=media_type)
    if title:
        preview["title"] = title

    if str(media_type).lower() == "image":
        data_url = _image_preview_from_bytes(content)
        if data_url:
            preview["data_url"] = data_url
    elif str(media_type).lower() == "video":
        thumb = _video_thumbnail_from_bytes(content=content, filename=filename)
        if thumb:
            preview["thumbnail_data_url"] = thumb

    return preview


def _build_video_face_not_detected_result(
    sampled_frames: int,
    failed_frames: int,
    sampling_meta: dict | None = None,
    frame_failure_reasons: Dict[str, int] | None = None,
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


def _extract_gradcam_overlay_url(rep_payload: Dict[str, Any]) -> str:
    if not isinstance(rep_payload, dict):
        return ""
    assets = rep_payload.get("assets")
    if not isinstance(assets, dict):
        return ""
    url = str(assets.get("gradcam_overlay_url") or "").strip()
    return url


def _cached_has_representative_gradcam(cached: Dict[str, Any]) -> bool:
    if not isinstance(cached, dict):
        return False
    rep = cached.get("representative_analysis")
    if not isinstance(rep, dict):
        return False
    url = _extract_gradcam_overlay_url(rep)
    return bool(url)


def _build_representative_analysis(
    successful_frames: List[np.ndarray],
    successful_frame_indices: List[int],
    scores: List[float],
    target_score: float,
    fusion_w: float,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []

    if not successful_frames or not scores:
        return None, ["대표 프레임 후보가 없습니다."]
    if detector.pixel_model is None or detector.freq_model is None:
        return None, ["대표 프레임 근거 생성을 위한 모델이 준비되지 않았습니다."]

    score_arr = np.asarray(scores, dtype=np.float32)
    if score_arr.size == 0:
        return None, ["대표 프레임 점수 배열이 비어 있습니다."]

    candidate_order = np.argsort(np.abs(score_arr - float(target_score))).tolist()
    max_candidates = min(5, len(candidate_order))

    for pos in candidate_order[:max_candidates]:
        idx = int(pos)
        sample_index = int(successful_frame_indices[idx]) if idx < len(successful_frame_indices) else idx
        frame_bgr = successful_frames[idx]
        frame_score = float(scores[idx])

        try:
            rep_faces = detect_faces_with_aligned_crops(
                image_bgr=frame_bgr,
                margin=0.15,
                target_size=224,
                max_faces=1,
                prioritize_frontal=True,
            )
        except Exception as exc:
            errors.append(f"sample {sample_index}: 얼굴 재검출 실패({type(exc).__name__})")
            continue

        if not rep_faces:
            errors.append(f"sample {sample_index}: 얼굴 미탐지")
            continue

        rep_cam = None
        try:
            rep_cam = GradCAM(detector.pixel_model, get_cam_target_layer(detector.pixel_model))
            rep_out = build_evidence_for_face(
                face_rgb_uint8=rep_faces[0]["crop_rgb"],
                landmarks=rep_faces[0]["landmarks"],
                rgb_model=detector.pixel_model,
                freq_model=detector.freq_model,
                cam=rep_cam,
                fusion_w=float(_normalize_pixel_weight(fusion_w)),
                evidence_level="mvp",
            )
        except Exception as exc:
            errors.append(f"sample {sample_index}: 근거 생성 실패({type(exc).__name__})")
            continue
        finally:
            if rep_cam is not None:
                rep_cam.close()

        gradcam_url = _extract_gradcam_overlay_url({"assets": rep_out.get("assets", {})})
        if not gradcam_url:
            errors.append(f"sample {sample_index}: Grad-CAM 에셋 누락")
            continue

        try:
            explanation = explain_from_evidence(
                evidence=rep_out.get("evidence", {}),
                score=rep_out.get("score", {}),
                media_mode_hint="video",
                use_openai=True,
            )
        except Exception as exc:
            errors.append(f"sample {sample_index}: 설명 생성 실패({type(exc).__name__})")
            explanation = {
                "summary": "대표 프레임 근거는 생성되었지만 설명 생성에 실패했습니다.",
                "summary_source": "fallback:error",
                "spatial_findings": [],
                "frequency_findings": [],
                "interpretation_guide": [],
                "next_steps": [],
                "caveats": [],
            }

        payload = {
            "sample_index": sample_index,
            "frame_score": round(frame_score, 2),
            "target_score": round(float(target_score), 2),
            "abs_diff": round(abs(frame_score - float(target_score)), 2),
            "assets": rep_out.get("assets", {}),
            "evidence": rep_out.get("evidence", {}),
            "explanation": explanation,
        }
        return payload, errors

    return None, errors


# =========================
# Image evidence inference
# =========================

def analyze_evidence_bytes(
    image_bytes: bytes,
    explain: bool = True,
    evidence_level: str = "mvp",
    fusion_w: float = 0.5,
    model_pixel_weight: float = 0.37,
) -> dict:
    request_id = str(uuid.uuid4())
    lv = _validate_evidence_level(evidence_level)
    normalized_fusion_w = _normalize_pixel_weight(fusion_w)
    normalized_model_pixel_weight = _normalize_pixel_weight(model_pixel_weight)

    img_arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    rgb_model = detector.pixel_model
    freq_model = detector.freq_model
    if rgb_model is None or freq_model is None:
        raise HTTPException(
            status_code=500,
            detail="RGB/Frequency 모델 로드 실패: backend/models/*.pth 경로를 확인하세요.",
        )

    try:
        faces = detect_faces_with_aligned_crops(
            image_bgr=bgr,
            margin=0.15,
            target_size=224,
            max_faces=1,
            prioritize_frontal=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"얼굴 분석 실패: {exc}") from exc

    if not faces:
        return {
            "request_id": request_id,
            "status": "ok",
            "score": {"p_rgb": 0.0, "p_freq": 0.0, "p_final": 0.0},
            "faces": [],
            "ai_comment": "얼굴이 선명하게 보이지 않아 판독을 진행하지 못했습니다. 얼굴이 크게 보이는 이미지로 다시 시도해 주세요.",
            "ai_comment_source": "fallback:no_face",
        }

    try:
        cam_target_layer = get_cam_target_layer(rgb_model)
        cam = GradCAM(rgb_model, cam_target_layer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM 초기화 실패: {exc}") from exc

    faces_out = []
    p_rgb_list, p_freq_list, p_final_list = [], [], []

    try:
        for i, face in enumerate(faces):
            try:
                out = build_evidence_for_face(
                    face_rgb_uint8=face["crop_rgb"],
                    landmarks=face["landmarks"],
                    rgb_model=rgb_model,
                    freq_model=freq_model,
                    cam=cam,
                    fusion_w=float(normalized_fusion_w),
                    evidence_level=lv,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"근거 생성 실패(face_id={i}): {exc}") from exc

            score = out["score"]
            evidence = out["evidence"]
            assets = out["assets"]

            p_rgb_list.append(float(score["p_rgb"]))
            p_freq_list.append(float(score["p_freq"]))
            p_final_list.append(float(score["p_final"]))

            item = {
                "face_id": i,
                "assets": assets,
                "evidence": evidence,
            }
            if explain:
                item["explanation"] = explain_from_evidence(
                    evidence=evidence,
                    score=score,
                    media_mode_hint="image",
                    use_openai=(i == 0),
                )

            faces_out.append(item)
    finally:
        cam.close()

    ai_comment = ""
    ai_comment_source = "rule_based"
    if explain and faces_out:
        first_explanation = faces_out[0].get("explanation", {})
        if isinstance(first_explanation, dict):
            ai_comment = str(first_explanation.get("summary", "")).strip()
            ai_comment_source = str(first_explanation.get("summary_source", "rule_based")).strip() or "rule_based"
    ai_comment = sanitize_ai_comment(ai_comment)

    result = {
        "request_id": request_id,
        "status": "ok",
        "score": {
            "p_rgb": _safe_score_agg(p_rgb_list),
            "p_freq": _safe_score_agg(p_freq_list),
            "p_final": _safe_score_agg(p_final_list),
        },
        "faces": faces_out,
        "ai_comment": ai_comment,
        "ai_comment_source": ai_comment_source,
    }

    # 프론트 종합점수는 model.py 점수 체계(real-confidence)와 동일 값으로 전달한다.
    pixel_real = _fake_prob_to_real_percent(result["score"]["p_rgb"])
    freq_real = _fake_prob_to_real_percent(result["score"]["p_freq"])
    final_real = _model_weighted_confidence(
        pixel_real,
        freq_real,
        pixel_weight=normalized_model_pixel_weight,
    )
    result["confidence"] = round(float(final_real), 2)
    result["pixel_score"] = round(float(pixel_real), 2)
    result["freq_score"] = round(float(freq_real), 2)
    result["is_fake"] = bool(result["confidence"] < 50.0)

    return result


# =========================
# Video inference
# =========================

def analyze_video_bytes(content: bytes, filename: str, model_pixel_weight: float = 0.37) -> dict:
    normalized_model_pixel_weight = _normalize_pixel_weight(model_pixel_weight)
    video_hash = sha256_bytes(content)
    video_cache_key = f"cache:video:{video_hash}"

    cached = redis_get_json(redis_db, video_cache_key)
    if cached is not None and has_frame_series(cached, "video_frame_pixel_scores") and has_frame_series(cached, "video_frame_freq_scores"):
        # 과거 캐시 중 대표 프레임 근거(Grad-CAM)가 누락된 케이스는 재생성한다.
        if bool(cached.get("inference_failed")) or _cached_has_representative_gradcam(cached):
            cached_response = dict(cached)
            cached_response["input_media_type"] = "video"
            return store_result_and_make_response(cached_response)

    suffix = os.path.splitext(filename or "")[1] or ".mp4"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        frames, meta = video_to_uniform_sampled_frames(
            tmp_path,
            max_side=VIDEO_MAX_SIDE,
            min_frames=VIDEO_MIN_FRAMES,
            max_frames=VIDEO_MAX_FRAMES_CAP,
            frames_per_minute=VIDEO_FRAMES_PER_MINUTE,
        )
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="비디오에서 프레임을 추출하지 못했습니다.")

        scores, pixel_scores, freq_scores = [], [], []
        successful_frames = []
        successful_frame_indices = []
        failed = 0
        frame_failure_reasons: Dict[str, int] = {}

        for frame_idx, fr in enumerate(frames):
            try:
                score, p_score, f_score, _ = detector.predict_from_bgr(
                    fr,
                    include_preprocess=False,
                    pixel_weight=normalized_model_pixel_weight,
                )
                scores.append(score)
                pixel_scores.append(p_score)
                freq_scores.append(f_score)
                successful_frames.append(fr)
                successful_frame_indices.append(frame_idx)
            except Exception as exc:
                failed += 1
                err_type = type(exc).__name__
                frame_failure_reasons[err_type] = int(frame_failure_reasons.get(err_type, 0)) + 1
                continue

        if len(scores) == 0:
            analysis_result = _build_video_face_not_detected_result(
                sampled_frames=len(frames),
                failed_frames=failed,
                sampling_meta=meta,
                frame_failure_reasons=frame_failure_reasons,
            )
            cache_payload = dict(analysis_result)
            redis_set_json(redis_db, video_cache_key, cache_payload, ex=CACHE_TTL_SEC)
            return store_result_and_make_response(analysis_result)

        video_score, trimmed_meta = trimmed_mean_confidence(
            scores,
            low_trim_ratio=VIDEO_TRIM_LOW_RATIO,
            high_trim_ratio=VIDEO_TRIM_HIGH_RATIO,
        )
        video_pixel = aggregate_scores(pixel_scores, mode=AGG_MODE_VIDEO, topk=TOPK)
        video_freq = aggregate_scores(freq_scores, mode=AGG_MODE_VIDEO, topk=TOPK)

        if video_score is None:
            raise HTTPException(status_code=500, detail="영상 점수 집계 실패")

        analysis_result = build_analysis_result(
            video_score,
            video_pixel,
            video_freq,
            real_mean=REAL_MEAN,
            real_std=REAL_STD,
        )
        analysis_result["video_representative_confidence"] = round(float(video_score), 2)
        analysis_result["video_frame_confidences"] = [round(float(s), 2) for s in scores]
        analysis_result["video_frame_pixel_scores"] = [round(float(s), 2) for s in pixel_scores]
        analysis_result["video_frame_freq_scores"] = [round(float(s), 2) for s in freq_scores]

        analysis_result["video_meta"] = {
            "used_frames": len(scores),
            "failed_frames": failed,
            "agg_mode": VIDEO_AGG_MODE_LABEL,
            "pixel_freq_agg_mode": AGG_MODE_VIDEO,
            "topk": TOPK,
        }
        if frame_failure_reasons:
            analysis_result["video_meta"]["frame_failure_reasons"] = frame_failure_reasons
        analysis_result["video_meta"].update(trimmed_meta)
        analysis_result["video_meta"].update(meta)

        representative_payload, representative_errors = _build_representative_analysis(
            successful_frames=successful_frames,
            successful_frame_indices=successful_frame_indices,
            scores=scores,
            target_score=float(video_score),
            fusion_w=normalized_model_pixel_weight,
        )

        if representative_payload is not None:
            analysis_result["representative_analysis"] = representative_payload
        elif representative_errors:
            analysis_result["representative_analysis_error"] = representative_errors[:5]

        ai_comment = generate_video_ai_comment(
            final_scores=[float(s) for s in scores],
            pixel_scores=[float(s) for s in pixel_scores],
            freq_scores=[float(s) for s in freq_scores],
            is_fake=bool(analysis_result.get("is_fake")) if isinstance(analysis_result.get("is_fake"), bool) else None,
        )
        ai_comment_source = "openai" if ai_comment else "rule_based"
        if not ai_comment:
            if bool(analysis_result.get("is_fake")):
                ai_comment = "영상 전체 흐름을 보면 조작 가능성이 조금 더 높게 보입니다. 아래 근거를 함께 확인해 주세요."
            else:
                ai_comment = "영상 전체 흐름을 보면 원본일 가능성이 조금 더 높게 보입니다. 아래 근거를 함께 확인해 주세요."
        analysis_result["ai_comment"] = sanitize_ai_comment(ai_comment)
        analysis_result["ai_comment_source"] = ai_comment_source
        analysis_result["input_media_type"] = "video"

        cache_payload = dict(analysis_result)
        redis_set_json(redis_db, video_cache_key, cache_payload, ex=CACHE_TTL_SEC)

        return store_result_and_make_response(analysis_result)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =========================
# URL inference
# =========================

def analyze_url_source(
    source_url: str,
    explain: bool = True,
    evidence_level: str = "mvp",
    fusion_w: float = 0.5,
    model_pixel_weight: float = 0.37,
) -> dict:
    try:
        downloaded = download_media_from_url(source_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"URL 미디어 다운로드 실패: {exc}") from exc

    source_preview = _build_source_preview_from_downloaded(
        source_url=downloaded.source_url,
        media_type=downloaded.media_type,
        filename=downloaded.filename,
        content=downloaded.content,
        title=downloaded.title,
    )
    source_meta = {
        "source_url": downloaded.source_url,
        "source_title": downloaded.title,
        "source_extractor": downloaded.extractor,
        "source_filename": downloaded.filename,
    }

    if downloaded.media_type == "video":
        result = analyze_video_bytes(
            content=downloaded.content,
            filename=downloaded.filename or "url_video.mp4",
            model_pixel_weight=model_pixel_weight,
        )
        if isinstance(result, dict):
            result["input_media_type"] = "video"
            result["source_preview"] = source_preview
            result["source_meta"] = source_meta

            data = result.get("data")
            if isinstance(data, dict):
                data["input_media_type"] = "video"
                data["source_preview"] = source_preview
                data["source_meta"] = source_meta
        return result

    if downloaded.media_type == "image":
        result = analyze_evidence_bytes(
            image_bytes=downloaded.content,
            explain=explain,
            evidence_level=evidence_level,
            fusion_w=fusion_w,
            model_pixel_weight=model_pixel_weight,
        )
        if isinstance(result, dict):
            result["input_media_type"] = "image"
            result["source_preview"] = source_preview
            result["source_meta"] = source_meta
        return result

    raise HTTPException(status_code=400, detail=f"지원하지 않는 미디어 타입입니다: {downloaded.media_type}")
