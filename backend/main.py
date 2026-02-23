from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import secrets
import os
import tempfile
import uuid
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from model import detector
from redis_client import redis_db

from utils import (
    sha256_bytes,
    redis_get_json,
    redis_set_json,
    video_to_uniform_sampled_frames,
    aggregate_scores,
    trimmed_mean_confidence,
    build_analysis_result,
    detect_faces_with_aligned_crops,
    get_cam_target_layer,
    GradCAM,
    build_evidence_for_face,
    explain_from_evidence,
)

app = FastAPI()

# --- Redis 연결 체크 ---
@app.on_event("startup")
def check_redis_connection():
    try:
        redis_db.ping()
        print("✅ Redis 연결 성공! (준비 완료)")
    except Exception as e:
        print(f"❌ Redis 연결 실패: {e}")
        print("   👉 Docker가 켜져 있는지, 'docker run -p 6379:6379 -d redis'를 했는지 확인하세요!")

# --- CORS ---
# NOTE: allow_credentials=True + allow_origins=["*"] 조합은 브라우저/환경에 따라 문제가 될 수 있음.
# (일단 기존 유지. 운영에서는 도메인을 명시하는 게 안전)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
VIDEO_TRIM_RATIO = 0.10

# Redis TTL
RESULT_TTL_SEC = 3600
CACHE_TTL_SEC = 24 * 3600
REMOTE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
REMOTE_IMAGE_TIMEOUT_SEC = 10


def store_result_and_make_response(analysis_result: dict, stored_result: dict = None) -> dict:
    """
    결과를 Redis(res:{token})에 저장하고 프론트가 쓰는 형태로 반환.
    """
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


def _validate_evidence_level(level: str) -> str:
    lv = (level or "mvp").strip().lower()
    if lv not in {"off", "mvp", "full"}:
        raise HTTPException(status_code=400, detail="evidence_level은 off/mvp/full 중 하나여야 합니다.")
    return lv


def _safe_score_agg(values):
    return float(max(values)) if values else 0.0


def _analyze_evidence_bytes(
    image_bytes: bytes,
    explain: bool = True,
    evidence_level: str = "mvp",
    fusion_w: float = 0.5,
) -> dict:
    request_id = str(uuid.uuid4())
    lv = _validate_evidence_level(evidence_level)

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
            max_faces=8,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"얼굴 분석 실패: {exc}") from exc

    if not faces:
        return {
            "request_id": request_id,
            "status": "ok",
            "score": {"p_rgb": 0.0, "p_freq": 0.0, "p_final": 0.0},
            "faces": [],
        }

    cam_target_layer = get_cam_target_layer(rgb_model)
    cam = GradCAM(rgb_model, cam_target_layer)

    faces_out = []
    p_rgb_list, p_freq_list, p_final_list = [], [], []

    try:
        for i, face in enumerate(faces):
            out = build_evidence_for_face(
                face_rgb_uint8=face["crop_rgb"],
                landmarks=face["landmarks"],
                rgb_model=rgb_model,
                freq_model=freq_model,
                cam=cam,
                fusion_w=float(fusion_w),
                evidence_level=lv,
            )

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
                item["explanation"] = explain_from_evidence(evidence=evidence, score=score)

            faces_out.append(item)
    finally:
        cam.close()

    return {
        "request_id": request_id,
        "status": "ok",
        "score": {
            "p_rgb": _safe_score_agg(p_rgb_list),
            "p_freq": _safe_score_agg(p_freq_list),
            "p_final": _safe_score_agg(p_final_list),
        },
        "faces": faces_out,
    }


@app.get("/test")
async def test():
    return {"message": "서버가 정상적으로 작동 중입니다."}


@app.post("/clear-cache")
async def clear_cache():
    """
    Redis 캐시 키 삭제.
    현재는 결과/비디오 캐시를 모두 정리한다.
    """
    try:
        patterns = ["cache:*", "res:*"]
        deleted_count = delete_keys_by_patterns(patterns)
        return {
            "message": "Redis cache cleared",
            "deleted_keys": deleted_count,
            "patterns": patterns,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {e}")


@app.post("/api/analyze")
@app.post("/api/analyze-evidence")
@app.post("/analyze-evidence")
async def analyze_with_evidence(
    file: UploadFile = File(...),
    explain: bool = Form(True),
    evidence_level: str = Form("mvp"),
    fusion_w: float = Form(0.5),
):
    data = await file.read()
    return _analyze_evidence_bytes(
        image_bytes=data,
        explain=explain,
        evidence_level=evidence_level,
        fusion_w=fusion_w,
    )


@app.post("/api/analyze-url")
@app.post("/analyze-url")
async def analyze_url_with_evidence(
    image_url: str = Form(...),
    explain: bool = Form(True),
    evidence_level: str = Form("mvp"),
    fusion_w: float = Form(0.5),
):
    parsed = urlparse((image_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="유효한 http/https 이미지 URL을 입력하세요.")

    try:
        resp = requests.get(parsed.geturl(), timeout=REMOTE_IMAGE_TIMEOUT_SEC)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"이미지 URL 다운로드 실패: {exc}") from exc

    data = resp.content or b""
    if not data:
        raise HTTPException(status_code=400, detail="다운로드한 이미지 데이터가 비어 있습니다.")
    if len(data) > REMOTE_IMAGE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"이미지 크기가 너무 큽니다. ({REMOTE_IMAGE_MAX_BYTES // (1024 * 1024)}MB 이하)",
        )

    content_type = (resp.headers.get("content-type") or "").lower()
    if content_type and "image" not in content_type:
        raise HTTPException(status_code=400, detail="이미지 URL만 지원합니다.")

    return _analyze_evidence_bytes(
        image_bytes=data,
        explain=explain,
        evidence_level=evidence_level,
        fusion_w=fusion_w,
    )


# =========================
# Image inference
# =========================
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        score, pixel_score, freq_score, preprocessed = detector.predict(
            image_bytes,
            include_preprocess=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Inference Error: {e}")

    analysis_result = build_analysis_result(
        score, pixel_score, freq_score,
        real_mean=REAL_MEAN, real_std=REAL_STD
    )
    if preprocessed is not None:
        analysis_result["preprocessed"] = preprocessed

    stored_result = dict(analysis_result)
    stored_result.pop("preprocessed", None)
    return store_result_and_make_response(analysis_result, stored_result=stored_result)


# =========================
# Video inference
# =========================
@app.post("/api/analyze-video")
@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    # 0) bytes read + hash
    content = await file.read()
    video_hash = sha256_bytes(content)
    video_cache_key = f"cache:video:{video_hash}"

    # 1) cache hit
    cached = redis_get_json(redis_db, video_cache_key)
    if cached is not None and has_frame_series(cached, "video_frame_pixel_scores") and has_frame_series(cached, "video_frame_freq_scores"):
        return store_result_and_make_response(cached)

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp_path = None

    try:
        # 2) temp save
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        # 3) full-span uniform sampling (전체 구간 대표 프레임)
        frames, meta = video_to_uniform_sampled_frames(
            tmp_path,
            max_side=VIDEO_MAX_SIDE,
            min_frames=VIDEO_MIN_FRAMES,
            max_frames=VIDEO_MAX_FRAMES_CAP,
            frames_per_minute=VIDEO_FRAMES_PER_MINUTE,
        )
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="비디오에서 프레임을 추출하지 못했습니다.")

        # 4) per-frame inference
        scores, pixel_scores, freq_scores = [], [], []
        failed = 0

        for fr in frames:
            try:
                score, p_score, f_score, _ = detector.predict_from_bgr(
                    fr,
                    include_preprocess=False,
                )
                scores.append(score)
                pixel_scores.append(p_score)
                freq_scores.append(f_score)
            except Exception:
                failed += 1
                continue

        if len(scores) == 0:
            raise HTTPException(
                status_code=500,
                detail=f"모든 프레임 추론 실패 (sampled={len(frames)}, failed={failed})."
            )

        # 5) aggregate
        video_score, trimmed_meta = trimmed_mean_confidence(
            scores,
            trim_ratio=VIDEO_TRIM_RATIO,
        )
        video_pixel = aggregate_scores(pixel_scores, mode=AGG_MODE_VIDEO, topk=TOPK)
        video_freq  = aggregate_scores(freq_scores, mode=AGG_MODE_VIDEO, topk=TOPK)

        if video_score is None:
            raise HTTPException(status_code=500, detail="영상 점수 집계 실패")

        analysis_result = build_analysis_result(
            video_score, video_pixel, video_freq,
            real_mean=REAL_MEAN, real_std=REAL_STD
        )
        analysis_result["video_representative_confidence"] = round(float(video_score), 2)
        analysis_result["video_frame_confidences"] = [round(float(s), 2) for s in scores]
        analysis_result["video_frame_pixel_scores"] = [round(float(s), 2) for s in pixel_scores]
        analysis_result["video_frame_freq_scores"] = [round(float(s), 2) for s in freq_scores]

        # 6) video meta + ✅ meta merge (여기가 update(meta) 위치)
        analysis_result["video_meta"] = {
            "used_frames": len(scores),
            "failed_frames": failed,
            "agg_mode": "trimmed_mean_10pct",
            "pixel_freq_agg_mode": AGG_MODE_VIDEO,
            "topk": TOPK,
        }
        analysis_result["video_meta"].update(trimmed_meta)
        analysis_result["video_meta"].update(meta)

        # 7) cache store
        redis_set_json(redis_db, video_cache_key, analysis_result, ex=CACHE_TTL_SEC)

        return store_result_and_make_response(analysis_result)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/get-result/{token}")
async def get_analysis_result(token: str):
    data = redis_get_json(redis_db, f"res:{token}")
    if data is None:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    return data
