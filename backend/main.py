from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import secrets
import os
import tempfile

from model import detector
from redis_client import redis_db

from utils import (
    sha256_bytes,
    redis_get_json,
    redis_set_json,
    video_to_sampled_frames_per_second,
    frame_to_jpeg_bytes,
    aggregate_scores,
    build_analysis_result,
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
VIDEO_SECONDS_STEP = 1.0    # 1초마다 1프레임
VIDEO_MAX_SIDE = 640        # 긴 변 기준 다운스케일
VIDEO_MAX_FRAMES_CAP = 60   # 너무 긴 영상 폭주 방지

# Aggregation
AGG_MODE_VIDEO = "mean"
TOPK = 5

# Redis TTL
RESULT_TTL_SEC = 3600
CACHE_TTL_SEC = 24 * 3600


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


@app.get("/test")
async def test():
    return {"message": "서버가 정상적으로 작동 중입니다."}


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
@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    # 0) bytes read + hash
    content = await file.read()
    video_hash = sha256_bytes(content)

    # 1) cache hit
    cached = redis_get_json(redis_db, f"cache:video:{video_hash}")
    if cached is not None:
        return store_result_and_make_response(cached)

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp_path = None

    try:
        # 2) temp save
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        # 3) sequential sampling per second
        frames, meta = video_to_sampled_frames_per_second(
            tmp_path,
            seconds_step=VIDEO_SECONDS_STEP,
            max_side=VIDEO_MAX_SIDE,
            max_frames=VIDEO_MAX_FRAMES_CAP,
        )
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="비디오에서 프레임을 추출하지 못했습니다.")

        # 4) per-frame inference
        scores, pixel_scores, freq_scores = [], [], []
        failed = 0

        for fr in frames:
            try:
                jpg_bytes = frame_to_jpeg_bytes(fr, quality=90)
                score, p_score, f_score, _ = detector.predict(
                    jpg_bytes,
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

        # 5) aggregate (topk_mean)
        video_score = aggregate_scores(scores, mode=AGG_MODE_VIDEO, topk=TOPK)
        video_pixel = aggregate_scores(pixel_scores, mode=AGG_MODE_VIDEO, topk=TOPK)
        video_freq  = aggregate_scores(freq_scores, mode=AGG_MODE_VIDEO, topk=TOPK)

        if video_score is None:
            raise HTTPException(status_code=500, detail="영상 점수 집계 실패")

        analysis_result = build_analysis_result(
            video_score, video_pixel, video_freq,
            real_mean=REAL_MEAN, real_std=REAL_STD
        )

        # 6) video meta + ✅ meta merge (여기가 update(meta) 위치)
        analysis_result["video_meta"] = {
            "used_frames": len(scores),
            "failed_frames": failed,
            "agg_mode": AGG_MODE_VIDEO,
            "topk": TOPK,
        }
        analysis_result["video_meta"].update(meta)

        # 7) cache store
        redis_set_json(redis_db, f"cache:video:{video_hash}", analysis_result, ex=CACHE_TTL_SEC)

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
