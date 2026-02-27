from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from log_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

from model import detector
from services.analysis_service import (
    analyze_evidence_bytes,
    analyze_url_source,
    analyze_video_bytes,
    clear_cache_entries,
    get_result_by_token,
    ping_redis,
)

app = FastAPI()

# Single source of truth for model fusion weight.
# - pixel weight: MODEL_PIXEL_WEIGHT
# - frequency weight: 1 - MODEL_PIXEL_WEIGHT
# 직접 수정 포인트: 여기 숫자만 바꾸면 전체 경로에 동일 반영됩니다.
MODEL_PIXEL_WEIGHT = 0.37
MODEL_READY = False
MODEL_STATUS_DETAIL = ""


def _assert_models_loaded() -> None:
    pixel_ready = getattr(detector, "pixel_model", None) is not None
    freq_ready = getattr(detector, "freq_model", None) is not None

    if not pixel_ready or not freq_ready:
        missing = []
        if not pixel_ready:
            missing.append("pixel")
        if not freq_ready:
            missing.append("frequency")
        raise RuntimeError(
            f"모델 로드 실패: {', '.join(missing)} 모델이 비활성화 상태입니다. "
            "IMG_MODEL_PATH/FREQUENCY_MODEL_PATH 및 모델 파일 경로를 확인하세요."
        )

    device = str(getattr(detector, "device", "unknown"))
    logger.info("모델 로드 성공 device=%s pixel=ready frequency=ready", device)


def _require_model_ready() -> None:
    if MODEL_READY:
        return
    detail = MODEL_STATUS_DETAIL or "모델이 아직 준비되지 않았습니다."
    raise HTTPException(status_code=503, detail=detail)


@app.on_event("startup")
def check_redis_connection():
    global MODEL_READY, MODEL_STATUS_DETAIL
    try:
        _assert_models_loaded()
        MODEL_READY = True
        MODEL_STATUS_DETAIL = "ready"
    except Exception as e:
        MODEL_READY = False
        MODEL_STATUS_DETAIL = str(e)
        logger.exception("모델 초기화 실패: %s", e)
        logger.warning("서비스는 기동되지만 분석 API는 503을 반환합니다.")

    try:
        ping_redis()
        logger.info("Redis 연결 성공 (준비 완료)")
        logger.info(
            "MODEL_PIXEL_WEIGHT=%.2f MODEL_FREQ_WEIGHT=%.2f",
            MODEL_PIXEL_WEIGHT,
            (1.0 - MODEL_PIXEL_WEIGHT),
        )
    except Exception as e:
        logger.exception("Redis 연결 실패: %s", e)
        logger.warning(
            "Docker/Redis 상태를 확인하세요. 서비스는 기동 유지됩니다."
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
@app.get("/api/test")
async def test():
    return {"message": "서버가 정상적으로 작동 중입니다."}


@app.post("/clear-cache")
@app.post("/api/clear-cache")
async def clear_cache():
    try:
        return clear_cache_entries()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {e}")


@app.post("/analyze")
@app.post("/api/analyze")
async def analyze_with_evidence(
    file: UploadFile = File(...),
    explain: bool = Form(True),
    evidence_level: str = Form("mvp"),
):
    _require_model_ready()
    data = await file.read()
    return analyze_evidence_bytes(
        image_bytes=data,
        explain=explain,
        evidence_level=evidence_level,
        fusion_w=MODEL_PIXEL_WEIGHT,
        model_pixel_weight=MODEL_PIXEL_WEIGHT,
    )


@app.post("/analyze-video")
@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    _require_model_ready()
    content = await file.read()
    return analyze_video_bytes(
        content=content,
        filename=file.filename or "upload.mp4",
        model_pixel_weight=MODEL_PIXEL_WEIGHT,
    )


@app.post("/analyze-url")
@app.post("/api/analyze-url")
async def analyze_url(
    source_url: str = Form(""),
    image_url: str = Form(""),
    explain: bool = Form(True),
    evidence_level: str = Form("mvp"),
):
    _require_model_ready()
    target_url = (source_url or image_url or "").strip()
    logger.info(
        "api:/analyze-url source_url=%r image_url=%r target_url=%r",
        source_url,
        image_url,
        target_url,
    )
    if not target_url:
        raise HTTPException(status_code=400, detail="URL을 입력해 주세요. (source_url 또는 image_url)")

    return analyze_url_source(
        source_url=target_url,
        explain=explain,
        evidence_level=evidence_level,
        fusion_w=MODEL_PIXEL_WEIGHT,
        model_pixel_weight=MODEL_PIXEL_WEIGHT,
    )


@app.get("/get-result/{token}")
@app.get("/api/get-result/{token}")
async def get_analysis_result(token: str):
    return get_result_by_token(token)
