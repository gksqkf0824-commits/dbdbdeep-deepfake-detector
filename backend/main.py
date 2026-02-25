from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from services.analysis_service import (
    analyze_evidence_bytes,
    analyze_url_source,
    analyze_video_bytes,
    clear_cache_entries,
    get_result_by_token,
    ping_redis,
)

app = FastAPI()


@app.on_event("startup")
def check_redis_connection():
    try:
        ping_redis()
        print("✅ Redis 연결 성공! (준비 완료)")
    except Exception as e:
        print(f"❌ Redis 연결 실패: {e}")
        print("   👉 Docker가 켜져 있는지, 'docker run -p 6379:6379 -d redis'를 했는지 확인하세요!")


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
    fusion_w: float = Form(0.5),
):
    data = await file.read()
    return analyze_evidence_bytes(
        image_bytes=data,
        explain=explain,
        evidence_level=evidence_level,
        fusion_w=fusion_w,
    )


@app.post("/analyze-video")
@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    content = await file.read()
    return analyze_video_bytes(content=content, filename=file.filename or "upload.mp4")


@app.post("/analyze-url")
@app.post("/api/analyze-url")
async def analyze_url(
    source_url: str = Form(""),
    image_url: str = Form(""),
    explain: bool = Form(True),
    evidence_level: str = Form("mvp"),
    fusion_w: float = Form(0.5),
):
    target_url = (source_url or image_url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL을 입력해 주세요. (source_url 또는 image_url)")

    return analyze_url_source(
        source_url=target_url,
        explain=explain,
        evidence_level=evidence_level,
        fusion_w=fusion_w,
    )


@app.get("/get-result/{token}")
@app.get("/api/get-result/{token}")
async def get_analysis_result(token: str):
    return get_result_by_token(token)
