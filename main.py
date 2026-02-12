from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import secrets
import json
import scipy.stats as stats
import os

from .model import detector  # 모델 로더
from .redis_client import redis_db  # redis_db 임포트 확인

app = FastAPI()

# --- [여기부터 복사] ---
@app.on_event("startup")
def check_redis_connection():
    try:
        redis_db.ping()
        print("✅ Redis 연결 성공! (준비 완료)")
    except Exception as e:
        print(f"❌ Redis 연결 실패: {e}")
        print("   👉 Docker가 켜져 있는지, 'docker run -p 6379:6379 -d redis'를 했는지 확인하세요!")
# --- [여기까지] ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 결과 이미지 저장 경로
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

REAL_MEAN = 15.0 
REAL_STD = 8.0

def calculate_p_value(score):
    z_score = (score - REAL_MEAN) / REAL_STD
    p_value = 1 - stats.norm.cdf(z_score)
    return round(max(p_value, 0.0001), 4)

@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # model.py의 detector 호출 (이미지 저장 로직 포함)
    try:
        score, pixel_score, freq_score, pixel_path, freq_path = detector.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Inference Error: {e}")

    p_val = calculate_p_value(score)
    
    analysis_result = {
        "confidence": score,       # 통합 점수
        "pixel_score": pixel_score, # ★ 픽셀 모델 점수 추가
        "freq_score": freq_score,   # ★ 주파수 모델 점수 추가
        "is_fake": score < 50,
        "p_value": p_val,
        "reliability": "매우 높음" if p_val < 0.01 else ("높음" if p_val < 0.05 else "보통"),
        "pixel_img_path": f"outputs/{pixel_path}", # 경로가 맞는지 확인 (static mount 경로)
        "freq_img_path": f"outputs/{freq_path}"
    }
    
    # Redis에 결과 저장 (1시간 후 만료)
    result_token = secrets.token_urlsafe(16)
    redis_db.set(f"res:{result_token}", json.dumps(analysis_result), ex=3600)
    
    return {
        "result_url": f"http://127.0.0.1:8000/get-result/{result_token}",
        "data": analysis_result
    }

@app.get("/get-result/{token}")
async def get_analysis_result(token: str):
    data = redis_db.get(f"res:{token}") # temp_db 대신 redis_db 사용
    if data is None:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    return json.loads(data)