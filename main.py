"""
main.py — DBDBDEEP FastAPI Backend

Endpoints
---------
POST /analyze          Upload an image → run ensemble inference → return result + token
GET  /get-result/{tok} Retrieve a cached result by token (valid for 1 hour)

Redis is used to cache results so the frontend can poll /get-result
without re-running the model.
"""

import json
import secrets

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model import detector          # DeepfakeDetectorEnsemble instance
from redis_client import redis_db   # Redis connection

app = FastAPI(title="DBDBDEEP API", version="1.0.0")

# ── CORS (allow all origins for dev; tighten in production) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files for output images (Grad-CAM, charts, etc.) ──────────────────
import os
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ── Redis connection check on startup ────────────────────────────────────────
@app.on_event("startup")
def check_redis():
    try:
        redis_db.ping()
        print("✅ Redis connected.")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Make sure Redis is running: docker run -p 6379:6379 -d redis")


# ── Analyze endpoint ──────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Upload an image file and receive a deepfake analysis result.

    Returns
    -------
    JSON with:
      result_url   : URL to retrieve the cached result later
      data         : full analysis result (scores, risk level, etc.)
    """
    image_bytes = await file.read()

    try:
        result = detector.predict(image_bytes)
    except ValueError as e:
        # e.g. "No face detected"
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # result keys: fake_score, real_score, p_image, p_freq, is_fake, risk_level
    analysis = {
        "fake_score":  result["fake_score"],   # 0–100 (higher = more fake)
        "real_score":  result["real_score"],   # 0–100 (higher = more real)
        "p_image":     result["p_image"],      # raw P(fake) from image model
        "p_freq":      result["p_freq"],       # raw P(fake) from freq model
        "is_fake":     result["is_fake"],      # bool
        "risk_level":  result["risk_level"],   # "Safe" | "Caution" | "Danger"
    }

    # Cache result in Redis for 1 hour
    token = secrets.token_urlsafe(16)
    redis_db.set(f"res:{token}", json.dumps(analysis), ex=3600)

    return {
        "result_url": f"/get-result/{token}",
        "data": analysis,
    }


# ── Result retrieval endpoint ─────────────────────────────────────────────────
@app.get("/get-result/{token}")
async def get_result(token: str):
    """Retrieve a previously cached analysis result by token."""
    data = redis_db.get(f"res:{token}")
    if data is None:
        raise HTTPException(status_code=404, detail="Result not found or expired.")
    return json.loads(data)
