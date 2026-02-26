# DBDBDEEP — Backend

FastAPI 기반 딥페이크 탐지 백엔드 서버입니다.  
이미지 · 영상 · SNS URL 입력을 모두 지원하며, Grad-CAM 시각화와 GPT 기반 AI 리포트를 포함한 완전한 분석 파이프라인을 제공합니다.

---

## 📁 디렉토리 구조

```
backend/
├── main.py                  # FastAPI 앱 진입점 · 라우터 정의
├── model.py                 # 모델 아키텍처 · DeepfakeDetectorEnsemble
├── requirements.txt         # Python 의존성
├── Dockerfile               # CUDA 11.8 기반 멀티스테이지 빌드
├── models/                  # 모델 가중치 (직접 배치 또는 env var로 경로 지정)
│   ├── image.pth            # Image Model (EfficientNet-V2-S, RGB 3ch)
│   └── freq.pt              # Frequency Model (EfficientNet-V2-S, SRM+Y 4ch)
└── services/
    ├── analysis_service.py  # 이미지 · 영상 · URL 분석 오케스트레이션
    ├── inference.py         # 얼굴 검출 · Grad-CAM · 전처리 유틸리티
    ├── evidence.py          # 공간 · 주파수 근거(Evidence) 생성
    ├── explain.py           # GPT API 기반 자연어 설명 생성
    ├── stats.py             # 점수 집계 · p-value · 신뢰도 레이블
    ├── storage.py           # Redis JSON 캐시 헬퍼
    ├── redis_client.py      # Redis 연결 싱글톤
    ├── url_media_utils.py   # YouTube · SNS URL 미디어 다운로드 (yt-dlp)
    └── video_utils.py       # 영상 프레임 균등 샘플링
```

---

## ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| **이미지 분석** | RGB Pixel + SRM+Y Frequency 듀얼 앙상블 (W=0.37/0.63) |
| **영상 분석** | 균등 프레임 샘플링 → 프레임별 추론 → Trimmed Mean 집계 |
| **URL 분석** | YouTube Shorts · 일반 URL → yt-dlp 자동 다운로드 후 분석 |
| **얼굴 검출** | InsightFace (RetinaFace) — 정면성 기반 우선순위 선택 |
| **Grad-CAM** | 픽셀 모델의 의심 영역 히트맵 시각화 |
| **AI 리포트** | Grad-CAM 결과 기반 GPT 자연어 설명 생성 |
| **Redis 캐싱** | 영상 결과 24h · 분석 토큰 1h 캐싱 |
| **위험 등급** | p_real 기반 3단계: REAL / WARNING / FAKE |

---

## 🚀 실행 방법

### Docker (권장)

```bash
# 1. 루트에서 모델 가중치를 models/ 폴더에 복사
mkdir -p models
cp ../image.pth models/image.pth
cp ../freq.pt   models/freq.pt

# 2. 빌드 및 실행 (Redis 컨테이너와 함께)
docker build -t dbdbdeep-backend .
docker run -p 8000:8000 \
  -e REDIS_HOST=host.docker.internal \
  -e OPENAI_API_KEY=sk-... \
  dbdbdeep-backend
```

> Docker Compose를 사용하면 Redis와 함께 자동으로 연결됩니다.

### 로컬 개발

```bash
pip install -r requirements.txt

# Redis 실행
docker run -p 6379:6379 -d redis

# 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/api/test` | 서버 상태 확인 |
| `POST` | `/api/analyze` | 이미지 업로드 분석 |
| `POST` | `/api/analyze-video` | 영상 업로드 분석 |
| `POST` | `/api/analyze-url` | SNS/YouTube URL 분석 |
| `GET` | `/api/get-result/{token}` | 캐시 결과 조회 (1h 유효) |
| `POST` | `/api/clear-cache` | Redis 캐시 전체 삭제 |

### `/api/analyze` 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `file` | File | 필수 | 이미지 파일 |
| `explain` | bool | `true` | GPT AI 코멘트 생성 여부 |
| `evidence_level` | string | `mvp` | `off` / `mvp` / `full` |
| `fusion_w` | float | `0.5` | 픽셀 모델 가중치 (0~1) |

### 응답 예시

```json
{
  "result_url": "http://localhost:8000/get-result/{token}",
  "data": {
    "confidence": 18.5,
    "pixel_score": 22.1,
    "freq_score": 16.3,
    "is_fake": true,
    "ai_comment": "주파수 도메인에서 GAN 특유의 격자 아티팩트가 감지되었습니다.",
    "faces": [
      {
        "face_id": 0,
        "assets": { "gradcam_overlay_url": "data:image/jpeg;base64,..." },
        "evidence": { "spatial": {...}, "frequency": {...} },
        "explanation": { "summary": "...", "spatial_findings": [...] }
      }
    ]
  }
}
```

---

## ⚙️ 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `IMG_MODEL_PATH` | `models/image.pth` | 이미지 모델 가중치 경로 |
| `FREQUENCY_MODEL_PATH` | `models/freq.pt` | 주파수 모델 가중치 경로 |
| `REDIS_HOST` | `redis` | Redis 호스트 (Docker: 서비스명) |
| `REDIS_PORT` | `6379` | Redis 포트 |
| `OPENAI_API_KEY` | — | GPT AI 리포트 사용 시 필요 |

---

## 🧠 모델 구조

```
Input Image
    │
    ├─ Face Detection (InsightFace RetinaFace)
    │       └─ Square crop + margin 0.15 → resize 224×224
    │
    ├─ [Image Model]  RGB 3ch → EfficientNet-V2-S → P(fake)_pixel
    │
    └─ [Freq Model]   SRM×3 + Y channel → 4ch → EfficientNet-V2-S → P(fake)_freq
                              │
                    Weighted Soft Voting
                  0.37 × s_pixel + 0.63 × s_freq
                              │
                    Real-Confidence Score (0~100)
                              │
              ┌───────────────┼───────────────┐
           REAL            WARNING           FAKE
        p_real > 52      33.5~52          p_real < 33.5
```

---

## 📦 주요 의존성

| 패키지 | 용도 |
|---|---|
| `fastapi` · `uvicorn` | API 서버 |
| `torch` · `torchvision` | 딥러닝 모델 추론 |
| `insightface` | RetinaFace 얼굴 검출 |
| `grad-cam` | Grad-CAM 시각화 |
| `opencv-python-headless` | 이미지/영상 처리 |
| `redis` | 결과 캐싱 |
| `yt-dlp` · `pytubefix` | YouTube/SNS URL 미디어 다운로드 |
