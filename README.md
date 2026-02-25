# 🕵️‍♂️ DBDBDEEP – Multimodal Deepfake Detector

사진 · 영상 · URL 기반 딥페이크 검증 AI 플랫폼  
**Upload once, verify instantly.**  
*"Can you trust what you see?"*

멋쟁이사자처럼 AI CV 단기심화 부트캠프 3기  
**디비디비딥(DBDBDEEP)** 팀의 최종 프로젝트입니다.

고도화된 생성형 AI 콘텐츠로 인한 딥페이크 범죄를 예방하기 위해,  
누구나 쉽게 사용할 수 있는 **실시간 딥페이크 검증 서비스**를 개발했습니다.

---

## 💡 Key Features

- 멀티 입력 지원 (Image / Video / URL)
- Pixel + Frequency 기반 이중 탐지 모델
- EfficientNet 기반 앙상블 구조
- 최신 Diffusion 생성 이미지 대응
- 딥페이크 확률 + Trust Score 제공
- Grad-CAM 기반 Explainable AI 시각화
- 웹 UI 실시간 분석

---

## 🧠 Detection Architecture
Input (Image / Video / URL)
↓
Pixel Model (EfficientNet-V2-S)
Frequency Model (SRM + Y Channel)
↓
Weighted Soft Voting Ensemble
↓
Fake Probability + Trust Score
↓
Grad-CAM Visualization

---

## 🛠 Tech Stack

### AI / ML
- PyTorch
- OpenCV
- InsightFace (RetinaFace)
- timm
- NumPy / Pandas
- scikit-learn

### Backend / Frontend
- FastAPI
- React

### Infra
- AWS EC2
- Docker
- NGINX
- GitHub Actions
- CUDA (A100 GPU)

---

## 📊 Dataset & Preprocessing

### Dataset

- FaceForensics++
- FFHQ
- Celeb-DF
- FaceSwapGAN
- Custom generated images (FLUX, Qwen, Kolors)

최신 생성 모델 데이터까지 직접 구축하여 일반화 성능을 강화했습니다.

### Preprocessing

- RetinaFace 기반 얼굴 검출
- Bounding box 확장
- 224×224 Crop & Resize
- 미세 위조 패턴 보존 중심 전처리

---

## 🧪 Model Design

### Image Model
- EfficientNet-V2-S
- 국소 텍스처 아티팩트 학습

### Frequency Model
- SRM 기반 고주파 특징 + Y Channel
- Custom EfficientNet-V2-S (4-channel input)

### Ensemble

Weighted Soft Voting:
Final = 0.37 * Image + 0.63 * Frequency

---

## 🏆 Performance

| Model | F1 (Macro) | AUC |
|------|-----------|-----|
| Image Model | 0.8013 | 0.8903 |
| Frequency Model | 0.9337 | 0.9840 |
| Ensemble | **0.9410** | **0.9789** |

앙상블 적용 시 단일 모델 대비 성능이 크게 향상되었습니다.

---
👥 Team DBDBDEEP

조영준 (팀장): 이미지 모델, 데이터 구축

권소윤: 이미지 모델, Frontend / Backend

주요셉: 이미지 모델, Frontend

신동혁: 주파수 모델, Backend

장은태: 주파수 모델, Frontend, 영상 제작
