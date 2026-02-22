# -------- 1) build stage --------
FROM python:3.11-slim AS build

WORKDIR /w

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ cmake \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /w/requirements.txt

# wheel 빌드를 위한 도구
RUN pip install --upgrade pip wheel

# requirements를 wheel로 미리 빌드 (insightface 포함)
RUN pip wheel --no-cache-dir --wheel-dir /w/wheels -r /w/requirements.txt


# -------- 2) runtime stage --------
FROM python:3.11-slim

WORKDIR /app

# 런타임에 필요한 최소 라이브러리(opencv/onnx 관련)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# (GPU) torch/torchvision은 런타임에서 설치
RUN pip install --upgrade pip \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 빌드 stage에서 만든 wheels를 복사해서 설치
COPY --from=build /w/wheels /w/wheels
RUN pip install --no-cache-dir /w/wheels/*

# 앱 코드 복사
COPY . /app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]