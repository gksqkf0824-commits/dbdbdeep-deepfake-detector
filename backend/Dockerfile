# -------- 1) build stage --------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /w

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential g++ cmake \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /w/requirements.txt

RUN python3 -m pip install --upgrade pip wheel

# requirements를 wheel로 미리 빌드(insightface 포함)
RUN pip wheel --no-cache-dir --wheel-dir /w/wheels -r /w/requirements.txt


# -------- 2) runtime stage --------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 런타임 최소 의존성(opencv 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# torch/torchvision (cu118)
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 빌드 stage에서 만든 wheels 설치
COPY --from=build /w/wheels /w/wheels
RUN pip install --no-cache-dir /w/wheels/*

# 앱 코드 복사
COPY . /app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]