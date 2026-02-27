# syntax=docker/dockerfile:1.7

# -------- 1) build stage --------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive \
    VENV_PATH=/opt/venv \
    UV_LINK_MODE=copy
WORKDIR /w

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    build-essential g++ cmake \
    ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

COPY requirements.txt /w/requirements.txt

RUN python3 -m venv "${VENV_PATH}"
ARG SOCIAL_DEPS_REFRESH=0

# torch/torchvision (cu118) + requirements 설치를 uv로 일원화
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" \
      torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
 && uv pip install --python "${VENV_PATH}/bin/python" -r /w/requirements.txt \
 && echo "social_deps_refresh=${SOCIAL_DEPS_REFRESH}" \
 && uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
      "yt-dlp[default]" instaloader pytubefix


# -------- 2) runtime stage --------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    VENV_PATH=/opt/venv \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"
ARG NODE_VERSION=20.19.0
WORKDIR /app

# 런타임 최소 의존성(opencv 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgl1 libglib2.0-0 \
    ca-certificates ffmpeg curl xz-utils \
 && update-ca-certificates \
 && ARCH="$(dpkg --print-architecture)" \
 && case "$ARCH" in \
      amd64) NODE_ARCH="x64" ;; \
      arm64) NODE_ARCH="arm64" ;; \
      *) echo "Unsupported architecture: $ARCH" && exit 1 ;; \
    esac \
 && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" -o /tmp/node.tar.xz \
 && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
 && rm -f /tmp/node.tar.xz \
 && node --version \
 && rm -rf /var/lib/apt/lists/*

# 빌드 stage에서 준비한 Python 환경 복사
COPY --from=build "${VENV_PATH}" "${VENV_PATH}"

# 앱 코드 복사
COPY . /app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
