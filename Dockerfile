# GPU image (CUDA 12.6 + cuDNN 9). Requires NVIDIA Container Toolkit on the host.
# StyleTTS2 Ukrainian needs a GPU for real-time synthesis; there is no CPU
# variant on purpose (CPU works but is too slow for telephony).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
# CUDA-matched torch wheels. Override per CUDA version if needed, e.g.
#   --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/home/app/.cache/huggingface \
    DATA_DIR=/data \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    build-essential \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /home/app

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY app /home/app/app

RUN mkdir -p /data /home/app/.cache/huggingface
VOLUME ["/home/app/.cache/huggingface", "/data"]

EXPOSE 8000

# start-period covers model load on startup, when heavy CPU work can briefly
# starve the health endpoint and cause false "unhealthy".
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["python", "-m", "app.main"]
