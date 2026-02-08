# =============================================================================
# Medical AI Drug Interaction Demo - Docker Container
# Optimized for HP ZGX Nano with NVIDIA GB10 Grace Blackwell Superchip
#
# Build & run:
#   ./start.sh --build          # First time / after code changes
#   ./start.sh                  # Subsequent runs
#   docker compose down         # Stop
#
# Prerequisites:
#   - NVIDIA Container Toolkit installed
#   - GGUF model file in ./models/ directory
# =============================================================================

# --- Stage 1: Build llama-cpp-python with CUDA support ---
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# Build llama-cpp-python with CUDA for Blackwell (sm_120)
# ============================================================================
# FIX: llama-cpp-python's cmake build links CLI tools (llama-mtmd-cli) that
# need libcuda.so.1, which isn't available during Docker builds. The actual
# driver is injected at runtime by nvidia-container-toolkit.
#
# --allow-shlib-undefined tells the linker to accept unresolved symbols from
# shared libraries — correct because libcuda.so.1 WILL be present at runtime.
# ============================================================================
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"
ENV FORCE_CMAKE=1

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements-docker.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir llama-cpp-python && \
    pip install --no-cache-dir -r /tmp/requirements.txt


# --- Stage 2: Runtime image (smaller, no compiler toolchain) ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=8

# libgomp1 is required by llama.cpp for OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up application directory
WORKDIR /app

# Copy application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Models are mounted at runtime, not baked into the image
VOLUME /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python3", "backend/main.py"]
