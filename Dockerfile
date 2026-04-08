# --- Stage 1: Builder ---
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 AS builder

# Install Python 3.11
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Copy uv binaries from official image
COPY --from=uv_bin /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for speed
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies using the lockfile
# Mounting the cache speeds up repeated builds significantly
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# --- Stage 2: Runtime ---
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install Python 3.11 runtime
RUN apt-get update && apt-get install -y python3.11 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment and project code
COPY --from=builder /app/.venv /app/.venv
COPY . /app

# Add venv to PATH so 'python' points to our locked environment
ENV PATH="/app/.venv/bin:$PATH"

# Default to training the VAE
CMD ["python", "train_vae.py"]