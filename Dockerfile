# =============================================================================
# Izwi Audio - Multi-stage Dockerfile (Rust-native runtime)
# Public native releases are CPU-only. CUDA is compiled and shipped through the
# Docker production-cuda target, which uses the NVIDIA CUDA runtime image.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build the React UI
# -----------------------------------------------------------------------------
FROM node:24-slim AS ui-builder

WORKDIR /app/ui

# Copy package files first for better caching
COPY ui/package*.json ./

# Install dependencies
RUN npm ci --ignore-scripts

# Copy source and build
COPY ui/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Build the Rust backend (CPU)
# -----------------------------------------------------------------------------
FROM rust:1.88-bookworm AS rust-builder-cpu

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build CPU release binary (server only for Docker)
RUN cargo build --release --locked --bin izwi-server

# -----------------------------------------------------------------------------
# Stage 3: Build the Rust backend (CUDA)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS rust-builder-cuda
ARG CUDA_COMPUTE_CAP=80

ENV PATH=/root/.cargo/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

WORKDIR /app

# Install build dependencies and Rust toolchain
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --profile minimal \
    --default-toolchain 1.88.0

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build CUDA release binary (server only for Docker)
RUN cargo build --release --locked --bin izwi-server --features cuda

# -----------------------------------------------------------------------------
# Stage 4: Production runtime (CPU)
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS production

LABEL org.opencontainers.image.title="Izwi Audio"
LABEL org.opencontainers.image.description="Rust-native audio inference engine"
LABEL org.opencontainers.image.vendor="Agentem"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 izwi

# Copy Rust binary
COPY --from=rust-builder-cpu /app/target/release/izwi-server /usr/local/bin/izwi-server

# Copy built UI
COPY --from=ui-builder /app/ui/dist /app/ui/dist

# Copy configuration (use Docker-specific config)
COPY config.docker.toml /app/config.toml

# Set up environment
ENV IZWI_CONFIG_PATH=/app/config.toml
ENV IZWI_UI_DIR=/app/ui/dist
ENV IZWI_MODELS_DIR=/app/models

# Create directories for models and data
RUN mkdir -p /app/models /app/data && \
    chown -R izwi:izwi /app

# Volume for model storage
VOLUME ["/app/models"]

# Expose server port
EXPOSE 8080

USER izwi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/readyz || exit 1

# Start the server
CMD ["izwi-server"]

# -----------------------------------------------------------------------------
# Stage 5: Production runtime with CUDA support (Docker-only CUDA artifact)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS production-cuda

LABEL org.opencontainers.image.title="Izwi Audio (CUDA)"
LABEL org.opencontainers.image.description="Rust-native audio inference engine with CUDA support"
LABEL org.opencontainers.image.vendor="Agentem"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 izwi

# Copy Rust binary
COPY --from=rust-builder-cuda /app/target/release/izwi-server /usr/local/bin/izwi-server

# Copy built UI
COPY --from=ui-builder /app/ui/dist /app/ui/dist

# Copy configuration (use Docker-specific config)
COPY config.docker.toml /app/config.toml

# Set up environment
ENV IZWI_CONFIG_PATH=/app/config.toml
ENV IZWI_UI_DIR=/app/ui/dist
ENV IZWI_MODELS_DIR=/app/models
ENV IZWI_BACKEND=cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create directories for models and data
RUN mkdir -p /app/models /app/data && \
    chown -R izwi:izwi /app

# Volume for model storage
VOLUME ["/app/models"]

# Expose server port
EXPOSE 8080

USER izwi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/readyz || exit 1

# Start the server
CMD ["izwi-server"]
