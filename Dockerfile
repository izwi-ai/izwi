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

# Build CPU release binaries (server, supervisor, worker); izwi-server ships
# with PostgreSQL support
RUN cargo build --release --locked --bin izwi-server --bin izwi-serving-supervisor --bin izwi-serving-worker --features izwi-server/db-postgres

# -----------------------------------------------------------------------------
# Stage 2b: Build the standalone gateway. It links no inference engine, so this
# stage builds none of the model or accelerator code.
# -----------------------------------------------------------------------------
FROM rust:1.88-bookworm AS rust-builder-gateway

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release --locked -p izwi-gateway --features db-postgres

# -----------------------------------------------------------------------------
# Stage 3: Build the Rust backend (CUDA)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS rust-builder-cuda
ARG CUDA_COMPUTE_CAP=80
ARG IZWI_CUDA_FEATURES=cuda,cudnn

ENV PATH=/root/.cargo/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
ENV IZWI_CUDA_FEATURES=${IZWI_CUDA_FEATURES}

WORKDIR /app

# Install build dependencies and Rust toolchain
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --profile minimal \
    --default-toolchain 1.88.0

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build CUDA release binaries (server, supervisor, worker)
RUN cargo build --release --locked --bin izwi-server --bin izwi-serving-supervisor --bin izwi-serving-worker --features "${IZWI_CUDA_FEATURES},izwi-server/db-postgres"

# -----------------------------------------------------------------------------
# Stage 4: Production runtime (CPU)
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS production

LABEL org.opencontainers.image.title="Izwi Audio"
LABEL org.opencontainers.image.description="Rust-native audio inference engine"
LABEL org.opencontainers.image.vendor="Izwi"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    debianutils \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    tar \
    unzip \
    zip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 izwi

# Copy Rust binaries
COPY --from=rust-builder-cpu /app/target/release/izwi-server /usr/local/bin/izwi-server
COPY --from=rust-builder-cpu /app/target/release/izwi-serving-supervisor /usr/local/bin/izwi-serving-supervisor
COPY --from=rust-builder-cpu /app/target/release/izwi-serving-worker /usr/local/bin/izwi-serving-worker

# Copy built UI
COPY --from=ui-builder /app/ui/dist /app/ui/dist

# License and notice for the binaries this image distributes
COPY LICENSE NOTICE /usr/share/doc/izwi/

# Set up environment. The server is configured through IZWI_* variables
# (docker run -e / compose environment).
ENV IZWI_UI_DIR=/app/ui/dist
ENV IZWI_MODELS_DIR=/app/models
# History (database) and media live on the /app/data volume, so they survive
# the container being recreated.
ENV IZWI_DB_PATH=/app/data/izwi.sqlite3
ENV IZWI_MEDIA_DIR=/app/data/media
# The container must listen on all interfaces; the server then refuses to
# start unless IZWI_API_KEY is provided at runtime (docker run -e IZWI_API_KEY).
ENV IZWI_HOST=0.0.0.0

# Create directories for models and data
RUN mkdir -p /app/models /app/data && \
    chown -R izwi:izwi /app

# Volumes for model storage and for the database and media
VOLUME ["/app/models", "/app/data"]

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
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS production-cuda

LABEL org.opencontainers.image.title="Izwi Audio (CUDA)"
LABEL org.opencontainers.image.description="Rust-native audio inference engine with CUDA support"
LABEL org.opencontainers.image.vendor="Izwi"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    debianutils \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    tar \
    unzip \
    zip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 izwi

# Copy Rust binaries
COPY --from=rust-builder-cuda /app/target/release/izwi-server /usr/local/bin/izwi-server
COPY --from=rust-builder-cuda /app/target/release/izwi-serving-supervisor /usr/local/bin/izwi-serving-supervisor
COPY --from=rust-builder-cuda /app/target/release/izwi-serving-worker /usr/local/bin/izwi-serving-worker

# Copy built UI
COPY --from=ui-builder /app/ui/dist /app/ui/dist

# License and notice for the binaries this image distributes
COPY LICENSE NOTICE /usr/share/doc/izwi/

# Set up environment. The server is configured through IZWI_* variables
# (docker run -e / compose environment).
ENV IZWI_UI_DIR=/app/ui/dist
ENV IZWI_MODELS_DIR=/app/models
# History (database) and media live on the /app/data volume, so they survive
# the container being recreated.
ENV IZWI_DB_PATH=/app/data/izwi.sqlite3
ENV IZWI_MEDIA_DIR=/app/data/media
# The container must listen on all interfaces; the server then refuses to
# start unless IZWI_API_KEY is provided at runtime (docker run -e IZWI_API_KEY).
ENV IZWI_HOST=0.0.0.0
ENV IZWI_BACKEND=cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create directories for models and data
RUN mkdir -p /app/models /app/data && \
    chown -R izwi:izwi /app

# Volumes for model storage and for the database and media
VOLUME ["/app/models", "/app/data"]

# Expose server port
EXPOSE 8080

USER izwi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/readyz || exit 1

# Start the server
CMD ["izwi-server"]

# -----------------------------------------------------------------------------
# Stage 6: Public API gateway (slim). The same image serves CPU, Metal and CUDA
# worker fleets: the gateway never runs a model.
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS production-gateway

LABEL org.opencontainers.image.title="Izwi Gateway"
LABEL org.opencontainers.image.description="Izwi public API gateway: OpenAI-compatible routing to approved inference workers"
LABEL org.opencontainers.image.vendor="Izwi"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 izwi \
    && mkdir -p /app/data \
    && chown -R izwi:izwi /app

COPY --from=rust-builder-gateway /app/target/release/izwi-gateway /usr/local/bin/izwi-gateway

# License and notice for the binary this image distributes
COPY LICENSE NOTICE /usr/share/doc/izwi/

# The scoped principal-key store (read when a key manifest is set) lives on
# the /app/data volume, so it survives the container being recreated.
ENV IZWI_DB_PATH=/app/data/izwi.sqlite3
VOLUME ["/app/data"]

# The container must listen on all interfaces; the gateway then refuses to
# start unless IZWI_GATEWAY_TRUSTED_INGRESS_TLS=1 asserts that TLS terminates
# in front of it.
ENV IZWI_HOST=0.0.0.0

EXPOSE 8080

USER izwi

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/readyz || exit 1

CMD ["izwi-gateway"]
