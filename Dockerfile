# syntax=docker/dockerfile:1
# check=error=true
# =============================================================================
# Izwi Audio - Multi-stage Dockerfile (Rust-native runtime)
#
# Runtime targets:
#   production          CPU server, supervisor and worker (amd64, arm64)
#   production-cuda     the same binaries built with CUDA (amd64); kernels are
#                       PTX for CUDA_COMPUTE_CAP (default 80, Ampere and newer)
#   production-gateway  the standalone API gateway; it links no inference engine
#
# Every base image is pinned by digest; Dependabot moves tags and digests.
# BuildKit cache mounts keep the cargo registry and target directory between
# builds, so `docker compose up --build` after a source change recompiles only
# what changed.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build the React UI
# -----------------------------------------------------------------------------
FROM node:26.13.0-trixie-slim@sha256:5adad64f9a1fca2d51972caa65983634c8a774eb300d72b20f7f2a0f60368072 AS ui-builder

WORKDIR /app/ui

COPY ui/package.json ui/package-lock.json ./
RUN --mount=type=cache,id=izwi-npm,target=/root/.npm \
    npm ci --ignore-scripts --no-audit --no-fund

COPY ui/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: The Rust toolchain. The full rust image already has the C and C++
# toolchain the -sys crates need; nothing links OpenSSL.
# -----------------------------------------------------------------------------
FROM rust:1.99.1-trixie@sha256:8a4e90c7946b74626488b1446f712c0fe0ed1a596cbd3d871684bc9c5433ae07 AS rust-toolchain

# -----------------------------------------------------------------------------
# Stage 2a: Build the Rust backend (CPU)
# -----------------------------------------------------------------------------
FROM rust-toolchain AS rust-builder-cpu
ARG TARGETARCH
# Reported by /internal/health; CUDA runtime evidence fails closed without it.
ARG IZWI_BUILD_GIT_SHA=

# cmake: libopus (izwi-audio-codec's Opus encoder) is built with it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# izwi-server ships with PostgreSQL support. The binaries are copied out of
# the cached target directory, which is not part of the image.
RUN --mount=type=cache,id=izwi-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=izwi-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=izwi-target-cpu-${TARGETARCH},target=/app/target,sharing=locked \
    IZWI_BUILD_GIT_SHA="${IZWI_BUILD_GIT_SHA}" cargo build --release --locked \
        --bin izwi-server --bin izwi-serving-supervisor --bin izwi-serving-worker \
        --features izwi-server/db-postgres \
    && mkdir -p /out \
    && cp target/release/izwi-server target/release/izwi-serving-supervisor \
        target/release/izwi-serving-worker /out/

# -----------------------------------------------------------------------------
# Stage 2b: Build the standalone gateway. It links no inference engine, so this
# stage builds none of the model or accelerator code.
# -----------------------------------------------------------------------------
FROM rust-toolchain AS rust-builder-gateway
ARG TARGETARCH

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN --mount=type=cache,id=izwi-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=izwi-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=izwi-target-gateway-${TARGETARCH},target=/app/target,sharing=locked \
    cargo build --release --locked -p izwi-gateway --features db-postgres \
    && mkdir -p /out/data \
    && cp target/release/izwi-gateway /out/

# -----------------------------------------------------------------------------
# Stage 3: Build the Rust backend (CUDA). The Rust toolchain is copied from the
# digest-pinned rust image above rather than fetched with curl | sh; rustup's
# release binaries run on any glibc distribution.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.5.0-cudnn-devel-ubuntu24.04@sha256:f24a2953ce729567af330c06d850a4f61f3a58df6901f9189d3319ea8881c107 AS rust-builder-cuda
ARG TARGETARCH
ARG CUDA_COMPUTE_CAP=80
ARG IZWI_CUDA_FEATURES=cuda,cudnn
ARG IZWI_BUILD_GIT_SHA=

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} \
    IZWI_CUDA_FEATURES=${IZWI_CUDA_FEATURES}

WORKDIR /app

# git: the FlashAttention build fetches CUTLASS. cmake: libopus
# (izwi-audio-codec's Opus encoder) is built with it. Nothing here comes from
# NVIDIA's apt repository (the image already has the CUDA toolkit), so its
# source goes first: a mirror NVIDIA is still syncing fails `apt-get update`
# with a size mismatch.
RUN rm -f /etc/apt/sources.list.d/cuda* \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --link --from=rust-toolchain /usr/local/rustup /usr/local/rustup
COPY --link --from=rust-toolchain /usr/local/cargo/bin /usr/local/cargo/bin

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN --mount=type=cache,id=izwi-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=izwi-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=izwi-target-cuda-${TARGETARCH},target=/app/target,sharing=locked \
    IZWI_BUILD_GIT_SHA="${IZWI_BUILD_GIT_SHA}" cargo build --release --locked \
        --bin izwi-server --bin izwi-serving-supervisor --bin izwi-serving-worker \
        --features "${IZWI_CUDA_FEATURES},izwi-server/db-postgres" \
    && mkdir -p /out \
    && cp target/release/izwi-server target/release/izwi-serving-supervisor \
        target/release/izwi-serving-worker /out/

# -----------------------------------------------------------------------------
# Stage 4: Production runtime (CPU)
# -----------------------------------------------------------------------------
FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2 AS production

LABEL org.opencontainers.image.title="Izwi Audio" \
      org.opencontainers.image.description="Rust-native audio inference engine" \
      org.opencontainers.image.vendor="Izwi" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/izwi-ai/izwi" \
      org.opencontainers.image.documentation="https://github.com/izwi-ai/izwi/blob/main/docs/user/installation/docker.md"

# Runtime tools the server runs: espeak-ng (Kokoro phonemizer) and zip/unzip
# (Parakeet .nemo repacking). `which` comes with the base image. The server
# runs as uid 1000, owns only its two volumes, and cannot modify its binaries
# or the UI it serves.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        espeak-ng \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 izwi \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /app/data \
        --shell /usr/sbin/nologin izwi \
    && install -d -o 1000 -g 1000 -m 0750 /app/models /app/data

COPY --link --from=rust-builder-cpu /out/ /usr/local/bin/
COPY --link --from=ui-builder /app/ui/dist /app/ui/dist
COPY --link LICENSE NOTICE THIRD_PARTY_LICENSES /usr/share/doc/izwi/

# The server is configured through IZWI_* variables (docker run -e / compose
# environment). History (the database) and media live on the /app/data
# volume, so they survive the container being recreated.
ENV IZWI_UI_DIR=/app/ui/dist \
    IZWI_MODELS_DIR=/app/models \
    IZWI_DB_PATH=/app/data/izwi.sqlite3 \
    IZWI_MEDIA_DIR=/app/data/media \
    IZWI_HOST=0.0.0.0
# The container listens on all interfaces, so the server refuses to start
# unless IZWI_API_KEY (or IZWI_API_KEY_FILE) is set, or IZWI_LOOPBACK_PUBLISHED
# says the port is published on the host's loopback only (see
# docker-compose.yml).

VOLUME ["/app/models", "/app/data"]

EXPOSE 8080

USER 1000:1000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --start-interval=2s --retries=3 \
    CMD ["izwi-server", "healthcheck"]

CMD ["izwi-server"]

# -----------------------------------------------------------------------------
# Stage 5: Production runtime with CUDA support (Docker-only CUDA artifact)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.5.0-cudnn-runtime-ubuntu24.04@sha256:0490edd1b0d88388497dfbe3762d0deab0ded3baf4828616636c17cbd44723e3 AS production-cuda

LABEL org.opencontainers.image.title="Izwi Audio (CUDA)" \
      org.opencontainers.image.description="Rust-native audio inference engine with CUDA support" \
      org.opencontainers.image.vendor="Izwi" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/izwi-ai/izwi" \
      org.opencontainers.image.documentation="https://github.com/izwi-ai/izwi/blob/main/docs/user/installation/docker.md"

# Ubuntu 24.04 images ship an `ubuntu` user with uid 1000; izwi takes it over
# so existing volumes (owned by 1000) keep working. NVIDIA's apt source goes
# first, as in the builder: nothing here comes from it.
RUN rm -f /etc/apt/sources.list.d/cuda* \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        espeak-ng \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && (userdel --remove ubuntu 2>/dev/null || true) \
    && groupadd --gid 1000 izwi \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /app/data \
        --shell /usr/sbin/nologin izwi \
    && install -d -o 1000 -g 1000 -m 0750 /app/models /app/data

COPY --link --from=rust-builder-cuda /out/ /usr/local/bin/
COPY --link --from=ui-builder /app/ui/dist /app/ui/dist
COPY --link LICENSE NOTICE THIRD_PARTY_LICENSES /usr/share/doc/izwi/

# The base image sets NVIDIA_VISIBLE_DEVICES and NVIDIA_DRIVER_CAPABILITIES;
# the compose device request decides which GPUs the container gets.
# izwi's kernels ship as compute_80 PTX, which the driver compiles for the
# GPU on first use and caches under $HOME/.nv (here /app/data, a volume, so
# it survives restarts). The cache is raised to its 4 GiB maximum so no
# compiled module is evicted and recompiled on a later start.
ENV IZWI_UI_DIR=/app/ui/dist \
    IZWI_MODELS_DIR=/app/models \
    IZWI_DB_PATH=/app/data/izwi.sqlite3 \
    IZWI_MEDIA_DIR=/app/data/media \
    IZWI_HOST=0.0.0.0 \
    IZWI_BACKEND=cuda \
    CUDA_CACHE_MAXSIZE=4294967296

VOLUME ["/app/models", "/app/data"]

EXPOSE 8080

USER 1000:1000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --start-interval=2s --retries=3 \
    CMD ["izwi-server", "healthcheck"]

CMD ["izwi-server"]

# -----------------------------------------------------------------------------
# Stage 6: Public API gateway. The same image serves CPU, Metal and CUDA worker
# fleets: the gateway never runs a model. Distroless: no shell or package
# manager, only glibc, libgcc and CA certificates.
# -----------------------------------------------------------------------------
FROM gcr.io/distroless/cc-debian13:nonroot@sha256:cb87acc574abe719b6e5c547384defa2f7b1f24af9605d3f618cf852495d8be2 AS production-gateway

LABEL org.opencontainers.image.title="Izwi Gateway" \
      org.opencontainers.image.description="Izwi public API gateway: OpenAI-compatible routing to approved inference workers" \
      org.opencontainers.image.vendor="Izwi" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/izwi-ai/izwi" \
      org.opencontainers.image.documentation="https://github.com/izwi-ai/izwi/blob/main/docs/dev/PRODUCTION_SERVING_SINGLE_NODE_RUNBOOK.md"

COPY --link --from=rust-builder-gateway /out/izwi-gateway /usr/local/bin/izwi-gateway
# The scoped principal-key store (read when a key manifest is set) lives on
# the /app/data volume, so it survives the container being recreated.
COPY --link --chown=1000:1000 --from=rust-builder-gateway /out/data /app/data
COPY --link LICENSE NOTICE THIRD_PARTY_LICENSES /usr/share/doc/izwi/

ENV IZWI_DB_PATH=/app/data/izwi.sqlite3 \
    IZWI_HOST=0.0.0.0
# The container listens on all interfaces; the gateway then refuses to start
# unless IZWI_GATEWAY_TRUSTED_INGRESS_TLS=1 asserts that TLS terminates in
# front of it.

VOLUME ["/app/data"]

EXPOSE 8080

USER 1000:1000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --start-interval=2s --retries=3 \
    CMD ["izwi-gateway", "healthcheck"]

CMD ["izwi-gateway"]
