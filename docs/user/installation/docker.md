---
title: "Docker"
description: "Run Izwi in Docker: an open development setup by default, production with one overlay."
sidebarTitle: "Docker"
icon: "docker"
---
One image and one Compose setup serve both development and production. Plain `docker compose up` gives you a development server that behaves like Izwi running natively on your machine. Production adds a second Compose file, which you switch on with two lines in `.env`.

---

## Requirements

- **Docker Engine 28.3.3 or newer** on Linux, or Docker Desktop on macOS and Windows. Older Linux engines could let other machines on the same network reach ports published on `127.0.0.1`, which the development setup relies on to stay private. See [Keep the host patched](#keep-the-host-patched).
- **Docker Compose v2** with support for the `!override` merge tag (used by the production overlay).
- **For CUDA**: NVIDIA driver 580 or newer and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The CUDA image targets Ampere (compute capability 8.0) or newer GPUs.

---

## Development (the default)

```bash
git clone https://github.com/izwi-ai/izwi.git
cd izwi
docker compose up -d
```

Open [http://localhost:8080](http://localhost:8080). No API key is needed.

For an NVIDIA GPU, start the CUDA service instead. Name it: both services publish the same port.

```bash
docker compose --profile cuda up -d izwi-cuda
```

After changing the source, rebuild and restart with `docker compose up -d --build`, or leave `docker compose watch` running to do that whenever a file under `crates/` or `ui/` changes. The build caches compiled dependencies, so a rebuild recompiles only what changed.

### Using a published image

Releases after 0.1.0 also ship signed images on the GitHub Container Registry, so you can skip the build. Verify the signature with [cosign](https://docs.sigstore.dev/cosign/system_config/installation/) 3 or newer, then tag the image as the one Compose uses:

```bash
VERSION=0.1.1   # the release you want, from the Releases page
docker pull "ghcr.io/izwi-ai/izwi:${VERSION}"
cosign verify "ghcr.io/izwi-ai/izwi:${VERSION}" \
  --certificate-identity-regexp '^https://github.com/izwi-ai/izwi/\.github/workflows/release\.yml@refs/tags/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
docker tag "ghcr.io/izwi-ai/izwi:${VERSION}" izwi:latest
docker compose up -d
```

For the CUDA service, use the `${VERSION}-cuda` tag and tag it `izwi:cuda`. The release notes list every image and the matching `gh attestation verify` command.

### What the development setup allows

- The port is published on the host's `127.0.0.1` only, so other machines cannot reach it.
- Inside the container the server runs in loopback-only mode (`IZWI_LOOPBACK_PUBLISHED=1`). It needs no API key and answers only `localhost`-style `Host` names, so a web page cannot reach it through DNS rebinding.
- Any program on this machine can use the API, just as with a native install.
- Other containers on the Compose network that call the server by service name need that name allowed: set `IZWI_ALLOWED_HOSTS=izwi` in `.env`.
- Setting `IZWI_API_KEY` in development is optional; when it is set, it is enforced.

---

## Production

Write a `.env` file next to `docker-compose.yml` that adds the production overlay and holds a new API key, then start the server as before:

```bash
cat > .env <<EOF
COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml
COMPOSE_PATH_SEPARATOR=:
IZWI_API_KEY=$(openssl rand -hex 32)
EOF
chmod 600 .env
docker compose up -d
```

Clients send the key as `Authorization: Bearer <key>`, and the web UI asks for it once. To read it back, run `grep IZWI_API_KEY .env`.

```bash
curl -H "Authorization: Bearer $(grep IZWI_API_KEY .env | cut -d= -f2)" http://127.0.0.1:8080/v1/models
```

The overlay (`docker-compose.prod.yml`) changes these settings and nothing else:

| Setting | Development | Production |
|---------|-------------|------------|
| API key | Optional | Required (`IZWI_API_KEY` or `IZWI_API_KEY_FILE`); `docker compose` refuses to start without it |
| Published on | `127.0.0.1` | `IZWI_PUBLISH_ADDR` (default `127.0.0.1`) |
| Root filesystem | Writable | Read-only, with a small `/tmp` |
| Limits | None | 4096 processes and threads; memory from `IZWI_MEM_LIMIT` (for example `24g`) when set |
| Restart policy | None | `unless-stopped` |
| Log format | Text | JSON |

Both modes run as the unprivileged `izwi` user (uid 1000) with every Linux capability dropped and privilege escalation disabled, and on a server with a key the public `/livez` and `/readyz` answer with only the status and version. Capacity, queues and startup warnings are on `/v1/health`, which needs the key.

### Reaching the server from other machines

The server speaks plain HTTP, so anything that crosses a network should go through a TLS-terminating reverse proxy. The production overlay includes one: Caddy, which gets and renews a certificate automatically. Point a DNS name at the host, open ports 80 and 443, and add to `.env`:

```bash
IZWI_DOMAIN=izwi.example.com
IZWI_ALLOWED_HOSTS=izwi.example.com
```

Then start it with the `tls` profile:

```bash
docker compose --profile tls up -d
```

For the CUDA service, also set `IZWI_TLS_UPSTREAM=izwi-cuda:8080`. To use your own proxy instead, run it on the same host, keep the default `IZWI_PUBLISH_ADDR=127.0.0.1`, and set `IZWI_ALLOWED_HOSTS` to the public host name, so the server answers only that name and `localhost`. Per-client rate limits belong in the proxy.

Setting `IZWI_PUBLISH_ADDR=0.0.0.0` publishes the port on every interface, and the API key then travels unencrypted. Docker's port rules bypass host firewalls such as UFW, so do this only on a network you trust.

Never set `IZWI_ALLOW_UNAUTHENTICATED_NETWORK` in a container. It removes the API key requirement for every client that can reach the port.

### Keeping the key out of the environment

A key in `.env` reaches the container as an environment variable, which `docker inspect` shows to anyone who can use Docker on the host. The server can read the key from a file instead: set `IZWI_API_KEY_FILE` to its absolute path inside the container and leave `IZWI_API_KEY` unset. Orchestrators that mount secrets (Docker Swarm, Kubernetes, systemd) do this directly. With Compose, add a file of your own, for example `docker-compose.secret.yml`:

```yaml
services:
  izwi:
    secrets: [izwi_api_key]
secrets:
  izwi_api_key:
    file: ./secrets/izwi_api_key
```

Then write the key to `./secrets/izwi_api_key` and, in `.env`, replace the `IZWI_API_KEY` line with `IZWI_API_KEY_FILE=/run/secrets/izwi_api_key` and append `:docker-compose.secret.yml` to `COMPOSE_FILE`. The file must be readable by uid 1000.

### Offline and proxied networks

- `IZWI_OFFLINE=1` stops the server from contacting Hugging Face at all: model downloads are refused and sizes come from built-in estimates. Copy model files into the `izwi-models` volume yourself.
- Downloads use `HTTPS_PROXY` (with `NO_PROXY` exceptions) when it is set.

Add either under `environment:` in a compose file of your own, listed after the others in `COMPOSE_FILE`.

---

## Data

- Models live in the `izwi-models` volume.
- History (the database) and media live in `izwi-data`.

Both survive `docker compose down` and `docker compose up --force-recreate`. `docker compose down -v` **deletes them**.

### Upgrading

```bash
git pull
docker compose up -d --build
```

The new container mounts the same volumes, and the server migrates the database when it starts. Back up `izwi-data` first.

The server runs as uid 1000 with no capabilities, so it cannot write files that another user owns. If models were downloaded by the development container from Izwi 0.1.0 or earlier, which ran as root, hand the volume to uid 1000 once:

```bash
docker run --rm -v izwi-models:/models debian:trixie-slim chown -R 1000:1000 /models
```

### Backup and restore

Stop the server so the database is consistent, archive the volume, then start it again:

```bash
docker compose stop
docker run --rm -v izwi-data:/data:ro -v "$PWD":/backup debian:trixie-slim \
  tar czf /backup/izwi-data-backup.tgz -C /data .
docker compose start
```

To restore, stop the server, then unpack the archive into the volume:

```bash
docker compose stop
docker run --rm -v izwi-data:/data -v "$PWD":/backup debian:trixie-slim \
  sh -c 'find /data -mindepth 1 -delete && tar xzf /backup/izwi-data-backup.tgz -C /data'
docker compose start
```

## Keep the host patched

Container isolation is only as strong as the host's container runtime. Keep Docker Engine (or Docker Desktop), runc, containerd and, for CUDA, the NVIDIA Container Toolkit up to date. In particular:

- **Docker Engine 28.3.3 or newer** on Linux. Older engines could expose ports published on `127.0.0.1` to the local network.
- **runc 1.2.8, 1.3.3 or newer**, which fix the November 2025 container-escape vulnerabilities (CVE-2025-31133, CVE-2025-52565, CVE-2025-52881).
- **NVIDIA Container Toolkit 1.17.8 or newer** (CVE-2025-23266).
- **NVIDIA driver 580 or newer** for the CUDA image, which is built with CUDA 13.0.

## Without Compose

The image refuses to start on `0.0.0.0` without a key, because it cannot tell how its port is published. For development, publish on loopback and say so:

```bash
docker run -d --name izwi \
  -p 127.0.0.1:8080:8080 -e IZWI_LOOPBACK_PUBLISHED=1 \
  -v izwi-models:/app/models -v izwi-data:/app/data \
  izwi:latest
```

Always use named volumes as above. Without `-v`, every `docker run` creates fresh anonymous volumes, and earlier history looks lost.
