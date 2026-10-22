---
title: "Docker"
description: "Run Izwi in Docker: an open development setup by default, production with one overlay."
sidebarTitle: "Docker"
icon: "docker"
---
One image and one Compose setup serve both development and production. Plain `docker compose up` gives you a development server that behaves like Izwi running natively on your machine. Production adds a second Compose file, which you switch on with two lines in `.env`.

---

## Requirements

- **Docker Engine 28.0 or newer** on Linux, or Docker Desktop on macOS and Windows. Older Linux engines let other machines on the same network reach ports published on `127.0.0.1`, which the development setup relies on to stay private.
- **Docker Compose v2** with support for the `!override` merge tag (used by the production overlay).
- **For CUDA**: an NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The CUDA image targets Ampere (compute capability 8.0) or newer GPUs.

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

After changing the source, rebuild and restart with `docker compose up -d --build`.

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
| API key | Optional | Required; `docker compose` refuses to start without it |
| Published on | `127.0.0.1` | `IZWI_PUBLISH_ADDR` (default `127.0.0.1`) |
| Restart policy | None | `unless-stopped` |
| Log format | Text | JSON |

### Reaching the server from other machines

The server speaks plain HTTP, so anything that crosses a network should go through a TLS-terminating reverse proxy:

- Run the proxy on the same host and keep the default `IZWI_PUBLISH_ADDR=127.0.0.1`.
- Set `IZWI_ALLOWED_HOSTS` to the public host name, so the server answers only that name and `localhost`.

Setting `IZWI_PUBLISH_ADDR=0.0.0.0` publishes the port on every interface, and the API key then travels unencrypted. Docker's port rules bypass host firewalls such as UFW, so do this only on a network you trust.

Never set `IZWI_ALLOW_UNAUTHENTICATED_NETWORK` in a container. It removes the API key requirement for every client that can reach the port.

---

## Data

- Models live in the `izwi-models` volume.
- History (the database) and media live in `izwi-data`.

Both survive `docker compose down` and `docker compose up --force-recreate`. `docker compose down -v` **deletes them**.

## Without Compose

The image refuses to start on `0.0.0.0` without a key, because it cannot tell how its port is published. For development, publish on loopback and say so:

```bash
docker run -d --name izwi \
  -p 127.0.0.1:8080:8080 -e IZWI_LOOPBACK_PUBLISHED=1 \
  -v izwi-models:/app/models -v izwi-data:/app/data \
  izwi:latest
```

Always use named volumes as above. Without `-v`, every `docker run` creates fresh anonymous volumes, and earlier history looks lost.
