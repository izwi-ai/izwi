#!/usr/bin/env python3
"""Closed-loop latency/throughput harness for the gateway/worker chat profile.

Drives gateway POST /v1/chat/completions (JSON and SSE) at a fixed client
concurrency and reports client-observed TTFT, latency percentiles,
throughput, and completed/rejected/failed rates. Requires --metadata JSON
describing deployed_sha, topology, worker backend, model, and artifact
revision; metadata is echoed into the evidence record unchanged. This is an
opt-in benchmark; fixtures make no speed claim.
"""
import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request

MAX_REQUEST_BODY_BYTES = 64 * 1024
MAX_RESPONSE_BODY_BYTES = 4 * 1024 * 1024
MAX_SSE_EVENTS = 8192
MAX_EVENT_LINE_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECS = 300


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def build_request(gateway, api_key, model, max_tokens, stream, request_id):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say the word ready."}],
        "max_tokens": max_tokens,
        "stream": stream,
    }).encode("utf-8")
    assert len(payload) <= MAX_REQUEST_BODY_BYTES
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-request-id": request_id,
    }
    return urllib.request.Request(
        f"{gateway}/v1/chat/completions", data=payload, headers=headers, method="POST"
    )


def run_one(gateway, api_key, model, max_tokens, stream, index):
    """Run one request; return (ok, rejected, ttft_ms, latency_ms, detail)."""
    request = build_request(gateway, api_key, model, max_tokens, stream, f"bench-{index:06d}")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECS) as response:
            if response.status == 429 or response.status == 503:
                response.read(4096)
                return (False, True, None, (time.monotonic() - started) * 1000.0, response.status)
            if response.status != 200:
                response.read(4096)
                return (False, False, None, (time.monotonic() - started) * 1000.0, response.status)
            ttft_ms = None
            total = 0
            events = 0
            buf = b""
            if stream:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_RESPONSE_BODY_BYTES:
                        return (False, False, None, (time.monotonic() - started) * 1000.0, "oversize")
                    for raw in chunk.split(b"\n"):
                        buf += raw
                        if len(buf) > MAX_EVENT_LINE_BYTES:
                            return (False, False, None, (time.monotonic() - started) * 1000.0, "oversize-line")
                        if raw.endswith(b"\r"):
                            raw = raw[:-1]
                        if raw.startswith(b"data:"):
                            line = raw[5:].strip()
                            if line == b"[DONE]":
                                buf = b""
                                break
                            events += 1
                            if events > MAX_SSE_EVENTS:
                                return (False, False, None, (time.monotonic() - started) * 1000.0, "too-many-events")
                            if ttft_ms is None:
                                ttft_ms = (time.monotonic() - started) * 1000.0
                            buf = b""
                    else:
                        continue
                    break
            else:
                body = response.read(MAX_RESPONSE_BODY_BYTES + 1)
                if len(body) > MAX_RESPONSE_BODY_BYTES:
                    return (False, False, None, (time.monotonic() - started) * 1000.0, "oversize")
                parsed = json.loads(body.decode("utf-8"))
                if not parsed.get("choices"):
                    return (False, False, None, (time.monotonic() - started) * 1000.0, "empty-choices")
            latency_ms = (time.monotonic() - started) * 1000.0
            return (True, False, ttft_ms, latency_ms, None)
    except urllib.error.HTTPError as error:
        try:
            error.read(4096)
        except Exception:
            pass
        return (False, error.code in (429, 503), None, (time.monotonic() - started) * 1000.0, error.code)
    except Exception as error:
        return (False, False, None, (time.monotonic() - started) * 1000.0, type(error).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", required=True, help="Gateway base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--api-key", required=True, help="Gateway inference API key")
    parser.add_argument("--model", required=True, help="Public model alias to request")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata", required=True, help="JSON hardware/deployment metadata echoed into evidence")
    parser.add_argument("--output", required=True, help="Where to write the JSON evidence record")
    args = parser.parse_args()

    if args.requests <= 0 or args.requests > 100000:
        parser.error("--requests must be between 1 and 100000")
    if args.concurrency <= 0 or args.concurrency > 64:
        parser.error("--concurrency must be between 1 and 64")
    if args.max_tokens <= 0 or args.max_tokens > 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    metadata = json.loads(args.metadata)
    if not isinstance(metadata, dict):
        parser.error("--metadata must be a JSON object")

    started = time.monotonic()
    completed = rejected = failed = 0
    ttfts = []
    latencies = []
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_one, args.gateway.rstrip("/"), args.api_key, args.model,
                        args.max_tokens, args.stream, index)
            for index in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            ok, was_rejected, ttft_ms, latency_ms, detail = future.result()
            latencies.append(latency_ms)
            if ok:
                completed += 1
                if ttft_ms is not None:
                    ttfts.append(ttft_ms)
            elif was_rejected:
                rejected += 1
            else:
                failed += 1
                failures[str(detail)] = failures.get(str(detail), 0) + 1
    wall_secs = time.monotonic() - started

    record = {
        "route": "POST /v1/chat/completions",
        "stream": args.stream,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_secs": wall_secs,
        "completed": completed,
        "rejected": rejected,
        "failed": failed,
        "failures": failures,
        "throughput_req_per_s": completed / wall_secs if wall_secs > 0 else 0.0,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "ttft_ms": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        } if ttfts else None,
        "metadata": metadata,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    print(json.dumps({
        "completed": completed, "rejected": rejected, "failed": failed,
        "throughput_req_per_s": record["throughput_req_per_s"],
        "latency_p95_ms": record["latency_ms"]["p95"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
