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

# Deterministic default filler vocabulary for workload prompt construction.
BENCH_VOCAB = [
    "amber", "basalt", "cobalt", "delta", "ember", "fjord", "granite",
    "harbor", "indigo", "juniper", "karst", "lagoon", "marble", "nectar",
    "opal", "pine", "quartz", "ridge", "slate", "timber",
]

# Uniqueness budget for per-request markers: the marker must keep the common
# token prefix of any two requests below one KV page (16 tokens) while
# remaining shorter than a page itself.
MARKER_UNIQUE_REQUESTS = 150000
MAX_MARKER_WORDS = 15


def _filler_words(count, salt, vocab):
    """Deterministic filler words; `salt` rotates the cycle per request."""
    return [vocab[(salt * 7 + i * 3) % len(vocab)] for i in range(count)]


def _marker_words(index, vocab):
    """Least-significant-first positional marker over `vocab`.

    Consecutive requests differ in the first marker word, so the shared token
    prefix of any two requests is at most the marker length — well under one
    KV page for vocabularies of at least two words.
    """
    base = len(vocab)
    digits = max(2, math.ceil(math.log(MARKER_UNIQUE_REQUESTS, base)))
    if digits > MAX_MARKER_WORDS:
        raise ValueError(
            f"vocabulary of {base} words needs {digits} marker words "
            f"(max {MAX_MARKER_WORDS}); use a larger --vocab"
        )
    return [vocab[(index // base**position) % base] for position in range(digits)]


def marker_word_count(vocab):
    base = len(vocab)
    return max(2, math.ceil(math.log(MARKER_UNIQUE_REQUESTS, base)))


def build_messages(workload, prefix_tokens, suffix_tokens, index, vocab=None):
    """Chat messages for the requested workload.

    Sizing treats one whitespace-delimited word as one token: approximate for
    a real tokenizer but deterministic, and the DS1.5 rig sets the
    chunked-prefill threshold above the whole prompt, so the estimate only
    needs to bound the prompt, not match it exactly. `vocab` should list words
    the target tokenizer maps to distinct token ids; with the DS1.5 fixture
    tokenizer an out-of-vocabulary word collapses to the unk token, which
    would make even "cold" prompts share prefixes.

    - default: the original fixed prompt, byte-identical to the pre-workload
      harness.
    - shared: an identical system prefix on every request plus a per-request
      unique user suffix; the cross-request share is exactly the prefix.
    - cold: a per-request unique user message of the same total length; the
      leading positional marker keeps the common token prefix below one page.
    """
    words = list(vocab) if vocab is not None else BENCH_VOCAB
    if workload == "default":
        return [{"role": "user", "content": "Say the word ready."}]
    marker_len = marker_word_count(words)
    if workload == "shared":
        if suffix_tokens < marker_len + 1:
            raise ValueError(
                f"workload shared needs --suffix-tokens >= {marker_len + 1} "
                f"for a {len(words)}-word vocabulary (marker uniqueness)"
            )
        prefix = " ".join(_filler_words(prefix_tokens, 0, words))
        suffix_words = _marker_words(index, words)
        suffix_words += _filler_words(suffix_tokens - len(suffix_words), index + 1, words)
        return [
            {"role": "system", "content": prefix},
            {"role": "user", "content": " ".join(suffix_words)},
        ]
    if workload == "cold":
        total = prefix_tokens + suffix_tokens
        if total < marker_len + 1:
            raise ValueError(
                f"workload cold needs --prefix-tokens + --suffix-tokens >= "
                f"{marker_len + 1} for a {len(words)}-word vocabulary "
                f"(marker uniqueness)"
            )
        user_words = _marker_words(index, words)
        user_words += _filler_words(total - len(user_words), index + 1, words)
        return [{"role": "user", "content": " ".join(user_words)}]
    raise ValueError(f"unknown workload: {workload}")


def build_request(gateway, api_key, model, messages, max_tokens, stream, request_id):
    payload = json.dumps({
        "model": model,
        "messages": messages,
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


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def run_one(gateway, api_key, model, max_tokens, stream, index,
            workload="default", prefix_tokens=0, suffix_tokens=0, vocab=None,
            max_retries=0):
    """Run one request; return (ok, rejected, ttft_ms, latency_ms, detail).

    A capacity rejection (429/503) is retried up to `max_retries` times with a
    short backoff before it is reported as rejected; the gateway sheds load on
    stale capacity credits, so a closed-loop client without retries sheds most
    of its own load no matter how lightly it is actually loaded.
    """
    messages = build_messages(workload, prefix_tokens, suffix_tokens, index, vocab)
    for attempt in range(max_retries + 1):
        request = build_request(
            gateway, api_key, model, messages, max_tokens, stream, f"bench-{index:06d}"
            if attempt == 0 else f"bench-{index:06d}-r{attempt}"
        )
        started = time.monotonic()
        ok, rejected, ttft_ms, latency_ms, detail = run_attempt(
            request, stream, started
        )
        if ok or not rejected or attempt == max_retries:
            return (ok, rejected and attempt == max_retries, ttft_ms, latency_ms, detail)
        time.sleep(min(0.025 * 2 ** attempt, 0.2))
    raise AssertionError("unreachable")


def run_attempt(request, stream, started):
    """Run one HTTP attempt; return (ok, rejected, ttft_ms, latency_ms, detail)."""
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
    parser.add_argument(
        "--workload",
        choices=["default", "shared", "cold"],
        default="default",
        help="default: fixed prompt; shared: constant system prefix + per-request "
        "unique user suffix (prefix-cache reuse expected); cold: unique "
        "per-request prompts of the same total length (no reuse expected)",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=64,
        help="approximate word count of the workload prefix (shared/cold)",
    )
    parser.add_argument(
        "--suffix-tokens",
        type=int,
        default=16,
        help="approximate word count of the per-request suffix (shared) / "
        "tail (cold)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="retry 429/503 rejections up to N times with short backoff "
        "(default 0: report the first rejection, legacy behavior)",
    )
    parser.add_argument(
        "--vocab",
        default=",".join(BENCH_VOCAB),
        help="comma-separated words the target tokenizer maps to distinct "
        "token ids; workload prompts are built from these (shared) and "
        "per-request markers are encoded over them",
    )
    parser.add_argument("--metadata", required=True, help="JSON hardware/deployment metadata echoed into evidence")
    parser.add_argument("--output", required=True, help="Where to write the JSON evidence record")
    args = parser.parse_args()

    if args.max_retries < 0 or args.max_retries > 1000:
        parser.error("--max-retries must be between 0 and 1000")
    if args.requests <= 0 or args.requests > 100000:
        parser.error("--requests must be between 1 and 100000")
    if args.concurrency <= 0 or args.concurrency > 64:
        parser.error("--concurrency must be between 1 and 64")
    if args.max_tokens <= 0 or args.max_tokens > 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    if args.workload in ("shared", "cold"):
        if args.prefix_tokens <= 0 or args.prefix_tokens > 4096:
            parser.error("--prefix-tokens must be between 1 and 4096 for shared/cold")
        if args.suffix_tokens <= 0 or args.suffix_tokens > 4096:
            parser.error("--suffix-tokens must be between 1 and 4096 for shared/cold")
        if args.prefix_tokens + args.suffix_tokens > 4096:
            parser.error("--prefix-tokens + --suffix-tokens must not exceed 4096")
    vocab = [word for word in args.vocab.split(",") if word]
    if len(vocab) < 2:
        parser.error("--vocab needs at least two distinct words")
    if len(set(vocab)) != len(vocab):
        parser.error("--vocab words must be distinct")
    if args.workload in ("shared", "cold"):
        try:
            build_messages(args.workload, args.prefix_tokens, args.suffix_tokens, 0, vocab)
        except ValueError as error:
            parser.error(str(error))
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
            pool.submit(
                run_one, args.gateway.rstrip("/"), args.api_key, args.model,
                args.max_tokens, args.stream, index,
                args.workload, args.prefix_tokens, args.suffix_tokens, vocab,
                args.max_retries,
            )
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
        "max_retries": args.max_retries,
        "stream": args.stream,
        "workload": args.workload,
        "prefix_tokens": args.prefix_tokens if args.workload in ("shared", "cold") else 0,
        "suffix_tokens": args.suffix_tokens if args.workload in ("shared", "cold") else 0,
        "vocab": vocab if args.workload in ("shared", "cold") else None,
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
