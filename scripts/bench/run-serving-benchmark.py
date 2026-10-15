#!/usr/bin/env python3
"""Open-loop serving benchmark for izwi chat completions (local role or gateway).

Sends streamed OpenAI chat completions with Poisson (or gamma) arrivals at one
or more request rates, optionally capped at a maximum concurrency, and records
a timestamp for every streamed delta. Reports TTFT, inter-token latency (every
gap of every request, not per-request means), TPOT, end-to-end latency,
request and token throughput, goodput under SLOs, and error/abort counts.

Results are JSON (schema "izwi.serving_benchmark", version 1) carrying git,
model, backend, host and server-config metadata, and, with --metrics-url,
server-side histogram and counter deltas per sweep point. Each sweep point is a
`reports[]` entry whose summary uses the same field names as `izwi bench`, so
`izwi bench compare` also reads these files.

Standard library only: runs with plain python3. See docs/dev/SERVING_BENCHMARK.md.
"""
import argparse
import asyncio
import collections
import datetime
import json
import math
import os
import platform
import random
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = "izwi.serving_benchmark"
SCHEMA_VERSION = 1
PERCENTILES = (50, 90, 95, 99)
MAX_EVENT_BYTES = 1 << 20
MAX_ERROR_BODY_BYTES = 4096
METADATA_TIMEOUT_SECS = 10

# Common English words; most BPE vocabularies map " word" to one token, so a
# prompt of N words is close to N prompt tokens. The server's usage block is
# the source of truth for the token counts the report uses.
SYNTHETIC_WORDS = (
    "about above across after again air all almost along also always among and animal another answer any "
    "appear area around ask away back base be became because become been before began begin behind being "
    "below best better between big black blue boat body book both bring brought build built call came can "
    "car carry cause center change check children city class clear close cold color come common complete "
    "could country course cover cross cut dark day deep did different direction do does dog done door "
    "down draw dry during each early earth east eat end enough even ever every eye face fact fall family "
    "far farm fast father feel feet few field figure fill final find fine fire first fish five fly follow "
    "food foot for force form found four free friend from front full game gave get girl give go gold good "
    "got great green ground group grow had half hand happen hard has have head hear heard heat help her "
    "here high hold home horse hot hour house how idea important inch include into island just keep kind "
    "king knew know land language large last late laugh lead learn leave left less letter life light like "
    "line list little live long look low machine made main make man many map mark may mean measure men "
    "might mile mind minute miss money moon more morning most mother mountain move much music must name "
    "near need never new next night north note nothing notice now number object ocean off often old once "
    "only open order other our out over own page paper part pass past pattern people perhaps person "
    "picture piece place plain plan plant play point port pose possible power press problem produce "
    "product pull put question quick rain ran reach read ready real record red remember rest right river "
    "road rock room round rule run said same saw say school science sea second see seem self sentence "
    "serve set several shape ship short should show side simple since sing size sleep slow small snow "
    "some song soon sound south space special spell stand star start state stay step still stood stop "
    "story street strong study such sun sure surface system table tail take talk teach tell ten test "
    "than that their them then there these they thing think this those though thought three through time "
    "together told too took top toward town travel tree true try turn two under unit until upon usual "
    "very voice vowel walk wall want warm watch water wave way week weight well went were west what wheel "
    "when where which while white whole why wide will wind winter wish with without wonder wood word work "
    "world would write year young"
).split()

PROFILES = {
    # Fixed-prompt regression check: seeded synthetic prompts, greedy decoding,
    # one single-stream point (latency) and one 4-way point (batching).
    "regression": {
        "dataset": "synthetic",
        "num_prompts": 24,
        "input_len": 128,
        "output_len": 32,
        "input_len_range": 0.0,
        "output_len_range": 0.0,
        "request_rate": "inf",
        "max_concurrency": "1,4",
        "burstiness": 1.0,
        "temperature": 0.0,
        "seed": 0,
        "warmup": 2,
    },
}

DEFAULTS = {
    "dataset": "synthetic",
    "num_prompts": 200,
    "input_len": 256,
    "output_len": 128,
    "input_len_range": 0.0,
    "output_len_range": 0.0,
    "request_rate": "inf",
    "max_concurrency": None,
    "burstiness": 1.0,
    "temperature": 0.0,
    "seed": 0,
    "warmup": 0,
}

# Metrics the report-only regression check compares: (name, summary path,
# lower is better).
REGRESSION_METRICS = (
    ("throughput_rps", ("throughput_rps",), False),
    ("output_throughput_tps", ("output_throughput_tps",), False),
    ("ttft_ms.p50", ("ttft_ms", "p50"), True),
    ("ttft_ms.p95", ("ttft_ms", "p95"), True),
    ("tpot_ms.p50", ("tpot_ms", "p50"), True),
    ("itl_ms.p95", ("itl_ms", "p95"), True),
    ("end_to_end_ms.p95", ("end_to_end_ms", "p95"), True),
)


# ---------------------------------------------------------------- statistics

def percentile(values, pct):
    """Linear-interpolation percentile (numpy's default), `pct` in [0, 100]."""
    if not values:
        return None
    return _percentile_sorted(sorted(values), pct)


def _percentile_sorted(ordered, pct):
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def min_samples_for(pct):
    """Samples needed before percentile `pct` is not simply the maximum."""
    return math.ceil(100.0 / (100.0 - pct))


def distribution(values):
    """count/mean/std/min/max and the report percentiles of `values`."""
    if not values:
        return None
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    out = {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": ordered[0],
        "max": ordered[-1],
    }
    for pct in PERCENTILES:
        out[f"p{pct}"] = _percentile_sorted(ordered, pct)
    return out


def low_sample_warnings(name, values):
    """Warn when a reported percentile is the sample maximum in disguise."""
    count = len(values)
    if count == 0:
        return []
    short = [pct for pct in PERCENTILES if count < min_samples_for(pct)]
    if not short:
        return []
    worst = max(short)
    listed = ", ".join(f"p{pct}" for pct in short)
    verb = "is" if len(short) == 1 else "are"
    return [
        f"{name}: {count} samples; p{worst} needs at least {min_samples_for(worst)} "
        f"({listed} {verb} at or near the maximum)"
    ]


# ------------------------------------------------------------------ arrivals

def parse_rate(raw):
    text = str(raw).strip().lower()
    if text in ("inf", "infinity"):
        return math.inf
    rate = float(text)
    if not rate > 0 or not math.isfinite(rate):
        raise ValueError(f"request rate must be positive or inf, got {raw!r}")
    return rate


def arrival_offsets(count, rate, burstiness, rng):
    """Send times (seconds from the point's start) for `count` requests.

    Inter-arrival gaps are gamma distributed with shape `burstiness` and mean
    1/rate: burstiness 1 is a Poisson process, below 1 burstier, above 1 more
    regular (vLLM's benchmark_serving semantics). An infinite rate sends
    everything at once.
    """
    if count <= 0:
        return []
    if math.isinf(rate):
        return [0.0] * count
    if burstiness <= 0:
        raise ValueError("burstiness must be positive")
    scale = 1.0 / (rate * burstiness)
    offsets = [0.0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + rng.gammavariate(burstiness, scale))
    return offsets


# ------------------------------------------------------------------- dataset

def sample_length(mean, range_ratio, rng):
    """Uniform integer in [mean*(1-r), mean*(1+r)], at least 1."""
    lo = max(1, int(mean * (1.0 - range_ratio)))
    hi = max(lo, int(mean * (1.0 + range_ratio)))
    return rng.randint(lo, hi)


def synthetic_requests(count, input_len, output_len, input_range, output_range, rng):
    """Seeded random-word prompts; each request draws its own words, so
    requests share no prefix beyond chance."""
    requests = []
    for _ in range(count):
        words = sample_length(input_len, input_range, rng)
        prompt = " ".join(rng.choice(SYNTHETIC_WORDS) for _ in range(words))
        requests.append({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": sample_length(output_len, output_range, rng),
            "prompt_words": words,
        })
    return requests


def estimate_tokens(text):
    """Rough token estimate for a reference reply (no tokenizer available)."""
    return max(1, math.ceil(len(text) / 4))


def dataset_entry(item, default_output_len):
    """One request from a JSONL/ShareGPT record, or None when unusable.

    Accepted shapes: {"messages": [...], "max_tokens"?: n},
    {"prompt": "...", "output_len"?: n}, and ShareGPT
    {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", ...}]}
    whose first human turn is the prompt and whose reply sizes the output.
    """
    if not isinstance(item, dict):
        return None
    explicit = item.get("max_tokens", item.get("output_len"))
    if isinstance(item.get("messages"), list) and item["messages"]:
        messages = item["messages"]
        output_len = explicit or default_output_len
    elif isinstance(item.get("prompt"), str) and item["prompt"]:
        messages = [{"role": "user", "content": item["prompt"]}]
        output_len = explicit or default_output_len
    elif isinstance(item.get("conversations"), list):
        turns = item["conversations"]
        first = next(
            (index for index, turn in enumerate(turns)
             if isinstance(turn, dict) and turn.get("from") in ("human", "user")
             and isinstance(turn.get("value"), str) and turn["value"]),
            None,
        )
        if first is None:
            return None
        messages = [{"role": "user", "content": turns[first]["value"]}]
        reply = turns[first + 1] if first + 1 < len(turns) else None
        if explicit:
            output_len = explicit
        elif isinstance(reply, dict) and isinstance(reply.get("value"), str) and reply["value"]:
            output_len = estimate_tokens(reply["value"])
        else:
            output_len = default_output_len
    else:
        return None
    return {"messages": messages, "max_tokens": max(1, int(output_len))}


def load_dataset(path, count, default_output_len, rng):
    """`count` requests sampled (seeded shuffle, cycled) from a JSONL file or
    a ShareGPT-style JSON array."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if text.lstrip().startswith("["):
        items = json.loads(text)
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    entries = [entry for entry in (dataset_entry(item, default_output_len) for item in items) if entry]
    if not entries:
        raise ValueError(f"{path}: no usable prompts")
    rng.shuffle(entries)
    return [entries[index % len(entries)] for index in range(count)]


# ----------------------------------------------------------------------- SSE

class SseParser:
    """Incremental text/event-stream parser: feed bytes as they arrive and
    get back the `data` of every event the bytes complete. Lines may end in
    CRLF, LF or CR, and may be split anywhere across feeds."""

    def __init__(self, max_event_bytes=MAX_EVENT_BYTES):
        self._buffer = b""
        self._data = []
        self._event_bytes = 0
        self._max_event_bytes = max_event_bytes

    def feed(self, chunk):
        self._buffer += chunk
        events = []
        while True:
            lf = self._buffer.find(b"\n")
            cr = self._buffer.find(b"\r")
            if lf < 0 and cr < 0:
                break
            if cr >= 0 and (lf < 0 or cr < lf):
                if cr == len(self._buffer) - 1:
                    break  # a CR at the end may be the first half of CRLF
                end = cr
                skip = 2 if self._buffer[cr + 1:cr + 2] == b"\n" else 1
            else:
                end = lf
                skip = 1
            line = self._buffer[:end]
            self._buffer = self._buffer[end + skip:]
            event = self._line(line)
            if event is not None:
                events.append(event)
        if len(self._buffer) > self._max_event_bytes:
            raise ValueError("SSE line exceeds the event size limit")
        return events

    def _line(self, line):
        if not line:
            if not self._data:
                return None
            event = "\n".join(self._data)
            self._data = []
            self._event_bytes = 0
            return event
        if line.startswith(b":"):
            return None
        field, sep, value = line.partition(b":")
        if not sep:
            value = b""
        elif value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            self._event_bytes += len(value)
            if self._event_bytes > self._max_event_bytes:
                raise ValueError("SSE event exceeds the event size limit")
            self._data.append(value.decode("utf-8"))
        return None


class StreamTrace:
    """What one chat completion stream delivered, with arrival times."""

    def __init__(self):
        self.token_times = []
        self.usage = None
        self.finish_reason = None
        self.done = False
        self.error = None

    def on_event(self, data, at):
        if self.error is not None:
            return
        if data.strip() == "[DONE]":
            self.done = True
            return
        try:
            payload = json.loads(data)
        except ValueError:
            self.error = "bad_event_json"
            return
        if not isinstance(payload, dict):
            self.error = "bad_event_json"
            return
        error = payload.get("error")
        if error is not None:
            code = error.get("code") or error.get("type") if isinstance(error, dict) else None
            self.error = f"stream_error:{code or 'unknown'}"
            return
        if isinstance(payload.get("usage"), dict):
            self.usage = payload["usage"]
        produced = False
        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if any(isinstance(delta.get(key), str) and delta[key]
                   for key in ("content", "reasoning_content")) or delta.get("tool_calls"):
                produced = True
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if produced:
            self.token_times.append(at)


def request_record(index, request, trace, started, ended, epoch, error=None):
    """Per-request metrics from a finished stream."""
    error = error or trace.error
    if error is None and not trace.done:
        error = "truncated"
    usage = trace.usage or {}
    ttft_ms = (trace.token_times[0] - started) * 1000.0 if trace.token_times else None
    itl_ms = [
        (later - earlier) * 1000.0
        for earlier, later in zip(trace.token_times, trace.token_times[1:])
    ]
    if isinstance(usage.get("completion_tokens"), int):
        output_tokens = usage["completion_tokens"]
        output_source = "usage"
    else:
        output_tokens = len(trace.token_times)
        output_source = "deltas"
    e2e_ms = (ended - started) * 1000.0
    tpot_ms = None
    if ttft_ms is not None and output_tokens > 1:
        tpot_ms = (e2e_ms - ttft_ms) / (output_tokens - 1)
    return {
        "index": index,
        "ok": error is None,
        "error": error,
        "start_offset_s": started - epoch,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "itl_ms": itl_ms,
        "deltas": len(trace.token_times),
        "input_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
        "output_tokens": output_tokens,
        "output_tokens_source": output_source,
        "requested_output_tokens": request["max_tokens"],
        "finish_reason": trace.finish_reason,
    }


# ---------------------------------------------------------------------- HTTP

async def _body_chunks(reader, headers):
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size_line = await reader.readline()
            if not size_line:
                raise ConnectionError("connection closed inside a chunked body")
            size = int(size_line.split(b";", 1)[0].strip(), 16)
            if size == 0:
                while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                    pass
                return
            data = await reader.readexactly(size)
            await reader.readexactly(2)
            yield data
    elif "content-length" in headers:
        remaining = int(headers["content-length"])
        while remaining > 0:
            data = await reader.read(min(65536, remaining))
            if not data:
                raise ConnectionError("connection closed before Content-Length bytes")
            remaining -= len(data)
            yield data
    else:
        while True:
            data = await reader.read(65536)
            if not data:
                return
            yield data


async def post_stream(url, body, headers, on_chunk):
    """POST `body` over a fresh HTTP/1.1 connection and hand every body chunk
    of a 200 response to `on_chunk(bytes, perf_counter_time)` as it arrives.
    Returns (status, error_body)."""
    parsed = urllib.parse.urlsplit(url)
    secure = parsed.scheme == "https"
    port = parsed.port or (443 if secure else 80)
    reader, writer = await asyncio.open_connection(
        parsed.hostname, port,
        ssl=ssl.create_default_context() if secure else None,
        limit=MAX_EVENT_BYTES,
    )
    try:
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        lines = [
            f"POST {target} HTTP/1.1",
            f"Host: {parsed.hostname}:{port}",
            "Content-Type: application/json",
            "Accept: text/event-stream",
            f"Content-Length: {len(body)}",
            "Connection: close",
        ]
        lines += [f"{name}: {value}" for name, value in headers.items()]
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
        await writer.drain()
        status_line = await reader.readline()
        parts = status_line.split(None, 2)
        if len(parts) < 2:
            raise ConnectionError("malformed HTTP status line")
        status = int(parts[1])
        response_headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin-1").partition(":")
            response_headers[name.strip().lower()] = value.strip()
        error_body = b""
        async for chunk in _body_chunks(reader, response_headers):
            if status == 200:
                on_chunk(chunk, time.perf_counter())
            elif len(error_body) < MAX_ERROR_BODY_BYTES:
                error_body += chunk[:MAX_ERROR_BODY_BYTES - len(error_body)]
        return status, error_body
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionError):
            pass


class _NoLimit:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def request_body(model, request, index, temperature, seed):
    body = {
        "model": model,
        "messages": request["messages"],
        "max_tokens": request["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
    }
    if temperature > 0:
        body["seed"] = seed + index
    return json.dumps(body).encode("utf-8")


async def run_request(index, request, config, limiter, epoch):
    async with limiter:
        trace = StreamTrace()
        parser = SseParser()

        def on_chunk(chunk, at):
            for data in parser.feed(chunk):
                trace.on_event(data, at)

        body = request_body(config["model"], request, index, config["temperature"], config["seed"])
        started = time.perf_counter()
        error = None
        try:
            status, _ = await asyncio.wait_for(
                post_stream(config["url"], body, config["headers"], on_chunk),
                config["timeout"],
            )
            if status != 200:
                error = f"http_{status}"
        except asyncio.TimeoutError:
            error = "timeout"
        except (OSError, ConnectionError, asyncio.IncompleteReadError, ValueError) as failure:
            error = f"client:{type(failure).__name__}"
        return request_record(index, request, trace, started, time.perf_counter(), epoch, error)


async def run_point(requests, rate, burstiness, max_concurrency, config, seed):
    """Open-loop run: request i is sent at its arrival offset whether or not
    earlier requests have finished (a max concurrency queues it client-side)."""
    offsets = arrival_offsets(len(requests), rate, burstiness, random.Random(seed))
    limiter = asyncio.Semaphore(max_concurrency) if max_concurrency else _NoLimit()
    start = time.perf_counter()
    tasks = []
    for index, (request, offset) in enumerate(zip(requests, offsets)):
        delay = start + offset - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.ensure_future(run_request(index, request, config, limiter, start)))
    records = await asyncio.gather(*tasks)
    return list(records), time.perf_counter() - start


# ------------------------------------------------------------------- summary

ABORT_ERRORS = ("timeout", "truncated")


def meets_slo(record, slo):
    if slo.get("ttft_ms") is not None and (record["ttft_ms"] is None or record["ttft_ms"] > slo["ttft_ms"]):
        return False
    if slo.get("tpot_ms") is not None and record["tpot_ms"] is not None and record["tpot_ms"] > slo["tpot_ms"]:
        return False
    if slo.get("e2e_ms") is not None and record["e2e_ms"] > slo["e2e_ms"]:
        return False
    return True


def goodput(records, duration, slo):
    """Requests that completed within every given SLO, per second of the run.
    None when no SLO was given."""
    active = {key: value for key, value in slo.items() if value is not None}
    if not active:
        return None
    good = sum(1 for record in records if record["ok"] and meets_slo(record, active))
    return {
        "slo": active,
        "requests": good,
        "rps": good / duration if duration > 0 else 0.0,
        "fraction": good / len(records) if records else 0.0,
    }


def summarize(records, duration, slo):
    completed = [record for record in records if record["ok"]]
    errors = collections.Counter(record["error"] for record in records if not record["ok"])
    ttft = [record["ttft_ms"] for record in completed if record["ttft_ms"] is not None]
    tpot = [record["tpot_ms"] for record in completed if record["tpot_ms"] is not None]
    itl = [gap for record in completed for gap in record["itl_ms"]]
    e2e = [record["e2e_ms"] for record in completed]
    input_tokens = [record["input_tokens"] for record in completed if record["input_tokens"] is not None]
    output_tokens = [record["output_tokens"] for record in completed]
    total_input = sum(input_tokens)
    total_output = sum(output_tokens)
    deltas = sum(record["deltas"] for record in completed)
    per_second = (lambda value: value / duration) if duration > 0 else (lambda value: 0.0)
    warnings = []
    for name, values in (("ttft_ms", ttft), ("tpot_ms", tpot), ("itl_ms", itl), ("end_to_end_ms", e2e)):
        warnings += low_sample_warnings(name, values)
    if completed and len(input_tokens) < len(completed):
        warnings.append("server sent no usage.prompt_tokens for some requests; input token totals are partial")
    if any(record["output_tokens_source"] == "deltas" for record in completed):
        warnings.append("server sent no usage.completion_tokens for some requests; their output tokens are delta counts")
    return {
        "total": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "aborted": sum(count for error, count in errors.items() if error in ABORT_ERRORS),
        "errors": dict(sorted(errors.items())),
        "duration_s": duration,
        "throughput_rps": per_second(len(completed)),
        "output_throughput_tps": per_second(total_output),
        "input_throughput_tps": per_second(total_input),
        "total_token_throughput_tps": per_second(total_input + total_output),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "tokens_per_delta": total_output / deltas if deltas else None,
        "ttft_ms": distribution(ttft),
        "tpot_ms": distribution(tpot),
        "itl_ms": distribution(itl),
        "end_to_end_ms": distribution(e2e),
        "input_tokens": distribution(input_tokens),
        "output_tokens": distribution(output_tokens),
        "goodput": goodput(records, duration, slo),
        "warnings": warnings,
    }


# ---------------------------------------------------------------- Prometheus

def _parse_labels(text):
    labels = {}
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in ", ":
            index += 1
        if index >= len(text):
            break
        eq = text.index("=", index)
        name = text[index:eq].strip()
        if text[eq + 1] != '"':
            raise ValueError("label value must be quoted")
        value = []
        index = eq + 2
        while text[index] != '"':
            if text[index] == "\\":
                index += 1
                value.append({"n": "\n"}.get(text[index], text[index]))
            else:
                value.append(text[index])
            index += 1
        labels[name] = "".join(value)
        index += 1
    return labels


def parse_prometheus(text):
    """Samples of a Prometheus text exposition as
    {(name, ((label, value), ...)): float}; comments and junk are skipped."""
    samples = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name, rest = line.split("{", 1)
                label_text, rest = rest.rsplit("}", 1)
                labels = _parse_labels(label_text)
            else:
                name, rest = line.split(None, 1)
                labels = {}
            value = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        if not math.isfinite(value):
            continue
        samples[(name.strip(), tuple(sorted(labels.items())))] = value
    return samples


def sample_delta(before, after):
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


def histogram_quantile(q, buckets):
    """Prometheus `histogram_quantile` over cumulative [(le, count)] buckets."""
    buckets = sorted(buckets, key=lambda bucket: bucket[0])
    if not buckets or buckets[-1][1] <= 0:
        return None
    total = buckets[-1][1]
    rank = q * total
    previous_le, previous_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if math.isinf(le):
                return previous_le
            if count == previous_count:
                return le
            return previous_le + (le - previous_le) * (rank - previous_count) / (count - previous_count)
        previous_le, previous_count = le, count
    return previous_le


def histograms(delta):
    """Histogram families in a sample delta:
    {family: {labels: {"buckets": [(le, n)], "sum": s, "count": n}}}."""
    families = {}
    for (name, labels), value in delta.items():
        for suffix in ("_bucket", "_sum", "_count"):
            if name.endswith(suffix):
                family = name[: -len(suffix)]
                break
        else:
            continue
        rest = tuple(item for item in labels if item[0] != "le")
        series = families.setdefault(family, {}).setdefault(rest, {"buckets": [], "sum": None, "count": None})
        if suffix == "_bucket":
            le = dict(labels).get("le")
            if le is None:
                continue
            series["buckets"].append((float(le), value))
        else:
            series[suffix[1:]] = value
    return {
        family: {labels: series for labels, series in by_labels.items()
                 if series["buckets"] and series["count"] is not None}
        for family, by_labels in families.items()
    }


def _merge_series(series_list):
    merged = collections.defaultdict(float)
    total_sum = total_count = 0.0
    for series in series_list:
        for le, count in series["buckets"]:
            merged[le] += count
        total_sum += series["sum"] or 0.0
        total_count += series["count"] or 0.0
    return {"buckets": sorted(merged.items()), "sum": total_sum, "count": total_count}


def histogram_summary(series, scale=1.0):
    count = series["count"]
    if not count:
        return None
    return {
        "count": count,
        "mean": series["sum"] / count * scale,
        "p50": _scaled(histogram_quantile(0.50, series["buckets"]), scale),
        "p90": _scaled(histogram_quantile(0.90, series["buckets"]), scale),
        "p99": _scaled(histogram_quantile(0.99, series["buckets"]), scale),
    }


def _scaled(value, scale):
    return None if value is None else value * scale


def _counter_total(delta, name):
    values = [value for (sample, _), value in delta.items() if sample == name]
    return sum(values) if values else None


def _label_key(family, labels):
    if not labels:
        return family
    return family + "{" + ",".join(f'{key}="{value}"' for key, value in labels) + "}"


def server_metrics_report(before_text, after_text, client_output_tokens):
    """Histogram and counter deltas over one sweep point, plus the counted
    per-token and per-step metrics the serving plan tracks."""
    delta = sample_delta(parse_prometheus(before_text), parse_prometheus(after_text))
    families = histograms(delta)
    histogram_report = {}
    for family, by_labels in sorted(families.items()):
        for labels, series in sorted(by_labels.items()):
            summary = histogram_summary(series)
            if summary:
                histogram_report[_label_key(family, labels)] = summary
    counters = {
        _label_key(name, labels): value
        for (name, labels), value in sorted(delta.items())
        if name.endswith("_total") and value
    }

    counted = {}
    generated = _counter_total(delta, "izwi_request_generation_tokens_total")
    tokens = generated if generated else client_output_tokens
    counted["output_tokens"] = tokens
    counted["output_tokens_source"] = "server" if generated else "client"
    host_reads = _counter_total(delta, "izwi_kernel_host_read_ops_total")
    if host_reads is not None and tokens:
        counted["device_readbacks_per_token"] = host_reads / tokens
    steps = families.get("izwi_engine_step_host_seconds", {})
    if steps:
        phases = {dict(labels).get("phase", "all"): series for labels, series in steps.items()}
        counted["step_host_us_by_phase"] = {
            phase: histogram_summary(series, 1e6) for phase, series in sorted(phases.items())
        }
        prepare = phases.get("prepare")
        if prepare and prepare["count"]:
            total_us = sum(series["sum"] or 0.0 for series in phases.values()) * 1e6
            counted["engine_steps"] = prepare["count"]
            counted["host_us_per_step"] = total_us / prepare["count"]
            if tokens:
                counted["engine_steps_per_token"] = prepare["count"] / tokens
    for family, key in (("izwi_engine_batch_rows", "batch_rows"), ("izwi_engine_batch_tokens", "batch_tokens")):
        if families.get(family):
            counted[key] = histogram_summary(_merge_series(families[family].values()))
    for family, key in (
        ("izwi_engine_inter_token_latency_seconds", "server_itl_ms"),
        ("izwi_gateway_chat_inter_token_latency_seconds", "gateway_itl_ms"),
        ("izwi_gateway_chat_ttft_seconds", "gateway_ttft_ms"),
    ):
        if families.get(family):
            counted[key] = histogram_summary(_merge_series(families[family].values()), 1000.0)
    return {"counted": counted, "histograms": histogram_report, "counters": counters}


def fetch_text(url, headers, timeout=METADATA_TIMEOUT_SECS):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def scrape(urls, headers):
    texts = {}
    for url in urls:
        try:
            texts[url] = fetch_text(url, headers)
        except (OSError, urllib.error.URLError, ValueError) as error:
            texts[url] = error
    return texts


# ------------------------------------------------------------------ metadata

def _run(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                timeout=METADATA_TIMEOUT_SECS, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_info():
    root = os.path.dirname(os.path.abspath(__file__))
    sha = _run(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def host_info():
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_model": None,
        "memory_bytes": None,
        "gpus": [],
    }
    if info["os"] == "Darwin":
        info["cpu_model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_bytes"] = int(memory) if memory and memory.isdigit() else None
    elif info["os"] == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        info["memory_bytes"] = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    gpus = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    for line in (gpus or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            info["gpus"].append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return info


def server_info(base_url, headers):
    """Server identity and configuration: /v1/health on the local role,
    /v1/models on both roles. Missing endpoints are recorded as null."""
    info = {"health": None, "models": None}
    try:
        health = json.loads(fetch_text(base_url + "/v1/health", headers))
        runtime = health.get("runtime") or {}
        info["health"] = {
            "version": health.get("version"),
            "build_git_sha": runtime.get("build_git_sha"),
            "requested_backend": runtime.get("requested_backend"),
            "selected_backend": runtime.get("selected_backend"),
            "detected_device": runtime.get("detected_device"),
            "dtype_policy": runtime.get("dtype_policy"),
            "kv_cache_policy": runtime.get("kv_cache_policy"),
            "chat_concurrency_policy": runtime.get("chat_concurrency_policy"),
            "loaded_models": runtime.get("loaded_models"),
        }
    except (OSError, urllib.error.URLError, ValueError, AttributeError):
        pass
    try:
        models = json.loads(fetch_text(base_url + "/v1/models", headers))
        info["models"] = [entry.get("id") for entry in models.get("data", []) if isinstance(entry, dict)]
    except (OSError, urllib.error.URLError, ValueError, AttributeError):
        pass
    return info


# ----------------------------------------------------------- results/compare

def json_rate(rate):
    return "inf" if math.isinf(rate) else rate


def point_name(rate, max_concurrency):
    name = f"rate={json_rate(rate)}"
    return name + (f",concurrency={max_concurrency}" if max_concurrency else "")


def point_report(rate, max_concurrency, burstiness, records, duration, slo, server_metrics, save_samples):
    report = {
        "command": "serving",
        "config": {
            "request_rate": json_rate(rate),
            "max_concurrency": max_concurrency,
            "burstiness": burstiness,
        },
        "summary": summarize(records, duration, slo),
        "server_metrics": server_metrics,
    }
    if save_samples:
        report["samples"] = records
    return report


def build_result(metadata, config, points, started_at, ended_at):
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "command": "serving",
        "started_at": started_at,
        "ended_at": ended_at,
        "metadata": metadata,
        "config": config,
        "reports": [{"name": name, "report": report} for name, report in points],
    }


def _metric(summary, path):
    value = summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def compare_results(current, baseline, tolerance_percent):
    """Report-only comparison of matching sweep points. A metric regresses
    when it is worse than the baseline by more than the tolerance."""
    notes = []
    for path in (("metadata", "model"), ("metadata", "backend"), ("metadata", "host", "cpu_model"),
                 ("config", "profile"), ("config", "dataset"), ("config", "input_len"),
                 ("config", "output_len"), ("config", "seed")):
        values = []
        for result in (current, baseline):
            value = result
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(value)
        if values[0] != values[1]:
            notes.append(f"{'.'.join(path)} differs: current {values[0]!r}, baseline {values[1]!r}")
    baseline_points = {entry["name"]: entry["report"]["summary"] for entry in baseline.get("reports", [])}
    checks = []
    for entry in current.get("reports", []):
        base = baseline_points.get(entry["name"])
        if base is None:
            notes.append(f"point {entry['name']} has no baseline")
            continue
        for metric, path, lower_is_better in REGRESSION_METRICS:
            now = _metric(entry["report"]["summary"], path)
            then = _metric(base, path)
            if now is None or then is None:
                continue
            change = (now - then) * 100.0 / then if then else (0.0 if now == then else math.inf)
            worse = change > tolerance_percent if lower_is_better else change < -tolerance_percent
            checks.append({
                "point": entry["name"],
                "metric": metric,
                "baseline": then,
                "current": now,
                "change_percent": change if math.isfinite(change) else None,
                "lower_is_better": lower_is_better,
                "status": "regression" if worse else "ok",
            })
    return {
        "tolerance_percent": tolerance_percent,
        "regressions": sum(1 for check in checks if check["status"] == "regression"),
        "checks": checks,
        "notes": notes,
    }


# ------------------------------------------------------------------- output

def _fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_point(name, summary, server_metrics, out=None):
    out = out or sys.stdout
    width = 60
    print(f" Serving benchmark: {name} ".center(width, "="), file=out)
    rows = [
        ("Completed / total requests", f"{summary['completed']} / {summary['total']}"),
        ("Failed (aborted)", f"{summary['failed']} ({summary['aborted']})"),
        ("Duration (s)", _fmt(summary["duration_s"])),
        ("Request throughput (req/s)", _fmt(summary["throughput_rps"])),
        ("Output token throughput (tok/s)", _fmt(summary["output_throughput_tps"])),
        ("Total token throughput (tok/s)", _fmt(summary["total_token_throughput_tps"])),
        ("Input / output tokens", f"{summary['total_input_tokens']} / {summary['total_output_tokens']}"),
        ("Tokens per streamed delta", _fmt(summary["tokens_per_delta"])),
    ]
    if summary["goodput"]:
        rows.append(("Goodput (req/s)", _fmt(summary["goodput"]["rps"])))
    for label, value in rows:
        print(f"{label:<40}{value:>20}", file=out)
    for key, label in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT"), ("itl_ms", "ITL"), ("end_to_end_ms", "E2E")):
        dist = summary[key]
        if dist is None:
            continue
        print(f" {label} (ms), n={dist['count']} ".center(width, "-"), file=out)
        for stat in ("mean", "p50", "p90", "p95", "p99", "max"):
            print(f"{stat:<40}{_fmt(dist[stat]):>20}", file=out)
    counted = (server_metrics or {}).get("counted") or {}
    if counted:
        print(" Server counted metrics ".center(width, "-"), file=out)
        for key in ("device_readbacks_per_token", "host_us_per_step", "engine_steps_per_token"):
            if counted.get(key) is not None:
                print(f"{key:<40}{_fmt(counted[key]):>20}", file=out)
        for key in ("batch_rows", "batch_tokens"):
            if counted.get(key):
                print(f"{key + ' (mean)':<40}{_fmt(counted[key]['mean']):>20}", file=out)
    for warning in summary["warnings"]:
        print(f"warning: {warning}", file=out)
    if summary["errors"]:
        print(f"errors: {json.dumps(summary['errors'])}", file=out)
    print("=" * width, file=out)


def print_comparison(comparison, out=None):
    out = out or sys.stdout
    print(f" Comparison (tolerance {comparison['tolerance_percent']}%) ".center(60, "="), file=out)
    for note in comparison["notes"]:
        print(f"note: {note}", file=out)
    for check in comparison["checks"]:
        change = "n/a" if check["change_percent"] is None else f"{check['change_percent']:+.1f}%"
        print(f"{check['status']:<11}{check['point']:<26}{check['metric']:<24}"
              f"{_fmt(check['baseline']):>10} -> {_fmt(check['current']):<10}{change}", file=out)
    print(f"regressions: {comparison['regressions']} (report-only unless --fail-on-regression)", file=out)


# ---------------------------------------------------------------------- main

def resolve_options(args):
    """Explicit flags win, then the profile, then the defaults."""
    profile = PROFILES.get(args.profile, {}) if args.profile else {}
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, profile.get(key, default))
    return args


def raise_file_limit(wanted):
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= wanted:
        return
    target = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    for limit in (target, 10240):
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(limit, target), hard))
            return
        except (ValueError, OSError):
            continue


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080",
                        help="server or gateway base URL (default %(default)s)")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", help="model id sent in each request (required unless --current)")
    parser.add_argument("--api-key", default=os.environ.get("IZWI_BENCH_API_KEY"),
                        help="bearer key (default $IZWI_BENCH_API_KEY); the gateway and non-loopback servers need one")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="named preset; explicit flags override it")
    parser.add_argument("--dataset", help="'synthetic' (default) or a JSONL / ShareGPT JSON file")
    parser.add_argument("--num-prompts", type=int, help="requests per sweep point (default 200)")
    parser.add_argument("--input-len", type=int, help="synthetic prompt length in words (default 256)")
    parser.add_argument("--output-len", type=int, help="max_tokens per request (default 128)")
    parser.add_argument("--input-len-range", type=float,
                        help="synthetic prompt lengths are uniform in input_len*(1±r) (default 0)")
    parser.add_argument("--output-len-range", type=float, help="max_tokens uniform in output_len*(1±r) (default 0)")
    parser.add_argument("--request-rate",
                        help="comma-separated requests/s to sweep; 'inf' sends all at once (default inf)")
    parser.add_argument("--burstiness", type=float,
                        help="gamma shape of inter-arrival gaps; 1 = Poisson (default 1)")
    parser.add_argument("--max-concurrency",
                        help="comma-separated in-flight caps to sweep (crossed with the rates); "
                             "with rate inf this is a fixed-concurrency run")
    parser.add_argument("--temperature", type=float, help="sampling temperature (default 0, greedy)")
    parser.add_argument("--seed", type=int, help="dataset, arrival and sampling seed (default 0)")
    parser.add_argument("--warmup", type=int, help="unmeasured sequential requests before the sweep (default 0)")
    parser.add_argument("--request-timeout", type=float, default=600.0,
                        help="client timeout per request in seconds; a timeout counts as aborted")
    parser.add_argument("--slo-ttft-ms", type=float, help="goodput SLO on time to first token")
    parser.add_argument("--slo-tpot-ms", type=float, help="goodput SLO on time per output token")
    parser.add_argument("--slo-e2e-ms", type=float, help="goodput SLO on end-to-end latency")
    parser.add_argument("--metrics-url", action="append", default=[],
                        help="Prometheus endpoint scraped before/after each point (repeatable), e.g. "
                             "BASE/v1/metrics/prometheus (local) or BASE/internal/metrics (gateway)")
    parser.add_argument("--metrics-api-key", default=os.environ.get("IZWI_BENCH_METRICS_API_KEY"),
                        help="bearer key for --metrics-url (default $IZWI_BENCH_METRICS_API_KEY, else --api-key)")
    parser.add_argument("--backend", help="backend label when the server does not report one (gateway)")
    parser.add_argument("--metadata", default="{}", help="extra JSON object recorded under metadata.extra")
    parser.add_argument("--output", help="write the results JSON here")
    parser.add_argument("--save-samples", action="store_true", help="include per-request records in the results")
    parser.add_argument("--baseline", help="results JSON to compare against (report-only)")
    parser.add_argument("--current", help="compare this existing results JSON with --baseline instead of running")
    parser.add_argument("--tolerance-percent", type=float, default=10.0)
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit 3 when the comparison finds a regression")
    args = parser.parse_args(argv)
    if args.current:
        if not args.baseline:
            parser.error("--current needs --baseline")
        return args
    if not args.model:
        parser.error("--model is required")
    resolve_options(args)
    try:
        args.rates = [parse_rate(value) for value in str(args.request_rate).split(",") if value.strip()]
        args.concurrencies = (
            [int(value) for value in str(args.max_concurrency).split(",") if value.strip()]
            if args.max_concurrency else [None]
        )
        extra = json.loads(args.metadata)
    except ValueError as error:
        parser.error(str(error))
    if not isinstance(extra, dict):
        parser.error("--metadata must be a JSON object")
    args.extra_metadata = extra
    if not args.rates:
        parser.error("--request-rate needs at least one rate")
    if any(value is not None and value < 1 for value in args.concurrencies):
        parser.error("--max-concurrency values must be at least 1")
    if args.num_prompts < 1 or args.input_len < 1 or args.output_len < 1:
        parser.error("--num-prompts, --input-len and --output-len must be at least 1")
    if not 0 <= args.input_len_range < 1 or not 0 <= args.output_len_range < 1:
        parser.error("length ranges must be in [0, 1)")
    if args.burstiness <= 0:
        parser.error("--burstiness must be positive")
    return args


def build_requests(args, point_index, count=None):
    """The prompts of one sweep point. Every point (and the warmup, index
    -1) draws its own seeded prompts, so a later point does not hit the
    prefix cache an earlier point filled."""
    rng = random.Random(args.seed * 1000003 + point_index)
    count = args.num_prompts if count is None else count
    if args.dataset == "synthetic" or point_index < 0:
        return synthetic_requests(count, args.input_len, args.output_len,
                                  args.input_len_range, args.output_len_range, rng)
    return load_dataset(args.dataset, count, args.output_len, rng)


def run_config(args):
    return {
        "profile": args.profile,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "dataset": args.dataset,
        "num_prompts": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "input_len_range": args.input_len_range,
        "output_len_range": args.output_len_range,
        "request_rates": [json_rate(rate) for rate in args.rates],
        "max_concurrency": args.concurrencies,
        "burstiness": args.burstiness,
        "temperature": args.temperature,
        "seed": args.seed,
        "warmup": args.warmup,
        "request_timeout_s": args.request_timeout,
        "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms, "e2e_ms": args.slo_e2e_ms},
        "metrics_urls": args.metrics_url,
    }


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


async def benchmark(args):
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    metrics_key = args.metrics_api_key or args.api_key
    metrics_headers = {"Authorization": f"Bearer {metrics_key}"} if metrics_key else {}
    base_url = args.base_url.rstrip("/")
    server = server_info(base_url, headers)
    reported_backend = (server["health"] or {}).get("selected_backend")
    metadata = {
        "model": args.model,
        "backend": reported_backend or args.backend,
        "backend_source": "server_health" if reported_backend else ("argument" if args.backend else None),
        "git": git_info(),
        "host": host_info(),
        "server": server,
        "extra": args.extra_metadata,
    }
    if reported_backend and args.backend and args.backend != reported_backend:
        print(f"warning: --backend {args.backend} but the server reports {reported_backend}", file=sys.stderr)
    config = {
        "url": base_url + args.endpoint,
        "headers": headers,
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "timeout": args.request_timeout,
    }
    raise_file_limit(args.num_prompts + 64)
    started_at = utc_now()
    if args.warmup:
        warmup = await run_point(build_requests(args, -1, args.warmup), math.inf, 1.0, 1, config, args.seed)
        failed = [record["error"] for record in warmup[0] if not record["ok"]]
        if failed:
            print(f"warning: {len(failed)} warmup request(s) failed: {sorted(set(failed))}", file=sys.stderr)
    slo = {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms, "e2e_ms": args.slo_e2e_ms}
    points = []
    for rate in args.rates:
        for max_concurrency in args.concurrencies:
            name = point_name(rate, max_concurrency)
            requests = build_requests(args, len(points))
            before = scrape(args.metrics_url, metrics_headers)
            records, duration = await run_point(requests, rate, args.burstiness, max_concurrency, config, args.seed)
            after = scrape(args.metrics_url, metrics_headers)
            output_tokens = sum(record["output_tokens"] for record in records if record["ok"])
            server_metrics = {}
            for url in args.metrics_url:
                if isinstance(before[url], Exception) or isinstance(after[url], Exception):
                    failure = before[url] if isinstance(before[url], Exception) else after[url]
                    server_metrics[url] = {"error": f"{type(failure).__name__}: {failure}"}
                else:
                    server_metrics[url] = server_metrics_report(before[url], after[url], output_tokens)
            report = point_report(rate, max_concurrency, args.burstiness, records, duration, slo,
                                  server_metrics, args.save_samples)
            points.append((name, report))
            print_point(name, report["summary"], next(iter(server_metrics.values()), None))
    return build_result(metadata, run_config(args), points, started_at, utc_now())


def main(argv=None):
    args = parse_args(argv)
    if args.current:
        with open(args.current, encoding="utf-8") as handle:
            result = json.load(handle)
    else:
        result = asyncio.run(benchmark(args))
    comparison = None
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as handle:
            baseline = json.load(handle)
        comparison = compare_results(result, baseline, args.tolerance_percent)
        result["comparison"] = dict(comparison, baseline=args.baseline)
        print_comparison(comparison)
    if args.output and not args.current:
        directory = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(f"results: {args.output}")
    if comparison and comparison["regressions"] and args.fail_on_regression:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
