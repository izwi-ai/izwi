#!/usr/bin/env python3
"""End-to-end checks of Izwi's speech API with the stock openai-python SDK.

Runs against a live izwi-server (any backend). Start it with the aliases the
alias scenario uses:

    IZWI_OPENAI_MODEL_ALIASES=tts-1=Kokoro-82M,whisper-1=Granite-Speech-5.0-2B-GGUF \
      target/release/izwi-server --port 18080 --backend metal

then:

    python scripts/e2e/speech-api/run_sdk.py --url http://127.0.0.1:18080

Each scenario prints PASS or FAIL with its reason; the exit status is the
number of failures. Audio is checked with ffprobe/ffmpeg (an independent
decoder). Scenarios that need a model the server lacks fail with the
server's error, so run them with the models installed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
FOX_WAV = ROOT / "data" / "fox.wav"
FOX_TEXT = (ROOT / "data" / "fox.md").read_text().strip() if (ROOT / "data" / "fox.md").exists() else ""
TTS = "Kokoro-82M"
CLONE = "Qwen3-TTS-12Hz-0.6B-Base"
FRENCH = "Chatterbox-Multilingual-500M"
TRANSLATE = "Granite-Speech-5.0-2B-GGUF"
TEXT = "Izwi speaks every format now. This sentence checks it."

RESULTS: list[tuple[str, bool, str]] = []


def probe(data: bytes, suffix: str) -> dict:
    """ffprobe's view of `data`: codec, rate, duration."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = handle.name
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=False,
    )
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    Path(path).unlink(missing_ok=True)
    info = json.loads(out.stdout or "{}")
    stream = (info.get("streams") or [{}])[0]
    return {
        "codec": stream.get("codec_name"),
        "rate": int(stream.get("sample_rate") or 0),
        "channels": stream.get("channels"),
        "duration": float((info.get("format") or {}).get("duration") or 0.0),
        "decode_errors": decoded.stderr.strip(),
    }


def scenario(name: str):
    def wrap(func):
        def run(*args, **kwargs):
            started = time.time()
            try:
                detail = func(*args, **kwargs) or ""
                RESULTS.append((name, True, detail))
                print(f"PASS {name} ({time.time() - started:.1f}s) {detail}")
            except Exception as error:  # noqa: BLE001 - report every failure
                RESULTS.append((name, False, repr(error)))
                print(f"FAIL {name}: {error}")
                traceback.print_exc(limit=2)

        return run

    return wrap


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@scenario("S1 every response_format decodes with its codec and Content-Type")
def s1_formats(client: OpenAI) -> str:
    expected = {
        "mp3": ("mp3", "audio/mpeg", ".mp3"),
        "opus": ("opus", "audio/ogg", ".opus"),
        "flac": ("flac", "audio/flac", ".flac"),
        "wav": ("pcm_s16le", "audio/wav", ".wav"),
    }
    seen = []
    for fmt, (codec, content_type, suffix) in expected.items():
        response = client.audio.speech.with_raw_response.create(
            model=TTS, voice="af_heart", input=TEXT, response_format=fmt
        )
        data = response.content
        check(response.headers["content-type"].startswith(content_type), f"{fmt}: {response.headers['content-type']}")
        info = probe(data, suffix)
        check(info["codec"] == codec, f"{fmt}: codec {info['codec']}")
        check(info["duration"] > 1.0, f"{fmt}: duration {info['duration']}")
        check(not info["decode_errors"], f"{fmt}: {info['decode_errors']}")
        seen.append(f"{fmt}={info['duration']:.2f}s@{info['rate']}")
    pcm = client.audio.speech.create(model=TTS, voice="af_heart", input=TEXT, response_format="pcm")
    check(len(pcm.content) % 2 == 0, "pcm is 16-bit")
    seen.append(f"pcm={len(pcm.content) / 2 / 24000:.2f}s@24000")
    return ", ".join(seen)


@scenario("S2 stream_format=audio arrives progressively and decodes")
def s2_audio_stream(client: OpenAI) -> str:
    started = time.time()
    first = None
    data = bytearray()
    with client.audio.speech.with_streaming_response.create(
        model=TTS, voice="af_heart", input=TEXT + " " + TEXT, response_format="mp3", stream_format="audio"
    ) as response:
        check(response.headers.get("transfer-encoding") == "chunked", "chunked body")
        speech_id = response.headers.get("x-izwi-speech-id")
        for chunk in response.iter_bytes():
            if first is None and chunk:
                first = time.time() - started
            data.extend(chunk)
    total = time.time() - started
    info = probe(bytes(data), ".mp3")
    check(info["codec"] == "mp3" and info["duration"] > 2, f"decoded {info}")
    check(not info["decode_errors"], info["decode_errors"])
    check(speech_id and speech_id.startswith("speech_"), "speech id header")
    return f"first byte {first:.2f}s of {total:.2f}s, {info['duration']:.2f}s audio, id {speech_id}"


@scenario("S3 SSE deltas concatenate into one file; done carries usage")
def s3_sse(base_url: str) -> str:
    deltas = bytearray()
    done = None
    with httpx.stream(
        "POST", f"{base_url}/v1/audio/speech",
        json={"model": TTS, "voice": "af_heart", "input": TEXT, "response_format": "opus", "stream_format": "sse"},
        timeout=300,
    ) as response:
        check(response.status_code == 200, f"status {response.status_code}")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if event.get("type") == "speech.audio.delta":
                deltas.extend(base64.b64decode(event["audio"]))
            elif event.get("type") == "speech.audio.done":
                done = event
            elif event.get("type") == "error":
                raise AssertionError(event)
    check(done is not None, "speech.audio.done")
    usage = done["usage"]
    check({"input_tokens", "output_tokens", "total_tokens"} <= usage.keys(), f"usage {usage}")
    info = probe(bytes(deltas), ".opus")
    check(info["codec"] == "opus" and info["duration"] > 1, f"decoded {info}")
    check(not info["decode_errors"], info["decode_errors"])
    return f"{len(deltas)} bytes, {info['duration']:.2f}s, usage {usage}"


@scenario("S4 an operator alias (tts-1) speaks with the mapped model")
def s4_alias(client: OpenAI) -> str:
    audio = client.audio.speech.create(model="tts-1", voice="alloy", input="Alias check.", response_format="wav")
    info = probe(audio.content, ".wav")
    check(info["duration"] > 0.3, f"{info}")
    return f"{info['duration']:.2f}s"


@scenario("S5 sample_rate resamples; the lookup reports the outcome")
def s5_rate_and_lookup(client: OpenAI, base_url: str) -> str:
    response = client.audio.speech.with_raw_response.create(
        model=TTS, voice="af_heart", input=TEXT, response_format="flac", extra_body={"sample_rate": 16000}
    )
    info = probe(response.content, ".flac")
    check(info["rate"] == 16000, f"rate {info['rate']}")
    speech_id = response.headers["x-izwi-speech-id"]
    outcome = httpx.get(f"{base_url}/v1/audio/speech/{speech_id}", timeout=30).json()
    check(outcome["status"] == "completed", f"outcome {outcome}")
    check(outcome["sample_rate"] == 16000, f"outcome rate {outcome['sample_rate']}")
    return f"{info['duration']:.2f}s at 16 kHz, finish_reason {outcome['finish_reason']}"


@scenario("S6 translations: French speech to English text in every format")
def s6_translations(client: OpenAI) -> str:
    french = client.audio.speech.create(
        model=FRENCH,
        voice="default",
        input="Bonjour, je m'appelle Marie. Aujourd'hui il fait beau à Paris.",
        response_format="wav",
        extra_body={"language": "fr"},
    ).content
    texts = []
    for fmt in ["json", "text", "verbose_json", "srt", "vtt"]:
        result = client.audio.translations.create(
            model="whisper-1", file=("french.wav", french, "audio/wav"), response_format=fmt
        )
        text = result if isinstance(result, str) else result.text
        check(bool(text.strip()), f"{fmt}: empty")
        if fmt == "verbose_json":
            check(getattr(result, "task", None) == "translate", f"task {getattr(result, 'task', None)}")
        texts.append(text.strip().splitlines()[0][:60] if fmt == "json" else fmt)
    return " | ".join(texts)


@scenario("S7 consents and voices: create, use, revoke, delete")
def s7_voices(client: OpenAI) -> str:
    consent = client.audio.voice_consents.create(
        name="E2E speaker", recording=FOX_WAV.open("rb"), language="en-US"
    )
    check(consent.id.startswith("cons_"), consent.id)
    listed = client.audio.voice_consents.list(limit=5)
    check(any(item.id == consent.id for item in listed.data), "listed")
    renamed = client.audio.voice_consents.update(consent.id, name="E2E speaker 2")
    check(renamed.name == "E2E speaker 2", renamed.name)
    voice = client.audio.voices.create(
        name=f"e2e-{int(time.time())}",
        audio_sample=FOX_WAV.open("rb"),
        consent=consent.id,
        extra_body={"ref_text": FOX_TEXT},
    )
    check(voice.id and voice.type == "audio_sample", f"{voice}")
    spoken = client.audio.speech.create(
        model=CLONE, voice={"id": voice.id}, input="This is my cloned voice.", response_format="mp3"
    )
    info = probe(spoken.content, ".mp3")
    check(info["duration"] > 0.5, f"{info}")
    client.audio.voice_consents.delete(consent.id)
    try:
        client.audio.speech.create(model=CLONE, voice={"id": voice.id}, input="Revoked.", response_format="wav")
        raise AssertionError("a revoked voice still spoke")
    except Exception as error:  # the SDK raises PermissionDeniedError (403)
        check("403" in str(error) or "permission" in str(error).lower(), f"{error}")
    client.audio.voices.delete(voice.id)
    return f"voice {voice.id} spoke {info['duration']:.2f}s, revoked with its consent"


@scenario("S12 SGLang-Omni voice upload: multipart with free-text consent, speak by name, delete by name")
def s12_sglang_voice(base_url: str) -> str:
    # The request SGLang-Omni's docs send (attestation mode: no API key).
    name = f"narrator-{int(time.time())}"
    with FOX_WAV.open("rb") as sample:
        created = httpx.post(
            f"{base_url}/v1/audio/voices",
            files={"audio_sample": ("fox.wav", sample, "audio/wav")},
            data={"name": name, "consent": "consent-recording-id", "ref_text": FOX_TEXT},
            timeout=120,
        )
    check(created.status_code == 200, f"{created.status_code} {created.text}")
    check(created.json()["name"] == name, created.text)
    names = httpx.get(f"{base_url}/v1/audio/voices", params={"names_only": "true"}, timeout=30).json()
    check(name in names["uploaded_voice_names"], f"{names}")
    spoken = httpx.post(
        f"{base_url}/v1/audio/speech",
        json={"model": CLONE, "voice": name, "input": "Spoken by an uploaded name.", "response_format": "wav"},
        timeout=900,
    )
    check(spoken.status_code == 200, f"{spoken.status_code} {spoken.text[:200]}")
    info = probe(spoken.content, ".wav")
    deleted = httpx.delete(f"{base_url}/v1/audio/voices/{name}", timeout=30).json()
    check(deleted.get("success") is True, f"{deleted}")
    return f"{name} spoke {info['duration']:.2f}s, deleted by name"


@scenario("S8 batch: items in order, item errors in their item")
def s8_batch(base_url: str) -> str:
    body = {
        "model": TTS,
        "voice": "af_heart",
        "response_format": "mp3",
        "items": [{"input": "One."}, {"input": "Two.", "speed": 1.2}, {"input": "Three.", "response_format": "m4a"}],
    }
    result = httpx.post(f"{base_url}/v1/audio/speech/batch", json=body, timeout=600).json()
    check(result["total"] == 3 and result["succeeded"] == 2 and result["failed"] == 1, f"{result.get('total')}")
    for item in result["results"][:2]:
        info = probe(base64.b64decode(item["audio_data"]), ".mp3")
        check(info["codec"] == "mp3", f"item {item['index']}: {info}")
    check(result["results"][2]["error"]["param"] == "items.2.response_format", f"{result['results'][2]}")
    return f"{result['succeeded']} ok, 1 refused"


@scenario("S13 files: upload, retrieve, list, content and delete with client.files")
def s13_files(client: OpenAI) -> str:
    content = b'{"custom_id": "probe"}\n'
    uploaded = client.files.create(file=("probe.jsonl", content), purpose="batch")
    check(uploaded.id.startswith("file-"), f"id {uploaded.id}")
    check(uploaded.bytes == len(content) and uploaded.purpose == "batch", f"{uploaded}")
    check(client.files.retrieve(uploaded.id).filename == "probe.jsonl", "retrieve")
    listed = [item.id for item in client.files.list(purpose="batch")]
    check(uploaded.id in listed, f"{uploaded.id} not in the list")
    check(client.files.content(uploaded.id).content == content, "content")
    deleted = client.files.delete(uploaded.id)
    check(deleted.deleted and deleted.id == uploaded.id, f"{deleted}")
    try:
        client.files.retrieve(uploaded.id)
    except Exception as error:  # noqa: BLE001 - the SDK's NotFoundError
        check(getattr(error, "status_code", None) == 404, f"after delete: {error}")
    else:
        raise AssertionError("the deleted file is still there")
    try:
        client.files.create(file=("probe.jsonl", content), purpose="assistants")
    except Exception as error:  # noqa: BLE001 - the SDK's BadRequestError
        check(getattr(error, "status_code", None) == 400, f"purpose: {error}")
    else:
        raise AssertionError("purpose assistants was accepted")
    return f"{uploaded.id} round-tripped {len(content)} bytes"


def wait_for_batch(client: OpenAI, batch_id: str, timeout: float = 900.0):
    """Poll a batch until it ends."""
    deadline = time.time() + timeout
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        check(time.time() < deadline, f"batch {batch_id} is still {batch.status}")
        time.sleep(1)


def jsonl(lines: list[dict]) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


@scenario("S14 batch transcriptions: audio by file_id and inline, a refused line in the error file")
def s14_batch_transcriptions(client: OpenAI) -> str:
    spoken = client.audio.speech.create(model=TTS, voice="af_heart", input=TEXT, response_format="wav").content
    audio = client.files.create(file=("spoken.wav", spoken), purpose="user_data")
    endpoint = "/v1/audio/transcriptions"
    requests = [
        {"custom_id": "by-file", "method": "POST", "url": endpoint,
         "body": {"model": "whisper-1", "file_id": audio.id}},
        {"custom_id": "inline", "method": "POST", "url": endpoint,
         "body": {"model": "whisper-1", "audio_base64": base64.b64encode(spoken).decode(),
                  "response_format": "verbose_json"}},
        {"custom_id": "refused", "method": "POST", "url": endpoint,
         "body": {"model": "no-such-model", "file_id": audio.id}},
    ]
    source = client.files.create(file=("requests.jsonl", jsonl(requests)), purpose="batch")
    batch = client.batches.create(
        input_file_id=source.id, endpoint=endpoint, completion_window="24h", metadata={"suite": "speech-e2e"}
    )
    check(batch.status == "validating" and batch.id.startswith("batch_"), f"{batch}")
    check(batch.id in [item.id for item in client.batches.list(limit=20)], "the batch is not listed")
    batch = wait_for_batch(client, batch.id)
    check(batch.status == "completed", f"status {batch.status}: {batch.errors}")
    counts = batch.request_counts
    check((counts.total, counts.completed, counts.failed) == (3, 2, 1), f"{counts}")
    output = [json.loads(line) for line in client.files.content(batch.output_file_id).text.splitlines()]
    check([line["custom_id"] for line in output] == ["by-file", "inline"], f"{output}")
    for line in output:
        check(line["response"]["status_code"] == 200 and line["error"] is None, f"{line}")
        check("speaks" in line["response"]["body"]["text"].lower(), f"{line['response']['body']}")
    check(output[1]["response"]["body"].get("task") == "transcribe", "verbose_json body")
    errors = [json.loads(line) for line in client.files.content(batch.error_file_id).text.splitlines()]
    check([line["custom_id"] for line in errors] == ["refused"], f"{errors}")
    check(errors[0]["response"]["status_code"] == 400 and errors[0]["error"]["message"], f"{errors[0]}")
    check(client.files.retrieve(batch.output_file_id).purpose == "batch_output", "output purpose")
    return f"{batch.id}: 2 transcribed, 1 refused"


@scenario("S15 batch speech: each line's audio is its own file; a bad input file fails; cancel ends cancelled")
def s15_batch_speech(client: OpenAI) -> str:
    endpoint = "/v1/audio/speech"

    def speech_line(custom_id: str, text: str, **extra) -> dict:
        body = {"model": TTS, "voice": "af_heart", "input": text, "response_format": "mp3", **extra}
        return {"custom_id": custom_id, "method": "POST", "url": endpoint, "body": body}

    source = client.files.create(
        file=("speech.jsonl", jsonl([speech_line("first", "One."), speech_line("second", "Two.", speed=1.2)])),
        purpose="batch",
    )
    batch = client.batches.create(input_file_id=source.id, endpoint=endpoint, completion_window="24h")
    batch = wait_for_batch(client, batch.id)
    check(batch.status == "completed" and batch.error_file_id is None, f"{batch}")
    output = [json.loads(line) for line in client.files.content(batch.output_file_id).text.splitlines()]
    for line, custom_id in zip(output, ["first", "second"], strict=True):
        body = line["response"]["body"]
        check(line["custom_id"] == custom_id and body["format"] == "mp3", f"{line}")
        audio = client.files.content(body["audio_file_id"]).content
        check(len(audio) == body["bytes"], f"{custom_id}: {len(audio)} bytes, body says {body['bytes']}")
        info = probe(audio, ".mp3")
        check(info["codec"] == "mp3" and not info["decode_errors"], f"{custom_id}: {info}")
        check(client.files.retrieve(body["audio_file_id"]).filename == f"{custom_id}.mp3", "audio filename")

    # A line that breaks the format fails the whole batch at validation.
    broken = jsonl([speech_line("dup", "One."), speech_line("dup", "Two."), speech_line("live", "Three.", stream=True)])
    source = client.files.create(file=("broken.jsonl", broken), purpose="batch")
    failed = wait_for_batch(
        client, client.batches.create(input_file_id=source.id, endpoint=endpoint, completion_window="24h").id
    )
    check(failed.status == "failed", f"status {failed.status}")
    codes = [(error.code, error.line) for error in failed.errors.data]
    check(codes == [("duplicate_custom_id", 2), ("streaming_not_supported", 3)], f"{codes}")

    # Cancel: cancelling, then cancelled, with whatever finished kept.
    many = jsonl([speech_line(f"line-{index}", f"Sentence number {index}.") for index in range(64)])
    source = client.files.create(file=("many.jsonl", many), purpose="batch")
    batch = client.batches.create(input_file_id=source.id, endpoint=endpoint, completion_window="24h")
    cancelling = client.batches.cancel(batch.id)
    check(cancelling.status in ("cancelling", "cancelled"), f"status {cancelling.status}")
    cancelled = wait_for_batch(client, batch.id, timeout=120)
    check(cancelled.status == "cancelled", f"status {cancelled.status}")
    done = cancelled.request_counts.completed
    check(done < 64, "every line ran before the cancel")
    return f"2 spoken, 1 batch failed validation, 1 cancelled after {done} of 64 lines"


async def socket_session(base_url: str, dialect: str, cancel: bool = False) -> dict:
    import websockets

    url = base_url.replace("http", "ws", 1) + f"/v1/audio/speech/stream?dialect={dialect}"
    segments: dict[int, bytearray] = {}
    events = []
    speech_id = None
    closed_by_server = False
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        config = {"model": TTS, "voice": "af_heart", "response_format": "opus"}
        if dialect == "sglang-omni":
            await ws.send(json.dumps({"type": "session.config", "session": config}))
        else:
            await ws.send(json.dumps({"type": "session.config", **config}))
        configured = json.loads(await ws.recv())
        events.append(configured["type"])
        for piece in ["Dr. Smith arrived at 3.5 p.m. ", "He was late. ", "Then everyone ", "went home."]:
            await ws.send(json.dumps({"type": "input.text", "text": piece}))
        if cancel:
            await ws.send(json.dumps({"type": "input.cancel"}))
        else:
            await ws.send(json.dumps({"type": "input.done"}))
        current = None
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=120)
            except websockets.ConnectionClosed:
                closed_by_server = True
                break
            if isinstance(message, bytes):
                segments.setdefault(current, bytearray()).extend(message)
                continue
            event = json.loads(message)
            events.append(event["type"])
            if event["type"] == "audio.start":
                current = event["sentence_index"]
            if event["type"] == "error" and event.get("fatal"):
                raise AssertionError(event)
            if event["type"] == "session.done":
                speech_id = event.get("speech_id")
                if dialect != "sglang-omni":
                    break
        if dialect == "sglang-omni" and not closed_by_server:
            try:
                await asyncio.wait_for(ws.recv(), timeout=5)
            except websockets.ConnectionClosed:
                closed_by_server = True
    return {"segments": segments, "events": events, "closed": closed_by_server, "speech_id": speech_id}


@scenario("S9 text-in socket (izwi): segments decode, abbreviations hold")
def s9_socket(base_url: str) -> str:
    result = asyncio.run(socket_session(base_url, "izwi"))
    events = result["events"]
    check(events[0] == "session.configured", f"{events[:3]}")
    check("session.done" in events, f"{events}")
    check(events.count("audio.start") == events.count("audio.done"), f"{events}")
    check(len(result["segments"]) >= 2, f"{len(result['segments'])} segments")
    durations = []
    for index, data in sorted(result["segments"].items()):
        info = probe(bytes(data), ".opus")
        check(info["codec"] == "opus" and not info["decode_errors"], f"segment {index}: {info}")
        durations.append(round(info["duration"], 2))
    outcome = httpx.get(f"{base_url}/v1/audio/speech/{result['speech_id']}", timeout=30).json()
    check(outcome["status"] == "completed" and outcome["mode"] == "socket_utterance", f"outcome {outcome}")
    usage = outcome["usage"]
    check(usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"], f"usage {usage}")
    return f"{len(durations)} segments {durations}, {usage['output_tokens']} output tokens"


@scenario("S10 text-in socket (sglang-omni dialect) closes after session.done; cancel works")
def s10_socket_dialect(base_url: str) -> str:
    result = asyncio.run(socket_session(base_url, "sglang-omni"))
    check(result["closed"], "socket closed after session.done")
    cancelled = asyncio.run(socket_session(base_url, "vllm-omni", cancel=True))
    check("session.done" in cancelled["events"], f"{cancelled['events']}")
    return f"events {result['events'][-2:]}, cancel {cancelled['events'][-2:]}"


def node_check(base_url: str, node_dir: Path | None) -> None:
    @scenario("S11 openai-node: speech mp3 bytes and a JSON translation object")
    def run() -> str:
        if node_dir is None or not (node_dir / "node_modules" / "openai").exists():
            raise AssertionError("pass --node-dir with openai installed")
        script = f"""
const OpenAI = require('openai').default;
const fs = require('fs');
(async () => {{
  const client = new OpenAI({{ apiKey: 'unused', baseURL: '{base_url}/v1' }});
  const speech = await client.audio.speech.create({{ model: '{TTS}', voice: 'af_heart', input: 'Node check.', response_format: 'mp3' }});
  const bytes = Buffer.from(await speech.arrayBuffer());
  fs.writeFileSync(process.argv[2], bytes);
  const result = await client.audio.translations.create({{
    model: 'whisper-1', file: await OpenAI.toFile(fs.readFileSync(process.argv[2]), 'a.mp3'), response_format: 'json' }});
  console.log(JSON.stringify({{ bytes: bytes.length, translationType: typeof result, text: result.text }}));
}})().catch((e) => {{ console.error(e); process.exit(1); }});
"""
        with tempfile.TemporaryDirectory() as tmp:
            js = Path(tmp) / "check.cjs"
            js.write_text(script)
            # Node resolves `require` from the script's directory, not cwd.
            env = {**os.environ, "NODE_PATH": str(Path(node_dir).resolve() / "node_modules")}
            out = subprocess.run(
                ["node", str(js), str(Path(tmp) / "a.mp3")],
                cwd=node_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=900,
            )
            check(out.returncode == 0, out.stderr[-500:])
            result = json.loads(out.stdout.strip().splitlines()[-1])
            check(result["translationType"] == "object", f"{result}")
            info = probe((Path(tmp) / "a.mp3").read_bytes(), ".mp3")
            check(info["codec"] == "mp3", f"{info}")
        return f"{result['bytes']} bytes; translation object {result['text'][:40]!r}"

    run()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--node-dir", type=Path, default=None, help="a directory with openai-node installed")
    parser.add_argument("--only", nargs="*", help="scenario numbers to run, e.g. 1 3 9")
    args = parser.parse_args()
    if shutil.which("ffprobe") is None:
        print("ffprobe is required")
        return 1
    client = OpenAI(api_key="unused", base_url=f"{args.url}/v1", timeout=900)
    plan = {
        "1": lambda: s1_formats(client),
        "2": lambda: s2_audio_stream(client),
        "3": lambda: s3_sse(args.url),
        "4": lambda: s4_alias(client),
        "5": lambda: s5_rate_and_lookup(client, args.url),
        "6": lambda: s6_translations(client),
        # Right after S6, while its translation model is still loaded: the
        # speech route pins what it loads, and S7 adds a clone model.
        "11": lambda: node_check(args.url, args.node_dir),
        "7": lambda: s7_voices(client),
        "8": lambda: s8_batch(args.url),
        "9": lambda: s9_socket(args.url),
        "10": lambda: s10_socket_dialect(args.url),
        "12": lambda: s12_sglang_voice(args.url),
        "13": lambda: s13_files(client),
        "14": lambda: s14_batch_transcriptions(client),
        "15": lambda: s15_batch_speech(client),
    }
    for number, run in plan.items():
        if args.only and number not in args.only:
            continue
        run()
    failures = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - failures}/{len(RESULTS)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(main())
