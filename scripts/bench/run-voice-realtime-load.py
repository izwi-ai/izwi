#!/usr/bin/env python3
"""Multi-session load test for izwi's realtime voice and transcription sockets.

`voice` mode opens N concurrent `/v1/voice/realtime/ws` sessions. Each session
speaks a WAV utterance in real time (20 ms frames) followed by trailing
silence, lets the server's VAD end the turn, and records when the server ended
the utterance (`user_speech_end`), finished the transcript
(`user_transcript_final`), sent the first assistant audio frame and finished
the turn (`turn_done`). It repeats for --turns turns.

`transcription` mode opens N concurrent `/v1/speech-to-text/realtime/ws`
sessions, streams the WAV in real time and records the first partial
transcript after the first frame and the session end after `session_stop`.

Reported per mode, over every turn of every session:
- `time_to_first_audio_ms`: user_speech_end to the first assistant audio frame
  (the server's response latency; the VAD's trailing silence is excluded);
- `transcript_latency_ms`: user_speech_end to user_transcript_final;
- `turn_ms`: user_speech_end to turn_done;
- transcription mode: `first_partial_ms` and `final_ms`.

With --metrics-url the run also reports the server's own voice latency
histograms (`izwi_voice_*_seconds`) as the p50/p95 of the bucket deltas over the
run, so the client and server views can be compared.

Standard library only (a minimal RFC 6455 client): runs with plain python3.
"""
import argparse
import asyncio
import base64
import json
import math
import os
import struct
import sys
import time
import urllib.parse
import urllib.request
import wave

SCHEMA = "izwi.voice_realtime_load"
SCHEMA_VERSION = 1
FRAME_MS = 20
VOICE_MAGIC = b"IVWS"
TRANSCRIPTION_MAGIC = b"ITRW"
CLIENT_PCM16_KIND = 1
ASSISTANT_HEADER_BYTES = 24
ASSISTANT_AUDIO_KINDS = {2: "pcm_i16", 3: "g711_ulaw", 4: "g711_alaw"}
MAX_MESSAGE_BYTES = 16 << 20
VOICE_HISTOGRAMS = (
    "izwi_voice_turn_first_audio_seconds",
    "izwi_voice_asr_final_seconds",
    "izwi_voice_llm_first_token_seconds",
    "izwi_voice_tts_first_audio_seconds",
)

OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


# --- RFC 6455 framing -------------------------------------------------------

def encode_ws_frame(opcode, payload, mask_key):
    """One final client frame; client frames are always masked."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 1 << 16:
        header.append(0x80 | 126)
        header += struct.pack("!H", length)
    else:
        header.append(0x80 | 127)
        header += struct.pack("!Q", length)
    header += mask_key
    masked = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
    return bytes(header) + masked


def parse_ws_header(head):
    """Opcode, fin and payload-length field size from the first two bytes:
    returns (fin, opcode, masked, length_or_marker)."""
    fin = bool(head[0] & 0x80)
    opcode = head[0] & 0x0F
    masked = bool(head[1] & 0x80)
    return fin, opcode, masked, head[1] & 0x7F


class WebSocket:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    @classmethod
    async def connect(cls, url, timeout=30.0):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "ws":
            raise ValueError(f"only ws:// URLs are supported, got {url}")
        port = parsed.port or 80
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(parsed.hostname, port, limit=MAX_MESSAGE_BYTES), timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        writer.write((
            f"GET {path} HTTP/1.1\r\nHost: {parsed.hostname}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), timeout)
        if b" 101 " not in status:
            raise ConnectionError(f"websocket upgrade refused: {status.decode(errors='replace').strip()}")
        while (await reader.readline()) not in (b"\r\n", b""):
            pass
        return cls(reader, writer)

    async def send(self, opcode, payload):
        self.writer.write(encode_ws_frame(opcode, payload, os.urandom(4)))
        await self.writer.drain()

    async def send_json(self, value):
        await self.send(OP_TEXT, json.dumps(value).encode())

    async def recv(self):
        """The next text (str) or binary (bytes) message, or None on close."""
        parts, message_opcode = [], None
        while True:
            fin, opcode, masked, length = parse_ws_header(await self.reader.readexactly(2))
            if length == 126:
                (length,) = struct.unpack("!H", await self.reader.readexactly(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", await self.reader.readexactly(8))
            mask = await self.reader.readexactly(4) if masked else None
            payload = await self.reader.readexactly(length)
            if mask:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            if opcode == OP_PING:
                await self.send(OP_PONG, payload)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                return None
            if opcode != OP_CONT:
                message_opcode = opcode
            parts.append(payload)
            if fin:
                data = b"".join(parts)
                return data.decode() if message_opcode == OP_TEXT else data

    async def close(self):
        try:
            await self.send(OP_CLOSE, struct.pack("!H", 1000))
        except (ConnectionError, RuntimeError):
            pass
        self.writer.close()


# --- izwi framing -----------------------------------------------------------

def client_pcm16_frame(magic, frame_seq, sample_rate, pcm):
    return magic + bytes([1, CLIENT_PCM16_KIND, 0, 0]) + struct.pack("<II", sample_rate, frame_seq) + pcm


def parse_assistant_audio_frame(frame):
    """(format, is_final, utterance_seq, chunk_seq, sample_rate, payload) of a
    voice assistant audio frame, or None for another binary message."""
    if len(frame) < ASSISTANT_HEADER_BYTES or frame[:4] != VOICE_MAGIC:
        return None
    audio_format = ASSISTANT_AUDIO_KINDS.get(frame[5])
    if audio_format is None:
        return None
    (flags,) = struct.unpack_from("<H", frame, 6)
    utterance_seq, chunk_seq, sample_rate = struct.unpack_from("<QII", frame, 8)
    return audio_format, bool(flags & 1), utterance_seq, chunk_seq, sample_rate, frame[ASSISTANT_HEADER_BYTES:]


def read_wav(path):
    """Mono PCM16 bytes and sample rate of a WAV file (the first channel)."""
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        channels, rate = handle.getnchannels(), handle.getframerate()
        data = handle.readframes(handle.getnframes())
    if channels > 1:
        samples = struct.unpack(f"<{len(data) // 2}h", data)
        data = struct.pack(f"<{len(samples) // channels}h", *samples[::channels])
    return data, rate


def pcm_frames(pcm, sample_rate, frame_ms=FRAME_MS):
    step = sample_rate * frame_ms // 1000 * 2
    return [pcm[offset:offset + step] for offset in range(0, len(pcm), step)]


# --- statistics -------------------------------------------------------------

def percentile(values, pct):
    """Linear-interpolated percentile (numpy's default)."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * pct / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def parse_histograms(text, names):
    """{name: ([(le, cumulative_count)], count)} for the named Prometheus
    histograms, summed over label sets."""
    out = {name: ({}, 0.0) for name in names}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, _, value = line.rpartition(" ")
        for name in names:
            if metric.startswith(name + "_bucket"):
                le = metric.split('le="', 1)[1].split('"', 1)[0]
                bound = math.inf if le == "+Inf" else float(le)
                buckets, count = out[name]
                buckets[bound] = buckets.get(bound, 0.0) + float(value)
            elif metric.split("{", 1)[0] == name + "_count":
                buckets, count = out[name]
                out[name] = (buckets, count + float(value))
    return {name: (sorted(buckets.items()), count) for name, (buckets, count) in out.items()}


def histogram_delta(before, after):
    """Bucket deltas of one histogram between two scrapes."""
    previous = dict(before[0])
    return [(bound, cumulative - previous.get(bound, 0.0)) for bound, cumulative in after[0]], after[1] - before[1]


def histogram_quantile(buckets, count, quantile):
    """Prometheus histogram_quantile over cumulative buckets."""
    if count <= 0 or not buckets:
        return None
    rank = quantile * count
    lower_bound, lower_count = 0.0, 0.0
    for bound, cumulative in buckets:
        if cumulative >= rank:
            if math.isinf(bound):
                return lower_bound
            if cumulative == lower_count:
                return bound
            return lower_bound + (bound - lower_bound) * (rank - lower_count) / (cumulative - lower_count)
        lower_bound, lower_count = bound, cumulative
    return lower_bound


def scrape(metrics_url):
    with urllib.request.urlopen(metrics_url, timeout=10) as response:
        return parse_histograms(response.read().decode(), VOICE_HISTOGRAMS)


def server_view(before, after):
    view = {}
    for name in VOICE_HISTOGRAMS:
        buckets, count = histogram_delta(before[name], after[name])
        view[name] = {
            "count": count,
            "p50_ms": None if count <= 0 else histogram_quantile(buckets, count, 0.5) * 1000,
            "p95_ms": None if count <= 0 else histogram_quantile(buckets, count, 0.95) * 1000,
        }
    return view


# --- sessions ---------------------------------------------------------------

async def stream_audio(ws, magic, frames, sample_rate, first_seq, silence_frames):
    """Send `frames` then `silence_frames` of silence in real time; returns the
    next frame sequence."""
    seq = first_seq
    silence = bytes(len(frames[0])) if frames else b""
    started = time.monotonic()
    for index, payload in enumerate(list(frames) + [silence] * silence_frames):
        await ws.send(OP_BINARY, client_pcm16_frame(magic, seq, sample_rate, payload))
        seq += 1
        delay = started + (index + 1) * FRAME_MS / 1000 - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    return seq


async def wait_for(ws, wanted, deadline, events):
    """Receive until a text event of type `wanted`, logging (time, event)."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"no {wanted} before the deadline")
        message = await asyncio.wait_for(ws.recv(), remaining)
        if message is None:
            raise ConnectionError(f"socket closed while waiting for {wanted}")
        now = time.monotonic()
        if isinstance(message, str):
            event = json.loads(message)
            events.append((now, event))
            if event.get("type") == "error" and event.get("fatal"):
                raise RuntimeError(event.get("message"))
            if event.get("type") == wanted:
                return event
        else:
            events.append((now, message))


def turn_record(events, speech_end_type="user_speech_end"):
    """Latencies of one voice turn from its (time, event) log."""
    marks = {}
    for now, event in events:
        if isinstance(event, dict):
            marks.setdefault(event.get("type"), now)
        elif parse_assistant_audio_frame(event) is not None:
            marks.setdefault("first_audio", now)
    start = marks.get(speech_end_type)
    done = [event for _, event in events if isinstance(event, dict) and event.get("type") == "turn_done"]

    def since(mark):
        return None if start is None or mark not in marks else (marks[mark] - start) * 1000

    errors = [event.get("message") for _, event in events
              if isinstance(event, dict) and event.get("type") == "error"]
    record = {
        "status": done[-1].get("status") if done else "missing",
        "time_to_first_audio_ms": since("first_audio"),
        "transcript_latency_ms": since("user_transcript_final"),
        "turn_ms": since("turn_done"),
    }
    if errors:
        record["errors"] = errors
    return record


async def run_voice_session(index, args, frames, sample_rate):
    ws = await WebSocket.connect(args.base_url.replace("http", "ws", 1) + "/v1/voice/realtime/ws")
    records, deadline = [], time.monotonic() + args.timeout
    try:
        events = []
        session_start = {"type": "session_start"}
        if args.system_prompt:
            session_start["system_prompt"] = args.system_prompt
        await ws.send_json(session_start)
        await wait_for(ws, "session_ready", deadline, events)
        start = {
            "type": "input_stream_start",
            "asr_model_id": args.asr_model,
            "text_model_id": args.text_model,
            "tts_model_id": args.tts_model,
            "input_sample_rate": sample_rate,
            "silence_duration_ms": args.silence_ms,
        }
        if args.speaker:
            start["speaker"] = args.speaker
        if args.max_output_tokens:
            start["max_output_tokens"] = args.max_output_tokens
        if args.output_audio_format:
            start["output_audio_format"] = args.output_audio_format
        await ws.send_json(start)
        await wait_for(ws, "input_stream_ready", deadline, events)
        seq = 1
        silence_frames = (args.silence_ms + args.trailing_silence_ms) // FRAME_MS
        for _ in range(args.turns):
            events = []
            sender = asyncio.ensure_future(
                stream_audio(ws, VOICE_MAGIC, frames, sample_rate, seq, silence_frames))
            try:
                await wait_for(ws, "turn_done", deadline, events)
            finally:
                seq = await sender
            records.append(turn_record(events))
    except Exception as error:  # noqa: BLE001 - one failed session is a result
        records.append({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    finally:
        await ws.close()
    return [{"session": index, "turn": turn, **record} for turn, record in enumerate(records)]


async def run_transcription_session(index, args, frames, sample_rate):
    ws = await WebSocket.connect(
        args.base_url.replace("http", "ws", 1) + "/v1/speech-to-text/realtime/ws")
    deadline = time.monotonic() + args.timeout
    try:
        events = []
        await ws.send_json({"type": "session_start", "model_id": args.asr_model})
        await wait_for(ws, "session_ready", deadline, events)
        first_frame = time.monotonic()
        receiver = asyncio.ensure_future(wait_for(ws, "session_done", deadline, events))
        await stream_audio(ws, TRANSCRIPTION_MAGIC, frames, sample_rate, 1, 0)
        stopped = time.monotonic()
        await ws.send_json({"type": "session_stop"})
        await receiver
        partials = [now for now, event in events
                    if isinstance(event, dict) and event.get("type") == "transcript_partial"]
        done = events[-1][0]
        record = {
            "status": "ok",
            "first_partial_ms": (partials[0] - first_frame) * 1000 if partials else None,
            "final_ms": (done - stopped) * 1000,
        }
    except Exception as error:  # noqa: BLE001
        record = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    finally:
        await ws.close()
    return [{"session": index, "turn": 0, **record}]


def build_report(args, records, server):
    fields = (("time_to_first_audio_ms", "transcript_latency_ms", "turn_ms")
              if args.mode == "voice" else ("first_partial_ms", "final_ms"))
    statuses = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "sessions": args.sessions,
        "turns": args.turns if args.mode == "voice" else 1,
        "models": {"asr": args.asr_model, "text": args.text_model, "tts": args.tts_model},
        "statuses": statuses,
        "summary": {field: summarize([record.get(field) for record in records]) for field in fields},
        "server": server,
        "records": records,
    }


async def run(args):
    pcm, sample_rate = read_wav(args.audio)
    frames = pcm_frames(pcm, sample_rate)
    before = scrape(args.metrics_url) if args.metrics_url else None
    session = run_voice_session if args.mode == "voice" else run_transcription_session
    results = await asyncio.gather(*(session(index, args, frames, sample_rate)
                                     for index in range(args.sessions)))
    server = server_view(before, scrape(args.metrics_url)) if before else None
    return build_report(args, [record for result in results for record in result], server)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--mode", choices=("voice", "transcription"), default="voice")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--audio", default="data/fox.wav")
    parser.add_argument("--asr-model", default="Parakeet-TDT-0.6B-v3")
    parser.add_argument("--text-model", default="LFM2.5-1.2B-Instruct-GGUF")
    parser.add_argument("--tts-model", default="Kokoro-82M")
    parser.add_argument("--speaker")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--system-prompt",
                        help="e.g. one that asks for several sentences, so the reply streams")
    parser.add_argument("--output-audio-format")
    parser.add_argument("--silence-ms", type=int, default=600,
                        help="VAD silence that ends an utterance")
    parser.add_argument("--trailing-silence-ms", type=int, default=400,
                        help="extra silence sent past the VAD window")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--metrics-url", help="e.g. http://127.0.0.1:8080/v1/metrics/prometheus")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    if args.sessions < 1 or args.turns < 1:
        parser.error("--sessions and --turns must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = asyncio.run(run(args))
    text = json.dumps(report, indent=2)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            handle.write(text + "\n")
    print(json.dumps({key: report[key] for key in ("mode", "sessions", "statuses", "summary", "server")}, indent=2))
    return 0 if set(report["statuses"]) <= {"ok", "no_input"} else 1


if __name__ == "__main__":
    sys.exit(main())
