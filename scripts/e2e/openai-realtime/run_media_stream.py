#!/usr/bin/env python3
"""Play a WAV into izwi's `/v1/realtime/media-stream` as a phone call.

Acts as the carrier side of a telephony media stream: `connected`, `start`,
the WAV as 20 ms frames of base64 8 kHz μ-law sent in real time, silence while
the call is answered, then `stop`. Prints the `transcript` frames as they
arrive and saves the speech the call returns (`media` frames) as a WAV.

Usage (the server must be running; models load on first use):
  python3 run_media_stream.py --intent transcription \\
      --transcription-model Voxtral-Mini-4B-Realtime-2602
  python3 run_media_stream.py --model izwi-voice --voice cedar --out reply.wav
  python3 run_media_stream.py --custom-parameters   # parameters in `start`, as
                                                    # carriers that drop the query string send them

Requires `websockets` (installed with `openai[realtime]`).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import time
import urllib.parse
import wave
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[3]
RATE = 8_000
FRAME_SAMPLES = 160  # 20 ms
STREAM_SID = "MZizwi0e2e"
CALL_SID = "CAizwi0e2e"


def linear_to_ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(sample), 32_635) + 0x84
    exponent = magnitude.bit_length() - 8
    return ~(sign | (exponent << 4) | ((magnitude >> (exponent + 3)) & 0x0F)) & 0xFF


def ulaw_to_linear(code: int) -> int:
    code = ~code & 0xFF
    magnitude = ((((code & 0x0F) << 3) + 0x84) << ((code >> 4) & 0x07)) - 0x84
    return -magnitude if code & 0x80 else magnitude


def read_wav_8k(path: Path) -> list[int]:
    """The WAV as mono 8 kHz samples; each one averages the input it spans."""
    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2, "PCM16 WAV"
        channels, rate = handle.getnchannels(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    mono = [sum(samples[i:i + channels]) // channels for i in range(0, len(samples), channels)]
    if rate == RATE:
        return mono
    step = rate / RATE
    out = []
    for index in range(int(len(mono) / step)):
        window = mono[int(index * step):max(int((index + 1) * step), int(index * step) + 1)]
        out.append(sum(window) // len(window))
    return out


def call_url(args: argparse.Namespace, params: dict[str, str]) -> str:
    base = args.url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    query = "" if args.custom_parameters else urllib.parse.urlencode(params)
    return f"{base}/v1/realtime/media-stream" + (f"?{query}" if query else "")


async def run(args: argparse.Namespace) -> int:
    params = {
        key: value
        for key, value in {
            "model": args.model,
            "intent": args.intent,
            "voice": args.voice,
            "instructions": args.instructions,
            "language": args.language,
            "transcription_model": args.transcription_model,
        }.items()
        if value
    }
    samples = read_wav_8k(Path(args.wav))
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    reply = bytearray()
    counts: dict[str, int] = {}
    completed: list[str] = []
    last_frame = time.monotonic()

    async with websockets.connect(call_url(args, params), additional_headers=headers) as call:
        async def send(frame: dict) -> None:
            await call.send(json.dumps(frame))

        async def receive() -> None:
            nonlocal last_frame
            async for raw in call:
                frame = json.loads(raw)
                event = frame.get("event")
                counts[event] = counts.get(event, 0) + 1
                last_frame = time.monotonic()
                if event == "media":
                    reply.extend(base64.b64decode(frame["media"]["payload"]))
                elif event == "mark":
                    # A carrier echoes a mark once the audio before it has played.
                    await send({"event": "mark", "streamSid": STREAM_SID, "mark": frame["mark"]})
                    print(f"[mark] {frame['mark']['name']}")
                elif event == "clear":
                    print("[clear]")
                elif event == "transcript" and frame["type"] == "completed":
                    completed.append(frame["text"])
                    print(f"[transcript] {frame['text']}")
                elif event == "transcript":
                    print(f"[partial] {frame['text']}")
                elif event == "error":
                    print(f"[error] {json.dumps(frame.get('error'))}")

        receiver = asyncio.create_task(receive())
        await send({"event": "connected", "protocol": "Call", "version": "1.0.0"})
        await send({
            "event": "start", "sequenceNumber": "1", "streamSid": STREAM_SID,
            "start": {
                "streamSid": STREAM_SID, "callSid": CALL_SID, "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": RATE, "channels": 1},
                "customParameters": params if args.custom_parameters else {},
            },
        })

        silence = bytes([0xFF]) * FRAME_SAMPLES
        started = time.monotonic()
        audio_frames = -(-len(samples) // FRAME_SAMPLES)
        index = 0
        while not receiver.done():
            if index < audio_frames:
                chunk = samples[index * FRAME_SAMPLES:(index + 1) * FRAME_SAMPLES]
                payload = bytes(linear_to_ulaw(sample) for sample in chunk)
                payload += silence[len(payload):]
            else:
                # The call stays open in silence until it has been quiet for
                # `--quiet` seconds after something came back, or times out.
                waited = time.monotonic() - started - audio_frames * 0.02
                answered = bool(counts) and time.monotonic() - last_frame >= args.quiet
                if answered or waited >= args.timeout:
                    break
                payload = silence
            await send({
                "event": "media", "streamSid": STREAM_SID,
                "media": {"track": "inbound", "chunk": str(index + 1),
                          "timestamp": str(index * 20),
                          "payload": base64.b64encode(payload).decode()},
            })
            index += 1
            # Real time: frame `index` is due 20 ms after the one before it.
            await asyncio.sleep(max(0.0, started + index * 0.02 - time.monotonic()))
        if not receiver.done():
            await send({"event": "stop", "streamSid": STREAM_SID, "stop": {"callSid": CALL_SID}})
        try:
            await asyncio.wait_for(receiver, timeout=5)
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            receiver.cancel()

    if reply and args.out:
        with wave.open(args.out, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(RATE)
            handle.writeframes(struct.pack(f"<{len(reply)}h", *map(ulaw_to_linear, reply)))
        print(f"saved {len(reply) / RATE:.2f} s of returned speech to {args.out}")
    print("frames received: " + (json.dumps(counts, sort_keys=True) if counts else "none"))
    answered = bool(completed) if args.intent == "transcription" else bool(reply)
    if not answered or counts.get("error"):
        print("FAIL: the call returned " + ("an error" if counts.get("error") else "nothing"))
        return 1
    print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="ws://127.0.0.1:18080", help="the server's base URL")
    parser.add_argument("--api-key", default=os.environ.get("IZWI_API_KEY"),
                        help="the server's API key or a client secret (default: $IZWI_API_KEY)")
    parser.add_argument("--wav", default=str(REPO / "data" / "fox.wav"), help="the caller's speech")
    parser.add_argument("--model", help="realtime profile, as on /v1/realtime")
    parser.add_argument("--intent", choices=["transcription"],
                        help="`transcription` for a call that is only transcribed")
    parser.add_argument("--transcription-model")
    parser.add_argument("--voice")
    parser.add_argument("--instructions")
    parser.add_argument("--language")
    parser.add_argument("--custom-parameters", action="store_true",
                        help="send the parameters in the start frame, not the query string")
    parser.add_argument("--out", default="media-stream-reply.wav",
                        help="where the returned speech is saved")
    parser.add_argument("--quiet", type=float, default=3.0,
                        help="seconds without a frame that end an answered call")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="seconds to wait after the WAV for an answer")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
