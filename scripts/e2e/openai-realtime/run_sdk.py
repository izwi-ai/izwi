#!/usr/bin/env python3
"""End-to-end check of izwi's `/v1/realtime` with the stock openai-python client.

Runs scenarios against a running izwi-server, using `client.realtime.connect()`
unchanged (only `base_url` points at izwi), and validates every server event
against openai-python's own GA `RealtimeServerEvent` union, so an event the
SDK cannot parse fails the run.

Scenarios:
  s1  server-VAD spoken turn: speech in, transcript, spoken answer
  s2  text item + response.create with output_modalities ["text"]
  s3  function call round trip: function_call item, function_call_output, answer
  s4  barge-in: speech during the answer cancels it (turn_detected), then truncate
  s5  manual turns: turn_detection null, commit, response.create
  s6  out-of-band response: conversation "none", input items, metadata
  s7  G.711 μ-law in and out
  s8  history replay as LiveKit sends it after a reconnect

Usage (the server must be running; models load on first use):
  python3 run_sdk.py --url http://127.0.0.1:18080 --model izwi-voice
  python3 run_sdk.py --model izwi-voice-compact --scenarios s1,s4
  python3 run_sdk.py --record traffic.jsonl   # keep every event for replay

Requires `openai[realtime]` (tested with 3.46.0).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import struct
import sys
import time
import wave
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.realtime import RealtimeServerEvent
from pydantic import TypeAdapter

REPO = Path(__file__).resolve().parents[3]
SERVER_EVENT = TypeAdapter(RealtimeServerEvent)


def read_wav_24k(path: Path) -> list[int]:
    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2 and handle.getnchannels() == 1, "mono PCM16 WAV"
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))
    if rate == 24_000:
        return samples
    out = []
    step = rate / 24_000
    for index in range(int(len(samples) / step)):
        position = index * step
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        frac = position - left
        out.append(int(samples[left] * (1 - frac) + samples[right] * frac))
    return out


def pcm16(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def ulaw(sample: int) -> int:
    bias, clip = 0x84, 32635
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(sample), clip) + bias
    exponent = max(magnitude.bit_length() - 8, 0)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def to_ulaw_8k(samples_24k: list[int]) -> bytes:
    return bytes(ulaw(sample) for sample in samples_24k[::3])


class Session:
    """One realtime connection: sends events, receives and validates every
    server event, and records the traffic."""

    def __init__(self, connection, record: list | None):
        self.connection = connection
        self.events: list[dict] = []
        self.record = record
        self.started = time.monotonic()

    def _log(self, direction: str, event: dict) -> None:
        if self.record is not None:
            logged = dict(event)
            for key in ("audio", "delta"):
                if key in logged and isinstance(logged[key], str) and len(logged[key]) > 64:
                    logged[key] = f"<{len(logged[key])} base64 chars>"
            self.record.append({"t_ms": int((time.monotonic() - self.started) * 1000), "dir": direction, "event": logged})

    async def send(self, event: dict) -> None:
        self._log("client", event)
        await self.connection.send(event)

    async def recv(self, timeout: float) -> dict:
        raw = await asyncio.wait_for(self.connection.recv_bytes(), timeout)
        event = json.loads(raw)
        SERVER_EVENT.validate_python(event)
        self._log("server", event)
        self.events.append(event)
        return event

    async def until(self, event_type: str, timeout: float = 120.0) -> list[dict]:
        """Events up to and including the next `event_type`."""
        deadline = time.monotonic() + timeout
        seen = []
        while True:
            event = await self.recv(max(deadline - time.monotonic(), 0.01))
            seen.append(event)
            if event["type"] == "error" and event_type != "error":
                raise AssertionError(f"server error: {event['error']}")
            if event["type"] == event_type:
                return seen

    async def append(self, samples: list[int], chunk_ms: int = 100) -> None:
        step = 24 * chunk_ms
        for start in range(0, len(samples), step):
            await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16(samples[start:start + step])).decode()})


def types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audio_ms(events: list[dict], bytes_per_ms: int = 48) -> int:
    total = sum(len(base64.b64decode(event["delta"])) for event in events if event["type"] == "response.output_audio.delta")
    return total // bytes_per_ms


def check_response_order(events: list[dict]) -> None:
    """The order LiveKit relies on: output_item.added before its deltas,
    conversation.item.added right after it, done events before response.done."""
    names = types(events)
    require("response.created" in names, "response.created")
    require(names[-1] == "response.done", f"response.done last, got {names[-1]}")
    if "response.output_audio.delta" in names or "response.output_audio_transcript.delta" in names:
        first_delta = min(names.index(name) for name in ("response.output_audio.delta", "response.output_audio_transcript.delta") if name in names)
        require(names.index("response.output_item.added") < first_delta, "output_item.added before deltas")
        require(names.index("response.content_part.added") < first_delta, "content_part.added before deltas")


async def s1_vad_turn(session: Session, speech: list[int]) -> dict:
    await session.send({"type": "session.update", "session": {"type": "realtime", "instructions": "You are a helpful voice assistant. Answer in one short sentence.", "audio": {"input": {"transcription": {"model": "whisper-1"}}}}})
    await session.until("session.updated")
    await session.append([0] * 7_200 + speech + [0] * 36_000)
    events = await session.until("response.done", 180)
    names = types(events)
    for name in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed", "conversation.item.added", "response.created"):
        require(name in names, f"{name} missing: {names}")
    require(names.index("input_audio_buffer.speech_started") < names.index("input_audio_buffer.speech_stopped") < names.index("input_audio_buffer.committed"), "VAD event order")
    check_response_order(events[names.index("response.created"):])
    done = events[-1]["response"]
    require(done["status"] == "completed", f"status {done['status']}: {done.get('status_details')}")
    spoken = audio_ms(events)
    require(spoken > 300, f"spoke only {spoken} ms")
    transcript = next((event["transcript"] for event in session.events if event["type"] == "conversation.item.input_audio_transcription.completed"), None)
    if transcript is None:
        transcript = (await session.until("conversation.item.input_audio_transcription.completed", 60))[-1]["transcript"]
    require("fox" in transcript.lower(), f"transcript {transcript!r}")
    answer = next(event["transcript"] for event in events if event["type"] == "response.output_audio_transcript.done")
    return {"transcript": transcript, "answer": answer, "audio_ms": spoken, "usage": done["usage"]}


async def s2_text_response(session: Session, _speech) -> dict:
    await session.send({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Reply with exactly the word: ready"}]}})
    await session.until("conversation.item.done")
    await session.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
    events = await session.until("response.done", 120)
    check_response_order(events)
    require(events[-1]["response"]["status"] == "completed", "completed")
    text = next(event["text"] for event in events if event["type"] == "response.output_text.done")
    require(text.strip() != "", "text answer")
    require("response.output_audio.delta" not in types(events), "no audio in a text response")
    return {"text": text}


WEATHER_TOOL = {"type": "function", "name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


async def s3_function_call(session: Session, _speech) -> dict:
    await session.send({"type": "session.update", "session": {"type": "realtime", "instructions": "You are a voice assistant. Use tools when asked about the weather. Answer briefly.", "tools": [WEATHER_TOOL], "tool_choice": "auto"}})
    await session.until("session.updated")
    await session.send({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is the weather in Harare right now?"}]}})
    await session.until("conversation.item.done")
    await session.send({"type": "response.create"})
    events = await session.until("response.done", 120)
    calls = [event["item"] for event in events if event["type"] == "response.output_item.done" and event["item"]["type"] == "function_call"]
    require(calls, f"a function call: {types(events)}")
    call = calls[0]
    require(call["name"] == "get_weather", call["name"])
    require("harare" in call["arguments"].lower(), call["arguments"])
    names = types(events)
    require("response.function_call_arguments.done" in names, "arguments.done (Pipecat)")
    added = [event for event in events if event["type"] == "conversation.item.added" and event["item"]["type"] == "function_call"]
    require(added and added[0]["item"]["call_id"] == call["call_id"], "conversation.item.added carries call_id (Pipecat)")
    await session.send({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps({"temp_c": 24, "sky": "sunny"})}})
    await session.until("conversation.item.done")
    await session.send({"type": "response.create"})
    events = await session.until("response.done", 180)
    require(events[-1]["response"]["status"] == "completed", "follow-up completed")
    answer = next(event["transcript"] for event in events if event["type"] == "response.output_audio_transcript.done")
    return {"call": call, "answer": answer}


async def s4_barge_in(session: Session, speech: list[int]) -> dict:
    await session.send({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Count slowly from one to forty, one number per sentence."}]}})
    await session.until("conversation.item.done")
    await session.send({"type": "response.create"})
    events = await session.until("response.output_audio.delta", 180)
    item_id = events[-1]["item_id"]
    await session.append(speech + [0] * 36_000)
    events = await session.until("response.done", 60)
    names = types(events)
    require("input_audio_buffer.speech_started" in names, "speech_started")
    done = events[-1]["response"]
    require(done["status"] == "cancelled", f"cancelled, got {done['status']}")
    require(done["status_details"]["reason"] == "turn_detected", str(done["status_details"]))
    require(names.index("input_audio_buffer.speech_started") < len(names) - 1, "cancel follows speech")
    await session.send({"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": 800})
    truncated = (await session.until("conversation.item.truncated", 30))[-1]
    require(truncated["item_id"] == item_id, "truncated item")
    # The new turn is answered.
    events = await session.until("response.done", 180)
    return {"cancelled_item": item_id, "next_status": events[-1]["response"]["status"]}


async def s5_manual_turn(session: Session, speech: list[int]) -> dict:
    await session.send({"type": "session.update", "session": {"type": "realtime", "audio": {"input": {"turn_detection": None}}}})
    updated = (await session.until("session.updated"))[-1]
    require(updated["session"]["audio"]["input"]["turn_detection"] is None, "manual mode")
    await session.append(speech)
    await session.send({"type": "input_audio_buffer.commit"})
    events = await session.until("input_audio_buffer.committed", 30)
    require("input_audio_buffer.speech_started" not in types(events), "no VAD events in manual mode")
    await session.send({"type": "response.create"})
    events = await session.until("response.done", 180)
    check_response_order(events)
    require(events[-1]["response"]["status"] == "completed", "completed")
    return {"audio_ms": audio_ms(events)}


async def s6_out_of_band(session: Session, _speech) -> dict:
    await session.send({"type": "conversation.item.create", "item": {"id": "item_oob_user", "type": "message", "role": "user", "content": [{"type": "input_text", "text": "I absolutely love this product."}]}})
    await session.until("conversation.item.done")
    await session.send({"type": "response.create", "response": {"conversation": "none", "metadata": {"topic": "classification"}, "output_modalities": ["text"], "instructions": "Classify the sentiment of the user's message. Reply with one word: positive or negative.", "input": [{"type": "item_reference", "id": "item_oob_user"}]}})
    events = await session.until("response.done", 120)
    require("conversation.item.added" not in types(events), "out-of-band output stays out of the conversation")
    done = events[-1]["response"]
    require(done["metadata"] == {"topic": "classification"}, "metadata echoed")
    require(done["conversation_id"] is None, "no conversation")
    text = next(event["text"] for event in events if event["type"] == "response.output_text.done")
    return {"classification": text}


async def s7_g711(session: Session, speech: list[int]) -> dict:
    await session.send({"type": "session.update", "session": {"type": "realtime", "audio": {"input": {"format": {"type": "audio/pcmu"}}, "output": {"format": {"type": "audio/pcmu"}}}}})
    await session.until("session.updated")
    audio = to_ulaw_8k([0] * 7_200 + speech + [0] * 36_000)
    for start in range(0, len(audio), 800):
        await session.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(audio[start:start + 800]).decode()})
    events = await session.until("response.done", 180)
    require(events[-1]["response"]["status"] == "completed", "completed")
    spoken = audio_ms(events, bytes_per_ms=8)
    require(spoken > 300, f"spoke {spoken} ms of μ-law")
    return {"audio_ms": spoken}


async def s8_history_replay(session: Session, _speech) -> dict:
    previous = "root"
    history = [
        {"id": "lk_sys", "type": "message", "role": "system", "content": [{"type": "input_text", "text": "The user's name is Tendai."}]},
        {"id": "lk_u1", "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi, remember my name."}]},
        {"id": "lk_a1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Of course, Tendai."}]},
        {"id": "lk_u2", "type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is my name? Answer in one word."}]},
    ]
    for item in history:
        await session.send({"type": "conversation.item.create", "previous_item_id": previous, "item": item})
        added = (await session.until("conversation.item.added"))[-1]
        require(added["previous_item_id"] == (None if previous == "root" else previous), f"previous_item_id {added['previous_item_id']}")
        await session.until("conversation.item.done")
        previous = item["id"]
    await session.send({"type": "response.create", "response": {"output_modalities": ["text"], "metadata": {"client_event_id": "lk_reply_1"}}})
    events = await session.until("response.done", 120)
    require(events[0]["response"]["metadata"] == {"client_event_id": "lk_reply_1"}, "metadata on response.created (LiveKit)")
    text = next(event["text"] for event in events if event["type"] == "response.output_text.done")
    require("tendai" in text.lower(), f"history used: {text!r}")
    return {"answer": text}


SCENARIOS = {"s1": s1_vad_turn, "s2": s2_text_response, "s3": s3_function_call, "s4": s4_barge_in, "s5": s5_manual_turn, "s6": s6_out_of_band, "s7": s7_g711, "s8": s8_history_replay}


async def run(args) -> int:
    speech = read_wav_24k(Path(args.audio))
    client = AsyncOpenAI(base_url=f"{args.url.rstrip('/')}/v1", api_key=args.api_key)
    record: list | None = [] if args.record else None
    results = {}
    failed = 0
    for name in args.scenarios.split(","):
        scenario = SCENARIOS[name.strip()]
        started = time.monotonic()
        try:
            async with client.realtime.connect(model=args.model) as connection:
                session = Session(connection, record)
                created = await session.recv(60)
                require(created["type"] == "session.created", created["type"])
                detail = await scenario(session, speech)
                results[name] = {"ok": True, "seconds": round(time.monotonic() - started, 1), **detail}
        except Exception as err:  # noqa: BLE001 - a scenario's failure is reported, the run goes on
            failed += 1
            results[name] = {"ok": False, "seconds": round(time.monotonic() - started, 1), "error": f"{type(err).__name__}: {err}"}
        print(json.dumps({name: results[name]}, default=str), flush=True)
    if args.record:
        with open(args.record, "w") as handle:
            for line in record:
                handle.write(json.dumps(line) + "\n")
    print(json.dumps({"model": args.model, "passed": len(results) - failed, "failed": failed}))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="izwi-voice")
    parser.add_argument("--api-key", default="izwi-local")
    parser.add_argument("--audio", default=str(REPO / "data" / "fox.wav"))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--record", help="write every event to this JSONL file")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
