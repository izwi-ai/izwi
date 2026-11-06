#!/usr/bin/env python3
"""End-to-end check of izwi's `/v1/realtime` with LiveKit Agents' OpenAI realtime model.

Drives `livekit.plugins.openai.realtime.RealtimeModel` (livekit-agents 1.8.11)
directly, without a LiveKit room: only `base_url` points at izwi. Checks what
LiveKit depends on (see tasks/openai-realtime-evidence-2026-11-06):
  - generate_reply resolves (response.created echoes metadata.client_event_id)
    and its message streams carry audio and text;
  - update_chat_ctx and update_tools are acknowledged without LiveKit's 5 s timeout;
  - a function tool call arrives on the generation's function stream;
  - pushed speech triggers input_speech_started and a server-initiated reply;
  - interrupt() cancels a reply.

Usage: python3 run_livekit.py --url http://127.0.0.1:18080 --model gpt-realtime
Requires livekit-agents==1.8.11 and livekit-plugins-openai==1.8.11.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
import wave
from pathlib import Path

import aiohttp
from livekit import rtc
from livekit.agents import function_tool, llm
from livekit.plugins import openai as lk_openai

REPO = Path(__file__).resolve().parents[3]


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return json.dumps({"city": city, "temp_c": 24, "sky": "sunny"})


def speech_frames(path: Path) -> list[rtc.AudioFrame]:
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == 24_000 and handle.getnchannels() == 1
        samples = list(struct.unpack(f"<{handle.getnframes()}h", handle.readframes(handle.getnframes())))
    samples = [0] * 4_800 + samples + [0] * 36_000
    frames = []
    for start in range(0, len(samples), 480):
        chunk = samples[start:start + 480]
        chunk += [0] * (480 - len(chunk))
        frames.append(rtc.AudioFrame(struct.pack(f"<{len(chunk)}h", *chunk), 24_000, 1, len(chunk)))
    return frames


async def consume(generation: llm.GenerationCreatedEvent, timeout: float = 120) -> dict:
    """Read a generation to its end: its messages' text and audio, its calls."""
    text, audio_samples, calls = [], 0, []

    async def messages():
        nonlocal audio_samples
        async for message in generation.message_stream:
            async def read_text():
                async for chunk in message.text_stream:
                    text.append(str(chunk))

            async def read_audio():
                nonlocal audio_samples
                async for frame in message.audio_stream:
                    audio_samples += frame.samples_per_channel

            await asyncio.gather(read_text(), read_audio())

    async def functions():
        async for call in generation.function_stream:
            calls.append(call)

    await asyncio.wait_for(asyncio.gather(messages(), functions()), timeout)
    return {"text": "".join(text), "audio_ms": audio_samples // 24, "calls": calls}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def run(args) -> int:
    results = {}
    async with aiohttp.ClientSession() as http:
        model = lk_openai.realtime.RealtimeModel(
            base_url=f"{args.url.rstrip('/')}/v1", api_key=args.api_key, model=args.model, http_session=http
        )
        session = model.session()
        server_events: list[dict] = []
        errors: list = []
        session.on("openai_server_event_received", lambda event: server_events.append(event))
        session.on("error", lambda error: errors.append(error))
        generations: asyncio.Queue = asyncio.Queue()
        session.on("generation_created", lambda event: generations.put_nowait(event))

        async def step(name, coro):
            started = time.monotonic()
            try:
                detail = await coro
                results[name] = {"ok": True, "seconds": round(time.monotonic() - started, 1), **(detail or {})}
            except Exception as err:  # noqa: BLE001
                results[name] = {"ok": False, "error": f"{type(err).__name__}: {err}"}
            print(json.dumps({name: results[name]}, default=str), flush=True)

        async def greeting():
            await session.update_instructions("You are a helpful voice assistant. Keep answers to one sentence.")
            generation = await asyncio.wait_for(session.generate_reply(instructions="Greet the user briefly."), 30)
            require(generation.user_initiated, "generate_reply bound by the echoed metadata")
            out = await consume(generation)
            require(out["audio_ms"] > 300 and out["text"].strip(), f"spoken greeting: {out}")
            return {"text": out["text"], "audio_ms": out["audio_ms"]}

        async def chat_ctx():
            started = time.monotonic()
            ctx = session.chat_ctx.copy()
            ctx.add_message(role="user", content="My name is Tendai. Please remember it.")
            await session.update_chat_ctx(ctx)
            elapsed = time.monotonic() - started
            require(elapsed < 4, f"update_chat_ctx acknowledged quickly ({elapsed:.1f} s)")
            generation = await asyncio.wait_for(session.generate_reply(instructions="Say the user's name."), 30)
            out = await consume(generation)
            require("tendai" in out["text"].lower(), f"history used: {out['text']!r}")
            return {"ack_s": round(elapsed, 2), "text": out["text"]}

        async def tools():
            started = time.monotonic()
            await session.update_tools([get_weather])
            require(time.monotonic() - started < 4, "update_tools quickly")
            ctx = session.chat_ctx.copy()
            ctx.add_message(role="user", content="What is the weather in Harare right now?")
            await session.update_chat_ctx(ctx)
            generation = await asyncio.wait_for(session.generate_reply(), 30)
            out = await consume(generation)
            require(out["calls"], f"a function call: {out}")
            call = out["calls"][0]
            require(call.name == "get_weather" and "harare" in call.arguments.lower(), f"{call.name} {call.arguments}")
            ctx = session.chat_ctx.copy()
            ctx.items.append(llm.FunctionCallOutput(call_id=call.call_id, name=call.name, output=await get_weather(city="Harare"), is_error=False))
            await session.update_chat_ctx(ctx)
            generation = await asyncio.wait_for(session.generate_reply(), 30)
            answer = await consume(generation)
            await session.update_tools([])
            return {"arguments": call.arguments, "answer": answer["text"]}

        async def speech():
            while not generations.empty():
                generations.get_nowait()
            for frame in speech_frames(Path(args.audio)):
                session.push_audio(frame)
            generation = await asyncio.wait_for(generations.get(), 120)
            require(not generation.user_initiated, "a server-initiated reply")
            out = await consume(generation)
            names = [event.get("type") for event in server_events]
            require("input_audio_buffer.speech_started" in names, "speech_started")
            transcripts = [event["transcript"] for event in server_events if event.get("type") == "conversation.item.input_audio_transcription.completed"]
            return {"transcripts": transcripts[-1:], "reply": out["text"], "audio_ms": out["audio_ms"]}

        async def interrupt():
            generation = await asyncio.wait_for(
                session.generate_reply(instructions="Count slowly from one to forty, one number per sentence."), 30
            )
            reader = asyncio.ensure_future(consume(generation))
            await asyncio.sleep(1.5)
            session.interrupt()
            out = await asyncio.wait_for(reader, 30)
            done = [event for event in server_events if event.get("type") == "response.done"][-1]
            require(done["response"]["status"] == "cancelled", f"cancelled: {done['response']['status']}")
            return {"audio_ms_before_cancel": out["audio_ms"]}

        for name, coro in [("greeting", greeting()), ("chat_ctx", chat_ctx()), ("tools", tools()), ("speech", speech()), ("interrupt", interrupt())]:
            await step(name, coro)
        fatal = [str(error) for error in errors if "Cancellation failed" not in str(error)]
        results["errors"] = fatal
        print(json.dumps({"errors": fatal}, default=str))
        await session.aclose()
        await model.aclose()
    failed = sum(1 for result in results.values() if isinstance(result, dict) and not result.get("ok"))
    print(json.dumps({"model": args.model, "failed": failed, "error_events": len(fatal)}))
    return 1 if failed or fatal else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="gpt-realtime")
    parser.add_argument("--api-key", default="izwi-local")
    parser.add_argument("--audio", default=str(REPO / "data" / "fox.wav"))
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
