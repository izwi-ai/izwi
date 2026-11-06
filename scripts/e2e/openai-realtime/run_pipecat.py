#!/usr/bin/env python3
"""End-to-end check of izwi's `/v1/realtime` with Pipecat's OpenAI realtime service.

Runs `OpenAIRealtimeLLMService` (pipecat-ai 1.15.0) in a real pipeline (context
aggregators, the service, a sink) without a transport: frames are queued on
the task, and only `base_url` points at izwi. Pipecat validates every server
event strictly and ends its receive loop on almost any `error` event, so the
run also fails on a parse failure or an unexpected error event.

Checks:
  - the handshake (session.created → session.update → session.updated);
  - a function call round trip through the context aggregator;
  - a spoken turn under server VAD: transcription and spoken answer;
  - barge-in: speech during the answer, Pipecat's truncate, the server's cancel.

Usage: python3 run_pipecat.py --url ws://127.0.0.1:18080 --model gpt-realtime
Requires pipecat-ai[openai]==1.15.0.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import wave
from pathlib import Path

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

REPO = Path(__file__).resolve().parents[3]
SERVER: list[dict] = []
CLIENT: list[dict] = []


def _tap_events() -> None:
    """Log every server event Pipecat parses and every client event it sends."""
    parse = events.parse_server_event

    def logged_parse(message):
        SERVER.append(json.loads(message))
        return parse(message)

    events.parse_server_event = logged_parse
    send = OpenAIRealtimeLLMService.send_client_event

    async def logged_send(self, event):
        CLIENT.append(event.model_dump(exclude_none=True))
        return await send(self, event)

    OpenAIRealtimeLLMService.send_client_event = logged_send


class Sink(FrameProcessor):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.audio_bytes = 0
        self.transcripts: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.audio_bytes += len(frame.audio)
        if isinstance(frame, TranscriptionFrame):
            self.transcripts.append(frame.text)
        await self.push_frame(frame, direction)


def speech(path: Path) -> list[bytes]:
    with wave.open(str(path)) as handle:
        samples = list(struct.unpack(f"<{handle.getnframes()}h", handle.readframes(handle.getnframes())))
    samples = [0] * 4_800 + samples + [0] * 36_000
    return [struct.pack(f"<{len(samples[i:i + 480])}h", *samples[i:i + 480]) for i in range(0, len(samples), 480)]


async def wait_for(predicate, timeout: float, what: str) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.05)


def count(event_type: str) -> int:
    return sum(1 for event in SERVER if event.get("type") == event_type)


async def run(args) -> int:
    _tap_events()
    calls: list[dict] = []

    async def get_weather(params):
        calls.append(dict(params.arguments))
        await params.result_callback({"city": params.arguments.get("city"), "temp_c": 24, "sky": "sunny"})

    llm = OpenAIRealtimeLLMService(
        api_key=args.api_key,
        base_url=f"{args.url.rstrip('/')}/v1/realtime",
        settings=OpenAIRealtimeLLMService.Settings(
            model=args.model,
            system_instruction="You are a helpful voice assistant. Answer in one short sentence.",
            session_properties=events.SessionProperties(
                audio=events.AudioConfiguration(
                    input=events.AudioInput(transcription=events.InputAudioTranscription(), turn_detection=events.TurnDetection())
                )
            ),
        ),
    )
    llm.register_function("get_weather", get_weather)
    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(name="get_weather", description="Get the current weather for a city.", properties={"city": {"type": "string"}}, required=["city"])
        ]
    )
    context = LLMContext(messages=[], tools=tools)
    aggregators = LLMContextAggregatorPair(context)
    upstream, downstream = Sink("upstream"), Sink("downstream")
    pipeline = Pipeline([upstream, aggregators.user(), llm, downstream, aggregators.assistant()])
    task = PipelineTask(pipeline, params=PipelineParams(audio_in_sample_rate=24_000, audio_out_sample_rate=24_000))
    runner = PipelineRunner(handle_sigint=False)
    running = asyncio.ensure_future(runner.run(task))
    results = {}

    async def step(name, coro):
        try:
            results[name] = {"ok": True, **(await coro or {})}
        except Exception as err:  # noqa: BLE001
            results[name] = {
                "ok": False,
                "error": f"{type(err).__name__}: {err}",
                "last_server": [event.get("type") for event in SERVER[-12:]],
                "last_client": [event.get("type") for event in CLIENT[-8:] if event.get("type") != "input_audio_buffer.append"],
                "appends_sent": sum(1 for event in CLIENT if event.get("type") == "input_audio_buffer.append"),
                "speech_started": count("input_audio_buffer.speech_started"),
            }
        print(json.dumps({name: results[name]}, default=str), flush=True)

    async def handshake():
        await wait_for(lambda: count("session.updated") >= 1, 30, "session.updated")
        names = [event["type"] for event in SERVER]
        assert names.index("session.created") < names.index("session.updated")
        return {"server_events": names[:4]}

    async def tool_turn():
        await task.queue_frames([LLMMessagesAppendFrame(messages=[{"role": "user", "content": "What is the weather in Harare right now?"}], run_llm=True)])
        await wait_for(lambda: calls, 120, "the get_weather call")
        await wait_for(lambda: count("response.done") >= 2, 180, "the answer after the tool result")
        outputs = [event for event in CLIENT if event.get("type") == "conversation.item.create" and event.get("item", {}).get("type") == "function_call_output"]
        assert outputs, "Pipecat sent the function_call_output"
        return {"arguments": calls[0], "audio_bytes": downstream.audio_bytes}

    async def spoken_turn():
        before = count("response.done")
        for chunk in speech(Path(args.audio)):
            await task.queue_frames([InputAudioRawFrame(audio=chunk, sample_rate=24_000, num_channels=1)])
            await asyncio.sleep(0.005)
        await wait_for(lambda: count("response.done") > before, 180, "the spoken answer")
        # The user aggregator takes Pipecat's TranscriptionFrame into the
        # context; the transcript itself is the server event.
        await wait_for(
            lambda: count("conversation.item.input_audio_transcription.completed") >= 1, 60, "the user transcript"
        )
        transcript = [event for event in SERVER if event.get("type") == "conversation.item.input_audio_transcription.completed"][-1]
        return {"transcript": transcript["transcript"], "in_context": [message.get("content") for message in context.get_messages() if message.get("role") == "user"][-1:]}

    async def barge_in():
        # Speech starts a spoken answer; more speech interrupts it. (Pipecat
        # sends a mid-session LLMMessagesAppendFrame through its context, which
        # the realtime service reads only when the conversation starts.)
        before = count("response.output_audio.delta")
        done_before = count("response.done")
        for chunk in speech(Path(args.audio)):
            await task.queue_frames([InputAudioRawFrame(audio=chunk, sample_rate=24_000, num_channels=1)])
            await asyncio.sleep(0.005)
        await wait_for(lambda: count("response.output_audio.delta") > before + 5, 120, "the spoken answer")
        truncated_before = count("conversation.item.truncated")
        for chunk in speech(Path(args.audio)):
            await task.queue_frames([InputAudioRawFrame(audio=chunk, sample_rate=24_000, num_channels=1)])
            await asyncio.sleep(0.005)
        # Pipecat stops playback and truncates the playing item on
        # speech_started; the server cancels the answer if it is still being
        # generated (it is often all sent already, faster than real time).
        await wait_for(lambda: count("conversation.item.truncated") > truncated_before, 60, "the truncate")
        await wait_for(lambda: count("response.done") >= done_before + 2, 180, "the next answer")
        statuses = [event["response"]["status"] for event in SERVER if event.get("type") == "response.done"][done_before:]
        truncates = [event for event in CLIENT if event.get("type") == "conversation.item.truncate"]
        return {"truncate": truncates[-1], "statuses": statuses}

    for name, coro in [("handshake", handshake()), ("tool_turn", tool_turn()), ("spoken_turn", spoken_turn()), ("barge_in", barge_in())]:
        await step(name, coro)
    if args.dump:
        with open(args.dump, "w") as handle:
            for event in SERVER:
                if event.get("type") != "response.output_audio.delta":
                    handle.write(json.dumps({key: value for key, value in event.items() if key != "delta"})[:400] + "\n")
    errors = [event["error"] for event in SERVER if event.get("type") == "error"]
    tolerated = {"response_cancel_not_active", "conversation_already_has_active_response", "item_retrieve_invalid_item_id"}
    fatal = [error for error in errors if error.get("code") not in tolerated]
    print(json.dumps({"error_events": errors}))
    await task.queue_frames([EndFrame()])
    try:
        await asyncio.wait_for(running, 15)
    except asyncio.TimeoutError:
        running.cancel()
    failed = sum(1 for result in results.values() if not result["ok"])
    print(json.dumps({"model": args.model, "failed": failed, "fatal_error_events": len(fatal)}))
    return 1 if failed or fatal else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:18080")
    parser.add_argument("--model", default="gpt-realtime")
    parser.add_argument("--api-key", default="izwi-local")
    parser.add_argument("--audio", default=str(REPO / "data" / "fox.wav"))
    parser.add_argument("--dump", help="write the server events (audio elided) to this file")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
