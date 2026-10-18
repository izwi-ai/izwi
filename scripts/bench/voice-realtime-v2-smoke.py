#!/usr/bin/env python3
"""Protocol smoke test for `/v1/voice/realtime/ws` as the desktop UI speaks it
(typed protocol `voice_realtime` version 2).

Scenarios (all run by default, or pick with --scenario):
- `turn`: one spoken turn; expects the model status before speech, a final
  transcript, assistant text and audio, and `turn_completed` ok.
- `barge-in`: speaks again as soon as the reply starts playing; expects the
  session to survive with a gap-free event sequence.
- `missing-model`: asks for an ASR model that is not downloaded; expects
  `models_status` to report it unavailable before any speech is sent.

Every scenario checks the UI's envelope rule: within one connection epoch each
event's `sequence` is the previous one plus one (`isValidVoiceRealtimeV2Successor`
in ui/src/features/voice/realtime/support.ts), so a gap here closes the UI's
socket.

Standard library only; reuses the websocket client of run-voice-realtime-load.py.
Exits non-zero when any scenario fails.
"""
import argparse
import asyncio
import importlib.util
import json
import pathlib
import sys
import time

_BENCH = pathlib.Path(__file__).with_name("run-voice-realtime-load.py")
_spec = importlib.util.spec_from_file_location("voice_realtime_load", _BENCH)
load = importlib.util.module_from_spec(_spec)
sys.dont_write_bytecode = True
_spec.loader.exec_module(load)

PROTOCOL = "voice_realtime"
# The last session's events, so errors are reported even when a scenario raises.
LAST_EVENTS = []
VERSION = 2


class Session:
    """One v2 socket: records every event and checks the envelope sequence."""

    def __init__(self, ws):
        self.ws = ws
        # Shared with the runner, which reports errors even when a scenario raises.
        LAST_EVENTS.clear()
        self.events = LAST_EVENTS
        self.audio_frames = 0
        self.previous = None
        self.sequence_errors = []

    def check_successor(self, event):
        previous = self.previous
        if previous is None:
            ok = event.get("sequence") == 0 and event.get("connection_epoch") == 0
        elif event.get("connection_epoch") == previous.get("connection_epoch"):
            ok = event.get("sequence") == previous.get("sequence", -1) + 1
        else:
            ok = (event.get("connection_epoch") == previous.get("connection_epoch", -1) + 1
                  and event.get("sequence") == 0)
        if not ok:
            self.sequence_errors.append(
                f"{event.get('type')} seq={event.get('sequence')} epoch={event.get('connection_epoch')}"
                f" after {previous and previous.get('type')} seq={previous and previous.get('sequence')}")
        self.previous = event

    async def recv_until(self, predicate, deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("deadline passed")
            message = await asyncio.wait_for(self.ws.recv(), remaining)
            if message is None:
                raise ConnectionError("socket closed")
            if isinstance(message, bytes):
                if load.parse_assistant_audio_frame(message) is not None:
                    self.audio_frames += 1
                continue
            event = json.loads(message)
            if event.get("protocol") == PROTOCOL:
                self.check_successor(event)
            self.events.append(event)
            if predicate(event):
                return event

    def types(self):
        return [event.get("type") for event in self.events]

    def first(self, event_type):
        return next((event for event in self.events if event.get("type") == event_type), None)


async def open_session(args, asr_model=None):
    ws = await load.WebSocket.connect(args.base_url.replace("http", "ws", 1) + "/v1/voice/realtime/ws")
    session = Session(ws)
    deadline = time.monotonic() + 30
    await ws.send_json({"type": "session_start", "protocol": PROTOCOL, "version": VERSION})
    await session.recv_until(lambda e: e.get("type") == "session_ready", deadline)
    if args.mode == "unified":
        start = {"type": "input_stream_start", "mode": "unified", "s2s_model_id": args.unified_model}
    else:
        start = {
            "type": "input_stream_start",
            "mode": "modular",
            "asr_model_id": asr_model or args.asr_model,
            "text_model_id": args.text_model,
            "tts_model_id": args.tts_model,
            "asr_language": "Auto",
        }
    start.update({
        "speaker": args.speaker,
        "max_output_tokens": 256,
        "max_utterance_ms": 20_000,
        "pre_roll_ms": 160,
        "input_sample_rate": args.sample_rate,
    })
    await ws.send_json(start)
    await session.recv_until(lambda e: e.get("type") == "session_started", deadline)
    return session


def model_status(session):
    """The models_status event, or None (B4: v2 used to drop it)."""
    return session.first("models_status")


async def scenario_turn(args, frames):
    session = await open_session(args)
    failures = []
    try:
        deadline = time.monotonic() + args.timeout
        try:
            status = await session.recv_until(
                lambda e: e.get("type") in ("models_status", "recoverable_error", "fatal_error"),
                time.monotonic() + args.load_timeout)
            if status.get("type") != "models_status":
                failures.append(f"expected models_status before speech, got {status}")
            elif not all(model.get("ready") for model in status["data"]["models"]):
                failures.append(f"models not ready: {status['data']['models']}")
        except TimeoutError:
            failures.append(f"no models_status within {args.load_timeout:.0f} s of session_started")
        sender = asyncio.ensure_future(load.stream_audio(
            session.ws, load.VOICE_MAGIC, frames, args.sample_rate, 1, args.silence_frames))
        done = await session.recv_until(lambda e: e.get("type") == "turn_completed", deadline)
        await sender
        final = session.first("transcript_final")
        if not final or not final["data"]["text"].strip():
            failures.append("no non-empty transcript_final")
        if not session.first("assistant_text_final"):
            failures.append("no assistant_text_final")
        if session.audio_frames == 0:
            failures.append("no assistant audio frames")
        if done["data"].get("status") != "ok":
            failures.append(f"turn_completed status {done['data'].get('status')}")
        detail = {"transcript": final and final["data"]["text"],
                  "reply": (session.first("assistant_text_final") or {}).get("data", {}).get("text"),
                  "audio_frames": session.audio_frames}
    finally:
        await session.ws.close()
    return failures + session.sequence_errors, detail, session.types()


async def scenario_barge_in(args, frames):
    session = await open_session(args)
    failures = []
    try:
        deadline = time.monotonic() + args.timeout
        sender = asyncio.ensure_future(load.stream_audio(
            session.ws, load.VOICE_MAGIC, frames, args.sample_rate, 1, args.silence_frames))
        await session.recv_until(lambda e: e.get("type") == "assistant_audio_started", deadline)
        seq = await sender
        # Speak over the reply, then let the second turn finish.
        await load.stream_audio(session.ws, load.VOICE_MAGIC, frames, args.sample_rate, seq, args.silence_frames)
        completed = 0

        def second_turn_done(event):
            nonlocal completed
            if event.get("type") == "turn_completed":
                completed += 1
            return completed >= 2 and event.get("type") == "turn_completed"

        await session.recv_until(second_turn_done, deadline)
        if not session.first("interruption"):
            failures.append("no interruption event for the barge-in")
    finally:
        await session.ws.close()
    return failures + session.sequence_errors, {"turns_completed": completed}, session.types()


async def scenario_missing_model(args, _frames):
    session = await open_session(args, asr_model=args.missing_asr_model)
    failures, detail = [], {}
    try:
        status = await session.recv_until(
            lambda e: e.get("type") in ("models_status", "recoverable_error", "fatal_error"),
            time.monotonic() + args.load_timeout)
        if status.get("type") != "models_status":
            failures.append(f"expected models_status, got {status}")
        else:
            asr = next((m for m in status["data"]["models"] if m.get("stage") == "asr"), None)
            detail = {"asr": asr}
            if not asr or asr.get("ready") or not asr.get("error"):
                failures.append(f"asr should be reported unavailable: {status['data']}")
    except TimeoutError:
        failures.append(f"no models_status within {args.load_timeout:.0f} s of session_started")
    finally:
        await session.ws.close()
    return failures + session.sequence_errors, detail, session.types()


SCENARIOS = {"turn": scenario_turn, "barge-in": scenario_barge_in, "missing-model": scenario_missing_model}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--mode", choices=["modular", "unified"], default="modular")
    parser.add_argument("--wav", default=str(pathlib.Path(__file__).parents[2] / "data" / "fox.wav"))
    parser.add_argument("--asr-model", default="Parakeet-TDT-0.6B-v3")
    parser.add_argument("--text-model", default="Qwen3.5-4B")
    parser.add_argument("--tts-model", default="Kokoro-82M")
    parser.add_argument("--unified-model", default="LFM2.5-Audio-1.5B-GGUF")
    parser.add_argument("--speaker", default="af_heart")
    parser.add_argument("--missing-asr-model", default="VibeVoice-ASR")
    parser.add_argument("--silence-ms", type=int, default=1500)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--load-timeout", type=float, default=180.0,
                        help="seconds to wait for models_status after session_started")
    return parser.parse_args(argv)


async def run(args):
    pcm, args.sample_rate = load.read_wav(args.wav)
    frames = load.pcm_frames(pcm, args.sample_rate)
    args.silence_frames = args.silence_ms // load.FRAME_MS
    failed = False
    for name in args.scenario or list(SCENARIOS):
        started = time.monotonic()
        LAST_EVENTS.clear()
        try:
            failures, detail, types = await SCENARIOS[name](args, frames)
        except Exception as error:  # noqa: BLE001 - report and continue
            failures, detail, types = [f"{type(error).__name__}: {error}"], {}, []
        errors = [event.get("data") for event in LAST_EVENTS
                  if event.get("type") in ("recoverable_error", "fatal_error")]
        failed |= bool(failures)
        print(json.dumps({
            "scenario": name,
            "ok": not failures,
            "seconds": round(time.monotonic() - started, 1),
            "failures": failures,
            "detail": detail,
            "errors": errors,
            "events": types,
        }, indent=2))
    return 1 if failed else 0


def main(argv=None):
    return asyncio.run(run(parse_args(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    sys.exit(main())
