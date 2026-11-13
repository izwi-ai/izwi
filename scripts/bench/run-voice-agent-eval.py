#!/usr/bin/env python3
"""Simulated-caller evaluation of a voice agent on a running Izwi server.

A caller played by a chat model talks to an agent over the OpenAI Realtime
socket (`/v1/realtime`), turn by turn, until the caller is done or the turn
limit is reached. The script answers the agent's function calls with the
scenario's canned results, then checks the conversation two ways:

  checks  rules a script can decide: the tools that had to be called (with
          the arguments that had to be in the call), the tools that must not
          be called, and text the agent had to say or must not say
  judge   a chat model reads the transcript against the scenario's rubric
          and answers pass or fail with a reason

A scenario passes a run when every check and the judge pass. Each scenario
runs `--samples` times; the report gives its pass rate and the latency from
`response.create` to the first text or audio (p50, p95), and the script exits
non-zero when a scenario's pass rate is under `--pass-rate`.

In text mode (the default) the caller's words reach the agent as text and the
agent answers in text: the agent's reasoning and tool use are tested without
speech. With `--audio` the caller's words are spoken by the server's TTS
(`/v1/audio/speech`), optionally with white noise (`--noise-db`, the noise
level under the speech in dB), and the agent hears and answers in speech; the
transcript the judge reads is then what the server transcribed and spoke.

Everything runs against the one server, so it works offline: the caller, the
judge, the agent and the speech all use local models.

Usage (standard library only):

    python3 scripts/bench/run-voice-agent-eval.py \
        --base-url http://127.0.0.1:8080 --agent-model Qwen3.5-4B \
        --out target/voice-agent-eval.json

`--scenarios FILE` replaces the built-in scenarios with a JSON list of the
same shape (see SCENARIOS below); `--only` filters them by name;
`--caller-model` and `--judge-model` default to the agent's model.
"""
import argparse
import asyncio
import base64
import importlib.util
import io
import json
import math
import os
import random
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
import wave

SCHEMA = "izwi.voice_agent_eval"
DONE = "[DONE]"


def _load_ws():
    """The minimal RFC 6455 client of the realtime load test, beside this file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run-voice-realtime-load.py")
    spec = importlib.util.spec_from_file_location("izwi_voice_realtime_load", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WebSocket


def fn(name, description, properties=None, required=None):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
    }


# A scenario:
#   name          a short id
#   persona       who the caller is and what they want (the caller model's prompt)
#   instructions  the agent's instructions
#   tools         the agent's function tools (Realtime form)
#   tool_results  what each tool answers: a value, or a list of
#                 {"when": {arg: value}, "result": ...} tried in order with a
#                 final {"result": ...} as the default
#   expect        calls:      [{"name", "args": {subset}}] that must be made
#                 no_calls:   [tool names] that must not be called
#                 says:       [substrings] the agent must say (any case)
#                 never_says: [substrings] the agent must not say
#                 judge:      the rubric the judge decides pass/fail on
#   max_turns     the most caller turns (default 6)
SCENARIOS = [
    {
        "name": "order_status",
        "persona": (
            "You are Thandi, calling a shop about your order. Your order number is 48213. "
            "You want to know when it arrives. Give the order number when asked. "
            "Once you know the delivery day, say thanks and end the call."
        ),
        "instructions": (
            "You are the phone assistant of a small online shop. Keep answers to one or two "
            "short sentences. To answer anything about an order, ask for the order number and "
            "look it up with the tool; never guess a delivery date."
        ),
        "tools": [
            fn("lookup_order", "Look up an order's status by its number.",
               {"order_number": {"type": "string"}}, ["order_number"]),
        ],
        "tool_results": {
            "lookup_order": [
                {"when": {"order_number": "48213"},
                 "result": {"status": "shipped", "arrives": "Thursday"}},
                {"result": {"error": "no such order"}},
            ],
        },
        "expect": {
            "calls": [{"name": "lookup_order", "args": {"order_number": "48213"}}],
            "says": ["thursday"],
            "judge": (
                "The agent asked for or used the order number, looked the order up, and told "
                "the caller it arrives on Thursday without inventing other details."
            ),
        },
    },
    {
        "name": "no_tool_small_talk",
        "persona": (
            "You are Sipho. You just want to know what the shop sells; ask once, then say "
            "thanks and end the call."
        ),
        "instructions": (
            "You are the phone assistant of a shop that sells garden tools and seeds. Keep "
            "answers to one or two short sentences. Use the order tool only for questions "
            "about an order."
        ),
        "tools": [
            fn("lookup_order", "Look up an order's status by its number.",
               {"order_number": {"type": "string"}}, ["order_number"]),
        ],
        "tool_results": {"lookup_order": {"error": "no such order"}},
        "expect": {
            "no_calls": ["lookup_order"],
            "judge": "The agent said what the shop sells (garden tools and seeds) and called no tool.",
        },
        "max_turns": 3,
    },
    {
        "name": "booking_two_fields",
        "persona": (
            "You are Lerato and want to book a table. You want it for four people on Friday. "
            "Only give the party size and the day when the agent asks for them, one at a time. "
            "When the booking is confirmed, say thanks and end the call."
        ),
        "instructions": (
            "You take table bookings for a restaurant by phone. Keep answers to one short "
            "sentence. You need the party size and the day; ask for what is missing, then make "
            "the booking with the tool and confirm it."
        ),
        "tools": [
            fn("book_table", "Book a table.",
               {"party_size": {"type": "integer"}, "day": {"type": "string"}},
               ["party_size", "day"]),
        ],
        "tool_results": {"book_table": {"confirmed": True, "reference": "R-77"}},
        "expect": {
            "calls": [{"name": "book_table", "args": {"party_size": 4}}],
            "judge": (
                "The agent collected both the party size (four) and the day (Friday) before "
                "booking, and confirmed the booking to the caller."
            ),
        },
    },
]


# --- HTTP -------------------------------------------------------------------

def post(base_url, path, body, api_key, timeout=300, raw=False):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"{path}: HTTP {err.code}: {err.read().decode(errors='replace')[:300]}")
    return data if raw else json.loads(data)


def chat(args, model, messages, max_tokens=200):
    reply = post(args.base_url, "/v1/chat/completions", {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.7, "enable_thinking": False,
    }, args.api_key)
    return (reply["choices"][0]["message"].get("content") or "").strip()


def caller_says(args, scenario, transcript):
    """The caller's next words, or None when the caller ends the call."""
    system = (
        scenario["persona"]
        + " You are on a phone call. Speak as the caller only: one or two short spoken "
        "sentences, no stage directions, no quotation marks. When you have what you came "
        f"for and have said goodbye, answer with exactly {DONE}."
    )
    # The caller sees the call from its own side: the agent's words are its `user`.
    messages = [{"role": "system", "content": system}]
    if not transcript:
        messages.append({"role": "user", "content": "(The call connects. You speak first.)"})
    for speaker, text in transcript:
        messages.append({"role": "assistant" if speaker == "caller" else "user", "content": text})
    words = chat(args, args.caller_model, messages, max_tokens=80)
    if DONE in words:
        words = words.replace(DONE, "").strip()
        return (words or None), True
    return words or None, False


def judge(args, scenario, transcript, calls):
    rubric = scenario["expect"].get("judge")
    if not rubric:
        return True, "no rubric"
    lines = "\n".join(f"{speaker.upper()}: {text}" for speaker, text in transcript)
    tools = "\n".join(f"- {call['name']}({call['arguments']})" for call in calls) or "(none)"
    verdict = chat(args, args.judge_model, [
        {"role": "system", "content": (
            "You grade a phone call between a caller and an automated agent. Decide whether "
            "the agent met the rubric. Answer with one JSON object and nothing else: "
            '{"pass": true or false, "reason": "one sentence"}.')},
        {"role": "user", "content": f"Rubric: {rubric}\n\nCall:\n{lines}\n\nTools the agent called:\n{tools}"},
    ], max_tokens=120)
    start, end = verdict.find("{"), verdict.rfind("}")
    try:
        parsed = json.loads(verdict[start:end + 1])
        return bool(parsed.get("pass")), str(parsed.get("reason", ""))[:300]
    except (ValueError, TypeError):
        return False, f"unreadable verdict: {verdict[:120]}"


# --- speech -----------------------------------------------------------------

def speak(args, text):
    """`text` as 24 kHz mono PCM16 bytes, with noise mixed in when asked."""
    data = post(args.base_url, "/v1/audio/speech", {
        "model": args.tts_model, "input": text, "voice": args.caller_voice,
        "response_format": "wav",
    }, args.api_key, raw=True)
    with wave.open(io.BytesIO(data)) as clip:
        rate, channels = clip.getframerate(), clip.getnchannels()
        frames = clip.readframes(clip.getnframes())
    samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))[::channels]
    if rate != 24_000:
        step = rate / 24_000
        samples = [samples[min(int(index * step), len(samples) - 1)]
                   for index in range(int(len(samples) / step))]
    if args.noise_db is not None and samples:
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) or 1.0
        level = rms * 10 ** (-args.noise_db / 20)
        samples = [max(-32768, min(32767, int(sample + random.gauss(0, level))))
                   for sample in samples]
    # Trailing silence, so the last word is not clipped by the commit.
    samples += [0] * 4_800
    return struct.pack(f"<{len(samples)}h", *samples)


# --- the call ---------------------------------------------------------------

def tool_result(scenario, name, arguments):
    spec = scenario.get("tool_results", {}).get(name, {"error": "tool not available"})
    if isinstance(spec, list):
        for rule in spec:
            when = rule.get("when")
            if when is None or all(str(arguments.get(key)) == str(value) for key, value in when.items()):
                return rule.get("result")
        return {"error": "no result"}
    return spec


async def agent_response(ws, scenario, calls, started):
    """Read one response to its end. Returns (text, function calls, first-output seconds)."""
    text, made, first = [], [], None
    while True:
        message = await asyncio.wait_for(ws.recv(), 300)
        if message is None:
            raise ConnectionError("the server closed the socket")
        if isinstance(message, bytes):
            continue
        event = json.loads(message)
        kind = event.get("type", "")
        if kind in ("response.output_text.delta", "response.output_audio_transcript.delta"):
            if first is None:
                first = time.monotonic() - started
            text.append(event.get("delta", ""))
        elif kind == "response.output_audio.delta" and first is None:
            first = time.monotonic() - started
        elif kind == "response.function_call_arguments.done":
            try:
                arguments = json.loads(event.get("arguments") or "{}")
            except ValueError:
                arguments = {}
            made.append({"call_id": event.get("call_id"), "name": event.get("name"),
                         "arguments": arguments})
        elif kind == "response.done":
            status = event.get("response", {}).get("status")
            if status not in ("completed", "incomplete"):
                detail = event.get("response", {}).get("status_details")
                raise RuntimeError(f"response {status}: {json.dumps(detail)[:300]}")
            calls.extend(made)
            return "".join(text).strip(), made, first
        elif kind == "error":
            raise RuntimeError(f"server error: {json.dumps(event.get('error'))[:300]}")


async def run_call(args, scenario, WebSocket):
    url = args.base_url.rstrip("/").replace("http://", "ws://", 1) + f"/v1/realtime?model={args.agent_model}"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    ws = await WebSocket.connect(url, headers=headers)
    transcript, calls, latencies = [], [], []
    try:
        await ws.send_json({"type": "session.update", "session": {
            "type": "realtime",
            "output_modalities": ["audio"] if args.audio else ["text"],
            "instructions": scenario["instructions"],
            "tools": scenario.get("tools", []),
            "tool_choice": "auto",
            "audio": {"input": {"turn_detection": None,
                                "transcription": {"model": "whisper-1"} if args.audio else None}},
        }})
        for _ in range(scenario.get("max_turns", 6)):
            words, done = await asyncio.to_thread(caller_says, args, scenario, transcript)
            if words:
                transcript.append(("caller", words))
                if args.audio:
                    pcm = await asyncio.to_thread(speak, args, words)
                    for offset in range(0, len(pcm), 48_000):
                        await ws.send_json({"type": "input_audio_buffer.append",
                                            "audio": base64.b64encode(pcm[offset:offset + 48_000]).decode()})
                    await ws.send_json({"type": "input_audio_buffer.commit"})
                else:
                    await ws.send_json({"type": "conversation.item.create", "item": {
                        "type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": words}]}})
                # A response that calls tools ends there; answer and ask again.
                said = []
                for _round in range(4):
                    started = time.monotonic()
                    await ws.send_json({"type": "response.create"})
                    text, made, first = await agent_response(ws, scenario, calls, started)
                    if first is not None:
                        latencies.append(first)
                    if text:
                        said.append(text)
                    if not made:
                        break
                    for call in made:
                        await ws.send_json({"type": "conversation.item.create", "item": {
                            "type": "function_call_output", "call_id": call["call_id"],
                            "output": json.dumps(tool_result(scenario, call["name"], call["arguments"]))}})
                if said:
                    transcript.append(("agent", " ".join(said)))
            if done:
                break
    finally:
        await ws.close()
    return transcript, calls, latencies


def check(scenario, transcript, calls):
    """The failures of the scripted checks."""
    expect, failures = scenario.get("expect", {}), []
    for wanted in expect.get("calls", []):
        def matches(call):
            return call["name"] == wanted["name"] and all(
                str(call["arguments"].get(key)).lower() == str(value).lower()
                for key, value in wanted.get("args", {}).items())
        if not any(matches(call) for call in calls):
            failures.append(f"no call {wanted['name']}({json.dumps(wanted.get('args', {}))})")
    for name in expect.get("no_calls", []):
        if any(call["name"] == name for call in calls):
            failures.append(f"called {name}")
    spoken = " ".join(text for speaker, text in transcript if speaker == "agent").lower()
    for phrase in expect.get("says", []):
        if phrase.lower() not in spoken:
            failures.append(f"never said {phrase!r}")
    for phrase in expect.get("never_says", []):
        if phrase.lower() in spoken:
            failures.append(f"said {phrase!r}")
    if not spoken:
        failures.append("the agent said nothing")
    return failures


def percentile(values, share):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(share * len(ordered)))] * 1000)


async def main_async(args):
    WebSocket = _load_ws()
    scenarios = SCENARIOS
    if args.scenarios:
        with open(args.scenarios) as handle:
            scenarios = json.load(handle)
    scenarios = [s for s in scenarios if not args.only or args.only in s["name"]]
    if not scenarios:
        sys.exit("no scenario matches")

    report = {"schema": SCHEMA, "mode": "audio" if args.audio else "text",
              "agent_model": args.agent_model, "caller_model": args.caller_model,
              "judge_model": args.judge_model, "samples": args.samples,
              "noise_db": args.noise_db, "scenarios": []}
    failed = False
    for scenario in scenarios:
        runs, latencies = [], []
        for sample in range(args.samples):
            try:
                transcript, calls, firsts = await run_call(args, scenario, WebSocket)
                failures = check(scenario, transcript, calls)
                verdict, reason = await asyncio.to_thread(judge, args, scenario, transcript, calls)
                if not verdict:
                    failures.append(f"judge: {reason}")
                latencies.extend(firsts)
                runs.append({"passed": not failures, "failures": failures, "judge_reason": reason,
                             "transcript": [list(turn) for turn in transcript],
                             "calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls]})
            except Exception as err:  # A broken run is a failed run, with its cause.
                runs.append({"passed": False, "failures": [f"run failed: {err}"],
                             "transcript": [], "calls": []})
            print(f"{scenario['name']} #{sample + 1}: "
                  f"{'pass' if runs[-1]['passed'] else 'FAIL ' + '; '.join(runs[-1]['failures'])}",
                  flush=True)
        rate = sum(run["passed"] for run in runs) / len(runs)
        failed |= rate < args.pass_rate
        report["scenarios"].append({
            "name": scenario["name"], "pass_rate": round(rate, 3),
            "first_output_ms_p50": percentile(latencies, 0.5),
            "first_output_ms_p95": percentile(latencies, 0.95),
            "first_output_ms_mean": round(statistics.mean(latencies) * 1000) if latencies else None,
            "runs": runs,
        })
    report["passed"] = not failed
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps({s["name"]: {"pass_rate": s["pass_rate"], "first_output_ms_p50": s["first_output_ms_p50"]}
                      for s in report["scenarios"]}, indent=2))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("IZWI_API_KEY"))
    parser.add_argument("--agent-model", required=True,
                        help="the Realtime model: a chat model id or a profile such as izwi-voice")
    parser.add_argument("--caller-model", help="chat model that plays the caller (default: --chat-model)")
    parser.add_argument("--judge-model", help="chat model that grades (default: --chat-model)")
    parser.add_argument("--chat-model", help="chat model for the caller and the judge (default: --agent-model)")
    parser.add_argument("--scenarios", help="a JSON file of scenarios instead of the built-in ones")
    parser.add_argument("--only", help="run scenarios whose name contains this")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--pass-rate", type=float, default=0.67,
                        help="the least share of a scenario's runs that must pass")
    parser.add_argument("--audio", action="store_true", help="speak the caller's words and hear the agent")
    parser.add_argument("--tts-model", default="Kokoro-82M")
    parser.add_argument("--caller-voice", default="af_heart")
    parser.add_argument("--noise-db", type=float,
                        help="with --audio: white noise this many dB under the caller's speech")
    parser.add_argument("--out", help="write the full report here")
    args = parser.parse_args()
    chat_model = args.chat_model or args.agent_model
    args.caller_model = args.caller_model or chat_model
    args.judge_model = args.judge_model or chat_model
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
