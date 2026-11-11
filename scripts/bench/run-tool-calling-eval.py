#!/usr/bin/env python3
"""Tool-calling evaluation against a running OpenAI-compatible server.

Sends a fixed set of function-calling conversations to `/v1/chat/completions`
and scores what comes back. It measures the serving stack (prompt rendering,
parsing, `tool_choice`), not model intelligence: the cases are easy on purpose,
so a miss usually points at the stack.

Per case it records:
  called        the answer made a tool call when one was expected, and none
                when none was
  valid_name    every call names a declared tool
  valid_schema  every call's arguments are a JSON object that satisfies the
                tool's parameter schema (types, required, enum)
  exact         the arguments equal the expected ones
  stream_equal  the streamed answer, reassembled, makes the same calls as the
                non-streamed one (run with a fixed seed)
  finish        `finish_reason` is `tool_calls` exactly when calls came back

Usage (standard library only):

    python3 scripts/bench/run-tool-calling-eval.py \
        --base-url http://127.0.0.1:8080 --model Qwen3.5-4B-GGUF \
        --out tasks/tool-calling-evidence-2026-11-11/qwen35-4b.json

`--thinking off|on|default` sets `enable_thinking`; `--only` filters cases by
substring; `--skip-forced` leaves out the `tool_choice` required / named /
`parallel_tool_calls` cases for servers that refuse them.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def fn(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


WEATHER = fn(
    "get_weather",
    "Get the current weather for a city.",
    {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    ["city"],
)
TIME = fn("get_time", "Get the current time in a city.", {"city": {"type": "string"}}, ["city"])
SEARCH = fn(
    "search_contacts",
    "Search the address book.",
    {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
        "include_archived": {"type": "boolean"},
    },
    ["query"],
)
LOOKUP = fn(
    "lookup_order",
    "Look up an order by its id. The id is a string of digits.",
    {"order_id": {"type": "string"}},
    ["order_id"],
)
SET_FLAG = fn(
    "set_label",
    "Attach a text label to an item.",
    {"item": {"type": "string"}, "label": {"type": "string"}},
    ["item", "label"],
)
WRITE = fn(
    "write_file",
    "Write text to a file.",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
EVENT = fn(
    "create_event",
    "Create a calendar event.",
    {
        "title": {"type": "string"},
        "attendees": {"type": "array", "items": {"type": "string"}},
        "location": {
            "type": "object",
            "properties": {"room": {"type": "string"}, "floor": {"type": "integer"}},
        },
    },
    ["title", "attendees"],
)
ALL = [WEATHER, TIME, SEARCH, LOOKUP, SET_FLAG, WRITE, EVENT]


def user(text):
    return {"role": "user", "content": text}


def expect(name, **arguments):
    return {"name": name, "arguments": arguments}


POEM = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(1, 41))

# name, messages, tools, expected calls (None: any), request extras
CASES = [
    ("single/weather", [user("What's the weather in Harare?")], [WEATHER], [expect("get_weather", city="Harare")], {}),
    ("single/weather_unit", [user("Weather in Oslo in fahrenheit please.")], [WEATHER], [expect("get_weather", city="Oslo", unit="fahrenheit")], {}),
    ("single/time", [user("What time is it in Tokyo?")], [WEATHER, TIME], [expect("get_time", city="Tokyo")], {}),
    ("single/pick_among_many", [user("What time is it in Lagos right now?")], ALL, [expect("get_time", city="Lagos")], {}),
    ("single/integer_and_bool", [user("Search my contacts for 'Moyo', at most 5 results, and include archived ones.")], [SEARCH], [expect("search_contacts", query="Moyo", limit=5, include_archived=True)], {}),
    ("single/nested", [user("Create an event titled Standup with attendees Ana and Ben in room Baobab on floor 3.")], [EVENT], [expect("create_event", title="Standup", attendees=["Ana", "Ben"], location={"room": "Baobab", "floor": 3})], {}),
    ("single/system_prompt", [{"role": "system", "content": "You are a terse assistant. Use tools when they help."}, user("Weather in Nairobi?")], [WEATHER], [expect("get_weather", city="Nairobi")], {}),
    ("single/non_ascii", [user("What's the weather in São Paulo?")], [WEATHER], [expect("get_weather", city="São Paulo")], {}),
    ("string_lookalike/digits", [user("Look up order 00123.")], [LOOKUP], [expect("lookup_order", order_id="00123")], {}),
    ("string_lookalike/number", [user("Look up order 4521.")], [LOOKUP], [expect("lookup_order", order_id="4521")], {}),
    ("string_lookalike/true", [user("Set the label of item A7 to the word true.")], [SET_FLAG], [expect("set_label", item="A7", label="true")], {}),
    ("string_lookalike/null", [user("Set the label of item B2 to the word null.")], [SET_FLAG], [expect("set_label", item="B2", label="null")], {}),
    ("long_argument/poem", [user("Write this text to notes/poem.txt exactly as given:\n" + POEM)], [WRITE], [expect("write_file", path="notes/poem.txt", content=POEM)], {"max_tokens": 2048}),
    ("parallel/two_cities", [user("What's the weather in Harare and in Lusaka?")], [WEATHER], [expect("get_weather", city="Harare"), expect("get_weather", city="Lusaka")], {}),
    ("parallel/two_tools", [user("Tell me the weather and the current time in Cairo.")], [WEATHER, TIME], [expect("get_weather", city="Cairo"), expect("get_time", city="Cairo")], {}),
    ("no_call/chitchat", [user("Say hello in one short sentence.")], [WEATHER, TIME], [], {}),
    ("no_call/arithmetic", [user("What is 2 + 2? Answer with the number only.")], [WEATHER, SEARCH], [], {}),
    ("no_call/unrelated_tools", [user("Name the capital of France in one word.")], ALL, [], {}),
    ("no_call/talk_about_tools", [user("Without calling anything, list in words which tools you have.")], [WEATHER, TIME], [], {}),
    ("no_call/tool_choice_none", [user("What's the weather in Harare?")], [WEATHER], [], {"tool_choice": "none"}),
    (
        "multi_turn/answer_from_result",
        [
            user("What's the weather in Harare?"),
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Harare\"}"}}]},
            {"role": "tool", "tool_call_id": "call_a", "content": "{\"temperature_c\": 24, \"sky\": \"clear\"}"},
        ],
        [WEATHER],
        [],
        {"expect_text": "24"},
    ),
    (
        "multi_turn/second_call",
        [
            user("What's the weather in Harare?"),
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Harare\"}"}}]},
            {"role": "tool", "tool_call_id": "call_a", "content": "24 C, clear"},
            {"role": "assistant", "content": "It is 24 C and clear in Harare."},
            user("And what time is it there?"),
        ],
        [WEATHER, TIME],
        [expect("get_time", city="Harare")],
        {},
    ),
    (
        "multi_turn/parallel_results",
        [
            user("Weather in Harare and Lusaka?"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Harare\"}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Lusaka\"}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "24 C"},
            {"role": "tool", "tool_call_id": "call_b", "content": "31 C"},
        ],
        [WEATHER],
        [],
        {"expect_text": "31"},
    ),
    ("forced/required", [user("Hello there.")], [WEATHER, TIME], None, {"tool_choice": "required", "forced": True}),
    ("forced/required_relevant", [user("What's the weather in Harare?")], [WEATHER, TIME], [expect("get_weather", city="Harare")], {"tool_choice": "required", "forced": True}),
    ("forced/named", [user("Harare, please.")], [WEATHER, TIME], [expect("get_time", city="Harare")], {"tool_choice": {"type": "function", "function": {"name": "get_time"}}, "forced": True}),
    ("forced/named_against_intent", [user("What's the weather in Harare?")], [WEATHER, TIME], [expect("get_time", city="Harare")], {"tool_choice": {"type": "function", "function": {"name": "get_time"}}, "forced": True}),
    ("forced/parallel_off", [user("What's the weather in Harare and in Lusaka?")], [WEATHER], None, {"parallel_tool_calls": False, "forced": True, "max_calls": 1, "min_calls": 1}),
]


def post(base_url, payload, timeout):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def complete(base_url, payload, timeout):
    with post(base_url, payload, timeout) as response:
        body = json.load(response)
    choice = body["choices"][0]
    message = choice["message"]
    calls = [
        {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
        for c in message.get("tool_calls") or []
    ]
    return {"content": message.get("content") or "", "calls": calls, "finish": choice.get("finish_reason")}


def complete_stream(base_url, payload, timeout):
    content, calls, finish = "", [], None
    with post(base_url, dict(payload, stream=True), timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(json.dumps(event["error"]))
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content += delta.get("content") or ""
                for fragment in delta.get("tool_calls") or []:
                    index = fragment["index"]
                    while len(calls) <= index:
                        calls.append({"name": "", "arguments": ""})
                    function = fragment.get("function", {})
                    calls[index]["name"] += function.get("name") or ""
                    calls[index]["arguments"] += function.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
    return {"content": content, "calls": calls, "finish": finish}


def schema_errors(value, schema, path="$"):
    """A small JSON Schema check: type, required, enum, properties, items."""
    errors = []
    kinds = schema.get("type")
    if kinds is not None:
        kinds = kinds if isinstance(kinds, list) else [kinds]
        checks = {
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "null": lambda v: v is None,
        }
        if not any(checks[kind](value) for kind in kinds if kind in checks):
            return [f"{path}: expected {'/'.join(kinds)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required {key!r}")
        for key, item in value.items():
            if key in schema.get("properties", {}):
                errors.extend(schema_errors(item, schema["properties"][key], f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(schema_errors(item, schema["items"], f"{path}[{index}]"))
    return errors


def parsed_calls(result):
    out = []
    for call in result["calls"]:
        try:
            arguments = json.loads(call["arguments"])
        except (TypeError, ValueError):
            arguments = None
        out.append({"name": call["name"], "arguments": arguments})
    return out


def normalized(text):
    return " ".join(str(text).split()).casefold()


def arguments_match(got, want):
    """Expected arguments must be present and equal (strings compared ignoring
    case and whitespace runs); extra optional arguments are allowed."""
    if not isinstance(got, dict):
        return False
    for key, value in want.items():
        if key not in got:
            return False
        if isinstance(value, str) and isinstance(got[key], str):
            if normalized(got[key]) != normalized(value):
                return False
        elif got[key] != value or type(got[key]) is not type(value):
            return False
    return True


def score(case, result, stream_result):
    name, _messages, tools, expected, extras = case
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    calls = parsed_calls(result)
    row = {"case": name, "calls": calls, "finish": result["finish"], "problems": []}
    if expected is None:
        low, high = extras.get("min_calls", 1), extras.get("max_calls", 99)
        row["called"] = low <= len(calls) <= high
    else:
        row["called"] = bool(calls) == bool(expected)
    row["valid_name"] = all(c["name"] in schemas for c in calls)
    row["valid_schema"] = True
    for call in calls:
        if not isinstance(call["arguments"], dict):
            row["valid_schema"] = False
            row["problems"].append(f"{call['name']}: arguments are not a JSON object")
        elif call["name"] in schemas:
            errors = schema_errors(call["arguments"], schemas[call["name"]])
            if errors:
                row["valid_schema"] = False
                row["problems"].extend(errors)
    if expected is None:
        row["exact"] = row["called"]
    elif not expected:
        row["exact"] = not calls
        wanted = extras.get("expect_text")
        if wanted and wanted not in result["content"]:
            row["exact"] = False
            row["problems"].append(f"answer does not mention {wanted!r}")
    else:
        remaining = list(calls)
        row["exact"] = len(calls) == len(expected)
        for want in expected:
            match = next(
                (c for c in remaining if c["name"] == want["name"] and arguments_match(c["arguments"], want["arguments"])),
                None,
            )
            if match is None:
                row["exact"] = False
                row["problems"].append(f"missing call {want}")
            else:
                remaining.remove(match)
    row["finish_ok"] = (result["finish"] == "tool_calls") == bool(calls)
    if stream_result is not None:
        row["stream_equal"] = parsed_calls(stream_result) == calls and normalized(
            stream_result["content"]
        ) == normalized(result["content"])
        if not row["stream_equal"]:
            row["stream_calls"] = parsed_calls(stream_result)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out")
    parser.add_argument("--only", default="")
    parser.add_argument("--thinking", choices=["on", "off", "default"], default="off")
    parser.add_argument("--skip-forced", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    rows = []
    for case in CASES:
        name, messages, tools, _expected, extras = case
        if args.only not in name or (args.skip_forced and extras.get("forced")):
            continue
        payload = {
            "model": args.model,
            "messages": messages,
            "tools": tools,
            "temperature": 0,
            "seed": 7,
            "max_tokens": extras.get("max_tokens", args.max_tokens),
        }
        if args.thinking != "default":
            payload["enable_thinking"] = args.thinking == "on"
        for key in ("tool_choice", "parallel_tool_calls"):
            if key in extras:
                payload[key] = extras[key]
        started = time.time()
        try:
            result = complete(args.base_url, payload, args.timeout)
            stream_result = None if args.no_stream else complete_stream(args.base_url, payload, args.timeout)
            row = score(case, result, stream_result)
        except urllib.error.HTTPError as error:
            row = {"case": name, "error": f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:300]}"}
        except Exception as error:  # noqa: BLE001 - one failed case must not end the run
            row = {"case": name, "error": f"{type(error).__name__}: {error}"}
        row["seconds"] = round(time.time() - started, 2)
        rows.append(row)
        flags = "".join(
            "." if row.get(key, True) else letter
            for key, letter in (("called", "C"), ("valid_name", "N"), ("valid_schema", "S"), ("exact", "E"), ("finish_ok", "F"), ("stream_equal", "X"))
        )
        print(f"{'ERR   ' if 'error' in row else flags}  {name:36s} {row['seconds']:7.2f}s  {row.get('error') or '; '.join(row['problems'])}")

    scored = [row for row in rows if "error" not in row]
    summary = {"model": args.model, "thinking": args.thinking, "cases": len(rows), "errors": len(rows) - len(scored)}
    for key in ("called", "valid_name", "valid_schema", "exact", "finish_ok", "stream_equal"):
        values = [row[key] for row in scored if key in row]
        summary[key] = f"{sum(values)}/{len(values)}"
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
