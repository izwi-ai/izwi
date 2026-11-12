#!/usr/bin/env python3
"""Generate the Qwen tool-calling prompt fixture from the published chat templates.

Renders tool-calling conversations with each Qwen chat family's own chat
template (HF's sandboxed settings, `tojson` as `json.dumps(ensure_ascii=False)`,
`{% generation %}` passed through) and records the prompt text. The Rust
prompt builders must reproduce it byte for byte:

    Qwen3      Qwen/Qwen3-4B                 (Hermes JSON calls)
    Qwen3.5    Qwen/Qwen3.5-9B               (XML calls)
    Qwen3.6    Qwen/Qwen3.6-35B-A3B          (XML calls, JSON literals)
    Qwen3.8    Qwen/Qwen3.8-27B              (XML calls, reasoning effort)

Each case stores the request as an OpenAI client sends it (`arguments` as a
JSON string, `reasoning_content` on assistant messages) and the template's
output. Arguments are parsed to mappings before rendering, as vLLM and SGLang
do; the templates iterate them.

Tool definitions and arguments are deliberately NOT in sorted key order, so a
renderer that sorts keys fails.

Usage (jinja2 only). Download the templates first:

    for repo in Qwen3-4B Qwen3.5-9B Qwen3.6-35B-A3B Qwen3.8-27B; do
        hf download Qwen/$repo tokenizer_config.json chat_template.jinja \
            --local-dir "$TEMPLATES/$repo" || true
    done
    python3 -I scripts/fixtures/generate_qwen_tool_prompt_reference.py "$TEMPLATES" \
        crates/izwi-core/tests/fixtures/qwen_tool_prompt_reference.json

The Rust tests are `tool_prompt_reference_tests` in the qwen3, qwen35,
qwen36moe and qwen38 chat modules.
"""
import copy
import hashlib
import json
import os
import sys

import jinja2.exceptions
import jinja2.ext
import jinja2.sandbox

FAMILIES = {
    "qwen3": "Qwen3-4B",
    "qwen35": "Qwen3.5-9B",
    "qwen36": "Qwen3.6-35B-A3B",
    "qwen38": "Qwen3.8-27B",
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Harare."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Current time in a timezone.",
        "parameters": {"type": "object", "properties": {}},
    },
}
SAVE_TOOL = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "Save a note — with tags.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "meta": {"type": "object"},
                "priority": {"type": "integer"},
                "pinned": {"type": "boolean"},
                "due": {"type": ["string", "null"]},
            },
            "required": ["title", "body"],
        },
    },
}


def call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def assistant(content=None, calls=None, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message


def tool(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


SYSTEM = {"role": "system", "content": "You are a concise assistant."}
ASK_WEATHER = {"role": "user", "content": "What is the weather in Harare?"}
WEATHER_CALL = call("call_1", "get_weather", {"unit": "celsius", "city": "Harare"})

# name -> (messages, tools, template kwargs)
CASES = {
    "tools_with_system": ([SYSTEM, ASK_WEATHER], [WEATHER_TOOL, TIME_TOOL], {}),
    "tools_thinking_off": (
        [SYSTEM, ASK_WEATHER],
        [WEATHER_TOOL],
        {"enable_thinking": False},
    ),
    "call_round_trip": (
        [
            SYSTEM,
            ASK_WEATHER,
            assistant(calls=[WEATHER_CALL]),
            tool("call_1", '{"temperature": 24, "sky": "clear"}'),
        ],
        [WEATHER_TOOL],
        {},
    ),
    "text_before_call": (
        [
            SYSTEM,
            ASK_WEATHER,
            assistant("Let me check that.", calls=[WEATHER_CALL]),
            tool("call_1", "24 C, clear"),
        ],
        [WEATHER_TOOL],
        {},
    ),
    "parallel_calls": (
        [
            SYSTEM,
            {"role": "user", "content": "Weather in Harare, and the time?"},
            assistant(calls=[WEATHER_CALL, call("call_2", "get_time", {})]),
            tool("call_1", "24 C, clear"),
            tool("call_2", "14:05"),
        ],
        [WEATHER_TOOL, TIME_TOOL],
        {},
    ),
    "nested_arguments": (
        [
            SYSTEM,
            {"role": "user", "content": "Save a note about the launch."},
            assistant(
                calls=[
                    call(
                        "call_1",
                        "save_note",
                        {
                            "title": "Launch",
                            "body": "Line one\nLine two — done",
                            "tags": ["izwi", "launch"],
                            "meta": {"owner": "ops", "attempt": 2},
                            "priority": 3,
                        },
                    )
                ]
            ),
            tool("call_1", "saved"),
        ],
        [SAVE_TOOL],
        {},
    ),
    "literal_arguments": (
        [
            SYSTEM,
            {"role": "user", "content": "Pin a note with no due date."},
            assistant(
                calls=[
                    call(
                        "call_1",
                        "save_note",
                        {"title": "Pin", "body": "x", "pinned": True, "due": None},
                    )
                ]
            ),
            tool("call_1", "saved"),
        ],
        [SAVE_TOOL],
        {},
    ),
    "reasoning_before_call": (
        [
            SYSTEM,
            ASK_WEATHER,
            assistant(
                calls=[WEATHER_CALL],
                reasoning="The user wants the weather, so I call get_weather.",
            ),
            tool("call_1", "24 C, clear"),
        ],
        [WEATHER_TOOL],
        {},
    ),
    "second_query_after_tool_round": (
        [
            SYSTEM,
            ASK_WEATHER,
            assistant(calls=[WEATHER_CALL], reasoning="Call the weather tool."),
            tool("call_1", "24 C, clear"),
            assistant("It is 24 C and clear in Harare.", reasoning="Report the result."),
            {"role": "user", "content": "And the time there?"},
        ],
        [WEATHER_TOOL, TIME_TOOL],
        {},
    ),
}

# The official Qwen3.5 template writes Python `True`/`None` for boolean and
# null arguments, which Qwen fixed in 3.6. izwi writes JSON literals for every
# family, so this case is recorded for Qwen3.5 but not asserted.
NOT_ASSERTED = {("qwen35", "literal_arguments")}


class Generation(jinja2.ext.Extension):
    """HF's `{% generation %}` block marks assistant spans for training masks;
    rendering passes its body through."""

    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(["name:endgeneration"], drop_needle=True)


def load_template(root, repo):
    jinja_path = os.path.join(root, repo, "chat_template.jinja")
    if os.path.exists(jinja_path):
        with open(jinja_path, encoding="utf-8") as f:
            return f.read()
    with open(os.path.join(root, repo, "tokenizer_config.json"), encoding="utf-8") as f:
        return json.load(f)["chat_template"]


def renderer(source):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols", Generation],
    )
    env.filters["tojson"] = lambda value, indent=None, **_: json.dumps(
        value, ensure_ascii=False, indent=indent
    )

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)

    def render(messages, tools, kwargs):
        # The templates iterate `arguments`; an OpenAI client sends a string.
        messages = copy.deepcopy(messages)
        for message in messages:
            if message.get("content") is None:
                message["content"] = ""
            for tool_call in message.get("tool_calls", []):
                function = tool_call["function"]
                function["arguments"] = json.loads(function["arguments"])
        return template.render(
            messages=messages, tools=tools, add_generation_prompt=True, **kwargs
        )

    return render


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root, out_path = sys.argv[1], sys.argv[2]
    families = {}
    for family, repo in FAMILIES.items():
        source = load_template(root, repo)
        render = renderer(source)
        cases = []
        for name, (messages, tools, kwargs) in CASES.items():
            cases.append(
                {
                    "name": name,
                    "asserted": (family, name) not in NOT_ASSERTED,
                    "messages": messages,
                    "tools": tools,
                    "enable_thinking": kwargs.get("enable_thinking", True),
                    "prompt": render(messages, tools, kwargs),
                }
            )
        families[family] = {
            "repo": f"Qwen/{repo}",
            "template_md5": hashlib.md5(source.encode("utf-8")).hexdigest(),
            "cases": cases,
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"families": families}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
