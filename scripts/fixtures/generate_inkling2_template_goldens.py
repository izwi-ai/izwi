#!/usr/bin/env python3
"""Render the published Inkling 2 chat template for a fixed set of conversations.

Izwi renders Inkling 2 prompts with a hand-written Rust renderer. This script
renders the same conversations with the published `chat_template.jinja`
(through jinja2, with the helpers transformers gives a chat template:
`tojson` with `sort_keys` / `separators`, `raise_exception`, and the
`generation` block) and writes them to `template_cases.json`. The Rust test
`inkling2::prompt::tests::renderer_matches_the_published_template` compares the
two byte for byte.

Usage (needs jinja2 only):

    python3 scripts/fixtures/generate_inkling2_template_goldens.py \
        crates/izwi-core/tests/fixtures/inkling2
"""

import json
import sys
from pathlib import Path

import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class GenerationBlock(Extension):
    """`{% generation %}...{% endgeneration %}` renders its body unchanged."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

    def _render(self, caller):
        return caller()


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"],
        },
    },
}
CLOCK = {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}

CASES = [
    {"name": "one_user_turn", "messages": [{"role": "user", "content": "Hello"}]},
    {
        "name": "system_then_user",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Why is the sky blue?"}],
    },
    {
        "name": "two_system_messages",
        "messages": [
            {"role": "system", "content": "One."},
            {"role": "system", "content": "Two."},
            {"role": "user", "content": "Go"},
        ],
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke — a short one."},
        ],
    },
    {
        "name": "assistant_with_reasoning",
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "reasoning_content": "Simple sum.", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
    },
    {"name": "effort_none", "reasoning_effort": "none", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_minimal", "reasoning_effort": "minimal", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_low", "reasoning_effort": "low", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_medium", "reasoning_effort": "medium", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_high", "reasoning_effort": "high", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_xhigh", "reasoning_effort": "xhigh", "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "effort_number", "reasoning_effort": 0.35, "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "only_system", "messages": [{"role": "system", "content": "Standing by."}]},
    {
        "name": "tools_declared",
        "tools": [WEATHER, CLOCK],
        "messages": [{"role": "user", "content": "Weather in Harare?"}],
    },
    {
        "name": "tool_round_trip",
        "tools": [WEATHER],
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Weather in Harare?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"unit": "c", "city": "Harare"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 27}'},
            {"role": "assistant", "content": "It is 27 °C in Harare."},
            {"role": "user", "content": "Thanks"},
        ],
    },
    {
        "name": "two_tool_calls_with_text",
        "tools": [WEATHER, CLOCK],
        "messages": [
            {"role": "user", "content": "Time and weather?"},
            {
                "role": "assistant",
                "reasoning_content": "Two calls.",
                "content": "Checking both.",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "now", "arguments": {}}},
                    {
                        "id": "b",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Bulawayo", "nested": {"b": 1, "a": [1, 2]}}},
                    },
                ],
            },
            {"role": "tool", "name": "now", "content": "12:00"},
            {"role": "tool", "tool_call_id": "b", "content": "sunny"},
        ],
    },
    {
        "name": "image_parts",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "What animal is this?"}],
            }
        ],
    },
    {
        "name": "text_parts",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Part one."}, {"type": "text", "text": "Part two."}]}
        ],
    },
]


def main() -> None:
    out = Path(sys.argv[1])
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[GenerationBlock])
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string((out / "chat_template.jinja").read_text())
    rendered = []
    for case in CASES:
        variables = {"messages": case["messages"], "tools": case.get("tools"), "add_generation_prompt": True}
        if "reasoning_effort" in case:
            variables["reasoning_effort"] = case["reasoning_effort"]
        rendered.append({**case, "rendered": template.render(**variables)})
        print(case["name"], "->", len(rendered[-1]["rendered"]), "chars")
    (out / "template_cases.json").write_text(json.dumps(rendered, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
