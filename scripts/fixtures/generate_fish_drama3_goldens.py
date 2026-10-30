#!/usr/bin/env python3
"""Generate the Fish Audio Drama 3 prompt and chunking goldens from upstream fish-speech.

What it does
------------
1. Puts `--fish-speech` (a fish-speech checkout at 0f8a12e8, "drama-3 small
   updates (#1368)") first on `sys.path` and imports, unmodified:
   * `fish_speech.models.text2semantic.fish_omni.generate`:
     `build_drama3_prompt`, `with_speaker_tag`, `validate_drama3_text`,
     `Drama3PromptTooLongError` and the prompt constants;
   * `fish_speech.models.text2semantic.fish_omni.chunking`: `_protected_pieces`,
     `split_by_length`, `parse_speaker_segments`, `split_drama3_text`,
     `Drama3History` and the chunking constants;
   * `fish_speech.models.text2semantic.fish_omni.config.FishOmniConfig`, built
     from `--checkpoint`'s `config.json`;
   * `fish_speech.models.text2semantic.inference.build_drama3_chunk_prompt`
     (the history policy of the chunk loop). If `inference.py` does not import
     (it also pulls in hydra, omegaconf and click), its 22 lines are copied
     verbatim below and the fixture records that.
   The checkout's HEAD must be 0f8a12e8 (`--skip-commit-check` records an
   unverified commit instead).
2. Loads `--checkpoint/tokenizer.json` with the `tokenizers` library (the one
   upstream uses). `config.json` and `tokenizer.json` must hash to the pinned
   `fishaudio/drama-3@78e5ab79` files.
3. Runs the cases below through upstream. A recording proxy around the
   tokenizer captures every literal `build_drama3_prompt` encodes, and a
   recording wrapper around `build_drama3_prompt` captures the history each
   chunk prompt kept. Row budgets other than 8192 are set by assigning
   `generate.DRAMA3_MAX_PROMPT_ROWS`, which upstream reads at call time (as its
   own tests do). Codes are deterministic and written into the fixtures.
4. Writes `prompt_goldens.json` and `chunk_goldens.json` to `--out`. izwi's
   `fish_drama3::prompt::tests::upstream_prompt_goldens`,
   `fish_drama3::chunking::tests::upstream_chunk_goldens` and
   `fish_drama3::chunking::tests::upstream_history_selection_goldens` replay
   them (ignored until the files exist; the first and last also need
   `IZWI_FISH_DRAMA3_TOKENIZER_PATH`).

Nothing here downloads, loads weights or runs a model: only the prompt
builder, the chunker and the tokenizer run.

Usage (CPU only; any Python >= 3.10 that can import fish-speech's
`fish_omni` package: torch, loguru, tqdm, safetensors, tokenizers; `-I` keeps
the current directory and user site-packages off the path, so use the Python
of the environment fish-speech is installed in):

    python3 -I scripts/fixtures/generate_fish_drama3_goldens.py \\
        --fish-speech /path/to/fish-speech \\
        --checkpoint /path/to/drama-3 \\
        [--out crates/izwi-core/tests/fixtures/fish_drama3] \\
        [--skip-commit-check]

`--checkpoint` is any directory holding the pinned `config.json` and
`tokenizer.json` (no weights are read); on the izwi dev Mac,
`tasks/fish-drama3-evidence-2026-10-29/hf-78e5ab79`.

Error records are `{"type": <upstream exception class>, "message": str(error)}`.
Codes are `[codebooks][frames]` integer lists.

prompt_goldens.json (`schema` = "izwi.fish_drama3.prompt_goldens", `schema_version` = 1):

{
  "schema", "schema_version", "generator", "provenance": {...},
  "constants": {"tts_instruction", "reference_instruction", "message_end",
                "generation_start", "structural_tokens": [str],
                "max_speaker_id", "unicode_white_space": [code point],
                "max_prompt_rows"},
  "literals": [{"text", "ids": [int]}],          # tokenizer.encode(text).ids
  "validate": [{"text", "error": null | error}], # validate_drama3_text(text, "text")
  "with_speaker_tag": [{"text", "tagged"}],
  "cases": [{"name", "text",
             "references": [{"transcript", "codes"}],
             "history": [{"text", "codes"}],
             "max_prompt_rows",
             "segments": [str],                 # literals encoded, in order
             "rows": null | [[main, c0, ..., c9]],
             "error": null | error}]
}

chunk_goldens.json (`schema` = "izwi.fish_drama3.chunk_goldens", `schema_version` = 1):

{
  "schema", "schema_version", "generator", "provenance": {...},
  "constants": {"chunk_bytes", "max_speakers", "chunk_max_new_tokens",
                "recent_turns", "sentence_ends", "clause_marks", "spaces"},
  "corpus": {name: text},
  "protected_pieces": [{"text", "pieces": [[str, bool]]}],
  "split_by_length": [{"name", "length", "chunks": [str]}],      # name: corpus key
  "parse_speaker_segments": [{"text", "segments": [[speaker, body]]}],
  "split_text": [{"name", "max_bytes", "max_speakers", "chunks": [str]}],
  "history_candidates": [{"recent_turns", "pushed": [str],
                          "candidates": [[str]]}],
  "history_selection": [{"name", "references": [{"transcript", "codes"}],
                         # one chunk per step; its codes are pushed into the
                         # history after the step unless null (the chunk
                         # ended before a frame), whatever the step did
                         "chunks": [{"text", "codes": null | codes}],
                         "runs": [{"max_prompt_rows",
                                   "steps": [{"history": null | [chunk index],
                                              "rows": null | int,
                                              "error": null | error}]}]}]
}
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

FISH_SPEECH_COMMIT = "0f8a12e8a651062fdee9c33a894bdccd3a8513f1"
CHECKPOINT_REVISION = "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
PINNED_SHA256 = {
    "config.json": "aea53f0f2b9a15023fc0f8eb0f42360663671c99814037bada6867d508010111",
    "tokenizer.json": "0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "crates/izwi-core/tests/fixtures/fish_drama3"
GENERATOR = "scripts/fixtures/generate_fish_drama3_goldens.py"
CODEBOOKS = 10
CODEBOOK_SIZE = 4096
RESIDUAL_CODEC_SIZE = 1024

EN0 = "We asked over twenty different people, and they all said it was his."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fish-speech", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-commit-check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fish_speech_commit(root: Path, skip: bool) -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        if skip:
            return "unverified"
        sys.exit(f"{root} is not a git checkout ({error}); pass --skip-commit-check")
    if head != FISH_SPEECH_COMMIT and not skip:
        sys.exit(f"{root} is at {head}, expected {FISH_SPEECH_COMMIT}")
    return head if head == FISH_SPEECH_COMMIT else f"unverified ({head})"


def make_codes(frames: int, seed: int, residual: int = RESIDUAL_CODEC_SIZE) -> list[list[int]]:
    """[10][frames]: codebook 0 below 4096, the rest below `residual`."""
    return [
        [
            (seed * 7919 + k * 131 + f * 977 + k * f * 31) % (CODEBOOK_SIZE if k == 0 else residual)
            for f in range(frames)
        ]
        for k in range(CODEBOOKS)
    ]


class RecordingTokenizer:
    """The real tokenizer, recording each literal it encodes."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.parts: list[str] = []

    def token_to_id(self, token: str):
        return self.inner.token_to_id(token)

    def encode(self, text: str, *args, **kwargs):
        self.parts.append(text)
        return self.inner.encode(text, *args, **kwargs)


class HistoryRecorder:
    """Wraps build_drama3_prompt; remembers the history of the last build that succeeded."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.last: list | None = None

    def __call__(self, *args, **kwargs):
        rows = self.inner(*args, **kwargs)
        self.last = list(kwargs["history"] if "history" in kwargs else (args[5] if len(args) > 5 else ()))
        return rows


def describe(error: Exception) -> dict:
    return {"type": type(error).__name__, "message": str(error)}


def dump(obj: dict, path: Path) -> None:
    """One top-level key per line; each case of a case list on its own line."""
    lines = ["{"]
    items = list(obj.items())
    for index, (key, value) in enumerate(items):
        comma = "," if index + 1 < len(items) else ""
        compact = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f"  {json.dumps(key)}: [")
            for position, item in enumerate(value):
                tail = "," if position + 1 < len(value) else ""
                lines.append(f"    {compact(item)}{tail}")
            lines.append(f"  ]{comma}")
        else:
            lines.append(f"  {json.dumps(key)}: {compact(value)}{comma}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

LITERALS = [
    "<|im_start|>system\n",
    "convert the provided text to speech",
    "convert the provided text to speech reference to the following:",
    "<|im_end|>\n",
    "<|im_start|>user\n",
    "<|im_start|>assistant\n<|voice|>",
    "<|speaker:0|> Hello, how are you?",
    f"\n\n[Reference 1]\nText: <|speaker:0|> {EN0}\nSpeech: ",
    "\n\n[Reference 2]\nText: <|speaker:1|> Always.\nSpeech: ",
    "<|speaker:0|> 今天天气很好，我们去公园散步。",
    "<|speaker:1|> Ready? <|speaker:0|> Always.",
    "<|speaker:0|> [sighs] <whisper>not now</whisper> Pi is 3.14.",
    "<|speaker:0|>   reference",
    "<|speaker:0|>hello",
    "",
    " ",
    "  <|speaker:0|>",
    "<|speaker:0|> café café",
    "<|speaker:0|> Ça va? Très bien… 😀👍🏽",
    "<|speaker:0|> こんにちは、元気ですか？",
    "<|speaker:0|> 안녕하세요. 반갑습니다!",
    "<|speaker:0|> مرحبا، كيف حالك؟",
    "<|speaker:0|> नमस्ते, आप कैसे हैं?",
    "<|speaker:0|> 12345.678 and 3,000,000 and ٣.١٤",
    "<|speaker:0|> line one\r\nline two\n\n\tindented",
    "<|speaker:0|> 　ideographic space",
    "<|speaker:3|> <|fim_prefix|> <|speaker_change|> <|speaker:u3|> <|endofturn|>",
    "<|speaker:0|> [with growing excitement] [language: fr] [audio tokens: 120] Bonjour!",
    "<|speaker:15|>x<|speaker:14|>y",
    "<|semantic:0|><|semantic:4095|>",
    "\u001c\u001d\u001e\u001f",
]

VALIDATE = [
    "hello",
    " ",
    "　\n",
    "\u001c",
    "hello <|voice|>",
    "<|speaker:16|> hello",
    "<|speaker:00|> hi",
    "<|speaker:01|> hi",
    "<|speaker:x|> hi",
    "<|speaker:u0|> hi",
    "<|speaker hi",
    "<|speaker_change|> hi",
    "a <|semantic:12|>",
    "a <|semantic:",
    "<|im_start|>",
    "x<|im_end|>",
    "<|text|>",
    "<|interleave|>",
    "<|speaker:15|> hi\n<|speaker:0|> yes",
    "<|fim_prefix|> hi",
    "<|endofturn|> hi",
    "[sighs] <whisper>hi</whisper>",
    "<|speaker:9999999999999999999999|> big",
]

WITH_SPEAKER_TAG = [
    "hello",
    " \t hi",
    "<|speaker:3|>hi",
    " <|speaker:2|> x",
    "\u001c<|speaker:1|> x",
    "​<|speaker:1|> x",
    "<|speaker:01|> x",
    "<|speaker:|> x",
    "\n\n<|speaker:15|>",
    "",
]


def prompt_cases() -> list[dict]:
    def ref(transcript, frames, seed, codes=None):
        return {"transcript": transcript, "codes": codes or make_codes(frames, seed)}

    def turn(text, frames, seed, residual=CODEBOOK_SIZE, codes=None):
        return {"text": text, "codes": codes or make_codes(frames, seed, residual)}

    def case(name, text, references=(), history=(), max_prompt_rows=8192):
        return {
            "name": name,
            "text": text,
            "references": list(references),
            "history": list(history),
            "max_prompt_rows": max_prompt_rows,
        }

    out_of_range = make_codes(3, 5)
    out_of_range[3][1] = CODEBOOK_SIZE
    nine_codebooks = make_codes(2, 6)[:9]
    return [
        case("plain", "Hello, how are you?"),
        case("plain_tagged", "<|speaker:3|> Hi there."),
        case("plain_leading_space", " \t hi"),
        case("plain_unicode_space_then_tag", " <|speaker:2|> Indented."),
        case("plain_separator_then_tag", "\u001c<|speaker:1|> sep"),
        case("one_reference", "Hello, how are you?", [ref(EN0, 6, 1)]),
        case(
            "one_reference_tagged",
            "<|speaker:2|> Same voice.",
            [ref("<|speaker:2|>ref text", 3, 2)],
        ),
        case(
            "reference_leading_space",
            "<|speaker:0|>hello",
            [ref("  <|speaker:0|> reference", 2, 3)],
        ),
        case(
            "reference_unicode_space",
            "<|speaker:4|> y",
            [ref("　 <|speaker:4|>x", 2, 4)],
        ),
        case(
            "two_references_two_speakers",
            "<|speaker:0|> Ready? <|speaker:1|> Always.",
            [ref("<|speaker:0|> First voice.", 4, 5), ref("<|speaker:1|> Second voice.", 3, 6)],
        ),
        case(
            "two_references_one_speaker",
            "One voice.",
            [ref("First clip.", 2, 7), ref("Second clip.", 2, 8)],
        ),
        case("cjk", "今天天气很好，我们去公园散步。", [ref("这是参考音频。", 3, 9)]),
        case(
            "inline_tags",
            "[sighs] <whisper>not now</whisper> Pi is 3.14. [laughs] <|fim_prefix|>",
        ),
        case(
            "history_two_turns",
            "Now the third line.",
            history=[turn("First line.", 4, 10), turn("<|speaker:1|> Second line.", 3, 11)],
        ),
        case(
            "references_and_history",
            "<|speaker:1|> after",
            [ref(EN0, 3, 12)],
            [turn("<|speaker:1|> before", 2, 13)],
        ),
        case("speaker_15", "<|speaker:15|> hi\n<|speaker:0|> yes"),
        case("fits_exactly", "Hello, how are you?", max_prompt_rows=27),
        case("too_long", "Hello, how are you?", max_prompt_rows=26),
        case(
            "too_long_with_reference",
            "Hello, how are you?",
            [ref(EN0, 40, 14)],
            max_prompt_rows=64,
        ),
        case("error_voice", "hello <|voice|>"),
        case("error_speaker_16", "<|speaker:16|> hello"),
        case("error_speaker_01", "<|speaker:01|> hi"),
        case("error_speaker_x", "<|speaker:x|> hi"),
        case("error_speaker_u0", "<|speaker:u0|> hi"),
        case("error_blank", "　 \n"),
        case("error_semantic", "a <|semantic:12|>"),
        case("error_reference_im_end", "hi", [ref("x<|im_end|>", 2, 15)]),
        case("error_reference_code_range", "hi", [ref("ref", 0, 0, codes=out_of_range)]),
        case("error_reference_codebooks", "hi", [ref("ref", 0, 0, codes=nine_codebooks)]),
        case("error_history_text", "hi", history=[turn("<|speaker:16|> no", 2, 16)]),
        case(
            "error_history_code_range",
            "hi",
            history=[turn("ok", 0, 0, codes=[[CODEBOOK_SIZE]] * CODEBOOKS)],
        ),
    ]


LONG_ENGLISH = (
    "Mr. Smith arrived at 3.14 p.m. sharp; nobody expected him. The room went quiet, "
    "and then, all at once, everyone started talking: about the weather, about the "
    "price of bread ($2.50, up from $1.99!), about the strange letter… Wait... what?! "
    "He raised a hand.\nSilence.\n\n\"I have news,\" he said, \"and you will not like it.\" "
    "The council had voted, e.g. seven to two, to close the old bridge for good. "
    "Repairs would cost 4.5 million, which the town did not have; the mayor had tried, "
    "truly tried, to find the money, but every door had closed. "
) * 3
LONG_CJK = (
    "今天天气很好，我们去公园散步。公园里有很多人：有的在跑步，有的在打太极拳、有的在下棋！"
    "你想喝茶吗？我们可以去那家新开的茶馆；听说那里的龙井很好喝……"
    "「本当に？」と彼女は言った。それから、静かに笑った。"
) * 4
TAGGED = (
    "She leaned in. <whisper>" + "Do not tell anyone about this, not a single soul. " * 8
    + "</whisper> [laughs] Then louder: [with growing excitement] we did it! "
    "<emphasis level=\"strong\">Really</emphasis> did it. <|fim_prefix|> stays whole, "
    "and so does [language: en]. An unclosed <b tag and a stray ] bracket, plus [nested [square] brackets]. "
) * 2
DECIMALS = (
    "Version 2.5.1 costs 3.99 dollars. In 1999. The value was .5 and 5. as well. "
    "Arabic-Indic ٣.١٤ and superscript x².³ y. Ten.Twenty. 1.2.3.4.5.6.7.8.9.10. "
) * 5
DIALOGUE = "".join(
    f"<|speaker:{i % 5}|>"
    + ("Line %d of the scene goes here, with a little more to say. " % i) * (1 + (i % 4) * 3)
    for i in range(24)
)
WORDS = " ".join(["unpunctuated"] * 80)
UNBROKEN = "字" * 400
EMOJI = "😀" * 120
SPACES = "word　" * 60 + "end \u001c sep " * 20
LEADING = "Narrator text before any tag. <|speaker:1|> One. <|speaker:2|>Two.<|speaker:1|> Three " * 6
EXACT_300 = "a" * 299 + "."
EXACT_301 = "a" * 300 + "."

SPLIT_CORPUS = {
    "long_english": LONG_ENGLISH,
    "long_cjk": LONG_CJK,
    "tagged": TAGGED,
    "decimals": DECIMALS,
    "words": WORDS,
    "unbroken": UNBROKEN,
    "emoji": EMOJI,
    "spaces": SPACES,
    "exact_300": EXACT_300,
    "exact_301": EXACT_301,
    "short": "  Hello there.  ",
    "blank": " \n　 ",
    "empty": "",
    "separator_only": "\u001c",
}
TEXT_CORPUS = {
    **SPLIT_CORPUS,
    "dialogue_5_speakers": DIALOGUE,
    "leading_text_then_tags": LEADING,
    "fourth_speaker": "<|speaker:0|>A1<|speaker:1|>B1<|speaker:0|>A2<|speaker:2|>C1<|speaker:3|>D1",
    "noncanonical_tags": "<|speaker:01|> a. <|speaker:1|> b. <|speaker:x|> c. <|speaker:20|> d.",
    "blank_segments": "<|speaker:0|> \n<|speaker:1|>Hi<|speaker:2|> Yo<|speaker:3|>　",
    "tagged_long_segment": "<|speaker:0|>" + "This sentence is long enough. " * 30 + "<|speaker:0|>Short.",
}
PROTECTED = [
    "a <|x|> [b] <w a=1>c</w> <a->d</a> e",
    "<b>x [y",
    "[a [b] c]",
    "<i>a</i>b</i>",
    "<em>x</e>",
    "<|a>b|>",
    "<aé>x</aé>",
    "<aா>x</aா>",
    "<á>x</a>",
    "<whisper>a.\nb.</whisper>!",
    "<x-y_z9>q</x-y_z9> <x->r</x-> <-x>s</-x> <9>t</9>",
    "<a><a>inner</a></a>",
    "<|speaker:0|><|unterminated <|ok|> [|] [] <|> <||>",
    "<tag attr='>'>v</tag>",
    "<Ⅷ>roman</Ⅷ> <aⅧ>roman</aⅧ> <a‿b>tie</a‿b>",
    TAGGED,
]
PARSE = [
    "intro <|speaker:1|>Hi<|speaker:2|> Yo",
    " <|speaker:01|>a<|speaker:3|> \n",
    "<|speaker:3|> \n<|speaker:x|>b",
    "no tags at all",
    "<|speaker:7|>",
    LEADING,
]


def main() -> None:
    args = parse_args()
    root = args.fish_speech.resolve()
    commit = fish_speech_commit(root, args.skip_commit_check)
    checkpoint = args.checkpoint.resolve()
    hashes = {}
    for name, expected in PINNED_SHA256.items():
        found = sha256(checkpoint / name)
        if found != expected:
            sys.exit(f"{checkpoint / name} hashes to {found}; the pinned file is {expected}")
        hashes[name] = found

    sys.path.insert(0, str(root))
    import torch
    import tokenizers
    from tokenizers import Tokenizer

    generate = importlib.import_module("fish_speech.models.text2semantic.fish_omni.generate")
    chunking = importlib.import_module("fish_speech.models.text2semantic.fish_omni.chunking")
    config_module = importlib.import_module("fish_speech.models.text2semantic.fish_omni.config")
    errors = importlib.import_module("fish_speech.models.text2semantic.errors")
    InvalidInputError = errors.InvalidInputError

    recorder = HistoryRecorder(generate.build_drama3_prompt)
    try:
        inference = importlib.import_module("fish_speech.models.text2semantic.inference")
        inference.build_drama3_prompt = recorder
        build_chunk_prompt = inference.build_drama3_chunk_prompt
        chunk_policy = "fish_speech.models.text2semantic.inference.build_drama3_chunk_prompt"
    except ImportError as error:
        print(f"inference.py does not import ({error}); using its verbatim copy", file=sys.stderr)

        # inference.py:1021-1042 @ 0f8a12e8, verbatim (build_drama3_prompt is the recorder).
        def build_chunk_prompt(model, text, prompt_text, prompt_tokens, history):
            """The prompt for one chunk with as much history as fits DRAMA3_MAX_PROMPT_ROWS."""
            error = None
            for turns in history.candidates():
                try:
                    return recorder(
                        text,
                        model.tokenizer,
                        model.config,
                        prompt_text,
                        prompt_tokens,
                        history=turns,
                    )
                except generate.Drama3PromptTooLongError as too_long:
                    error = too_long
            raise error

        chunk_policy = "verbatim copy of inference.py:1021-1042"

    with (checkpoint / "config.json").open(encoding="utf-8") as handle:
        config = config_module.FishOmniConfig.from_dict(json.load(handle))
    base_tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    default_rows = generate.DRAMA3_MAX_PROMPT_ROWS

    def tensor(codes):
        return torch.tensor(codes, dtype=torch.long)

    def with_rows(limit, run):
        generate.DRAMA3_MAX_PROMPT_ROWS = limit
        try:
            return run()
        finally:
            generate.DRAMA3_MAX_PROMPT_ROWS = default_rows

    provenance = {
        "fish_speech_commit": commit,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "sha256": hashes,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "torch": torch.__version__,
        "tokenizers": tokenizers.__version__,
        "unicode": unicodedata.unidata_version,
        "chunk_policy": chunk_policy,
    }

    # -- prompt goldens -----------------------------------------------------
    cases = []
    for spec in prompt_cases():
        tokenizer = RecordingTokenizer(base_tokenizer)
        record = dict(spec)
        try:
            rows = with_rows(
                spec["max_prompt_rows"],
                lambda: generate.build_drama3_prompt(
                    spec["text"],
                    tokenizer,
                    config,
                    [r["transcript"] for r in spec["references"]],
                    [tensor(r["codes"]) for r in spec["references"]],
                    history=[(t["text"], tensor(t["codes"])) for t in spec["history"]],
                ),
            )
            record.update(rows=rows.tolist(), error=None)
        except InvalidInputError as error:
            record.update(rows=None, error=describe(error))
        record["segments"] = tokenizer.parts
        cases.append(record)

    validate = []
    for text in VALIDATE:
        try:
            generate.validate_drama3_text(text, "text")
            validate.append({"text": text, "error": None})
        except InvalidInputError as error:
            validate.append({"text": text, "error": describe(error)})

    prompt_goldens = {
        "schema": "izwi.fish_drama3.prompt_goldens",
        "schema_version": 1,
        "generator": GENERATOR,
        "provenance": provenance,
        "constants": {
            "tts_instruction": generate.DRAMA3_TTS_INSTRUCTION,
            "reference_instruction": generate.DRAMA3_REFERENCE_INSTRUCTION,
            "message_end": generate.DRAMA3_MESSAGE_END,
            "generation_start": generate.DRAMA3_GENERATION_START,
            "structural_tokens": list(generate.DRAMA3_STRUCTURAL_TOKENS),
            "max_speaker_id": generate.MAX_SPEAKER_ID,
            "unicode_white_space": [ord(c) for c in generate.UNICODE_WHITE_SPACE],
            "max_prompt_rows": default_rows,
        },
        "literals": [{"text": t, "ids": base_tokenizer.encode(t).ids} for t in LITERALS],
        "validate": validate,
        "with_speaker_tag": [
            {"text": t, "tagged": generate.with_speaker_tag(t)} for t in WITH_SPEAKER_TAG
        ],
        "cases": cases,
    }

    # -- chunk goldens ------------------------------------------------------
    split_by_length = [
        {"name": name, "length": length, "chunks": chunking.split_by_length(text, length)}
        for name, text in SPLIT_CORPUS.items()
        for length in (20, 60, 100, 300)
    ]
    split_text = [
        {
            "name": name,
            "max_bytes": max_bytes,
            "max_speakers": max_speakers,
            "chunks": chunking.split_drama3_text(text, max_bytes, max_speakers),
        }
        for name, text in TEXT_CORPUS.items()
        for max_bytes, max_speakers in ((300, 3), (100, 3), (60, 2), (150, 1))
    ]
    history_candidates = []
    for recent_turns in (0, 1, 2, 3):
        for pushes in range(6):
            history = chunking.Drama3History(recent_turns=recent_turns)
            pushed = [f"chunk {i}" for i in range(pushes)]
            for text in pushed:
                history.push(text, tensor([[0]] * CODEBOOKS))
            history_candidates.append(
                {
                    "recent_turns": recent_turns,
                    "pushed": pushed,
                    "candidates": [[t for t, _ in turns] for turns in history.candidates()],
                }
            )

    model = SimpleNamespace(tokenizer=base_tokenizer, config=config)

    def run(references, chunks, max_prompt_rows):
        history = chunking.Drama3History()
        texts = [r["transcript"] for r in references]
        tokens = [tensor(r["codes"]) for r in references]
        pushed = {}  # id of a pushed codes tensor -> its chunk index
        steps = []
        for index, (text, codes) in enumerate(chunks):
            recorder.last = None
            try:
                rows = with_rows(
                    max_prompt_rows,
                    lambda: build_chunk_prompt(model, text, texts, tokens, history),
                )
                kept = [pushed[id(turn_codes)] for _, turn_codes in recorder.last]
                steps.append({"history": kept, "rows": len(rows), "error": None})
            except InvalidInputError as error:
                steps.append({"history": None, "rows": None, "error": describe(error)})
            if codes is not None:
                turn_codes = tensor(codes)
                pushed[id(turn_codes)] = index
                history.push(text, turn_codes)
        return {"max_prompt_rows": max_prompt_rows, "steps": steps}

    def bare_rows(text, references):
        return len(
            with_rows(
                1 << 30,
                lambda: generate.build_drama3_prompt(
                    text,
                    base_tokenizer,
                    config,
                    [r["transcript"] for r in references],
                    [tensor(r["codes"]) for r in references],
                ),
            )
        )

    dialogue_chunks = chunking.split_drama3_text(DIALOGUE)
    english_chunks = chunking.split_drama3_text(LONG_ENGLISH)
    reference = [{"transcript": EN0, "codes": make_codes(20, 21)}]
    plans = {
        "dialogue": ([], [(c, make_codes(2 + 3 * (i % 4), 30 + i, CODEBOOK_SIZE)) for i, c in enumerate(dialogue_chunks)]),
        "english_with_reference": (
            reference,
            [(c, make_codes(4 + 5 * (i % 3), 50 + i)) for i, c in enumerate(english_chunks)],
        ),
        "eos_before_a_frame_and_bad_text": (
            [],
            [
                ("First chunk.", make_codes(6, 70)),
                ("Second chunk, no frames.", None),
                ("<|speaker:16|> bad", make_codes(4, 71)),
                ("Fourth chunk.", make_codes(5, 72)),
                ("Fifth chunk.", make_codes(3, 73)),
            ],
        ),
    }
    history_selection = []
    for name, (references, chunks) in plans.items():
        valid = [text for text, _ in chunks if text != "<|speaker:16|> bad"]
        widest = max(bare_rows(text, references) for text in valid)
        budgets = (default_rows, widest + 40, widest + 25, widest + 10, widest, widest - 1)
        history_selection.append(
            {
                "name": name,
                "references": references,
                "chunks": [{"text": text, "codes": codes} for text, codes in chunks],
                "runs": [run(references, chunks, budget) for budget in budgets],
            }
        )

    chunk_goldens = {
        "schema": "izwi.fish_drama3.chunk_goldens",
        "schema_version": 1,
        "generator": GENERATOR,
        "provenance": provenance,
        "constants": {
            "chunk_bytes": chunking.DRAMA3_CHUNK_BYTES,
            "max_speakers": chunking.DRAMA3_CHUNK_MAX_SPEAKERS,
            "chunk_max_new_tokens": chunking.DRAMA3_CHUNK_MAX_NEW_TOKENS,
            "recent_turns": chunking.DRAMA3_HISTORY_RECENT_TURNS,
            "sentence_ends": "".join(sorted(chunking.SENTENCE_ENDS)),
            "clause_marks": "".join(sorted(chunking.CLAUSE_MARKS)),
            "spaces": "".join(sorted(chunking.SPACES)),
        },
        "corpus": TEXT_CORPUS,
        "protected_pieces": [
            {"text": t, "pieces": [[p, bool(protected)] for p, protected in chunking._protected_pieces(t)]}
            for t in PROTECTED
        ],
        "split_by_length": split_by_length,
        "parse_speaker_segments": [
            {"text": t, "segments": [list(s) for s in chunking.parse_speaker_segments(t)]} for t in PARSE
        ],
        "split_text": split_text,
        "history_candidates": history_candidates,
        "history_selection": history_selection,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    dump(prompt_goldens, args.out / "prompt_goldens.json")
    dump(chunk_goldens, args.out / "chunk_goldens.json")
    print(
        f"wrote {len(cases)} prompt cases, {len(split_by_length) + len(split_text)} split cases and "
        f"{sum(len(plan['runs']) for plan in history_selection)} history runs to {args.out}"
    )


if __name__ == "__main__":
    main()
