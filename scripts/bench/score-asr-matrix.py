#!/usr/bin/env python3
"""Score a run-asr-matrix.sh results.jsonl into a table and summary.json.

Per transcription it reports normalized WER against the repository references
(fox, test), word count, duration and rtf as returned, how many long-form
chunks stopped on their token budget, and whether silence/noise fixtures came
back empty. SSE responses (stream=true) are scored from their final event.
"""
import argparse
import json
import pathlib
import re
import sys
import unicodedata

REFERENCES = {"fox": "data/fox.md", "test": "data/test.md"}
EMPTY_EXPECTED = {"silence", "noise"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("—", " ").replace("-", " ")
    return re.sub(r"[^\w\s']", " ", text).split()


def wer(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        previous, row[0] = row[0], i
        for j in range(1, len(hyp) + 1):
            current = min(row[j] + 1, row[j - 1] + 1, previous + (ref[i - 1] != hyp[j - 1]))
            previous, row[j] = row[j], current
    return row[len(hyp)] / max(1, len(ref))


def parse_body(raw):
    """Return the response object, reading the last SSE event for streams."""
    raw = raw.strip()
    if not raw.startswith("data:"):
        return json.loads(raw)
    final, deltas = None, []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        if "error" in event:
            return event
        if "delta" in event:
            deltas.append(event["delta"])
        if event.get("text") is not None:
            final = event
    if final is None:
        final = {"text": "".join(deltas)}
    final["_stream_delta_text"] = "".join(deltas)
    return final


def capped_chunks(body):
    chunks = (((body.get("izwi_asr_diagnostics") or {}).get("chunking") or {})
              .get("chunk_transcriptions") or [])
    capped = sum(1 for chunk in chunks
                 if ((chunk.get("model_diagnostics") or {}).get("decode") or {})
                 .get("stop_reason") == "max_tokens")
    return capped, len(chunks)


def score(rows, root):
    summary = {"loads": [], "transcriptions": [], "footprints": []}
    for row in rows:
        if row["kind"] == "load":
            summary["loads"].append({k: row[k] for k in ("model", "http", "secs")})
        elif row["kind"] == "footprint":
            summary["footprints"].append(row)
        elif row["kind"] == "transcription":
            entry = {k: row[k] for k in ("model", "fixture", "http", "secs")}
            try:
                body = parse_body(pathlib.Path(row["body"]).read_text())
            except (OSError, ValueError) as error:
                entry["error"] = f"unreadable response: {error}"
                summary["transcriptions"].append(entry)
                continue
            if "text" not in body:
                entry["error"] = json.dumps(body.get("error", body))[:300]
                summary["transcriptions"].append(entry)
                continue
            text = body["text"]
            entry.update({
                "words": len(normalize(text)),
                "duration": body.get("duration"),
                "rtf": body.get("rtf"),
                "segments": len(body.get("segments") or []),
                "head": text[:60],
            })
            entry["capped_chunks"], entry["chunks"] = capped_chunks(body)
            reference = REFERENCES.get(row["fixture"])
            if reference:
                entry["wer"] = round(wer((root / reference).read_text(), text), 4)
            if row["fixture"] in EMPTY_EXPECTED:
                entry["empty_as_expected"] = not normalize(text)
            if "_stream_delta_text" in body:
                entry["stream_matches_final"] = (
                    normalize(body["_stream_delta_text"]) == normalize(text))
            summary["transcriptions"].append(entry)
    return summary


def render(summary):
    lines = []
    for load in summary["loads"]:
        lines.append(f"load {load['model']} http={load['http']} secs={load['secs']}")
    for t in summary["transcriptions"]:
        if "error" in t:
            lines.append(f"{t['model']:34s} {t['fixture']:8s} http={t['http']} ERROR {t['error']}")
            continue
        extras = []
        if "wer" in t:
            extras.append(f"wer={t['wer']:.3f}")
        if "empty_as_expected" in t:
            extras.append(f"empty={'yes' if t['empty_as_expected'] else 'NO'}")
        if t["chunks"]:
            extras.append(f"capped={t['capped_chunks']}/{t['chunks']}")
        if "stream_matches_final" in t:
            extras.append(f"stream==final={'yes' if t['stream_matches_final'] else 'NO'}")
        lines.append(
            f"{t['model']:34s} {t['fixture']:8s} http={t['http']} secs={t['secs']} "
            f"words={t['words']} dur={t['duration']} rtf={t['rtf']} segs={t['segments']} "
            + " ".join(extras))
    for f in summary["footprints"]:
        lines.append(f"peak {f['model']} {f['peak_mib']} MiB guard_kills={f['guard_kills']}")
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    summary = score(rows, args.root)
    (args.results.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
