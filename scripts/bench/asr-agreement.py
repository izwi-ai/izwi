#!/usr/bin/env python3
"""Cross-model ASR agreement on shared clips, for recordings with no ground truth.

Each ASR matrix point keeps its inputs (source, offset, length) and every request's text.
Matrices cut their clips from the same seeded pools, so two models' c=1 points share
inputs. For each model this prints its word disagreement (normalized word error rate, in
percent) against reference models on the clips both transcribed, per shape. A model far
above the others on every reference has an accuracy problem (a broken kernel, a dtype
fallback) even without ground truth.

  asr-agreement.py MATRIX_DIR [MATRIX_DIR ...] [--refs MODEL,...] [--shapes A-c1,B-c1]

Each MATRIX_DIR holds <model>/<shape>.json; a model found in several takes the last.
"""
import argparse
import json
import pathlib
import re
import sys
import unicodedata

DEFAULT_REFS = ['Whisper-Large-v3-Turbo', 'Parakeet-TDT-0.6B-v3',
                'Granite-Speech-4.1-2B-Plus', 'Qwen3-ASR-1.7B-GGUF']


def words(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return re.sub(r"[^\w\s']", ' ', text).split()


def word_errors(reference, hypothesis):
    """(edit distance in words, reference word count)."""
    ref, hyp = words(reference), words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        diagonal, row[0] = row[0], i
        for j in range(1, len(hyp) + 1):
            current = min(row[j] + 1, row[j - 1] + 1, diagonal + (ref[i - 1] != hyp[j - 1]))
            diagonal, row[j] = row[j], current
    return row[len(hyp)], len(ref)


def transcripts(path):
    """Input key -> text for a point's successful requests."""
    data = json.loads(path.read_text())
    keys = [(item.get('source'), round(item.get('offset_secs', item.get('start_secs', 0.0)), 2),
             round(item['audio_secs'], 2)) for item in data['inputs']]
    return {keys[r['index']]: r.get('text', '') for r in data['records']
            if 'error' not in r and r['index'] < len(keys)}


def disagreement(a, b):
    """Percent word error of `b` against `a` over shared inputs, and how many."""
    shared = set(a) & set(b)
    errors = total = 0
    for key in shared:
        e, n = word_errors(a[key], b[key])
        errors, total = errors + e, total + n
    return (100.0 * errors / total if total else None), len(shared)


def table(models, shape, refs):
    data = {m: transcripts(d / f'{shape}.json') for m, d in models.items() if (d / f'{shape}.json').exists()}
    lines = [f'### {shape}: % word disagreement against each reference model (shared clips)',
             '| model | ' + ' | '.join(refs) + ' | mean |', '|---|' + '---|' * (len(refs) + 1)]
    for model in sorted(data):
        cells, values = [], []
        for ref in refs:
            if ref == model or ref not in data:
                cells.append('-')
                continue
            value, shared = disagreement(data[ref], data[model])
            if value is None:
                cells.append('n/a')
            else:
                cells.append(f'{value:.1f} ({shared})')
                values.append(value)
        mean = f'{sum(values) / len(values):.1f}' if values else '-'
        lines.append(f'| {model} | ' + ' | '.join(cells) + f' | {mean} |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dirs', nargs='+', type=pathlib.Path)
    parser.add_argument('--refs', default=','.join(DEFAULT_REFS))
    parser.add_argument('--shapes', default='A-c1,B-c1,L-c1')
    args = parser.parse_args(argv)
    models = {}
    for root in args.dirs:
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            models[model_dir.name] = model_dir
    if not models:
        print('no model directories found', file=sys.stderr)
        return 1
    refs = [r for r in args.refs.split(',') if r]
    print('\n\n'.join(table(models, shape, refs) for shape in args.shapes.split(',')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
