#!/usr/bin/env python3
"""Tabulate a serving matrix: throughput, latency, busy time, dispatch widths, defers and phase split.

Reads the per-point JSON both matrix drivers write:

  ROOT/asr/<model>/<shape>-c<N>.json   ASR matrix points (records, summary, a Prometheus `delta`)
  ROOT/tts/<model>/<shape>-c<N>.json   run-tts-concurrency.py points (records, summary,
                                       `metrics_before` / `metrics_after` Prometheus text)

For each point it prints one JSON line (throughput, latency, executor busy seconds and
utilization, parity, `physical_defers_total` by reason, ASR phase seconds) and, per
executor phase, the dispatch count, seconds, mean and max width and milliseconds per
dispatch. Those show where a plateau comes from: a decode width that stops growing,
capacity defers, or a width-1 audio stage taking the busy time.

  analyze-serving-matrix.py ROOT [--markdown]
"""
import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

WIDTH = re.compile(
    r'izwi_engine_executor_physical_dispatch_width_(dispatches|micros)_total'
    r'\{width="(\d+)",phase="(\w+)"\}')
BUSY = 'izwi_engine_executor_physical_dispatch_seconds_total'


def parse_prometheus(text):
    """Sample name with labels -> value, for every sample line."""
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        key, _, value = line.rpartition(' ')
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def delta(before, after):
    """Changed samples between two scrapes, histogram buckets left out."""
    return {k: v - before.get(k, 0.0) for k, v in after.items()
            if v - before.get(k, 0.0) and '_bucket' not in k}


def phase_widths(changes):
    """Per executor phase: dispatches, seconds, mean/max width, ms per dispatch."""
    counts = defaultdict(lambda: defaultdict(float))
    micros = defaultdict(lambda: defaultdict(float))
    for key, value in changes.items():
        match = WIDTH.fullmatch(key)
        if match:
            (counts if match[1] == 'dispatches' else micros)[match[3]][int(match[2])] += value
    phases = {}
    for phase, by_width in counts.items():
        n = sum(by_width.values())
        secs = sum(micros[phase].values()) / 1e6
        rows = sum(width * count for width, count in by_width.items())
        phases[phase] = {
            'dispatches': int(n),
            'secs': round(secs, 2),
            'mean_width': round(rows / n, 1) if n else 0.0,
            'max_width': max(by_width) if by_width else 0,
            'ms_per_dispatch': round(1000 * secs / n, 2) if n else 0.0,
        }
    return phases


def defers(changes):
    return {re.search(r'reason="(\w+)"', k)[1]: int(v)
            for k, v in changes.items() if 'physical_defers_total' in k and 'reason=' in k}


def asr_point(data):
    summary, changes = data['summary'], data.get('delta', {})
    latency = summary.get('latency_s') or {}
    parity = summary.get('parity') or {}
    wall = summary.get('duration_s') or 0.0
    busy = changes.get(BUSY, 0.0)
    return {
        'n': summary.get('requests'), 'ok': summary.get('completed'),
        'audio_s_per_s': round(summary.get('audio_throughput') or 0.0, 1),
        'lat_p50': round(latency.get('p50', 0.0), 2), 'lat_p95': round(latency.get('p95', 0.0), 2),
        'wall_s': round(wall, 1), 'busy_s': round(busy, 2),
        'util': round(busy / wall, 2) if wall else 0.0,
        'parity': f"{parity.get('checked', 0) - parity.get('failed', 0)}/{parity.get('checked', 0)}",
        'defers': defers(changes),
        'asr_phase_s': {re.search(r'phase="(\w+)"', k)[1]: round(v, 2) for k, v in changes.items()
                        if k.startswith('izwi_asr_phase_seconds_sum') and 'phase=' in k},
    }, phase_widths(changes)


def tts_point(data):
    summary = data['summary']
    changes = delta(parse_prometheus(data.get('metrics_before', '')),
                    parse_prometheus(data.get('metrics_after', '')))
    wall = summary.get('wall_s') or 0.0
    busy = changes.get(BUSY, 0.0)
    parity = summary.get('parity') or {}
    first = summary.get('first_audio_p50')
    return {
        'n': summary.get('n'), 'ok': summary.get('ok'), 'errors': summary.get('errors'),
        'audio_s_per_s': round(summary.get('audio_s_per_s') or 0.0, 2),
        'rtf_p50': round(summary.get('rtf_p50') or 0.0, 3),
        'lat_p50': round(summary.get('lat_p50') or 0.0, 2), 'lat_p95': round(summary.get('lat_p95') or 0.0, 2),
        'first_audio_p50': None if first is None else round(first, 3),
        'wall_s': round(wall, 1), 'busy_s': round(busy, 1),
        'util': round(busy / wall, 2) if wall else 0.0,
        'gpu_util': summary.get('gpu_util_mean'), 'gpu_mem_mib': summary.get('gpu_mem_max_mib'),
        'parity': f"{parity.get('exact', '-')}/{parity.get('compared', '-')}",
        'defers': defers(changes),
    }, phase_widths(changes)


def point_order(path):
    shape, _, level = path.stem.partition('-c')
    return shape, int(level) if level.isdigit() else 0


def analyze(root):
    """(kind, model, point name, row, phases) for every point under ROOT."""
    rows = []
    for kind, reader in (('asr', asr_point), ('tts', tts_point)):
        for model_dir in sorted((root / kind).glob('*')):
            if not model_dir.is_dir():
                continue
            for path in sorted(model_dir.glob('[A-Z]-c*.json'), key=point_order):
                row, phases = reader(json.loads(path.read_text()))
                rows.append((kind, model_dir.name, path.stem, row, phases))
    return rows


def markdown(rows):
    lines = ['| model | point | audio-s/s | p50 s | util | decode width mean/max | defers | parity |',
             '|---|---|---|---|---|---|---|---|']
    for _, model, name, row, phases in rows:
        decode = phases.get('decode', {})
        width = f"{decode.get('mean_width', '-')}/{decode.get('max_width', '-')}" if decode else '-'
        lines.append(f"| {model} | {name} | {row['audio_s_per_s']} | {row['lat_p50']} | {row['util']} | "
                     f"{width} | {sum(row['defers'].values()) or 0} | {row['parity']} |")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('root', type=pathlib.Path)
    parser.add_argument('--markdown', action='store_true', help='one Markdown table instead of JSON lines')
    args = parser.parse_args(argv)
    rows = analyze(args.root)
    if not rows:
        print(f'no matrix points under {args.root}', file=sys.stderr)
        return 1
    if args.markdown:
        print(markdown(rows))
        return 0
    current = None
    for kind, model, name, row, phases in rows:
        if (kind, model) != current:
            current = (kind, model)
            print(f'\n## {kind} {model}')
        print(json.dumps({'point': name, **row}))
        for phase, stats in sorted(phases.items()):
            print('   ', phase, json.dumps(stats))
    return 0


if __name__ == '__main__':
    sys.exit(main())
