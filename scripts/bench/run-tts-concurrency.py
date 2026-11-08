#!/usr/bin/env python3
"""TTS serving concurrency matrix against a running izwi-server (Python stdlib only).

Loads one TTS model at a time and drives it with closed-loop request levels:

  S  buffered WAV, 20-40 word inputs (about 8-14 s of audio),
     c = 1, 2, 4, 8, 16, 32, 64 by default; n = max(4, 2c) requests a level
  T  streamed WAV (stream_format=audio), same inputs, c = 1, 8, 32;
     n = max(4, c). Records the time to the first 100 ms of audio.

Request i carries seed i + 1 and the same text at every level, so each level's
outputs are compared with the c = 1 outputs (md5 of the PCM). A seeded request
that sounds the same alone and under load gives exact parity at every level.
With --greedy-every N, every Nth request samples at temperature 0, so a level
mixes sampling settings in one decode step.

A level whose wall time passes --level-budget-s, or where half the requests
fail, stops that shape's climb (the next level roughly doubles it).

Each point writes OUT/<model>/<shape>-c<N>.json with the client records
(including every X-* response header, such as X-Izwi-Tts-Diagnostics), a
summary, nvidia-smi samples when the tool exists, filtered Prometheus metrics
and /v1/health before and after.

  run-tts-concurrency.py run --out OUT --models Kokoro-82M,FishAudio-S2-Pro
  run-tts-concurrency.py report OUT [--require-parity]

`report` prints a Markdown table; --require-parity exits 1 when any level's
outputs differ from that model's c = 1 outputs.
"""
import argparse
import base64
import hashlib
import json
import pathlib
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[2]
REQUEST_TIMEOUT_S = 1800
REFERENCE_TEXT = 'The quick brown fox jumps over the lazy dog.'
CLONE = object()  # Marker: the request carries the reference clip.

# Per-model request fields; every request is English text.
MODELS = {
    'Kokoro-82M': {'voice': 'af_heart'},
    'Qwen3-TTS-12Hz-0.6B-CustomVoice': {'voice': 'Ryan', 'language': 'English'},
    'Qwen3-TTS-12Hz-1.7B-CustomVoice': {'voice': 'Ryan', 'language': 'English'},
    'Qwen3-TTS-12Hz-0.6B-Base': {CLONE: True, 'language': 'English'},
    'Qwen3-TTS-12Hz-1.7B-Base': {CLONE: True, 'language': 'English'},
    'Qwen3-TTS-12Hz-1.7B-VoiceDesign': {
        'instructions': 'A calm adult male narrator with a warm, clear voice and steady pacing.',
        'language': 'English'},
    'Voxtral-4B-TTS-2603': {'voice': 'casual_male'},
    'VibeVoice-1.5B': {CLONE: True},
    'FishAudio-S2-Pro': {CLONE: True},
    'FishAudio-Drama-3': {},
    'Chatterbox-500M': {'voice': 'default'},
    'Chatterbox-Multilingual-500M': {'voice': 'default', 'language': 'en'},
    'Maya1-3B': {'voice': 'ava'},
    'Maya1-3B-GGUF': {'voice': 'ava'},
    'Maya2-Native-2.5B': {'voice': 'Ahana', 'language': 'en'},
}

SHAPES = {
    'S': {'stream': False, 'n': lambda c: max(4, 2 * c)},
    'T': {'stream': True, 'n': lambda c: max(4, c)},
}
DEFAULT_LEVELS = {'S': [1, 2, 4, 8, 16, 32, 64], 'T': [1, 8, 32]}

# Plain declarative sentences; each input joins two to four of them.
SENTENCES = [
    'The morning train was late again, so we walked along the river to the old market.',
    'She opened the window and the smell of rain filled the quiet kitchen.',
    'Our team will meet on Thursday to review the plan for the new library.',
    'He said the bridge would be finished before the end of the summer.',
    'A small boat drifted past the harbour lights as the evening grew cold.',
    'Please remember to bring your notes, a pencil, and a bottle of water.',
    'The children laughed when the dog chased its tail around the garden.',
    'By the time the music started, every seat in the hall was taken.',
    'We planted tomatoes, beans, and sunflowers along the southern fence.',
    'The doctor explained the results slowly and answered every question.',
    'At the top of the hill there is a bench with a view of the whole valley.',
    'Most of the shops close early on Sunday, so plan your errands for Saturday.',
    'The old clock in the hallway has not kept the right time for years.',
    'When the storm passed, the streets were covered in leaves and branches.',
    'Her first novel was written by hand in a notebook she carried everywhere.',
    'The coffee shop on the corner serves the best bread in the neighbourhood.',
    'If the weather holds, we will climb to the lake and camp for the night.',
    'The engineer checked each cable twice before turning the power back on.',
    'Nobody expected the quiet student to win the debate, but she did.',
    'The museum is free on the first Monday of every month.',
    'Thousands of birds gather on the marsh each autumn before flying south.',
    'He fixed the bicycle chain with an old tool he found in the shed.',
    'The festival ends with a parade, fireworks, and music until midnight.',
    'Fresh snow had fallen overnight, and the whole town was silent.',
]


def inputs(count, seed=31):
    """`count` texts of 20-40 words, the same for every run."""
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        picked = rng.sample(SENTENCES, rng.choice([2, 3, 3]))
        while len(' '.join(picked).split()) > 40:
            picked = picked[:-1]
        while len(' '.join(picked).split()) < 20:
            picked.append(rng.choice(SENTENCES))
        texts.append(' '.join(picked))
    return texts


POOL = inputs(128)


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def request_body(model, index, stream, reference_b64, greedy_every=0):
    """The speech request for input `index`: its text, seed and model fields."""
    body = {}
    for key, value in MODELS[model].items():
        if key is CLONE:
            body.update({'reference_audio': reference_b64, 'reference_text': REFERENCE_TEXT})
        else:
            body[key] = value
    body.update({'model': model, 'input': POOL[index % len(POOL)],
                 'response_format': 'wav', 'seed': index + 1})
    if greedy_every and index % greedy_every == greedy_every - 1:
        body['temperature'] = 0.0
    if stream:
        body['stream_format'] = 'audio'
    return body


def wav_pcm(payload):
    """(rate, channels, bits, pcm) of a RIFF WAV; tolerates streamed sizes."""
    if payload[:4] != b'RIFF':
        raise ValueError(f'not a WAV: {payload[:16]!r}')
    pos, rate, channels, bits = 12, 0, 1, 16
    while pos + 8 <= len(payload):
        chunk_id = payload[pos:pos + 4]
        size = int.from_bytes(payload[pos + 4:pos + 8], 'little')
        if chunk_id == b'fmt ':
            channels = int.from_bytes(payload[pos + 10:pos + 12], 'little')
            rate = int.from_bytes(payload[pos + 12:pos + 16], 'little')
            bits = int.from_bytes(payload[pos + 22:pos + 24], 'little')
        if chunk_id == b'data':
            return rate, channels, bits, payload[pos + 8:]
        pos += 8 + size
    raise ValueError('no data chunk')


class Client:
    def __init__(self, base):
        self.base = base.rstrip('/')

    def json(self, method, path, timeout=1800):
        request = urllib.request.Request(
            self.base + path, method=method, data=b'{}' if method == 'POST' else None,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        try:
            return json.loads(raw or b'{}')
        except ValueError:
            return {'raw': raw.decode('utf-8', 'replace')}

    def health(self):
        try:
            return self.json('GET', '/v1/health', timeout=30)
        except Exception as error:  # noqa: BLE001
            return {'error': repr(error)}

    def metrics(self):
        try:
            with urllib.request.urlopen(self.base + '/v1/metrics/prometheus', timeout=60) as response:
                text = response.read().decode('utf-8')
        except Exception as error:  # noqa: BLE001
            return repr(error)
        keep = ('tts', 'speech', 'engine', 'step', 'batch', 'queue', 'cuda', 'row', 'audio')
        return '\n'.join(line for line in text.splitlines()
                         if not line.startswith('#') and any(k in line.lower() for k in keep))

    def loaded(self):
        return [m.get('variant_id') for m in self.health().get('runtime', {}).get('loaded_models', [])]

    def synthesize(self, body, stream):
        request = urllib.request.Request(
            self.base + '/v1/audio/speech', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        record = {'words': len(body['input'].split()), 'temperature': body.get('temperature')}
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                chunks, total, first_byte, first_audio = [], 0, None, None
                while True:
                    chunk = response.read1(65536) if hasattr(response, 'read1') else response.read(65536)
                    if not chunk:
                        break
                    now = time.monotonic() - started
                    if first_byte is None:
                        first_byte = now
                    chunks.append(chunk)
                    total += len(chunk)
                    if first_audio is None and stream and total >= 44 + 4800:
                        first_audio = now  # Header plus 100 ms of 24 kHz mono s16.
                payload = b''.join(chunks)
                headers = {k: v for k, v in response.headers.items() if k.lower().startswith('x-')}
            rate, channels, bits, pcm = wav_pcm(payload)
            frames = len(pcm) // max(1, channels * bits // 8)
            record.update({
                'rate': rate, 'audio_s': frames / rate if rate else 0.0, 'bytes': len(payload),
                'md5': hashlib.md5(pcm).hexdigest(), 'pcm_len': len(pcm),
                'first_byte_s': first_byte, 'first_audio_s': first_audio, 'headers': headers,
            })
        except urllib.error.HTTPError as error:
            record['error'] = f"HTTP {error.code}: {error.read(2048).decode('utf-8', 'replace')}"
        except Exception as error:  # noqa: BLE001
            record['error'] = f'{type(error).__name__}: {error}'
        record['latency_s'] = time.monotonic() - started
        return record


class GpuSampler:
    """nvidia-smi utilisation, memory and power once a second, when available."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout
                util, memory, power = [float(x) for x in out.strip().splitlines()[0].split(',')[:3]]
                self.samples.append((round(time.monotonic(), 2), util, memory, power))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)


def run_point(synthesize, n, c):
    """Release `n` requests with at most `c` in flight; records in index order."""
    gate = threading.BoundedSemaphore(c)
    records = [None] * n
    t0 = time.monotonic()

    def worker(index):
        try:
            started = time.monotonic() - t0
            record = synthesize(index)
            record.update({'index': index, 'start_offset_s': started,
                           'finished_at': time.monotonic() - t0})
            records[index] = record
        finally:
            gate.release()

    threads = []
    for index in range(n):
        gate.acquire()
        thread = threading.Thread(target=worker, args=(index,), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return records, time.monotonic() - t0


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def diagnostics_medians(records):
    """Median of each numeric `*_ms` field in X-Izwi-Tts-Diagnostics' latency breakdown."""
    fields = {}
    for record in records:
        raw = (record.get('headers') or {}).get('X-Izwi-Tts-Diagnostics')
        try:
            breakdown = json.loads(raw).get('latency_breakdown') or {} if raw else {}
        except (ValueError, AttributeError):
            continue
        for key, value in breakdown.items():
            if key.endswith('_ms') and isinstance(value, (int, float)):
                fields.setdefault(key, []).append(float(value))
    return {key: statistics.median(values) for key, values in sorted(fields.items())}


def summarize(records, wall, solo):
    ok = [r for r in records if 'error' not in r]
    errors = [r['error'][:300] for r in records if 'error' in r]
    audio = sum(r['audio_s'] for r in ok)
    latency = [r['latency_s'] for r in ok]
    rtf = [r['latency_s'] / r['audio_s'] for r in ok if r['audio_s'] > 0]
    first_audio = [r['first_audio_s'] for r in ok if r.get('first_audio_s') is not None]
    parity = {'compared': 0, 'exact': 0, 'same_len': 0, 'max_len_delta_s': 0.0, 'differing': []}
    for record in ok if solo else []:
        alone = solo.get(record['index'])
        if alone is None:
            continue
        parity['compared'] += 1
        parity['exact'] += int(alone['md5'] == record['md5'])
        parity['same_len'] += int(alone['pcm_len'] == record['pcm_len'])
        parity['max_len_delta_s'] = max(parity['max_len_delta_s'],
                                        abs(alone['audio_s'] - record['audio_s']))
        if alone['md5'] != record['md5']:
            parity['differing'].append(record['index'])
    return {
        'n': len(records), 'ok': len(ok), 'errors': len(errors), 'error_samples': errors[:3],
        'wall_s': wall, 'audio_s': audio,
        'audio_s_per_s': audio / wall if wall else None,
        'lat_p50': percentile(latency, .5), 'lat_p95': percentile(latency, .95),
        'lat_max': max(latency) if latency else None,
        'rtf_p50': percentile(rtf, .5), 'rtf_p95': percentile(rtf, .95),
        'first_audio_p50': percentile(first_audio, .5),
        'first_audio_p95': percentile(first_audio, .95),
        'mean_audio_s': audio / len(ok) if ok else None,
        'diagnostics_ms_p50': diagnostics_medians(ok),
        'parity': parity,
    }


def wait_downloaded(client, model, limit_s=4 * 3600):
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        try:
            listing = client.json('GET', '/v1/admin/models', timeout=60)
            items = listing if isinstance(listing, list) else listing.get('models', [])
            status = next((m['status'] for m in items if m.get('variant') == model), 'missing')
        except Exception as error:  # noqa: BLE001
            status = repr(error)
        if status in ('downloaded', 'ready', 'loaded'):
            return True
        if status in ('not_downloaded', 'error', 'failed', 'missing'):
            log(f'{model} download status {status}')
            return False
        time.sleep(30)
    return False


def load_model(client, model):
    for other in client.loaded():
        if other and other != model:
            log(f'unloading {other}')
            try:
                client.json('POST', f'/v1/admin/models/{other}/unload', timeout=600)
            except Exception as error:  # noqa: BLE001
                log(f'unload {other} failed: {error!r}')
    started = time.monotonic()
    response = client.json('POST', f'/v1/admin/models/{model}/load', timeout=3600)
    for _ in range(600):
        if model in client.loaded():
            break
        time.sleep(2)
    return {'response': response, 'load_s': time.monotonic() - started, 'loaded': client.loaded()}


def run_model(client, out, model, args, reference_b64):
    model_dir = out / model
    model_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_load:
        if not wait_downloaded(client, model):
            (model_dir / 'not-downloaded.txt').write_text('download did not finish\n')
            return
        try:
            info = load_model(client, model)
        except Exception as error:  # noqa: BLE001
            info = {'error': repr(error)}
        (model_dir / 'load.json').write_text(json.dumps(info, indent=1))
        log(f'{model} load: {json.dumps(info)[:300]}')
        if model not in client.loaded():
            log(f'{model} did not load; skipping')
            return

    def synthesize(stream):
        return lambda index: client.synthesize(
            request_body(model, index, stream, reference_b64, args.greedy_every), stream)

    warm, _ = run_point(synthesize(False), 2, 2)
    (model_dir / 'warmup.json').write_text(json.dumps(warm, indent=1))
    if all('error' in r for r in warm):
        log(f"{model} warm-up failed: {warm[0].get('error', '')[:300]}")
        (model_dir / 'warmup-failed.txt').write_text(json.dumps(warm, indent=1))
        return
    for shape in args.shapes:
        spec, solo = SHAPES[shape], None
        for c in args.levels[shape]:
            n = spec['n'](c)
            health_before, metrics_before = client.health(), client.metrics()
            with GpuSampler() as gpu:
                records, wall = run_point(synthesize(spec['stream']), n, c)
            health_after, metrics_after = client.health(), client.metrics()
            summary = summarize(records, wall, solo)
            utils = [s[1] for s in gpu.samples]
            summary['gpu_util_mean'] = statistics.mean(utils) if utils else None
            summary['gpu_mem_max_mib'] = max((s[2] for s in gpu.samples), default=None)
            if c == 1:
                solo = {r['index']: r for r in records if 'error' not in r}
            point = {'model': model, 'shape': shape, 'c': c, 'summary': summary, 'records': records,
                     'gpu': gpu.samples, 'metrics_before': metrics_before,
                     'metrics_after': metrics_after, 'health_before': health_before,
                     'health_after': health_after}
            (model_dir / f'{shape}-c{c}.json').write_text(json.dumps(point, indent=1))
            s = summary
            log(f"{model} {shape} c={c} n={n} ok={s['ok']} err={s['errors']} wall={wall:.1f}s "
                f"thr={s['audio_s_per_s'] or 0:.2f} audio-s/s rtf_p50={s['rtf_p50'] or 0:.3f} "
                f"lat_p50={s['lat_p50'] or 0:.2f} p95={s['lat_p95'] or 0:.2f} "
                f"fa_p50={s['first_audio_p50'] or 0:.2f} util={s['gpu_util_mean'] or 0:.0f}% "
                f"par={s['parity']['exact']}/{s['parity']['compared']}"
                + (f" ERR {s['error_samples'][0][:200]}" if s['errors'] else ''))
            if s['errors'] and s['errors'] >= s['n'] / 2:
                log(f'{model} {shape}: half or more failed; stopping the climb')
                break
            if wall > args.level_budget_s:
                log(f'{model} {shape}: wall {wall:.0f}s over budget; stopping the climb')
                break


def fmt(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def report(out):
    """Markdown rows for every point under `out`; returns (lines, parity_failures)."""
    out = pathlib.Path(out)
    lines = ['| model | shape | c | ok/n | wall s | audio-s/s | x c1 | RTF p50 | lat p50 | lat p95 | '
             'first audio p50 | GPU util % | parity exact | codec ms p50 | decode ms p50 |',
             '|' + '---|' * 15]
    failures = []
    for model_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        if (model_dir / 'warmup-failed.txt').exists():
            warm = json.loads((model_dir / 'warmup-failed.txt').read_text())
            lines.append(f"| {model_dir.name} | warm-up failed: {warm[0].get('error', '')[:160]} |"
                         + ' |' * 13)
            continue
        if (model_dir / 'not-downloaded.txt').exists():
            lines.append(f'| {model_dir.name} | not downloaded |' + ' |' * 13)
            continue
        points = []
        for path in model_dir.glob('*-c*.json'):
            match = re.fullmatch(r'(\w)-c(\d+)\.json', path.name)
            if match:
                points.append((match.group(1), int(match.group(2)), path))
        base = {}
        for shape, c, path in sorted(points):
            s = json.loads(path.read_text())['summary']
            throughput = s['audio_s_per_s']
            if c == 1:
                base[shape] = throughput
            gain = throughput / base[shape] if base.get(shape) and throughput else None
            parity = s['parity']
            exact = f"{parity['exact']}/{parity['compared']}" if parity['compared'] else '-'
            if parity['compared'] and parity['exact'] != parity['compared']:
                failures.append(f'{model_dir.name} {shape} c={c}: {exact} exact')
            ms = s.get('diagnostics_ms_p50') or {}
            first_audio = fmt(s['first_audio_p50']) if shape == 'T' else '-'
            lines.append(
                f"| {model_dir.name} | {shape} | {c} | {s['ok']}/{s['n']} | {fmt(s['wall_s'], 1)} | "
                f"{fmt(throughput)} | {fmt(gain, 1)} | {fmt(s['rtf_p50'], 3)} | {fmt(s['lat_p50'])} | "
                f"{fmt(s['lat_p95'])} | {first_audio} | {fmt(s['gpu_util_mean'], 0)} | {exact} | "
                f"{fmt(ms.get('codec_ms'), 0)} | {fmt(ms.get('decode_ms'), 0)} |")
            if s['errors']:
                sample = s['error_samples'][0][:180].replace('|', '/')
                lines.append(f"|  | errors: {s['errors']} ({sample}) |" + ' |' * 13)
    return lines, failures


def parse_levels(text):
    levels = [int(x) for x in text.split(',') if x]
    if not levels or levels[0] != 1 or any(x < 1 for x in levels):
        raise argparse.ArgumentTypeError('levels start at 1 (the parity reference) and stay positive')
    return levels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='drive the server and write one JSON per point')
    run.add_argument('--out', required=True, type=pathlib.Path)
    run.add_argument('--models', required=True, help='comma-separated catalog variants')
    run.add_argument('--base', default='http://127.0.0.1:8080')
    run.add_argument('--shapes', default='S,T')
    run.add_argument('--levels', type=parse_levels, help='buffered levels (default 1,2,4,8,16,32,64)')
    run.add_argument('--stream-levels', type=parse_levels, help='streamed levels (default 1,8,32)')
    run.add_argument('--level-budget-s', type=float, default=600.0)
    run.add_argument('--greedy-every', type=int, default=0,
                     help='every Nth request samples at temperature 0 (0 = never)')
    run.add_argument('--reference-wav', type=pathlib.Path, default=REPO / 'data' / 'fox.wav',
                     help=f'voice-clone reference; its text must be "{REFERENCE_TEXT}"')
    run.add_argument('--no-load', action='store_true',
                     help='use the models as loaded; do not download, load or unload')
    summary = commands.add_parser('report', help='print a Markdown table of a run')
    summary.add_argument('out', type=pathlib.Path)
    summary.add_argument('--require-parity', action='store_true',
                         help='exit 1 when any level differs from the c = 1 outputs')
    args = parser.parse_args(argv)

    if args.command == 'report':
        lines, failures = report(args.out)
        print('\n'.join(lines))
        if failures:
            print('\nParity differs from c = 1:\n' + '\n'.join(f'- {f}' for f in failures))
        return 1 if args.require_parity and failures else 0

    models = [m for m in args.models.split(',') if m]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        parser.error(f"unknown models: {', '.join(unknown)} (known: {', '.join(MODELS)})")
    args.shapes = [s for s in args.shapes.split(',') if s]
    if any(s not in SHAPES for s in args.shapes):
        parser.error('shapes are S and T')
    args.levels = {'S': args.levels or DEFAULT_LEVELS['S'],
                   'T': args.stream_levels or DEFAULT_LEVELS['T']}
    reference_b64 = base64.b64encode(args.reference_wav.read_bytes()).decode()
    client = Client(args.base)
    args.out.mkdir(parents=True, exist_ok=True)
    for model in models:
        log(f'=== {model}')
        try:
            run_model(client, args.out, model, args, reference_b64)
        except Exception as error:  # noqa: BLE001
            log(f'{model} crashed the driver: {error!r}')
        if not args.no_load:
            try:
                client.json('POST', f'/v1/admin/models/{model}/unload', timeout=600)
            except Exception as error:  # noqa: BLE001
                log(f'unload {model}: {error!r}')
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
