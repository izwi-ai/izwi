#!/usr/bin/env python3
"""Deterministic checks for the TTS concurrency driver, against a local fake server."""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import pathlib
import struct
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location(
    'driver', pathlib.Path(__file__).with_name('run-tts-concurrency.py'))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def wav(samples, rate=24000, streamed=False):
    """A mono s16 WAV; `streamed` writes the unknown-size sizes a stream sends."""
    data = struct.pack('<%dh' % len(samples), *samples)
    riff, data_size = (0xFFFFFFFF, 0xFFFFFFFF) if streamed else (36 + len(data), len(data))
    header = (b'RIFF' + struct.pack('<I', riff) + b'WAVE' + b'fmt '
              + struct.pack('<IHHIIHH', 16, 1, 1, rate, rate * 2, 2, 16)
              + b'data' + struct.pack('<I', data_size))
    return header + data


class FakeSpeech(http.server.BaseHTTPRequestHandler):
    """Audio that depends on the request's seed and, when `load_dependent`, on how
    many requests are in flight (a server whose output changes under load)."""

    load_dependent = False
    in_flight = 0
    lock = threading.Lock()
    bodies = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/v1/health':
            payload = json.dumps({'runtime': {'loaded_models': []}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        cls = type(self)
        with cls.lock:
            cls.bodies.append(body)
            cls.in_flight += 1
            width = cls.in_flight
        try:
            threading.Event().wait(0.05)  # Let concurrent requests overlap.
            value = body['seed'] * 7 + (width if cls.load_dependent else 0)
            payload = wav([value % 3000] * 4800, streamed='stream_format' in body)
            diagnostics = {'latency_breakdown': {'codec_ms': 10.0 + body['seed'], 'decode_ms': 40.0,
                                                 'note': 'ignored'}}
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('X-Izwi-Tts-Diagnostics', json.dumps(diagnostics))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            with cls.lock:
                cls.in_flight -= 1


@contextlib.contextmanager
def fake_server(load_dependent):
    handler = type('Handler', (FakeSpeech,), {'load_dependent': load_dependent, 'in_flight': 0,
                                              'lock': threading.Lock(), 'bodies': []})
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}', handler
    finally:
        server.shutdown()
        server.server_close()


class Requests(unittest.TestCase):
    def test_request_i_carries_seed_i_plus_one_and_its_text_at_every_level(self):
        body = driver.request_body('Kokoro-82M', 5, False, 'REF')
        self.assertEqual(body['seed'], 6)
        self.assertEqual(body['input'], driver.POOL[5])
        self.assertEqual(body['voice'], 'af_heart')
        self.assertNotIn('stream_format', body)
        self.assertNotIn('temperature', body)
        self.assertEqual(driver.request_body('Kokoro-82M', 5, True, 'REF')['stream_format'], 'audio')

    def test_clone_models_send_the_reference_clip_and_its_text(self):
        body = driver.request_body('FishAudio-S2-Pro', 0, False, 'REF')
        self.assertEqual(body['reference_audio'], 'REF')
        self.assertEqual(body['reference_text'], driver.REFERENCE_TEXT)
        self.assertNotIn(driver.CLONE, body)
        json.dumps(body)  # Serializable: the marker never reaches the wire.

    def test_greedy_every_samples_every_nth_request_at_temperature_zero(self):
        temps = [driver.request_body('Kokoro-82M', i, False, 'R', greedy_every=3).get('temperature')
                 for i in range(6)]
        self.assertEqual(temps, [None, None, 0.0, None, None, 0.0])

    def test_inputs_are_fixed_and_twenty_to_forty_words(self):
        self.assertEqual(driver.inputs(16), driver.inputs(16))
        for text in driver.POOL:
            self.assertTrue(20 <= len(text.split()) <= 46, text)

    def test_levels_start_at_one(self):
        self.assertEqual(driver.parse_levels('1,8,32'), [1, 8, 32])
        with self.assertRaises(Exception):
            driver.parse_levels('2,4')


class Parsing(unittest.TestCase):
    def test_wav_pcm_reads_buffered_and_streamed_headers(self):
        for streamed in (False, True):
            rate, channels, bits, pcm = driver.wav_pcm(wav([1, 2, 3], streamed=streamed))
            self.assertEqual((rate, channels, bits, len(pcm)), (24000, 1, 16, 6))
        with self.assertRaises(ValueError):
            driver.wav_pcm(b'{"error": "x"}')

    def test_summary_counts_parity_against_the_solo_outputs(self):
        solo = {0: {'md5': 'a', 'pcm_len': 10, 'audio_s': 1.0},
                1: {'md5': 'b', 'pcm_len': 10, 'audio_s': 1.0}}
        records = [{'index': 0, 'md5': 'a', 'pcm_len': 10, 'audio_s': 1.0, 'latency_s': 0.5},
                   {'index': 1, 'md5': 'c', 'pcm_len': 12, 'audio_s': 1.2, 'latency_s': 0.5},
                   {'index': 2, 'error': 'HTTP 503: busy', 'latency_s': 0.1}]
        s = driver.summarize(records, 1.0, solo)
        self.assertEqual((s['ok'], s['errors']), (2, 1))
        self.assertEqual(s['parity']['compared'], 2)
        self.assertEqual(s['parity']['exact'], 1)
        self.assertEqual(s['parity']['differing'], [1])
        self.assertAlmostEqual(s['parity']['max_len_delta_s'], 0.2)
        self.assertAlmostEqual(s['audio_s_per_s'], 2.2)

    def test_diagnostics_medians_take_numeric_ms_fields_only(self):
        records = [{'headers': {'X-Izwi-Tts-Diagnostics': json.dumps(
            {'latency_breakdown': {'codec_ms': v, 'decode_ms': 2 * v, 'label': 'x'}})}}
            for v in (1.0, 3.0, 5.0)]
        records.append({'headers': {'X-Izwi-Tts-Diagnostics': 'not json'}})
        records.append({'headers': {}})
        self.assertEqual(driver.diagnostics_medians(records), {'codec_ms': 3.0, 'decode_ms': 6.0})


class EndToEnd(unittest.TestCase):
    def run_driver(self, load_dependent, shapes='S'):
        with tempfile.TemporaryDirectory() as tmp, fake_server(load_dependent) as (base, handler):
            out = pathlib.Path(tmp) / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                code = driver.main(['run', '--out', str(out), '--models', 'Kokoro-82M',
                                    '--base', base, '--shapes', shapes, '--levels', '1,4',
                                    '--stream-levels', '1,4', '--no-load'])
            self.assertEqual(code, 0)
            point = json.loads((out / 'Kokoro-82M' / 'S-c4.json').read_text())
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                gate = driver.main(['report', str(out), '--require-parity'])
            return point, buffer.getvalue(), gate, handler.bodies

    def test_a_load_invariant_server_passes_the_parity_gate(self):
        point, table, gate, bodies = self.run_driver(load_dependent=False)
        summary = point['summary']
        self.assertEqual((summary['n'], summary['ok']), (8, 8))
        self.assertEqual(summary['parity']['exact'], 4)  # c1 ran inputs 0-3.
        self.assertEqual(summary['parity']['compared'], 4)
        self.assertEqual(summary['diagnostics_ms_p50']['decode_ms'], 40.0)
        self.assertEqual(gate, 0)
        self.assertIn('| Kokoro-82M | S | 4 | 8/8 |', table)
        self.assertEqual(sorted({b['seed'] for b in bodies}), list(range(1, 9)))
        record = point['records'][3]
        expected = hashlib.md5(struct.pack('<4800h', *([(4 * 7) % 3000] * 4800))).hexdigest()
        self.assertEqual((record['index'], record['md5']), (3, expected))

    def test_a_server_whose_audio_changes_under_load_fails_the_parity_gate(self):
        point, table, gate, _ = self.run_driver(load_dependent=True)
        self.assertLess(point['summary']['parity']['exact'], point['summary']['parity']['compared'])
        self.assertEqual(gate, 1)
        self.assertIn('Parity differs from c = 1', table)

    def test_streamed_points_record_the_first_audio(self):
        with tempfile.TemporaryDirectory() as tmp, fake_server(False) as (base, handler):
            out = pathlib.Path(tmp) / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                driver.main(['run', '--out', str(out), '--models', 'Kokoro-82M', '--base', base,
                             '--shapes', 'T', '--stream-levels', '1', '--no-load'])
            point = json.loads((out / 'Kokoro-82M' / 'T-c1.json').read_text())
            self.assertTrue(all(r['first_audio_s'] is not None for r in point['records']))
            self.assertTrue(all(b.get('stream_format') == 'audio' for b in handler.bodies[2:]))


if __name__ == '__main__':
    unittest.main()
