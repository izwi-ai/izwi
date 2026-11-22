#!/usr/bin/env python3
"""Deterministic checks for the serving matrix analyzer and the ASR agreement tool."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = load('analyzer', 'analyze-serving-matrix.py')
agreement = load('agreement', 'asr-agreement.py')

METRICS_BEFORE = '\n'.join([
    '# HELP x',
    'izwi_engine_executor_physical_dispatch_seconds_total 1.0',
    'izwi_engine_executor_physical_dispatch_width_dispatches_total{width="1",phase="decode"} 10',
    'izwi_engine_executor_physical_dispatch_width_micros_total{width="1",phase="decode"} 100000',
])
METRICS_AFTER = '\n'.join([
    'izwi_engine_executor_physical_dispatch_seconds_total 9.0',
    'izwi_engine_executor_physical_dispatch_width_dispatches_total{width="1",phase="decode"} 20',
    'izwi_engine_executor_physical_dispatch_width_micros_total{width="1",phase="decode"} 200000',
    'izwi_engine_executor_physical_dispatch_width_dispatches_total{width="4",phase="decode"} 30',
    'izwi_engine_executor_physical_dispatch_width_micros_total{width="4",phase="decode"} 600000',
    'izwi_engine_executor_physical_dispatch_width_dispatches_total{width="1",phase="other"} 5',
    'izwi_engine_executor_physical_dispatch_width_micros_total{width="1",phase="other"} 95000',
    'izwi_engine_executor_physical_defers_total{reason="managed_cache_capacity"} 7',
    'izwi_engine_executor_physical_dispatch_width_dispatches_total_bucket{le="1"} 99',
])


class Analyzer(unittest.TestCase):
    def test_phase_widths_from_a_scrape_delta(self):
        changes = analyzer.delta(analyzer.parse_prometheus(METRICS_BEFORE),
                                 analyzer.parse_prometheus(METRICS_AFTER))
        self.assertNotIn('izwi_engine_executor_physical_dispatch_width_dispatches_total_bucket{le="1"}',
                         changes)
        phases = analyzer.phase_widths(changes)
        # 10 more width-1 and 30 width-4 decode dispatches: mean (10 + 120) / 40.
        self.assertEqual(phases['decode'], {'dispatches': 40, 'secs': 0.7, 'mean_width': 3.2,
                                            'max_width': 4, 'ms_per_dispatch': 17.5})
        self.assertEqual(phases['other']['ms_per_dispatch'], 19.0)
        self.assertEqual(analyzer.defers(changes), {'managed_cache_capacity': 7})

    def test_tts_and_asr_points_are_read_in_shape_and_level_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            tts = root / 'tts' / 'Model'
            tts.mkdir(parents=True)
            for level in (16, 2):
                (tts / f'S-c{level}.json').write_text(json.dumps({
                    'summary': {'n': 4, 'ok': 4, 'errors': 0, 'audio_s_per_s': 7.5, 'rtf_p50': 0.5,
                                'lat_p50': 3.0, 'lat_p95': 4.0, 'wall_s': 10.0,
                                'parity': {'exact': 1, 'compared': 4}},
                    'metrics_before': METRICS_BEFORE, 'metrics_after': METRICS_AFTER}))
            asr = root / 'asr' / 'Asr'
            asr.mkdir(parents=True)
            (asr / 'A-c1.json').write_text(json.dumps({
                'summary': {'requests': 2, 'completed': 2, 'audio_throughput': 100.0,
                            'latency_s': {'p50': 0.1, 'p95': 0.2}, 'duration_s': 2.0,
                            'parity': {'checked': 0, 'failed': 0}},
                'delta': {'izwi_engine_executor_physical_dispatch_seconds_total': 1.5,
                          'izwi_asr_phase_seconds_sum{phase="encoder"}': 0.4}}))
            rows = analyzer.analyze(root)
        self.assertEqual([(kind, name) for kind, _, name, _, _ in rows],
                         [('asr', 'A-c1'), ('tts', 'S-c2'), ('tts', 'S-c16')])
        asr_row = rows[0][3]
        self.assertEqual((asr_row['util'], asr_row['asr_phase_s']), (0.75, {'encoder': 0.4}))
        tts_row = rows[1][3]
        self.assertEqual((tts_row['busy_s'], tts_row['util'], tts_row['parity']), (8.0, 0.8, '1/4'))
        self.assertIn('| Model | S-c2 | 7.5 |', analyzer.markdown(rows))


class Agreement(unittest.TestCase):
    def test_word_errors_ignore_case_and_punctuation(self):
        self.assertEqual(agreement.word_errors('Hello, World.', 'hello world'), (0, 2))
        self.assertEqual(agreement.word_errors('a b c d', 'a x c'), (2, 4))

    def test_disagreement_covers_only_shared_inputs(self):
        a = {('s', 0.0, 5.0): 'one two three four', ('s', 5.0, 5.0): 'five'}
        b = {('s', 0.0, 5.0): 'one two three for', ('t', 0.0, 5.0): 'other'}
        self.assertEqual(agreement.disagreement(a, b), (25.0, 1))
        self.assertEqual(agreement.disagreement(a, {}), (None, 0))


if __name__ == '__main__':
    unittest.main()
