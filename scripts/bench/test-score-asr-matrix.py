#!/usr/bin/env python3
"""Deterministic checks for the ASR matrix scorer."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'scorer', pathlib.Path(__file__).with_name('score-asr-matrix.py'))
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

ROOT = pathlib.Path(__file__).resolve().parents[2]


class Scorer(unittest.TestCase):
    def score_one(self, fixture, body):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'response.json'
            path.write_text(body if isinstance(body, str) else json.dumps(body))
            row = {'kind': 'transcription', 'model': 'm', 'fixture': fixture,
                   'http': '200', 'secs': 1.0, 'body': str(path)}
            return scorer.score([row], ROOT)['transcriptions'][0]

    def test_wer_ignores_case_and_punctuation(self):
        self.assertEqual(scorer.wer('The quick, brown fox.', 'the quick brown fox'), 0.0)
        self.assertAlmostEqual(scorer.wer('a b c d', 'a x c'), 0.5)

    def test_reference_fixture_reports_wer(self):
        entry = self.score_one('fox', {'text': 'The quick brown fox jumps over the lazy dog.',
                                       'duration': 3.6, 'rtf': 0.1})
        self.assertEqual(entry['wer'], 0.0)
        self.assertEqual(entry['duration'], 3.6)

    def test_silence_must_be_empty(self):
        self.assertFalse(self.score_one('silence', {'text': 'you'})['empty_as_expected'])
        self.assertTrue(self.score_one('noise', {'text': ' '})['empty_as_expected'])

    def test_counts_chunks_that_hit_the_token_budget(self):
        decode = lambda reason: {'model_diagnostics': {'decode': {'stop_reason': reason}}}
        body = {'text': 'x', 'izwi_asr_diagnostics': {'chunking': {'chunk_transcriptions': [
            decode('max_tokens'), decode('eos'), decode('max_tokens')]}}}
        entry = self.score_one('diar116', body)
        self.assertEqual((entry['capped_chunks'], entry['chunks']), (2, 3))

    def test_stream_scores_final_event_and_checks_deltas(self):
        events = [{'type': 'transcript.text.delta', 'delta': 'very expensive. '},
                  {'type': 'transcript.text.delta', 'delta': 'Expensive, and more'},
                  {'type': 'transcript.text.done', 'text': 'very expensive, and more'}]
        body = '\n\n'.join('data: ' + json.dumps(event) for event in events)
        entry = self.score_one('diar60', body)
        self.assertEqual(entry['words'], 4)
        self.assertFalse(entry['stream_matches_final'])

    def test_errors_are_reported_not_scored(self):
        entry = self.score_one('test', {'error': {'message': 'boom'}})
        self.assertIn('boom', entry['error'])


if __name__ == '__main__':
    unittest.main()
