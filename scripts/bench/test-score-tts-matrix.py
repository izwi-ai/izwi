#!/usr/bin/env python3
"""Deterministic checks for the TTS matrix client and scorer."""
import importlib.util
import json
import math
import pathlib
import struct
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location(
    'scorer', pathlib.Path(__file__).with_name('score-tts-matrix.py'))
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def tone_wav(path, rate=24000, seconds=1.0, lead=0.2, trail=0.3):
    """A 440 Hz tone with leading and trailing silence."""
    samples = []
    total = int(rate * (lead + seconds + trail))
    for i in range(total):
        t = i / rate
        voiced = lead <= t < lead + seconds
        samples.append(int(12000 * math.sin(2 * math.pi * 440 * t)) if voiced else 0)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(struct.pack('<%dh' % len(samples), *samples))


class Scorer(unittest.TestCase):
    def test_wer_ignores_case_punctuation_and_spells_small_numbers(self):
        self.assertEqual(scorer.wer('The quick, brown fox.', 'the quick brown fox'), 0.0)
        self.assertAlmostEqual(scorer.wer('a b c d', 'a x c'), 0.5)
        self.assertEqual(scorer.wer('3 items', 'three items'), 0.0)

    def test_audio_stats_measure_duration_and_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'tone.wav'
            tone_wav(path)
            stats = scorer.audio_stats(path)
        self.assertAlmostEqual(stats['duration_s'], 1.5, places=2)
        self.assertAlmostEqual(stats['lead_sil_s'], 0.2, delta=0.03)
        self.assertAlmostEqual(stats['trail_sil_s'], 0.3, delta=0.03)
        self.assertGreater(stats['peak'], 0.3)

    def test_streaming_wav_with_open_size_is_read(self):
        pcm = struct.pack('<4h', 1, 2, 3, 4)
        header = (b'RIFF' + struct.pack('<I', 0xFFFFFFFF) + b'WAVEfmt '
                  + struct.pack('<IHHIIHH', 16, 1, 1, 24000, 48000, 2, 16)
                  + b'data' + struct.pack('<I', 0xFFFFFFFF))
        payload, rate, channels = scorer.wav_payload(header + pcm)
        self.assertEqual((payload, rate, channels), (pcm, 24000, 1))

    def test_case_reports_budget_stop_and_decode_counts(self):
        record = {
            'case': 'med', 'model': 'Qwen3-TTS-12Hz-0.6B-CustomVoice', 'http': 200,
            'text_key': 'med', 'stream': False,
            'headers': {
                'x-stop-reason': 'length', 'x-tokens-generated': '400',
                'x-izwi-tts-diagnostics': json.dumps(
                    {'latency_breakdown': {'decode_steps': 400, 'decode_tokens': 400}}),
            },
        }
        row = scorer.score_case(record, None)
        self.assertEqual(row['stop_reason'], 'length')
        self.assertEqual((row['decode_steps'], row['decode_tokens']), (400, 400))

    def test_stream_terminal_event_and_stop_reason_come_from_sse(self):
        record = {'case': 's', 'model': 'm', 'http': 200, 'text_key': 'short', 'stream': True,
                  'events': [{'event': 'audio.chunk'},
                             {'event': 'audio.done', 'stop_reason': 'stop'}]}
        row = scorer.score_case(record, None)
        self.assertEqual(row['terminal_event'], 'audio.done')
        self.assertEqual(row['stop_reason'], 'stop')
        record['events'] = [{'event': 'audio.chunk'}]
        self.assertIsNone(scorer.score_case(record, None)['terminal_event'])

    def test_case_wer_uses_the_round_trip_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'short.wav'
            tone_wav(path)
            record = {'case': 'short', 'model': 'm', 'http': 200, 'text_key': 'short'}
            row = scorer.score_case(
                record, str(path),
                asr=lambda _: ('The quick brown fox jumps over the lazy dog.', None))
        self.assertEqual(row['wer'], 0.0)

    def test_every_family_has_cases_with_known_texts(self):
        for family in ('kokoro', 'lfm', 'qwen_cv', 'qwen_base', 'qwen_greedy', 'vibevoice',
                       'voxtral', 'fish'):
            for case_id, key, _, _ in scorer.family_cases(family, ('ref', 'text')):
                self.assertIn(key, scorer.TEXTS, (family, case_id))
        with self.assertRaises(SystemExit):
            scorer.family_cases('unknown', ('ref', 'text'))


if __name__ == '__main__':
    unittest.main()
