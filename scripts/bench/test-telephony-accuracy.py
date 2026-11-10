#!/usr/bin/env python3
"""Unit tests for run-telephony-accuracy.py: the G.711 codec, the telephone
band filter, reference loading, scoring, and end-to-end runs against a local
fake server that hears telephone audio worse than the original."""
import contextlib
import http.server
import importlib.util
import io
import json
import math
import pathlib
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location(
    "telephony_accuracy", pathlib.Path(__file__).with_name("run-telephony-accuracy.py"))
telephony = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telephony)
load = telephony.load

REFERENCE = "The quick brown fox jumps over the lazy dog."


def tone(rate, secs, freq, amplitude=8000):
    return [int(amplitude * math.sin(2 * math.pi * freq * i / rate)) for i in range(int(rate * secs))]


def rms(samples):
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class Codec(unittest.TestCase):
    def test_codes_match_the_standard_anchors(self):
        self.assertEqual([telephony.linear_to_ulaw(s) for s in (0, 32_767, -32_768)], [0xFF, 0x80, 0x00])
        self.assertEqual([telephony.linear_to_alaw(s) for s in (0, 32_767, -32_768)], [0xD5, 0xAA, 0x2A])

    def test_round_trips_within_one_quantization_step(self):
        for sample in range(-32_768, 32_768, 7):
            for name, encode, decode in (
                    ("ulaw", telephony.linear_to_ulaw, telephony.ulaw_to_linear),
                    ("alaw", telephony.linear_to_alaw, telephony.alaw_to_linear)):
                code = encode(sample)
                self.assertTrue(0 <= code <= 0xFF)
                error = abs(decode(code) - sample)
                self.assertLessEqual(error, max(abs(sample) // 16, 140), f"{name} {sample}")

    def test_companding_keeps_length_and_changes_samples(self):
        samples = tone(8_000, 0.1, 440.0)
        for encode, decode in ((telephony.linear_to_ulaw, telephony.ulaw_to_linear),
                               (telephony.linear_to_alaw, telephony.alaw_to_linear)):
            companded = telephony.compand(samples, encode, decode)
            self.assertEqual(len(companded), len(samples))
            self.assertNotEqual(companded, samples)


class TelephoneBand(unittest.TestCase):
    def test_speech_band_passes_and_the_band_above_is_removed(self):
        for rate in (16_000, 24_000, 44_100):
            passed = telephony.telephone_band(rate, tone(rate, 0.5, 1_000.0))
            self.assertEqual(len(passed), 4_000, rate)
            self.assertAlmostEqual(rms(passed[400:-400]) / (8000 / math.sqrt(2)), 1.0, delta=0.03)
            removed = telephony.telephone_band(rate, tone(rate, 0.5, 6_000.0))
            self.assertLess(rms(removed[400:-400]), 80, rate)

    def test_level_is_kept_and_8_khz_input_keeps_its_length(self):
        flat = telephony.telephone_band(16_000, [1000] * 3_200)
        self.assertTrue(all(abs(sample - 1000) <= 1 for sample in flat[100:-100]))
        self.assertEqual(len(telephony.telephone_band(8_000, tone(8_000, 0.25, 500.0))), 2_000)

    def test_conditions_present_a_clip_as_named(self):
        samples = tone(16_000, 0.25, 500.0)
        self.assertEqual(telephony.condition_audio("original", 16_000, samples), (16_000, samples))
        rate, narrow = telephony.condition_audio("narrowband", 16_000, samples)
        self.assertEqual((rate, len(narrow)), (8_000, 2_000))
        for law in ("ulaw", "alaw"):
            rate, companded = telephony.condition_audio(law, 16_000, samples)
            self.assertEqual((rate, len(companded)), (8_000, 2_000))
            self.assertNotEqual(companded, narrow)
            self.assertAlmostEqual(rms(companded) / rms(narrow), 1.0, delta=0.05)


class Scoring(unittest.TestCase):
    def test_edit_distance_counts_each_kind_of_error(self):
        self.assertEqual(telephony.edit_distance("abcd", "abcd"), 0)
        self.assertEqual(telephony.edit_distance(["a", "b", "c", "d"], ["a", "x", "c"]), 2)
        self.assertEqual(telephony.edit_distance([], ["a", "b"]), 2)

    def test_clip_errors_agree_with_the_matrix_scorer(self):
        errors = telephony.clip_errors("The quick, brown fox.", "the quik brown fox")
        self.assertEqual((errors["word_errors"], errors["reference_words"]), (1, 4))
        self.assertEqual((errors["char_errors"], errors["reference_chars"]), (1, 19))
        self.assertAlmostEqual(errors["word_errors"] / errors["reference_words"],
                               telephony.scoring.wer("The quick, brown fox.", "the quik brown fox"))

    def test_rates_are_over_the_whole_set_and_skip_failed_clips(self):
        records = [
            {"clip": "a", **telephony.clip_errors("one two three four", "one two three four")},
            {"clip": "b", **telephony.clip_errors("five six", "five")},
            {"clip": "c", "error": "HTTP 500"},
        ]
        summary = telephony.summarize(records)
        self.assertEqual((summary["clips"], summary["failed"]), (2, 1))
        self.assertAlmostEqual(summary["wer"], 1 / 6)
        self.assertAlmostEqual(summary["cer"], 4 / 26)
        self.assertIsNone(telephony.summarize([{"clip": "c", "error": "x"}])["wer"])

    def test_degradation_is_absolute_and_relative(self):
        fall = telephony.degradation({"wer": 0.10, "cer": 0.04}, {"wer": 0.15, "cer": 0.05})
        self.assertAlmostEqual(fall["wer_absolute"], 0.05)
        self.assertAlmostEqual(fall["wer_relative"], 0.5)
        self.assertAlmostEqual(fall["cer_relative"], 0.25)
        perfect = telephony.degradation({"wer": 0.0, "cer": 0.0}, {"wer": 0.2, "cer": 0.1})
        self.assertAlmostEqual(perfect["wer_absolute"], 0.2)
        self.assertIsNone(perfect["wer_relative"], "nothing is relative to a zero baseline")
        missing = telephony.degradation({"wer": None, "cer": None}, {"wer": 0.2, "cer": 0.1})
        self.assertIsNone(missing["wer_absolute"])


class References(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        for name in ("a", "b", "untranscribed"):
            (self.root / f"{name}.wav").write_bytes(load.wav_bytes(16_000, tone(16_000, 0.1, 440.0)))
        (self.root / "a.txt").write_text("first\nclip\n")
        (self.root / "b.md").write_text("second clip")

    def test_a_directory_pairs_each_wav_with_its_transcript(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            clips = telephony.directory_clips(self.root)
        self.assertEqual([(clip["name"], clip["reference"]) for clip in clips],
                         [("a", "first clip"), ("b", "second clip")])
        self.assertIn("untranscribed.wav", stderr.getvalue())

    def test_a_manifest_names_text_or_a_text_file(self):
        manifest = self.root / "set.jsonl"
        manifest.write_text("\n".join([
            json.dumps({"audio": "a.wav", "text": "inline  text"}),
            "",
            json.dumps({"audio": "b.wav", "text_file": "b.md", "name": "second"}),
        ]))
        clips = telephony.manifest_clips(manifest)
        self.assertEqual([(clip["name"], clip["reference"]) for clip in clips],
                         [("a", "inline text"), ("second", "second clip")])
        self.assertEqual(clips[0]["audio"], str(self.root / "a.wav"))
        manifest.write_text(json.dumps({"audio": "a.wav"}))
        with self.assertRaises(ValueError):
            telephony.manifest_clips(manifest)


class FakeHandler(http.server.BaseHTTPRequestHandler):
    """Transcribes wideband audio exactly and 8 kHz audio with one word wrong;
    the model `broken` fails."""

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
        model = body.split(b'name="model"\r\n\r\n', 1)[1].split(b"\r\n", 1)[0].decode()
        part = body.split(b"filename=", 1)[1].split(b"\r\n\r\n", 1)[1]
        rate, _ = load.wav_samples(part.rsplit(b"\r\n--" + boundary + b"--", 1)[0])
        self.server.rates.append(rate)
        if model == "broken":
            self.send_response(500)
            self.end_headers()
            return
        text = REFERENCE if rate != 8_000 else REFERENCE.replace("quick", "thick")
        payload = json.dumps({"text": text}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextlib.contextmanager
def fake_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    server.daemon_threads = True
    server.rates = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = pathlib.Path(self.tmp.name)
        for name in ("one", "two"):
            (root / f"{name}.wav").write_bytes(load.wav_bytes(16_000, tone(16_000, 0.3, 440.0)))
            (root / f"{name}.txt").write_text(REFERENCE)
        self.audio = root
        self.output = root / "result.json"

    def run_script(self, base_url, *extra):
        argv = ["--audio", str(self.audio), "--base-url", base_url, "--output", str(self.output), *extra]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            status = telephony.main(argv)
        return status, json.loads(self.output.read_text()), stdout.getvalue()

    def test_each_model_is_scored_under_each_condition(self):
        with fake_server() as (server, base_url):
            status, result, table = self.run_script(base_url, "--model", "fake,other", "--save-samples")
        self.assertEqual(status, 0)
        self.assertEqual((result["schema"], result["schema_version"]), ("izwi.telephony_accuracy", 1))
        self.assertEqual(result["config"]["conditions"], ["original", "ulaw", "alaw"])
        self.assertEqual([clip["reference_words"] for clip in result["config"]["clips"]], [9, 9])
        # Two models, three conditions, two clips: the original at its own
        # rate, both laws at 8 kHz.
        self.assertEqual(sorted(server.rates), [8_000] * 8 + [16_000] * 4)
        for model in ("fake", "other"):
            report = result["models"][model]
            self.assertEqual(report["conditions"]["original"]["wer"], 0.0)
            for law in ("ulaw", "alaw"):
                self.assertEqual(report["conditions"][law]["clips"], 2)
                self.assertAlmostEqual(report["conditions"][law]["wer"], 1 / 9)
                self.assertAlmostEqual(report["degradation"][law]["wer_absolute"], 1 / 9)
                self.assertIsNone(report["degradation"][law]["wer_relative"])
            self.assertEqual(report["samples"]["ulaw"][0]["text"], REFERENCE.replace("quick", "thick"))
        self.assertIn("WER vs original", table)
        self.assertIn("+11.11pt", table)

    def test_conditions_are_selectable_and_checked(self):
        with fake_server() as (server, base_url):
            status, result, _ = self.run_script(base_url, "--model", "fake",
                                                "--conditions", "narrowband,ulaw")
        self.assertEqual(status, 0)
        self.assertEqual(sorted(result["models"]["fake"]["conditions"]), ["narrowband", "ulaw"])
        self.assertNotIn("degradation", result["models"]["fake"], "nothing to compare without the original")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            telephony.parse_args(["--audio", "x", "--model", "m", "--conditions", "gsm"])

    def test_failed_clips_are_counted_and_fail_the_run(self):
        with fake_server() as (_, base_url):
            status, result, table = self.run_script(base_url, "--model", "broken")
        self.assertEqual(status, 1)
        summary = result["models"]["broken"]["conditions"]["ulaw"]
        self.assertEqual((summary["clips"], summary["failed"], summary["wer"]), (0, 2, None))
        self.assertIn("n/a", table)

    def test_a_directory_without_transcripts_is_refused(self):
        empty = pathlib.Path(self.tmp.name) / "empty"
        empty.mkdir()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(telephony.main(["--audio", str(empty), "--model", "m"]), 1)


if __name__ == "__main__":
    unittest.main()
