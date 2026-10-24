#!/usr/bin/env python3
"""Unit tests for run-audio-load.py: input generation, WAV handling, parity
checks, and end-to-end runs against a local fake server that can corrupt a
response whenever requests overlap, the way a batching bug would."""
import contextlib
import http.server
import importlib.util
import io
import json
import math
import pathlib
import random
import tempfile
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location(
    "audio_load", pathlib.Path(__file__).with_name("run-audio-load.py"))
load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load)


def tone(rate, secs, freq=440.0):
    return [int(8000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(int(rate * secs))]


class Inputs(unittest.TestCase):
    def test_clips_are_seeded_bounded_and_distinct(self):
        sources = [("long.wav", 16_000, tone(16_000, 20.0)), ("short.wav", 8_000, tone(8_000, 1.0))]
        clips = load.cut_clips(sources, 40, 2.0, 6.0, random.Random(3))
        again = load.cut_clips(sources, 40, 2.0, 6.0, random.Random(3))
        self.assertEqual([(c["source"], c["offset_secs"], c["audio_secs"]) for c in clips],
                         [(c["source"], c["offset_secs"], c["audio_secs"]) for c in again])
        for clip in clips:
            if clip["source"] == "short.wav":
                self.assertEqual(clip["audio_secs"], 1.0, "a short source is used whole")
            else:
                self.assertTrue(2.0 <= clip["audio_secs"] <= 6.0)
                self.assertLessEqual(clip["offset_secs"] + clip["audio_secs"], 20.0)
            self.assertEqual(len(clip["samples"]), round(clip["audio_secs"] * clip["rate"]))
        long_clips = {(c["offset_secs"], c["audio_secs"]) for c in clips if c["source"] == "long.wav"}
        self.assertGreater(len(long_clips), 10, "clips differ from one another")

    def test_tts_inputs_have_mixed_lengths_in_range(self):
        texts = load.tts_inputs(50, 4, 40, random.Random(1))
        counts = [len(text.split()) for text in texts]
        self.assertTrue(all(4 <= count <= 40 for count in counts))
        self.assertGreater(len(set(counts)), 10)
        self.assertTrue(all(text.endswith(".") and text[0].isupper() for text in texts))
        self.assertEqual(texts, load.tts_inputs(50, 4, 40, random.Random(1)))

    def test_bad_ranges_are_rejected(self):
        with self.assertRaises(ValueError):
            load.cut_clips([("a", 16_000, tone(16_000, 1.0))], 1, 5.0, 2.0, random.Random(0))
        with self.assertRaises(ValueError):
            load.tts_inputs(1, 0, 3, random.Random(0))


class Audio(unittest.TestCase):
    def test_wav_round_trip(self):
        samples = tone(24_000, 0.25)
        rate, decoded = load.wav_samples(load.wav_bytes(24_000, samples))
        self.assertEqual(rate, 24_000)
        self.assertEqual(list(decoded), samples)

    def test_stereo_is_averaged_to_mono(self):
        buffer = io.BytesIO()
        import wave
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(8_000)
            handle.writeframes(b"".join(
                left.to_bytes(2, "little", signed=True) + right.to_bytes(2, "little", signed=True)
                for left, right in [(100, 300), (-200, 0)]))
        _, mono = load.wav_samples(buffer.getvalue())
        self.assertEqual(list(mono), [200, -100])


class Parity(unittest.TestCase):
    def test_asr_exact_and_normalized(self):
        check = load.asr_parity({"text": "Hello, world!"}, {"text": "hello world"})
        self.assertEqual(check, {"exact": False, "normalized": True})
        self.assertTrue(load.asr_parity({"text": "a b"}, {"text": "a b"})["exact"])

    def test_tts_seeds_skip_the_clock_seeded_zero(self):
        self.assertEqual([load.request_seed(0, index) for index in range(3)], [1, 2, 3])

    def test_tts_tolerance_and_length(self):
        solo = {"samples": [0, 100, -100]}
        self.assertTrue(load.tts_parity(solo, {"samples": [0, 100, -100]}, 0.01)["exact"])
        near = load.tts_parity(solo, {"samples": [0, 200, -100]}, 0.01)
        self.assertEqual((near["exact"], near["close"]), (False, True))
        far = load.tts_parity(solo, {"samples": [0, 1000, -100]}, 0.01)
        self.assertFalse(far["close"])
        shorter = load.tts_parity(solo, {"samples": [0, 100]}, 0.01)
        self.assertEqual((shorter["close"], shorter["length_delta"]), (False, -1))


class FakeAudioServer(http.server.ThreadingHTTPServer):
    """Deterministic transcription and speech. With `corrupt_overlap`, a
    request that runs while another is in flight gets a wrong answer."""

    daemon_threads = True

    def __init__(self, corrupt_overlap, unseeded):
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.corrupt_overlap = corrupt_overlap
        self.unseeded = unseeded
        self.lock = threading.Lock()
        self.in_flight = 0
        self.overlapped = 0


class FakeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        server = self.server
        with server.lock:
            server.in_flight += 1
            overlapped = server.in_flight > 1
        time.sleep(0.05)
        try:
            corrupt = server.corrupt_overlap and overlapped
            server.overlapped += overlapped
            if self.path == "/v1/audio/transcriptions":
                self.transcription(body, corrupt)
            elif self.path == "/v1/audio/speech":
                self.speech(json.loads(body), corrupt)
            else:
                self.send_response(404)
                self.end_headers()
        finally:
            with server.lock:
                server.in_flight -= 1

    def transcription(self, body, corrupt):
        boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
        part = body.split(b"filename=", 1)[1].split(b"\r\n\r\n", 1)[1]
        wav = part.rsplit(b"\r\n--" + boundary + b"--", 1)[0]
        _, samples = load.wav_samples(wav)
        text = f"clip {len(samples)} {sum(samples) % 9973}" + (" extra" if corrupt else "")
        self.reply("application/json", json.dumps({"text": text}).encode())

    def speech(self, request, corrupt):
        rng = random.Random() if self.server.unseeded else random.Random(
            f"{request['input']}|{request['seed']}")
        samples = [rng.randint(-1000, 1000) for _ in range(40 * len(request["input"].split()))]
        if corrupt:
            samples[0] += 5000
        self.reply("audio/wav", load.wav_bytes(24_000, samples))

    def reply(self, content_type, payload):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextlib.contextmanager
def fake_server(corrupt_overlap=False, unseeded=False):
    server = FakeAudioServer(corrupt_overlap, unseeded)
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
        self.audio = pathlib.Path(self.tmp.name) / "source.wav"
        self.audio.write_bytes(load.wav_bytes(16_000, tone(16_000, 12.0)))
        self.output = pathlib.Path(self.tmp.name) / "result.json"

    def run_load(self, base_url, task, *extra):
        argv = ["--task", task, "--base-url", base_url, "--model", "fake", "--num-requests", "6",
                "--request-rate", "inf", "--output", str(self.output), *extra]
        if task == "asr":
            argv += ["--audio", str(self.audio), "--min-secs", "1", "--max-secs", "4"]
        with contextlib.redirect_stdout(io.StringIO()):
            status = load.main(argv)
        return status, json.loads(self.output.read_text())

    def test_asr_parity_holds_on_an_honest_server(self):
        with fake_server() as (server, base_url):
            status, result = self.run_load(base_url, "asr", "--require-parity")
        self.assertEqual(status, 0)
        self.assertGreater(server.overlapped, 0, "the load overlapped requests")
        self.assertEqual(result["schema"], "izwi.audio_load")
        summary = result["reports"][0]["report"]["summary"]
        self.assertEqual((summary["completed"], summary["errors"]), (6, 0))
        self.assertEqual(summary["parity"]["checked"], 6)
        self.assertEqual(summary["parity"]["failed"], 0)
        self.assertGreater(summary["audio_throughput"], 0)
        self.assertEqual(len(result["config"]["inputs"]), 6)

    def test_asr_overlap_corruption_fails_parity(self):
        with fake_server(corrupt_overlap=True) as (_, base_url):
            status, result = self.run_load(base_url, "asr", "--require-parity")
        self.assertEqual(status, load.PARITY_FAILURE_EXIT)
        parity = result["reports"][0]["report"]["summary"]["parity"]
        self.assertGreater(parity["failed"], 0)
        self.assertEqual(result["solo"]["summary"]["errors"], 0)

    def test_tts_parity_holds_and_catches_overlap_corruption(self):
        with fake_server() as (_, base_url):
            status, result = self.run_load(base_url, "tts", "--require-parity")
        self.assertEqual(status, 0)
        parity = result["reports"][0]["report"]["summary"]["parity"]
        self.assertEqual((parity["exact"], parity["close"], parity["failed"]), (6, 6, 0))
        self.assertIs(result["solo"]["repeatable"], True)

        with fake_server(corrupt_overlap=True) as (_, base_url):
            status, result = self.run_load(base_url, "tts", "--require-parity")
        self.assertEqual(status, load.PARITY_FAILURE_EXIT)
        self.assertGreater(result["reports"][0]["report"]["summary"]["parity"]["failed"], 0)

    def test_a_model_that_does_not_repeat_is_not_held_to_parity(self):
        with fake_server(unseeded=True) as (_, base_url):
            status, result = self.run_load(base_url, "tts")
        self.assertEqual(status, 0)
        self.assertIs(result["solo"]["repeatable"], False)
        self.assertNotIn("parity", result["reports"][0]["report"]["summary"])

        with fake_server(unseeded=True) as (_, base_url):
            status, _ = self.run_load(base_url, "tts", "--require-parity")
        self.assertEqual(status, load.PARITY_FAILURE_EXIT)

    def test_report_only_without_require_parity(self):
        with fake_server(corrupt_overlap=True) as (_, base_url):
            status, _ = self.run_load(base_url, "tts")
        self.assertEqual(status, 0)

    def test_server_errors_are_counted_not_raised(self):
        with fake_server() as (_, base_url):
            status, result = self.run_load(base_url + "/missing", "tts", "--warmup", "0", "--no-solo")
        self.assertEqual(status, 0)
        summary = result["reports"][0]["report"]["summary"]
        self.assertEqual((summary["completed"], summary["errors"]), (0, 6))
        self.assertIn("HTTP 404", summary["error_samples"][0])


if __name__ == "__main__":
    unittest.main()
