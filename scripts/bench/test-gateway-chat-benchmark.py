#!/usr/bin/env python3
"""Unit tests for run-gateway-chat-benchmark.py."""
import http.server
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).with_name("run-gateway-chat-benchmark.py")
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PercentileTests(unittest.TestCase):
    def test_percentile_empty(self):
        self.assertIsNone(benchmark.percentile([], 0.5))

    def test_percentile_exact(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(benchmark.percentile(values, 0.5), 30.0)
        self.assertEqual(benchmark.percentile(values, 0.0), 10.0)
        self.assertEqual(benchmark.percentile(values, 1.0), 50.0)

    def test_percentile_interpolated(self):
        values = [10.0, 20.0]
        self.assertEqual(benchmark.percentile(values, 0.5), 15.0)


class MockGatewayHandler(http.server.BaseHTTPRequestHandler):
    mode = "json"  # or "stream", "reject", "error", "flaky"
    last_payload = None
    attempts = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        payload = json.loads(body.decode("utf-8"))
        assert "model" in payload
        assert "messages" in payload
        MockGatewayHandler.last_payload = payload

        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            self.send_response(401)
            self.end_headers()
            return

        if self.mode == "flaky":
            MockGatewayHandler.attempts += 1
            if MockGatewayHandler.attempts < 2:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"busy","type":"overloaded"}}')
                return

        if self.mode == "reject":
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"rate limited","type":"rate_limit_error"}}')
            return

        if self.mode == "error":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"internal error","type":"server_error"}}')
            return

        if self.mode == "stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = (
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"ready"}}]}\n\n'
                b'data: [DONE]\n\n'
            )
            self.wfile.write(chunk)
            return

        # default json
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json.dumps({
            "id": "chatcmpl-1",
            "choices": [{"message": {"role": "assistant", "content": "ready"}}],
        }).encode("utf-8")
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass  # suppress logging during tests


class WorkloadMessageTests(unittest.TestCase):
    DEFAULT_VOCAB = ["a", "b", "c"]

    def test_default_messages_are_unchanged(self):
        self.assertEqual(
            benchmark.build_messages("default", 64, 8, 7),
            [{"role": "user", "content": "Say the word ready."}],
        )

    def test_shared_prefix_is_identical_and_suffix_unique(self):
        first = benchmark.build_messages("shared", 24, 6, 1)
        second = benchmark.build_messages("shared", 24, 6, 2)
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertEqual(first[0]["content"], second[0]["content"])
        self.assertNotEqual(first[1]["content"], second[1]["content"])
        self.assertEqual(len(first[0]["content"].split()), 24)
        self.assertEqual(len(first[1]["content"].split()), 6)

    def test_cold_messages_are_unique_same_length(self):
        first = benchmark.build_messages("cold", 24, 6, 1)
        second = benchmark.build_messages("cold", 24, 6, 2)
        self.assertEqual([m["role"] for m in first], ["user"])
        self.assertNotEqual(first[0]["content"], second[0]["content"])
        self.assertEqual(len(first[0]["content"].split()), 30)
        self.assertEqual(len(second[0]["content"].split()), 30)

    def test_build_messages_rejects_unknown_workload(self):
        with self.assertRaises(ValueError):
            benchmark.build_messages("bursty", 4, 4, 1)

    def test_fixture_vocab_keeps_cold_common_prefix_below_one_page(self):
        # The DS1.5 fixture tokenizer only knows a/b/c; cold prompts must stay
        # unique through a positional marker whose shared prefix stays below
        # one KV page (16 tokens) for every request pair.
        vocab = self.DEFAULT_VOCAB
        marker_len = benchmark.marker_word_count(vocab)
        self.assertEqual(marker_len, 11)
        for index in (0, 1, 2, 3, 17, 1000):
            messages = benchmark.build_messages("cold", 24, 8, index, vocab)
            words = messages[0]["content"].split()
            self.assertEqual(len(words), 32)
            self.assertEqual(words[:marker_len], benchmark._marker_words(index, vocab))

    def test_fixture_vocab_rejects_suffix_below_marker_floor(self):
        with self.assertRaises(ValueError):
            benchmark.build_messages("shared", 24, 6, 1, self.DEFAULT_VOCAB)
        # One word above the marker floor is accepted.
        messages = benchmark.build_messages(
            "shared", 24, benchmark.marker_word_count(self.DEFAULT_VOCAB) + 1,
            1, self.DEFAULT_VOCAB,
        )
        self.assertEqual(len(messages[1]["content"].split()), 12)

    def test_shared_fixture_vocab_suffixes_diverge_after_the_prefix(self):
        vocab = self.DEFAULT_VOCAB
        first = benchmark.build_messages("shared", 64, 12, 1, vocab)
        second = benchmark.build_messages("shared", 64, 12, 2, vocab)
        self.assertEqual(first[0]["content"], second[0]["content"])
        self.assertNotEqual(first[1]["content"], second[1]["content"])


class GatewayBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), MockGatewayHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)

    def test_run_one_json_success(self):
        MockGatewayHandler.mode = "json"
        ok, rejected, ttft_ms, latency_ms, detail = benchmark.run_one(
            f"http://127.0.0.1:{self.port}", "test-api-key", "test-model", 32, False, 1
        )
        self.assertTrue(ok)
        self.assertFalse(rejected)
        self.assertIsNone(ttft_ms)  # non-streaming has no ttft
        self.assertGreater(latency_ms, 0)
        self.assertIsNone(detail)

    def test_run_one_stream_success(self):
        MockGatewayHandler.mode = "stream"
        ok, rejected, ttft_ms, latency_ms, detail = benchmark.run_one(
            f"http://127.0.0.1:{self.port}", "test-api-key", "test-model", 32, True, 1
        )
        self.assertTrue(ok)
        self.assertFalse(rejected)
        self.assertIsNotNone(ttft_ms)
        self.assertGreater(latency_ms, 0)
        self.assertIsNone(detail)

    def test_run_one_rejection(self):
        MockGatewayHandler.mode = "reject"
        ok, rejected, ttft_ms, latency_ms, detail = benchmark.run_one(
            f"http://127.0.0.1:{self.port}", "test-api-key", "test-model", 32, False, 1
        )
        self.assertFalse(ok)
        self.assertTrue(rejected)
        self.assertEqual(detail, 429)

    def test_run_one_error(self):
        MockGatewayHandler.mode = "error"
        ok, rejected, ttft_ms, latency_ms, detail = benchmark.run_one(
            f"http://127.0.0.1:{self.port}", "test-api-key", "test-model", 32, False, 1
        )
        self.assertFalse(ok)
        self.assertFalse(rejected)
        self.assertEqual(detail, 500)

    def test_run_one_shared_workload_sends_system_and_unique_user(self):
        MockGatewayHandler.mode = "json"
        ok, rejected, ttft_ms, latency_ms, detail = benchmark.run_one(
            f"http://127.0.0.1:{self.port}", "test-api-key", "test-model", 32,
            False, 3, workload="shared", prefix_tokens=8, suffix_tokens=6,
        )
        self.assertTrue(ok)
        payload = MockGatewayHandler.last_payload
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        self.assertEqual(len(payload["messages"][0]["content"].split()), 8)
        self.assertEqual(len(payload["messages"][1]["content"].split()), 6)
        self.assertTrue(
            all(word in benchmark.BENCH_VOCAB
                for word in payload["messages"][1]["content"].split())
        )

    def test_run_one_retries_rejection_then_succeeds(self):
        MockGatewayHandler.mode = "flaky"
        MockGatewayHandler.attempts = 0
        ok, rejected, ttft_ms, latency_ms, detail = benchmark.run_one(
            f"http://127.0.0.1:{self.port}", "test-api-key", "test-model", 32,
            False, 1, max_retries=3,
        )
        self.assertTrue(ok)
        self.assertFalse(rejected)
        self.assertEqual(MockGatewayHandler.attempts, 2)

    def test_run_one_rejection_exhausts_retries(self):
        MockGatewayHandler.mode = "reject"
        ok, rejected, ttft_ms, latency_ms, detail = benchmark.run_one(
            f"http://127.0.0.1:{self.port}", "test-api-key", "test-model", 32,
            False, 1, max_retries=2,
        )
        self.assertFalse(ok)
        self.assertTrue(rejected)
        self.assertEqual(detail, 429)

    def test_run_one_passes_vocab_through(self):
        MockGatewayHandler.mode = "json"
        ok, rejected, ttft_ms, latency_ms, detail = benchmark.run_one(
            f"http://127.0.0.1:{self.port}", "test-api-key", "test-model", 32,
            False, 1, workload="cold", prefix_tokens=12, suffix_tokens=4,
            vocab=["a", "b", "c"],
        )
        self.assertTrue(ok)
        payload = MockGatewayHandler.last_payload
        self.assertEqual([m["role"] for m in payload["messages"]], ["user"])
        self.assertTrue(
            all(word in ("a", "b", "c")
                for word in payload["messages"][0]["content"].split())
        )
        self.assertEqual(len(payload["messages"][0]["content"].split()), 16)


if __name__ == "__main__":
    unittest.main()
