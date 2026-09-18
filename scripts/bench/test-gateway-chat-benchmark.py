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
    mode = "json"  # or "stream", "reject", "error"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        payload = json.loads(body.decode("utf-8"))
        assert "model" in payload
        assert "messages" in payload

        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            self.send_response(401)
            self.end_headers()
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


if __name__ == "__main__":
    unittest.main()
