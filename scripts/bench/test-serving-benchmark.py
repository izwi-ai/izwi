#!/usr/bin/env python3
"""Unit tests for run-serving-benchmark.py: arrivals, statistics, SSE
parsing, goodput, Prometheus deltas, the result schema, the regression
comparison, and one end-to-end run against a local fake SSE server."""
import asyncio
import http.server
import importlib.util
import json
import math
import os
import pathlib
import random
import statistics
import tempfile
import threading
import time
import unittest
import unittest.mock

spec = importlib.util.spec_from_file_location(
    "serving", pathlib.Path(__file__).with_name("run-serving-benchmark.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Arrivals(unittest.TestCase):
    def gaps(self, rate, burstiness, count=20000, seed=7):
        offsets = bench.arrival_offsets(count, rate, burstiness, random.Random(seed))
        return [later - earlier for earlier, later in zip(offsets, offsets[1:])]

    def test_poisson_gaps_have_mean_one_over_rate_and_unit_cv(self):
        gaps = self.gaps(4.0, 1.0)
        mean = statistics.fmean(gaps)
        self.assertAlmostEqual(mean, 0.25, delta=0.25 * 0.03)
        self.assertAlmostEqual(statistics.pstdev(gaps) / mean, 1.0, delta=0.05)

    def test_burstiness_keeps_the_rate_and_changes_the_spread(self):
        regular = self.gaps(4.0, 4.0)
        self.assertAlmostEqual(statistics.fmean(regular), 0.25, delta=0.25 * 0.03)
        self.assertAlmostEqual(statistics.pstdev(regular) / statistics.fmean(regular), 0.5, delta=0.05)

    def test_offsets_start_at_zero_increase_and_repeat_with_the_seed(self):
        first = bench.arrival_offsets(50, 2.0, 1.0, random.Random(3))
        self.assertEqual(first[0], 0.0)
        self.assertEqual(first, sorted(first))
        self.assertEqual(first, bench.arrival_offsets(50, 2.0, 1.0, random.Random(3)))

    def test_infinite_rate_is_a_burst(self):
        self.assertEqual(bench.arrival_offsets(5, math.inf, 1.0, random.Random(0)), [0.0] * 5)
        self.assertEqual(bench.parse_rate("inf"), math.inf)
        with self.assertRaises(ValueError):
            bench.parse_rate("0")


class Statistics(unittest.TestCase):
    def test_percentiles_interpolate_linearly(self):
        values = list(range(1, 11))
        self.assertEqual(bench.percentile(values, 50), 5.5)
        self.assertAlmostEqual(bench.percentile(values, 90), 9.1)
        self.assertAlmostEqual(bench.percentile(values, 99), 9.91)
        self.assertEqual(bench.percentile([4.0], 99), 4.0)
        self.assertIsNone(bench.percentile([], 50))

    def test_distribution_reports_every_percentile(self):
        dist = bench.distribution([3.0, 1.0, 2.0])
        self.assertEqual((dist["count"], dist["min"], dist["max"], dist["mean"]), (3, 1.0, 3.0, 2.0))
        self.assertEqual(sorted(key for key in dist if key.startswith("p")), ["p50", "p90", "p95", "p99"])

    def test_low_sample_counts_warn_about_tail_percentiles(self):
        self.assertEqual(bench.min_samples_for(99), 100)
        self.assertEqual(bench.min_samples_for(95), 20)
        warning = bench.low_sample_warnings("ttft_ms", [1.0] * 24)
        self.assertEqual(len(warning), 1)
        self.assertIn("p99 needs at least 100", warning[0])
        self.assertNotIn("p95", warning[0])
        self.assertIn("p95", bench.low_sample_warnings("ttft_ms", [1.0] * 8)[0])
        self.assertEqual(bench.low_sample_warnings("ttft_ms", [1.0] * 100), [])
        self.assertEqual(bench.low_sample_warnings("ttft_ms", []), [])


STREAM = (
    b": keep-alive\r\n\r\n"
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n'
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\r\r'
    b"event: message\r\ndata: first line\r\ndata: second line\r\n\r\n"
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
    b"data: unterminated"
)
EXPECTED_EVENTS = [
    '{"choices":[{"delta":{"role":"assistant"}}]}',
    '{"choices":[{"delta":{"content":"Hel"}}]}',
    '{"choices":[{"delta":{"content":"lo"}}]}',
    "first line\nsecond line",
    '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
    "[DONE]",
]


class Sse(unittest.TestCase):
    def test_whole_stream(self):
        self.assertEqual(bench.SseParser().feed(STREAM), EXPECTED_EVENTS)

    def test_every_two_way_split_gives_the_same_events(self):
        for cut in range(len(STREAM) + 1):
            parser = bench.SseParser()
            events = parser.feed(STREAM[:cut]) + parser.feed(STREAM[cut:])
            self.assertEqual(events, EXPECTED_EVENTS, f"split at {cut}")

    def test_byte_at_a_time(self):
        parser = bench.SseParser()
        events = []
        for index in range(len(STREAM)):
            events += parser.feed(STREAM[index:index + 1])
        self.assertEqual(events, EXPECTED_EVENTS)

    def test_split_multibyte_character(self):
        payload = "data: café\n\n".encode("utf-8")
        parser = bench.SseParser()
        self.assertEqual(parser.feed(payload[:9]) + parser.feed(payload[9:]), ["café"])

    def test_oversized_line_is_rejected(self):
        with self.assertRaises(ValueError):
            bench.SseParser(max_event_bytes=16).feed(b"data: " + b"x" * 64)


def trace_of(events):
    trace = bench.StreamTrace()
    for data, at in events:
        trace.on_event(data, at)
    return trace


def delta(text):
    return json.dumps({"choices": [{"delta": {"content": text}}]})


class Records(unittest.TestCase):
    def test_token_times_give_ttft_itl_and_tpot(self):
        trace = trace_of([
            (json.dumps({"choices": [{"delta": {"role": "assistant"}}]}), 10.05),
            (delta("a"), 10.1), (delta("b"), 10.15), (delta("c"), 10.3),
            (json.dumps({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4}}), 10.31),
            ("[DONE]", 10.31),
        ])
        record = bench.request_record(0, {"max_tokens": 8}, trace, 10.0, 10.4, 9.0)
        self.assertTrue(record["ok"])
        self.assertAlmostEqual(record["ttft_ms"], 100.0)
        self.assertEqual([round(gap, 6) for gap in record["itl_ms"]], [50.0, 150.0])
        self.assertEqual((record["input_tokens"], record["output_tokens"]), (7, 4))
        self.assertEqual(record["output_tokens_source"], "usage")
        self.assertAlmostEqual(record["tpot_ms"], (400.0 - 100.0) / 3)
        self.assertAlmostEqual(record["start_offset_s"], 1.0)

    def test_without_usage_tokens_are_counted_from_deltas(self):
        trace = trace_of([(delta("a"), 1.0), (delta("b"), 2.0), ("[DONE]", 2.0)])
        record = bench.request_record(0, {"max_tokens": 8}, trace, 0.0, 2.0, 0.0)
        self.assertEqual((record["output_tokens"], record["output_tokens_source"]), (2, "deltas"))

    def test_error_event_and_missing_done(self):
        error = trace_of([(delta("a"), 1.0),
                          (json.dumps({"error": {"type": "server_error", "code": "overloaded"}}), 2.0)])
        self.assertEqual(bench.request_record(0, {"max_tokens": 8}, error, 0, 2, 0)["error"],
                         "stream_error:overloaded")
        truncated = trace_of([(delta("a"), 1.0)])
        self.assertEqual(bench.request_record(0, {"max_tokens": 8}, truncated, 0, 2, 0)["error"], "truncated")


def record(ttft, tpot, e2e, ok=True, out=10, itl=(), error=None):
    return {"ok": ok, "error": error, "ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e,
            "itl_ms": list(itl), "deltas": out, "input_tokens": 5, "output_tokens": out,
            "output_tokens_source": "usage"}


class Goodput(unittest.TestCase):
    def test_counts_requests_within_every_slo(self):
        records = [
            record(100, 10, 500),
            record(300, 10, 500),          # TTFT over
            record(100, 30, 500),          # TPOT over
            record(100, None, 150, out=1),  # one token: no TPOT, still good
            record(None, None, 50, ok=False, error="http_503"),
        ]
        result = bench.goodput(records, 2.0, {"ttft_ms": 200, "tpot_ms": 20, "e2e_ms": None})
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["rps"], 1.0)
        self.assertEqual(result["fraction"], 0.4)
        self.assertEqual(result["slo"], {"ttft_ms": 200, "tpot_ms": 20})
        self.assertIsNone(bench.goodput(records, 2.0, {"ttft_ms": None, "tpot_ms": None, "e2e_ms": None}))

    def test_summary_pools_every_gap_and_counts_failures(self):
        records = [
            record(100, 10, 500, itl=[1, 2, 3]),
            record(120, 12, 600, itl=[4, 5]),
            record(None, None, 30, ok=False, error="timeout"),
            record(None, None, 30, ok=False, error="http_429"),
        ]
        summary = bench.summarize(records, 2.0, {"ttft_ms": None, "tpot_ms": None, "e2e_ms": None})
        self.assertEqual(summary["itl_ms"]["count"], 5)
        self.assertEqual(summary["itl_ms"]["max"], 5)
        self.assertEqual((summary["completed"], summary["failed"], summary["aborted"]), (2, 2, 1))
        self.assertEqual(summary["errors"], {"http_429": 1, "timeout": 1})
        self.assertEqual(summary["throughput_rps"], 1.0)
        self.assertEqual(summary["output_throughput_tps"], 10.0)
        self.assertEqual(summary["total_token_throughput_tps"], 15.0)


PROM_BEFORE = """# TYPE izwi_engine_step_host_seconds histogram
izwi_engine_step_host_seconds_bucket{phase="prepare",le="0.0001"} 0
izwi_engine_step_host_seconds_bucket{phase="prepare",le="0.001"} 0
izwi_engine_step_host_seconds_bucket{phase="prepare",le="+Inf"} 0
izwi_engine_step_host_seconds_sum{phase="prepare"} 0
izwi_engine_step_host_seconds_count{phase="prepare"} 0
izwi_engine_step_host_seconds_bucket{phase="commit",le="0.0001"} 1
izwi_engine_step_host_seconds_bucket{phase="commit",le="0.001"} 1
izwi_engine_step_host_seconds_bucket{phase="commit",le="+Inf"} 1
izwi_engine_step_host_seconds_sum{phase="commit"} 0.00005
izwi_engine_step_host_seconds_count{phase="commit"} 1
izwi_engine_batch_rows_bucket{le="1"} 0
izwi_engine_batch_rows_bucket{le="2"} 0
izwi_engine_batch_rows_bucket{le="4"} 0
izwi_engine_batch_rows_bucket{le="+Inf"} 0
izwi_engine_batch_rows_sum 0
izwi_engine_batch_rows_count 0
izwi_kernel_host_read_ops_total 100
izwi_request_generation_tokens_total{model="m",task="chat"} 10
izwi_requests_active 3
"""
PROM_AFTER = """izwi_engine_step_host_seconds_bucket{phase="prepare",le="0.0001"} 4
izwi_engine_step_host_seconds_bucket{phase="prepare",le="0.001"} 10
izwi_engine_step_host_seconds_bucket{phase="prepare",le="+Inf"} 10
izwi_engine_step_host_seconds_sum{phase="prepare"} 0.002
izwi_engine_step_host_seconds_count{phase="prepare"} 10
izwi_engine_step_host_seconds_bucket{phase="commit",le="0.0001"} 11
izwi_engine_step_host_seconds_bucket{phase="commit",le="0.001"} 11
izwi_engine_step_host_seconds_bucket{phase="commit",le="+Inf"} 11
izwi_engine_step_host_seconds_sum{phase="commit"} 0.00105
izwi_engine_step_host_seconds_count{phase="commit"} 11
izwi_engine_batch_rows_bucket{le="1"} 2
izwi_engine_batch_rows_bucket{le="2"} 6
izwi_engine_batch_rows_bucket{le="4"} 10
izwi_engine_batch_rows_bucket{le="+Inf"} 10
izwi_engine_batch_rows_sum 24
izwi_engine_batch_rows_count 10
izwi_kernel_host_read_ops_total 140
izwi_request_generation_tokens_total{model="m",task="chat"} 30
izwi_requests_active 0
"""


class Prometheus(unittest.TestCase):
    def test_parse_labels_escapes_and_skips_comments(self):
        samples = bench.parse_prometheus(
            '# HELP x y\nx_total{a="q\\"uote",b="back\\\\slash"} 3\ny 1.5 1700000000\nz NaN\n')
        self.assertEqual(samples[("x_total", (("a", 'q"uote'), ("b", "back\\slash")))], 3.0)
        self.assertEqual(samples[("y", ())], 1.5)
        self.assertNotIn(("z", ()), samples)

    def test_histogram_quantile_matches_prometheus(self):
        buckets = [(1.0, 2.0), (2.0, 6.0), (4.0, 10.0), (math.inf, 10.0)]
        self.assertAlmostEqual(bench.histogram_quantile(0.5, buckets), 1.75)
        self.assertAlmostEqual(bench.histogram_quantile(0.1, buckets), 0.5)
        self.assertAlmostEqual(bench.histogram_quantile(0.9, buckets), 3.5)
        self.assertEqual(bench.histogram_quantile(0.99, [(1.0, 1.0), (math.inf, 5.0)]), 1.0)
        self.assertIsNone(bench.histogram_quantile(0.5, [(1.0, 0.0), (math.inf, 0.0)]))

    def test_counted_metrics_come_from_deltas(self):
        report = bench.server_metrics_report(PROM_BEFORE, PROM_AFTER, client_output_tokens=99)
        counted = report["counted"]
        self.assertEqual(counted["output_tokens"], 20)
        self.assertEqual(counted["output_tokens_source"], "server")
        self.assertAlmostEqual(counted["device_readbacks_per_token"], 2.0)
        self.assertEqual(counted["engine_steps"], 10)
        # (0.002 prepare + 0.001 commit) seconds over 10 steps.
        self.assertAlmostEqual(counted["host_us_per_step"], 300.0)
        self.assertAlmostEqual(counted["step_host_us_by_phase"]["prepare"]["mean"], 200.0)
        self.assertAlmostEqual(counted["step_host_us_by_phase"]["commit"]["mean"], 100.0)
        self.assertAlmostEqual(counted["engine_steps_per_token"], 0.5)
        self.assertAlmostEqual(counted["batch_rows"]["mean"], 2.4)
        self.assertAlmostEqual(counted["batch_rows"]["p50"], 1.75)
        self.assertEqual(report["counters"]['izwi_kernel_host_read_ops_total'], 40)
        self.assertNotIn("izwi_requests_active", report["counters"])

    def test_client_tokens_are_the_fallback(self):
        counted = bench.server_metrics_report("", "izwi_kernel_host_read_ops_total 8\n", 4)["counted"]
        self.assertEqual((counted["output_tokens_source"], counted["device_readbacks_per_token"]), ("client", 2.0))


class Datasets(unittest.TestCase):
    def test_synthetic_lengths_are_seeded_and_in_range(self):
        first = bench.synthetic_requests(50, 100, 20, 0.5, 0.25, random.Random(1))
        again = bench.synthetic_requests(50, 100, 20, 0.5, 0.25, random.Random(1))
        self.assertEqual(first, again)
        for request in first:
            self.assertTrue(50 <= len(request["messages"][0]["content"].split()) <= 150)
            self.assertTrue(15 <= request["max_tokens"] <= 25)
        self.assertEqual(len({request["messages"][0]["content"] for request in first}), 50)

    def test_jsonl_and_sharegpt_records(self):
        self.assertEqual(bench.dataset_entry({"prompt": "hi", "output_len": 5}, 9),
                         {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
        self.assertEqual(bench.dataset_entry({"messages": [{"role": "user", "content": "x"}]}, 9)["max_tokens"], 9)
        sharegpt = {"conversations": [{"from": "system", "value": "s"}, {"from": "human", "value": "q"},
                                      {"from": "gpt", "value": "x" * 40}]}
        self.assertEqual(bench.dataset_entry(sharegpt, 9),
                         {"messages": [{"role": "user", "content": "q"}], "max_tokens": 10})
        self.assertIsNone(bench.dataset_entry({"conversations": [{"from": "gpt", "value": "a"}]}, 9))

    def test_load_dataset_cycles_a_small_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "prompts.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"prompt": "a"}\n\n{"prompt": "b"}\n{"bogus": 1}\n')
            requests = bench.load_dataset(path, 5, 3, random.Random(0))
        self.assertEqual(len(requests), 5)
        self.assertEqual({request["messages"][0]["content"] for request in requests}, {"a", "b"})


def summary_with(throughput, ttft_p50, tpot_p50):
    return {"throughput_rps": throughput, "output_throughput_tps": throughput * 10,
            "ttft_ms": {"p50": ttft_p50, "p95": ttft_p50 * 2}, "tpot_ms": {"p50": tpot_p50},
            "itl_ms": None, "end_to_end_ms": {"p95": 100.0}}


def result_with(summaries, backend="cpu"):
    return bench.build_result({"model": "m", "backend": backend, "host": {"cpu_model": "c"}},
                              {"profile": "regression", "dataset": "synthetic", "seed": 0},
                              [(name, {"command": "serving", "summary": summary})
                               for name, summary in summaries.items()],
                              "2026-10-15T00:00:00+00:00", "2026-10-15T00:01:00+00:00")


class Comparison(unittest.TestCase):
    def test_regressions_respect_direction_and_tolerance(self):
        baseline = result_with({"rate=inf,concurrency=1": summary_with(10.0, 100.0, 20.0)})
        current = result_with({"rate=inf,concurrency=1": summary_with(8.5, 105.0, 19.0)}, backend="metal")
        comparison = bench.compare_results(current, baseline, 10.0)
        status = {check["metric"]: check["status"] for check in comparison["checks"]}
        self.assertEqual(status["throughput_rps"], "regression")      # -15%
        self.assertEqual(status["output_throughput_tps"], "regression")
        self.assertEqual(status["ttft_ms.p50"], "ok")                 # +5%
        self.assertEqual(status["tpot_ms.p50"], "ok")                 # faster
        self.assertNotIn("itl_ms.p95", status)                        # missing on both
        self.assertEqual(comparison["regressions"], 2)
        self.assertTrue(any("metadata.backend differs" in note for note in comparison["notes"]))

    def test_main_compares_existing_files_and_only_fails_when_asked(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, throughput in (("base", 10.0), ("now", 5.0)):
                paths[name] = os.path.join(directory, f"{name}.json")
                with open(paths[name], "w", encoding="utf-8") as handle:
                    json.dump(result_with({"p": summary_with(throughput, 100.0, 20.0)}), handle)
            argv = ["--current", paths["now"], "--baseline", paths["base"]]
            with open(os.devnull, "w") as sink, unittest.mock.patch("sys.stdout", sink):
                self.assertEqual(bench.main(argv), 0)
                self.assertEqual(bench.main(argv + ["--fail-on-regression"]), 3)


class FakeChatHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    tokens = 4
    gap_secs = 0.02

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][0]["content"]
        if content == "reject":
            payload = b'{"error":{"type":"overloaded"}}'
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.chunk(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        for index in range(self.tokens):
            time.sleep(self.gap_secs)
            event = b'data: {"choices":[{"delta":{"content":"t%d"}}]}\n\n' % index
            # Split one event over two chunks to exercise reassembly.
            self.chunk(event[:10])
            self.chunk(event[10:])
        if content == "truncate":
            self.wfile.write(b"0\r\n\r\n")
            return
        usage = {"prompt_tokens": len(content.split()), "completion_tokens": self.tokens}
        self.chunk(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\ndata: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")

    def chunk(self, data):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeChatHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.config = {
            "url": f"http://127.0.0.1:{cls.server.server_address[1]}/v1/chat/completions",
            "headers": {"Authorization": "Bearer test"},
            "model": "fake",
            "temperature": 0.0,
            "seed": 0,
            "timeout": 30.0,
        }

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def run_point(self, requests, rate=math.inf, max_concurrency=None):
        return asyncio.run(bench.run_point(requests, rate, 1.0, max_concurrency, self.config, 0))

    def test_streams_are_timed_per_delta(self):
        requests = bench.synthetic_requests(6, 8, 4, 0.0, 0.0, random.Random(0))
        records, duration = self.run_point(requests, max_concurrency=2)
        summary = bench.summarize(records, duration, {"ttft_ms": 10000.0, "tpot_ms": None, "e2e_ms": None})
        self.assertEqual((summary["completed"], summary["failed"]), (6, 0))
        self.assertEqual(summary["itl_ms"]["count"], 6 * 3)
        self.assertGreater(summary["itl_ms"]["p50"], 10.0)
        self.assertGreater(summary["ttft_ms"]["min"], 10.0)
        self.assertEqual(summary["total_output_tokens"], 24)
        self.assertEqual(summary["total_input_tokens"], 48)
        self.assertEqual(summary["tokens_per_delta"], 1.0)
        self.assertEqual(summary["goodput"]["requests"], 6)
        # Two at a time, each ~80 ms: at least three waves.
        self.assertGreater(duration, 0.2)

    def test_rejections_and_truncated_streams_are_failures(self):
        requests = [{"messages": [{"role": "user", "content": text}], "max_tokens": 4}
                    for text in ("reject", "truncate", "fine")]
        records, duration = self.run_point(requests)
        summary = bench.summarize(records, duration, {})
        self.assertEqual(summary["errors"], {"http_429": 1, "truncated": 1})
        self.assertEqual((summary["completed"], summary["aborted"]), (1, 1))

    def test_open_loop_sends_on_schedule(self):
        # Arrivals ~20 ms apart while each stream takes ~80 ms: a closed loop
        # would send the last request hundreds of milliseconds late.
        requests = bench.synthetic_requests(5, 4, 4, 0.0, 0.0, random.Random(0))
        records, _ = self.run_point(requests, rate=50.0)
        expected = bench.arrival_offsets(5, 50.0, 1.0, random.Random(0))
        for record_, offset in zip(sorted(records, key=lambda r: r["index"]), expected):
            self.assertGreaterEqual(record_["start_offset_s"], offset - 0.001)
            self.assertLess(record_["start_offset_s"], offset + 0.1)

    def test_result_is_strict_json_readable_by_bench_compare(self):
        requests = bench.synthetic_requests(2, 4, 4, 0.0, 0.0, random.Random(0))
        records, duration = self.run_point(requests)
        report = bench.point_report(math.inf, None, 1.0, records, duration, {}, {}, True)
        result = bench.build_result({"model": "fake", "backend": "cpu", "git": bench.git_info(),
                                     "host": bench.host_info()},
                                    {"request_rates": ["inf"]}, [(bench.point_name(math.inf, None), report)],
                                    "a", "b")
        decoded = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual((decoded["schema"], decoded["schema_version"]), ("izwi.serving_benchmark", 1))
        for key in ("model", "backend", "git", "host"):
            self.assertIn(key, decoded["metadata"])
        entry = decoded["reports"][0]
        self.assertEqual(entry["name"], "rate=inf")
        self.assertEqual(entry["report"]["config"]["request_rate"], "inf")
        # The fields `izwi bench compare` reads.
        summary = entry["report"]["summary"]
        for path in (("ttft_ms", "p95"), ("end_to_end_ms", "p95"), ("throughput_rps",)):
            self.assertIsInstance(bench._metric(summary, path), float)
        self.assertEqual(len(entry["report"]["samples"]), 2)


if __name__ == "__main__":
    unittest.main()
