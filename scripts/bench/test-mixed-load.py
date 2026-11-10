#!/usr/bin/env python3
"""Unit tests for run-mixed-load.py: percentile and chunk-gap math, the ramp
plan, refusal counting, turn latencies, Prometheus deltas, SLO verdicts, the
result schema, and end-to-end ramps against a local fake server that speaks
the speech, transcription, metrics and `/v1/realtime` routes."""
import argparse
import base64
import contextlib
import http.server
import importlib.util
import io
import json
import pathlib
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import wave

spec = importlib.util.spec_from_file_location(
    "mixed_load", pathlib.Path(__file__).with_name("run-mixed-load.py"))
load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load)

LIMITS = {"asr_first_partial_s": 0.5, "asr_final_s": 0.8, "tts_first_audio_s": 0.5, "degradation": 0.25,
          "partial_field": "first_partial_s", "final_field": "final_after_vad_s"}


def p95(value):
    return {"count": 4, "mean": value, "p50": value, "p95": value, "max": value}


def asr_summary(partial, final):
    return {"first_partial_s": p95(partial), "first_partial_after_vad_s": p95(partial),
            "final_after_vad_s": p95(final), "final_after_audio_s": p95(final)}


class Statistics(unittest.TestCase):
    def test_stats_interpolate_percentiles_and_skip_missing_values(self):
        summary = load.stats([None, 4.0, 1.0, 3.0, 2.0, None])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 2.5)
        self.assertAlmostEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertEqual(summary["max"], 4.0)

    def test_stats_of_nothing_keep_the_schema(self):
        self.assertEqual(load.stats([None]), {"count": 0, "mean": None, "p50": None, "p95": None, "max": None})

    def test_first_audio_is_when_the_header_and_100ms_have_arrived(self):
        arrivals = [(0.10, 44), (0.25, 2400), (0.30, 2400), (0.70, 4800)]
        summary = load.arrival_summary(arrivals)
        self.assertEqual(summary["first_byte_s"], 0.10)
        self.assertEqual(summary["first_audio_s"], 0.30)
        self.assertEqual(summary["chunks"], 4)
        self.assertEqual(summary["bytes"], 44 + 9600)
        for gap, expected in zip(summary["gaps_s"], [0.15, 0.05, 0.40]):
            self.assertAlmostEqual(gap, expected)

    def test_a_stream_too_short_for_first_audio_reports_none(self):
        summary = load.arrival_summary([(0.2, 100)])
        self.assertIsNone(summary["first_audio_s"])
        self.assertEqual(summary["gaps_s"], [])
        self.assertIsNone(load.arrival_summary([])["first_byte_s"])


class Ramp(unittest.TestCase):
    def test_levels_parse_with_the_baseline_first(self):
        self.assertEqual(load.parse_levels("0,4,16,64"), [0, 4, 16, 64])
        self.assertEqual(load.parse_levels("8, 2"), [8, 2])
        for bad in ("", "4,0", "1,1", "-1", "a"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                load.parse_levels(bad)

    def test_each_level_is_a_named_step_and_level_zero_is_the_baseline(self):
        steps = load.plan_steps([0, 4, 16])
        self.assertEqual([step["name"] for step in steps],
                         ["step-00-batch-c0", "step-01-batch-c4", "step-02-batch-c16"])
        self.assertEqual([step["baseline"] for step in steps], [True, False, False])
        self.assertEqual([step["batch_concurrency"] for step in steps], [0, 4, 16])

    def test_batch_defaults_follow_the_real_time_side(self):
        args = load.parse_args(["--out", "x", "--realtime-kind", "tts", "--realtime-model", "Kokoro-82M"])
        config = load.build_config(args)
        self.assertEqual((config["batch_kind"], config["batch_model"]), ("tts", "Kokoro-82M"))
        self.assertEqual(config["batch_levels"], [0, 4, 16, 64])
        self.assertEqual(config["metrics_url"], "http://127.0.0.1:8080/v1/metrics/prometheus")
        self.assertEqual(config["limits"], LIMITS)
        self.assertNotIn("headers", load.public_config(config))


class Outcomes(unittest.TestCase):
    def test_only_429_503_and_close_1013_are_refusals(self):
        self.assertEqual(load.refusal_kind(http_status=429), "http_429")
        self.assertEqual(load.refusal_kind(http_status=503), "http_503")
        self.assertEqual(load.refusal_kind(close_code=1013), "ws_1013")
        self.assertIsNone(load.refusal_kind(http_status=500, close_code=1011))
        self.assertIsNone(load.refusal_kind())

    def test_failures_are_classified_from_http_upgrade_and_close_statuses(self):
        http_error = urllib.error.HTTPError("u", 429, "Too Many", {}, io.BytesIO(b'{"error": "busy"}'))
        record = load.failure_record(http_error)
        self.assertEqual((record["status"], record["refusal"]), ("refused", "http_429"))
        self.assertIn("busy", record["error"])
        upgrade = ConnectionError("websocket upgrade refused: HTTP/1.1 503 Service Unavailable")
        self.assertEqual(load.failure_record(upgrade)["refusal"], "http_503")
        closed = ConnectionError("socket closed (code 1013) while waiting for session.created")
        self.assertEqual(load.failure_record(closed, 1013)["refusal"], "ws_1013")
        plain = load.failure_record(ValueError("HTTP/1.1 503 in a body is not an upgrade status"))
        self.assertEqual(plain["status"], "failed")
        self.assertNotIn("refusal", plain)

    def test_outcomes_count_each_refusal_kind_apart_from_errors(self):
        records = [{"status": "ok"}, {"status": "ok"},
                   {"status": "failed", "error": "boom"},
                   {"status": "refused", "refusal": "http_429", "error": "HTTP 429: busy"},
                   {"status": "refused", "refusal": "ws_1013", "error": "closed"},
                   {"status": "refused", "refusal": "ws_1013", "error": "closed"}]
        counts = load.outcomes(records)
        self.assertEqual((counts["requests"], counts["completed"], counts["errors"]), (6, 2, 1))
        self.assertEqual(counts["refused"], {"http_429": 1, "http_503": 0, "ws_1013": 2, "total": 3})
        self.assertEqual(counts["error_samples"], ["HTTP 429: busy", "boom", "closed"])


class RealTime(unittest.TestCase):
    def test_turn_latencies_use_the_first_event_of_each_kind(self):
        events = [(10.9, {"type": "input_audio_buffer.speech_started"}),
                  (12.6, {"type": load.SPEECH_STOPPED}),
                  (12.7, {"type": load.TRANSCRIPT_DELTA, "delta": "the"}),
                  (12.8, {"type": load.TRANSCRIPT_DELTA, "delta": " fox"}),
                  (13.0, {"type": load.TRANSCRIPT_COMPLETED, "transcript": "the fox"})]
        record = load.asr_turn_record(events, started=10.0, audio_end=12.0)
        self.assertEqual(record["status"], "ok")
        self.assertAlmostEqual(record["first_partial_s"], 2.7)
        self.assertAlmostEqual(record["first_partial_after_vad_s"], 0.1)
        self.assertAlmostEqual(record["final_after_vad_s"], 0.4)
        self.assertAlmostEqual(record["final_after_audio_s"], 1.0)
        self.assertAlmostEqual(record["audio_s"], 2.0)
        self.assertEqual(record["transcript"], "the fox")

    def test_a_partial_during_speech_is_negative_against_the_vad(self):
        events = [(10.3, {"type": load.TRANSCRIPT_DELTA}), (12.6, {"type": load.SPEECH_STOPPED}),
                  (12.9, {"type": load.TRANSCRIPT_COMPLETED, "transcript": "x"})]
        record = load.asr_turn_record(events, 10.0, 12.0)
        self.assertAlmostEqual(record["first_partial_s"], 0.3)
        self.assertAlmostEqual(record["first_partial_after_vad_s"], -2.3)

    def test_a_failed_transcription_is_a_failed_turn_with_its_message(self):
        events = [(1.0, {"type": load.SPEECH_STOPPED}),
                  (1.2, {"type": load.TRANSCRIPT_FAILED, "error": {"message": "ASR failed: no model"}})]
        record = load.asr_turn_record(events, 0.0, 0.5)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "ASR failed: no model")
        self.assertIsNone(record["first_partial_s"])
        self.assertIsNone(record["final_after_vad_s"])

    def test_transcription_sessions_name_the_model_format_and_vad(self):
        update = load.transcription_session_update(
            {"realtime_model": "Parakeet-TDT-0.6B-v3", "language": "en", "silence_ms": 300})
        session = update["session"]
        self.assertEqual((update["type"], session["type"]), ("session.update", "transcription"))
        audio = session["audio"]["input"]
        self.assertEqual(audio["format"], {"type": "audio/pcm", "rate": 24_000})
        self.assertEqual(audio["transcription"], {"model": "Parakeet-TDT-0.6B-v3", "language": "en"})
        self.assertEqual(audio["turn_detection"], {"type": "server_vad", "silence_duration_ms": 300})

    def test_sse_audio_decodes_deltas_and_raises_on_a_failed_stream(self):
        delta = json.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(b"abcd").decode()})
        done = json.dumps({"type": "speech.audio.done", "usage": {}})
        self.assertEqual(load.sse_audio([delta, done, "[DONE]"]), [b"abcd"])
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            load.sse_audio([json.dumps({"type": "error", "error": {"message": "out of memory"}})])

    def test_speech_bodies_reuse_the_concurrency_drivers_fields(self):
        body = load.speech_body("Kokoro-82M", 5, "audio", None)
        self.assertEqual((body["voice"], body["seed"], body["stream_format"]), ("af_heart", 6, "audio"))
        self.assertEqual(body["input"], load.tts.POOL[5])
        self.assertNotIn("service_tier", body)
        unknown = load.speech_body("Some-New-TTS", 2, None, None, {"voice": "x"}, "batch")
        self.assertEqual(unknown, {"model": "Some-New-TTS", "input": load.tts.POOL[2], "response_format": "wav",
                                   "seed": 3, "voice": "x", "service_tier": "batch"})
        clone = load.speech_body("FishAudio-S2-Pro", 0, None, "REF")
        self.assertEqual(clone["reference_audio"], "REF")

    def test_the_service_tier_is_sent_only_when_set(self):
        self.assertEqual(load.service_tier_headers(None), {})
        self.assertEqual(load.service_tier_headers("batch"), {"X-Izwi-Service-Tier": "batch"})
        base = {"batch_model": "m", "language": None, "batch_service_tier": None}
        self.assertEqual(load.transcription_fields(base), {"model": "m", "response_format": "json"})
        self.assertEqual(load.transcription_fields(dict(base, batch_service_tier="flex"))["service_tier"], "flex")


class Summaries(unittest.TestCase):
    def test_batch_throughput_is_completed_audio_over_the_wall_time(self):
        records = [{"status": "ok", "audio_s": 10.0, "latency_s": 1.0},
                   {"status": "ok", "audio_s": 20.0, "latency_s": 3.0},
                   {"status": "refused", "refusal": "http_503", "error": "HTTP 503: full", "latency_s": 0.1}]
        summary = load.batch_summary(records, 5.0)
        self.assertEqual(summary["audio_seconds"], 30.0)
        self.assertEqual(summary["audio_seconds_per_second"], 6.0)
        self.assertEqual(summary["latency_s"]["p50"], 2.0)
        self.assertAlmostEqual(summary["latency_s"]["p95"], 2.9)
        self.assertEqual(summary["refused"]["http_503"], 1)

    def test_a_level_zero_batch_reports_nothing_rather_than_dividing_by_zero(self):
        summary = load.batch_summary([], 0.0)
        self.assertEqual(summary["requests"], 0)
        self.assertIsNone(summary["audio_seconds_per_second"])
        self.assertEqual(summary["latency_s"]["count"], 0)

    def test_tts_gaps_pool_every_stream_and_asr_reports_each_reference(self):
        records = [{"status": "ok", "first_byte_s": 0.1, "first_audio_s": 0.2, "gaps_s": [0.1, 0.3], "audio_s": 4.0},
                   {"status": "ok", "first_byte_s": 0.3, "first_audio_s": 0.4, "gaps_s": [0.2], "audio_s": 6.0},
                   {"status": "failed", "error": "x"}]
        summary = load.realtime_summary("tts", records, 5.0)
        self.assertEqual(summary["inter_chunk_gap_s"]["count"], 3)
        self.assertAlmostEqual(summary["inter_chunk_gap_s"]["p50"], 0.2)
        self.assertEqual(summary["inter_chunk_gap_s"]["max"], 0.3)
        self.assertAlmostEqual(summary["first_audio_s"]["p50"], 0.3)
        self.assertEqual(summary["audio_seconds_per_second"], 2.0)
        self.assertEqual((summary["completed"], summary["errors"]), (2, 1))
        turns = [{"status": "ok", "first_partial_s": 2.5, "first_partial_after_vad_s": 0.1,
                  "final_after_vad_s": 0.3, "final_after_audio_s": 0.9, "transcript": ""}]
        asr = load.realtime_summary("asr", turns, 5.0)
        self.assertEqual(asr["first_partial_s"]["p95"], 2.5)
        self.assertEqual(asr["final_after_audio_s"]["p95"], 0.9)
        self.assertEqual(asr["empty_transcripts"], 1)


def histogram(name, labels, buckets, total):
    """Exposition lines of one histogram series; `buckets` is {le: cumulative}."""
    prefix = labels + "," if labels else ""
    lines = [f'{name}_bucket{{{prefix}le="{le}"}} {count}' for le, count in buckets.items()]
    lines.append(f'{name}_sum{{{labels}}} {total}')
    lines.append(f'{name}_count{{{labels}}} {buckets["+Inf"]}')
    return lines


class Metrics(unittest.TestCase):
    BEFORE = "\n".join([
        "# TYPE izwi_request_ttft_seconds histogram",
        *histogram("izwi_request_ttft_seconds", 'workload_class="realtime"',
                   {"0.1": 2, "0.5": 4, "+Inf": 4}, 0.8),
        *histogram("izwi_request_ttft_seconds", 'workload_class="batch"',
                   {"0.1": 0, "0.5": 1, "+Inf": 1}, 0.3),
        *histogram("izwi_request_duration_seconds", "", {"1": 5, "+Inf": 5}, 2.0),
        *histogram("izwi_engine_batch_rows", 'model="Kokoro-82M"', {"1": 3, "8": 3, "+Inf": 3}, 3),
        'izwi_tts_output_audio_seconds_total{model="Kokoro-82M",workload_class="batch"} 10',
        'izwi_engine_executor_physical_defers_total{reason="workspace_capacity"} 1',
        'izwi_engine_executor_model_tensor_batch_width_calls_total{width="1"} 7',
    ])
    AFTER = "\n".join([
        *histogram("izwi_request_ttft_seconds", 'workload_class="realtime"',
                   {"0.1": 2, "0.5": 12, "+Inf": 12}, 3.2),
        *histogram("izwi_request_ttft_seconds", 'workload_class="batch"',
                   {"0.1": 0, "0.5": 1, "+Inf": 1}, 0.3),
        *histogram("izwi_request_duration_seconds", "", {"1": 7, "+Inf": 9}, 12.0),
        *histogram("izwi_engine_batch_rows", 'model="Kokoro-82M"', {"1": 3, "8": 13, "+Inf": 13}, 63),
        'izwi_tts_output_audio_seconds_total{model="Kokoro-82M",workload_class="batch"} 40',
        'izwi_tts_output_audio_seconds_total{model="Kokoro-82M",workload_class="realtime"} 12.5',
        'izwi_engine_executor_physical_defers_total{reason="workspace_capacity"} 4',
        'izwi_engine_executor_model_tensor_batch_width_calls_total{width="1"} 7',
        'izwi_engine_executor_model_tensor_batch_width_calls_total{width="8"} 5',
        "junk line without a value",
    ])

    def test_request_histograms_are_deltas_by_workload_class(self):
        report = load.metrics_delta(self.BEFORE, self.AFTER)
        ttft = report["histograms"]["izwi_request_ttft_seconds"]
        self.assertEqual(list(ttft), ["realtime"], "a class without new requests is left out")
        self.assertEqual(ttft["realtime"]["count"], 8)
        self.assertAlmostEqual(ttft["realtime"]["mean"], 0.3)
        self.assertAlmostEqual(ttft["realtime"]["p50"], 0.3)
        self.assertAlmostEqual(ttft["realtime"]["p95"], 0.48)

    def test_a_histogram_without_the_class_label_is_reported_unlabelled(self):
        duration = load.metrics_delta(self.BEFORE, self.AFTER)["histograms"]["izwi_request_duration_seconds"]
        self.assertEqual(list(duration), ["unlabelled"])
        self.assertEqual(duration["unlabelled"]["count"], 4)
        self.assertAlmostEqual(duration["unlabelled"]["mean"], 2.5)

    def test_widths_and_defers_are_deltas_per_label_set(self):
        report = load.metrics_delta(self.BEFORE, self.AFTER)
        rows = report["histograms"]["izwi_engine_batch_rows"]["Kokoro-82M"]
        self.assertEqual(rows["count"], 10)
        self.assertAlmostEqual(rows["mean"], 6.0)
        counters = report["counters"]
        self.assertEqual(counters["izwi_engine_executor_physical_defers_total"],
                         {"total": 3.0, "series": {"reason=workspace_capacity": 3.0}})
        self.assertEqual(counters["izwi_engine_executor_model_tensor_batch_width_calls_total"]["series"],
                         {"width=8": 5.0})
        audio = counters["izwi_tts_output_audio_seconds_total"]
        self.assertEqual(audio["total"], 42.5)
        self.assertEqual(audio["series"]["model=Kokoro-82M,workload_class=realtime"], 12.5)

    def test_series_the_server_lacks_are_listed_not_fatal(self):
        report = load.metrics_delta(self.BEFORE, self.AFTER)
        self.assertEqual(report["missing"], [
            "izwi_request_queue_wait_seconds", "izwi_asr_preparation_batch_rows",
            "izwi_engine_executor_physical_dispatch_width_dispatches_total"])
        empty = load.metrics_delta("", "")
        self.assertEqual((empty["histograms"], empty["counters"]), ({}, {}))
        self.assertEqual(len(empty["missing"]),
                         len(load.REQUEST_HISTOGRAMS + load.WIDTH_HISTOGRAMS + load.COUNTERS))

    def test_an_unreachable_endpoint_is_recorded_as_unavailable(self):
        config = {"metrics_url": "http://127.0.0.1:1/v1/metrics/prometheus", "headers": {}}
        text, error = load.scrape(config)
        self.assertIsNone(text)
        report = load.server_metrics(config, (text, error), (text, error))
        self.assertFalse(report["available"])
        self.assertTrue(report["error"])
        self.assertNotIn("histograms", report)
        self.assertEqual(load.scrape({"metrics_url": None, "headers": {}}), (None, "disabled"))


class Slo(unittest.TestCase):
    def test_the_baseline_step_checks_latency_and_skips_degradation(self):
        verdict = load.evaluate_slos("asr", asr_summary(0.4, 0.7), None, LIMITS)
        checks = verdict["checks"]
        self.assertEqual(set(checks), {"asr_first_partial_p95", "asr_final_p95",
                                       "degradation_asr_first_partial_p95", "degradation_asr_final_p95"})
        self.assertEqual(checks["asr_first_partial_p95"], {"value": 0.4, "limit": 0.5, "pass": True})
        self.assertTrue(checks["asr_final_p95"]["pass"])
        self.assertIsNone(checks["degradation_asr_final_p95"]["pass"])
        self.assertTrue(verdict["pass"])

    def test_a_latency_over_its_limit_fails_the_step(self):
        verdict = load.evaluate_slos("asr", asr_summary(0.4, 0.81), None, LIMITS)
        self.assertFalse(verdict["checks"]["asr_final_p95"]["pass"])
        self.assertFalse(verdict["pass"])
        self.assertTrue(load.evaluate_slos("asr", asr_summary(0.5, 0.8), None, LIMITS)["pass"], "limits are inclusive")

    def test_degradation_is_the_relative_rise_over_the_baseline(self):
        baseline = {"first_audio_s": p95(0.20)}
        within = load.evaluate_slos("tts", {"first_audio_s": p95(0.25)}, baseline, LIMITS)
        check = within["checks"]["degradation_tts_first_audio_p95"]
        self.assertAlmostEqual(check["value"], 0.25)
        self.assertEqual((check["baseline"], check["limit"], check["pass"]), (0.20, 0.25, True))
        beyond = load.evaluate_slos("tts", {"first_audio_s": p95(0.30)}, baseline, LIMITS)
        self.assertFalse(beyond["checks"]["degradation_tts_first_audio_p95"]["pass"])
        self.assertTrue(beyond["checks"]["tts_first_audio_p95"]["pass"])
        self.assertFalse(beyond["pass"])
        faster = load.evaluate_slos("tts", {"first_audio_s": p95(0.10)}, baseline, LIMITS)
        self.assertTrue(faster["checks"]["degradation_tts_first_audio_p95"]["pass"])

    def test_a_step_without_samples_fails_and_a_baseline_without_them_is_not_compared(self):
        empty = {"first_audio_s": load.stats([])}
        verdict = load.evaluate_slos("tts", empty, {"first_audio_s": p95(0.2)}, LIMITS)
        self.assertFalse(verdict["checks"]["tts_first_audio_p95"]["pass"])
        self.assertFalse(verdict["checks"]["degradation_tts_first_audio_p95"]["pass"])
        unknown = load.evaluate_slos("tts", {"first_audio_s": p95(0.2)}, empty, LIMITS)
        self.assertIsNone(unknown["checks"]["degradation_tts_first_audio_p95"]["pass"])
        self.assertTrue(unknown["pass"])

    def test_the_references_pick_which_distribution_an_slo_reads(self):
        realtime = dict(asr_summary(0.1, 0.1), first_partial_s=p95(3.0), final_after_audio_s=p95(1.2))
        self.assertFalse(load.evaluate_slos("asr", realtime, None, LIMITS)["pass"])
        by_vad = dict(LIMITS, partial_field="first_partial_after_vad_s")
        self.assertTrue(load.evaluate_slos("asr", realtime, None, by_vad)["pass"])
        by_audio = dict(by_vad, final_field="final_after_audio_s")
        self.assertFalse(load.evaluate_slos("asr", realtime, None, by_audio)["checks"]["asr_final_p95"]["pass"])

    def test_the_summary_fails_an_slo_any_step_failed(self):
        baseline = {"first_audio_s": p95(0.2)}
        steps = [{"step": {"name": name}, "slo": load.evaluate_slos("tts", {"first_audio_s": p95(value)}, base, LIMITS)}
                 for name, value, base in (("s0", 0.2, None), ("s1", 0.22, baseline), ("s2", 0.4, baseline))]
        summary = load.summarize_slos(steps)
        self.assertEqual(summary["checks"], {"tts_first_audio_p95": True, "degradation_tts_first_audio_p95": False})
        self.assertFalse(summary["pass"])
        self.assertEqual(summary["failed_steps"], ["s2"])
        alone = load.summarize_slos(steps[:1])
        self.assertEqual(alone["checks"]["degradation_tts_first_audio_p95"], None)
        self.assertTrue(alone["pass"])


# ------------------------------------------------------------- fake server

def wav_header(rate=24_000):
    """The header of a streamed mono s16 WAV (unknown sizes)."""
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", 0xFFFFFFFF))


def server_frame(opcode, payload):
    """An unmasked server-to-client WebSocket frame."""
    header = bytearray([0x80 | opcode])
    if len(payload) < 126:
        header.append(len(payload))
    else:
        header.append(126)
        header += struct.pack("!H", len(payload))
    return bytes(header) + payload


def read_client_frame(stream):
    """(opcode, payload) of one masked client frame, or None at the end."""
    head = stream.read(2)
    if len(head) < 2:
        return None
    length = head[1] & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", stream.read(2))
    elif length == 127:
        (length,) = struct.unpack("!Q", stream.read(8))
    mask = stream.read(4)
    payload = stream.read(length)
    return head[0] & 0x0F, bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))


class FakeIzwi(http.server.BaseHTTPRequestHandler):
    """Speech (buffered, chunked and SSE), transcriptions, Prometheus metrics
    and `/v1/realtime` transcription sessions. Every speech response is 0.5 s
    of audio in five chunks."""

    frames_per_turn = 7
    refuse_batch_every = 0   # every Nth buffered request answers 429
    refuse_sessions = 0      # this many sockets are closed with 1013 first
    upgrade_status = None    # answer the upgrade with this status instead
    metrics = True

    def log_message(self, *args):
        pass

    def count(self, key, amount=1):
        with self.server.lock:
            self.server.counts[key] = self.server.counts.get(key, 0) + amount
            return self.server.counts[key]

    def reply(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/v1/realtime"):
            return self.realtime()
        if self.path == "/v1/metrics/prometheus" and self.metrics:
            with self.server.lock:
                counts = dict(self.server.counts)
            done = counts.get("speech", 0) + counts.get("transcriptions", 0)
            lines = histogram("izwi_request_duration_seconds", 'workload_class="batch"',
                              {"1": done, "+Inf": done}, done * 0.05)
            lines.append(f'izwi_tts_output_audio_seconds_total{{workload_class="batch"}} '
                         f'{counts.get("speech", 0) * 0.5}')
            return self.reply(200, "text/plain", "\n".join(lines).encode())
        self.reply(404, "application/json", b"{}")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with self.server.lock:
            self.server.requests.append((self.path, dict(self.headers), body))
        if self.path == "/v1/audio/transcriptions":
            time.sleep(0.02)
            self.count("transcriptions")
            return self.reply(200, "application/json", json.dumps({"text": "the quick brown fox"}).encode())
        request = json.loads(body)
        chunks = [bytes(4800)] * 5
        stream_format = request.get("stream_format")
        if not stream_format:
            if self.refuse_batch_every and self.count("buffered") % self.refuse_batch_every == 0:
                return self.reply(429, "application/json", b'{"error": {"message": "queue full"}}')
            time.sleep(0.02)
            self.count("speech")
            return self.reply(200, "audio/wav", wav_header() + b"".join(chunks))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream_format == "sse" else "audio/wav")
        self.end_headers()
        for index, chunk in enumerate([wav_header() + chunks[0]] + chunks[1:]):
            if index:
                time.sleep(0.03)
            if stream_format == "sse":
                event = {"type": "speech.audio.delta", "audio": base64.b64encode(chunk).decode()}
                chunk = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(chunk)
            self.wfile.flush()
        if stream_format == "sse":
            self.wfile.write(b'data: {"type": "speech.audio.done"}\n\ndata: [DONE]\n\n')

    def realtime(self):
        self.server.lock.acquire()
        self.server.sockets.append(dict(self.headers, path=self.path))
        self.server.lock.release()
        if self.upgrade_status:
            return self.reply(self.upgrade_status, "application/json", b"{}")
        self.wfile.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                         b"Connection: Upgrade\r\n\r\n")

        def send(event):
            self.wfile.write(server_frame(1, json.dumps(event).encode()))
            self.wfile.flush()

        if self.count("sockets") <= self.refuse_sessions:
            send({"type": "error", "error": {"code": "session_capacity", "message": "capacity is exhausted"}})
            self.wfile.write(server_frame(8, struct.pack("!H", 1013) + b"realtime_session_capacity"))
            self.wfile.flush()
            return
        send({"type": "session.created", "session": {}})
        frames = 0
        while True:
            frame = read_client_frame(self.rfile)
            if frame is None or frame[0] == 8:
                return
            event = json.loads(frame[1])
            if event["type"] == "session.update":
                with self.server.lock:
                    self.server.updates.append(event["session"])
                send({"type": "session.updated", "session": event["session"]})
            elif event["type"] == "input_audio_buffer.append":
                frames += 1
                if frames % self.frames_per_turn == 0:
                    send({"type": load.SPEECH_STOPPED, "audio_end_ms": 0, "item_id": "item_1"})
                    time.sleep(0.01)
                    send({"type": load.TRANSCRIPT_DELTA, "item_id": "item_1", "delta": "the quick"})
                    time.sleep(0.02)
                    send({"type": load.TRANSCRIPT_COMPLETED, "item_id": "item_1",
                          "transcript": "the quick brown fox"})


@contextlib.contextmanager
def fake_izwi(**settings):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), type("Handler", (FakeIzwi,), settings))
    server.lock, server.counts = threading.Lock(), {}
    server.requests, server.sockets, server.updates = [], [], []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        server.shutdown()
        server.server_close()


def write_speech(directory, rate=24_000, secs=0.1):
    path = pathlib.Path(directory) / "speech.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(int(rate * secs) * 2))
    return path


def run(arguments):
    """(exit code, printed output) of one driver run."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = load.main(arguments)
    return code, out.getvalue()


STEP_KEYS = {"schema", "schema_version", "started_at", "ended_at", "step", "config", "realtime", "batch",
             "server_metrics", "slo"}
SIDE_KEYS = {"kind", "model", "requests", "completed", "errors", "refused", "error_samples", "duration_s"}


class EndToEnd(unittest.TestCase):
    def test_asr_sessions_hold_while_batch_transcriptions_ramp(self):
        with tempfile.TemporaryDirectory() as directory, fake_izwi() as (base, server):
            out = pathlib.Path(directory) / "run"
            # 5 speech frames + (20 + 20) ms of silence = 7 frames a turn.
            code, printed = run([
                "--base-url", base, "--api-key", "secret-key-for-tests", "--out", str(out),
                "--realtime-kind", "asr", "--realtime-model", "Parakeet-TDT-0.6B-v3",
                "--realtime-streams", "2", "--batch-model", "Whisper-Large-v3-Turbo",
                "--batch-levels", "0,3", "--batch-service-tier", "batch",
                "--step-secs", "0.6", "--batch-lead-s", "0.05", "--batch-clips", "4",
                "--batch-min-secs", "0.05", "--batch-max-secs", "0.1",
                "--audio", str(write_speech(directory)), "--silence-ms", "20", "--trailing-silence-ms", "20",
                "--partial-reference", "vad", "--slo-degradation", "50", "--save-records", "--enforce-slo"])
            self.assertEqual(code, 0, printed)
            self.assertEqual(sorted(path.name for path in out.iterdir()),
                             ["step-00-batch-c0.json", "step-01-batch-c3.json", "summary.json"])
            baseline = json.loads((out / "step-00-batch-c0.json").read_text())
            loaded = json.loads((out / "step-01-batch-c3.json").read_text())
            summary = json.loads((out / "summary.json").read_text())

        for step in (baseline, loaded):
            self.assertEqual(set(step), STEP_KEYS | {"records"})
            self.assertEqual((step["schema"], step["schema_version"]), ("izwi.mixed_load.step", 1))
            self.assertLessEqual(SIDE_KEYS, set(step["realtime"]))
            self.assertLessEqual(SIDE_KEYS, set(step["batch"]))
            self.assertNotIn("headers", step["config"])
            self.assertNotIn("secret-key-for-tests", json.dumps(step))
            realtime = step["realtime"]
            self.assertGreaterEqual(realtime["completed"], 4, "two sessions speak several turns a step")
            self.assertEqual(realtime["completed"], realtime["requests"])
            self.assertEqual(realtime["empty_transcripts"], 0)
            self.assertGreater(realtime["first_partial_s"]["p50"], 0.1, "counted from the first frame")
            self.assertLess(realtime["first_partial_after_vad_s"]["p95"], 0.4)
            self.assertGreater(realtime["final_after_vad_s"]["p50"], 0.02)
            self.assertGreater(realtime["final_after_audio_s"]["p50"], realtime["final_after_vad_s"]["p50"])
            self.assertTrue(step["slo"]["pass"])
        self.assertEqual(baseline["batch"]["requests"], 0)
        self.assertIsNone(baseline["batch"]["audio_seconds_per_second"])
        self.assertIsNone(baseline["slo"]["checks"]["degradation_asr_final_p95"]["pass"])
        batch = loaded["batch"]
        self.assertEqual((batch["kind"], batch["model"], batch["concurrency"], batch["service_tier"]),
                         ("asr", "Whisper-Large-v3-Turbo", 3, "batch"))
        self.assertGreater(batch["completed"], 6)
        self.assertGreater(batch["audio_seconds_per_second"], 0.5)
        self.assertGreater(batch["latency_s"]["p50"], 0.02)
        self.assertIsNotNone(loaded["slo"]["checks"]["degradation_asr_final_p95"]["pass"])

        metrics = loaded["server_metrics"]
        self.assertTrue(metrics["available"])
        duration = metrics["histograms"]["izwi_request_duration_seconds"]["batch"]
        self.assertEqual(duration["count"], batch["completed"])
        self.assertIn("izwi_request_ttft_seconds", metrics["missing"])
        self.assertEqual(baseline["server_metrics"]["histograms"]["izwi_request_duration_seconds"], {})

        self.assertEqual((summary["schema"], summary["schema_version"]), ("izwi.mixed_load", 1))
        self.assertEqual(set(summary), {"schema", "schema_version", "started_at", "ended_at", "metadata",
                                        "config", "slo_limits", "steps", "slo"})
        self.assertEqual([step["file"] for step in summary["steps"]],
                         ["step-00-batch-c0.json", "step-01-batch-c3.json"])
        self.assertEqual(summary["steps"][1]["batch"]["audio_seconds_per_second"], batch["audio_seconds_per_second"])
        self.assertEqual(summary["steps"][1]["realtime"]["final_after_vad_s"]["p95"],
                         loaded["realtime"]["final_after_vad_s"]["p95"])
        self.assertTrue(summary["slo"]["pass"])
        self.assertEqual(summary["slo_limits"]["asr_final_s"], 0.8)

        # The wire: transcription sessions with the bearer token; the batch
        # tier as a form field and a header.
        self.assertTrue(all(socket["path"] == "/v1/realtime?intent=transcription" for socket in server.sockets))
        self.assertTrue(all(socket["Authorization"] == "Bearer secret-key-for-tests" for socket in server.sockets))
        self.assertEqual(server.updates[0]["audio"]["input"]["transcription"]["model"], "Parakeet-TDT-0.6B-v3")
        posts = [(headers, body) for path, headers, body in server.requests if path == "/v1/audio/transcriptions"]
        self.assertEqual(len(server.requests), len(posts))
        for headers, body in posts:
            self.assertEqual(headers["X-Izwi-Service-Tier"], "batch")
            self.assertIn(b'name="service_tier"\r\n\r\nbatch', body)
            self.assertIn(b'name="model"\r\n\r\nWhisper-Large-v3-Turbo', body)
        self.assertIn("step-01-batch-c3: real-time", printed)

    def test_tts_streams_report_gaps_and_batch_refusals(self):
        for stream_format in ("audio", "sse"):
            with self.subTest(stream_format), tempfile.TemporaryDirectory() as directory, \
                    fake_izwi(refuse_batch_every=2) as (base, server):
                out = pathlib.Path(directory) / "run"
                code, printed = run([
                    "--base-url", base, "--out", str(out), "--realtime-kind", "tts",
                    "--realtime-model", "Kokoro-82M", "--realtime-streams", "2",
                    "--batch-model", "Some-New-TTS", "--batch-levels", "0,2",
                    "--step-secs", "0.5", "--batch-lead-s", "0.05", "--retry-delay", "0.01",
                    "--tts-stream-format", stream_format, "--no-warmup"])
                self.assertEqual(code, 0, printed)
                baseline = json.loads((out / "step-00-batch-c0.json").read_text())
                loaded = json.loads((out / "step-01-batch-c2.json").read_text())
            self.assertEqual(set(loaded), STEP_KEYS)
            realtime = loaded["realtime"]
            self.assertGreaterEqual(realtime["completed"], 4, "each stream is replaced when it ends")
            self.assertEqual(realtime["errors"], 0)
            self.assertLess(realtime["first_audio_s"]["p95"], 0.4)
            gaps = realtime["inter_chunk_gap_s"]
            # Five chunks a stream; a read may split or join one.
            self.assertGreaterEqual(gaps["count"], realtime["completed"] * 3)
            self.assertGreater(gaps["p50"], 0.02)
            self.assertGreaterEqual(gaps["max"], gaps["p95"])
            self.assertAlmostEqual(realtime["audio_seconds"], realtime["completed"] * 0.5)
            batch = loaded["batch"]
            self.assertGreater(batch["completed"], 0)
            self.assertGreater(batch["refused"]["http_429"], 0)
            self.assertEqual(batch["refused"]["total"], batch["refused"]["http_429"])
            self.assertEqual(batch["errors"], 0)
            self.assertEqual(batch["requests"], batch["completed"] + batch["refused"]["total"])
            self.assertAlmostEqual(batch["audio_seconds"], batch["completed"] * 0.5)
            self.assertIn("queue full", batch["error_samples"][0])
            audio = loaded["server_metrics"]["counters"]["izwi_tts_output_audio_seconds_total"]
            self.assertAlmostEqual(audio["total"], batch["audio_seconds"])
            self.assertEqual(baseline["batch"]["requests"], 0)
            # No tier was asked for: nothing about it is sent.
            for path, headers, body in server.requests:
                self.assertNotIn("X-Izwi-Service-Tier", headers)
                self.assertNotIn("service_tier", json.loads(body))
            streamed = [json.loads(body) for _, _, body in server.requests if b"stream_format" in body]
            self.assertTrue(all(body["model"] == "Kokoro-82M" and body["stream_format"] == stream_format
                                for body in streamed))

    def test_refused_sessions_are_counted_and_reopened(self):
        with tempfile.TemporaryDirectory() as directory, fake_izwi(refuse_sessions=2, metrics=False) as (base, _):
            out = pathlib.Path(directory) / "run"
            code, printed = run([
                "--base-url", base, "--out", str(out), "--realtime-kind", "asr", "--realtime-model", "m",
                "--realtime-streams", "2", "--batch-levels", "0", "--step-secs", "0.5",
                "--audio", str(write_speech(directory)), "--silence-ms", "20", "--trailing-silence-ms", "20",
                "--retry-delay", "0.01", "--no-warmup"])
            self.assertEqual(code, 0, printed)
            step = json.loads((out / "step-00-batch-c0.json").read_text())
        realtime = step["realtime"]
        self.assertEqual(realtime["refused"], {"http_429": 0, "http_503": 0, "ws_1013": 2, "total": 2})
        self.assertEqual(realtime["errors"], 0)
        self.assertGreaterEqual(realtime["completed"], 2, "the sessions reopen and speak")
        self.assertIn("capacity is exhausted", realtime["error_samples"][0])
        # A server without the metrics endpoint: recorded, not fatal.
        self.assertFalse(step["server_metrics"]["available"])
        self.assertIn("404", step["server_metrics"]["error"])
        self.assertIn("server metrics not scraped", printed)

    def test_a_refused_upgrade_is_an_http_refusal(self):
        with tempfile.TemporaryDirectory() as directory, fake_izwi(upgrade_status=503) as (base, _):
            out = pathlib.Path(directory) / "run"
            arguments = ["--base-url", base, "--out", str(out), "--realtime-kind", "asr", "--realtime-model", "m",
                         "--realtime-streams", "1", "--batch-levels", "0", "--step-secs", "0.2",
                         "--audio", str(write_speech(directory)), "--retry-delay", "0.05", "--no-warmup"]
            code, printed = run(arguments)
            self.assertEqual(code, 0, "an SLO failure does not fail the run unless it is enforced")
            step = json.loads((out / "step-00-batch-c0.json").read_text())
            self.assertEqual(run(arguments + ["--enforce-slo"])[0], load.SLO_FAILURE_EXIT)
            summary = json.loads((out / "summary.json").read_text())
        realtime = step["realtime"]
        self.assertEqual(realtime["completed"], 0)
        self.assertEqual(realtime["refused"]["http_503"], realtime["requests"])
        self.assertFalse(step["slo"]["checks"]["asr_first_partial_p95"]["pass"])
        self.assertIn("SLO FAIL asr_first_partial_p95, asr_final_p95", printed)
        self.assertEqual(summary["slo"]["failed_steps"], ["step-00-batch-c0"])
        self.assertEqual(summary["slo"]["checks"]["asr_final_p95"], False)

    def test_a_failed_warm_up_stops_the_run(self):
        with tempfile.TemporaryDirectory() as directory, fake_izwi(upgrade_status=500) as (base, _):
            out = pathlib.Path(directory) / "run"
            code, printed = run(["--base-url", base, "--out", str(out), "--realtime-kind", "asr",
                                 "--realtime-model", "m", "--audio", str(write_speech(directory)),
                                 "--retry-delay", "0.01"])
            self.assertEqual(code, 1)
            self.assertIn("warm-up failed", printed)
            self.assertEqual(list(out.iterdir()), [])

    def test_real_time_asr_needs_24khz_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            code, printed = run(["--out", str(pathlib.Path(directory) / "run"), "--realtime-kind", "asr",
                                 "--realtime-model", "m", "--audio", str(write_speech(directory, rate=16_000))])
        self.assertEqual(code, 2)
        self.assertIn("24 kHz", printed)


if __name__ == "__main__":
    unittest.main()
