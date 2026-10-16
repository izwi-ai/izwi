#!/usr/bin/env python3
"""Unit tests for run-voice-realtime-load.py: WebSocket framing, izwi audio
frames, statistics, Prometheus histogram deltas, per-turn latencies, and one
end-to-end voice run against a local fake voice socket."""
import argparse
import asyncio
import importlib.util
import json
import os
import pathlib
import struct
import tempfile
import time
import unittest
import wave

spec = importlib.util.spec_from_file_location(
    "voiceload", pathlib.Path(__file__).with_name("run-voice-realtime-load.py"))
load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load)


def server_frame(opcode, payload):
    """An unmasked server-to-client frame."""
    header = bytearray([0x80 | opcode])
    if len(payload) < 126:
        header.append(len(payload))
    else:
        header.append(126)
        header += struct.pack("!H", len(payload))
    return bytes(header) + payload


class Framing(unittest.TestCase):
    def test_client_frames_are_masked_with_the_right_length_encoding(self):
        mask = b"\x01\x02\x03\x04"
        for length, header_len in ((5, 6), (200, 8), (70_000, 14)):
            payload = bytes(range(256)) * (length // 256) + bytes(length % 256)
            frame = load.encode_ws_frame(load.OP_BINARY, payload, mask)
            fin, opcode, masked, marker = load.parse_ws_header(frame[:2])
            self.assertTrue(fin and masked)
            self.assertEqual(opcode, load.OP_BINARY)
            self.assertEqual(len(frame), header_len + length)
            body = frame[header_len:]
            unmasked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(body))
            self.assertEqual(unmasked, payload)
            self.assertEqual(marker, length if length < 126 else (126 if length < 1 << 16 else 127))

    def test_client_pcm16_frame_layout(self):
        frame = load.client_pcm16_frame(load.VOICE_MAGIC, 7, 24_000, b"\x01\x00")
        self.assertEqual(frame[:4], b"IVWS")
        self.assertEqual(frame[4:6], bytes([1, 1]))
        self.assertEqual(struct.unpack_from("<II", frame, 8), (24_000, 7))
        self.assertEqual(frame[16:], b"\x01\x00")

    def test_assistant_frames_parse_for_every_output_format(self):
        for kind, name in ((2, "pcm_i16"), (3, "g711_ulaw"), (4, "g711_alaw")):
            frame = b"IVWS" + bytes([1, kind]) + struct.pack("<HQII", 1, 9, 3, 8_000) + b"xy"
            self.assertEqual(load.parse_assistant_audio_frame(frame), (name, True, 9, 3, 8_000, b"xy"))
        self.assertIsNone(load.parse_assistant_audio_frame(b"IVWS" + bytes([1, 1]) + bytes(18)))
        self.assertIsNone(load.parse_assistant_audio_frame(b"short"))

    def test_wav_is_read_mono_and_cut_into_20ms_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stereo.wav")
            with wave.open(path, "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(struct.pack("<4h", 1, -1, 2, -2) * 400)
            pcm, rate = load.read_wav(path)
        self.assertEqual(rate, 16_000)
        self.assertEqual(struct.unpack("<4h", pcm[:8]), (1, 2, 1, 2))
        frames = load.pcm_frames(pcm, rate)
        self.assertEqual([len(frame) for frame in frames], [640, 640, 320])


class Statistics(unittest.TestCase):
    def test_percentiles_interpolate_linearly(self):
        values = list(range(1, 11))
        self.assertEqual(load.percentile(values, 50), 5.5)
        self.assertAlmostEqual(load.percentile(values, 90), 9.1)
        self.assertIsNone(load.percentile([], 50))
        summary = load.summarize([3.0, None, 1.0, 2.0])
        self.assertEqual((summary["count"], summary["p50"], summary["max"]), (3, 2.0, 3.0))
        self.assertEqual(load.summarize([None]), {"count": 0})

    def test_histogram_quantiles_come_from_bucket_deltas(self):
        def scrape(counts):
            lines = ["# TYPE izwi_voice_turn_first_audio_seconds histogram"]
            for bound, count in zip(("0.5", "1", "2", "+Inf"), counts):
                lines.append(f'izwi_voice_turn_first_audio_seconds_bucket{{le="{bound}"}} {count}')
            lines.append(f"izwi_voice_turn_first_audio_seconds_count {counts[-1]}")
            return "\n".join(lines)

        name = "izwi_voice_turn_first_audio_seconds"
        before = load.parse_histograms(scrape((1, 1, 1, 1)), (name,))[name]
        after = load.parse_histograms(scrape((1, 3, 5, 5)), (name,))[name]
        buckets, count = load.histogram_delta(before, after)
        self.assertEqual(count, 4)
        self.assertEqual(buckets, [(0.5, 0.0), (1.0, 2.0), (2.0, 4.0), (float("inf"), 4.0)])
        # Two of four samples fall in (0.5, 1]: the median is that bucket's top.
        self.assertAlmostEqual(load.histogram_quantile(buckets, count, 0.5), 1.0)
        self.assertAlmostEqual(load.histogram_quantile(buckets, count, 0.75), 1.5)
        self.assertIsNone(load.histogram_quantile(buckets, 0, 0.5))

    def test_turn_latencies_run_from_the_servers_speech_end(self):
        audio = b"IVWS" + bytes([1, 2]) + struct.pack("<HQII", 0, 1, 0, 24_000) + b"\x00\x00"
        events = [
            (10.0, {"type": "user_speech_start"}),
            (11.0, {"type": "user_speech_end"}),
            (11.3, {"type": "user_transcript_final"}),
            (11.9, audio),
            (12.0, audio),
            (12.5, {"type": "error", "message": "boom"}),
            (13.0, {"type": "turn_done", "status": "ok"}),
        ]
        record = load.turn_record(events)
        self.assertEqual(record["status"], "ok")
        self.assertAlmostEqual(record["time_to_first_audio_ms"], 900.0)
        self.assertAlmostEqual(record["transcript_latency_ms"], 300.0)
        self.assertAlmostEqual(record["turn_ms"], 2000.0)
        self.assertEqual(record["errors"], ["boom"])


class FakeVoiceServer:
    """Answers the voice protocol: ready events, then once the client has
    sent `frames_per_turn` audio frames, one turn's events and one audio
    frame."""

    def __init__(self, frames_per_turn):
        self.frames_per_turn = frames_per_turn
        self.sessions = 0

    async def handle(self, reader, writer):
        self.sessions += 1
        while (await reader.readline()) not in (b"\r\n", b""):
            pass
        writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                     b"Connection: Upgrade\r\n\r\n")
        socket = load.WebSocket(reader, writer)

        def send_json(value):
            writer.write(server_frame(load.OP_TEXT, json.dumps(value).encode()))

        frames, turn = 0, 0
        while True:
            try:
                message = await socket.recv()
            except asyncio.IncompleteReadError:
                break
            if message is None:
                break
            if isinstance(message, str):
                kind = json.loads(message)["type"]
                if kind == "session_start":
                    send_json({"type": "session_ready"})
                elif kind == "input_stream_start":
                    send_json({"type": "input_stream_ready"})
                continue
            frames += 1
            if frames % self.frames_per_turn == 0:
                turn += 1
                send_json({"type": "user_speech_end", "utterance_seq": turn})
                await asyncio.sleep(0.01)
                send_json({"type": "user_transcript_final", "text": "hi"})
                await asyncio.sleep(0.02)
                writer.write(server_frame(load.OP_BINARY, b"IVWS" + bytes([1, 2]) +
                                          struct.pack("<HQII", 1, turn, 0, 24_000) + b"\x00\x00"))
                send_json({"type": "turn_done", "status": "ok"})
            await writer.drain()
        writer.close()


class EndToEnd(unittest.TestCase):
    def test_concurrent_voice_sessions_report_every_turn(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "speech.wav")
                with wave.open(path, "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16_000)
                    handle.writeframes(bytes(16_000 * 2 // 10))  # 100 ms: 5 frames
                # 5 speech frames + (20 + 20) ms of silence = 7 frames a turn.
                fake = FakeVoiceServer(frames_per_turn=7)
                server = await asyncio.start_server(fake.handle, "127.0.0.1", 0)
                port = server.sockets[0].getsockname()[1]
                args = load.parse_args([
                    "--base-url", f"http://127.0.0.1:{port}", "--sessions", "3",
                    "--turns", "2", "--audio", path, "--silence-ms", "20",
                    "--trailing-silence-ms", "20", "--timeout", "20",
                ])
                started = time.monotonic()
                report = await load.run(args)
                server.close()
                await server.wait_closed()
                return report, fake, time.monotonic() - started

        report, fake, elapsed = asyncio.run(scenario())
        self.assertEqual(fake.sessions, 3)
        self.assertEqual(report["statuses"], {"ok": 6})
        ttfa = report["summary"]["time_to_first_audio_ms"]
        self.assertEqual(ttfa["count"], 6)
        self.assertGreater(ttfa["p50"], 20.0)
        self.assertGreater(report["summary"]["transcript_latency_ms"]["p50"], 5.0)
        self.assertIsNone(report["server"])
        self.assertLess(elapsed, 15.0, "sessions run concurrently")


if __name__ == "__main__":
    unittest.main()
