#!/usr/bin/env python3
"""Rebuild the 16 kHz inputs the Sortformer probability-parity tests feed.

The NeMo references in crates/izwi-core/tests/fixtures/sortformer_parity/
were produced from 16 kHz audio resampled outside izwi, and a different
resampler moves Sortformer probabilities by far more than the 2e-3 parity
tolerance (izwi linear vs ffmpeg: max |dp| 0.5 on Nemotron). The tests
therefore read these exact samples instead of izwi's own decoder output:

* ``*.soxr16k.f32``: ffmpeg decode to float at the native rate, mean
  downmix, then ``soxr.resample(..., "HQ")``. The v2.1 references used it.
* ``*.ffmpeg16k.f32``: ``ffmpeg -ac 1 -ar 16000`` (swr) to float. The
  Nemotron-3 diarization-2 references used it.
* ``diarization_10s.ffmpeg16k_s16.f32``: ``ffmpeg -t 10 -ac 1 -ar 16000
  -c:a pcm_s16le``, then int16 / 32768. The Nemotron-3 10 s fixture.
* ``sparse2.soxr16k.f32``: the SRT-1 synthetic built from the soxr
  diarization-2 samples: [0,5) s + 40 s of zeros + [42,62) s + 5 s of
  zeros + [63,75) s + [5,15) s (92 s).

Every file is raw little-endian f32 mono at 16 kHz.

Usage (needs ffmpeg, numpy and python-soxr; run from the repo root):

    python3 -I scripts/fixtures/prepare_sortformer_parity_audio.py \
        data target/sortformer-parity-audio

The ignored real-checkpoint tests read ``IZWI_SORTFORMER_PARITY_AUDIO_DIR``
(default ``target/sortformer-parity-audio``) and skip when a file is absent.
"""
import pathlib
import subprocess
import sys

import numpy as np
import soxr

SR = 16_000


def ffmpeg_f32(path, *args):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), *args, "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype="<f4")


def native_rate_and_channels(path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rate, channels = out.split(",")
    return int(rate), int(channels)


def soxr16k(path):
    rate, channels = native_rate_and_channels(path)
    pcm = ffmpeg_f32(path).reshape(-1, channels).mean(axis=1)
    return soxr.resample(pcm, rate, SR, "HQ").astype(np.float32)


def ffmpeg16k(path):
    return ffmpeg_f32(path, "-ac", "1", "-ar", str(SR))


def ffmpeg16k_s16(path, seconds):
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-t", str(seconds),
            "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", "-f", "s16le", "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return (np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0)


def sparse2(diar2):
    def span(start, end):
        return diar2[int(start * SR):int(end * SR)]

    def zeros(seconds):
        return np.zeros(int(seconds * SR), dtype=np.float32)

    return np.concatenate(
        [span(0, 5), zeros(40), span(42, 62), zeros(5), span(63, 75), span(5, 15)]
    )


def main():
    data, out = map(pathlib.Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=True)
    diar1, diar2 = data / "diarization.wav", data / "diarization-2.mp3"
    diar2_soxr = soxr16k(diar2)
    files = {
        "diarization.soxr16k.f32": soxr16k(diar1),
        "diarization-2.soxr16k.f32": diar2_soxr,
        "diarization-2.ffmpeg16k.f32": ffmpeg16k(diar2),
        "diarization_10s.ffmpeg16k_s16.f32": ffmpeg16k_s16(diar1, 10),
        "sparse2.soxr16k.f32": sparse2(diar2_soxr),
    }
    for name, samples in files.items():
        samples.astype("<f4").tofile(out / name)
        print(f"{name}: {len(samples)} samples")


if __name__ == "__main__":
    main()
