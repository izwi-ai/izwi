"""Regenerate whisper-hf-features-fox.json (golden Whisper log-mel features).

Run from the repo root with transformers 4.57.3 + torch:
    HF_HUB_OFFLINE=1 python -I crates/izwi-core/tests/fixtures/whisper-hf-features-fox.py \
        data/fox.wav <whisper-large-v3-turbo dir> crates/izwi-core/tests/fixtures/whisper-hf-features-fox.json

The input is data/fox.wav's int16 PCM / 32768 read directly as 16 kHz samples
(no resampling), so izwi's test can rebuild it bit-exactly. The "click" case
trims the tiled audio to exactly 30 s and ends it with an 80-sample 2 kHz
burst, so the loudest STFT frame is the extra 3001st one that Whisper drops
before taking the normalization max.
"""
import json
import sys
import wave

import numpy as np
from transformers import WhisperFeatureExtractor

wav_path, model_dir, out = sys.argv[1:4]
w = wave.open(wav_path)
assert w.getsampwidth() == 2 and w.getnchannels() == 1
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
fe = WhisperFeatureExtractor.from_pretrained(model_dir)
cases = {
    "fox_full": (0, len(pcm), 1, False),
    "fox_half_second": (16000, 8000, 1, False),
    "fox_tiled_over_30s": (0, len(pcm), 6, False),
    "fox_30s_ending_in_click": (0, len(pcm), 6, True),
}
rng = np.random.default_rng(20261013)
res = {"source": "transformers 4.57.3 WhisperFeatureExtractor (torch path), whisper-large-v3-turbo preprocessor_config",
       "input": "data/fox.wav int16 PCM / 32768 read as 16 kHz samples; slice [start, start+len) tiled `tile` times",
       "cases": {}}
for name, (start, length, tile, click) in cases.items():
    x = np.tile(pcm[start:start + length], tile)
    if click:
        x = x[:480000].copy()
        x[-80:] = np.array([0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5] * 10, dtype=np.float32)
    f_t = fe(x, sampling_rate=16000, return_tensors="np")["input_features"][0]
    padded = np.pad(x[:480000], (0, max(0, 480000 - len(x))))[None]
    f_n = fe._np_extract_fbank_features(padded, "cpu")[0]
    print(name, len(x), f_t.shape, "torch-vs-numpy max abs diff", np.abs(f_t - f_n).max())
    nframes = f_t.shape[1]
    audio_frames = min(nframes, len(x) // 160 + 1)
    tail = min(nframes, audio_frames + 8) if tile == 1 else 0
    frames = [0, 1, audio_frames - 1, audio_frames, audio_frames + 1, nframes - 2, nframes - 1]
    frames += [int(f) for f in rng.choice(nframes, 5, replace=False)]
    frames = sorted({f for f in frames if 0 <= f < nframes})
    res["cases"][name] = {
        "start": start, "len": length, "tile": tile, "click": click,
        "frames": nframes, "mels": int(f_t.shape[0]),
        "max": float(f_t.max()), "min": float(f_t.min()),
        "mean": float(f_t.astype(np.float64).mean()),
        "frame_means": [round(float(v), 6) for v in f_t[:, :tail].astype(np.float64).mean(axis=0)],
        "floor": float(f_t[:, tail:].max()) if tile == 1 else None,
        "sampled_frames": frames,
        "sampled": [[round(float(v), 6) for v in f_t[:, f]] for f in frames],
    }
json.dump(res, open(out, "w"), separators=(",", ":"))
