# Sortformer probability-parity fixtures

NeMo reference outputs for the ignored real-checkpoint tests in
`src/models/architectures/sortformer/diarization/parity_tests.rs`.

## Provenance

Recorded on 2026-10-14 (Apple M1 Pro, CPU, F32) by the diarization audit:

- **v2.1** (`v21-*.json`): `ref_sortformer.py`, a minimal PyTorch port of NeMo main's
  `SortformerEncLabelModel` streaming path (features, dw_striding, Conformer, transformer, head,
  `streaming_update`, `_compress_spkcache`, `ts_vad_post_processing`), loading the
  `diar_streaming_sortformer_4spk-v2.1.nemo` weights (sha256 `8abd3283…2ef8`, HF main). Its card
  very-high-latency run on linearly resampled diarization-2 reproduces izwi's 17 golden segments
  exactly.
- **Nemotron-3** (`nemotron3-*.json`): `ref_nemotron.py`, a port of NeMo main@0c658ae for
  `Nemotron-3-Diarization.nemo` (sha256 `867c53f5…4d96d`, HF rev f667ed73), cross-checked against
  HF transformers 5.19.0 `Nemotron3DiarizationForAudioFrameClassification`: max |Δp| ≤ 1.52e-6,
  no binary mismatches.

The generator scripts were not kept. The fixtures are field selections from their output JSONs
(`scripts/fixtures/extract_sortformer_parity_fixtures.py`); no value was changed.

| File | Audio | Runs | Probabilities |
|---|---|---|---|
| `v21-diarization.json` | `diarization.soxr16k.f32` | model, card_high, card_low | 80 ms, 4 decimals (model, card_high) |
| `v21-diarization-2.json` | `diarization-2.soxr16k.f32` | model, card_high, card_low | 80 ms, 4 decimals (model, card_high) |
| `v21-sparse2.json` | `sparse2.soxr16k.f32` | model, card_high, and card_high with izwi's pre-SRT-1 cache order | per-channel summaries only |
| `nemotron3-diarization-10s.json` | `diarization_10s.ffmpeg16k_s16.f32` | test_small, test_small_rc | 10 ms, all 8 channels |
| `nemotron3-diarization-2.json` | `diarization-2.ffmpeg16k.f32` | yaml, card_offline | 100 ms means of the active channels, per-channel maxima |

Profiles (80 ms frames, chunk / right context / FIFO / update period / speaker cache):

- v2.1 `model` = checkpoint yaml 188/1/0/188/188; `card_high` = card very high latency
  340/40/40/300/188; `card_low` = card low latency 6/7/188/144/188 (NeMo's FIFO pop arithmetic).
- Nemotron-3 `yaml` = checkpoint 264/0/0/264/264; `card_offline` = card very high latency
  340/40/40/300/264, left context 0; `test_small` 16/0/0/16/32 and `test_small_rc` 16/4/8/12/32
  exercise compression, FIFO and right context in 10 s.

Nemotron-3 `yaml` on diarization-2 is asserted only up to its first speaker-cache compression
(42.24 s): that compression meets an exact score tie across the top-k cut, and which tied frames
`torch.topk` keeps is implementation-defined (izwi keeps the lowest indices).

Segments are NeMo's default post-processing (onset = offset = 0.5, no padding or minimum
durations) on raw channel ids, as `[start, end, channel]`.

## Inputs

A different resampler moves Sortformer probabilities far beyond the 2e-3 tolerance, so the tests
read the exact samples the references consumed. Rebuild them with:

```
python3 -I scripts/fixtures/prepare_sortformer_parity_audio.py data target/sortformer-parity-audio
```

The tests read `IZWI_SORTFORMER_PARITY_AUDIO_DIR` (default `target/sortformer-parity-audio`) and
`IZWI_MODELS_DIR`, and skip when either input is missing. Run them in release:

```
cargo test --release -p izwi-core --lib parity_tests -- --ignored --nocapture
```
