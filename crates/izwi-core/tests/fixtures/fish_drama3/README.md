# Fish Audio Drama 3 fixtures

Parity fixtures for `fishaudio/drama-3` at revision `78e5ab79e49aa9a25abe8799eff2c4d73a1c3216`, produced from
upstream fish-speech at `0f8a12e8a651062fdee9c33a894bdccd3a8513f1`. Every generator runs with `python3 -I` from
the repository root, and imports fish-speech from a checkout passed as `--fish-speech`. The checkout must be at
the pinned commit; `--skip-commit-check` records the run as unverified instead of failing. Each script's
docstring holds the exact output schema. The shared definitions (the xorshift generator, checksums, trace entries
and the recorded-noise layout) are in `scripts/fixtures/fish_drama3_fixture_common.py`.

| Level | Script | Writes | Consumer |
|---|---|---|---|
| L0 | (hand-made) | `config.json` (the pinned file), `tensor_census.txt`, `published_names.txt` | config and census unit tests |
| L0 | `generate_fish_drama3_goldens.py` | `prompt_goldens.json`, `chunk_goldens.json` | `prompt::tests`, `chunking::tests` (`IZWI_FISH_DRAMA3_TOKENIZER_PATH`) |
| L1 | `generate_fish_drama3_tiny.py` | `tiny/`: checkpoint, `tokenizer.json`, `expected.json`, `expected.safetensors` | `reference_tests::tiny_*` (no env var) |
| L1.5 | `fetch_fish_drama3_slices.py`, then `generate_fish_drama3_slices.py` | `slice_{a,b}.safetensors`, `manifest.json`, `slice_expected.{json,safetensors}` | `reference_tests::slice_*` (`IZWI_FISH_DRAMA3_SLICE_DIR`) |
| L2 | `generate_fish_drama3_reference.py` | `reference.{json,safetensors}`, `audio/*.f32` | `reference_tests::reference_*` (`IZWI_FISH_DRAMA3_REFERENCE_DIR`, `IZWI_FISH_DRAMA3_MODEL_DIR`, `IZWI_FISH_DRAMA3_BACKEND`) |

Only `tiny/` lives in this directory. The slice and reference outputs default to `target/` and are never
committed.

Every `reference_tests` test except `slice_inputs_are_the_documented_xorshift` is `#[ignore]`d, and each fails at
once, naming the generator to run, when its fixture or env var is missing. The harness, its tolerances and why
each has its value are in `src/models/architectures/fish_drama3/reference_tests.rs`. Each test prints the worst
measured value of every stage beside its bound, so the first real run calibrates the bounds. The L1.5 and L2 tests
run real weights: build them with `--release` (a release test build needs several GB of `target/`).

## L1: tiny upstream fixture (any Python + CPU torch box)

```sh
python3 -I scripts/fixtures/generate_fish_drama3_tiny.py --fish-speech /path/to/fish-speech
cargo test -p izwi-core --lib fish_drama3::reference_tests::tiny -- --ignored
# Metal (fails clearly until the family's Metal residency lands in Phase 3):
cargo test -p izwi-core --lib --features metal fish_drama3::reference_tests::tiny -- --ignored
```

| Test | Proves |
|---|---|
| `tiny_checkpoint_is_the_rust_synthetic_checkpoint` | `SyntheticCheckpoint::write(seed)` hashes to `checkpoint.tensor_digest`, so the generator's Python port of the weights is exact; `tiny/config.json` and a committed checkpoint match too, when present |
| `tiny_prompts_match_upstream_rows` | the four prompt shapes build upstream's rows exactly; the ids and layer types agree |
| `tiny_prefill_and_forced_steps_match_upstream_on_cpu` | every trunk layer's residual, the mixer and MoE outputs of layers 0 and 3, the routes of every layer (as sets, near ties counted), the final norm, all 13 tower layers, the teacher-forced head logits, the main logits, and two forced decode steps (hidden, main and head logits) |
| `tiny_rollouts_replay_upstream_on_cpu` | the five rollouts under the recorded noise: RAS windows, both main draws, RAS decisions and every code, exactly until a recorded near tie; the model loop itself makes the same frames |
| `tiny_matches_upstream_on_metal` (`--features metal`) | the same on Metal; a TODO until Phase 3 |

The tests load the checkpoint from `tiny/` when the generator's checkpoint files are committed there. Otherwise
they build `SyntheticCheckpoint::write(seed)` and use `tiny/tokenizer.json`, after checking the digest. So only
`expected.json`, `expected.safetensors` and `tokenizer.json` need to be committed.

- **Geometry.** The checkpoint has the L0 synthetic geometry (`synthetic.rs::trunk_geometry_json`).
- **Weights.** Its tensors are a port of `synthetic.rs::trained_like_tensors(config, seed)`.
  `expected.json` → `checkpoint.tensor_digest` lets the Rust test check that `SyntheticCheckpoint::write(seed)`
  produces the same tensors.
- **Run.** Upstream loads the directory with `FishOmniModel.from_pretrained` and runs it in float32 (W8A16).
- **Outputs.** It writes prompt rows, traces, forced-code head logits, and five 32-frame rollouts under recorded
  noise.
- **Cost.** It takes about 1–2 minutes and less than 2 GB of RAM.

## L1.5: real-weight slices (this Mac; about 4.2 GB of downloads, less than 8 GB of RAM)

```sh
python3 -I scripts/fixtures/fetch_fish_drama3_slices.py --out target/fish-drama3-slices
python3 -I scripts/fixtures/generate_fish_drama3_slices.py --fish-speech /path/to/fish-speech \
    --slice-dir target/fish-drama3-slices
IZWI_FISH_DRAMA3_SLICE_DIR=$PWD/target/fish-drama3-slices \
    cargo test --release -p izwi-core --lib fish_drama3::reference_tests::slice -- --ignored --test-threads 1
```

`--test-threads 1` keeps one F32 load in memory at a time (peak about 7 GB).

| Test | Proves |
|---|---|
| `slice_inputs_are_the_documented_xorshift` (not ignored) | the generators' xorshift, written out from the docstring, is `numeric_check::xorshift_values` bit for bit |
| `slice_trunk_deltanet_layer_0_matches_upstream` | layer 0 (Gated DeltaNet) over prefill chunks [0,128), [128,256), [256,300) and 4 recurrent decode steps: layer output, mixer and MoE per decode row, with checksums and history traces; decode routes as sets. RED: tiled value heads |
| `slice_trunk_attention_layer_3_matches_upstream` | layer 3 (gated attention), the same run. RED: interleaved RoPE pairs in the trunk |
| `slice_tower_matches_upstream_on_a_forced_step` | the tower teacher-forced over 300 rows, then a forced step at row 300 with codes ≥ 1024: every layer's prefill trace and step output, and the 9 heads' logits. RED: half-split RoPE in the tower |
| `slice_main_head_and_embedding_match_upstream` | the compact head over the final norm of xorshift rows, and the trunk embedding of 8 semantic rows. RED: the final norm without its `+1` |

The slices hold only some tensors, so the tests read them through `FishDrama3Checkpoint::open_view`. It renames
slice B's layer `L` to the index it takes in a cut-down trunk config, and names slice A's row slices as whole
tables of a re-based vocabulary. A trunk's paged domain needs an attention layer, so layer 3 rides behind layer 0
in the layer-0 test.

- **What the fetch does.** It range-downloads only the needed tensors at the pin. It checks free disk space first,
  resumes after an interruption, and sends `HF_TOKEN` when set.
- **Slice A.** The tower, the 4,097 compact-head rows, the semantic rows of `embed_tokens`, and
  `codebook_embeddings`.
- **Slice B.** Trunk layers 0 (DeltaNet) and 3 (attention), each with its router and its 256 experts.

## L2: whole-model reference (a GPU with at least 80 GB, or a CPU box with at least 64 GB)

```sh
python3 -I scripts/fixtures/generate_fish_drama3_reference.py --model-dir /path/to/drama-3 \
    --fish-speech /path/to/fish-speech --device cuda --out target/fish-drama3-reference
# Prompts, chunk histories and the take edit (tokenizer only), and the codec PCM (codec only): any machine.
IZWI_FISH_DRAMA3_MODEL_DIR=/path/to/drama-3 IZWI_FISH_DRAMA3_REFERENCE_DIR=$PWD/target/fish-drama3-reference \
    cargo test --release -p izwi-core --lib fish_drama3::reference_tests::reference_ -- --ignored \
    --skip reference_generations
# The whole model: cpu loads F32 weights (about 140 GiB of RAM); cpu-q8_0 the production Q8_0 form (about 40 GiB).
IZWI_FISH_DRAMA3_MODEL_DIR=/path/to/drama-3 IZWI_FISH_DRAMA3_REFERENCE_DIR=$PWD/target/fish-drama3-reference \
IZWI_FISH_DRAMA3_BACKEND=cpu \
    cargo test --release -p izwi-core --lib fish_drama3::reference_tests::reference_generations -- --ignored
```

| Test | Proves |
|---|---|
| `reference_prompts_match_upstream` | every generation's prompt rows; the long text's split, each chunk's prompt and kept history (izwi's chunk session fed upstream's frames); the take edit's history turn holds exactly the utterance's codes |
| `reference_utterance_pcm_matches_the_upstream_codec` | the family codec on the utterance's recorded codes against upstream's codec PCM; it also reports how often izwi's encoder agrees with the recorded reference codes |
| `reference_generations_match_upstream` | every generation: the input rows, hidden checksums at trunk layers 0, 1, 3 and 39 and tower layers 0, 4 and 12, routes, the main logits (all and the top 16), and the first 64 frames under the recorded noise. On `cpu` the frames must match until the first recorded near tie (gap < 1e-4), and the frame where they diverge is reported. On `cpu-q8_0` the divergence is reported only |

`IZWI_FISH_DRAMA3_BACKEND` takes `cpu` (default), `cpu-q8_0`, `metal` or `cuda`. The last two fail with the
family's own refusal until its Metal and CUDA residencies exist (Phase 3).

- **Cases.**
  - `plain`, `one_reference` (`data/fox.wav`) and `two_speakers` (`data/fox.wav` + `data/test.wav`): 64 frames each.
  - `long_text`: three chunks (65, 288 and 57 bytes) run through upstream's own `generate_long_drama3` loop with
    the fox reference. Chunks 2 and 3 receive as history the frames upstream generated for the earlier chunks
    (the pinned first chunk plus the latest one).
  - `utterance_10s`: a plain case run to eos, or to the 10 s cap of 216 frames.
  - `take_edit` (plan §8.4): the utterance becomes a completed history turn, followed by the edited text;
    64 frames.
- **Audio.** ffmpeg converts each reference clip once to 44.1 kHz f32. The upstream codec encodes it in F32, and
  the codes are recorded. The utterance's full codes and the upstream codec's F32 PCM decode of them (44.1 kHz)
  are recorded too.
- **Per generation.** The script records the prompt rows, the main logits at the last prompt row, the frames
  under recorded noise, and per-layer checksums. Trunk layers are 0, 1, 3 and 39; tower layers are 0, 4 and 12.
- **Self-check.** Cached decode must match the full forward.
- **Precision.** The experts stay FP8-resident (W8A16).
- **Cost.** About 20–25 minutes on the GPU, or 1–1.5 hours on the CPU. Peak memory stays around 52 GB, because
  the codec is loaded only after the LM is freed. `reference.safetensors` is about 200–250 MB.
- **Leg B.** The official sglang or vllm engine run is not part of this script. It belongs to the e2e kit
  (`scripts/e2e/fish-drama3/`).

## Recorded noise in one paragraph

izwi's race picks `argmax(p / E)` with one exponential per candidate, in candidate order. Upstream draws its
noise in rank order. The generators therefore patch `tower.sample_top_k_top_p` so that upstream reads
`E[candidate ids]`, and they save `E`.

Per decode step, the saved noise is:
- `E_first[C]`, then `E_retry[C]` (the RAS draw, always present);
- then `E_head1..9[V]`.

C is the number of compact candidates (eos first) and V the number of codes. On a step whose main token is eos,
the 9 head draws upstream still makes are cut from the tape. Replays run with izwi's first-frame eos mask off.
