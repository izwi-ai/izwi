//! Parity against upstream fish-speech (`0f8a12e8`), level by level (plan
//! §10.2–10.4). Every test but the xorshift check is ignored: each reads what
//! a generator in `scripts/fixtures/` wrote (its docstring holds the schema)
//! and fails at once, naming the generator, when that output is absent.
//!
//! * **L1 `tiny_*`** read `tests/fixtures/fish_drama3/tiny/`
//!   (`generate_fish_drama3_tiny.py`): upstream run in F32 on the §10.1
//!   synthetic checkpoint. The checkpoint is the generator's when it is
//!   committed there, else `SyntheticCheckpoint::write(seed)`; either way its
//!   tensors must hash to the recorded `tensor_digest`. Prompt rows, every
//!   trunk and tower trace, routes, head and main logits, and five rollouts
//!   replayed under the recorded noise.
//! * **L1.5 `slice_*`** read `IZWI_FISH_DRAMA3_SLICE_DIR`
//!   (`fetch_fish_drama3_slices.py`, then `generate_fish_drama3_slices.py`):
//!   real weights a piece at a time from xorshift inputs, through a
//!   checkpoint view over the slice files and configs cut to match. RED
//!   variants of the RoPE layout, the `1 + w` norm gain and the grouped value
//!   heads must miss.
//! * **L2 `reference_*`** read `IZWI_FISH_DRAMA3_REFERENCE_DIR` and
//!   `IZWI_FISH_DRAMA3_MODEL_DIR` (`generate_fish_drama3_reference.py`): the
//!   prompts and the codec need no LM; the generations load the whole model
//!   on `IZWI_FISH_DRAMA3_BACKEND`.
//!
//! **How they compare.** Rows and token ids exactly. Tensors by relative L2
//! (and the largest absolute difference where it matters), NaN-safe: a
//! non-finite value on either side fails. A trace entry also by every row's
//! RMS and by its checksum. Router decisions as expert sets: a row where
//! either side's top-k margin is under the near-tie threshold may differ, and
//! is counted instead of failed. Recorded-noise rollouts step by step (the
//! RAS window, both main draws, the RAS decision, every code) until the
//! first step whose recorded race or top-p gap is under 1e-4: a divergence
//! before it fails, one at or after it is reported with its frame. The worst
//! measured value of every stage is printed beside its bound, so the first
//! run calibrates the bounds.
//!
//! [`prefill`] and [`replay`] run the model loop's cells and steps
//! (`model/mod.rs`: `prefill_span`, `decode_quantum`) component by component
//! so each can be observed; on the CPU the tiny rollouts also run through the
//! loop itself, which must make the same frames.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use half::bf16;
use memmap2::Mmap;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::checkpoint::{trunk_layer, FishDrama3Checkpoint, INDEX_FILE, LM_HEAD, TRUNK_PREFIX};
use super::chunking::plan_chunks;
use super::codec::{
    FishDrama3CodecArtifact, FishDrama3DacConfig, FishDrama3DacDecoder, FishDrama3DacStreamState,
};
use super::config::{
    FishDrama3Config, FishDrama3ConfigPolicy, FishDrama3LayerKind, FishDrama3RopeLayout,
};
use super::embed::{prompt_rows, FishDrama3Embedding};
use super::head::FishDrama3CompactHead;
use super::linear::FishDrama3KernelPhase;
use super::model::{
    FishDrama3CancelCheck, FishDrama3ChunkSession, FishDrama3ChunkSessionOptions,
    FishDrama3FrameCodec, FishDrama3GenerationParams, FishDrama3Model, FishDrama3ModelOptions,
    FishDrama3PagedDomain, FishDrama3StopReason, MAX_REQUEST_POSITIONS,
};
use super::prompt::{
    build_prompt, FishDrama3Codes, FishDrama3HistoryTurn, FishDrama3Prompt, FishDrama3Reference,
};
use super::residency::FishDrama3CpuPrecision;
use super::sampling::{FishDrama3MainDraw, FishDrama3RowSampler, FishDrama3SamplingParams};
use super::synthetic::{temp_dir, trunk_geometry, SyntheticCheckpoint, TempDir, Written};
use super::tokenizer::FishDrama3Tokenizer;
use super::tower::{
    FishDrama3Tower, FishDrama3TowerForcedRow, FishDrama3TowerOptions, FishDrama3TowerStepRow,
};
use super::trunk::{
    norm_gain, rms_norm_rows, route, FishDrama3StepRow, FishDrama3Trunk, FishDrama3TrunkOptions,
    FishDrama3TrunkState, LayerTrace, NormGain,
};
use crate::error::{Error, Result};
use crate::kernels::numeric_check::{max_abs_diff, relative_l2, xorshift_values};
use crate::models::shared::attention::physical::{
    test_physical_caches_for_model_layers_on, PhysicalPagedKvCache,
};

const SLICE_DIR_ENV: &str = "IZWI_FISH_DRAMA3_SLICE_DIR";
const MODEL_DIR_ENV: &str = "IZWI_FISH_DRAMA3_MODEL_DIR";
const REFERENCE_DIR_ENV: &str = "IZWI_FISH_DRAMA3_REFERENCE_DIR";
const BACKEND_ENV: &str = "IZWI_FISH_DRAMA3_BACKEND";

const TINY_SCHEMA: &str = "izwi.fish_drama3.tiny_expected";
const SLICE_SCHEMA: &str = "izwi.fish_drama3.slice_expected";
const REFERENCE_SCHEMA: &str = "izwi.fish_drama3.reference";

const TINY_HOW: &str = "generate the L1 fixture once with `python3 -I \
    scripts/fixtures/generate_fish_drama3_tiny.py --fish-speech /path/to/fish-speech` \
    (fish-speech at 0f8a12e8; tests/fixtures/fish_drama3/README.md)";
const SLICE_HOW: &str = "point IZWI_FISH_DRAMA3_SLICE_DIR at the directory `python3 -I \
    scripts/fixtures/fetch_fish_drama3_slices.py --out DIR` and then `python3 -I \
    scripts/fixtures/generate_fish_drama3_slices.py --fish-speech /path/to/fish-speech \
    --slice-dir DIR` wrote (tests/fixtures/fish_drama3/README.md, L1.5)";
const REFERENCE_HOW: &str = "point IZWI_FISH_DRAMA3_REFERENCE_DIR at the --out of `python3 -I \
    scripts/fixtures/generate_fish_drama3_reference.py` and IZWI_FISH_DRAMA3_MODEL_DIR at \
    fishaudio/drama-3 at 78e5ab79 (tests/fixtures/fish_drama3/README.md, L2)";

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// One stage's bound: relative L2 over the compared values and the largest
/// absolute difference (infinite where it adds nothing). A non-finite value
/// on either side fails both.
#[derive(Debug, Clone, Copy)]
struct Bound {
    rel_l2: f64,
    max_abs: f64,
}

const fn bound(rel_l2: f64, max_abs: f64) -> Bound {
    Bound { rel_l2, max_abs }
}

const fn rel(rel_l2: f64) -> Bound {
    bound(rel_l2, f64::INFINITY)
}

/// The bounds of one run against upstream's F32 reference, stage by stage.
#[derive(Debug, Clone, Copy)]
struct Tolerances {
    /// The trunk's input rows: BF16 tables widened to F32 on both sides, so
    /// only the `/ sqrt(2)` of semantic rows may round differently.
    embedding: Bound,
    /// Trunk traces: each layer's residual, mixer and MoE outputs, the final
    /// norm.
    trunk: Bound,
    /// Tower traces and per-layer step outputs.
    tower: Bound,
    /// Main and head logits at prefill and at forced steps.
    logits: Bound,
    /// A rollout's per-step hidden and main logits.
    decode: Bound,
    /// A router margin `p[k-1] - p[k]` under this is a near tie (the
    /// generators' `--route-tie`).
    route_tie: f32,
    /// A race or top-p gap under this makes a draw a near tie (§10.4).
    frame_tie: f64,
    /// Frames that must replay exactly unless a near tie comes first.
    frames_required: usize,
}

/// L1: izwi's F32 load (the Q8_0 path is not used: it would add its own
/// ~0.4% weight error) against upstream in F32 with the experts dequantized
/// exactly (W8A16) and the router weights rounded to BF16 on both sides, so
/// only summation order differs: matmul blocking, izwi's chunked DeltaNet
/// against upstream's, softmax order. At width 128 that is about 1e-6 per
/// layer and the untrained 8-layer stack amplifies a perturbation about
/// tenfold (trunk/tests.rs), so 1e-4 leaves a tenfold margin and sits a
/// thousand times under a RED miss. Decode values carry up to 32 steps of
/// recurrent state and KV: 3e-4. Logits also get an absolute bound: the race
/// reads them.
const TINY: Tolerances = Tolerances {
    embedding: bound(1e-6, 1e-5),
    trunk: rel(1e-4),
    tower: rel(1e-4),
    logits: bound(1e-4, 2e-3),
    decode: rel(3e-4),
    route_tie: 1e-5,
    frame_tie: 1e-4,
    frames_required: usize::MAX,
};

/// L1 on Metal: the Metal form (F16 weights and activations, F32 residual,
/// F16 KV) against upstream's F32 run. F16 rounds each layer's activations
/// to 11 bits, about 5e-4, and the untrained tiny trunk grows that to about
/// 2e-2 by its last layer (the synthetic checkpoint's Metal tests measure
/// 2.0e-2 on the trunk, 6e-3 on the tower, 3e-4 on the compact head).
/// Router margins move with it, so ties widen to 1e-3, and a race whose
/// gap is under 1e-2 may flip: frames replay until the first such tie and
/// a divergence after it is reported, never failed. Uncalibrated: the
/// first run's worst values set them.
const TINY_METAL: Tolerances = Tolerances {
    embedding: bound(2e-7, 1e-5),
    trunk: rel(5e-2),
    tower: rel(2e-2),
    logits: bound(2e-2, 0.2),
    decode: rel(5e-2),
    route_tie: 1e-3,
    frame_tie: 1e-2,
    frames_required: usize::MAX,
};

/// L1 on CUDA: the CUDA form (BF16 weights, activations, residual and KV)
/// against upstream's F32 run. BF16 rounds to 8 bits, about four times
/// F16's step, and the residual itself rounds too (upstream's own BF16
/// numerics), so the tiny trunk's untrained depth grows it to about four
/// times the Metal bounds; ties widen to 5e-3. Uncalibrated: the first run
/// on a GPU sets them.
const TINY_CUDA: Tolerances = Tolerances {
    embedding: bound(2e-7, 1e-5),
    trunk: rel(2e-1),
    tower: rel(8e-2),
    logits: bound(8e-2, 0.8),
    decode: rel(2e-1),
    route_tie: 5e-3,
    frame_tie: 5e-2,
    frames_required: usize::MAX,
};

/// L2 `cuda`: the published model's CUDA form (a ≥ 48 GB GPU) against
/// upstream's F32 W8A16 run: BF16 throughout over 40 layers, a few percent
/// on traces; §10.6 asks for 32 replayed frames before the first near tie.
/// Uncalibrated.
const L2_CUDA: Tolerances = Tolerances {
    embedding: bound(2e-7, 1e-5),
    trunk: rel(1e-1),
    tower: rel(1e-1),
    logits: bound(1e-1, 1.0),
    decode: rel(2e-1),
    route_tie: 5e-3,
    frame_tie: 5e-2,
    frames_required: 32,
};

/// L2 `metal`: the published model's Metal form (a ≥ 64 GB Mac) against
/// upstream's F32 W8A16 run: F16 activations over 40 layers, about a percent
/// on traces; ties and frames as on [`TINY_METAL`], and §10.6 asks for 32
/// replayed frames before the first near tie. Uncalibrated.
const L2_METAL: Tolerances = Tolerances {
    embedding: bound(2e-7, 1e-5),
    trunk: rel(5e-2),
    tower: rel(5e-2),
    logits: bound(5e-2, 0.5),
    decode: rel(1e-1),
    route_tie: 1e-3,
    frame_tie: 1e-2,
    frames_required: 32,
};

/// L1.5: the same F32-against-F32 comparison on real weights, one layer (or
/// the tower, or the head) from exact xorshift inputs: sums of 2,048 (6,144
/// in the tower's feed-forward) in another order, the chunked DeltaNet
/// regrouped, router weights BF16 on both sides. Expected near 1e-6; bounded
/// at 1e-4, a thousand times under the RED misses.
const SLICE: Tolerances = Tolerances {
    embedding: bound(1e-6, 1e-5),
    trunk: rel(1e-4),
    tower: rel(1e-4),
    logits: bound(1e-4, 1e-3),
    decode: rel(1e-4),
    route_tie: 1e-5,
    frame_tie: 1e-4,
    frames_required: usize::MAX,
};

/// L1.5 on Metal: the same real-weight pieces in the Metal form (F16
/// weights and activations, F32 residual, F16 KV, the fused F16 MoE) against
/// upstream's F32 run. Every BF16 weight in F16's normal range is exact in
/// F16, so what differs is F16 rounding of each projection's inputs and
/// outputs (11 bits, about 5e-4 a rounding, a handful per layer): about 1e-3
/// for one trunk layer, where the synthetic checkpoint's Metal tests
/// measure up to 6.3e-3 over eight untrained layers, so 1e-2. The tower's 13
/// layers and the codebook heads after them take the synthetic tower's
/// bound, 2e-2 (measured 6.3e-3), and logits an absolute 0.2 over values of
/// order ten. Router margins move about 1e-3 with the F16-rounded input, so
/// ties widen to it. The embedding's F16 tables hold every BF16 value, so
/// only the semantic rows' `/ sqrt(2)` may round. Uncalibrated: the first
/// run with the slices sets them.
const SLICE_METAL: Tolerances = Tolerances {
    embedding: bound(2e-7, 1e-5),
    trunk: rel(1e-2),
    tower: rel(2e-2),
    logits: bound(2e-2, 0.2),
    decode: rel(1e-2),
    route_tie: 1e-3,
    frame_tie: 1e-2,
    frames_required: usize::MAX,
};

/// L1.5 on CUDA: the CUDA form (BF16 weights, activations, residual and
/// KV). BF16's 8-bit rounding is four times F16's, and the residual rounds
/// as well, so each stage takes four times [`SLICE_METAL`]'s bound.
/// Uncalibrated: the first run on a GPU sets them.
const SLICE_CUDA: Tolerances = Tolerances {
    embedding: bound(2e-7, 1e-5),
    trunk: rel(4e-2),
    tower: rel(8e-2),
    logits: bound(8e-2, 0.8),
    decode: rel(4e-2),
    route_tie: 5e-3,
    frame_tie: 5e-2,
    frames_required: usize::MAX,
};

/// L2 `cpu`: izwi's F32 load of the whole model (about 140 GiB) against
/// upstream's F32 W8A16 run. Forty layers at width 2048 grow F32 noise to
/// about 1e-5, and router near ties flip a few rows of the deep layers (a
/// flipped row is ~1e-3 off), so traces get 1e-3. Logits span about ±20:
/// 2e-2 absolute on the top 16. 64 decode steps: 3e-3. Frames must replay 64
/// deep unless a recorded near tie comes first (§10.6).
const L2_CPU: Tolerances = Tolerances {
    embedding: bound(1e-6, 1e-5),
    trunk: rel(1e-3),
    tower: rel(1e-3),
    logits: bound(1e-3, 2e-2),
    decode: rel(3e-3),
    route_tie: 1e-5,
    frame_tie: 1e-4,
    frames_required: REFERENCE_FRAMES,
};

/// L2 `cpu-q8_0`: the production CPU form (Q8_0 projections and experts,
/// about 40 GiB, the only one a 64 GiB box holds). Q8_0's ~0.4% per weight
/// puts each projection a few tenths of a percent off and the residual a few
/// percent off by layer 39: 5e-2 on traces, 0.5 absolute on logits, 0.1 on
/// decode values; router margins move ~1e-3, so ties widen to 5e-3. Races
/// with gaps of a percent flip under that noise, so a frame divergence is
/// reported, never required. Uncalibrated: the first run's worst values set
/// them.
const L2_CPU_Q8_0: Tolerances = Tolerances {
    embedding: bound(1e-6, 1e-5),
    trunk: rel(5e-2),
    tower: rel(5e-2),
    logits: bound(5e-2, 0.5),
    decode: rel(1e-1),
    route_tie: 5e-3,
    frame_tie: 1e-4,
    frames_required: 0,
};

/// The codec on recorded codes: F32 on both sides, its convolutions and
/// windowed transformer summed in other orders; samples lie in [-1, 1].
const PCM: Bound = bound(1e-3, 1e-2);

/// A RED variant must miss by this many times its stage's bound.
const RED_FACTOR: f64 = 100.0;
/// [`RED_FACTOR`] on a 16-bit device form, whose bounds are a hundred times
/// the F32 ones or more: a RED miss of a convention is of order one, which
/// must still be five times the bound.
const RED_FACTOR_DEVICE: f64 = 5.0;

/// Frames each L2 generation replays (§10.4).
const REFERENCE_FRAMES: usize = 64;

/// `measured <= limit`, a NaN failing.
fn within(measured: f64, limit: f64) -> bool {
    measured <= limit
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// Verdicts collected so one run reports every stage.
#[derive(Default)]
struct Report {
    failures: Vec<String>,
    notes: Vec<String>,
    /// Per stage, the worst measured value and its bound.
    worst: BTreeMap<String, (f64, f64)>,
}

impl Report {
    fn fail(&mut self, message: String) {
        self.failures.push(message);
    }

    fn note(&mut self, message: String) {
        self.notes.push(message);
    }

    fn check(&mut self, ok: bool, message: impl FnOnce() -> String) {
        if !ok {
            self.failures.push(message());
        }
    }

    /// `measured` within `limit`; `waived` (past a near tie) reports a miss
    /// as a note and leaves the stage's worst value alone.
    fn bound(&mut self, stage: &str, label: &str, measured: f64, limit: f64, waived: bool) {
        if !waived {
            let worst = self.worst.entry(stage.to_string()).or_insert((0.0, limit));
            if !within(measured, worst.0) {
                *worst = (measured, limit);
            }
        }
        if !within(measured, limit) {
            let message = format!("{label}: {measured:.3e} over {limit:.1e}");
            if waived {
                self.notes
                    .push(format!("{message} (waived: past a near tie)"));
            } else {
                self.failures.push(message);
            }
        }
    }

    fn finish(self, what: &str) {
        for note in &self.notes {
            eprintln!("{what}: {note}");
        }
        for (stage, (worst, limit)) in &self.worst {
            eprintln!("{what}: worst {stage} {worst:.3e} (bound {limit:.1e})");
        }
        assert!(
            self.failures.is_empty(),
            "{what}: {} failures:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fish_drama3")
}

fn tiny_dir() -> PathBuf {
    fixture_dir().join("tiny")
}

/// The directory `name` holds, or a failure saying how to make it.
fn env_dir(name: &str, how: &str) -> PathBuf {
    let dir = std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} is unset: {how}"));
    assert!(
        dir.is_dir(),
        "{name}={} is not a directory: {how}",
        dir.display()
    );
    dir
}

/// A generator's sidecar safetensors file (F32 tensors).
struct Sidecar {
    path: PathBuf,
    mmap: Mmap,
    tensors: BTreeMap<String, (Dtype, Range<usize>)>,
}

impl Sidecar {
    fn open(path: &Path, how: &str) -> Self {
        let file = std::fs::File::open(path)
            .unwrap_or_else(|error| panic!("{} is missing ({error}): {how}", path.display()));
        // SAFETY: read-only, and nothing writes a fixture while a test runs.
        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let (header, metadata) = safetensors::SafeTensors::read_metadata(&mmap)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let start = 8 + header;
        let tensors = metadata
            .tensors()
            .into_iter()
            .map(|(name, info)| {
                let (from, to) = info.data_offsets;
                (name, (info.dtype, start + from..start + to))
            })
            .collect();
        Self {
            path: path.to_path_buf(),
            mmap,
            tensors,
        }
    }

    /// The F32 tensor the JSON string `name` names.
    fn f32(&self, name: &Value) -> Vec<f32> {
        let name = name
            .as_str()
            .unwrap_or_else(|| panic!("{name} does not name a tensor"));
        let Some((dtype, range)) = self.tensors.get(name) else {
            panic!("{} holds no tensor {name}", self.path.display())
        };
        assert_eq!(
            *dtype,
            Dtype::F32,
            "{}: {name} is {dtype:?}",
            self.path.display()
        );
        self.mmap[range.clone()]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
}

/// A generator's JSON and its sidecar.
struct Fixture {
    json: Value,
    sidecar: Sidecar,
}

/// `dir/json_name`, a `schema` version 1 file, and the sidecar it names,
/// whose sha256 must be the recorded one. A missing file fails with `how`.
fn open_fixture(dir: &Path, json_name: &str, schema: &str, how: &str) -> Fixture {
    let path = dir.join(json_name);
    let raw = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("{} is missing ({error}): {how}", path.display()));
    let json: Value =
        serde_json::from_slice(&raw).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(
        json["schema"].as_str(),
        Some(schema),
        "{} is not a {schema} file",
        path.display()
    );
    assert_eq!(
        json["schema_version"].as_u64(),
        Some(1),
        "{}: this harness reads {schema} version 1",
        path.display()
    );
    let file = text(&json["tensors_file"]);
    let sidecar = Sidecar::open(&dir.join(file), how);
    let digest = hex::encode(Sha256::digest(&sidecar.mmap[..]));
    assert_eq!(
        Some(digest.as_str()),
        json["tensors_sha256"].as_str(),
        "{file} is not the sidecar {json_name} was written with"
    );
    Fixture { json, sidecar }
}

fn int(value: &Value) -> usize {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("{value} is not a count")) as usize
}

fn float(value: &Value) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("{value} is not a number"))
}

fn text(value: &Value) -> &str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("{value} is not a string"))
}

fn list(value: &Value) -> &[Value] {
    value
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_else(|| panic!("{value} is not a list"))
}

fn ids(value: &Value) -> Vec<u32> {
    list(value).iter().map(|v| int(v) as u32).collect()
}

fn counts(value: &Value) -> Vec<usize> {
    list(value).iter().map(int).collect()
}

fn table(value: &Value) -> Vec<Vec<u32>> {
    list(value).iter().map(ids).collect()
}

/// `[[start, end]]` row ranges.
fn spans(value: &Value) -> Vec<(usize, usize)> {
    list(value)
        .iter()
        .map(|span| {
            let span = counts(span);
            (span[0], span[1])
        })
        .collect()
}

/// Rows `[[main, c0, …]]` as a prompt.
fn prompt_of(rows: &Value) -> FishDrama3Prompt {
    let rows = table(rows);
    let width = rows.first().map_or(0, Vec::len);
    FishDrama3Prompt {
        values: (0..width)
            .map(|k| rows.iter().map(|row| row[k]).collect())
            .collect(),
        rows: rows.len(),
    }
}

fn rows_of(prompt: &FishDrama3Prompt) -> Vec<Vec<u32>> {
    (0..prompt.rows).filter_map(|t| prompt.row(t)).collect()
}

/// Frames `[[main, c0, …, c9]]` as codes `[codebooks][frames]`.
fn codes_of_frames(frames: &[Vec<u32>]) -> FishDrama3Codes {
    let codebooks = frames.first().map_or(0, |frame| frame.len() - 1);
    FishDrama3Codes::new(
        (0..codebooks)
            .map(|k| frames.iter().map(|frame| frame[1 + k]).collect())
            .collect(),
    )
}

fn references_of(value: &Value) -> Vec<FishDrama3Reference> {
    list(value)
        .iter()
        .map(|reference| FishDrama3Reference {
            transcript: text(&reference["transcript"]).to_string(),
            codes: FishDrama3Codes::new(table(&reference["codes"])),
        })
        .collect()
}

fn turns_of(value: &Value) -> Vec<FishDrama3HistoryTurn> {
    list(value)
        .iter()
        .map(|turn| FishDrama3HistoryTurn {
            text: text(&turn["text"]).to_string(),
            codes: FishDrama3Codes::new(table(&turn["codes"])),
        })
        .collect()
}

fn kind_name(kind: FishDrama3LayerKind) -> &'static str {
    match kind {
        FishDrama3LayerKind::LinearAttention => "linear_attention",
        FishDrama3LayerKind::FullAttention => "full_attention",
    }
}

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// `tensor_digest` (`fish_drama3_fixture_common.py`): sha256 over the
/// tensors in name order, each `"{name}\0{dtype}\0{shape joined by ','}\0{byte
/// length}\0"` then its bytes.
fn tensor_digest(checkpoint: &FishDrama3Checkpoint) -> String {
    let mut digest = Sha256::new();
    for name in checkpoint.tensor_names() {
        let (dtype, shape) = checkpoint.info(name).unwrap();
        let dtype = match dtype {
            Dtype::BF16 => "BF16",
            Dtype::F32 => "F32",
            Dtype::F8_E4M3 => "F8_E4M3",
            other => panic!("{name} is {other:?}, which the census never holds"),
        };
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let bytes = checkpoint.bytes(name).unwrap();
        digest.update(format!("{name}\0{dtype}\0{shape}\0{}\0", bytes.len()).as_bytes());
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}

/// No audio: the tiny fixture has no codec weights and the LM-only L2 test
/// needs none. It keeps the frame contract the load checks.
struct NoCodec;

fn no_codec() -> Error {
    Error::InferenceError("the parity harness loads no codec".into())
}

impl FishDrama3FrameCodec for NoCodec {
    fn sample_rate(&self) -> u32 {
        FishDrama3DacConfig::current().sample_rate
    }

    fn samples_per_frame(&self) -> Result<usize> {
        FishDrama3DacConfig::current().samples_per_frame()
    }

    fn codebooks(&self) -> usize {
        FishDrama3DacConfig::current().num_codebooks()
    }

    fn stream_history_bound_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    fn stream_frames(
        &self,
        _state: &mut FishDrama3DacStreamState,
        _codebooks: &[Vec<u32>],
        _check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<f32>> {
        Err(no_codec())
    }

    fn stream_frames_batch(
        &self,
        _states: &mut [FishDrama3DacStreamState],
        _codebooks: &[Vec<Vec<u32>>],
        _check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        Err(no_codec())
    }

    fn encode_reference(
        &self,
        _samples: &[f32],
        _sample_rate: u32,
        _check: FishDrama3CancelCheck<'_>,
    ) -> Result<FishDrama3Codes> {
        Err(no_codec())
    }

    fn reference_frames(&self, samples: usize, sample_rate: u32) -> Result<usize> {
        FishDrama3DacConfig::current().reference_frame_count(samples, sample_rate)
    }
}

fn model_options(
    precision: FishDrama3CpuPrecision,
    prefill_span_rows: usize,
    config_policy: FishDrama3ConfigPolicy,
) -> FishDrama3ModelOptions {
    FishDrama3ModelOptions {
        trunk: FishDrama3TrunkOptions {
            precision,
            ..Default::default()
        },
        tower: FishDrama3TowerOptions {
            precision,
            ..Default::default()
        },
        prefill_span_rows,
        config_policy,
        activations: None,
        cpu_kv_dtype: DType::F32,
    }
}

/// Upstream prefills in 128-row chunks; the same grid cell lines izwi's
/// DeltaNet chunks up with upstream's.
fn prefill_cell(json: &Value) -> usize {
    json["reference_semantics"]["prefill_chunk_rows"]
        .as_u64()
        .map_or(128, |rows| rows as usize)
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

fn compare_values(
    report: &mut Report,
    stage: &str,
    label: &str,
    ours: &[f32],
    upstream: &[f32],
    bound: Bound,
    waived: bool,
) {
    if ours.len() != upstream.len() {
        let message = format!(
            "{label}: {} values, upstream {}",
            ours.len(),
            upstream.len()
        );
        if waived {
            report.note(message);
        } else {
            report.fail(message);
        }
        return;
    }
    report.bound(
        stage,
        &format!("{label} rel L2"),
        relative_l2(ours, upstream),
        bound.rel_l2,
        waived,
    );
    if bound.max_abs.is_finite() {
        report.bound(
            &format!("{stage} max abs"),
            &format!("{label} max abs"),
            f64::from(max_abs_diff(ours, upstream)),
            bound.max_abs,
            waived,
        );
    }
}

/// `checksum` (`fish_drama3_fixture_common.py`), accumulated in F64.
struct Checksum {
    count: usize,
    sum: f64,
    abs_sum: f64,
    sum_sq: f64,
    max_abs: f64,
}

impl Checksum {
    fn of(values: &[f32]) -> Self {
        values.iter().fold(
            Self {
                count: 0,
                sum: 0.0,
                abs_sum: 0.0,
                sum_sq: 0.0,
                max_abs: 0.0,
            },
            |c, &v| {
                let v = f64::from(v);
                Self {
                    count: c.count + 1,
                    sum: c.sum + v,
                    abs_sum: c.abs_sum + v.abs(),
                    sum_sq: c.sum_sq + v * v,
                    max_abs: if v.is_nan() || v.abs() > c.max_abs {
                        v.abs()
                    } else {
                        c.max_abs
                    },
                }
            },
        )
    }

    fn rms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum_sq / self.count as f64).sqrt()
        }
    }
}

/// Our checksum against a recorded one: the count exactly; the RMS, the
/// absolute sum and the largest magnitude relative to themselves; the signed
/// sum relative to the absolute sum (a sum near zero has no scale of its
/// own). The largest magnitude is one element: ten times the bound.
fn compare_checksum(
    report: &mut Report,
    stage: &str,
    label: &str,
    expected: &Value,
    ours: &[f32],
    rel: f64,
) {
    let c = Checksum::of(ours);
    report.check(int(&expected["count"]) == c.count, || {
        format!(
            "{label}: checksum over {} values, upstream {}",
            c.count, expected["count"]
        )
    });
    let off = |a: f64, b: f64, scale: f64| (a - b).abs() / scale.abs().max(f64::MIN_POSITIVE);
    let (rms, abs_sum, max_abs) = (
        float(&expected["rms"]),
        float(&expected["abs_sum"]),
        float(&expected["max_abs"]),
    );
    for (what, measured) in [
        ("rms", off(c.rms(), rms, rms)),
        ("abs_sum", off(c.abs_sum, abs_sum, abs_sum)),
        ("sum", off(c.sum, float(&expected["sum"]), abs_sum)),
    ] {
        report.bound(
            stage,
            &format!("{label} checksum {what}"),
            measured,
            rel,
            false,
        );
    }
    report.bound(
        &format!("{stage} checksum max_abs"),
        &format!("{label} checksum max_abs"),
        off(c.max_abs, max_abs, max_abs),
        10.0 * rel,
        false,
    );
}

fn gather(values: &[f32], width: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|&r| values[r * width..(r + 1) * width].iter().copied())
        .collect()
}

fn row_rms(values: &[f32], width: usize) -> Vec<f32> {
    values
        .chunks_exact(width)
        .map(|row| {
            (row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / width as f64).sqrt() as f32
        })
        .collect()
}

/// A trace entry against ours (`[rows, width]` row-major): the shape, the
/// stored rows, every row's RMS and the checksum.
#[allow(clippy::too_many_arguments)]
fn compare_trace(
    report: &mut Report,
    stage: &str,
    label: &str,
    entry: &Value,
    sidecar: &Sidecar,
    ours: &[f32],
    width: usize,
    bound: Bound,
) {
    let rows = ours.len() / width;
    let shape = counts(&entry["shape"]);
    if shape != [rows, width] {
        report.fail(format!(
            "{label}: izwi holds [{rows}, {width}], upstream {shape:?}"
        ));
        return;
    }
    let picks = counts(&entry["rows"]);
    compare_values(
        report,
        stage,
        &format!("{label} rows {picks:?}"),
        &gather(ours, width, &picks),
        &sidecar.f32(&entry["values"]),
        bound,
        false,
    );
    report.bound(
        stage,
        &format!("{label} row RMS"),
        relative_l2(&row_rms(ours, width), &sidecar.f32(&entry["row_rms"])),
        bound.rel_l2,
        false,
    );
    compare_checksum(report, stage, label, &entry["checksum"], ours, bound.rel_l2);
}

/// A prompt izwi built against upstream's rows.
fn compare_prompt(report: &mut Report, label: &str, rows: &Value, built: Result<FishDrama3Prompt>) {
    let built = match built {
        Ok(prompt) => prompt,
        Err(error) => {
            report.fail(format!("{label}: izwi refuses the prompt: {error}"));
            return;
        }
    };
    let (ours, theirs) = (rows_of(&built), table(rows));
    let first = ours.iter().zip(&theirs).position(|(a, b)| a != b);
    report.check(ours == theirs, || {
        format!(
            "{label}: {} prompt rows, upstream {}; first differing row {first:?}",
            ours.len(),
            theirs.len()
        )
    });
}

/// Upstream's tower-layer labels follow §3.5: shared layers inject c0;
/// residual layer j injects c_j and its head predicts c_{j+1}.
fn check_tower_kind(report: &mut Report, label: &str, layer: &Value, shared: usize) {
    let index = int(&layer["index"]);
    let (kind, injects, predicts) = if index < shared {
        ("shared", "c0".to_string(), Value::Null)
    } else {
        let j = index - shared;
        (
            "residual",
            format!("c{j}"),
            Value::from(format!("c{}", j + 1)),
        )
    };
    report.check(
        layer["kind"] == kind
            && layer["injects"] == injects.as_str()
            && layer["head_predicts"] == predicts,
        || format!("{label}: tower layer {index} is labelled {layer:?}"),
    );
}

/// `p[k-1] - p[k]` of the F32 softmax over `logits`, as `route_entry`
/// takes it; 1 when every expert is kept.
fn route_margin(logits: &[f32], top_k: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let total = exps.iter().sum::<f32>();
    let mut p = exps.iter().map(|e| e / total).collect::<Vec<_>>();
    p.sort_by(|a, b| b.total_cmp(a));
    if p.len() > top_k {
        p[top_k - 1] - p[top_k]
    } else {
        1.0
    }
}

/// Rows of one layer's router logits `[n, experts]` with a near tie.
fn near_tie_rows(logits: &[f32], experts: usize, top_k: usize, tie: f32) -> usize {
    logits
        .chunks_exact(experts)
        .filter(|row| route_margin(row, top_k) < tie)
        .count()
}

fn bf16_ulps(a: f32, b: f32) -> u16 {
    bf16::from_f32(a)
        .to_bits()
        .abs_diff(bf16::from_f32(b).to_bits())
}

/// A recorded `route_entry` against our routing of its rows (`logits`
/// `[n, experts]` covers every row): the expert sets agree unless either
/// side's margin is under the near-tie threshold (the row is then counted),
/// and the BF16 weights within one ulp. Returns the rows whose sets differ
/// at a near tie.
fn compare_routes(
    report: &mut Report,
    label: &str,
    entry: &Value,
    logits: &[f32],
    experts: usize,
    top_k: usize,
    tie: f32,
) -> Vec<usize> {
    let tie = tie.max(entry["near_tie_threshold"].as_f64().unwrap_or(0.0) as f32);
    let (sets, weights) = (list(&entry["experts"]), list(&entry["weights"]));
    let margins = list(&entry["margins"]);
    let (mut equal, mut flipped) = (0, Vec::new());
    for (i, &row) in counts(&entry["rows"]).iter().enumerate() {
        let row_logits = &logits[row * experts..(row + 1) * experts];
        let ours = route(row_logits, top_k);
        let (our_margin, their_margin) =
            (route_margin(row_logits, top_k), float(&margins[i]) as f32);
        let theirs = counts(&sets[i]);
        let (mut a, mut b) = (ours.experts.clone(), theirs.clone());
        a.sort_unstable();
        b.sort_unstable();
        if a != b {
            if our_margin < tie || their_margin < tie {
                flipped.push(row);
            } else {
                report.fail(format!(
                    "{label} row {row}: experts {a:?}, upstream {b:?} (margins {our_margin:.1e}, upstream {their_margin:.1e})"
                ));
            }
            continue;
        }
        equal += 1;
        let their_weights = list(&weights[i]);
        let worst = theirs
            .iter()
            .zip(their_weights)
            .map(|(e, w)| {
                let at = ours.experts.iter().position(|x| x == e).unwrap();
                bf16_ulps(ours.weights[at], float(w) as f32)
            })
            .max()
            .unwrap_or(0);
        report.check(worst <= 1, || {
            format!("{label} row {row}: router weights {worst} BF16 ulps off")
        });
    }
    report.note(format!(
        "{label}: {equal} rows route alike, {} near-tie rows differ; near ties over every row: \
         ours {}, upstream {}",
        flipped.len(),
        near_tie_rows(logits, experts, top_k, tie),
        entry["near_ties"]
    ));
    flipped
}

/// A RED variant must miss upstream by more than `factor` times its
/// stage's bound.
fn red(report: &mut Report, name: &str, ours: &[f32], upstream: &[f32], bound: f64, factor: f64) {
    let error = relative_l2(ours, upstream);
    let floor = factor * bound;
    report.note(format!(
        "RED {name}: rel L2 {error:.3e} (must exceed {floor:.1e})"
    ));
    report.check(error > floor, || {
        format!(
            "RED {name}: misses upstream by only {error:.3e}, under {floor:.1e} ({factor} times \
             its bound): the fixture cannot tell this convention apart"
        )
    });
}

// ---------------------------------------------------------------------------
// Prefill and replay outside the model loop
// ---------------------------------------------------------------------------

/// A one-row F32 paged cache.
fn cache(
    device: &Device,
    layers: &[u32],
    kv_heads: usize,
    head_dim: usize,
    positions: usize,
) -> PhysicalPagedKvCache {
    cache_in(device, DType::F32, layers, kv_heads, head_dim, positions)
}

/// A one-row paged cache storing `dtype`.
fn cache_in(
    device: &Device,
    dtype: DType,
    layers: &[u32],
    kv_heads: usize,
    head_dim: usize,
    positions: usize,
) -> PhysicalPagedKvCache {
    test_physical_caches_for_model_layers_on(
        device, dtype, layers, kv_heads, head_dim, positions, 1,
    )
    .remove(0)
}

fn append_layers(all: &mut Vec<Vec<f32>>, part: Vec<Vec<f32>>) {
    if all.is_empty() {
        *all = part;
    } else {
        for (all, part) in all.iter_mut().zip(part) {
            all.extend(part);
        }
    }
}

fn append_trace(all: &mut LayerTrace, part: LayerTrace) {
    append_layers(&mut all.outputs, part.outputs);
    append_layers(&mut all.mixers, part.mixers);
    append_layers(&mut all.moes, part.moes);
    append_layers(&mut all.router_logits, part.router_logits);
}

/// One request's state outside the model loop, so every step can be
/// watched.
struct Session<'m> {
    model: &'m FishDrama3Model,
    state: FishDrama3TrunkState,
    trunk_cache: PhysicalPagedKvCache,
    tower_cache: PhysicalPagedKvCache,
    /// `h_t`, `[1, hidden]`.
    hidden: Tensor,
    /// `F_t`, `[main, c0, …]`.
    row: Vec<u32>,
}

impl Session<'_> {
    /// The compact head's logits at `h_t`.
    fn logits(&self) -> Vec<f32> {
        host(&self.model.head().logits(&self.hidden).unwrap())
    }

    /// The tower at `t` under `label`'s codes: the heads' logits
    /// `[heads, V]` and every layer's output `[layers, dim]`.
    fn forced_tower(&mut self, label: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let model = self.model;
        let mut rows = [FishDrama3TowerForcedRow {
            cache: &mut self.tower_cache,
            input: &self.row,
            label,
        }];
        model
            .tower()
            .forward_step_forced_traced(model.embedding(), &self.hidden, &mut rows)
            .unwrap()
    }

    /// The tower at `t` sampling its heads with `sampler`: `F_{t+1}`.
    fn sampled_tower(&mut self, main: u32, sampler: &mut FishDrama3RowSampler) -> Vec<u32> {
        let model = self.model;
        let mut rows = [FishDrama3TowerStepRow {
            cache: &mut self.tower_cache,
            input: &self.row,
            main,
            sampler,
        }];
        model
            .tower()
            .forward_step(model.embedding(), &self.hidden, &mut rows)
            .unwrap()
    }

    /// The trunk at `t + 1` on `frame`.
    fn advance(&mut self, frame: Vec<u32>) {
        let model = self.model;
        let embeds = model
            .embedding()
            .embed_tensor(&frame, model.device())
            .unwrap();
        let mut rows = [FishDrama3StepRow {
            state: &mut self.state,
            cache: &mut self.trunk_cache,
        }];
        self.hidden = model.trunk().forward_step(&embeds, &mut rows).unwrap();
        self.row = frame;
    }
}

/// What a traced prefill computed.
struct Prefill {
    /// The trunk's input rows `[P, hidden]`.
    input: Vec<f32>,
    /// Every trunk layer over every row.
    trunk: LayerTrace,
    /// The final-norm hidden `[P, hidden]`.
    final_norm: Vec<f32>,
    /// Every tower layer's output over the teacher-forced rows `[P - 1, dim]`.
    tower: Vec<Vec<f32>>,
}

/// `prompt` through the model's prefill grid as `prefill_span` runs it (the
/// trunk over a cell, then the tower teacher-forced over its rows short of
/// the last prompt row), traced, with room for `frames` more positions.
fn prefill<'m>(
    model: &'m FishDrama3Model,
    prompt: &FishDrama3Prompt,
    frames: usize,
) -> (Prefill, Session<'m>) {
    let device = model.device();
    let geometry = model.state_geometry();
    let positions = prompt.rows + frames + 2;
    let domain_cache =
        |d: &FishDrama3PagedDomain| cache(device, &d.layers, d.kv_heads, d.head_dim, positions);
    let mut state = model.trunk().new_state().unwrap();
    let mut trunk_cache = domain_cache(&geometry.trunk);
    let mut tower_cache = domain_cache(&geometry.tower);
    let mut out = Prefill {
        input: Vec::new(),
        trunk: LayerTrace::default(),
        final_norm: Vec::new(),
        tower: Vec::new(),
    };
    let (rows, cell) = (prompt.rows, model.prefill_span_rows());
    let mut last = None;
    let mut a = 0;
    while a < rows {
        let b = ((a / cell + 1) * cell).min(rows);
        let span = prompt_rows(prompt, a..b).unwrap();
        let embeds = model.embedding().embed_tensor(&span, device).unwrap();
        out.input.extend(host(&embeds));
        let (hidden, trace) = model
            .trunk()
            .forward_span_traced(&embeds, a, &mut state, &mut trunk_cache)
            .unwrap();
        append_trace(&mut out.trunk, trace);
        out.final_norm.extend(host(&hidden));
        let forced = FishDrama3Tower::teacher_forced_rows(a..b, rows);
        if !forced.is_empty() {
            let labels = prompt_rows(prompt, forced.start..forced.end + 1).unwrap();
            let trace = model
                .tower()
                .forward_span_traced(
                    model.embedding(),
                    &hidden.narrow(0, 0, forced.len()).unwrap(),
                    &labels,
                    forced.start,
                    &mut tower_cache,
                )
                .unwrap();
            append_layers(&mut out.tower, trace);
        }
        last = Some(
            hidden
                .narrow(0, b - a - 1, 1)
                .unwrap()
                .contiguous()
                .unwrap(),
        );
        a = b;
    }
    let session = Session {
        model,
        state,
        trunk_cache,
        tower_cache,
        hidden: last.expect("a prompt has rows"),
        row: prompt.row(rows - 1).unwrap(),
    };
    (out, session)
}

/// One decode step of a replay.
struct ReplayStep {
    /// The RAS window before the draw, oldest first.
    window: Vec<u32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    draw: FishDrama3MainDraw,
    /// `F_{t+1}`; `None` on eos.
    frame: Option<Vec<u32>>,
}

/// Up to `steps` decode steps as `decode_quantum` runs them, eos allowed
/// from the first frame: upstream has no first-frame mask (§7.5).
fn replay(
    session: &mut Session<'_>,
    sampler: &mut FishDrama3RowSampler,
    steps: usize,
) -> Vec<ReplayStep> {
    let eos = session.model.config().ids.eos;
    let mut out = Vec::with_capacity(steps);
    while out.len() < steps {
        let (hidden, logits, window) = (host(&session.hidden), session.logits(), sampler.window());
        let draw = sampler.sample_main(&logits, true).unwrap();
        let frame = (draw.token != eos).then(|| session.sampled_tower(draw.token, sampler));
        out.push(ReplayStep {
            window,
            hidden,
            logits,
            draw,
            frame: frame.clone(),
        });
        match frame {
            Some(frame) if out.len() < steps => session.advance(frame),
            _ => break,
        }
    }
    out
}

/// A rollout's `params`, the main token and the heads alike
/// (`from_request(t, p, Some(k))`).
fn sampling_of(params: &Value) -> FishDrama3SamplingParams {
    FishDrama3SamplingParams::from_request(
        Some(float(&params["temperature"]) as f32),
        Some(float(&params["top_p"]) as f32),
        Some(int(&params["top_k"])),
        0,
    )
    .unwrap()
}

/// A rollout's recorded exponentials, checked against their layout: `2C`
/// per step, then `9V` per frame made.
fn noise_of(
    report: &mut Report,
    label: &str,
    rollout: &Value,
    sidecar: &Sidecar,
    config: &FishDrama3Config,
) -> Vec<f32> {
    let noise = &rollout["noise"];
    let values = sidecar.f32(&noise["tensor"]);
    let (c, v) = (config.ids.semantic_codes() + 1, config.tower.codebook_vocab);
    let heads = config.tower.residual_layers();
    let steps = list(&rollout["steps"]);
    let frames = steps.iter().filter(|step| !step["frame"].is_null()).count();
    let layout = steps.len() * 2 * c + frames * heads * v;
    report.check(
        values.len() == layout
            && int(&noise["count"]) == layout
            && int(&noise["candidates"]) == c
            && int(&noise["codes"]) == v,
        || {
            format!(
                "{label}: {} recorded exponentials; {} steps and {frames} frames of C {c}, V {v} need {layout}",
                values.len(),
                steps.len()
            )
        },
    );
    report.check(rollout["eos_mask_first_frame"] == false, || {
        format!("{label}: recorded with a first-frame eos mask")
    });
    values
}

/// The first draw of a recorded step whose race or top-p gap is under `tie`.
fn near_tie(step: &Value, tie: f64) -> Option<String> {
    list(&step["draws"]).iter().find_map(|draw| {
        let kind = draw["kind"].as_str().unwrap_or("draw");
        let race = draw["race_gap"].as_f64().unwrap_or(f64::INFINITY);
        if race < tie {
            return Some(format!("{kind} race gap {race:.1e}"));
        }
        draw["top_p_gap"]
            .as_f64()
            .filter(|gap| *gap < tie)
            .map(|gap| format!("{kind} top-p gap {gap:.1e}"))
    })
}

/// How our step differs from the recorded one, if it does.
fn step_difference(step: &Value, ours: &ReplayStep) -> Option<String> {
    let window = ids(&step["window"]);
    if window != ours.window {
        return Some(format!("RAS window {:?}, upstream {window:?}", ours.window));
    }
    let main = &step["main"];
    let theirs = (
        int(&main["first"]) as u32,
        int(&main["retry"]) as u32,
        main["redrawn"].as_bool(),
        int(&main["token"]) as u32,
    );
    let draw = ours.draw;
    let mine = (draw.first, draw.retry, Some(draw.redrawn), draw.token);
    if mine != theirs {
        let gap = |i: usize| step["draws"][i]["race_gap"].as_f64();
        return Some(format!(
            "main draws (first, retry, redrawn, token) {mine:?}, upstream {theirs:?} (recorded race gaps {:?}, {:?})",
            gap(0),
            gap(1)
        ));
    }
    match (step["frame"].is_null(), &ours.frame) {
        (true, None) => None,
        (false, Some(frame)) => {
            let recorded = ids(&step["frame"]);
            if recorded.len() != frame.len() {
                return Some(format!(
                    "a frame of {} values, upstream {}",
                    frame.len(),
                    recorded.len()
                ));
            }
            // Index 1 + j holds c_j, drawn by head j (draw 1 + j).
            let k = frame.iter().zip(&recorded).position(|(a, b)| a != b)?;
            let value = match k {
                0 => "the main token".to_string(),
                k => format!("c{}", k - 1),
            };
            Some(format!(
                "{value} is {}, upstream {} (recorded race gap {:?})",
                frame[k],
                recorded[k],
                step["draws"][k]["race_gap"].as_f64()
            ))
        }
        (true, Some(_)) => Some("a frame where upstream drew eos".into()),
        (false, None) => Some("eos where upstream made a frame".into()),
    }
}

/// A replay against a recorded rollout (see the module doc). Returns
/// whether every replayed step matched.
fn compare_rollout(
    report: &mut Report,
    label: &str,
    rollout: &Value,
    sidecar: &Sidecar,
    ours: &[ReplayStep],
    tol: &Tolerances,
    hidden: usize,
) -> bool {
    let steps = list(&rollout["steps"]);
    let hiddens = sidecar.f32(&rollout["hidden"]);
    let logits = sidecar.f32(&rollout["main_logits"]);
    let candidates = logits.len() / steps.len().max(1);
    let mut first_tie: Option<(usize, String)> = None;
    let mut divergence: Option<(usize, String)> = None;
    for (s, (step, mine)) in steps.iter().zip(ours).enumerate() {
        if first_tie.is_none() {
            first_tie = near_tie(step, tol.frame_tie).map(|why| (s, why));
        }
        let waived = first_tie.is_some();
        compare_values(
            report,
            "decode",
            &format!("{label} step {s} hidden"),
            &mine.hidden,
            &hiddens[s * hidden..(s + 1) * hidden],
            tol.decode,
            waived,
        );
        compare_values(
            report,
            "decode",
            &format!("{label} step {s} main logits"),
            &mine.logits,
            &logits[s * candidates..(s + 1) * candidates],
            tol.decode,
            waived,
        );
        if let Some(why) = step_difference(step, mine) {
            divergence = Some((s, why));
            break;
        }
    }
    let tie_at = first_tie.as_ref().map_or(usize::MAX, |(s, _)| *s);
    let tie_text = first_tie
        .as_ref()
        .map_or("none".to_string(), |(s, why)| format!("frame {s} ({why})"));
    let required = tol.frames_required.min(steps.len());
    match &divergence {
        Some((s, why)) if *s < tie_at && *s < required => report.fail(format!(
            "{label}: frame {s} diverges before any recorded near tie: {why}"
        )),
        Some((s, why)) => report.note(format!(
            "{label}: frame {s} diverges ({why}); first recorded near tie: {tie_text}"
        )),
        None if ours.len() < required => report.fail(format!(
            "{label}: replayed {} of the {required} frames required",
            ours.len()
        )),
        None => {}
    }
    let matched = divergence
        .as_ref()
        .map_or(ours.len().min(steps.len()), |(s, _)| *s);
    report.note(format!(
        "{label}: {matched} of {} recorded steps replayed exactly ({} replayed); first near tie: \
         {tie_text}; upstream made {} redraws and stopped by {}",
        steps.len(),
        ours.len(),
        rollout["redraws"],
        rollout["finish_reason"]
    ));
    divergence.is_none()
}

// ---------------------------------------------------------------------------
// L1: the tiny upstream fixture
// ---------------------------------------------------------------------------

fn tiny_fixture() -> Fixture {
    let dir = tiny_dir();
    let fixture = open_fixture(&dir, "expected.json", TINY_SCHEMA, TINY_HOW);
    let tokenizer = dir.join("tokenizer.json");
    assert!(
        tokenizer.is_file(),
        "{} is missing: {TINY_HOW}",
        tokenizer.display()
    );
    fixture
}

/// `SyntheticCheckpoint::write(seed)` must be what the generator's port of
/// it wrote.
fn assert_port(synthetic: &SyntheticCheckpoint, seed: u64, recorded: &str) {
    let ours = tensor_digest(&synthetic.open());
    assert_eq!(
        ours, recorded,
        "the Python port of the weight generator diverged: SyntheticCheckpoint::write({seed}) \
         hashes to {ours}, but generate_fish_drama3_tiny.py's port of \
         synthetic.rs::trained_like_tensors wrote tensors hashing to {recorded}. Fix the port and \
         regenerate, or commit the generator's checkpoint into tests/fixtures/fish_drama3/tiny/"
    );
}

/// The checkpoint upstream ran: tiny/ itself when the generator's checkpoint
/// is committed there, else `SyntheticCheckpoint::write(seed)` with tiny/'s
/// tokenizer. Its tensors hash to the recorded digest either way.
struct TinyCheckpoint {
    dir: PathBuf,
    /// The written checkpoint, removed when dropped.
    _written: Option<TempDir>,
}

fn tiny_checkpoint(fixture: &Fixture) -> TinyCheckpoint {
    let tiny = tiny_dir();
    let recorded = &fixture.json["checkpoint"];
    let digest = text(&recorded["tensor_digest"]);
    if tiny.join(INDEX_FILE).is_file() {
        let found = tensor_digest(&FishDrama3Checkpoint::open(&tiny).unwrap());
        assert_eq!(
            found, digest,
            "tests/fixtures/fish_drama3/tiny/ holds a checkpoint hashing to {found}, not the one \
             expected.json was made from ({digest}): regenerate the fixture"
        );
        return TinyCheckpoint {
            dir: tiny,
            _written: None,
        };
    }
    let seed = int(&recorded["seed"]) as u64;
    let synthetic = SyntheticCheckpoint::write(seed);
    assert_port(&synthetic, seed, digest);
    std::fs::copy(
        tiny.join("tokenizer.json"),
        synthetic.path().join("tokenizer.json"),
    )
    .unwrap();
    TinyCheckpoint {
        dir: synthetic.path().to_path_buf(),
        _written: Some(synthetic.dir),
    }
}

/// The tiny model, F32 like upstream's run.
fn tiny_model(fixture: &Fixture, device: &Device) -> (TinyCheckpoint, FishDrama3Model) {
    let checkpoint = tiny_checkpoint(fixture);
    let options = model_options(
        FishDrama3CpuPrecision::F32,
        prefill_cell(&fixture.json),
        FishDrama3ConfigPolicy::Synthetic,
    );
    let model =
        FishDrama3Model::load_with_codec(&checkpoint.dir, device, options, Arc::new(NoCodec))
            .unwrap();
    (checkpoint, model)
}

/// The digest of `SyntheticCheckpoint::write(seed)` is the one the
/// generator recorded: its Python port of `trained_like_tensors` writes the
/// same tensors, so the Rust-built checkpoint may stand in for a committed
/// one. The geometry (`tiny/config.json`) and a committed checkpoint are
/// checked when present.
#[test]
#[ignore = "needs tests/fixtures/fish_drama3/tiny/ from generate_fish_drama3_tiny.py"]
fn tiny_checkpoint_is_the_rust_synthetic_checkpoint() {
    let fixture = tiny_fixture();
    let recorded = &fixture.json["checkpoint"];
    let (seed, digest) = (
        int(&recorded["seed"]) as u64,
        text(&recorded["tensor_digest"]),
    );
    let synthetic = SyntheticCheckpoint::write(seed);
    assert_port(&synthetic, seed, digest);
    assert_eq!(synthetic.open().tensor_count(), int(&recorded["tensors"]));
    let tiny = tiny_dir();
    if tiny.join("config.json").is_file() {
        let config =
            FishDrama3Config::load_with_policy(&tiny, FishDrama3ConfigPolicy::Synthetic).unwrap();
        assert_eq!(
            config,
            trunk_geometry(),
            "tiny/config.json is not synthetic.rs::trunk_geometry_json()"
        );
    }
    if tiny.join(INDEX_FILE).is_file() {
        let committed = tensor_digest(&FishDrama3Checkpoint::open(&tiny).unwrap());
        assert_eq!(committed, digest, "the committed tiny checkpoint differs");
    }
}

/// The four prompt shapes (plain, one reference, two references with two
/// speakers, history) build upstream's rows exactly with the tiny
/// tokenizer.
#[test]
#[ignore = "needs tests/fixtures/fish_drama3/tiny/ from generate_fish_drama3_tiny.py"]
fn tiny_prompts_match_upstream_rows() {
    let fixture = tiny_fixture();
    let config = trunk_geometry();
    let tokenizer = FishDrama3Tokenizer::load(&tiny_dir(), &config).unwrap();
    let mut report = Report::default();
    let recorded = &fixture.json["ids"];
    let ids = config.ids;
    for (name, ours) in [
        ("pad", ids.pad),
        ("im_start", tokenizer.im_start()),
        ("eos", ids.eos),
        ("audio_pad", ids.audio_pad),
        ("voice", ids.voice),
        ("speaker_first", ids.speaker_first),
        ("semantic_start", ids.semantic_start),
        ("semantic_end", ids.semantic_end),
    ] {
        report.check(int(&recorded[name]) == ours as usize, || {
            format!("id {name}: izwi {ours}, upstream {}", recorded[name])
        });
    }
    let kinds = config
        .trunk
        .layers
        .iter()
        .map(|k| kind_name(*k))
        .collect::<Vec<_>>();
    let recorded_kinds = list(&fixture.json["layer_types"])
        .iter()
        .map(text)
        .collect::<Vec<_>>();
    report.check(kinds == recorded_kinds, || {
        format!("layer types {kinds:?}, upstream {recorded_kinds:?}")
    });
    for entry in list(&fixture.json["prompts"]) {
        let name = text(&entry["name"]);
        let references = references_of(&entry["references"]);
        let history = turns_of(&entry["history"]);
        let built = build_prompt(&tokenizer, &references, &history, text(&entry["text"]));
        report.check(
            built
                .as_ref()
                .is_ok_and(|prompt| prompt.rows == int(&entry["prompt_rows"])),
            || format!("{name}: prompt_rows {}", entry["prompt_rows"]),
        );
        compare_prompt(&mut report, name, &entry["rows"], built);
    }
    report.finish("L1 tiny prompts");
}

/// Every prompt's prefill (the trunk's input, each layer's residual, the
/// mixer and MoE outputs of layers 0 and 3, routes of every layer, the
/// final norm, every tower layer, the teacher-forced head logits) and two
/// forced decode steps (hidden, main logits, head logits under forced
/// codes).
fn run_tiny_tensors(device: &Device, backend: &str) {
    let fixture = tiny_fixture();
    let (_checkpoint, model) = tiny_model(&fixture, device);
    let (json, sidecar) = (&fixture.json, &fixture.sidecar);
    let config = model.config();
    let (h, d) = (config.trunk.hidden, config.tower.dim);
    let (experts, top_k) = (config.trunk.moe.experts, config.trunk.moe.top_k);
    let (shared, heads) = (config.tower.shared_layers, config.tower.residual_layers());
    let tol = tiny_tolerances(device);
    let mut report = Report::default();
    for entry in list(&json["prompts"]) {
        let label = format!("{backend} {}", text(&entry["name"]));
        let prompt = prompt_of(&entry["rows"]);
        let p = prompt.rows;
        let (prefill, mut session) = prefill(&model, &prompt, 2);
        let trunk = &entry["trunk"];
        compare_trace(
            &mut report,
            "embedding",
            &format!("{label} input"),
            &trunk["input"],
            sidecar,
            &prefill.input,
            h,
            tol.embedding,
        );
        for layer in list(&trunk["layers"]) {
            let l = int(&layer["layer"]);
            compare_trace(
                &mut report,
                "trunk",
                &format!("{label} layer {l}"),
                &layer["output"],
                sidecar,
                &prefill.trunk.outputs[l],
                h,
                tol.trunk,
            );
        }
        for component in list(&trunk["components"]) {
            let (l, part) = (int(&component["layer"]), text(&component["part"]));
            let ours = match part {
                "mixer" => &prefill.trunk.mixers[l],
                _ => &prefill.trunk.moes[l],
            };
            compare_trace(
                &mut report,
                "trunk",
                &format!("{label} layer {l} {part}"),
                &component["output"],
                sidecar,
                ours,
                h,
                tol.trunk,
            );
        }
        compare_trace(
            &mut report,
            "trunk",
            &format!("{label} final norm"),
            &trunk["final_norm"],
            sidecar,
            &prefill.final_norm,
            h,
            tol.trunk,
        );
        for routes in list(&trunk["routes"]) {
            let l = int(&routes["layer"]);
            compare_routes(
                &mut report,
                &format!("{label} layer {l} routes"),
                routes,
                &prefill.trunk.router_logits[l],
                experts,
                top_k,
                tol.route_tie,
            );
        }
        let tower = &entry["tower"];
        report.check(int(&tower["prefill_rows"]) == p - 1, || {
            format!(
                "{label}: the tower prefilled {} rows of {p}",
                tower["prefill_rows"]
            )
        });
        for layer in list(&tower["layers"]) {
            let i = int(&layer["index"]);
            check_tower_kind(&mut report, &label, layer, shared);
            compare_trace(
                &mut report,
                "tower",
                &format!("{label} tower layer {i}"),
                &layer["output"],
                sidecar,
                &prefill.tower[i],
                d,
                tol.tower,
            );
        }
        // Head j on residual layer j's teacher-forced output, `[rows, 9, V]`.
        let teacher = &tower["teacher_forced_head_logits"];
        let mut ours = Vec::new();
        for r in counts(&teacher["rows"]) {
            for j in 0..heads {
                let layer = &prefill.tower[shared + j][r * d..(r + 1) * d];
                ours.extend(
                    model
                        .tower()
                        .head_logits(j, layer, 1, FishDrama3KernelPhase::Span)
                        .unwrap(),
                );
            }
        }
        compare_values(
            &mut report,
            "logits",
            &format!("{label} teacher-forced head logits"),
            &ours,
            &sidecar.f32(&teacher["tensor"]),
            tol.logits,
            false,
        );
        // The main logits at the last prompt row over the whole vocabulary:
        // the compact head's candidates of them.
        let forced = &entry["forced"];
        let full = sidecar.f32(&forced["main_logits_full"]);
        let candidates = config
            .ids
            .compact_head_rows()
            .iter()
            .map(|&row| full[row])
            .collect::<Vec<_>>();
        compare_values(
            &mut report,
            "logits",
            &format!("{label} main logits at row {}", p - 1),
            &session.logits(),
            &candidates,
            tol.logits,
            false,
        );
        for (s, (step, frame)) in list(&forced["steps"])
            .iter()
            .zip(table(&forced["frames"]))
            .enumerate()
        {
            let at = format!("{label} forced step {s}");
            report.check(int(&step["position"]) == p - 1 + s, || {
                format!("{at}: recorded at position {}", step["position"])
            });
            compare_values(
                &mut report,
                "trunk",
                &format!("{at} hidden"),
                &host(&session.hidden),
                &sidecar.f32(&step["hidden"]),
                tol.trunk,
                false,
            );
            compare_values(
                &mut report,
                "logits",
                &format!("{at} main logits"),
                &session.logits(),
                &sidecar.f32(&step["main_logits"]),
                tol.logits,
                false,
            );
            let (head_logits, _) = session.forced_tower(&frame);
            compare_values(
                &mut report,
                "logits",
                &format!("{at} head logits"),
                &head_logits,
                &sidecar.f32(&step["head_logits"]),
                tol.logits,
                false,
            );
            session.advance(frame);
        }
    }
    report.finish(&format!("L1 tiny prefill and forced steps on {backend}"));
}

/// The model loop itself over the same prompt and noise makes the replay's
/// frames and stops for upstream's reason. CPU only: its private caches
/// are.
fn check_model_loop(
    report: &mut Report,
    label: &str,
    model: &FishDrama3Model,
    prompt: FishDrama3Prompt,
    rollout: &Value,
    mut noise: Vec<f32>,
    replayed: &[ReplayStep],
) {
    let max_frames = int(&rollout["max_new_tokens"]);
    // After a full budget the loop draws once more (a frame-limit stop
    // reads one more main step, which upstream never drew).
    let candidates = model.config().ids.semantic_codes() + 1;
    noise.extend(std::iter::repeat_n(1.0, 2 * candidates));
    let params = FishDrama3GenerationParams {
        sampling: sampling_of(&rollout["params"]),
        max_frames,
        mask_first_eos: false,
    };
    let check: FishDrama3CancelCheck<'_> = &|| Ok(());
    let mut state = model.new_private_replay(prompt, params, noise).unwrap();
    model.run_prefill(&mut state, check).unwrap();
    model.run_decode(&mut [&mut state], check).unwrap();
    let made = (0..state.frames())
        .map(|f| {
            state
                .codes()
                .iter()
                .map(|codebook| codebook[f])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let frames = replayed
        .iter()
        .filter_map(|step| step.frame.as_ref().map(|frame| frame[1..].to_vec()))
        .collect::<Vec<_>>();
    report.check(made == frames, || {
        format!(
            "{label}: the model loop made {} frames that differ from the replay's {}",
            made.len(),
            frames.len()
        )
    });
    let stop = state.stop_reason();
    let expected = match text(&rollout["finish_reason"]) {
        "eos" => stop == Some(FishDrama3StopReason::EndOfSpeech),
        // Upstream stopped at its budget without drawing; izwi's extra draw
        // may then be eos or not.
        _ => made.len() == max_frames && stop.is_some(),
    };
    report.check(expected, || {
        format!(
            "{label}: the model loop stopped by {stop:?} after {} frames; upstream by {}",
            made.len(),
            rollout["finish_reason"]
        )
    });
}

/// The five rollouts under the recorded noise, the first-frame eos mask
/// off: every RAS window, both main draws, the RAS decision, every code.
fn run_tiny_rollouts(device: &Device, backend: &str) {
    let fixture = tiny_fixture();
    let (_checkpoint, model) = tiny_model(&fixture, device);
    let (json, sidecar) = (&fixture.json, &fixture.sidecar);
    let config = model.config();
    let mut report = Report::default();
    let prompts = list(&json["prompts"]);
    for rollout in list(&json["rollouts"]) {
        let label = format!("{backend} {}", text(&rollout["name"]));
        let name = text(&rollout["prompt"]);
        let entry = prompts
            .iter()
            .find(|prompt| prompt["name"] == name)
            .unwrap_or_else(|| panic!("{label}: expected.json has no prompt {name}"));
        let prompt = prompt_of(&entry["rows"]);
        let steps = list(&rollout["steps"]).len();
        let noise = noise_of(&mut report, &label, rollout, sidecar, config);
        if rollout["goal_met"] != true {
            report.note(format!(
                "{label}: the generator did not meet its {} goal",
                rollout["goal"]
            ));
        }
        let (_, mut session) = prefill(&model, &prompt, steps);
        let mut sampler = FishDrama3RowSampler::with_recorded_noise(
            sampling_of(&rollout["params"]),
            config.ids,
            prompt.main_ids(),
            noise.clone(),
        )
        .unwrap();
        let ours = replay(&mut session, &mut sampler, steps);
        let exact = compare_rollout(
            &mut report,
            &label,
            rollout,
            sidecar,
            &ours,
            &tiny_tolerances(device),
            config.trunk.hidden,
        );
        if device.is_cpu() && exact {
            check_model_loop(&mut report, &label, &model, prompt, rollout, noise, &ours);
        }
    }
    report.finish(&format!("L1 tiny rollouts on {backend}"));
}

#[test]
#[ignore = "needs tests/fixtures/fish_drama3/tiny/ from generate_fish_drama3_tiny.py"]
fn tiny_prefill_and_forced_steps_match_upstream_on_cpu() {
    run_tiny_tensors(&Device::Cpu, "cpu");
}

#[test]
#[ignore = "needs tests/fixtures/fish_drama3/tiny/ from generate_fish_drama3_tiny.py"]
fn tiny_rollouts_replay_upstream_on_cpu() {
    run_tiny_rollouts(&Device::Cpu, "cpu");
}

/// L1 on Metal (§10.6 requires it): the same comparisons on the Metal form
/// with [`TINY_METAL`]'s bounds. The model-loop cross-check stays on the
/// CPU. The tiny fixture's 32-dim DeltaNet heads take Metal's Candle
/// recurrence (the published 128-dim heads take `gdn_decode`).
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs tests/fixtures/fish_drama3/tiny/ from generate_fish_drama3_tiny.py"]
fn tiny_matches_upstream_on_metal() {
    let _fixture = tiny_fixture();
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("tiny_matches_upstream_on_metal")
    else {
        return;
    };
    run_tiny_tensors(&device, "metal");
    run_tiny_rollouts(&device, "metal");
}

/// L1 on CUDA (§10.6): the same comparisons on the CUDA form with
/// [`TINY_CUDA`]'s bounds.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs tests/fixtures/fish_drama3/tiny/ from generate_fish_drama3_tiny.py"]
fn tiny_matches_upstream_on_cuda() {
    let _fixture = tiny_fixture();
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    run_tiny_tensors(&device, "cuda");
    run_tiny_rollouts(&device, "cuda");
}

/// The L1 bounds of `device`'s form.
fn tiny_tolerances(device: &Device) -> Tolerances {
    match device {
        Device::Cpu => TINY,
        Device::Metal(_) => TINY_METAL,
        Device::Cuda(_) => TINY_CUDA,
    }
}

// ---------------------------------------------------------------------------
// L1.5: real-weight slices
// ---------------------------------------------------------------------------

/// The slices' inputs are `numeric_check::xorshift_values`: the generators'
/// formula (`fish_drama3_fixture_common.py`), written out from its
/// docstring, gives the same bits.
#[test]
fn slice_inputs_are_the_documented_xorshift() {
    fn documented(count: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // `unit` and `unit * 2 - 1` are exact; one rounding to F32.
                let unit = (state >> 40) as f64 / f64::from(1u32 << 24);
                ((unit * 2.0 - 1.0) * f64::from(scale)) as f32
            })
            .collect()
    }
    for (seed, scale) in [
        (1000, 1.0),
        (1003, 1.0),
        (2000, 1.0),
        (3000, 1.0),
        (7, (3.0f32 / 128.0).sqrt()),
    ] {
        let ours = xorshift_values(4096, seed, scale);
        let theirs = documented(4096, seed, scale);
        assert!(
            ours.iter()
                .zip(&theirs)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "seed {seed}"
        );
        assert!(ours.iter().all(|v| (-scale..scale).contains(v)));
    }
}

/// Where a slice test runs, and the bounds it holds there.
struct SliceBackend {
    name: &'static str,
    device: Device,
    tol: Tolerances,
    /// How far a RED variant must miss, in multiples of its stage's bound.
    red_factor: f64,
    /// Both paged domains' dtype: F32 on the CPU, F16 in the Metal form.
    kv: DType,
}

impl SliceBackend {
    /// The CPU's F32 load, upstream's own numerics.
    fn cpu() -> Self {
        Self {
            name: "cpu",
            device: Device::Cpu,
            tol: SLICE,
            red_factor: RED_FACTOR,
            kv: DType::F32,
        }
    }

    /// The Metal form on `device` (§10.6 requires L1.5 there).
    #[cfg(feature = "metal")]
    fn metal(device: Device) -> Self {
        Self {
            name: "metal",
            device,
            tol: SLICE_METAL,
            red_factor: RED_FACTOR_DEVICE,
            kv: DType::F16,
        }
    }

    /// The CUDA form on `device` (§10.6 requires L1.5 there too).
    #[cfg(feature = "cuda")]
    fn cuda(device: Device) -> Self {
        Self {
            name: "cuda",
            device,
            tol: SLICE_CUDA,
            red_factor: RED_FACTOR_DEVICE,
            kv: DType::BF16,
        }
    }
}

/// The slice directory: the fetched slices, their pinned config and the
/// generator's outputs.
struct Slices {
    dir: PathBuf,
    fixture: Fixture,
    config: FishDrama3Config,
}

fn slices() -> Slices {
    let dir = env_dir(SLICE_DIR_ENV, SLICE_HOW);
    for name in ["slice_a.safetensors", "slice_b.safetensors", "config.json"] {
        let path = dir.join(name);
        assert!(path.is_file(), "{} is missing: {SLICE_HOW}", path.display());
    }
    let fixture = open_fixture(&dir, "slice_expected.json", SLICE_SCHEMA, SLICE_HOW);
    // The outputs came from these very slices: the fetch wrote each one's
    // sha256 beside it, and the generator recorded it.
    for key in ["a", "b"] {
        let recorded = &fixture.json["model"]["slices"][key];
        let file = text(&recorded["file"]);
        let path = dir.join(format!("{file}.sha256"));
        let sha = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}: {SLICE_HOW}", path.display()));
        assert_eq!(
            sha.split_whitespace().next(),
            recorded["sha256"].as_str(),
            "{file} is not the slice slice_expected.json was made from: rerun generate_fish_drama3_slices.py"
        );
    }
    let config = FishDrama3Config::load(&dir).unwrap();
    Slices {
        dir,
        fixture,
        config,
    }
}

/// A record's xorshift input (`{"seed", "rows", "width", "scale",
/// "checksum"}`), checked against the checksum the generator took of its
/// own: the same exact values, so only the F64 summation order may differ.
fn slice_input(report: &mut Report, label: &str, record: &Value) -> Vec<f32> {
    let (seed, rows, width) = (
        int(&record["seed"]) as u64,
        int(&record["rows"]),
        int(&record["width"]),
    );
    let values = xorshift_values(rows * width, seed, float(&record["scale"]) as f32);
    compare_checksum(
        report,
        "inputs",
        &format!("{label} input"),
        &record["checksum"],
        &values,
        1e-9,
    );
    values
}

/// Slice B's `layers` as a trunk of that many layers in that order (slice
/// layer `layers[i]` named `i`), slice A's final norm after them, and
/// `extra` files laying replacement tensors over them (RED variants). F32
/// throughout, as upstream ran.
fn slice_trunk(
    slices: &Slices,
    layers: &[usize],
    extra: &[PathBuf],
    edit: impl FnOnce(&mut FishDrama3Config),
    backend: &SliceBackend,
) -> (FishDrama3Trunk, FishDrama3Config) {
    let mut config = slices.config.clone();
    config.trunk.layers = layers
        .iter()
        .map(|&l| slices.config.trunk.layers[l])
        .collect();
    edit(&mut config);
    let mut files = vec![
        slices.dir.join("slice_b.safetensors"),
        slices.dir.join("slice_a.safetensors"),
    ];
    files.extend_from_slice(extra);
    let prefix = format!("{TRUNK_PREFIX}.layers.");
    let norm = format!("{TRUNK_PREFIX}.norm.weight");
    let view = FishDrama3Checkpoint::open_view(&files, |name| {
        if name == norm {
            return Some(name.to_string());
        }
        let (layer, rest) = name.strip_prefix(prefix.as_str())?.split_once('.')?;
        let index = layers.iter().position(|l| l.to_string() == layer)?;
        Some(format!("{prefix}{index}.{rest}"))
    })
    .unwrap();
    let trunk = FishDrama3Trunk::load(
        &view,
        &config,
        &backend.device,
        FishDrama3TrunkOptions {
            precision: FishDrama3CpuPrecision::F32,
            ..Default::default()
        },
    )
    .unwrap();
    (trunk, config)
}

/// The slice run of `trunk_layers`: prefill chunks, then one-row decode
/// steps, every layer traced over every row.
fn run_slice_trunk(
    trunk: &FishDrama3Trunk,
    x: &[f32],
    chunks: &[(usize, usize)],
    decode: &[usize],
    kv: DType,
) -> LayerTrace {
    let config = trunk.config();
    let h = config.hidden;
    let attention = &config.attention;
    let mut cache = cache_in(
        trunk.device(),
        kv,
        trunk.attention_layers(),
        attention.kv_heads,
        attention.head_dim,
        x.len() / h + 1,
    );
    let mut state = trunk.new_state().unwrap();
    let mut trace = LayerTrace::default();
    let rows = |from: usize, to: usize| {
        Tensor::from_slice(&x[from * h..to * h], (to - from, h), &Device::Cpu).unwrap()
    };
    for &(from, to) in chunks {
        let (_, part) = trunk
            .forward_span_traced(&rows(from, to), from, &mut state, &mut cache)
            .unwrap();
        append_trace(&mut trace, part);
    }
    for &position in decode {
        let mut row = [FishDrama3StepRow {
            state: &mut state,
            cache: &mut cache,
        }];
        let (_, part) = trunk
            .forward_step_traced(&rows(position, position + 1), &mut row)
            .unwrap();
        append_trace(&mut trace, part);
    }
    trace
}

fn slice_layer_entry(slices: &Slices, layer: usize) -> &Value {
    let entry = list(&slices.fixture.json["trunk_layers"])
        .iter()
        .find(|entry| int(&entry["layer"]) == layer)
        .unwrap_or_else(|| panic!("slice_expected.json has no trunk layer {layer}"));
    assert_eq!(
        text(&entry["kind"]),
        kind_name(slices.config.trunk.layers[layer]),
        "slice layer {layer}"
    );
    entry
}

/// One trunk layer's record against trace slot `slot`: the decode rows,
/// their checksums and the history trace of the layer output, the mixer and
/// the MoE block, and the decode rows' routes.
fn compare_slice_layer(
    report: &mut Report,
    label: &str,
    entry: &Value,
    sidecar: &Sidecar,
    trace: &LayerTrace,
    slot: usize,
    config: &FishDrama3Config,
    tol: &Tolerances,
) {
    let h = config.trunk.hidden;
    let (experts, top_k) = (config.trunk.moe.experts, config.trunk.moe.top_k);
    let decode = counts(&entry["decode_positions"]);
    let history = decode[0];
    let routes = &entry["routes"];
    let flipped = compare_routes(
        report,
        &format!("{label} decode routes"),
        &routes["decode"],
        &trace.router_logits[slot],
        experts,
        top_k,
        tol.route_tie,
    );
    report.note(format!(
        "{label}: history near ties ours {}, upstream {} (upstream min margin {})",
        near_tie_rows(
            &trace.router_logits[slot][..history * experts],
            experts,
            top_k,
            tol.route_tie
        ),
        routes["history"]["near_ties"],
        routes["history"]["min_margin"]
    ));
    for (part, ours) in [
        ("output", &trace.outputs[slot]),
        ("mixer", &trace.mixers[slot]),
        ("moe", &trace.moes[slot]),
    ] {
        let record = &entry[part];
        let upstream = sidecar.f32(&record["decode"]);
        // A row whose experts flipped at a near tie leaves the MoE and the
        // layer output; the mixer never routes.
        let kept = (0..decode.len())
            .filter(|&i| part == "mixer" || !flipped.contains(&decode[i]))
            .collect::<Vec<_>>();
        let positions = kept.iter().map(|&i| decode[i]).collect::<Vec<_>>();
        compare_values(
            report,
            "trunk",
            &format!("{label} {part} at decode rows {positions:?}"),
            &gather(ours, h, &positions),
            &gather(&upstream, h, &kept),
            tol.trunk,
            false,
        );
        if kept.len() == decode.len() {
            compare_checksum(
                report,
                "trunk",
                &format!("{label} {part} decode"),
                &record["decode_checksum"],
                &ours[history * h..],
                tol.trunk.rel_l2,
            );
        }
        compare_trace(
            report,
            "trunk",
            &format!("{label} {part} history"),
            &record["history"],
            sidecar,
            &ours[..history * h],
            h,
            tol.trunk,
        );
    }
}

/// Upstream's mixer output at every row it stored (the history picks, then
/// the decode rows) and ours (`[rows, hidden]`) at the same rows.
fn mixer_rows(entry: &Value, sidecar: &Sidecar, ours: &[f32], h: usize) -> (Vec<f32>, Vec<f32>) {
    let mixer = &entry["mixer"];
    let mut upstream = sidecar.f32(&mixer["history"]["values"]);
    upstream.extend(sidecar.f32(&mixer["decode"]));
    let rows = counts(&mixer["history"]["rows"])
        .into_iter()
        .chain(counts(&entry["decode_positions"]))
        .collect::<Vec<_>>();
    (gather(ours, h, &rows), upstream)
}

/// RED: slice B layer `layer`'s value heads reordered so the grouped kernel
/// (value head `i` reads key head `i / g`) computes the tiled pairing (value
/// head `j` reads key head `j % K`): slot `i` holds value head
/// `(i % g) * K + i / g` in every value-head tensor. The q/k channels and the
/// per-dim output norm stay. Written into `dir` under the slice's own names,
/// to lie over slice B.
fn tiled_value_heads(slices: &Slices, layer: usize, dir: &Path) -> PathBuf {
    let la = slices.config.trunk.linear;
    let (k, v, dk, dv) = (
        la.key_heads,
        la.value_heads,
        la.key_head_dim,
        la.value_head_dim,
    );
    let groups = v / k;
    let source = (0..v)
        .map(|i| (i % groups) * k + i / groups)
        .collect::<Vec<_>>();
    let view = FishDrama3Checkpoint::open_view(&[slices.dir.join("slice_b.safetensors")], |name| {
        Some(name.to_string())
    })
    .unwrap();
    let x = format!("{}.linear_attn", trunk_layer(layer));
    let mut tensors: Vec<Written> = Vec::new();
    // (tensor, its first value-head row, rows per head) along the leading
    // dimension.
    for (suffix, first, per_head) in [
        ("in_proj_qkv.weight", 2 * k * dk, dv),
        ("conv1d.weight", 2 * k * dk, dv),
        ("in_proj_z.weight", 0, dv),
        ("in_proj_a.weight", 0, 1),
        ("in_proj_b.weight", 0, 1),
        ("A_log", 0, 1),
        ("dt_bias", 0, 1),
    ] {
        let name = format!("{x}.{suffix}");
        let (dtype, shape) = view.info(&name).unwrap();
        let bytes = view.bytes(&name).unwrap();
        let row = bytes.len() / shape[0];
        let mut out = bytes.to_vec();
        for (slot, &head) in source.iter().enumerate() {
            let (to, from) = (
                (first + slot * per_head) * row,
                (first + head * per_head) * row,
            );
            out[to..to + per_head * row].copy_from_slice(&bytes[from..from + per_head * row]);
        }
        tensors.push((name, dtype, shape.to_vec(), out));
    }
    // out_proj `[hidden, value heads * dv]`: blocks of dv columns.
    let name = format!("{x}.out_proj.weight");
    let (dtype, shape) = view.info(&name).unwrap();
    let bytes = view.bytes(&name).unwrap();
    let element = bytes.len() / (shape[0] * shape[1]);
    let (row, block) = (shape[1] * element, dv * element);
    let mut out = bytes.to_vec();
    for r in 0..shape[0] {
        for (slot, &head) in source.iter().enumerate() {
            let (to, from) = (r * row + slot * block, r * row + head * block);
            out[to..to + block].copy_from_slice(&bytes[from..from + block]);
        }
    }
    tensors.push((name, dtype, shape.to_vec(), out));
    let path = dir.join("tiled_value_heads.safetensors");
    let views = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::serialize_to_file(views, None, &path).unwrap();
    path
}

/// Layer 0 (Gated DeltaNet) on real weights: three prefill chunks continue
/// the chunked rule's state, then four recurrent decode steps; RED: tiled
/// value heads.
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 7 GB of RAM; run with --release"]
fn slice_trunk_deltanet_layer_0_matches_upstream() {
    run_slice_layer_0(&slices(), &SliceBackend::cpu());
}

/// [`slice_trunk_deltanet_layer_0_matches_upstream`] in the Metal form.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 7 GB of RAM; run with --release"]
fn slice_trunk_deltanet_layer_0_matches_upstream_on_metal() {
    let slices = slices();
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "slice_trunk_deltanet_layer_0_matches_upstream_on_metal",
    ) else {
        return;
    };
    run_slice_layer_0(&slices, &SliceBackend::metal(device));
}

/// [`slice_trunk_deltanet_layer_0_matches_upstream`] in the CUDA form.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 7 GB of RAM; run with --release"]
fn slice_trunk_deltanet_layer_0_matches_upstream_on_cuda() {
    let slices = slices();
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    run_slice_layer_0(&slices, &SliceBackend::cuda(device));
}

fn run_slice_layer_0(slices: &Slices, backend: &SliceBackend) {
    let sidecar = &slices.fixture.sidecar;
    let mut report = Report::default();
    let entry = slice_layer_entry(slices, 0);
    let x = slice_input(&mut report, "layer 0", &entry["input"]);
    let (chunks, decode) = (
        spans(&entry["prefill_chunks"]),
        counts(&entry["decode_positions"]),
    );
    // A trunk's paged domain needs an attention layer (a KV arena holds at
    // least one), so slice layer 3 rides behind layer 0 as layer 1; only
    // layer 0's record is read.
    {
        let (trunk, config) = slice_trunk(slices, &[0, 3], &[], |_| {}, backend);
        let trace = run_slice_trunk(&trunk, &x, &chunks, &decode, backend.kv);
        compare_slice_layer(
            &mut report,
            "layer 0",
            entry,
            sidecar,
            &trace,
            0,
            &config,
            &backend.tol,
        );
    }
    let red_dir = temp_dir();
    let tiled = tiled_value_heads(slices, 0, &red_dir.0);
    let (trunk, config) = slice_trunk(slices, &[0, 3], &[tiled], |_| {}, backend);
    let trace = run_slice_trunk(&trunk, &x, &chunks, &decode, backend.kv);
    let (ours, upstream) = mixer_rows(entry, sidecar, &trace.mixers[0], config.trunk.hidden);
    red(
        &mut report,
        "tiled value heads (layer 0 mixer)",
        &ours,
        &upstream,
        backend.tol.trunk.rel_l2,
        backend.red_factor,
    );
    report.finish(&format!(
        "L1.5 trunk layer 0 (Gated DeltaNet) on {}",
        backend.name
    ));
}

/// Layer 3 (gated full attention) on real weights: prefill chunks read the
/// cache, then four decode steps; RED: interleaved RoPE pairs.
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 4 GB of RAM; run with --release"]
fn slice_trunk_attention_layer_3_matches_upstream() {
    run_slice_layer_3(&slices(), &SliceBackend::cpu());
}

/// [`slice_trunk_attention_layer_3_matches_upstream`] in the Metal form.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 4 GB of RAM; run with --release"]
fn slice_trunk_attention_layer_3_matches_upstream_on_metal() {
    let slices = slices();
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "slice_trunk_attention_layer_3_matches_upstream_on_metal",
    ) else {
        return;
    };
    run_slice_layer_3(&slices, &SliceBackend::metal(device));
}

/// [`slice_trunk_attention_layer_3_matches_upstream`] in the CUDA form.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 4 GB of RAM; run with --release"]
fn slice_trunk_attention_layer_3_matches_upstream_on_cuda() {
    let slices = slices();
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    run_slice_layer_3(&slices, &SliceBackend::cuda(device));
}

fn run_slice_layer_3(slices: &Slices, backend: &SliceBackend) {
    let sidecar = &slices.fixture.sidecar;
    let mut report = Report::default();
    let entry = slice_layer_entry(slices, 3);
    let x = slice_input(&mut report, "layer 3", &entry["input"]);
    let (chunks, decode) = (
        spans(&entry["prefill_chunks"]),
        counts(&entry["decode_positions"]),
    );
    {
        let (trunk, config) = slice_trunk(slices, &[3], &[], |_| {}, backend);
        let trace = run_slice_trunk(&trunk, &x, &chunks, &decode, backend.kv);
        compare_slice_layer(
            &mut report,
            "layer 3",
            entry,
            sidecar,
            &trace,
            0,
            &config,
            &backend.tol,
        );
    }
    let (trunk, config) = slice_trunk(
        slices,
        &[3],
        &[],
        |config| {
            config.trunk.attention.rope_layout = FishDrama3RopeLayout::InterleavedPairs;
        },
        backend,
    );
    let trace = run_slice_trunk(&trunk, &x, &chunks, &decode, backend.kv);
    let (ours, upstream) = mixer_rows(entry, sidecar, &trace.mixers[0], config.trunk.hidden);
    red(
        &mut report,
        "interleaved RoPE in the trunk (layer 3 mixer)",
        &ours,
        &upstream,
        backend.tol.trunk.rel_l2,
        backend.red_factor,
    );
    report.finish(&format!(
        "L1.5 trunk layer 3 (gated attention) on {}",
        backend.name
    ));
}

/// Slice A with its two row slices named as the whole tables they stand
/// for; the re-based configs below index them.
fn slice_a_view(slices: &Slices) -> FishDrama3Checkpoint {
    let embed = format!("{TRUNK_PREFIX}.embed_tokens.weight");
    let semantic = format!("{embed}.semantic_rows");
    let compact = format!("{LM_HEAD}.compact_rows");
    FishDrama3Checkpoint::open_view(&[slices.dir.join("slice_a.safetensors")], |name| {
        Some(if name == semantic {
            embed.clone()
        } else if name == compact {
            LM_HEAD.to_string()
        } else {
            name.to_string()
        })
    })
    .unwrap()
}

/// The ids re-based onto slice A's `lm_head` rows (eos, then the band): eos
/// 0 and the band 1..=4096, so the compact head reads all 4,097 in order.
fn compact_rows_config(config: &FishDrama3Config) -> FishDrama3Config {
    let mut rebased = config.clone();
    let codes = config.ids.semantic_codes() as u32;
    rebased.ids.eos = 0;
    rebased.ids.semantic_start = 1;
    rebased.ids.semantic_end = codes;
    rebased.trunk.vocab = codes as usize + 1;
    rebased
}

/// The band re-based onto slice A's `embed_tokens` rows (the band only):
/// `semantic_start` 0, so a semantic row's main id becomes its code. The
/// tower reads only the codebook table through it.
fn semantic_rows_config(config: &FishDrama3Config) -> FishDrama3Config {
    let mut rebased = config.clone();
    let codes = config.ids.semantic_codes() as u32;
    rebased.ids.semantic_start = 0;
    rebased.ids.semantic_end = codes - 1;
    rebased.trunk.vocab = codes as usize;
    rebased
}

fn slice_tower(
    view: &FishDrama3Checkpoint,
    config: &FishDrama3Config,
    edit: impl FnOnce(&mut FishDrama3Config),
    device: &Device,
) -> FishDrama3Tower {
    let mut config = config.clone();
    edit(&mut config);
    FishDrama3Tower::load(
        view,
        &config,
        device,
        FishDrama3TowerOptions {
            precision: FishDrama3CpuPrecision::F32,
            ..Default::default()
        },
    )
    .unwrap()
}

/// What the slice tower run computed.
struct TowerRun {
    /// Every layer's output over the prefilled rows.
    prefill: Vec<Vec<f32>>,
    /// Every layer's output at the forced step, `[layers, dim]`.
    step: Vec<f32>,
    /// The heads' logits at the forced step, `[heads, V]`.
    heads: Vec<f32>,
}

/// The slice tower run: `chunks` teacher-forced (row `p` labelled with row
/// `p + 1`), then a forced step at `step.0` under the frame `step.1`.
fn run_slice_tower(
    tower: &FishDrama3Tower,
    embedding: &FishDrama3Embedding,
    hidden: &[f32],
    frames: &[Vec<u32>],
    chunks: &[(usize, usize)],
    step: Option<(usize, &[u32])>,
    kv: DType,
) -> TowerRun {
    let config = tower.config();
    let d = config.dim;
    let attention = &config.attention;
    let mut cache = cache_in(
        tower.device(),
        kv,
        tower.attention_layers(),
        attention.kv_heads,
        attention.head_dim,
        frames.len() + 1,
    );
    let rows = |from: usize, to: usize| {
        Tensor::from_slice(&hidden[from * d..to * d], (to - from, d), &Device::Cpu).unwrap()
    };
    let mut prefill = Vec::new();
    for &(from, to) in chunks {
        let labels = frames[from..=to].concat();
        let trace = tower
            .forward_span_traced(embedding, &rows(from, to), &labels, from, &mut cache)
            .unwrap();
        append_layers(&mut prefill, trace);
    }
    let (heads, step) = match step {
        Some((position, label)) => {
            let mut forced = [FishDrama3TowerForcedRow {
                cache: &mut cache,
                input: &frames[position],
                label,
            }];
            tower
                .forward_step_forced_traced(embedding, &rows(position, position + 1), &mut forced)
                .unwrap()
        }
        None => (Vec::new(), Vec::new()),
    };
    TowerRun {
        prefill,
        step,
        heads,
    }
}

/// The tower on real weights: 300 teacher-forced rows (40 text rows, then
/// frames with head-range codes ≥ 1024), a forced decode step at row 300
/// with every layer's output and the 9 heads' logits; RED: the trunk's
/// half-split RoPE in the tower.
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 6 GB of RAM; run with --release"]
fn slice_tower_matches_upstream_on_a_forced_step() {
    run_slice_tower_test(&slices(), &SliceBackend::cpu());
}

/// [`slice_tower_matches_upstream_on_a_forced_step`] in the Metal form.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 6 GB of RAM; run with --release"]
fn slice_tower_matches_upstream_on_a_forced_step_on_metal() {
    let slices = slices();
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "slice_tower_matches_upstream_on_a_forced_step_on_metal",
    ) else {
        return;
    };
    run_slice_tower_test(&slices, &SliceBackend::metal(device));
}

/// [`slice_tower_matches_upstream_on_a_forced_step`] in the CUDA form.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); about 6 GB of RAM; run with --release"]
fn slice_tower_matches_upstream_on_a_forced_step_on_cuda() {
    let slices = slices();
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    run_slice_tower_test(&slices, &SliceBackend::cuda(device));
}

fn run_slice_tower_test(slices: &Slices, backend: &SliceBackend) {
    let (json, sidecar) = (&slices.fixture.json, &slices.fixture.sidecar);
    let config = &slices.config;
    let entry = &json["tower"];
    let tol = &backend.tol;
    let mut report = Report::default();
    let view = slice_a_view(slices);
    let embedding =
        FishDrama3Embedding::load_on(&view, &semantic_rows_config(config), &backend.device)
            .unwrap();
    let hidden = slice_input(&mut report, "tower", &entry["hidden"]);
    let frames = table(&entry["frames"]);
    let forced = ids(&entry["forced_frame"]);
    let (chunks, position) = (spans(&entry["prefill_chunks"]), int(&entry["position"]));
    report.check(
        frames
            .iter()
            .chain([&forced])
            .any(|frame| frame[2..].iter().any(|&code| code >= 1024)),
        || "the slice frames hold no head-range code (>= 1024)".into(),
    );
    let d = config.tower.dim;
    let layers = list(&entry["layers"]);
    {
        let tower = slice_tower(&view, config, |_| {}, &backend.device);
        let run = run_slice_tower(
            &tower,
            &embedding,
            &hidden,
            &frames,
            &chunks,
            Some((position, forced.as_slice())),
            backend.kv,
        );
        for layer in layers {
            let i = int(&layer["index"]);
            check_tower_kind(&mut report, "slice", layer, config.tower.shared_layers);
            compare_trace(
                &mut report,
                "tower",
                &format!("tower layer {i} prefill"),
                &layer["prefill"],
                sidecar,
                &run.prefill[i],
                d,
                tol.tower,
            );
            compare_values(
                &mut report,
                "tower",
                &format!("tower layer {i} at the forced step"),
                &run.step[i * d..(i + 1) * d],
                &sidecar.f32(&layer["step"]),
                tol.tower,
                false,
            );
        }
        compare_values(
            &mut report,
            "logits",
            "tower head logits at the forced step",
            &run.heads,
            &sidecar.f32(&entry["head_logits"]),
            tol.logits,
            false,
        );
    }
    // RED over the first chunk: the last layer at its stored rows there.
    let tower = slice_tower(
        &view,
        config,
        |config| {
            config.tower.attention.rope_layout = FishDrama3RopeLayout::HalfSplit;
        },
        &backend.device,
    );
    let run = run_slice_tower(
        &tower,
        &embedding,
        &hidden,
        &frames,
        &chunks[..1],
        None,
        backend.kv,
    );
    let last = layers.last().expect("tower layers");
    let picks = counts(&last["prefill"]["rows"]);
    let kept = (0..picks.len())
        .filter(|&i| picks[i] < chunks[0].1)
        .collect::<Vec<_>>();
    let rows = kept.iter().map(|&i| picks[i]).collect::<Vec<_>>();
    red(
        &mut report,
        "half-split RoPE in the tower (last layer)",
        &gather(&run.prefill[int(&last["index"])], d, &rows),
        &gather(&sidecar.f32(&last["prefill"]["values"]), d, &kept),
        tol.tower.rel_l2,
        backend.red_factor,
    );
    report.finish(&format!("L1.5 tower on {}", backend.name));
}

/// The compact main head on the final norm of xorshift rows (RED: the final
/// norm without its `+1`), and the trunk embedding of 8 semantic rows.
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); run with --release"]
fn slice_main_head_and_embedding_match_upstream() {
    run_slice_head_test(&slices(), &SliceBackend::cpu());
}

/// [`slice_main_head_and_embedding_match_upstream`] in the Metal form.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); run with --release"]
fn slice_main_head_and_embedding_match_upstream_on_metal() {
    let slices = slices();
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "slice_main_head_and_embedding_match_upstream_on_metal",
    ) else {
        return;
    };
    run_slice_head_test(&slices, &SliceBackend::metal(device));
}

/// [`slice_main_head_and_embedding_match_upstream`] in the CUDA form.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_SLICE_DIR (fetch_fish_drama3_slices.py, then generate_fish_drama3_slices.py); run with --release"]
fn slice_main_head_and_embedding_match_upstream_on_cuda() {
    let slices = slices();
    let Some(device) = crate::kernels::cuda::cuda_test_device() else {
        return;
    };
    run_slice_head_test(&slices, &SliceBackend::cuda(device));
}

fn run_slice_head_test(slices: &Slices, backend: &SliceBackend) {
    let (json, sidecar) = (&slices.fixture.json, &slices.fixture.sidecar);
    let config = &slices.config;
    let h = config.trunk.hidden;
    let tol = &backend.tol;
    let mut report = Report::default();
    let view = slice_a_view(slices);

    let entry = &json["main_head"];
    let candidates = format!(
        "eos {}, then {}..{}",
        config.ids.eos, config.ids.semantic_start, config.ids.semantic_end
    );
    report.check(entry["candidates"] == candidates.as_str(), || {
        format!(
            "main head candidates {}, izwi {candidates}",
            entry["candidates"]
        )
    });
    let x = slice_input(&mut report, "main head", &entry["input"]);
    let head =
        FishDrama3CompactHead::load(&view, &compact_rows_config(config), &backend.device).unwrap();
    // The final norm on the host in F32 (the trunk's own, in either form,
    // hands the head F32 rows), then the head in its form.
    let logits = |gain: NormGain| {
        let gain = norm_gain(&view, &format!("{TRUNK_PREFIX}.norm.weight"), h, gain).unwrap();
        let normed = rms_norm_rows(&x, h, &gain, config.trunk.rms_norm_eps as f32);
        let rows = Tensor::from_vec(normed, (x.len() / h, h), &backend.device).unwrap();
        host(&head.logits(&rows).unwrap())
    };
    let upstream = sidecar.f32(&entry["logits"]);
    compare_values(
        &mut report,
        "logits",
        "main head logits",
        &logits(NormGain::ZeroCentred),
        &upstream,
        tol.logits,
        false,
    );
    red(
        &mut report,
        "no 1 + w on the final norm (main head logits)",
        &logits(NormGain::Plain),
        &upstream,
        tol.logits.rel_l2,
        backend.red_factor,
    );

    let entry = &json["embedding"];
    let embedding =
        FishDrama3Embedding::load_on(&view, &semantic_rows_config(config), &backend.device)
            .unwrap();
    let rows = table(&entry["frames"])
        .into_iter()
        .flat_map(|mut frame| {
            frame[0] = config
                .ids
                .semantic_code(frame[0])
                .expect("the embedding rows are semantic");
            frame
        })
        .collect::<Vec<_>>();
    compare_values(
        &mut report,
        "embedding",
        "semantic-row embedding",
        &embedding.embed(&rows).unwrap(),
        &sidecar.f32(&entry["values"]),
        tol.embedding,
        false,
    );
    report.finish(&format!("L1.5 main head and embedding on {}", backend.name));
}

// ---------------------------------------------------------------------------
// L2: the whole-model reference
// ---------------------------------------------------------------------------

struct ReferenceRun {
    model_dir: PathBuf,
    dir: PathBuf,
    fixture: Fixture,
}

fn reference_run() -> ReferenceRun {
    let dir = env_dir(REFERENCE_DIR_ENV, REFERENCE_HOW);
    let model_dir = env_dir(MODEL_DIR_ENV, REFERENCE_HOW);
    let fixture = open_fixture(&dir, "reference.json", REFERENCE_SCHEMA, REFERENCE_HOW);
    ReferenceRun {
        model_dir,
        dir,
        fixture,
    }
}

/// Every upstream `generate_drama3` call of a case, labelled: a long text's
/// chunks, or the case itself.
fn generations(case: &Value) -> Vec<(String, &Value)> {
    let name = text(&case["name"]);
    if case["kind"] == "chunks" {
        list(&case["chunks"])
            .iter()
            .map(|chunk| (format!("{name} chunk {}", int(&chunk["index"])), chunk))
            .collect()
    } else {
        vec![(name.to_string(), case)]
    }
}

/// `long_text`: izwi's chunker against upstream's split, then every chunk's
/// prompt and the history turns it kept, izwi's session fed the frames
/// upstream made for the earlier chunks (as upstream's own loop was).
fn compare_chunks(
    report: &mut Report,
    case: &Value,
    tokenizer: &FishDrama3Tokenizer,
    config: &FishDrama3Config,
) {
    let name = text(&case["name"]);
    let whole = text(&case["text"]);
    let split = list(&case["split"])
        .iter()
        .map(|s| text(s).to_string())
        .collect::<Vec<_>>();
    let ours = plan_chunks(whole, false);
    report.check(ours == split, || {
        format!("{name}: izwi splits into {ours:?}, upstream {split:?}")
    });
    let mut options = FishDrama3ChunkSessionOptions::new(
        FishDrama3SamplingParams::from_request(None, None, None, 0).unwrap(),
    );
    options.max_tokens = Some(int(&case["chunker"]["requested_frames"]));
    options.mask_first_eos = false;
    let mut session =
        FishDrama3ChunkSession::new(whole, references_of(&case["references"]), options).unwrap();
    let positions = config.trunk.max_positions.min(MAX_REQUEST_POSITIONS);
    let mut histories = Vec::new();
    for chunk in list(&case["chunks"]) {
        let label = format!("{name} chunk {}", int(&chunk["index"]));
        let Some(plan) = session.plan_next(tokenizer, positions).unwrap() else {
            report.fail(format!("{label}: izwi has no chunk left"));
            break;
        };
        report.check(plan.text == text(&chunk["text"]), || {
            format!("{label}: izwi's chunk is {:?}", plan.text)
        });
        let history = counts(&chunk["history"]);
        report.check(plan.history == history, || {
            format!("{label}: history {:?}, upstream {history:?}", plan.history)
        });
        compare_prompt(report, &label, &chunk["rows"], Ok(plan.prompt.clone()));
        if plan.params.max_frames != int(&chunk["budget"]) {
            report.note(format!(
                "{label}: izwi's budget {}, upstream's {}",
                plan.params.max_frames, chunk["budget"]
            ));
        }
        histories.push(plan.history.clone());
        session
            .complete(&plan, &codes_of_frames(&table(&chunk["rollout"]["frames"])))
            .unwrap();
    }
    let expected = list(&case["self_check_chunks"]["expected_histories"])
        .iter()
        .map(counts)
        .collect::<Vec<_>>();
    report.check(histories == expected, || {
        format!("{name}: histories {histories:?}, expected {expected:?}")
    });
    report.check(session.is_done(), || {
        format!("{name}: izwi plans more chunks than upstream ran")
    });
}

/// `take_edit` (§8.4): the previous take is a completed history turn
/// holding exactly the utterance's codes. The case has no references, so
/// its only audio rows are the take's.
fn take_edit_shape(
    report: &mut Report,
    case: &Value,
    utterance: Option<&Value>,
    config: &FishDrama3Config,
) {
    let Some(utterance) = utterance else {
        report.fail("take_edit: reference.json has no utterance case".into());
        return;
    };
    let codes = table(&utterance["codes"]);
    let takes = list(&case["previous_takes"]);
    report.check(
        takes.len() == 1
            && table(&takes[0]["codes"]) == codes
            && takes[0]["text"] == utterance["text"],
        || "take_edit: its previous take is not the utterance".into(),
    );
    let audio = table(&case["rows"])
        .into_iter()
        .filter(|row| config.ids.is_semantic(row[0]))
        .collect::<Vec<_>>();
    let held = (0..codes.len())
        .map(|k| audio.iter().map(|row| row[1 + k]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    report.check(held == codes, || {
        format!(
            "take_edit: its prompt holds {} audio rows, the take {} frames",
            audio.len(),
            codes.first().map_or(0, Vec::len)
        )
    });
}

/// Every generation's prompt rows (references with their recorded codes,
/// dialogue tags, the take edit's history turn), the long text's split,
/// chunk prompts and kept histories, and the take edit's shape. Needs only
/// the tokenizer.
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_REFERENCE_DIR (generate_fish_drama3_reference.py) and IZWI_FISH_DRAMA3_MODEL_DIR (its config.json and tokenizer.json)"]
fn reference_prompts_match_upstream() {
    let run = reference_run();
    let config = FishDrama3Config::load(&run.model_dir).unwrap();
    let tokenizer = FishDrama3Tokenizer::load(&run.model_dir, &config).unwrap();
    let cases = list(&run.fixture.json["cases"]);
    let utterance = cases.iter().find(|case| case["kind"] == "utterance");
    let mut report = Report::default();
    for case in cases {
        let name = text(&case["name"]);
        match text(&case["kind"]) {
            "chunks" => compare_chunks(&mut report, case, &tokenizer, &config),
            kind => {
                let references = references_of(&case["references"]);
                let history = match kind {
                    "take_edit" => turns_of(&case["previous_takes"]),
                    _ => Vec::new(),
                };
                let built = build_prompt(&tokenizer, &references, &history, text(&case["text"]));
                compare_prompt(&mut report, name, &case["rows"], built);
                if kind == "take_edit" {
                    take_edit_shape(&mut report, case, utterance, &config);
                }
            }
        }
    }
    report.finish("L2 prompts");
}

fn load_codec(model_dir: &Path) -> FishDrama3DacDecoder {
    let weights = FishDrama3CodecArtifact::load(model_dir)
        .unwrap()
        .load_weights(&Device::Cpu, DType::F32)
        .unwrap();
    FishDrama3DacDecoder::load(FishDrama3DacConfig::current(), weights.var_builder()).unwrap()
}

/// The family's codec on the utterance's recorded codes (head-range codes
/// clamped by the codec, as upstream's) against upstream's codec PCM. Also
/// reports how often izwi's encoder agrees with upstream's codes on the very
/// samples upstream encoded.
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_REFERENCE_DIR (generate_fish_drama3_reference.py) and IZWI_FISH_DRAMA3_MODEL_DIR (its codec.pth); run with --release"]
fn reference_utterance_pcm_matches_the_upstream_codec() {
    let run = reference_run();
    let (json, sidecar) = (&run.fixture.json, &run.fixture.sidecar);
    let case = list(&json["cases"])
        .iter()
        .find(|case| case["kind"] == "utterance")
        .expect("reference.json has no utterance case (--cases left utterance_10s out)");
    let pcm = &case["pcm"];
    let mut report = Report::default();
    let decoder = load_codec(&run.model_dir);
    let rate = decoder.config().sample_rate;
    report.check(int(&pcm["sample_rate"]) == rate as usize, || {
        format!(
            "upstream PCM at {} Hz, izwi's codec at {rate}",
            pcm["sample_rate"]
        )
    });
    let codes = FishDrama3Codes::new(table(&case["codes"]));
    let ours = decoder.decode_vq_codes(&codes).unwrap();
    compare_values(
        &mut report,
        "pcm",
        &format!("PCM of {} frames", codes.frames()),
        &ours,
        &sidecar.f32(&pcm["tensor"]),
        PCM,
        false,
    );
    compare_checksum(
        &mut report,
        "pcm",
        "PCM",
        &pcm["checksum"],
        &ours,
        PCM.rel_l2,
    );
    for clip in list(&json["clips"]) {
        let name = text(&clip["name"]);
        let samples = read_f32(&run.dir.join(text(&clip["raw_file"])));
        let theirs = table(&clip["codes"]);
        match decoder.encode_reference_audio(&samples, rate) {
            Ok(ours) => {
                let (same, total) = ours
                    .codebooks
                    .iter()
                    .zip(&theirs)
                    .flat_map(|(a, b)| a.iter().zip(b))
                    .fold((0, 0), |(same, total), (a, b)| {
                        (same + usize::from(a == b), total + 1)
                    });
                report.note(format!(
                    "clip {name}: izwi's encoder makes {} frames (upstream {}); {same} of {total} codes agree",
                    ours.frames(),
                    theirs.first().map_or(0, Vec::len)
                ));
            }
            Err(error) => report.note(format!("clip {name}: izwi's encoder refuses it: {error}")),
        }
    }
    report.finish("L2 utterance PCM");
}

/// What keeps the shared Metal device to the L2 run (its test lease), held
/// for the whole run; nothing on the CPU.
type DeviceLease = Option<Box<dyn std::any::Any>>;

/// `IZWI_FISH_DRAMA3_BACKEND`: the L2 run's name, bounds, CPU precision,
/// device and the device's lease.
fn reference_backend() -> (
    String,
    Tolerances,
    FishDrama3CpuPrecision,
    Device,
    DeviceLease,
) {
    let backend = std::env::var(BACKEND_ENV).unwrap_or_else(|_| "cpu".into());
    let (tolerances, precision, device, lease) = match backend.as_str() {
        "cpu" => (L2_CPU, FishDrama3CpuPrecision::F32, Device::Cpu, None),
        "cpu-q8_0" => (L2_CPU_Q8_0, FishDrama3CpuPrecision::Q8_0, Device::Cpu, None),
        #[cfg(feature = "metal")]
        "metal" => {
            let (device, lease) =
                crate::backends::metal_test_device_or_skip("reference_generations_match_upstream")
                    .unwrap_or_else(|| panic!("{BACKEND_ENV}=metal: no Metal device"));
            let lease: DeviceLease = Some(Box::new(lease));
            (L2_METAL, FishDrama3CpuPrecision::Q8_0, device, lease)
        }
        #[cfg(not(feature = "metal"))]
        "metal" => panic!("{BACKEND_ENV}=metal needs a test build with --features metal"),
        #[cfg(feature = "cuda")]
        "cuda" => {
            let device = crate::kernels::cuda::cuda_test_device()
                .unwrap_or_else(|| panic!("{BACKEND_ENV}=cuda: no CUDA device"));
            if let Err(error) =
                super::residency::FishDrama3Residency::Cuda.require_supported(&device)
            {
                panic!("{BACKEND_ENV}=cuda: {error}");
            }
            (L2_CUDA, FishDrama3CpuPrecision::Q8_0, device, None)
        }
        #[cfg(not(feature = "cuda"))]
        "cuda" => panic!("{BACKEND_ENV}=cuda needs a test build with --features cuda"),
        other => panic!("{BACKEND_ENV}={other}: expected cpu, cpu-q8_0, metal or cuda"),
    };
    (backend, tolerances, precision, device, lease)
}

/// One upstream generation (a `<generation>` record) against izwi's traced
/// prefill and its replay under the recorded noise, up to
/// [`REFERENCE_FRAMES`] frames.
fn compare_generation(
    report: &mut Report,
    label: &str,
    generation: &Value,
    sidecar: &Sidecar,
    model: &FishDrama3Model,
    tol: &Tolerances,
) {
    let config = model.config();
    let (h, d) = (config.trunk.hidden, config.tower.dim);
    let (experts, top_k) = (config.trunk.moe.experts, config.trunk.moe.top_k);
    let prompt = prompt_of(&generation["rows"]);
    report.check(prompt.rows == int(&generation["prompt_rows"]), || {
        format!("{label}: prompt_rows {}", generation["prompt_rows"])
    });
    let rollout = &generation["rollout"];
    let steps = list(&rollout["steps"]).len().min(REFERENCE_FRAMES);
    let noise = noise_of(report, label, rollout, sidecar, config);
    let (prefill, mut session) = prefill(model, &prompt, steps);

    let trunk = &generation["trunk"];
    compare_trace(
        report,
        "embedding",
        &format!("{label} input"),
        &trunk["input"],
        sidecar,
        &prefill.input,
        h,
        tol.embedding,
    );
    for layer in list(&trunk["layers"]) {
        let l = int(&layer["layer"]);
        compare_trace(
            report,
            "trunk",
            &format!("{label} layer {l}"),
            &layer["output"],
            sidecar,
            &prefill.trunk.outputs[l],
            h,
            tol.trunk,
        );
    }
    compare_trace(
        report,
        "trunk",
        &format!("{label} final norm"),
        &trunk["final_norm"],
        sidecar,
        &prefill.final_norm,
        h,
        tol.trunk,
    );
    let mut tie = tol.route_tie;
    for routes in list(&trunk["routes"]) {
        let l = int(&routes["layer"]);
        tie = tie.max(routes["near_tie_threshold"].as_f64().unwrap_or(0.0) as f32);
        compare_routes(
            report,
            &format!("{label} layer {l} routes"),
            routes,
            &prefill.trunk.router_logits[l],
            experts,
            top_k,
            tol.route_tie,
        );
    }
    let by_layer = &trunk["near_ties_by_layer"];
    let (mut ours, mut theirs) = (0, 0);
    for (l, logits) in prefill.trunk.router_logits.iter().enumerate() {
        ours += near_tie_rows(logits, experts, top_k, tie);
        theirs += by_layer[l.to_string()].as_u64().unwrap_or(0) as usize;
    }
    report.note(format!(
        "{label}: router near ties (margin < {tie:.0e}) over every layer and row: ours {ours}, upstream {theirs}"
    ));
    for layer in list(&generation["tower"]["layers"]) {
        let i = int(&layer["index"]);
        compare_trace(
            report,
            "tower",
            &format!("{label} tower layer {i}"),
            &layer["output"],
            sidecar,
            &prefill.tower[i],
            d,
            tol.tower,
        );
    }

    // The main logits at the last prompt row: all of them, and upstream's
    // top 16 by absolute difference.
    let logits = session.logits();
    let main = &generation["main_logits"];
    compare_values(
        report,
        "logits",
        &format!("{label} main logits"),
        &logits,
        &sidecar.f32(&main["tensor"]),
        tol.logits,
        false,
    );
    let top16 = list(&main["top16"]);
    let worst = top16
        .iter()
        .map(|entry| {
            let entry = list(entry);
            (f64::from(logits[int(&entry[0])]) - float(&entry[2])).abs()
        })
        .fold(
            0f64,
            |worst, x| if x.is_nan() || x > worst { x } else { worst },
        );
    report.bound(
        "top-16 logits max abs",
        &format!("{label} top-16 main logits max abs"),
        worst,
        tol.logits.max_abs,
        false,
    );

    let mut sampler = FishDrama3RowSampler::with_recorded_noise(
        sampling_of(&rollout["params"]),
        config.ids,
        prompt.main_ids(),
        noise,
    )
    .unwrap();
    let replayed = replay(&mut session, &mut sampler, steps);
    compare_rollout(report, label, rollout, sidecar, &replayed, tol, h);
    report.note(format!(
        "{label}: upstream's cached decode against its full forward: rel L2 {}",
        generation["self_check"]["relative_l2"]
    ));
}

/// Every generation of every case (plain, one reference, the two-speaker
/// dialogue, each long-text chunk with its history, the 10 s utterance, the
/// take edit): the input rows, hidden checksums at the recorded trunk and
/// tower layers, routes, the main logits, and the first 64 frames under the
/// recorded noise.
#[test]
#[ignore = "needs IZWI_FISH_DRAMA3_REFERENCE_DIR, IZWI_FISH_DRAMA3_MODEL_DIR and IZWI_FISH_DRAMA3_BACKEND (cpu: F32, about 140 GiB of RAM; cpu-q8_0: about 40 GiB; metal: a Mac with at least 64 GB); run with --release"]
fn reference_generations_match_upstream() {
    let run = reference_run();
    let (backend, tolerances, precision, device, _lease) = reference_backend();
    let (json, sidecar) = (&run.fixture.json, &run.fixture.sidecar);
    let options = model_options(
        precision,
        prefill_cell(json),
        FishDrama3ConfigPolicy::Published,
    );
    let model =
        FishDrama3Model::load_with_codec(&run.model_dir, &device, options, Arc::new(NoCodec))
            .unwrap();
    eprintln!("{}", model.diagnostics_summary());
    let mut report = Report::default();
    for case in list(&json["cases"]) {
        for (label, generation) in generations(case) {
            let started = std::time::Instant::now();
            compare_generation(
                &mut report,
                &format!("{backend} {label}"),
                generation,
                sidecar,
                &model,
                &tolerances,
            );
            eprintln!("{label}: {:.0} s", started.elapsed().as_secs_f64());
        }
    }
    report.finish(&format!("L2 generations on {backend}"));
}
