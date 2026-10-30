//! The model loop on the synthetic checkpoint (§10.1), with a stub codec of
//! 2,048 samples a frame.

use std::cell::Cell;
use std::sync::{Arc, OnceLock};

use serde_json::{json, Value};

use super::super::config::tests::published;
use super::super::config::FishDrama3ConfigPolicy;
use super::super::prompt::{
    build_prompt, planned_rows, FishDrama3Codes, FishDrama3HistoryTurn, FishDrama3Reference,
};
use super::super::sampling::{FishDrama3RowSampler, FishDrama3SamplingParams};
use super::super::synthetic::SyntheticCheckpoint;
use super::super::tokenizer::{semantic_token, speaker_token};
use super::*;
use crate::kernels::numeric_check::relative_l2;

/// A stateless codec: each frame's 2,048 samples come from its codes alone.
pub(crate) struct StubCodec;

const STUB_SAMPLES: usize = 2048;

fn stub_frame(codes: &[u32]) -> impl Iterator<Item = f32> + '_ {
    let seed = codes
        .iter()
        .enumerate()
        .map(|(k, &code)| (k as u32 + 1) * code)
        .sum::<u32>() as f32;
    (0..STUB_SAMPLES).map(move |j| (seed * 0.01 + j as f32 * 1e-3).sin() * 0.5)
}

impl FishDrama3FrameCodec for StubCodec {
    fn sample_rate(&self) -> u32 {
        44_100
    }

    fn samples_per_frame(&self) -> Result<usize> {
        Ok(STUB_SAMPLES)
    }

    fn codebooks(&self) -> usize {
        10
    }

    fn stream_history_bound_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    fn stream_frames(
        &self,
        _state: &mut FishDrama3DacStreamState,
        codebooks: &[Vec<u32>],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<f32>> {
        check()?;
        let frames = codebooks[0].len();
        Ok((0..frames)
            .flat_map(|f| {
                let codes = codebooks.iter().map(|row| row[f]).collect::<Vec<_>>();
                stub_frame(&codes).collect::<Vec<_>>()
            })
            .collect())
    }

    fn stream_frames_batch(
        &self,
        states: &mut [FishDrama3DacStreamState],
        codebooks: &[Vec<Vec<u32>>],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        states
            .iter_mut()
            .zip(codebooks)
            .map(|(state, codes)| self.stream_frames(state, codes, check))
            .collect()
    }

    fn encode_reference(
        &self,
        samples: &[f32],
        _sample_rate: u32,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<FishDrama3Codes> {
        check()?;
        let frames = samples.len().div_ceil(STUB_SAMPLES);
        Ok(FishDrama3Codes::new(
            (0..10)
                .map(|k| {
                    (0..frames)
                        .map(|f| ((k * 7 + f * 13) % 64) as u32)
                        .collect()
                })
                .collect(),
        ))
    }
}

/// A WordLevel tokenizer laid out like the synthetic checkpoint's ids: pad
/// 468, `<|im_start|>` 469, eos 470, audio pad 480, `<|voice|>` 494,
/// speakers 496..=511, the semantic band 512..=575.
fn tokenizer_json() -> Value {
    let mut controls: Vec<(String, u32)> = vec![
        ("<|endoftext|>".into(), 468),
        ("<|im_start|>".into(), 469),
        ("<|im_end|>".into(), 470),
        ("<|audio_pad|>".into(), 480),
        ("<|voice|>".into(), 494),
        ("<|interleave|>".into(), 495),
    ];
    controls.extend((0..16).map(|index| (speaker_token(index), 496 + index as u32)));
    controls.extend((0..64).map(|code| (semantic_token(code), 512 + code as u32)));
    let words = [
        "system",
        "user",
        "assistant",
        "convert",
        "the",
        "provided",
        "text",
        "to",
        "speech",
        "reference",
        "following:",
        "hello",
        "world.",
        "Text:",
        "Speech:",
        "one",
        "two",
        "three",
    ];
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".into(), 0.into());
    for (index, word) in words.iter().enumerate() {
        vocab.insert((*word).into(), (index + 1).into());
    }
    for (token, id) in &controls {
        vocab.insert(token.clone(), (*id).into());
    }
    let added = controls
        .iter()
        .map(|(token, id)| {
            json!({
                "id": id, "content": token, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        })
        .collect::<Vec<_>>();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
}

/// The synthetic checkpoint with its tokenizer: a model directory the
/// family loads with [`StubCodec`] (the engine's tests use it too).
pub(crate) fn write_synthetic_model(seed: u64) -> SyntheticCheckpoint {
    let synthetic = SyntheticCheckpoint::write(seed);
    std::fs::write(
        synthetic.path().join("tokenizer.json"),
        tokenizer_json().to_string(),
    )
    .unwrap();
    synthetic
}

/// The synthetic checkpoint with its tokenizer at `dir` (a runtime's
/// `models_dir/<variant>`), marked for the lifecycle: a test loads it
/// through `RuntimeService` as it would the published one.
pub(crate) fn write_lifecycle_synthetic_model(dir: &std::path::Path, seed: u64) {
    super::super::synthetic::write_checkpoint_at(dir, seed);
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json().to_string()).unwrap();
    std::fs::write(dir.join(super::super::synthetic::LIFECYCLE_MARKER), b"").unwrap();
}

fn load_synthetic(seed: u64, cell: usize) -> FishDrama3Model {
    let synthetic = write_synthetic_model(seed);
    FishDrama3Model::load_with_codec(
        synthetic.path(),
        &Device::Cpu,
        FishDrama3ModelOptions {
            config_policy: FishDrama3ConfigPolicy::Synthetic,
            prefill_span_rows: cell,
            ..Default::default()
        },
        Arc::new(StubCodec),
    )
    .unwrap()
}

/// The same weights with a prefill grid of `cell` rows (1, 7 or 256), one
/// load each shared by every test (each load quantizes the weights).
fn model_with(cell: usize) -> &'static FishDrama3Model {
    static MODELS: [OnceLock<FishDrama3Model>; 3] = [const { OnceLock::new() }; 3];
    let slot = match cell {
        1 => 0,
        7 => 1,
        256 => 2,
        _ => panic!("no shared model with {cell}-row cells"),
    };
    MODELS[slot].get_or_init(|| load_synthetic(31, cell))
}

/// The default grid.
fn model() -> &'static FishDrama3Model {
    model_with(DEFAULT_PREFILL_SPAN_ROWS)
}

fn ok() -> Result<()> {
    Ok(())
}

fn sampling(seed: u64) -> FishDrama3SamplingParams {
    FishDrama3SamplingParams::from_request(None, None, None, seed).unwrap()
}

/// A reference of `frames` frames whose codes cover the semantic band.
fn reference(frames: usize, salt: usize) -> FishDrama3Reference {
    FishDrama3Reference {
        transcript: "hello world.".into(),
        codes: FishDrama3Codes::new(
            (0..10)
                .map(|k| {
                    (0..frames)
                        .map(|f| ((k * 11 + f * 5 + salt * 3) % 64) as u32)
                        .collect()
                })
                .collect(),
        ),
    }
}

fn prompt(text: &str, references: &[FishDrama3Reference]) -> FishDrama3Prompt {
    build_prompt(model().tokenizer(), references, [], text).unwrap()
}

/// Every piece of a request's retained state, bit for bit.
#[derive(Debug, Clone, PartialEq)]
struct Snapshot {
    prefilled: usize,
    codes: Vec<Vec<u32>>,
    stop: Option<FishDrama3StopReason>,
    last_hidden: Option<Vec<u32>>,
    last_row: Vec<u32>,
    trunk: Vec<u32>,
    position: usize,
    kv: (usize, usize),
    sampler: FishDrama3SamplerCheckpoint,
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn snapshot(state: &FishDrama3RequestState) -> Snapshot {
    Snapshot {
        prefilled: state.prefilled,
        codes: state.codes.clone(),
        stop: state.stop,
        last_hidden: state
            .last_hidden
            .as_ref()
            .map(|hidden| bits(&host_values(hidden).unwrap())),
        last_row: state.last_row.clone(),
        trunk: bits(&state.trunk.values()),
        position: state.trunk.position(),
        kv: (state.trunk_kv_len(), state.tower_kv_len()),
        sampler: state.sampler.checkpoint(),
    }
}

fn params(seed: u64, max_frames: usize) -> FishDrama3GenerationParams {
    FishDrama3GenerationParams::new(sampling(seed), max_frames)
}

/// The request prefilled by the model of `cell`-row cells.
fn prefilled(
    prompt: &FishDrama3Prompt,
    params: FishDrama3GenerationParams,
    cell: usize,
) -> FishDrama3RequestState {
    let model = model_with(cell);
    let mut state = model.new_private_request(prompt.clone(), params).unwrap();
    model.run_prefill(&mut state, &ok).unwrap();
    state
}

/// The request decoded alone to its stop.
fn solo(
    prompt: &FishDrama3Prompt,
    params: FishDrama3GenerationParams,
    cell: usize,
) -> FishDrama3RequestState {
    let mut state = prefilled(prompt, params, cell);
    model_with(cell).run_decode(&mut [&mut state], &ok).unwrap();
    state
}

/// One quantum of `state`: its next prefill cell.
fn one_cell(model: &FishDrama3Model, state: &mut FishDrama3RequestState) -> FishDrama3Step {
    let checkpoint = model.begin_quantum(state).unwrap();
    let step = model.prefill_span(state).unwrap();
    model.commit_quantum(state, &checkpoint).unwrap();
    step
}

/// A check that cancels from its `after + 1`-th call on.
fn cancel_after(after: usize) -> impl Fn() -> Result<()> {
    let calls = Cell::new(0);
    move || {
        calls.set(calls.get() + 1);
        if calls.get() > after {
            Err(Error::Cancelled("test".into()))
        } else {
            Ok(())
        }
    }
}

/// The PCM a whole decode of `codes` gives.
fn whole_pcm(codes: &FishDrama3Codes) -> Vec<f32> {
    (0..codes.frames())
        .flat_map(|f| {
            let frame = codes.codebooks.iter().map(|row| row[f]).collect::<Vec<_>>();
            stub_frame(&frame).collect::<Vec<_>>()
        })
        .collect()
}

/// A prompt of a short text after a two-frame reference: text and audio
/// rows, 30 of them.
fn short_prompt() -> FishDrama3Prompt {
    prompt("hello", &[reference(2, 0)])
}

/// A seed whose generation stops on eos after `frames` frames, under
/// `budget`.
fn eos_seed(prompt: &FishDrama3Prompt, frames: std::ops::Range<usize>, budget: usize) -> u64 {
    (0..32)
        .find(|&seed| {
            let state = solo(prompt, params(seed, budget), DEFAULT_PREFILL_SPAN_ROWS);
            state.stop == Some(FishDrama3StopReason::EndOfSpeech)
                && frames.contains(&state.frames())
        })
        .expect("a seed that stops on eos in range")
}

#[test]
fn the_synthetic_model_loads_with_its_state_geometry() {
    let model = model();
    let geometry = model.state_geometry();
    assert_eq!(geometry.trunk.layers, model.trunk().attention_layers());
    assert_eq!(geometry.tower.layers, model.tower().attention_layers());
    assert_eq!(geometry.trunk.layers, vec![3, 7]);
    assert_eq!(geometry.tower.layers, (8..21).collect::<Vec<_>>());
    let state = model
        .new_private_request(short_prompt(), params(0, 4))
        .unwrap();
    assert_eq!(geometry.tensor_bytes(), state.trunk_state().bytes());
    assert_eq!(model.max_positions(), 4_096);
    assert_eq!(model.prefill_span_rows(), DEFAULT_PREFILL_SPAN_ROWS);

    // The published geometry: 62.8 MiB of tensors and 46 KiB of BF16 KV a
    // position (×2 for F32), §4.2.
    let published = FishDrama3StateGeometry::of(&published());
    assert_eq!(
        published.trunk.layers,
        (0..10).map(|i| 4 * i + 3).collect::<Vec<_>>()
    );
    assert_eq!(published.tower.layers, (40..53).collect::<Vec<_>>());
    assert_eq!(published.recurrent_bytes, 30 * 32 * 128 * 128 * 4);
    assert_eq!(published.conv_bytes, 30 * 3 * 8192 * 4);
    assert_eq!(published.tensor_bytes(), 65_863_680);
    let bf16 = published.trunk.bytes_per_position(DType::BF16)
        + published.tower.bytes_per_position(DType::BF16);
    assert_eq!(bf16, 46 * 1024);
    assert_eq!(
        published.request_bytes(101, DType::F32),
        65_863_680 + 101 * 40 * 1024 + 100 * 52 * 1024
    );
}

fn as_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|b| f32::from_bits(*b)).collect()
}

/// Prefill grids of 1, 7 and 256 rows reach the same clocks and rows and
/// agree to F32 rounding (a span's bits depend on its split, which is why
/// the grid is pinned); one grid always gives the same bits.
/// Decode then runs to eos, and the streamed codec gives the first audio
/// after 4 frames, then 16-frame chunks, and exactly `frames × 2048`
/// samples, the whole decode's.
#[test]
fn spans_agree_and_the_stream_decodes_the_frames() {
    let model = model();
    let prompt = short_prompt();
    assert_eq!(prompt.rows, 30);
    let seed = eos_seed(&prompt, 22..120, 200);
    let expected = solo(&prompt, params(seed, 200), 256);
    assert_eq!(expected.stop, Some(FishDrama3StopReason::EndOfSpeech));
    let whole = snapshot(&prefilled(&prompt, params(seed, 200), 256));
    for span in [1, 7] {
        let mut state = prefilled(&prompt, params(seed, 200), span);
        assert_eq!(model_with(span).prefill_span_rows(), span);
        let split = snapshot(&state);
        assert_eq!(
            (split.kv, split.prefilled, split.position, &split.last_row),
            (whole.kv, whole.prefilled, whole.position, &whole.last_row)
        );
        assert_eq!(split.sampler, whole.sampler);
        let hidden = |s: &Snapshot| as_f32(s.last_hidden.as_ref().unwrap());
        let drift = relative_l2(&hidden(&split), &hidden(&whole));
        assert!(drift < 1e-5, "span {span}: h_t {drift:e}");
        let drift = relative_l2(&as_f32(&split.trunk), &as_f32(&whole.trunk));
        assert!(drift < 1e-5, "span {span}: state {drift:e}");
        if span == 7 {
            assert_eq!(snapshot(&prefilled(&prompt, params(seed, 200), 7)), split);
        }
        let steps = model_with(span).decode_round(&mut [&mut state]).unwrap();
        assert_eq!(steps, [FishDrama3Step::Frame { frames: 1 }]);
    }

    let mut chunks = Vec::new();
    let synthesis = model
        .synthesize(prompt.clone(), params(seed, 200), &ok, &mut |chunk| {
            chunks.push(chunk.len());
            Ok(())
        })
        .unwrap();
    let frames = expected.frames();
    assert_eq!(synthesis.codes.codebooks, expected.codes);
    assert_eq!(synthesis.sample_rate, 44_100);
    assert_eq!(synthesis.samples.len(), frames * 2048);
    assert_eq!(synthesis.samples, whole_pcm(&synthesis.codes));
    // 4, then 16s, then the rest after eos.
    let mut cadence = vec![4];
    let mut left = frames - 4;
    while left > 0 {
        cadence.push(left.min(16));
        left -= left.min(16);
    }
    assert_eq!(synthesis.audio_quanta, cadence);
    assert_eq!(chunks, cadence.iter().map(|f| f * 2048).collect::<Vec<_>>());
    assert_eq!(
        model
            .generate_frames(prompt, params(seed, 200), &ok)
            .unwrap()
            .codebooks,
        expected.codes
    );
}

/// The grid belongs to the load: cells are `[k·S, min((k + 1)·S, P))`
/// whoever calls, and a prompt one row past a whole cell ends in a one-row
/// cell (the GEMV path, and the last prompt row the tower leaves to decode)
/// that gives the same bits every time.
#[test]
fn the_grid_is_fixed_and_a_one_row_final_cell_repeats_its_bits() {
    let model = model_with(7);
    let prompt = prompt("hello", &[reference(1, 0)]);
    assert_eq!(prompt.rows % 7, 1);
    let mut state = model
        .new_private_request(prompt.clone(), params(9, 24))
        .unwrap();
    let mut cells = Vec::new();
    while let Some(cell) = model.prefill_cell(&state) {
        let step = one_cell(model, &mut state);
        assert_eq!(
            step,
            FishDrama3Step::Prefill {
                consumed: cell.len(),
                position: cell.end,
                complete: cell.end == prompt.rows,
            }
        );
        cells.push(cell);
    }
    let rows = prompt.rows;
    let expected = (0..rows)
        .step_by(7)
        .map(|start| start..(start + 7).min(rows))
        .collect::<Vec<_>>();
    assert_eq!(cells, expected);
    assert_eq!(cells.last().unwrap().len(), 1);
    assert_eq!(state.kv(), (rows, rows - 1));
    // Prefilled, the request has no cell left to run.
    let checkpoint = model.begin_quantum(&mut state).unwrap();
    assert!(model.prefill_span(&mut state).is_err());
    model.abort_quantum(&mut state, checkpoint).unwrap();

    let first = snapshot(&state);
    assert_eq!(snapshot(&prefilled(&prompt, params(9, 24), 7)), first);
    model.run_decode(&mut [&mut state], &ok).unwrap();
    assert_eq!(snapshot(&solo(&prompt, params(9, 24), 7)), snapshot(&state));
}

/// A request prefilled and decoded alone makes the same bits as with other
/// requests' prefill cells and decode quanta between its cells and beside
/// its frames: load never moves its grid.
#[test]
fn other_requests_between_the_cells_change_no_bit() {
    let model = model_with(7);
    let prompts = [
        short_prompt(),
        prompt("hello world. one two three", &[reference(3, 1)]),
        prompt("two", &[reference(5, 2)]),
    ];
    let budgets = [20, 12, 16];
    let request = |i: usize| params(60 + i as u64, budgets[i]);
    let solos = (0..3)
        .map(|i| snapshot(&solo(&prompts[i], request(i), 7)))
        .collect::<Vec<_>>();

    let mut a = model
        .new_private_request(prompts[0].clone(), request(0))
        .unwrap();
    let mut b = model
        .new_private_request(prompts[1].clone(), request(1))
        .unwrap();
    let mut c = prefilled(&prompts[2], request(2), 7);
    let mut cells = 0;
    while !a.is_prefilled() {
        if !b.is_prefilled() {
            one_cell(model, &mut b);
        }
        if !c.is_finished() {
            model.decode_round(&mut [&mut c]).unwrap();
        }
        one_cell(model, &mut a);
        cells += 1;
    }
    assert!(cells >= 4, "{cells}");
    model.run_prefill(&mut b, &ok).unwrap();
    model
        .run_decode(&mut [&mut c, &mut a, &mut b], &ok)
        .unwrap();
    assert_eq!(snapshot(&a), solos[0]);
    assert_eq!(snapshot(&b), solos[1]);
    assert_eq!(snapshot(&c), solos[2]);
}

impl FishDrama3RequestState {
    fn kv(&self) -> (usize, usize) {
        (self.trunk_kv_len(), self.tower_kv_len())
    }
}

/// A request decoded alone and beside requests of other lengths, in any
/// order, makes the same bits; so does the codec batch.
#[test]
fn a_row_decodes_the_same_alone_and_beside_others() {
    let model = model();
    let prompts = [
        prompt("hello", &[]),
        prompt("hello world. one two three", &[reference(3, 1)]),
        prompt("two", &[reference(5, 2), reference(1, 3)]),
    ];
    let rows = prompts.iter().map(|p| p.rows).collect::<Vec<_>>();
    assert!(rows[0] < rows[1] && rows[1] != rows[2], "{rows:?}");
    let budgets = [24, 9, 30];
    let solos = (0..3)
        .map(|i| snapshot(&solo(&prompts[i], params(40 + i as u64, budgets[i]), 256)))
        .collect::<Vec<_>>();
    for order in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
        let mut states = (0..3)
            .map(|i| prefilled(&prompts[i], params(40 + i as u64, budgets[i]), 256))
            .collect::<Vec<_>>();
        {
            let mut slots = states.iter_mut().map(Some).collect::<Vec<_>>();
            let mut batch = order
                .iter()
                .map(|&i| slots[i].take().unwrap())
                .collect::<Vec<_>>();
            model.run_decode(&mut batch, &ok).unwrap();
        }
        for (i, state) in states.iter().enumerate() {
            assert_eq!(snapshot(state), solos[i], "row {i} in order {order:?}");
        }
    }
    // A pair of rows: the third waits.
    let mut a = prefilled(&prompts[0], params(40, budgets[0]), 256);
    let mut c = prefilled(&prompts[2], params(42, budgets[2]), 256);
    model.run_decode(&mut [&mut c, &mut a], &ok).unwrap();
    assert_eq!(snapshot(&a), solos[0]);
    assert_eq!(snapshot(&c), solos[2]);

    // The codec batch equals scalar quanta, streams at different ages.
    let finished = (0..3)
        .map(|i| solo(&prompts[i], params(40 + i as u64, budgets[i]), 256))
        .collect::<Vec<_>>();
    assert!(finished.iter().all(|state| state.frames() >= 5));
    let mut streams = vec![FishDrama3AudioStream::new(); 3];
    let first = model
        .stage_audio(&finished[1], &streams[1], 4, &ok)
        .unwrap();
    streams[1].commit(first).unwrap();
    let rows = finished
        .iter()
        .zip(&streams)
        .map(|(state, stream)| (state, stream, 5))
        .collect::<Vec<_>>();
    let batch = model.stage_audio_batch(&rows, &ok).unwrap();
    for ((state, stream), candidate) in finished.iter().zip(&streams).zip(&batch) {
        let scalar = model.stage_audio(state, stream, 5, &ok).unwrap();
        assert_eq!(candidate.samples(), scalar.samples());
        assert_eq!(candidate.frames(), scalar.frames());
    }
    for (stream, candidate) in streams.iter_mut().zip(batch) {
        stream.commit(candidate).unwrap();
    }
    let decoded = streams
        .iter()
        .map(|s| s.decoded_frames())
        .collect::<Vec<_>>();
    assert_eq!(decoded, [5, finished[1].frames().min(9), 5]);
}

/// An aborted quantum, prefill or decode, restores every piece bit for
/// bit, and its replay gives the same bits. A failed batch aborts every
/// row. Stale or foreign checkpoints are refused.
#[test]
fn an_aborted_quantum_restores_the_request_exactly() {
    let model = model_with(7);
    let prompt = short_prompt();
    let mut state = model
        .new_private_request(prompt.clone(), params(3, 40))
        .unwrap();
    let before = snapshot(&state);
    let checkpoint = model.begin_quantum(&mut state).unwrap();
    assert!(model.begin_quantum(&mut state).is_err());
    let step = model.prefill_span(&mut state).unwrap();
    assert_eq!(
        step,
        FishDrama3Step::Prefill {
            consumed: 7,
            position: 7,
            complete: false
        }
    );
    let spanned = snapshot(&state);
    assert_eq!(spanned.kv, (7, 7));
    model.abort_quantum(&mut state, checkpoint).unwrap();
    assert_eq!(snapshot(&state), before);
    assert!(!state.in_quantum());
    let checkpoint = model.begin_quantum(&mut state).unwrap();
    model.prefill_span(&mut state).unwrap();
    assert_eq!(snapshot(&state), spanned);
    model.commit_quantum(&mut state, &checkpoint).unwrap();
    // A committed checkpoint is stale.
    assert!(model.abort_quantum(&mut state, checkpoint).is_err());
    model.run_prefill(&mut state, &ok).unwrap();
    assert_eq!(state.kv(), (30, 29));
    // Steps need an open quantum.
    assert!(model.decode_quantum(&mut [&mut state]).is_err());

    // Three decode quanta, the middle one aborted and replayed.
    let run = |state: &mut FishDrama3RequestState| {
        let checkpoint = model.begin_quantum(state).unwrap();
        let steps = model.decode_quantum(&mut [state]).unwrap();
        (checkpoint, steps)
    };
    let (checkpoint, _) = run(&mut state);
    model.commit_quantum(&mut state, &checkpoint).unwrap();
    let mid = snapshot(&state);
    let (checkpoint, steps) = run(&mut state);
    let decoded = snapshot(&state);
    assert_eq!(decoded.kv, (mid.kv.0 + 1, mid.kv.1 + 1));
    assert_ne!(decoded.sampler, mid.sampler);
    model.abort_quantum(&mut state, checkpoint).unwrap();
    assert_eq!(snapshot(&state), mid);
    let (checkpoint, again) = run(&mut state);
    assert_eq!(again, steps);
    assert_eq!(snapshot(&state), decoded);
    model.commit_quantum(&mut state, &checkpoint).unwrap();

    // A batch with a row that cannot decode (not prefilled) fails whole
    // and leaves the ready row as it was.
    let mut fresh = model.new_private_request(prompt, params(4, 40)).unwrap();
    let ready = snapshot(&state);
    assert!(model.decode_round(&mut [&mut state, &mut fresh]).is_err());
    assert_eq!(snapshot(&state), ready);
    assert!(!state.in_quantum() && !fresh.in_quantum());

    // Another request's checkpoint does not fit this one.
    model.run_prefill(&mut fresh, &ok).unwrap();
    let foreign = model.begin_quantum(&mut fresh).unwrap();
    let own = model.begin_quantum(&mut state).unwrap();
    assert!(model.commit_quantum(&mut state, &foreign).is_err());
    assert!(model.abort_quantum(&mut state, foreign).is_err());
    model.abort_quantum(&mut state, own).unwrap();
    assert_eq!(snapshot(&state), ready);
}

/// Cancelling between spans or between quanta stops at a committed
/// boundary; resuming finishes exactly as an uninterrupted run.
#[test]
fn cancellation_stops_between_quanta_and_the_request_resumes() {
    let model = model_with(7);
    let prompt = short_prompt();
    let budget = 12;
    let expected = solo(&prompt, params(7, budget), 7);
    assert!(expected.frames() > 3);

    let mut state = model
        .new_private_request(prompt.clone(), params(7, budget))
        .unwrap();
    let err = model.run_prefill(&mut state, &cancel_after(2)).unwrap_err();
    assert!(matches!(err, Error::Cancelled(_)), "{err}");
    assert!(!state.in_quantum());
    assert_eq!((state.prefilled(), state.kv()), (14, (14, 14)));
    model.run_prefill(&mut state, &ok).unwrap();
    let err = model
        .run_decode(&mut [&mut state], &cancel_after(3))
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled(_)), "{err}");
    assert!(!state.in_quantum());
    assert_eq!(state.frames(), 3);
    assert_eq!(state.kv(), (30 + 3, 29 + 3));
    model.run_decode(&mut [&mut state], &ok).unwrap();
    assert_eq!(snapshot(&state), snapshot(&expected));

    // The whole-generation helpers stop the same way.
    assert!(matches!(
        model.generate_frames(prompt.clone(), params(7, budget), &cancel_after(0)),
        Err(Error::Cancelled(_))
    ));
    // A codec quantum cancelled mid-push leaves its stream as it was.
    let stream = FishDrama3AudioStream::new();
    assert!(model
        .stage_audio(&expected, &stream, 4, &cancel_after(0))
        .is_err());
    assert_eq!(stream.decoded_frames(), 0);
}

/// Running out of budget is an error, not a short take; the first frame
/// never stops on eos unless the mask is off, when a zero-frame stop is
/// a finished generation.
#[test]
fn the_budget_is_an_error_and_the_first_frame_mask_holds() {
    let model = model();
    let prompt = prompt("hello", &[]);
    let ids = model.config().ids;

    // A one-frame budget whose second draw is not eos.
    let seed = (0..16)
        .find(|&seed| {
            solo(&prompt, params(seed, 1), 256).stop == Some(FishDrama3StopReason::FrameLimit)
        })
        .expect("a seed that runs past its budget");
    let state = solo(&prompt, params(seed, 1), 256);
    assert_eq!(state.frames(), 1);
    let err = model
        .generate_frames(prompt.clone(), params(seed, 1), &ok)
        .unwrap_err();
    assert!(err.to_string().contains("frame_limit"), "{err}");
    let err = model
        .synthesize(prompt.clone(), params(seed, 1), &ok, &mut |_| Ok(()))
        .unwrap_err();
    assert!(err.to_string().contains("frame_limit"), "{err}");
    assert!(model
        .finalize_audio(&state, &mut FishDrama3AudioStream::new(), &ok)
        .is_err());

    // Budgets and positions are checked before anything runs.
    for frames in [0, MAX_EXPLICIT_FRAMES + 1] {
        assert!(matches!(
            model.new_private_request(prompt.clone(), params(0, frames)),
            Err(Error::InvalidInput(_))
        ));
    }
    assert!(matches!(
        model.new_private_request(prompt.clone(), params(0, 4_096 - prompt.rows + 1)),
        Err(Error::ContextLengthExceeded(_))
    ));
    model
        .new_private_request(prompt.clone(), params(0, 4_096 - prompt.rows))
        .unwrap();

    // A seed whose first draw at a flat temperature is eos.
    let hot =
        |seed| FishDrama3SamplingParams::from_request(Some(4.0), Some(1.0), None, seed).unwrap();
    let state = prefilled(&prompt, params(0, 8), 256);
    let logits = host_values(&model.head().logits(state.last_hidden().unwrap()).unwrap()).unwrap();
    let seed = (0..5_000)
        .find(|&seed| {
            let mut sampler = FishDrama3RowSampler::new(hot(seed), ids, prompt.main_ids()).unwrap();
            sampler.sample_main(&logits, true).unwrap().token == ids.eos
        })
        .expect("a seed whose first draw is eos");
    let mask_off = FishDrama3GenerationParams {
        sampling: hot(seed),
        max_frames: 8,
        mask_first_eos: false,
    };
    let codes = model
        .generate_frames(prompt.clone(), mask_off, &ok)
        .unwrap();
    assert_eq!(codes.frames(), 0);
    assert_eq!(codes.codebooks.len(), 10);
    let synthesis = model
        .synthesize(prompt.clone(), mask_off, &ok, &mut |_| Ok(()))
        .unwrap();
    assert!(synthesis.samples.is_empty() && synthesis.audio_quanta.is_empty());
    // With the mask the same seed makes a frame.
    let mut state = prefilled(
        &prompt,
        FishDrama3GenerationParams {
            mask_first_eos: true,
            ..mask_off
        },
        256,
    );
    assert_eq!(
        model.decode_round(&mut [&mut state]).unwrap(),
        [FishDrama3Step::Frame { frames: 1 }]
    );
    assert_ne!(state.codes[0].len(), 0);
}

fn turn_codes(frames: usize, salt: usize) -> FishDrama3Codes {
    reference(frames, salt).codes
}

/// The prompt holds `codes` as consecutive audio rows.
fn holds(prompt: &FishDrama3Prompt, codes: &FishDrama3Codes) -> bool {
    let frames = codes.frames();
    (0..=prompt.rows - frames).any(|start| {
        (0..frames)
            .all(|f| (0..10).all(|k| prompt.values[1 + k][start + f] == codes.codebooks[k][f]))
    })
}

/// The session keeps the pinned first chunk and the latest one with their
/// codes, skips chunks that made no frames, drops history oldest first
/// then the pinned chunk when the prompt is too long, and refuses a chunk
/// that cannot fit at all.
#[test]
fn the_chunk_session_keeps_pinned_and_latest_history() {
    let model = model();
    let tokenizer = model.tokenizer();
    let text = "one two three hello world. ".repeat(48);
    let references = vec![reference(3, 9)];
    let mut options = FishDrama3ChunkSessionOptions::new(sampling(100));
    options.max_tokens = Some(8);
    let mut session = FishDrama3ChunkSession::new(&text, references.clone(), options).unwrap();
    assert!(session.chunks().len() >= 4, "{}", session.chunks().len());
    let made = [30, 0, 2];
    let mut plans = Vec::new();
    for (index, frames) in made.into_iter().enumerate() {
        let plan = session.plan_next(tokenizer, usize::MAX).unwrap().unwrap();
        assert_eq!(plan.index, index);
        assert_eq!(plan.params.sampling.seed, 100 + index as u64);
        assert_eq!(plan.params.max_frames, 8);
        session.complete(&plan, &turn_codes(frames, index)).unwrap();
        plans.push(plan);
    }
    let plan = session.plan_next(tokenizer, usize::MAX).unwrap().unwrap();
    // Chunk 1 made nothing, so it never joined the history.
    let histories = plans
        .iter()
        .chain([&plan])
        .map(|plan| plan.history.clone())
        .collect::<Vec<_>>();
    assert_eq!(histories, [vec![], vec![0], vec![0], vec![0, 2]]);
    assert!(holds(&plan.prompt, &turn_codes(30, 0)));
    assert!(holds(&plan.prompt, &turn_codes(2, 2)));
    assert!(holds(&plan.prompt, &references[0].codes));
    assert!(!holds(&plans[0].prompt, &turn_codes(30, 0)));

    // Too long: the latest goes first, then the pinned one; references stay.
    let turn = |index: usize, frames| FishDrama3HistoryTurn {
        text: session.chunks()[index].clone(),
        codes: turn_codes(frames, index),
    };
    let text3 = &session.chunks()[3];
    let rows = |turns: &[FishDrama3HistoryTurn]| {
        planned_rows(tokenizer, &references, turns.iter(), text3).unwrap()
    };
    let (full, pinned, latest, none) = (
        plan.prompt.rows,
        rows(&[turn(0, 30)]),
        rows(&[turn(2, 2)]),
        rows(&[]),
    );
    assert_eq!(full, rows(&[turn(0, 30), turn(2, 2)]));
    assert!(full > pinned && pinned > latest && latest > none);
    let limited = |rows: usize| session.plan_next(tokenizer, rows + 8);
    assert_eq!(limited(full - 1).unwrap().unwrap().history, [0]);
    // The latest alone would fit, but the pinned chunk goes only after it.
    assert_eq!(
        limited(pinned - 1).unwrap().unwrap().history,
        Vec::<usize>::new()
    );
    let fitted = limited(none).unwrap().unwrap();
    assert!(fitted.history.is_empty() && holds(&fitted.prompt, &references[0].codes));
    assert!(matches!(
        limited(none - 1),
        Err(Error::ContextLengthExceeded(_))
    ));
    // A chunk completes once, in order.
    assert!(session.complete(&plans[0], &turn_codes(1, 0)).is_err());

    // The whole loop over the model: every chunk its own generation, the
    // next prompt holding the codes of the last.
    let short = "hello world. ".repeat(30);
    let mut options = FishDrama3ChunkSessionOptions::new(
        FishDrama3SamplingParams::from_request(Some(2.0), None, None, 0).unwrap(),
    );
    options.max_tokens = Some(400);
    let seed = (0..8u64)
        .find(|&seed| {
            options.sampling.seed = seed;
            let mut session = FishDrama3ChunkSession::new(&short, Vec::new(), options).unwrap();
            assert_eq!(session.chunks().len(), 2);
            model.run_chunk_session(&mut session, &ok).is_ok()
        })
        .expect("a seed whose chunks all stop on eos");
    options.sampling.seed = seed;
    let mut session = FishDrama3ChunkSession::new(&short, Vec::new(), options).unwrap();
    let results = model.run_chunk_session(&mut session, &ok).unwrap();
    assert!(session.is_done());
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].history, Vec::<usize>::new());
    assert_eq!(results[1].history, [0]);
    assert!(results.iter().all(|result| result.codes.frames() > 0));
    let first = FishDrama3HistoryTurn {
        text: session.chunks()[0].clone(),
        codes: results[0].codes.clone(),
    };
    assert_eq!(
        results[1].prompt_rows,
        planned_rows(tokenizer, &[], [&first], &session.chunks()[1]).unwrap()
    );
}

/// References go through the cache (a repeat hits), at most 8 of them and
/// 60 s each.
#[test]
fn references_encode_through_the_cache() {
    let model = model();
    let samples = vec![0.25f32; 4_100];
    let audio = FishDrama3ReferenceAudio {
        samples: &samples,
        channels: 1,
        sample_rate: 44_100,
    };
    let (codes, hit) = model.encode_reference(&audio, &ok).unwrap();
    assert!(!hit);
    assert_eq!((codes.codebooks.len(), codes.frames()), (10, 3));
    let (again, hit) = model.encode_reference(&audio, &ok).unwrap();
    assert!(hit);
    assert_eq!(again, codes);
    let references = model
        .encode_references(&[("hello world.", audio), ("two", audio)], &ok)
        .unwrap();
    assert_eq!(references[1].codes, codes);
    let prompt = build_prompt(model.tokenizer(), &references, [], "hello").unwrap();
    assert!(holds(&prompt, &codes));
    let nine = vec![("hello", audio); 9];
    assert!(matches!(
        model.encode_references(&nine, &ok),
        Err(Error::InvalidInput(_))
    ));
    let long = vec![0.0f32; 61 * 100];
    let long = FishDrama3ReferenceAudio {
        samples: &long,
        channels: 1,
        sample_rate: 100,
    };
    assert!(matches!(
        model.encode_reference(&long, &ok),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        model.encode_references(&[("hello", audio)], &cancel_after(0)),
        Err(Error::Cancelled(_))
    ));
}
