//! Real-checkpoint check of Granite Speech 5.0's retained route against its
//! atomic `generate`: each leg runs the same greedy transcription another
//! way and must emit the atomic route's token ids and text:
//! 1. one prefill span, then `decode_step`;
//! 2. engine-sized prefill spans, each quantum on a fresh scheduler-style
//!    cache view (`begin_managed_quantum`), then `decode_step`;
//! 3. two rows prepared one at a time and a third (cancelled and rolled back
//!    mid-decode) through `decode_step_batch`, the survivor finishing alone;
//! 4. the same rows prepared by the batched encoder;
//! 5. a translation request and a keyword-biased one.
//!
//! The checkpoint comes from `IZWI_GRANITE_SPEECH5_MODEL_DIR` (default: the
//! models directory's Granite Speech 5.0 2B GGUF), the clips from
//! `IZWI_GRANITE_SPEECH5_AUDIO` / `IZWI_GRANITE_SPEECH5_LONG_AUDIO`
//! (default: the repository's `data/fox.wav` and `data/test.wav`):
//! `cargo test -p izwi-core --release --lib granite_speech5_retained -- --ignored --test-threads 1`.

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;

use super::*;
use crate::backends::kv::{KvArena, KvArenaConfig, KvLayerConfig};
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};

const VARIANT: ModelVariant = ModelVariant::GraniteSpeech502BGguf;
const PAGE_TOKENS: usize = 16;
/// Room for a 30 s clip's prompt (about 350 tokens) and its 300-token
/// automatic budget.
const ROW_TOKENS: usize = 1024;

fn model_dir() -> PathBuf {
    std::env::var_os("IZWI_GRANITE_SPEECH5_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models")
                .join(VARIANT.dir_name())
        })
}

fn clip(env: &str, default: &str) -> (Vec<f32>, u32) {
    let path = std::env::var_os(env).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(default)
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    crate::runtime::audio_io::decode_audio_bytes_canonical(&bytes).unwrap()
}

fn load(profile: DeviceProfile) -> Option<GraniteSpeech5AsrModel> {
    let dir = model_dir();
    if ensure_granite_speech5_artifacts(&dir, VARIANT).is_err() {
        eprintln!("skipping: no {VARIANT} checkpoint in {}", dir.display());
        return None;
    }
    Some(GraniteSpeech5AsrModel::load(&dir, VARIANT, profile).unwrap())
}

/// One scheduler-style KV arena on the model's device, holding `rows`
/// sequences of `pages_per_row` pages each.
struct TestArena {
    arena: Arc<dyn KvArena>,
    bindings: Vec<KvLayerBinding>,
    pages_per_row: usize,
}

impl TestArena {
    fn new(model: &GraniteSpeech5AsrModel, rows: usize, tokens_per_row: usize) -> Self {
        let text = &model.config.text_config;
        let head_dim = model.config.decoder_head_dim();
        let device = model.device.device.clone();
        let bindings = (0..text.num_hidden_layers as u32)
            .map(|layer| KvLayerBinding {
                model_layer: layer,
                physical_layer: layer,
            })
            .collect::<Vec<_>>();
        let pages_per_row = tokens_per_row.div_ceil(PAGE_TOKENS);
        let config = KvArenaConfig {
            id: KvArenaId {
                model_instance: crate::engine::ModelInstanceId::new(u64::MAX - 11),
                backend: crate::backends::backend_kind_for_device(&device),
                device_ordinal: (!device.is_cpu()).then_some(0),
                generation: 1,
            },
            group: KvGroupId::new(0),
            page_tokens: PAGE_TOKENS as u32,
            capacity_pages: (rows * pages_per_row) as u32,
            growth: None,
            dtype: model.runtime.kv_dtype(),
            layers: bindings
                .iter()
                .map(|binding| KvLayerConfig {
                    binding: *binding,
                    num_kv_heads: text.num_key_value_heads as u32,
                    key_head_dim: head_dim as u32,
                    value_head_dim: head_dim as u32,
                })
                .collect(),
        };
        let arena: Arc<dyn KvArena> = match device {
            Device::Cpu => Arc::new(crate::backends::kv::CpuKvArena::new(config).unwrap()),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Arc::new(
                crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                    config,
                    device.clone(),
                )
                .unwrap(),
            ),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            _ => unreachable!("an accelerator test needs the cuda or metal feature"),
        };
        Self {
            arena,
            bindings,
            pages_per_row,
        }
    }

    /// The view of `row` a scheduler quantum hands the model: the pages
    /// covering `tokens` tokens, with `context` already written.
    fn view(&self, row: usize, context: usize, tokens: usize) -> PhysicalPagedKvCache {
        let pages = tokens.div_ceil(PAGE_TOKENS).clamp(1, self.pages_per_row);
        let blocks = (row * self.pages_per_row..row * self.pages_per_row + pages)
            .map(|index| CacheBlockRef {
                arena: self.arena.id(),
                group: self.arena.config().group,
                index: index as u32,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(self.arena.clone(), self.bindings.clone(), blocks, context)
            .unwrap()
    }
}

fn options(samples: usize, rate: u32) -> GraniteSpeech5GenerationOptions {
    GraniteSpeech5GenerationOptions {
        max_new_tokens: granite_speech5_auto_max_tokens_for_duration(samples as f32 / rate as f32),
        ..GraniteSpeech5GenerationOptions::default()
    }
}

/// The atomic route's greedy token ids and its cleaned text.
fn atomic(
    model: &GraniteSpeech5AsrModel,
    audio: &[f32],
    rate: u32,
    request: &GraniteSpeech5RequestOptions,
) -> (Vec<u32>, String) {
    let features = model.preprocessor.prepare(audio, rate).unwrap();
    let (audio_embeds, _) = model.runtime.audio_embeddings(&features).unwrap();
    let prompt = model
        .build_prompt(&request.prompt_options().unwrap())
        .unwrap();
    let options = options(audio.len(), rate);
    let arena = TestArena::new(model, 1, ROW_TOKENS);
    let mut cache = arena.view(0, 0, ROW_TOKENS);
    let special = model.prompt_tokenizer.special_tokens().clone();
    let mut decode = |ids: &[u32]| model.prompt_tokenizer.decode(ids);
    let generation = model
        .runtime
        .generate(
            &prompt,
            &special,
            &audio_embeds,
            options.max_new_tokens,
            &[],
            &[],
            &mut decode,
            &mut cache,
            &mut |_| {},
            false,
        )
        .unwrap();
    (
        generation.token_ids,
        clean_granite_speech5_output(&generation.text),
    )
}

/// Prefill `state` in `span` sized quanta, each on a fresh cache view.
fn prefill(
    model: &GraniteSpeech5AsrModel,
    arena: &TestArena,
    row: usize,
    artifact: Arc<GraniteSpeech5PreparedPromptArtifact>,
    options: GraniteSpeech5GenerationOptions,
    span: usize,
) -> GraniteSpeech5DecodeState {
    let prompt_tokens = artifact.prompt_tokens();
    let first = span.min(prompt_tokens);
    let mut state = model
        .begin_resumable_prefill_managed(artifact, options, arena.view(row, 0, first))
        .unwrap();
    let mut start = 0;
    while start < prompt_tokens {
        let end = (start + span).min(prompt_tokens);
        let mut checkpoint = (start > 0).then(|| {
            state
                .begin_managed_quantum(arena.view(row, start, end))
                .unwrap()
        });
        let complete = model
            .continue_resumable_prefill(&mut state, start, end)
            .unwrap();
        assert_eq!(complete, end == prompt_tokens);
        state.take_managed_write_completions();
        if let Some(checkpoint) = checkpoint.as_mut() {
            state.commit_managed_quantum(checkpoint).unwrap();
        }
        start = end;
    }
    state
}

/// One decode quantum on a fresh view.
fn step(
    model: &GraniteSpeech5AsrModel,
    arena: &TestArena,
    row: usize,
    state: &mut GraniteSpeech5DecodeState,
) -> GraniteSpeech5DecodeStep {
    let pos = state.sequence_position();
    let mut checkpoint = state
        .begin_managed_quantum(arena.view(row, pos, pos + 1))
        .unwrap();
    let step = model.decode_step(state).unwrap();
    state.take_managed_write_completions();
    state.commit_managed_quantum(&mut checkpoint).unwrap();
    step
}

fn finish(
    model: &GraniteSpeech5AsrModel,
    arena: &TestArena,
    row: usize,
    state: &mut GraniteSpeech5DecodeState,
) -> String {
    let mut text = String::new();
    while !state.finished() {
        text = step(model, arena, row, state).text;
    }
    text
}

fn assert_tokens(leg: &str, expected: &[u32], state: &GraniteSpeech5DecodeState) {
    let actual = state.generated_ids();
    eprintln!(
        "{leg}: {} tokens, stop {}",
        actual.len(),
        state.stop_reason()
    );
    if let Some(index) =
        (0..expected.len().max(actual.len())).find(|&i| expected.get(i) != actual.get(i))
    {
        panic!(
            "{leg} diverged from the atomic route at token {index}: expected {:?}, got {:?}",
            &expected[index..expected.len().min(index + 8)],
            &actual[index.min(actual.len())..actual.len().min(index + 8)]
        );
    }
}

fn row<'a>(
    audio: &'a [f32],
    sample_rate: u32,
    request: &'a GraniteSpeech5RequestOptions,
) -> GraniteSpeech5PreparationBatchRow<'a> {
    GraniteSpeech5PreparationBatchRow {
        audio,
        sample_rate,
        request,
    }
}

fn compare(profile: DeviceProfile) {
    let Some(model) = load(profile) else {
        return;
    };
    let (short, rate) = clip("IZWI_GRANITE_SPEECH5_AUDIO", "fox.wav");
    let (long, long_rate) = clip("IZWI_GRANITE_SPEECH5_LONG_AUDIO", "test.wav");
    assert_eq!(rate, long_rate);
    let plain = GraniteSpeech5RequestOptions::default();

    let (short_ids, short_text) = atomic(&model, &short, rate, &plain);
    eprintln!("atomic short: {} tokens: {short_text:?}", short_ids.len());
    assert!(!short_ids.is_empty());
    let (long_ids, long_text) = atomic(&model, &long, rate, &plain);
    eprintln!("atomic long: {} tokens: {long_text:?}", long_ids.len());

    let single = |audio: &[f32], request: &GraniteSpeech5RequestOptions| {
        let artifact = model
            .prepare_prompt_artifact_batch(&[row(audio, rate, request)])
            .unwrap()
            .remove(0);
        let geometry = model
            .retained_preparation_geometry(audio, rate, request)
            .unwrap();
        assert_eq!(artifact.prompt_tokens(), geometry.prompt_tokens);
        assert_eq!(artifact.audio_tokens(), geometry.audio_tokens);
        assert_eq!(
            artifact.resident_tensor_bytes().unwrap(),
            geometry.retained_device_bytes
        );
        artifact
    };

    // Leg 1: one prefill span on one view, then scalar decode.
    let arena = TestArena::new(&model, 1, ROW_TOKENS);
    let artifact = single(&short, &plain);
    let prompt_tokens = artifact.prompt_tokens();
    let mut state = prefill(
        &model,
        &arena,
        0,
        artifact,
        options(short.len(), rate),
        prompt_tokens,
    );
    let text = finish(&model, &arena, 0, &mut state);
    assert_tokens("one span", &short_ids, &state);
    assert_eq!(text, short_text);

    // Leg 2: engine-sized spans (and a ragged one) on fresh views.
    for span in [64, 37] {
        let arena = TestArena::new(&model, 1, ROW_TOKENS);
        let mut state = prefill(
            &model,
            &arena,
            0,
            single(&long, &plain),
            options(long.len(), rate),
            span,
        );
        let text = finish(&model, &arena, 0, &mut state);
        assert_tokens(&format!("{span}-token spans"), &long_ids, &state);
        assert_eq!(text, long_text);
    }

    // Legs 3 and 4: both clips and a third row, cancelled mid-decode,
    // decoded together as the executor runs a continuous batch: every row
    // begins its quantum, one batched step runs, the cancelled row rolls
    // back and leaves, the others commit; a row that finishes leaves and the
    // survivor decodes alone. Leg 3 prepares each row alone, leg 4 in one
    // encoder batch.
    for (leg, batched_encoder) in [("leg 3", false), ("leg 4", true)] {
        let audios = [&short[..], &long[..], &long[..]];
        let artifacts = if batched_encoder {
            model
                .prepare_prompt_artifact_batch(&audios.map(|audio| row(audio, rate, &plain)))
                .unwrap()
        } else {
            audios.iter().map(|audio| single(audio, &plain)).collect()
        };
        let arena = TestArena::new(&model, 3, ROW_TOKENS);
        let mut rows = artifacts
            .into_iter()
            .zip(audios)
            .enumerate()
            .map(|(row, (artifact, audio))| {
                Some(prefill(
                    &model,
                    &arena,
                    row,
                    artifact,
                    options(audio.len(), rate),
                    64,
                ))
            })
            .collect::<Vec<_>>();
        const CANCEL_AT_STEP: usize = 3;
        let mut batched_steps = 0;
        loop {
            let live = (0..rows.len())
                .filter(|&row| rows[row].as_ref().is_some_and(|state| !state.finished()))
                .collect::<Vec<_>>();
            if live.len() < 2 {
                break;
            }
            let mut checkpoints = live
                .iter()
                .map(|&row| {
                    let state = rows[row].as_mut().unwrap();
                    let pos = state.sequence_position();
                    state
                        .begin_managed_quantum(arena.view(row, pos, pos + 1))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            {
                let mut states = rows
                    .iter_mut()
                    .enumerate()
                    .filter(|(row, _)| live.contains(row))
                    .map(|(_, state)| state.as_mut().unwrap())
                    .collect::<Vec<_>>();
                model.decode_step_batch(&mut states).unwrap();
            }
            for (&row, checkpoint) in live.iter().zip(checkpoints.iter_mut()) {
                let state = rows[row].as_mut().unwrap();
                if row == 2 && batched_steps == CANCEL_AT_STEP {
                    let pos = state.sequence_position();
                    state.rollback_managed_quantum(checkpoint).unwrap();
                    assert_eq!(state.sequence_position(), pos - 1);
                    assert_eq!(state.generated_ids(), &long_ids[..CANCEL_AT_STEP]);
                    rows[row] = None;
                    continue;
                }
                state.take_managed_write_completions();
                state.commit_managed_quantum(checkpoint).unwrap();
            }
            batched_steps += 1;
        }
        eprintln!("{leg}: decode_step_batch ran {batched_steps} steps (3 rows, then 2)");
        for (row, state) in rows.iter_mut().enumerate() {
            if let Some(state) = state.as_mut() {
                finish(&model, &arena, row, state);
            }
        }
        assert!(rows[2].is_none());
        assert_tokens(
            &format!("{leg} short"),
            &short_ids,
            rows[0].as_ref().unwrap(),
        );
        assert_tokens(&format!("{leg} long"), &long_ids, rows[1].as_ref().unwrap());
    }

    // Leg 5: translation and keyword prompts ride the route too.
    for (leg, request) in [
        (
            "translate",
            GraniteSpeech5RequestOptions {
                language: Some("en".into()),
                translate_to: Some("de".into()),
                ..GraniteSpeech5RequestOptions::default()
            },
        ),
        (
            "keywords",
            GraniteSpeech5RequestOptions {
                keywords: vec!["Izwi".into(), "quick brown fox".into()],
                ..GraniteSpeech5RequestOptions::default()
            },
        ),
    ] {
        let (ids, expected_text) = atomic(&model, &short, rate, &request);
        eprintln!("atomic {leg}: {expected_text:?}");
        let arena = TestArena::new(&model, 1, ROW_TOKENS);
        let artifact = single(&short, &request);
        let prompt_tokens = artifact.prompt_tokens();
        let mut state = prefill(
            &model,
            &arena,
            0,
            artifact,
            options(short.len(), rate),
            prompt_tokens,
        );
        let text = finish(&model, &arena, 0, &mut state);
        assert_tokens(leg, &ids, &state);
        assert_eq!(text, expected_text);
    }
}

#[test]
#[ignore = "needs the Granite Speech 5.0 2B GGUF checkpoint"]
fn granite_speech5_retained_decode_matches_atomic_on_cpu() {
    compare(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the Granite Speech 5.0 2B GGUF checkpoint and a Metal device"]
fn granite_speech5_retained_decode_matches_atomic_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let Some(_lease) = crate::backends::metal_test_device_or_skip(
        "granite_speech5_retained_decode_matches_atomic_on_metal",
    ) else {
        return;
    };
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    compare(profile);
}
