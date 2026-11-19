//! The retained route on the tiny synthetic checkpoint: it must decode what
//! the atomic route decodes, alone or beside other rows, and survive a
//! rolled-back quantum.

use std::sync::Arc;

use crate::backends::DeviceProfile;
use crate::catalog::ModelVariant;
use crate::models::shared::attention::physical::{test_physical_caches_on, PhysicalPagedKvCache};

use super::retained::{RumbleDecodeState, RumblePreparedPromptArtifact};
use super::synthetic::SyntheticRumbleCheckpoint;
use super::{RumbleAsrModel, RumbleGenerationOptions, RumbleRequestOptions};

fn load(checkpoint: &SyntheticRumbleCheckpoint) -> RumbleAsrModel {
    RumbleAsrModel::load(
        checkpoint.path(),
        ModelVariant::Rumble2Flash,
        DeviceProfile::cpu(),
    )
    .expect("synthetic Rumble loads")
    // The route is off by default on CPU.
    .with_retained_decode(true)
}

fn caches(model: &RumbleAsrModel, rows: usize) -> Vec<PhysicalPagedKvCache> {
    let text = &model.config().text_config;
    test_physical_caches_on(
        &model.device().device,
        text.num_hidden_layers,
        text.num_key_value_heads,
        model.config().decoder_head_dim(),
        model.sequence_tokens(),
        rows,
    )
}

fn tone(samples: usize, pitch: f32) -> Vec<f32> {
    (0..samples)
        .map(|index| ((index as f32 * pitch).sin() + (index as f32 * 0.0071).cos()) * 0.2)
        .collect()
}

fn options(max_new_tokens: usize) -> RumbleGenerationOptions {
    RumbleGenerationOptions {
        max_new_tokens,
        ..RumbleGenerationOptions::default()
    }
}

fn request() -> RumbleRequestOptions {
    RumbleRequestOptions {
        language: Some("English".into()),
        ..RumbleRequestOptions::default()
    }
}

fn artifact(model: &RumbleAsrModel, audio: &[f32]) -> Arc<RumblePreparedPromptArtifact> {
    model
        .prepare_prompt_artifact(audio, 16_000, &request())
        .expect("artifact prepares")
}

/// Prefill `state` in spans of `span` tokens, each on the state's own cache.
fn prefill(model: &RumbleAsrModel, state: &mut RumbleDecodeState, span: usize) {
    let total = state.prefill_token_count();
    let mut start = 0;
    while start < total {
        let end = (start + span).min(total);
        let complete = model
            .continue_resumable_prefill(state, start, end)
            .expect("span prefills");
        assert_eq!(complete, end == total);
        let _ = state.take_managed_write_completions();
        start = end;
    }
    assert_eq!(state.prefill_progress(), total);
    assert_eq!(state.sequence_position(), total);
}

/// Decode `state` alone to its end; returns its final text.
fn decode_alone(model: &RumbleAsrModel, state: &mut RumbleDecodeState) -> String {
    let mut streamed = String::new();
    loop {
        let step = model.decode_step(state).expect("step decodes");
        let _ = state.take_managed_write_completions();
        streamed.push_str(&step.delta);
        if step.finished {
            assert_eq!(streamed.trim(), step.text, "deltas are the transcript");
            return step.text;
        }
    }
}

fn atomic(model: &RumbleAsrModel, audio: &[f32], max_new_tokens: usize) -> (String, usize) {
    let mut cache = caches(model, 1).pop().unwrap();
    let transcription = model
        .transcribe_physical(
            audio,
            16_000,
            &request(),
            options(max_new_tokens),
            &mut cache,
            None,
        )
        .expect("atomic route transcribes");
    let generated = transcription.diagnostics.unwrap()["generated_tokens"]
        .as_u64()
        .unwrap() as usize;
    (transcription.text, generated)
}

#[test]
fn the_retained_route_decodes_what_the_atomic_route_decodes() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    assert!(model.supports_resumable_prefill());
    assert!(model.supports_incremental_decode());
    assert!(model.supports_continuous_decode_batch());
    let audio = tone(24_000, 0.031);
    for budget in [1, 5, 14] {
        let (atomic_text, atomic_generated) = atomic(&model, &audio, budget);
        let prepared = artifact(&model, &audio);
        // One prefill span, as the atomic route prefills.
        let mut state = model
            .begin_resumable_prefill_managed(
                prepared.clone(),
                options(budget),
                caches(&model, 1).pop().unwrap(),
            )
            .unwrap();
        prefill(&model, &mut state, usize::MAX / 2);
        let text = decode_alone(&model, &mut state);
        assert_eq!(text, atomic_text, "budget {budget}");
        assert_eq!(
            state.generated_ids().len(),
            atomic_generated,
            "budget {budget}"
        );
        assert_eq!(state.language(), Some("English"));
        assert!(state.finished());
        let one_span = state.generated_ids().to_vec();
        let diagnostics = state.retained_terminal_diagnostics();
        assert_eq!(diagnostics["language"], "English");
        assert_eq!(diagnostics["generated_tokens"], atomic_generated);

        // Resumed in short spans, the row stages the same tokens.
        let mut state = model
            .begin_resumable_prefill_managed(
                prepared,
                options(budget),
                caches(&model, 1).pop().unwrap(),
            )
            .unwrap();
        prefill(&model, &mut state, 7);
        assert_eq!(decode_alone(&model, &mut state), atomic_text);
        assert_eq!(state.generated_ids(), one_span, "budget {budget}");
    }
}

/// Every row decodes the same alone as beside rows of other lengths and
/// budgets, and a batch keeps going when one row ends first.
#[test]
fn rows_decode_the_same_alone_and_batched() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let clips = [
        (tone(16_000, 0.031), 9),
        (tone(40_000, 0.012), 4),
        (tone(8_000, 0.07), 12),
    ];
    let alone = clips
        .iter()
        .map(|(audio, budget)| {
            let mut state = model
                .begin_resumable_prefill_managed(
                    artifact(&model, audio),
                    options(*budget),
                    caches(&model, 1).pop().unwrap(),
                )
                .unwrap();
            prefill(&model, &mut state, usize::MAX / 2);
            let text = decode_alone(&model, &mut state);
            (state.generated_ids().to_vec(), text)
        })
        .collect::<Vec<_>>();

    let mut states = clips
        .iter()
        .zip(caches(&model, clips.len()))
        .map(|((audio, budget), cache)| {
            let mut state = model
                .begin_resumable_prefill_managed(artifact(&model, audio), options(*budget), cache)
                .unwrap();
            prefill(&model, &mut state, usize::MAX / 2);
            state
        })
        .collect::<Vec<_>>();
    let mut texts = vec![None; states.len()];
    let mut widths = Vec::new();
    loop {
        let mut live = states
            .iter_mut()
            .enumerate()
            .filter(|(_, state)| !state.finished())
            .collect::<Vec<_>>();
        if live.is_empty() {
            break;
        }
        widths.push(live.len());
        let indices = live.iter().map(|(index, _)| *index).collect::<Vec<_>>();
        let mut rows = live
            .iter_mut()
            .map(|(_, state)| &mut **state)
            .collect::<Vec<_>>();
        let steps = model.decode_step_batch(&mut rows).expect("batch decodes");
        for (index, step) in indices.into_iter().zip(steps) {
            if step.finished {
                texts[index] = Some(step.text);
            }
        }
        for state in states.iter_mut() {
            let _ = state.take_managed_write_completions();
        }
    }
    assert!(widths.contains(&3) && widths.contains(&1), "{widths:?}");
    for (index, (ids, text)) in alone.iter().enumerate() {
        assert_eq!(states[index].generated_ids(), ids, "row {index}");
        assert_eq!(texts[index].as_deref(), Some(text.as_str()), "row {index}");
    }
    // The clips are different audio, so their rows are not alike.
    assert_ne!(alone[0].0, alone[2].0);
}

#[test]
fn misuse_of_a_retained_row_is_refused() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let other_checkpoint = SyntheticRumbleCheckpoint::write();
    let other = load(&other_checkpoint);
    let prepared = artifact(&model, &tone(16_000, 0.031));
    // Another loaded model's artifact.
    assert!(other
        .begin_resumable_prefill_managed(
            prepared.clone(),
            options(4),
            caches(&other, 1).pop().unwrap()
        )
        .is_err());
    let mut state = model
        .begin_resumable_prefill_managed(prepared, options(4), caches(&model, 1).pop().unwrap())
        .unwrap();
    let total = state.prefill_token_count();
    // Spans must be monotonic and in range; decode needs a finished prefill.
    assert!(model.continue_resumable_prefill(&mut state, 3, 9).is_err());
    assert!(model
        .continue_resumable_prefill(&mut state, 0, total + 1)
        .is_err());
    assert!(model.decode_step_batch(&mut [&mut state]).is_err());
    assert!(model.continue_resumable_prefill(&mut state, 0, 5).is_ok());
    assert!(model.continue_resumable_prefill(&mut state, 0, 5).is_err());

    // The budget is narrowed to the room one sequence leaves, and a request
    // past it does not fit the retained route.
    assert!(RumbleAsrModel::retained_sequence_fits(1_024, 512));
    assert!(!RumbleAsrModel::retained_sequence_fits(1_025, 512));
    assert!(!RumbleAsrModel::retained_sequence_fits(100, usize::MAX));
    let seal = model
        .scalar_preparation_stage_seal(crate::backends::BackendKind::Cpu)
        .unwrap();
    assert_eq!(seal.max_work_units, 390);
    assert!(seal.max_workspace_bytes > 0);
    assert!(model
        .scalar_preparation_stage_seal(crate::backends::BackendKind::Metal)
        .is_err());
    let geometry = model
        .retained_preparation_geometry(&tone(16_000, 0.031), 16_000, &request())
        .unwrap();
    assert_eq!(geometry.audio_tokens, 13);
    assert_eq!(geometry.mel_frames, 100);
    assert_eq!(geometry.prompt_tokens, total);
    assert!(geometry.preparation_workspace_bytes <= seal.max_workspace_bytes);
    assert!(model.continuous_decode_workspace_per_row_bytes().unwrap() > 0);
}

/// The decoder's CUDA decode graphs belong to the retained route and to
/// CUDA: a CPU load reports why it has none.
#[test]
fn decode_graphs_stay_off_off_cuda() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = RumbleAsrModel::load(
        checkpoint.path(),
        ModelVariant::Rumble2Flash,
        DeviceProfile::cpu(),
    )
    .expect("synthetic Rumble loads");
    let off = model.decode_graph_diagnostics()["off"]
        .as_str()
        .expect("a CPU load has no graphs")
        .to_string();
    if model.retained_decode() {
        // The switch forced the route on: the graphs were asked for.
        assert!(off.contains("CUDA only") || off.contains("IZWI_"), "{off}");
    } else {
        assert!(off.contains("not enabled"), "{off}");
    }
    let registry = crate::models::registry::NativeAsrModel::Rumble(model);
    assert!(registry.decode_graph_diagnostics().is_some());
}

/// Requests encoded in one tower pass get the artifacts they get alone:
/// the same prompt, the same tokens on decode, and embeddings equal up to
/// the tower's rounding.
#[test]
fn artifacts_encoded_together_match_those_encoded_alone() {
    let checkpoint = SyntheticRumbleCheckpoint::write();
    let model = load(&checkpoint);
    let clips = [tone(16_000, 0.031), tone(40_000, 0.012), tone(8_000, 0.07)];
    let requests = [
        request(),
        RumbleRequestOptions::default(),
        RumbleRequestOptions {
            prompt: Some("Izwi".into()),
            ..request()
        },
    ];
    let features = clips
        .iter()
        .map(|audio| model.prepare_features(audio, 16_000).unwrap())
        .collect::<Vec<_>>();
    let rows = features.iter().zip(&requests).collect::<Vec<_>>();
    let together = model
        .prepare_prompt_artifact_batch_from_features(&rows)
        .expect("batch encodes");
    assert_eq!(together.len(), rows.len());
    for (index, (features, request)) in rows.iter().enumerate() {
        let alone = model
            .prepare_prompt_artifact_from_features(features, request)
            .unwrap();
        let batched = &together[index];
        assert_eq!(
            batched.prompt_tokens(),
            alone.prompt_tokens(),
            "row {index}"
        );
        assert_eq!(batched.audio_tokens(), alone.audio_tokens(), "row {index}");
        assert_eq!(
            batched.resident_tensor_bytes().unwrap(),
            alone.resident_tensor_bytes().unwrap()
        );
        let values = |artifact: &RumblePreparedPromptArtifact| {
            artifact
                .embeddings
                .to_dtype(candle_core::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (batched_values, alone_values) = (values(batched), values(&alone));
        assert_eq!(batched_values.len(), alone_values.len());
        for (lhs, rhs) in batched_values.iter().zip(&alone_values) {
            assert!((lhs - rhs).abs() <= 1e-4, "row {index}: {lhs} vs {rhs}");
        }
        let decode = |artifact: Arc<RumblePreparedPromptArtifact>| {
            let mut state = model
                .begin_resumable_prefill_managed(
                    artifact,
                    options(8),
                    caches(&model, 1).pop().unwrap(),
                )
                .unwrap();
            prefill(&model, &mut state, usize::MAX / 2);
            while !model
                .decode_step(&mut state)
                .expect("step decodes")
                .finished
            {
                let _ = state.take_managed_write_completions();
            }
            state.generated_ids().to_vec()
        };
        assert_eq!(decode(batched.clone()), decode(alone), "row {index}");
    }
    assert!(model
        .prepare_prompt_artifact_batch_from_features(&[])
        .is_err());
}
