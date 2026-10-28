//! Real-checkpoint parity for LFM3-1.7B chat (QAD-Q4_0 GGUF) against the
//! independent F32 reference in `tests/fixtures/lfm3_chat_reference.json`
//! (`scripts/fixtures/generate_lfm3_reference.py`).
//!
//! The tests are ignored because they need the published GGUF. They look in
//! `IZWI_LFM25_MODELS_DIR`, else the default models directory, and skip when
//! `LFM3-1.7B-GGUF` is missing. Each backend leg checks:
//! * prompt ids equal the checkpoint chat template's rendering (thinking on
//!   and off, preserved and stripped reasoning, tools, tool round trips);
//! * one-shot and two-span prefill agree, the spans cut inside the 256-token
//!   sliding window of a 1,042-token prompt;
//! * teacher-forced decode, scalar and two-row batched, tracks the reference
//!   top-8 logits, and the argmax equals the reference token wherever the
//!   reference is not a near tie.
//!
//! Run: `cargo test -p izwi-core --features metal --lib lfm2::chat::lfm3_reference_tests -- --ignored --nocapture`
use super::reference_tests::{
    abs_err, messages, models_dir, prefill, row, FixtureStep, Sessions, StepStats, Tolerance,
};
use super::*;
use candle_core::{DType, IndexOp};
use std::path::PathBuf;

/// 1,280 tokens per session: the 1,042-token prompt plus its decode.
const PAGES_PER_SESSION: u32 = 80;
const FOLDER: &str = "LFM3-1.7B-GGUF";

#[derive(serde::Deserialize)]
struct Fixture {
    template_cases: std::collections::BTreeMap<String, TemplateCase>,
    prompts: std::collections::BTreeMap<String, Prompt>,
}

#[derive(serde::Deserialize)]
struct TemplateCase {
    messages: Vec<(String, String)>,
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    #[serde(default)]
    template_kwargs: TemplateKwargs,
    prompt_ids: Vec<u32>,
}

#[derive(serde::Deserialize, Default)]
struct TemplateKwargs {
    enable_thinking: Option<bool>,
    preserve_thinking: Option<bool>,
}

#[derive(serde::Deserialize)]
struct Prompt {
    messages: Vec<(String, String)>,
    tools: Vec<serde_json::Value>,
    enable_thinking: bool,
    prompt_ids: Vec<u32>,
    steps: Vec<FixtureStep>,
}

fn fixture() -> Fixture {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lfm3_chat_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn is_eos(token: u32) -> bool {
    matches!(token, 124_900 | 124_895)
}

fn request(
    tools: &[serde_json::Value],
    enable_thinking: Option<bool>,
    preserve_thinking: Option<bool>,
) -> ChatRequestConfig {
    ChatRequestConfig {
        tools: tools.to_vec(),
        enable_thinking,
        preserve_thinking,
        ..ChatRequestConfig::default()
    }
}

fn run_leg(profile: DeviceProfile, tol: Tolerance) {
    let dir = models_dir().join(FOLDER);
    if !dir.exists() {
        eprintln!("skipping: {} is missing", dir.display());
        return;
    }
    let fixture = fixture();
    let model = Lfm2ChatModel::load(&dir, ModelVariant::Lfm317BGguf, profile).unwrap();
    assert_eq!(
        model.max_context_tokens().unwrap(),
        32_768,
        "default context"
    );
    let features = model.config.lfm3.as_ref().expect("LFM3 trunk features");
    assert!(features.attention_output_gate && features.sandwich_norm);
    // 6 x 2048 + 3 x 7680 F32 per prefill token; a decode row adds the
    // 128,000-entry logits row.
    assert_eq!(model.prefill_workspace_per_token_bytes().unwrap(), 141_312);

    for (name, case) in &fixture.template_cases {
        let ids = model
            .prompt_token_ids_for_request(
                &messages(&case.messages),
                &request(
                    &case.tools,
                    case.template_kwargs.enable_thinking,
                    case.template_kwargs.preserve_thinking,
                ),
            )
            .unwrap();
        assert_eq!(ids, case.prompt_ids, "{name}: template prompt ids");
    }

    let dtype = match super::lfm2_chat_kv_dtype(model.device_kind()) {
        crate::kv::v2::StateDType::F16 => DType::F16,
        _ => DType::F32,
    };
    let sessions = Sessions::sized(&model, dtype, PAGES_PER_SESSION);
    let device = model.device.device.clone();
    let backend = format!("{:?}", model.device_kind());
    let prompts: Vec<(&String, &Prompt)> = fixture.prompts.iter().collect();

    // Scalar path: one-shot prefill, then one token per forward.
    let mut first_rows = Vec::new();
    for (index, (name, prompt)) in prompts.iter().enumerate() {
        let label = format!("{backend}/{name}/scalar");
        let ids = model
            .prompt_token_ids_for_request(
                &messages(&prompt.messages),
                &request(&prompt.tools, Some(prompt.enable_thinking), None),
            )
            .unwrap();
        assert_eq!(ids, prompt.prompt_ids, "{label}: prompt ids");
        let mut state = prefill(&model, &ids, sessions.session(index as u32), &[]);
        let mut logits = state.unconsumed_output.take().unwrap();
        let mut stats = StepStats::default();
        for (step, reference) in prompt.steps.iter().enumerate() {
            let values = row(&logits);
            if step == 0 {
                first_rows.push(values.clone());
            }
            stats.check(&label, step, &values, reference);
            if is_eos(reference.next) {
                break;
            }
            let input = Tensor::from_slice(&[reference.next], (1, 1), &device).unwrap();
            logits = model
                .text_model
                .forward_tokens_retained(
                    &input,
                    ids.len() + step,
                    &mut state.physical_kv,
                    &mut state.shortconv,
                    true,
                )
                .unwrap()
                .unwrap();
        }
        stats.assert_within(&label, &tol);
    }

    // Batched path: a two-span prefill per prompt (cut at 300 tokens, inside
    // the long prompt's sliding window, or at 5), then every prompt decodes
    // in one batch, as continuous batching runs them.
    let mut states = Vec::new();
    let mut pending = Vec::new();
    for (index, (name, prompt)) in prompts.iter().enumerate() {
        let cut = if prompt.prompt_ids.len() > 600 {
            300
        } else {
            5
        };
        let mut state = prefill(
            &model,
            &prompt.prompt_ids,
            sessions.session((prompts.len() + index) as u32),
            &[cut],
        );
        let values = row(&state.unconsumed_output.take().unwrap());
        let drift = values
            .iter()
            .zip(&first_rows[index])
            .map(|(lhs, rhs)| abs_err(*lhs, *rhs))
            .fold(0f32, f32::max);
        eprintln!("{backend}/{name}: two-span prefill drift {drift:.2e}");
        assert!(
            drift <= tol.chunked,
            "{backend}/{name}: two-span prefill drift {drift}"
        );
        state.position = prompt.prompt_ids.len();
        states.push(state);
        pending.push(prompt.steps[0].next);
    }
    let mut stats: Vec<StepStats> = prompts.iter().map(|_| StepStats::default()).collect();
    let mut live: Vec<bool> = pending.iter().map(|token| !is_eos(*token)).collect();
    for step in 1.. {
        let rows: Vec<usize> = (0..prompts.len())
            .filter(|&index| live[index] && step < prompts[index].1.steps.len())
            .collect();
        if rows.is_empty() {
            break;
        }
        let tokens: Vec<u32> = rows.iter().map(|&index| pending[index]).collect();
        let positions: Vec<usize> = rows.iter().map(|&index| states[index].position).collect();
        let mut shortconv = Vec::new();
        let mut caches = Vec::new();
        for (index, state) in states.iter_mut().enumerate() {
            if rows.contains(&index) {
                shortconv.push(&mut state.shortconv);
                caches.push(&mut state.physical_kv);
            }
        }
        let logits = model
            .text_model
            .forward_token_ids_retained_batch(&tokens, &positions, &mut shortconv, &mut caches)
            .unwrap();
        drop(shortconv);
        drop(caches);
        for (slot, &index) in rows.iter().enumerate() {
            let (name, prompt) = prompts[index];
            let label = format!("{backend}/{name}/batched");
            let values = row(&logits.i(slot).unwrap());
            stats[index].check(&label, step, &values, &prompt.steps[step]);
            pending[index] = prompt.steps[step].next;
            live[index] = !is_eos(pending[index]);
            states[index].position += 1;
        }
    }
    for (index, (name, _)) in prompts.iter().enumerate() {
        stats[index].assert_within(&format!("{backend}/{name}/batched"), &tol);
    }
}

#[test]
#[ignore = "needs the LFM3-1.7B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_chat_matches_the_f32_reference_on_cpu() {
    // CPU QMatMul quantizes activations to Q8 for Q4_0 weights.
    run_leg(
        DeviceProfile::cpu(),
        Tolerance {
            logit: 1.5,
            chunked: 1e-3,
        },
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM3-1.7B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_chat_matches_the_f32_reference_on_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    use candle_core::Device;
    let _lease =
        crate::backends::metal_test_device_or_skip("lfm3_chat_matches_the_f32_reference_on_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    // Metal mul_mv keeps F32 activations: measured worst error 0.0076.
    run_leg(
        profile,
        Tolerance {
            logit: 0.05,
            chunked: 2e-2,
        },
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the LFM3-1.7B GGUF (IZWI_LFM25_MODELS_DIR)"]
fn lfm3_chat_matches_the_f32_reference_on_cuda() {
    use crate::backends::{BackendPreference, DeviceSelector};
    use candle_core::Device;
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Cuda).unwrap();
    if !matches!(profile.device, Device::Cuda(_)) {
        eprintln!("skipping: no CUDA device");
        return;
    }
    run_leg(
        profile,
        Tolerance {
            logit: 1.5,
            chunked: 2e-2,
        },
    );
}
