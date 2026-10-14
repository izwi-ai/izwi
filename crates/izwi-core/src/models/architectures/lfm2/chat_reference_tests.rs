//! Real-checkpoint parity for LFM2.5-1.2B chat (Instruct and Thinking GGUF)
//! against the independent F32 reference in
//! `tests/fixtures/lfm25_chat_reference.json`
//! (`scripts/fixtures/generate_lfm25_reference.py`).
//!
//! The tests are ignored because they need the published GGUFs. They look in
//! `IZWI_LFM25_MODELS_DIR`, else the default models directory, and skip a
//! model whose folder is missing. Each backend leg checks:
//! * prompt ids equal the checkpoint chat template's rendering, including
//!   whitespace, empty-turn and past-thinking cases;
//! * one-shot and two-span (5 + rest) prefill agree;
//! * teacher-forced decode, scalar and two-row batched, tracks the reference
//!   top-8 logits, and the argmax equals the reference token wherever the
//!   reference is not a near tie.
//!
//! Run: `cargo test -p izwi-core --features metal --lib lfm2::chat::reference_tests -- --ignored --nocapture`
use super::*;
use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::{BackendPreference, DeviceProfile, DeviceSelector};
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::architectures::lfm2::physical::Lfm2StateLayout;
use candle_core::Device;
use std::path::PathBuf;
use std::sync::Arc;

const PAGE_TOKENS: u32 = 16;
const PAGES_PER_SESSION: u32 = 16;
const SESSIONS: u32 = 4;
/// Reference steps whose top-2 margin is below this are near ties; any
/// quantized backend may legitimately pick either token there.
const NEAR_TIE: f32 = 0.5;

struct Tolerance {
    /// Max |izwi - reference| over each step's reference top-8 logits.
    logit: f32,
    /// Max |one-shot - two-span| over the prefill logits.
    chunked: f32,
}

#[derive(serde::Deserialize)]
struct Fixture {
    models: std::collections::BTreeMap<String, FixtureModel>,
}

#[derive(serde::Deserialize)]
struct FixtureModel {
    folder: String,
    prompts: std::collections::BTreeMap<String, FixturePrompt>,
    /// Prompt-id-only cases rendered by the checkpoint's chat template.
    template_cases: std::collections::BTreeMap<String, FixtureTemplateCase>,
}

#[derive(serde::Deserialize)]
struct FixtureTemplateCase {
    messages: Vec<(String, String)>,
    prompt_ids: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct FixturePrompt {
    messages: Vec<(String, String)>,
    prompt_ids: Vec<u32>,
    steps: Vec<FixtureStep>,
}

#[derive(serde::Deserialize)]
struct FixtureStep {
    next: u32,
    margin: f32,
    top: Vec<(u32, f32)>,
}

fn fixture() -> Fixture {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lfm25_chat_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn models_dir() -> PathBuf {
    std::env::var_os("IZWI_LFM25_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models")
        })
}

fn variant(key: &str) -> ModelVariant {
    match key {
        "instruct" => ModelVariant::Lfm2512BInstructGguf,
        "thinking" => ModelVariant::Lfm2512BThinkingGguf,
        other => panic!("unknown fixture model {other}"),
    }
}

fn messages(raw: &[(String, String)]) -> Vec<ChatMessage> {
    raw.iter()
        .map(|(role, content)| ChatMessage {
            role: match role.as_str() {
                "system" => ChatRole::System,
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                other => panic!("unknown fixture role {other}"),
            },
            content: content.clone(),
        })
        .collect()
}

struct Sessions {
    arena: Arc<dyn KvArena>,
    id: KvArenaId,
    group: KvGroupId,
    bindings: Vec<KvLayerBinding>,
}

impl Sessions {
    fn new(model: &Lfm2ChatModel) -> Self {
        let layout = Lfm2StateLayout::from_config(&model.config).unwrap();
        let bindings: Vec<KvLayerBinding> = layout
            .attention_model_layers
            .iter()
            .enumerate()
            .map(|(physical, model_layer)| KvLayerBinding {
                model_layer: *model_layer,
                physical_layer: physical as u32,
            })
            .collect();
        let device = &model.device.device;
        let backend = model.device_kind();
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(2512),
            backend,
            device_ordinal: (!device.is_cpu()).then_some(0),
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let head_dim = model.config.embedding_length / model.config.attention_head_count;
        let config = KvArenaConfig {
            id,
            group,
            page_tokens: PAGE_TOKENS,
            capacity_pages: PAGES_PER_SESSION * SESSIONS,
            growth: None,
            dtype: DType::F32,
            layers: bindings
                .iter()
                .map(|binding| KvLayerConfig {
                    binding: *binding,
                    num_kv_heads: model.config.attention_head_count_kv[binding.model_layer as usize]
                        as u32,
                    key_head_dim: head_dim as u32,
                    value_head_dim: head_dim as u32,
                })
                .collect(),
        };
        let arena: Arc<dyn KvArena> = match backend {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            BackendKind::Metal | BackendKind::Cuda => Arc::new(
                crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                    config,
                    device.clone(),
                )
                .unwrap(),
            ),
            _ => Arc::new(CpuKvArena::new(config).unwrap()),
        };
        Self {
            arena,
            id,
            group,
            bindings,
        }
    }

    fn session(&self, index: u32) -> PhysicalPagedKvCache {
        assert!(index < SESSIONS);
        let blocks = (index * PAGES_PER_SESSION..(index + 1) * PAGES_PER_SESSION)
            .map(|block| CacheBlockRef {
                arena: self.id,
                group: self.group,
                index: block,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(self.arena.clone(), self.bindings.clone(), blocks, 0).unwrap()
    }
}

fn row(logits: &Tensor) -> Vec<f32> {
    logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn argmax_of(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}

/// Non-finite values saturate to infinity, so they can never pass.
fn abs_err(lhs: f32, rhs: f32) -> f32 {
    let diff = (lhs - rhs).abs();
    if diff.is_finite() {
        diff
    } else {
        f32::INFINITY
    }
}

#[derive(Default)]
struct StepStats {
    steps: usize,
    decisive: usize,
    agreed: usize,
    worst_logit: f32,
}

impl StepStats {
    fn check(&mut self, label: &str, step: usize, values: &[f32], reference: &FixtureStep) {
        self.steps += 1;
        for (id, expected) in &reference.top {
            let err = abs_err(values[*id as usize], *expected);
            self.worst_logit = self.worst_logit.max(err);
        }
        let chosen = argmax_of(values);
        if reference.margin >= NEAR_TIE {
            self.decisive += 1;
            if chosen == reference.next {
                self.agreed += 1;
            } else {
                eprintln!(
                    "{label}: step {step} argmax {chosen} != reference {} (margin {:.3})",
                    reference.next, reference.margin
                );
            }
        }
    }

    fn assert_within(&self, label: &str, tolerance: &Tolerance) {
        eprintln!(
            "{label}: {} steps, {}/{} decisive argmax agree, worst top-8 logit err {:.4}",
            self.steps, self.agreed, self.decisive, self.worst_logit
        );
        assert!(self.steps > 0, "{label}: no steps checked");
        assert_eq!(
            self.agreed, self.decisive,
            "{label}: argmax disagrees with a decisive reference step"
        );
        assert!(
            self.worst_logit <= tolerance.logit,
            "{label}: top-8 logit error {} exceeds {}",
            self.worst_logit,
            tolerance.logit
        );
    }
}

fn is_eos(token: u32) -> bool {
    matches!(token, 2 | 7)
}

fn prefill(
    model: &Lfm2ChatModel,
    ids: &[u32],
    cache: PhysicalPagedKvCache,
    spans: &[usize],
) -> ChatDecodeState {
    let config = ChatGenerationConfig::default();
    let mut state = model
        .begin_resumable_prefill_state_managed(ids, 64, &config, cache)
        .unwrap();
    let mut start = 0;
    for end in spans.iter().copied().chain(std::iter::once(ids.len())) {
        model
            .continue_resumable_prefill_managed(&mut state, ids, start, end)
            .unwrap();
        start = end;
    }
    state
}

fn run_model(profile: &DeviceProfile, key: &str, model_fixture: &FixtureModel, tol: &Tolerance) {
    let dir = models_dir().join(&model_fixture.folder);
    if !dir.exists() {
        eprintln!("skipping {key}: {} is missing", dir.display());
        return;
    }
    let model = Lfm2ChatModel::load(&dir, variant(key), profile.clone()).unwrap();
    assert_eq!(model.max_context_tokens().unwrap(), 32_768, "{key}: model-card context");
    for (name, case) in &model_fixture.template_cases {
        let ids = model.prompt_token_ids(&messages(&case.messages)).unwrap();
        assert_eq!(ids, case.prompt_ids, "{key}/{name}: template prompt ids");
    }
    let sessions = Sessions::new(&model);
    let device = model.device.device.clone();
    let backend = format!("{:?}", model.device_kind());
    let prompts: Vec<(&String, &FixturePrompt)> = model_fixture.prompts.iter().collect();

    // Scalar path: one-shot prefill, then one token per forward.
    let mut first_rows = Vec::new();
    for (index, (name, prompt)) in prompts.iter().enumerate() {
        let label = format!("{backend}/{key}/{name}/scalar");
        let ids = model.prompt_token_ids(&messages(&prompt.messages)).unwrap();
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
        stats.assert_within(&label, tol);
    }

    // Batched path: two-span prefill per prompt, then both prompts decode
    // together as one two-row batch, as continuous batching runs them.
    let mut states = Vec::new();
    let mut pending = Vec::new();
    for (index, (name, prompt)) in prompts.iter().enumerate() {
        let mut state = prefill(
            &model,
            &prompt.prompt_ids,
            sessions.session((prompts.len() + index) as u32),
            &[5],
        );
        let values = row(&state.unconsumed_output.take().unwrap());
        let drift = values
            .iter()
            .zip(&first_rows[index])
            .map(|(lhs, rhs)| abs_err(*lhs, *rhs))
            .fold(0f32, f32::max);
        eprintln!("{backend}/{key}/{name}: two-span prefill drift {drift:.2e}");
        assert!(
            drift <= tol.chunked,
            "{backend}/{key}/{name}: two-span prefill drift {drift}"
        );
        state.position = prompt.prompt_ids.len();
        states.push(state);
        pending.push(prompt.steps[0].next);
    }
    let batch_steps = prompts
        .iter()
        .map(|(_, prompt)| prompt.steps.len())
        .min()
        .unwrap();
    let mut stats: Vec<StepStats> = prompts.iter().map(|_| StepStats::default()).collect();
    for step in 1..batch_steps {
        if pending.iter().any(|token| is_eos(*token)) {
            break;
        }
        let positions: Vec<usize> = states.iter().map(|state| state.position).collect();
        let mut shortconv = Vec::new();
        let mut caches = Vec::new();
        for state in states.iter_mut() {
            shortconv.push(&mut state.shortconv);
            caches.push(&mut state.physical_kv);
        }
        let logits = model
            .text_model
            .forward_token_ids_retained_batch(&pending, &positions, &mut shortconv, &mut caches)
            .unwrap();
        drop(shortconv);
        drop(caches);
        for (index, (name, prompt)) in prompts.iter().enumerate() {
            let label = format!("{backend}/{key}/{name}/batched");
            let values = row(&logits.i(index).unwrap());
            stats[index].check(&label, step, &values, &prompt.steps[step]);
            pending[index] = prompt.steps[step].next;
            states[index].position += 1;
        }
    }
    for (index, (name, _)) in prompts.iter().enumerate() {
        stats[index].assert_within(&format!("{backend}/{key}/{name}/batched"), tol);
    }
}

fn run_leg(profile: DeviceProfile, tol: Tolerance) {
    let fixture = fixture();
    for (key, model_fixture) in &fixture.models {
        run_model(&profile, key, model_fixture, &tol);
    }
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUFs (IZWI_LFM25_MODELS_DIR)"]
fn lfm25_chat_matches_the_f32_reference_on_cpu() {
    // CPU QMatMul quantizes activations to Q8_K (as llama.cpp does), which
    // moves logits by up to ~1.0 (measured 0.97 on Thinking).
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
#[ignore = "needs the LFM2.5-1.2B GGUFs (IZWI_LFM25_MODELS_DIR)"]
fn lfm25_chat_matches_the_f32_reference_on_metal() {
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !matches!(profile.device, Device::Metal(_)) {
        eprintln!("skipping: no Metal device");
        return;
    }
    // Metal mul_mv keeps F32 activations: measured worst error 0.33.
    run_leg(
        profile,
        Tolerance {
            logit: 0.6,
            chunked: 1e-2,
        },
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUFs (IZWI_LFM25_MODELS_DIR)"]
fn lfm25_chat_matches_the_f32_reference_on_cuda() {
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Cuda).unwrap();
    if !matches!(profile.device, Device::Cuda(_)) {
        eprintln!("skipping: no CUDA device");
        return;
    }
    // CUDA MMVQ quantizes activations to Q8_1, like the CPU path.
    run_leg(
        profile,
        Tolerance {
            logit: 1.5,
            chunked: 1e-2,
        },
    );
}
