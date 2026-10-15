//! Phase 1.1 chat request controls through the real `RuntimeService`,
//! scheduler and LFM2 decode path on the tiny LFM2 fixture.

mod common;

#[cfg(feature = "metal")]
use common::metal_leg_device;
use common::write_tiny_lfm_fixture;
use izwi_core::backends::RuntimeDeviceAssignment;
use izwi_core::{
    ChatGeneration, ChatMessage, ChatRole, EngineConfig, GenerationParams, ModelVariant,
    RuntimeChatInvocationEvent, RuntimeChatInvocationRequest, RuntimeRequestContext,
    RuntimeService, WorkloadClass,
};
use std::time::{Duration, Instant};

const VARIANT: ModelVariant = ModelVariant::Lfm2512BInstructGguf;

async fn tiny_lfm_runtime(
    models: &std::path::Path,
    assignment: RuntimeDeviceAssignment,
) -> RuntimeService {
    write_tiny_lfm_fixture(models);
    let engine = EngineConfig {
        models_dir: models.to_path_buf(),
        max_loaded_models: Some(1),
        num_threads: 1,
        ..EngineConfig::default()
    };
    let runtime = RuntimeService::new_assigned(engine, assignment).expect("runtime");
    runtime
        .load_model(VARIANT)
        .await
        .expect("tiny LFM fixture loads");
    runtime
}

/// One finished chat: the text deltas it streamed (none when not streaming)
/// and its terminal generation.
struct ChatRun {
    deltas: Vec<String>,
    generation: ChatGeneration,
}

impl ChatRun {
    /// The sampled tokens. The fixture's word-level vocabulary makes each
    /// token's text its identity.
    fn tokens(&self) -> Vec<String> {
        self.generation
            .logprobs
            .iter()
            .map(|entry| entry.token.clone())
            .collect()
    }
}

async fn run_chat(
    runtime: &RuntimeService,
    params: GenerationParams,
    streaming: bool,
) -> izwi_core::Result<ChatRun> {
    let runtime_context = RuntimeRequestContext::new(WorkloadClass::Interactive)
        .with_deadline(Instant::now() + Duration::from_secs(30));
    let mut invocation = runtime
        .start_chat_invocation(RuntimeChatInvocationRequest {
            variant: VARIANT,
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".into(),
            }],
            params,
            chat_config: Default::default(),
            correlation_id: None,
            runtime_context,
            streaming,
        })
        .await?;
    let mut deltas = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(30), invocation.next_event())
            .await
            .expect("chat finishes within 30 s")?
        {
            Some(RuntimeChatInvocationEvent::TextDelta { text, .. }) => deltas.push(text),
            Some(RuntimeChatInvocationEvent::Completed(generation)) => {
                invocation.wait_for_teardown().await?;
                return Ok(ChatRun { deltas, generation });
            }
            None => panic!("chat ended without a completion"),
        }
    }
}

/// Hot sampling: the fixture's logits are peaked, so a high temperature
/// makes the sampled tokens depend on the seed.
fn sampled(seed: Option<u64>) -> GenerationParams {
    GenerationParams {
        temperature: 5.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        max_tokens: 12,
        seed,
        logprobs: true,
        ..GenerationParams::default()
    }
}

/// EXE-F9: the same seed and parameters sample the same tokens, streamed or
/// not, and different seeds sample different tokens.
async fn assert_seed_reproduces_sampling(runtime: &RuntimeService) {
    let first = run_chat(runtime, sampled(Some(7)), false).await.unwrap();
    let second = run_chat(runtime, sampled(Some(7)), false).await.unwrap();
    let streamed = run_chat(runtime, sampled(Some(7)), true).await.unwrap();
    assert!(!first.tokens().is_empty(), "the fixture samples tokens");
    assert_eq!(first.tokens(), second.tokens());
    assert_eq!(first.generation.text, second.generation.text);
    assert_eq!(first.tokens(), streamed.tokens());
    assert_eq!(first.generation.text, streamed.deltas.concat());

    let mut others = Vec::new();
    for seed in 8..12 {
        others.push(
            run_chat(runtime, sampled(Some(seed)), false)
                .await
                .unwrap()
                .tokens(),
        );
    }
    assert!(
        others.iter().any(|tokens| *tokens != first.tokens()),
        "different seeds must sample different tokens: {:?} vs {others:?}",
        first.tokens()
    );
}

#[tokio::test]
async fn a_request_seed_reproduces_sampling_on_cpu() {
    let models = tempfile::tempdir().unwrap();
    let runtime = tiny_lfm_runtime(models.path(), RuntimeDeviceAssignment::Cpu).await;
    assert_seed_reproduces_sampling(&runtime).await;
}

#[cfg(feature = "metal")]
#[tokio::test]
async fn a_request_seed_reproduces_sampling_on_metal() {
    let Some(expected_device_id) = metal_leg_device("seed metal leg") else {
        return;
    };
    let models = tempfile::tempdir().unwrap();
    let runtime = tiny_lfm_runtime(
        models.path(),
        RuntimeDeviceAssignment::Metal {
            process_local_device_index: 0,
            expected_device_id,
        },
    )
    .await;
    assert_seed_reproduces_sampling(&runtime).await;
}

/// A stop string of `len` characters that starts at the first character
/// other than `a` or a space (past the first character), with the text before
/// it, or `None` when `text` has no such window.
fn stop_inside(text: &str, len: usize) -> Option<(String, String)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let start =
        (1..chars.len().saturating_sub(len)).find(|&index| !matches!(chars[index].1, 'a' | ' '))?;
    let from = chars[start].0;
    let stop = &text[from..chars[start + len].0];
    (text.find(stop) == Some(from)).then(|| (stop.to_string(), text[..from].to_string()))
}

/// EXE-F9 / EDGE-F4: a stop string ends generation, is excluded from the
/// output, and is never streamed, even split across deltas.
#[tokio::test]
async fn a_stop_string_ends_generation_without_being_returned() {
    let models = tempfile::tempdir().unwrap();
    let runtime = tiny_lfm_runtime(models.path(), RuntimeDeviceAssignment::Cpu).await;
    // A seeded run whose text has a stop string that spans several tokens
    // (so several streamed deltas) and starts after varied text.
    let mut found = None;
    for seed in 0..64 {
        let full = run_chat(&runtime, sampled(Some(seed)), true).await.unwrap();
        if let Some(stop) = stop_inside(&full.deltas.concat(), 4) {
            found = Some((seed, full, stop));
            break;
        }
    }
    let (seed, full, (stop, expected)) = found.expect("a seed samples varied text");
    assert!(
        full.deltas.len() > expected.len().min(2),
        "one delta per token"
    );

    let with_stop = |streaming| {
        let mut params = sampled(Some(seed));
        params.stop_sequences = vec![stop.clone()];
        run_chat(&runtime, params, streaming)
    };
    let streamed = with_stop(true).await.unwrap();
    assert_eq!(streamed.deltas.concat(), expected);
    assert_eq!(streamed.generation.text, expected);
    assert!(streamed.deltas.iter().all(|delta| !delta.contains(&stop)));
    assert_eq!(
        streamed.generation.finish_reason,
        Some(izwi_core::engine::OutputFinishReason::StopToken)
    );
    assert!(streamed.generation.tokens_generated < full.generation.tokens_generated);

    let collected = with_stop(false).await.unwrap();
    assert_eq!(collected.generation.text, expected);
    // Sampling stops at the token that completes the stop string: the
    // tokens are the unstopped run's prefix, and usage counts exactly them.
    let tokens = collected.tokens();
    assert!(tokens.len() < full.tokens().len());
    assert_eq!(tokens[..], full.tokens()[..tokens.len()]);
    assert_eq!(collected.generation.tokens_generated, tokens.len());
    assert_eq!(
        collected.generation.finish_reason,
        Some(izwi_core::engine::OutputFinishReason::StopToken)
    );
}
