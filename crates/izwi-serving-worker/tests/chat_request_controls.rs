//! Phase 1.1 chat request controls through the real `RuntimeService`,
//! scheduler and LFM2 decode path on the tiny LFM2 fixture.

mod common;

#[cfg(feature = "metal")]
use common::metal_leg_device;
use common::write_tiny_lfm_fixture;
use izwi_core::backends::RuntimeDeviceAssignment;
use izwi_core::{
    ChatGeneration, ChatMessage, ChatRequestConfig, ChatRole, ChatTruncation, EngineConfig,
    GenerationParams, ModelVariant, RuntimeChatInvocationEvent, RuntimeChatInvocationRequest,
    RuntimeRequestContext, RuntimeService, WorkloadClass, CHAT_MAX_TOKENS_UNSET,
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
    /// The tokens of the logprob entries the deltas carried.
    delta_tokens: Vec<String>,
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
    run_conversation(
        runtime,
        vec![message(ChatRole::User, "hello")],
        params,
        Default::default(),
        streaming,
    )
    .await
}

fn message(role: ChatRole, content: &str) -> ChatMessage {
    ChatMessage {
        role,
        content: content.into(),
    }
}

async fn run_conversation(
    runtime: &RuntimeService,
    messages: Vec<ChatMessage>,
    params: GenerationParams,
    chat_config: ChatRequestConfig,
    streaming: bool,
) -> izwi_core::Result<ChatRun> {
    run_capped_conversation(runtime, messages, params, chat_config, streaming, None).await
}

async fn run_capped_conversation(
    runtime: &RuntimeService,
    messages: Vec<ChatMessage>,
    params: GenerationParams,
    chat_config: ChatRequestConfig,
    streaming: bool,
    max_tokens_ceiling: Option<usize>,
) -> izwi_core::Result<ChatRun> {
    let runtime_context = RuntimeRequestContext::new(WorkloadClass::Interactive)
        .with_deadline(Instant::now() + Duration::from_secs(30));
    let mut invocation = runtime
        .start_chat_invocation(RuntimeChatInvocationRequest {
            variant: VARIANT,
            messages,
            params,
            chat_config,
            correlation_id: None,
            runtime_context,
            streaming,
            max_tokens_ceiling,
        })
        .await?;
    let mut deltas = Vec::new();
    let mut delta_tokens = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(30), invocation.next_event())
            .await
            .expect("chat finishes within 30 s")?
        {
            Some(RuntimeChatInvocationEvent::TextDelta { text, logprobs }) => {
                deltas.push(text);
                delta_tokens.extend(logprobs.into_iter().map(|entry| entry.token));
            }
            Some(RuntimeChatInvocationEvent::Completed(generation)) => {
                invocation.wait_for_teardown().await?;
                return Ok(ChatRun {
                    deltas,
                    delta_tokens,
                    generation,
                });
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
    // The stream carries the logprobs of every sampled token, including the
    // ones whose text the stop string discarded, like the terminal list.
    assert_eq!(streamed.delta_tokens, streamed.tokens());
    assert_eq!(
        streamed.tokens().len(),
        streamed.generation.tokens_generated
    );
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

/// Stop strings that first complete at different tokens of `run`, past
/// its first six: (stop, text before it, tokens through the one that
/// completes it). The fixture decodes words joined by spaces.
fn stops_across_quanta(run: &ChatRun) -> Vec<(String, String, usize)> {
    let tokens = run.tokens();
    let text = tokens.join(" ");
    assert_eq!(text, run.deltas.concat(), "fixture text is its words");
    let mut stops = Vec::new();
    for end in 6..tokens.len() {
        for width in 2..=3 {
            let start = end + 1 - width;
            let stop = tokens[start..=end].join(" ");
            let at = tokens[..start].join(" ").len() + 1;
            if text.find(&stop) == Some(at) {
                stops.push((stop, text[..at].to_string(), end + 1));
                break;
            }
        }
    }
    stops
}

/// Stop strings are checked after every token of a multi-token decode
/// quantum (four tokens on the CPU, eight on Metal for LFM2), so a stop
/// string ends the quantum at the token that completes it: the text stops
/// before it, usage counts exactly the tokens through it, and the request
/// still decodes several tokens per quantum.
async fn assert_stop_string_ends_a_multi_token_quantum(runtime: &RuntimeService) {
    // Stops completing at several different tokens, so some end inside a
    // quantum rather than at its last token.
    let mut found = None;
    for seed in 0..64 {
        let mut params = sampled(Some(seed));
        params.max_tokens = 24;
        let full = run_chat(runtime, params, true).await.unwrap();
        let stops = stops_across_quanta(&full);
        if stops.len() >= 4 {
            found = Some((seed, full, stops));
            break;
        }
    }
    let (seed, full, stops) = found.expect("a seed samples a long varied text");
    for (stop, expected, count) in stops.into_iter().take(8) {
        for streaming in [true, false] {
            let mut params = sampled(Some(seed));
            params.max_tokens = 24;
            params.stop_sequences = vec![stop.clone()];
            let run = run_chat(runtime, params, streaming).await.unwrap();
            if streaming {
                assert_eq!(run.deltas.concat(), expected);
                assert_eq!(run.delta_tokens, run.tokens());
            }
            assert_eq!(run.generation.text, expected);
            assert_eq!(
                run.generation.finish_reason,
                Some(izwi_core::engine::OutputFinishReason::StopToken)
            );
            assert_eq!(run.tokens()[..], full.tokens()[..count], "{stop:?}");
            assert_eq!(run.generation.tokens_generated, count);
            let breakdown = run
                .generation
                .latency_breakdown
                .as_ref()
                .expect("latency breakdown");
            let decode_tokens = breakdown.decode_tokens.expect("decode tokens");
            assert!(
                (breakdown.decode_steps as usize) < decode_tokens,
                "stop strings forced one token per quantum: {breakdown:?}"
            );
        }
    }
}

#[tokio::test]
async fn a_stop_string_ends_a_multi_token_quantum_on_cpu() {
    let models = tempfile::tempdir().unwrap();
    let runtime = tiny_lfm_runtime(models.path(), RuntimeDeviceAssignment::Cpu).await;
    assert_stop_string_ends_a_multi_token_quantum(&runtime).await;
}

#[cfg(feature = "metal")]
#[tokio::test]
async fn a_stop_string_ends_an_eight_token_quantum_on_metal() {
    let Some(expected_device_id) = metal_leg_device("stop-string quantum metal leg") else {
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
    assert_stop_string_ends_a_multi_token_quantum(&runtime).await;
}

fn greedy(max_tokens: usize) -> GenerationParams {
    GenerationParams {
        temperature: 0.0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens,
        ..GenerationParams::default()
    }
}

fn truncation(truncation: ChatTruncation) -> ChatRequestConfig {
    ChatRequestConfig {
        truncation,
        ..ChatRequestConfig::default()
    }
}

/// A conversation of `turns` complete user/assistant turns and a new user
/// message; on the fixture's 32-token context ten turns overflow.
fn conversation(turns: usize) -> Vec<ChatMessage> {
    let mut messages = vec![message(ChatRole::System, "be brief")];
    for _ in 0..turns {
        messages.push(message(ChatRole::User, "a"));
        messages.push(message(ChatRole::Assistant, "b"));
    }
    messages.push(message(ChatRole::User, "hello"));
    messages
}

fn expect_context_length_error(result: izwi_core::Result<ChatRun>, expected: &str) {
    match result {
        Err(izwi_core::Error::ContextLengthExceeded(message)) => {
            assert!(message.contains(expected), "{message}")
        }
        Err(other) => panic!("expected a context-length error, got {other}"),
        Ok(run) => panic!(
            "expected a context-length error, got a {}-token prompt",
            run.generation.prompt_tokens
        ),
    }
}

/// D8: a conversation that outgrows the context is rejected by default, the
/// way OpenAI and vLLM reject it, and drops its oldest turns only when the
/// request opts into truncation.
#[tokio::test]
async fn prompt_overflow_is_rejected_unless_truncation_is_requested() {
    let models = tempfile::tempdir().unwrap();
    let runtime = tiny_lfm_runtime(models.path(), RuntimeDeviceAssignment::Cpu).await;
    let disabled = run_conversation(
        &runtime,
        conversation(10),
        greedy(CHAT_MAX_TOKENS_UNSET),
        ChatRequestConfig::default(),
        false,
    )
    .await;
    expect_context_length_error(disabled, "maximum context length is 32 tokens");

    let truncated = run_conversation(
        &runtime,
        conversation(10),
        greedy(CHAT_MAX_TOKENS_UNSET),
        truncation(ChatTruncation::Auto),
        false,
    )
    .await
    .expect("auto truncation drops the oldest turns");
    assert!(truncated.generation.prompt_tokens < 32);
}

/// D8, following vLLM: an explicit `max_tokens` that does not fit beside the
/// prompt is rejected, an unset one takes the rest of the context, and
/// truncation shortens an explicit one instead.
#[tokio::test]
async fn an_output_budget_that_does_not_fit_is_rejected_unless_unset_or_truncated() {
    let models = tempfile::tempdir().unwrap();
    let runtime = tiny_lfm_runtime(models.path(), RuntimeDeviceAssignment::Cpu).await;
    let conversation = || vec![message(ChatRole::User, "hello")];

    let fits = run_conversation(
        &runtime,
        conversation(),
        greedy(4),
        Default::default(),
        false,
    )
    .await
    .expect("a budget that fits runs");
    let prompt = fits.generation.prompt_tokens;
    let remaining = 32 - prompt;

    expect_context_length_error(
        run_conversation(
            &runtime,
            conversation(),
            greedy(remaining + 1),
            Default::default(),
            false,
        )
        .await,
        &format!(
            "({prompt} in the messages, {} in the completion)",
            remaining + 1
        ),
    );
    for (params, chat_config) in [
        (greedy(CHAT_MAX_TOKENS_UNSET), ChatRequestConfig::default()),
        (greedy(remaining + 1), truncation(ChatTruncation::Auto)),
    ] {
        let run = run_conversation(&runtime, conversation(), params, chat_config, false)
            .await
            .expect("an unset or truncated budget runs");
        assert!(run.generation.tokens_generated <= remaining);
    }
}

/// A worker caps the budget at its gateway's ceiling after D8 resolves it:
/// an unset budget runs to the ceiling instead of the rest of the context,
/// and an explicit budget that does not fit is still rejected.
#[tokio::test]
async fn a_ceiling_caps_the_budget_without_hiding_an_overflow() {
    let models = tempfile::tempdir().unwrap();
    let runtime = tiny_lfm_runtime(models.path(), RuntimeDeviceAssignment::Cpu).await;
    let conversation = || vec![message(ChatRole::User, "hello")];
    let uncapped = run_conversation(
        &runtime,
        conversation(),
        greedy(CHAT_MAX_TOKENS_UNSET),
        Default::default(),
        false,
    )
    .await
    .unwrap();
    assert!(
        uncapped.generation.tokens_generated > 3,
        "the fixture runs on"
    );
    let prompt = uncapped.generation.prompt_tokens;

    let capped = run_capped_conversation(
        &runtime,
        conversation(),
        greedy(CHAT_MAX_TOKENS_UNSET),
        Default::default(),
        false,
        Some(3),
    )
    .await
    .unwrap();
    assert_eq!(capped.generation.tokens_generated, 3);
    assert_eq!(
        capped.generation.finish_reason,
        Some(izwi_core::engine::OutputFinishReason::MaxTokens)
    );

    let too_long = 32 - prompt + 1;
    expect_context_length_error(
        run_capped_conversation(
            &runtime,
            conversation(),
            greedy(too_long),
            Default::default(),
            false,
            Some(3),
        )
        .await,
        &format!("({prompt} in the messages, {too_long} in the completion)"),
    );
}
