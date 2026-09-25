//! DS1.5 regression: a session that attaches a published tensor snapshot /
//! reused paged prefix at cursor C must complete. The scheduler still issues
//! the logical span from 0, so the model side clips its feed to the attached
//! physical cursor (`continue_chunked_prefill_physical`); before that clip
//! landed, the paged append overran the block table ("physical paged append
//! ends at N, beyond capacity M") and the stream closed without a final
//! marker.

mod common;

use common::write_tiny_qwen38_hybrid_fixture;
use izwi_core::{
    ChatMessage, ChatRole, EngineConfig, GenerationParams, ModelVariant,
    RuntimeChatInvocationRequest, RuntimeRequestContext, RuntimeService, WorkloadClass,
};
use std::time::{Duration, Instant};

async fn run_chat(
    runtime: &RuntimeService,
    id: &str,
    system: &str,
    user: &str,
    max_tokens: usize,
) -> Result<String, izwi_core::Error> {
    let runtime_context = RuntimeRequestContext::new(WorkloadClass::Streaming)
        .with_deadline(Instant::now() + Duration::from_secs(30));
    let mut invocation = runtime
        .start_chat_invocation(RuntimeChatInvocationRequest {
            variant: ModelVariant::Qwen3827BFp8,
            messages: vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: system.into(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: user.into(),
                },
            ],
            params: GenerationParams {
                temperature: 0.0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                max_tokens,
                ..GenerationParams::default()
            },
            chat_config: Default::default(),
            correlation_id: Some(id.into()),
            runtime_context,
            streaming: true,
        })
        .await?;
    let mut text = String::new();
    let deadline = tokio::time::sleep(Duration::from_secs(30));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            () = &mut deadline => {
                invocation.request_cancel();
                let _ = invocation.wait_for_teardown().await;
                panic!("chat {id} timed out");
            }
            event = invocation.next_event() => match event {
                Ok(Some(izwi_core::RuntimeChatInvocationEvent::TextDelta(delta))) => {
                    text.push_str(&delta);
                }
                Ok(Some(izwi_core::RuntimeChatInvocationEvent::Completed(generation))) => {
                    return Ok(generation.text);
                }
                Ok(None) => panic!("chat {id} ended without completion"),
                Err(error) => {
                    let _ = invocation.wait_for_teardown().await;
                    return Err(error);
                }
            },
        }
    }
}

#[tokio::test]
async fn attach_after_publish_completes_without_error() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_new("izwi_core=trace,warn").unwrap())
        .with_test_writer()
        .try_init();
    // Serving-policy env must be set before the runtime is constructed.
    std::env::set_var("IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY", "1");
    std::env::set_var("IZWI_CUDA_MTP", "off");
    std::env::set_var("IZWI_ENABLE_PREFIX_CACHING", "1");
    std::env::set_var("IZWI_MANAGED_PREFIX_CACHE_SALT", "repro-salt");
    std::env::set_var("IZWI_MAX_PREFIX_CACHE_PAGES", "64");
    std::env::set_var("IZWI_ENABLE_CHUNKED_PREFILL", "1");
    std::env::set_var("IZWI_CHUNKED_PREFILL_THRESHOLD", "8");
    std::env::set_var("IZWI_MAX_SEQUENCE_LENGTH", "4096");
    std::env::set_var("IZWI_KV_PAGE_SIZE", "16");
    std::env::set_var("IZWI_CPU_MEMORY_BUDGET_BYTES", "2147483648");

    let models = tempfile::tempdir().unwrap();
    write_tiny_qwen38_hybrid_fixture(models.path());
    let engine = EngineConfig {
        models_dir: models.path().to_path_buf(),
        max_loaded_models: Some(1),
        max_queued_requests: 1,
        max_scheduler_batch_size: 1,
        max_retained_sequences: 1,
        max_staged_transactions: 1,
        num_threads: 1,
        enable_prefix_caching: true,
        managed_prefix_cache_salt: Some("repro-salt".into()),
        max_prefix_cache_pages: 64,
        enable_chunked_prefill: true,
        chunked_prefill_threshold: 8,
        max_sequence_length: izwi_core::config::ContextLengthPreference::explicit(4096).unwrap(),
        ..EngineConfig::default()
    };
    let runtime =
        RuntimeService::new_assigned(engine, izwi_core::backends::RuntimeDeviceAssignment::Cpu)
            .expect("runtime");
    runtime
        .load_model(ModelVariant::Qwen3827BFp8)
        .await
        .expect("model loads");
    let snapshot = runtime.telemetry_snapshot().await;
    for model in &snapshot.engine.kv_cache.models {
        println!(
            "DEBUG model single_sequence_token_capacity={} aggregate={}",
            model.single_sequence_token_capacity, model.aggregate_token_capacity
        );
        for arena in &model.arenas {
            println!(
                "DEBUG arena page_tokens={} token_capacity={} capacity_pages={}",
                arena.page_tokens, arena.token_capacity, arena.coordinator.capacity_pages
            );
        }
    }

    let system = (0..50)
        .map(|i| ["alpha", "beta", "gamma", "delta", "epsilon"][i % 5])
        .collect::<Vec<_>>()
        .join(" ");
    // First request: cold prefill, publishes aligned snapshots.
    let first = run_chat(&runtime, "publish-1", &system, "tail one", 4).await;
    println!("publish request result: {first:?}");
    // Second request: shares the system prefix; must attach (or at worst
    // prefill fully) without failing.
    let second = run_chat(&runtime, "attach-1", &system, "tail two", 4).await;
    println!("attach request result: {second:?}");

    let snapshot = runtime.telemetry_snapshot().await;
    let counters = &snapshot.engine.kv_cache.counters;
    println!(
        "publishes={} attaches={} truncations={} hits={} misses={} reused={}",
        counters.tensor_snapshot_publishes,
        counters.tensor_snapshot_attaches,
        counters.tensor_snapshot_truncations,
        counters.prefix_hits,
        counters.prefix_misses,
        counters.reused_tokens,
    );
    assert!(counters.tensor_snapshot_publishes >= 1, "publish must fire");
    assert!(
        counters.tensor_snapshot_attaches >= 1,
        "second request must attach the published snapshot"
    );
    assert!(
        counters.reused_tokens >= 16,
        "attach must reuse committed prefix tokens"
    );
    assert!(
        second.is_ok(),
        "attach request must complete without error: {second:?}"
    );
}
