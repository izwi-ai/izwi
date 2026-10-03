//! DS4.4 acceptance: hierarchical KV offload with a device arena too small to
//! hold every committed prefix. Sessions share one multi-page system prefix;
//! once the arena's pressure crosses the high watermark the manager demotes
//! the committed chain into the bounded host pool, and later requests
//! continue their prefix lookup into the host tier and promote the pages back
//! into the fresh pages their reservation holds before executing. Greedy
//! output for an identical prompt must match the cold run byte for byte —
//! the restored pages feed attention — and the cycle must show up in the
//! engine counters.
//!
//! The tiny LFM fixture arena holds two 16-token pages, so the watermarks are
//! calibrated to fixture scale: a committed chain occupying half the arena is
//! already "over pressure". That is the same undersized-arena regime the DS4.5
//! benchmark exercises with production-sized geometry.

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
        .with_deadline(Instant::now() + Duration::from_secs(60));
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
    let deadline = tokio::time::sleep(Duration::from_secs(60));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            () = &mut deadline => {
                invocation.request_cancel();
                let _ = invocation.wait_for_teardown().await;
                panic!("chat {id} timed out");
            }
            event = invocation.next_event() => match event {
                Ok(Some(izwi_core::RuntimeChatInvocationEvent::TextDelta { text: delta, .. })) => {
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
async fn shared_prefix_survives_host_offload_with_matching_outputs() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_new("warn").unwrap())
        .with_test_writer()
        .try_init();
    // Serving-policy env must be set before the runtime is constructed.
    std::env::set_var("IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY", "1");
    std::env::set_var("IZWI_CUDA_MTP", "off");
    std::env::set_var("IZWI_ENABLE_PREFIX_CACHING", "1");
    std::env::set_var("IZWI_MANAGED_PREFIX_CACHE_SALT", "ds4-offload-salt");
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
        max_queued_requests: 8,
        max_scheduler_batch_size: 4,
        max_retained_sequences: 8,
        max_staged_transactions: 8,
        num_threads: 2,
        enable_prefix_caching: true,
        managed_prefix_cache_salt: Some("ds4-offload-salt".into()),
        max_prefix_cache_pages: 64,
        enable_chunked_prefill: true,
        chunked_prefill_threshold: 8,
        max_sequence_length: izwi_core::config::ContextLengthPreference::explicit(4096).unwrap(),
        kv_host_pool_budget_bytes: 8 << 20,
        // Fixture-scale watermarks: the synthetic arena resolves to 32 pages,
        // so a committed chain holding ~15% of it is already "over pressure".
        kv_offload_high_watermark: 0.15,
        kv_offload_low_watermark: 0.05,
        ..EngineConfig::default()
    };
    let runtime =
        RuntimeService::new_assigned(engine, izwi_core::backends::RuntimeDeviceAssignment::Cpu)
            .expect("runtime");
    runtime
        .load_model(ModelVariant::Qwen3827BFp8)
        .await
        .expect("model loads");
    let loaded = runtime.telemetry_snapshot().await;
    let page_bytes = loaded
        .engine
        .kv_cache
        .models
        .first()
        .and_then(|model| model.arenas.first().map(|arena| arena.bytes_per_page))
        .expect("loaded model exposes its arena");
    for model in &loaded.engine.kv_cache.models {
        for arena in &model.arenas {
            println!(
                "DS4DBG arena page_tokens={} capacity_pages={}",
                arena.page_tokens, arena.coordinator.capacity_pages
            );
        }
    }

    // The shared system prefix spans several complete pages; each session's
    // tail is request-private.
    let system = (0..50)
        .map(|i| ["alpha", "beta", "gamma", "delta", "epsilon"][i % 5])
        .collect::<Vec<_>>()
        .join(" ");

    // Cold run: the reference output for the shared-prefix conversation.
    let cold = run_chat(&runtime, "cold", &system, "tail one", 4)
        .await
        .expect("cold chat completes");
    assert!(!cold.is_empty(), "cold run must produce output");

    // Concurrent shared-prefix batch: every request reuses the committed
    // system chain while the two-page arena cannot retain every session's
    // private pages, so committed prefixes demote into the host tier and
    // later admission promotes them back.
    let shared_runtime = std::sync::Arc::new(runtime);
    let outputs = futures::future::join_all(
        ["tail two", "tail three", "tail four", "tail five"]
            .iter()
            .map(|user| {
                let runtime = shared_runtime.clone();
                let system = system.clone();
                async move { run_chat(&runtime, "batch", &system, user, 4).await }
            })
            .collect::<Vec<_>>(),
    )
    .await;
    for (index, output) in outputs.iter().enumerate() {
        assert!(
            output.is_ok(),
            "concurrent shared-prefix chat {index} failed: {output:?}"
        );
    }

    // Byte-identical greedy output with the cold run: attention consumed the
    // restored host pages, not zeros.
    let replay = run_chat(&shared_runtime, "replay", &system, "tail one", 4)
        .await
        .expect("replay chat completes");
    assert_eq!(
        replay, cold,
        "restored pages must reproduce the cold-run output"
    );

    let snapshot = shared_runtime.telemetry_snapshot().await;
    let counters = &snapshot.engine.kv_cache.counters;
    assert!(
        counters.demotions_total >= 1,
        "committed prefixes must have been demoted to the host tier: {counters:?}"
    );
    assert!(
        counters.promotions_total >= 1,
        "host-resident pages must have been promoted back: {counters:?}"
    );
    let budget_pages = (8_u64 << 20).div_ceil(page_bytes.max(1));
    assert!(
        counters.kv_host_pages <= budget_pages,
        "host usage {} must stay inside the pool budget ({budget_pages} pages)",
        counters.kv_host_pages
    );
}
