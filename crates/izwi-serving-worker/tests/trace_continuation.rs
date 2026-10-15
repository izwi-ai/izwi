//! The worker continues a caller's W3C trace: its JSON log line for an
//! invocation carries the caller's trace id and names the caller's span as
//! the parent. One test per binary, because tracing caches callsite interest
//! process-wide and a scoped subscriber in a shared test binary misses events.

use izwi_serving_client::{TraceContext, WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::*;
use izwi_serving_worker::test_support::{TestWorker, TestWorkerConfig};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use tracing_subscriber::layer::SubscriberExt;

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn id<T: TryFrom<String>>(value: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.to_string()).expect("test identity")
}

fn invocation(config: &TestWorkerConfig) -> InvocationRequest {
    InvocationRequest {
        schema_version: PROTOCOL_V1,
        request_id: id("request-trace"),
        attempt_id: id("attempt-trace"),
        expected_worker_incarnation: config.incarnation_id.clone(),
        deployment_id: config.deployment_id.clone(),
        expected_model_generation: config.model_generation,
        caller: GatewayAttestedCallerContext {
            tenant_id: id("tenant-trace"),
            caller_id: id("caller-trace"),
            policy_revision: id("policy-trace"),
            permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
            allowed_data_regions: vec!["local".into()],
        },
        task: TaskKind::Chat,
        service_class: ServiceClass::Interactive,
        remaining_time_ms: 2_000,
        max_queue_wait_ms: 25,
        output_limits: OutputLimits {
            max_tokens: 32,
            max_bytes: 4096,
        },
        requested_output_format: OutputFormat::Text,
        session_id: None,
        request_digest: id("digest-trace"),
        input: InvocationInput::Chat {
            input: ChatInput {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "hello worker".into(),
                }],
            },
            parameters: ChatParameters::default(),
        },
    }
}

#[tokio::test]
async fn worker_json_logs_continue_the_callers_trace() {
    let capture = Capture::default();
    let writer = capture.clone();
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(izwi_serving_worker::json_log_layer(move || writer.clone())),
    )
    .expect("the only subscriber in this binary");

    let worker = TestWorker::spawn(TestWorkerConfig::default())
        .await
        .expect("test worker binds");
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig::default(),
    )
    .expect("client");
    let caller = TraceContext::new_root();
    let mut stream = client
        .invoke_traced(invocation(worker.config()), Some(&caller))
        .await
        .expect("invocation accepted");
    while stream.next_event().await.expect("event").is_some() {}

    let logs = String::from_utf8(capture.0.lock().unwrap().clone()).expect("UTF-8 logs");
    let span = logs
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSON log line"))
        .filter(|line| line["message"] == "worker request completed")
        .filter_map(|line| {
            line["spans"]
                .as_array()?
                .iter()
                .find(|span| span["name"] == "worker_request" && span["path"] == INVOCATIONS_PATH)
                .cloned()
        })
        .next()
        .unwrap_or_else(|| panic!("no invocation log line in:\n{logs}"));
    assert_eq!(span["trace_id"], caller.trace_id());
    assert_eq!(span["parent_span_id"], caller.span_id());
    assert_ne!(span["span_id"], caller.span_id());
}
