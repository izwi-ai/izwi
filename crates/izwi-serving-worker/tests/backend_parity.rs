//! DS0.8 backend parity harness: the same tiny LFM2 fixture and greedy chat
//! fixtures must produce identical outputs across every buildable backend
//! lane. The CPU leg always runs and doubles as a determinism gate (two
//! independent worker processes must agree). The Metal leg is compiled only
//! when the crate is built with the `metal` feature and executes on hosts
//! with Apple silicon; a real model with a documented tolerance budget
//! replaces this smoke fixture before any performance claim.
//!
//! Hardware lanes without a build/device on the executing host stay
//! `not run` (serving plan INV-18) — they are never reported as passed.

mod common;

use common::{id, write_tiny_lfm_fixture};
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::*;
use std::{collections::BTreeSet, net::TcpListener, process::Stdio, time::Duration};

struct ChildGuard(tokio::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

const PARITY_PROMPTS: [&str; 3] = ["hello", "ab", "a"];
const STARTUP_DEADLINE: Duration = Duration::from_secs(30);

fn uid<T: TryFrom<String>>(prefix: &str, suffix: usize) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(format!("{prefix}-{suffix}")).expect("unique test identity")
}

async fn collect_lane_outputs(
    models_dir: &std::path::Path,
    backend_env: &[(&str, &str)],
    startup_deadline: Duration,
) -> Vec<String> {
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_izwi-serving-worker"))
        .env("IZWI_WORKER_BIND", address.to_string())
        .env("IZWI_MODELS_DIR", models_dir)
        .env("IZWI_WORKER_CREDENTIAL_ID", "parity-credential")
        .env("IZWI_WORKER_BEARER_TOKEN", "parity-secret")
        .env("IZWI_WORKER_ARTIFACT_REVISION", "tiny-lfm-fixture-v1")
        .env("IZWI_WORKER_CPU_THREADS", "1")
        .env("IZWI_WORKER_MAX_ACTIVE", "1")
        .env("RUST_LOG", "warn")
        .envs(backend_env.iter().copied())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut child = ChildGuard(child);

    let credentials = ServiceCredentials {
        credential_id: id("parity-credential"),
        bearer_token: ServiceBearerToken::new("parity-secret").unwrap(),
    };
    let client = WorkerClient::new(
        &format!("http://{address}/"),
        credentials,
        WorkerClientConfig {
            connect_timeout: Duration::from_millis(100),
            request_timeout: Duration::from_millis(500),
            progress_timeout: Duration::from_secs(5),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();
    let descriptor = tokio::time::timeout(startup_deadline, async {
        loop {
            match client.descriptor().await {
                Ok(descriptor) => break descriptor,
                Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
            }
        }
    })
    .await
    .expect("worker loads and warms the tiny model before its startup deadline");

    let mut outputs = Vec::new();
    for (index, prompt) in PARITY_PROMPTS.iter().enumerate() {
        let events = client
            .invoke_collect(InvocationRequest {
                schema_version: PROTOCOL_V1,
                request_id: uid("parity-request", index),
                attempt_id: uid("parity-attempt", index),
                expected_worker_incarnation: descriptor.incarnation_id.clone(),
                deployment_id: id("lfm25-cpu-v1"),
                expected_model_generation: ModelGeneration::new(1).unwrap(),
                caller: GatewayAttestedCallerContext {
                    tenant_id: id("local-test"),
                    caller_id: id("parity-harness"),
                    policy_revision: id("test-policy-v1"),
                    permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
                    allowed_data_regions: vec!["local".into()],
                },
                task: TaskKind::Chat,
                service_class: ServiceClass::Interactive,
                remaining_time_ms: 10_000,
                max_queue_wait_ms: 2_000,
                output_limits: OutputLimits {
                    max_tokens: 4,
                    max_bytes: 1024,
                },
                requested_output_format: OutputFormat::Text,
                session_id: None,
                request_digest: uid("sha256:parity-fixture", index),
                input: InvocationInput::Chat {
                    input: ChatInput {
                        messages: vec![ChatMessage {
                            role: ChatRole::User,
                            content: (*prompt).into(),
                        }],
                    },
                    parameters: ChatParameters {
                        temperature: Some(0.0),
                        top_p: Some(1.0),
                        seed: Some(0),
                        stop: Vec::new(),
                    },
                },
            })
            .await
            .unwrap_or_else(|error| panic!("prompt {index} ({prompt:?}) failed: {error}"));
        assert!(matches!(
            events.last().map(|event| &event.event),
            Some(InvocationEventKind::Completed { .. })
        ));
        let text: String = events
            .iter()
            .filter_map(|event| match &event.event {
                InvocationEventKind::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        outputs.push(text);
    }
    drop(child);
    outputs
}

#[tokio::test]
async fn cpu_outputs_are_deterministic_across_worker_processes() {
    let models = tempfile::tempdir().unwrap();
    write_tiny_lfm_fixture(models.path());
    let first = collect_lane_outputs(models.path(), &[], STARTUP_DEADLINE).await;
    let second = collect_lane_outputs(models.path(), &[], STARTUP_DEADLINE).await;
    assert_eq!(
        first, second,
        "the CPU lane must be deterministic run to run"
    );
    assert!(
        first.iter().all(|output| !output.is_empty()),
        "parity fixtures must produce non-empty outputs"
    );
}

/// The strict Metal assignment lane requires the operator-provided device
/// identity (`metal:<registryID>`). Tests discover it from the host's default
/// GPU; when that is impossible the leg is recorded as not-run, never passed.
#[cfg(feature = "metal")]
fn host_metal_device_id() -> Option<String> {
    let output = std::process::Command::new("swift")
        .args(["-e", "import Metal; if let d = MTLCreateSystemDefaultDevice() { print(\"metal:\" + String(d.registryID)) }"])
        .output()
        .ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    let id = stdout.trim().to_string();
    (output.status.success() && id.starts_with("metal:")).then_some(id)
}

#[cfg(feature = "metal")]
#[tokio::test]
async fn metal_outputs_match_cpu_outputs_on_the_parity_fixture() {
    let Some(device_id) = host_metal_device_id() else {
        println!("metal parity leg not run: host default GPU registryID unavailable");
        return;
    };
    let models = tempfile::tempdir().unwrap();
    write_tiny_lfm_fixture(models.path());
    let cpu = collect_lane_outputs(models.path(), &[], STARTUP_DEADLINE).await;
    // Cold Metal starts compile the MSL shader cache; allow a generous deadline.
    let metal = collect_lane_outputs(
        models.path(),
        &[
            ("IZWI_BACKEND", "metal"),
            ("IZWI_WORKER_EXPECTED_DEVICE_ID", device_id.as_str()),
            ("IZWI_METAL_DEVICE_ORDINAL", "0"),
        ],
        Duration::from_secs(120),
    )
    .await;
    assert_eq!(
        cpu, metal,
        "greedy decode on the unified-memory lane must match the CPU lane on the parity fixture"
    );
}
