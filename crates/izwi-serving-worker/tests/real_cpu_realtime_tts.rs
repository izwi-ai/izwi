//! Real TTS evidence (DS3 TTS-stream follow-up): the real worker binary
//! loading the real Kokoro-82M artifact and executing genuine streaming
//! synthesis through the `izwi-realtime-v1` WebSocket subprotocol —
//! admission with an announced output spec, text input, live audio frames,
//! a final-flagged terminal frame, one terminal outcome, and a clean close.
//!
//! Ignored by default: set `IZWI_REAL_TTS_MODELS_DIR` to the local models
//! root holding `Kokoro-82M` and run with `--ignored`. The artifact is
//! staged read-only (symlinks plus a generated deployment manifest) so the
//! run never mutates the local cache.
//!
//! Backend selection: `IZWI_RT_EVIDENCE_BACKEND=cpu|metal` (default cpu).
//! The Metal lane additionally requires the metal-feature worker binary,
//! `IZWI_WORKER_EXPECTED_DEVICE_ID=metal:<registryID>`, and
//! `IZWI_METAL_DEVICE_ORDINAL` — same contract as the parity tests.

use izwi_core::artifacts::ArtifactManifest;
use izwi_core::ModelVariant;
use izwi_serving_client::realtime::{connect, RealtimeClientConfig};
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::*;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

struct ChildGuard(tokio::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

#[tokio::test]
#[ignore = "requires the real Kokoro TTS artifact (IZWI_REAL_TTS_MODELS_DIR)"]
async fn real_worker_streams_tts_through_the_realtime_subprotocol() {
    let models_root = std::env::var_os("IZWI_REAL_TTS_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .expect("set IZWI_REAL_TTS_MODELS_DIR to the local models root");
    let variant = ModelVariant::Kokoro82M;
    let source_dir = models_root.join(variant.dir_name());
    assert!(
        source_dir.exists(),
        "the real artifact must exist at {}",
        source_dir.display()
    );

    // Stage the artifact: symlinks to the real files plus a generated
    // deployment manifest naming exactly those files.
    let staging = tempfile::tempdir().unwrap();
    let model_dir = staging.path().join(variant.dir_name());
    std::fs::create_dir_all(&model_dir).unwrap();
    let mut files = Vec::new();
    for entry in walk_real_files(&source_dir) {
        let relative = entry
            .strip_prefix(&source_dir)
            .expect("relative artifact path")
            .to_path_buf();
        let link = model_dir.join(&relative);
        std::fs::create_dir_all(link.parent().unwrap()).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&entry, &link).unwrap();
        #[cfg(not(unix))]
        std::fs::copy(&entry, &link).unwrap();
        files.push(relative.to_string_lossy().to_string());
    }
    std::fs::write(
        model_dir.join(izwi_core::artifacts::ARTIFACT_MANIFEST_FILE),
        serde_json::to_vec_pretty(&ArtifactManifest {
            schema_version: 1,
            variant,
            repo_id: variant.repo_id().to_string(),
            revision: "local-real-artifact".to_string(),
            files,
        })
        .unwrap(),
    )
    .unwrap();

    let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let backend = std::env::var("IZWI_RT_EVIDENCE_BACKEND")
        .unwrap_or_else(|_| "cpu".to_string())
        .to_ascii_lowercase();
    let expected_backend = match backend.as_str() {
        "metal" => BackendKind::Metal,
        _ => BackendKind::Cpu,
    };
    let deployment_id = format!("kokoro-tts-{backend}-v1");
    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_izwi-serving-worker"));
    command
        .env("IZWI_WORKER_BIND", address.to_string())
        .env("IZWI_WORKER_TASK", "text_to_speech")
        .env("IZWI_WORKER_MODEL", variant.dir_name())
        .env("IZWI_MODELS_DIR", staging.path())
        .env("IZWI_WORKER_DEPLOYMENT_ID", &deployment_id)
        .env("IZWI_WORKER_ARTIFACT_REVISION", "local-real-artifact")
        .env("IZWI_WORKER_CREDENTIAL_ID", "real-tts-credential")
        .env("IZWI_WORKER_BEARER_TOKEN", "real-tts-secret")
        .env("IZWI_WORKER_MAX_ACTIVE", "2")
        .env("RUST_LOG", "warn")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if backend == "metal" {
        for (name, value) in [
            ("IZWI_BACKEND", "metal"),
            ("IZWI_WORKER_EXPECTED_DEVICE_ID", ""),
            ("IZWI_METAL_DEVICE_ORDINAL", "0"),
        ] {
            let inherited = std::env::var(name).unwrap_or(value.to_string());
            command.env(name, inherited);
        }
    }
    let child = ChildGuard(command.spawn().unwrap());

    let credentials = ServiceCredentials {
        credential_id: CredentialId::new("real-tts-credential").unwrap(),
        bearer_token: ServiceBearerToken::new("real-tts-secret").unwrap(),
    };
    let client = WorkerClient::new(
        &format!("http://{address}/"),
        credentials.clone(),
        WorkerClientConfig::default(),
    )
    .unwrap();

    // Startup includes the real model load plus a bounded short-text
    // synthesis warm-up: allow minutes, not seconds (Metal also compiles MSL
    // shaders cold).
    let descriptor = tokio::time::timeout(Duration::from_secs(900), async {
        loop {
            match client.descriptor().await {
                Ok(descriptor) => break descriptor,
                Err(_) => tokio::time::sleep(Duration::from_millis(250)).await,
            }
        }
    })
    .await
    .expect("real worker loads and warms the TTS model before the deadline");
    assert_eq!(descriptor.assignment.backend(), expected_backend);
    assert!(descriptor.features.contains(&WorkerFeature::RealtimeSocket));
    let status = client.status().await.expect("worker status");
    let deployment = status
        .deployments
        .iter()
        .find(|deployment| deployment.task == TaskKind::TextToSpeech)
        .expect("text_to_speech deployment advertised");
    assert!(deployment.capability.realtime);
    assert_eq!(deployment.deployment_id.as_str(), deployment_id);

    let admit = RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: SessionId::new("real-tts-session-1").unwrap(),
        request_id: RequestId::new("real-tts-request-1").unwrap(),
        attempt_id: AttemptId::new("real-tts-attempt-1").unwrap(),
        expected_worker_incarnation: descriptor.incarnation_id.clone(),
        deployment_id: DeploymentId::new(deployment_id.clone()).unwrap(),
        expected_model_generation: ModelGeneration::new(1).unwrap(),
        caller: GatewayAttestedCallerContext {
            tenant_id: TenantId::new("local-evidence").unwrap(),
            caller_id: CallerId::new("realtime-tts-evidence").unwrap(),
            policy_revision: PolicyRevision::new("evidence-policy-1").unwrap(),
            permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
            allowed_data_regions: vec!["local".into()],
        },
        task: TaskKind::TextToSpeech,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: 300_000,
        input: RealtimeStageInput::TextStream,
    };

    let mut session = connect(
        &format!("http://{address}/"),
        &credentials,
        admit,
        RealtimeClientConfig {
            admit_timeout: Duration::from_secs(60),
            event_timeout: Duration::from_secs(120),
            ..RealtimeClientConfig::default()
        },
    )
    .await
    .expect("realtime session admitted by the real worker");

    // The admission announces the synthesized audio spec at the model's rate.
    let spec = session
        .admission()
        .output_audio
        .expect("TTS admission announces output_audio");
    assert_eq!(spec.codec, RealtimeAudioCodec::PcmI16Le);
    assert_eq!(spec.channels, 1);
    assert!(
        (8_000..=192_000).contains(&spec.sample_rate),
        "announced rate {} within the protocol band",
        spec.sample_rate
    );

    let accepted = session
        .next_event()
        .await
        .expect("event stream")
        .expect("accepted event");
    assert!(matches!(
        accepted.event,
        InvocationEventKind::Accepted { .. }
    ));

    // Push the fixture text, then commit synthesis. The sentence mirrors the
    // DS3.7 ASR evidence transcript.
    session
        .send_text("The quick brown fox jumps over the lazy dog.")
        .await
        .expect("text accepted");
    session.finish().await.expect("finish accepted");

    let mut audio_frames = 0usize;
    let mut audio_bytes = 0u64;
    let mut nonzero_bytes = 0u64;
    let mut saw_final = false;
    while let Some(chunk) = session
        .next_audio()
        .await
        .expect("worker audio frames decode")
    {
        assert!(!saw_final, "no audio follows the final frame");
        saw_final = chunk.is_final;
        audio_frames += 1;
        audio_bytes = audio_bytes.saturating_add(chunk.payload.len() as u64);
        nonzero_bytes += chunk
            .payload
            .iter()
            .map(|b| u64::from(*b != 0))
            .sum::<u64>();
    }
    let completed = session
        .next_event()
        .await
        .expect("terminal event")
        .expect("completed terminal");
    assert!(matches!(
        completed.event,
        InvocationEventKind::Completed { .. }
    ));
    assert!(session.is_terminal());
    assert!(session.next_event().await.unwrap().is_none());

    assert!(
        saw_final,
        "the worker emitted a final-flagged terminal frame"
    );
    assert!(
        audio_frames > 0 && audio_bytes > 0,
        "the real worker must synthesize audio"
    );
    assert!(
        nonzero_bytes * 4 > audio_bytes,
        "synthesized audio must not be silence (nonzero {nonzero_bytes} of {audio_bytes} bytes)"
    );
    let seconds = audio_bytes as f64 / 2.0 / f64::from(spec.sample_rate);
    println!(
        "DS3 TTS-stream {backend} evidence: rate={} audio_frames={audio_frames} \
         audio_bytes={audio_bytes} approx_seconds={seconds:.2}",
        spec.sample_rate
    );

    // The attempt stays queryable over HTTP through the shared attempt table.
    let query = client
        .query_attempt(&AttemptIdentity {
            request_id: RequestId::new("real-tts-request-1").unwrap(),
            attempt_id: AttemptId::new("real-tts-attempt-1").unwrap(),
            tenant_id: TenantId::new("local-evidence").unwrap(),
            caller_id: CallerId::new("realtime-tts-evidence").unwrap(),
            incarnation_id: descriptor.incarnation_id.clone(),
            deployment_id: DeploymentId::new(deployment_id).unwrap(),
            model_generation: ModelGeneration::new(1).unwrap(),
        })
        .await
        .expect("shared attempt table resolves realtime sessions");
    assert_eq!(query.state, AttemptState::Completed);
    drop(child);
}

fn walk_real_files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read artifact dir") {
            let entry = entry.expect("artifact entry").path();
            if entry.is_dir() {
                stack.push(entry);
            } else {
                files.push(entry);
            }
        }
    }
    files.sort();
    files
}
