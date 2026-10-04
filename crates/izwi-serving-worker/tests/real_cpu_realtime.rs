//! Real CPU evidence (DS3.7): the real worker binary loading the real
//! Nemotron streaming-ASR artifact and executing genuine audio through the
//! `izwi-realtime-v1` WebSocket subprotocol — admission, live deltas, a
//! final transcript, one terminal outcome, and a clean close.
//!
//! Ignored by default: set `IZWI_REAL_ASR_MODELS_DIR` to the local models
//! root holding `Nemotron-3.5-ASR-Streaming-0.6B` and run with
//! `--ignored`. The artifact is staged read-only (symlinks plus a generated
//! deployment manifest) so the run never mutates the local cache.
//!
//! Backend selection: `IZWI_RT_EVIDENCE_BACKEND=cpu|metal` (default cpu).
//! The Metal lane additionally requires the metal-feature worker binary,
//! `IZWI_WORKER_EXPECTED_DEVICE_ID=metal:<registryID>`, and
//! `IZWI_METAL_DEVICE_ORDINAL` — same contract as the parity tests.

mod common;

use common::id;
use izwi_core::artifacts::ArtifactManifest;
use izwi_core::ModelVariant;
use izwi_serving_client::realtime::{connect, RealtimeClientConfig, RealtimeClientError};
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

/// Minimal RIFF/WAVE reader for mono 16-bit PCM fixtures.
fn read_wav_pcm16_mono(path: &Path) -> (u32, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read wav fixture");
    assert_eq!(&bytes[0..4], b"RIFF", "wav fixture must be RIFF");
    assert_eq!(&bytes[8..12], b"WAVE", "wav fixture must be WAVE");
    let mut offset = 12;
    let (mut sample_rate, mut channels, mut bits) = (0u32, 0u16, 0u16);
    let mut pcm = None;
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_len =
            u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let body = &bytes[offset + 8..(offset + 8 + chunk_len).min(bytes.len())];
        match chunk_id {
            b"fmt " => {
                sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
            }
            b"data" => pcm = Some(body.to_vec()),
            _ => {}
        }
        offset += 8 + chunk_len + (chunk_len & 1);
    }
    let pcm = pcm.expect("wav fixture carries a data chunk");
    assert_eq!((channels, bits), (1, 16), "fixture must be mono 16-bit");
    (sample_rate, pcm)
}

#[tokio::test]
#[ignore = "requires the real Nemotron ASR artifact (IZWI_REAL_ASR_MODELS_DIR)"]
async fn real_cpu_worker_streams_asr_through_the_realtime_subprotocol() {
    let models_root = std::env::var_os("IZWI_REAL_ASR_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .expect("set IZWI_REAL_ASR_MODELS_DIR to the local models root");
    let variant = ModelVariant::Nemotron35AsrStreaming06B;
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
    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_izwi-serving-worker"));
    command
        .env("IZWI_WORKER_BIND", address.to_string())
        .env("IZWI_WORKER_TASK", "speech_to_text")
        .env("IZWI_WORKER_MODEL", variant.dir_name())
        .env("IZWI_MODELS_DIR", staging.path())
        .env("IZWI_WORKER_DEPLOYMENT_ID", "nemotron-asr-cpu-v1")
        .env("IZWI_WORKER_ARTIFACT_REVISION", "local-real-artifact")
        .env("IZWI_WORKER_CREDENTIAL_ID", "real-rt-credential")
        .env("IZWI_WORKER_BEARER_TOKEN", "real-rt-secret")
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
        credential_id: id("real-rt-credential"),
        bearer_token: ServiceBearerToken::new("real-rt-secret").unwrap(),
    };
    let client = WorkerClient::new(
        &format!("http://{address}/"),
        credentials.clone(),
        WorkerClientConfig::default(),
    )
    .unwrap();

    // Startup includes the real model load plus a bounded synthetic-utterance
    // ASR warm-up: allow minutes, not seconds (Metal also compiles MSL
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
    .expect("real worker loads and warms the ASR model before the deadline");
    assert_eq!(descriptor.assignment.backend(), expected_backend);
    assert!(descriptor.features.contains(&WorkerFeature::RealtimeSocket));
    let status = client.status().await.expect("worker status");
    let deployment = status
        .deployments
        .iter()
        .find(|deployment| deployment.task == TaskKind::SpeechToText)
        .expect("speech_to_text deployment advertised");
    assert!(deployment.capability.realtime);
    assert_eq!(deployment.deployment_id.as_str(), "nemotron-asr-cpu-v1");

    // Real audio: the repo's speech fixture, mono 16-bit at its native rate.
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav");
    let (sample_rate, pcm) = read_wav_pcm16_mono(&fixture);

    let admit = RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: id("real-rt-session-1"),
        request_id: id("real-rt-request-1"),
        attempt_id: id("real-rt-attempt-1"),
        expected_worker_incarnation: descriptor.incarnation_id.clone(),
        deployment_id: id("nemotron-asr-cpu-v1"),
        expected_model_generation: ModelGeneration::new(1).unwrap(),
        caller: GatewayAttestedCallerContext {
            tenant_id: id("local-evidence"),
            caller_id: id("realtime-evidence"),
            policy_revision: id("evidence-policy-1"),
            permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
            allowed_data_regions: vec!["local".into()],
        },
        task: TaskKind::SpeechToText,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: 120_000,
        input: RealtimeStageInput::AudioStream {
            spec: RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate,
                channels: 1,
            },
            language: None,
        },
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

    let accepted = session
        .next_event()
        .await
        .expect("event stream")
        .expect("accepted event");
    assert!(matches!(
        accepted.event,
        InvocationEventKind::Accepted { .. }
    ));

    // Push 100 ms frames of real audio, then finish.
    let bytes_per_frame = usize::try_from(sample_rate).unwrap() * 2 / 10;
    let mut frames = 0usize;
    for chunk in pcm.chunks(bytes_per_frame) {
        session
            .send_audio(chunk)
            .await
            .expect("audio frame accepted");
        frames += 1;
    }
    session.finish().await.expect("finish accepted");

    let mut transcript = String::new();
    let mut deltas = 0usize;
    let mut terminal = None;
    loop {
        match session.next_event().await {
            Ok(Some(event)) => match event.event {
                InvocationEventKind::TextDelta { text, .. } => {
                    transcript.push_str(&text);
                    deltas += 1;
                }
                InvocationEventKind::Completed { .. } => {
                    terminal = Some("completed");
                    break;
                }
                InvocationEventKind::Error { code, message } => {
                    panic!("real worker failed the session: {code:?}: {message}");
                }
                other => panic!("unexpected event: {other:?}"),
            },
            Ok(None) => break,
            Err(RealtimeClientError::Protocol(message)) => {
                panic!("contract violation against the real worker: {message}");
            }
            Err(error) => panic!("session error: {error}"),
        }
    }
    assert_eq!(terminal, Some("completed"), "exactly one terminal outcome");
    assert!(session.is_terminal());
    assert!(
        !transcript.trim().is_empty(),
        "the real worker must produce a transcript for the fixture audio"
    );
    println!(
        "DS3.7 realtime {backend} evidence: frames={frames} deltas={deltas} transcript={transcript:?}"
    );

    // The attempt stays queryable over HTTP through the shared attempt table.
    let query = client
        .query_attempt(&AttemptIdentity {
            request_id: id("real-rt-request-1"),
            attempt_id: id("real-rt-attempt-1"),
            tenant_id: id("local-evidence"),
            caller_id: id("realtime-evidence"),
            incarnation_id: descriptor.incarnation_id.clone(),
            deployment_id: id("nemotron-asr-cpu-v1"),
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
