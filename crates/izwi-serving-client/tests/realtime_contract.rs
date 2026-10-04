#![cfg(feature = "mock-worker")]
//! Realtime session contract tests (T32/T33): the bounded
//! `izwi-realtime-v1` client against the deterministic mock worker over real
//! sockets. Ordering, bounds, one-terminal-outcome, owner-loss interruption,
//! stale-identity fencing, and HTTP/WS cancellation parity.

use futures::StreamExt;
use izwi_serving_client::realtime::{connect, RealtimeClientConfig, RealtimeClientError};
use izwi_serving_client::{
    mock::{MockRealtimeKnobs, MockTtsRealtimeKnobs, MockWorker, MockWorkerConfig},
    WorkerClient, WorkerClientConfig,
};
use izwi_serving_protocol::*;
use std::collections::BTreeSet;
use std::time::Duration;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value).unwrap()
}

fn credentials() -> ServiceCredentials {
    ServiceCredentials {
        credential_id: id("mock-credential-1"),
        bearer_token: ServiceBearerToken::new("mock-secret-token").unwrap(),
    }
}

fn realtime_config(knobs: MockRealtimeKnobs) -> MockWorkerConfig {
    MockWorkerConfig {
        deployment_id: id("mock-asr-v1"),
        public_model: id("mock-asr"),
        realtime: Some(knobs),
        ..MockWorkerConfig::default()
    }
}

fn tts_config(knobs: MockTtsRealtimeKnobs) -> MockWorkerConfig {
    MockWorkerConfig {
        deployment_id: id("mock-tts-v1"),
        public_model: id("mock-tts"),
        realtime_tts: Some(knobs),
        ..MockWorkerConfig::default()
    }
}

fn admission(session: &'static str, attempt: &'static str) -> RealtimeSessionAdmit {
    RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: id(session),
        request_id: id("request-1"),
        attempt_id: id(attempt),
        expected_worker_incarnation: id("mock-incarnation-1"),
        deployment_id: id("mock-asr-v1"),
        expected_model_generation: ModelGeneration::new(1).unwrap(),
        caller: GatewayAttestedCallerContext {
            tenant_id: id("tenant-1"),
            caller_id: id("caller-1"),
            policy_revision: id("policy-1"),
            permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
            allowed_data_regions: vec!["local".into()],
        },
        task: TaskKind::SpeechToText,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: 10_000,
        input: RealtimeStageInput::AudioStream {
            spec: RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate: 16_000,
                channels: 1,
            },
            language: None,
        },
    }
}

async fn established(
    endpoint: &str,
    admit: RealtimeSessionAdmit,
) -> izwi_serving_client::realtime::RealtimeSession {
    connect(
        endpoint,
        &credentials(),
        admit,
        RealtimeClientConfig {
            event_timeout: Duration::from_secs(5),
            ..RealtimeClientConfig::default()
        },
    )
    .await
    .expect("session established")
}

async fn http_client(worker: &MockWorker) -> WorkerClient {
    WorkerClient::new(
        &worker.endpoint(),
        credentials(),
        WorkerClientConfig::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn realtime_session_orders_events_and_ends_with_one_terminal() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs::default()))
        .await
        .unwrap();
    let mut session = established(&worker.endpoint(), admission("session-1", "attempt-1")).await;

    let admission = session.admission().clone();
    assert_eq!(admission.session_id, "session-1");
    assert_eq!(admission.attempt_id, "attempt-1");
    assert_eq!(admission.worker_id, "mock-worker-1");
    assert_eq!(
        admission.bounds.max_frame_bytes,
        MAX_REALTIME_AUDIO_FRAME_BYTES
    );

    let accepted = session.next_event().await.unwrap().expect("accepted");
    assert!(matches!(
        accepted.event,
        InvocationEventKind::Accepted { .. }
    ));
    assert_eq!(accepted.sequence, 0);

    for sequence in 1..=2u64 {
        session
            .send_audio(&[0i16.to_le_bytes(); 16].concat())
            .await
            .unwrap();
        let delta = session.next_event().await.unwrap().expect("delta");
        assert!(matches!(delta.event, InvocationEventKind::TextDelta { .. }));
        assert_eq!(delta.sequence, sequence);
    }

    session.finish().await.unwrap();
    let final_delta = session.next_event().await.unwrap().expect("final delta");
    let InvocationEventKind::TextDelta { text, .. } = final_delta.event else {
        panic!("expected final transcript delta");
    };
    assert_eq!(text, "mock realtime transcript");
    let completed = session.next_event().await.unwrap().expect("completed");
    assert!(matches!(
        completed.event,
        InvocationEventKind::Completed { .. }
    ));

    // Exactly one terminal outcome, then a clean close.
    assert!(session.is_terminal());
    assert!(session.next_event().await.unwrap().is_none());
    assert!(session.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn realtime_stale_incarnation_is_closed_before_admission() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs::default()))
        .await
        .unwrap();
    let admit = RealtimeSessionAdmit {
        expected_worker_incarnation: id("mock-incarnation-stale"),
        ..admission("session-1", "attempt-1")
    };
    let error = connect(
        &worker.endpoint(),
        &credentials(),
        admit,
        RealtimeClientConfig::default(),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, RealtimeClientError::ClosedBeforeAdmit(4412)),
        "unexpected error: {error}"
    );
    assert!(error.proves_session_unaccepted());
}

#[tokio::test]
async fn realtime_frames_above_the_negotiated_bound_never_leave_the_client() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs::default()))
        .await
        .unwrap();
    let mut session = established(&worker.endpoint(), admission("session-1", "attempt-1")).await;
    let oversized = vec![0u8; session.admission().bounds.max_frame_bytes + 1];
    let error = session.send_audio(&oversized).await.unwrap_err();
    assert!(matches!(
        error,
        RealtimeClientError::AudioFrameTooLarge { .. }
    ));
    // The session is still usable: a well-sized frame flows normally.
    session.send_audio(&[0, 0]).await.unwrap();
    assert!(session.next_event().await.unwrap().is_some());
}

#[tokio::test]
async fn realtime_owner_loss_interrupts_without_any_terminal_event() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs {
        disconnect_after_frames: Some(2),
        ..MockRealtimeKnobs::default()
    }))
    .await
    .unwrap();
    let mut session = established(&worker.endpoint(), admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some()); // accepted

    session.send_audio(&[0, 0]).await.unwrap();
    assert!(matches!(
        session.next_event().await.unwrap().expect("delta").event,
        InvocationEventKind::TextDelta { .. }
    ));
    // Second push triggers the abrupt owner loss: no terminal event, no
    // close frame — the transport simply ends.
    session.send_audio(&[0, 0]).await.unwrap();
    let error = session.next_event().await.unwrap_err();
    assert!(
        matches!(error, RealtimeClientError::Protocol(_)),
        "unexpected error: {error}"
    );
    assert!(!session.is_terminal());

    // The attempt remains owned in the worker's table; HTTP still resolves it.
    let client = http_client(&worker).await;
    let query = client
        .query_attempt(&AttemptIdentity {
            request_id: id("request-1"),
            attempt_id: id("attempt-1"),
            tenant_id: id("tenant-1"),
            caller_id: id("caller-1"),
            incarnation_id: id("mock-incarnation-1"),
            deployment_id: id("mock-asr-v1"),
            model_generation: ModelGeneration::new(1).unwrap(),
        })
        .await
        .unwrap();
    assert!(!query.state.is_terminal());
}

#[tokio::test]
async fn realtime_in_session_cancel_ends_with_one_cancelled_terminal() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs {
        push_cadence: Duration::from_millis(1),
        ..MockRealtimeKnobs::default()
    }))
    .await
    .unwrap();
    let mut session = established(&worker.endpoint(), admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some());
    session.send_audio(&[0, 0]).await.unwrap();
    assert!(session.next_event().await.unwrap().is_some());

    session.cancel().await.unwrap();
    let cancelled = session.next_event().await.unwrap().expect("cancelled");
    assert!(matches!(
        cancelled.event,
        InvocationEventKind::Cancelled { .. }
    ));
    assert!(session.is_terminal());
    assert!(session.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn realtime_http_cancel_reaches_the_ws_session_through_the_shared_table() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs {
        push_cadence: Duration::from_millis(1),
        ..MockRealtimeKnobs::default()
    }))
    .await
    .unwrap();
    let mut session = established(&worker.endpoint(), admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some());

    let client = http_client(&worker).await;
    let response = client
        .cancel_attempt(&AttemptIdentity {
            request_id: id("request-1"),
            attempt_id: id("attempt-1"),
            tenant_id: id("tenant-1"),
            caller_id: id("caller-1"),
            incarnation_id: id("mock-incarnation-1"),
            deployment_id: id("mock-asr-v1"),
            model_generation: ModelGeneration::new(1).unwrap(),
        })
        .await
        .unwrap();
    assert_eq!(response.disposition, CancelDisposition::Requested);

    let cancelled = session.next_event().await.unwrap().expect("cancelled");
    assert!(matches!(
        cancelled.event,
        InvocationEventKind::Cancelled { .. }
    ));
    assert!(session.is_terminal());
    assert!(session.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn realtime_attempt_id_reuse_is_closed_as_duplicate() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs::default()))
        .await
        .unwrap();
    let mut first = established(&worker.endpoint(), admission("session-1", "attempt-1")).await;
    assert!(first.next_event().await.unwrap().is_some());

    let second = connect(
        &worker.endpoint(),
        &credentials(),
        admission("session-2", "attempt-1"),
        RealtimeClientConfig::default(),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(second, RealtimeClientError::ClosedBeforeAdmit(4409)),
        "unexpected error: {second}"
    );

    first.finish().await.unwrap();
    while first.next_event().await.unwrap().is_some() {}
}

#[tokio::test]
async fn realtime_endpoint_rejects_missing_subprotocol_and_denies_policy() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs::default()))
        .await
        .unwrap();

    // No subprotocol offered: rejected at the HTTP layer.
    let mut request = format!(
        "ws://{}internal/v1/realtime",
        worker.endpoint().trim_start_matches("http://")
    )
    .into_client_request()
    .unwrap();
    use tokio_tungstenite::tungstenite::http::HeaderValue;
    let headers = request.headers_mut();
    headers.insert(
        SERVICE_AUTHORIZATION_HEADER,
        HeaderValue::from_static("Bearer mock-secret-token"),
    );
    headers.insert(
        SERVICE_CREDENTIAL_ID_HEADER,
        HeaderValue::from_static("mock-credential-1"),
    );
    let error = tokio_tungstenite::connect_async(request).await.unwrap_err();
    assert!(
        matches!(error, tokio_tungstenite::tungstenite::Error::Http(ref response)
            if response.status().as_u16() == 400),
        "unexpected error: {error}"
    );

    // Admit without the invoke permission: closed with PolicyDenied.
    let admit = RealtimeSessionAdmit {
        caller: GatewayAttestedCallerContext {
            permitted_actions: BTreeSet::new(),
            ..admission("session-1", "attempt-1").caller
        },
        ..admission("session-1", "attempt-1")
    };
    let error = connect(
        &worker.endpoint(),
        &credentials(),
        admit,
        RealtimeClientConfig::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        RealtimeClientError::ClosedBeforeAdmit(4403)
    ));

    // Chat workers built without realtime knobs never expose the route.
    let chat_worker = MockWorker::spawn(MockWorkerConfig::default())
        .await
        .unwrap();
    let error = connect(
        &chat_worker.endpoint(),
        &credentials(),
        admission("session-1", "attempt-1"),
        RealtimeClientConfig::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, RealtimeClientError::Connect(_)));

    // Keep the used worker alive until the end of the test.
    let _ = futures::stream::iter(vec![worker]).next().await;
}

// ---------------------------------------------------------------------------
// TTS-stream stage: text in, synthesized audio out (protocol minor 2).
// ---------------------------------------------------------------------------

fn tts_admission(session: &'static str, attempt: &'static str) -> RealtimeSessionAdmit {
    RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: id(session),
        request_id: id("request-1"),
        attempt_id: id(attempt),
        expected_worker_incarnation: id("mock-incarnation-1"),
        deployment_id: id("mock-tts-v1"),
        expected_model_generation: ModelGeneration::new(1).unwrap(),
        caller: GatewayAttestedCallerContext {
            tenant_id: id("tenant-1"),
            caller_id: id("caller-1"),
            policy_revision: id("policy-1"),
            permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
            allowed_data_regions: vec!["local".into()],
        },
        task: TaskKind::TextToSpeech,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: 10_000,
        input: RealtimeStageInput::TextStream,
    }
}

#[tokio::test]
async fn tts_session_streams_audio_frames_and_ends_with_one_terminal() {
    let worker = MockWorker::spawn(tts_config(MockTtsRealtimeKnobs {
        chunk_bytes: 16,
        chunk_count: 3,
        ..MockTtsRealtimeKnobs::default()
    }))
    .await
    .unwrap();
    let mut session =
        established(&worker.endpoint(), tts_admission("session-1", "attempt-1")).await;

    // The admission announces the synthesized audio spec.
    let admission = session.admission().clone();
    let spec = admission
        .output_audio
        .expect("TTS admissions announce output_audio");
    assert_eq!(spec.codec, RealtimeAudioCodec::PcmI16Le);
    assert_eq!(spec.sample_rate, 24_000);
    assert_eq!(spec.channels, 1);

    let accepted = session.next_event().await.unwrap().expect("accepted");
    assert!(matches!(
        accepted.event,
        InvocationEventKind::Accepted { .. }
    ));
    assert_eq!(accepted.sequence, 0);

    session.send_text("Hello ").await.unwrap();
    session.send_text("world").await.unwrap();
    session.finish().await.unwrap();

    // Audio frames arrive in order; the terminal frame is zero-payload with
    // the final flag; the Completed event is buffered behind the audio.
    let mut seen_final = false;
    let mut payloads = Vec::new();
    while let Some(chunk) = session.next_audio().await.unwrap() {
        assert!(!seen_final, "no audio follows the final frame");
        seen_final = chunk.is_final;
        payloads.push((chunk.sequence, chunk.payload));
    }
    assert!(seen_final, "the final frame arrived");
    assert_eq!(payloads.len(), 4, "three payload frames plus the final");
    assert_eq!(payloads[3].0, 4);
    assert!(payloads[3].1.is_empty());
    for (index, (sequence, payload)) in payloads.iter().take(3).enumerate() {
        assert_eq!(*sequence, u32::try_from(index + 1).unwrap());
        let expected: Vec<u8> = (0..16)
            .map(|position| ((index * 31 + position * 7) % 251 + 1) as u8)
            .collect();
        assert_eq!(*payload, expected, "payload frame {index} content");
    }

    let completed = session.next_event().await.unwrap().expect("completed");
    assert!(matches!(
        completed.event,
        InvocationEventKind::Completed { .. }
    ));
    assert!(session.is_terminal());
    assert!(session.next_event().await.unwrap().is_none());
    assert!(session.next_audio().await.unwrap().is_none());
}

#[tokio::test]
async fn tts_session_client_rejects_text_bound_violations_locally() {
    let worker = MockWorker::spawn(tts_config(MockTtsRealtimeKnobs::default()))
        .await
        .unwrap();
    let mut session =
        established(&worker.endpoint(), tts_admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some());

    let oversized = "x".repeat(MAX_REALTIME_INPUT_TEXT_BYTES + 1);
    let error = session.send_text(&oversized).await.unwrap_err();
    assert!(matches!(error, RealtimeClientError::Protocol(_)));
    // Nothing left the process: a valid piece still flows and the session
    // completes normally.
    session.send_text("ok").await.unwrap();
    session.finish().await.unwrap();
    while session.next_audio().await.unwrap().is_some() {}
    assert!(session.is_terminal());
}

#[tokio::test]
async fn tts_session_in_session_cancel_ends_with_one_cancelled_terminal() {
    let worker = MockWorker::spawn(tts_config(MockTtsRealtimeKnobs::default()))
        .await
        .unwrap();
    let mut session =
        established(&worker.endpoint(), tts_admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some());
    session.send_text("hi").await.unwrap();

    // Cancelling before the finish-commit reaches the stage's select loop,
    // which answers with exactly one Cancelled terminal.
    session.cancel().await.unwrap();
    loop {
        match session
            .next_event()
            .await
            .unwrap()
            .expect("terminal event")
            .event
        {
            InvocationEventKind::Cancelled { .. } => break,
            _ => continue,
        }
    }
    assert!(session.is_terminal());
    assert!(session.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn tts_owner_loss_interrupts_without_any_terminal_event() {
    let worker = MockWorker::spawn(tts_config(MockTtsRealtimeKnobs {
        disconnect_after_frames: Some(2),
        ..MockTtsRealtimeKnobs::default()
    }))
    .await
    .unwrap();
    let mut session =
        established(&worker.endpoint(), tts_admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some());

    session.send_text("hi").await.unwrap();
    session.finish().await.unwrap();
    // Payload frames flow until the disconnect threshold fires before the
    // final frame; then the transport simply ends.
    loop {
        match session.next_audio().await {
            Ok(Some(_)) => continue,
            Ok(None) => panic!("session ended cleanly without a terminal outcome"),
            Err(error) => {
                assert!(
                    matches!(error, RealtimeClientError::Protocol(_)),
                    "unexpected error: {error}"
                );
                break;
            }
        }
    }
    assert!(!session.is_terminal());

    // The attempt remains owned in the worker's table; HTTP still resolves it.
    let client = http_client(&worker).await;
    let query = client
        .query_attempt(&AttemptIdentity {
            request_id: id("request-1"),
            attempt_id: id("attempt-1"),
            tenant_id: id("tenant-1"),
            caller_id: id("caller-1"),
            incarnation_id: id("mock-incarnation-1"),
            deployment_id: id("mock-tts-v1"),
            model_generation: ModelGeneration::new(1).unwrap(),
        })
        .await
        .unwrap();
    assert!(!query.state.is_terminal());
}

#[tokio::test]
async fn tts_admit_on_an_asr_worker_is_incompatible() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs::default()))
        .await
        .unwrap();
    // The admit targets the ASR worker's deployment with the TTS task, so the
    // fencing ladder reaches the task check and answers IncompatibleTask.
    let admit = RealtimeSessionAdmit {
        deployment_id: id("mock-asr-v1"),
        ..tts_admission("session-1", "attempt-1")
    };
    let error = connect(
        &worker.endpoint(),
        &credentials(),
        admit,
        RealtimeClientConfig::default(),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, RealtimeClientError::ClosedBeforeAdmit(4413)),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn tts_http_cancel_reaches_the_ws_session_through_the_shared_table() {
    // A long emission window guarantees the HTTP cancel lands mid-stream.
    let worker = MockWorker::spawn(tts_config(MockTtsRealtimeKnobs {
        chunk_count: 10,
        push_cadence: Duration::from_millis(20),
        ..MockTtsRealtimeKnobs::default()
    }))
    .await
    .unwrap();
    let mut session =
        established(&worker.endpoint(), tts_admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some());
    session.send_text("hi").await.unwrap();
    session.finish().await.unwrap();

    let client = http_client(&worker).await;
    let response = client
        .cancel_attempt(&AttemptIdentity {
            request_id: id("request-1"),
            attempt_id: id("attempt-1"),
            tenant_id: id("tenant-1"),
            caller_id: id("caller-1"),
            incarnation_id: id("mock-incarnation-1"),
            deployment_id: id("mock-tts-v1"),
            model_generation: ModelGeneration::new(1).unwrap(),
        })
        .await
        .unwrap();
    assert_eq!(response.disposition, CancelDisposition::Requested);

    loop {
        match session
            .next_event()
            .await
            .unwrap()
            .expect("terminal event")
            .event
        {
            InvocationEventKind::Cancelled { .. } => break,
            _ => continue,
        }
    }
    assert!(session.is_terminal());
    assert!(session.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn tts_session_empty_finish_completes_without_audio() {
    let worker = MockWorker::spawn(tts_config(MockTtsRealtimeKnobs::default()))
        .await
        .unwrap();
    let mut session =
        established(&worker.endpoint(), tts_admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some());
    session.finish().await.unwrap();
    let completed = session.next_event().await.unwrap().expect("completed");
    assert!(matches!(
        completed.event,
        InvocationEventKind::Completed { .. }
    ));
    assert!(session.is_terminal());
    assert!(session.next_audio().await.unwrap().is_none());
}

#[tokio::test]
async fn text_input_is_rejected_on_audio_stream_sessions_before_writing() {
    let worker = MockWorker::spawn(realtime_config(MockRealtimeKnobs::default()))
        .await
        .unwrap();
    let mut session = established(&worker.endpoint(), admission("session-1", "attempt-1")).await;
    assert!(session.next_event().await.unwrap().is_some());
    let error = session.send_text("hi").await.unwrap_err();
    assert!(matches!(error, RealtimeClientError::Protocol(_)));
    // The ASR session is unaffected: audio still flows.
    session.send_audio(&[0, 0]).await.unwrap();
    assert!(session.next_event().await.unwrap().is_some());
}
