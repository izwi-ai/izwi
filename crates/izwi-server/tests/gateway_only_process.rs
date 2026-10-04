//! Process-level T01 (serving plan §16.1): the gateway binary starts, serves
//! its probes, and routes an inference request through an approved worker —
//! without constructing an engine. The negative control is structural and
//! observable: the process starts with no model artifacts or models directory
//! anywhere, which is only possible because gateway mode owns no
//! RuntimeService. An inference request succeeding through the mock worker
//! proves the request was executed remotely, not locally.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use izwi_core::ModelVariant;
use izwi_serving_client::mock::{MockWorker, MockWorkerConfig};
use izwi_serving_protocol::ModelAlias;

const GATEWAY_API_KEY: &str = "t01-gateway-api-key";

#[tokio::test]
async fn gateway_process_serves_probes_and_remote_inference_without_an_engine() {
    let model = ModelVariant::Qwen354BGguf;
    let model_alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    let worker_config = MockWorkerConfig {
        public_model: model_alias,
        output_text: "gateway-process-t01".to_string(),
        // The CLI dispatcher requires capacity for its 4096-token output
        // budget; the mock's 1024-token default would fail eligibility.
        max_output_tokens: 8_192,
        // DS9.1: the worker reports managed-prefix cached tokens through
        // protocol minor 3 and the gateway must surface them in the public
        // OpenAI usage shape.
        usage_cached_input_tokens: Some(64),
        // DS9.3: the worker reports per-token logprobs through protocol
        // minor 3; the gateway must map them into the public choice shape.
        delta_logprobs: Some(vec![izwi_serving_protocol::TokenLogprob {
            token: "gateway".into(),
            logprob: -0.75,
            bytes: b"gateway".to_vec(),
            top_logprobs: vec![izwi_serving_protocol::TopTokenLogprob {
                token: "gateway".into(),
                logprob: -0.75,
                bytes: b"gateway".to_vec(),
            }],
        }]),
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(worker_config)
        .await
        .expect("mock worker should bind a real loopback TCP listener");

    // Reserve a port the same way an operator would: bind, learn it, release.
    let port = {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe listener binds");
        probe.local_addr().expect("probe addr").port()
    };
    let _endpoint = format!("http://127.0.0.1:{port}");

    // Standalone compatibility approval: URL|TASK|PUBLIC_MODEL|DEPLOYMENT|GEN.
    // The mock's static deployment identity is mock-chat-v1 @ generation 1.
    let approval = format!(
        "{}|chat|{}|mock-chat-v1|1",
        worker.endpoint(),
        model.dir_name()
    );

    let mut gateway = Command::new(env!("CARGO_BIN_EXE_izwi-server"))
        .args([
            "--role",
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--public-model",
            model.dir_name(),
            "--gateway-worker-approval",
            &approval,
        ])
        .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
        .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
        .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
        .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
        .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gateway binary spawns");
    // Drain diagnostics so bounded startup/restart logs cannot fill the pipe.
    if let Some(stderr) = gateway.stderr.take() {
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for _ in BufReader::new(stderr).lines() {}
        });
    }

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);

    // /livez answers while the gateway is running.
    let livez = loop {
        match client.get(format!("{base}/livez")).send().await {
            Ok(response) => break response,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await
            }
            Err(error) => panic!("gateway never answered /livez: {error}"),
        }
    };
    assert_eq!(livez.status(), 200, "liveness must answer 200");

    // /readyz covers registry-backed readiness: the approved worker answered
    // its initial authenticated descriptor/status exchange.
    let readyz = loop {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => break response,
            _ if Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(100)).await,
            other => panic!(
                "gateway never became ready: {:?}",
                other.map(|r| r.status())
            ),
        }
    };
    assert_eq!(readyz.status(), 200);

    // Inference: authenticated request must be executed by the mock worker and
    // relayed with the exact public JSON shape (no local engine is possible:
    // the process started without any model artifacts on disk).
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {GATEWAY_API_KEY}"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("chat request reaches the gateway");
    assert_eq!(
        response.status(),
        200,
        "remote chat request must succeed: {}",
        response.text().await.unwrap_or_default()
    );
    let body: serde_json::Value = response.json().await.expect("JSON completion body");
    assert_eq!(
        body["object"], "chat.completion",
        "public completion shape must be preserved through the worker boundary"
    );
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("message content present");
    assert!(
        content.contains("gateway-process-t01"),
        "output must come from the approved worker, got {content:?}"
    );
    assert_eq!(
        body["usage"]["prompt_tokens_details"]["cached_tokens"], 64,
        "worker cached-token measurement must reach the public OpenAI usage shape"
    );
    assert_eq!(
        body["usage"]["prompt_tokens"].as_u64(),
        Some(1),
        "cached tokens are a subset of prompt tokens"
    );
    let logprobs = &body["choices"][0]["logprobs"]["content"];
    assert_eq!(
        logprobs[0]["token"], "gateway",
        "worker logprobs must reach the public OpenAI choice shape: {body}"
    );
    assert_eq!(logprobs[0]["logprob"], -0.75);
    assert_eq!(
        logprobs[0]["bytes"],
        serde_json::json!([103, 97, 116, 101, 119, 97, 121])
    );
    assert_eq!(logprobs[0]["top_logprobs"][0]["token"], "gateway");

    // Unauthenticated callers are rejected at the perimeter.
    let rejected = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("unauthenticated request reaches the gateway");
    assert_eq!(
        rejected.status(),
        401,
        "missing credentials must be rejected"
    );

    let _ = gateway.kill();
    let _ = gateway.wait();
}

const SCOPED_INFERENCE_KEY: &str = "ds05-scoped-inference-key";
const SCOPED_OPS_KEY: &str = "ds05-scoped-ops-metrics-key";

/// DS0.5 process leg: scoped per-principal keys are provisioned from a bounded
/// manifest into the durable store at boot, authenticate through the real
/// gateway perimeter with role and tenant semantics, and leave the shared
/// root key fully backward compatible.
#[tokio::test]
async fn gateway_process_authenticates_scoped_principal_keys_from_the_durable_store() {
    let model = ModelVariant::Qwen354BGguf;
    let model_alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    let worker_config = MockWorkerConfig {
        public_model: model_alias,
        output_text: "gateway-process-ds05".to_string(),
        max_output_tokens: 8_192,
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(worker_config)
        .await
        .expect("mock worker should bind a real loopback TCP listener");

    let storage = tempfile::tempdir().expect("temp storage dir");
    let manifest_path = storage.path().join("principals.json");
    std::fs::write(
        &manifest_path,
        r#"{"version":1,"principals":[
                {"principal_id":"svc-alpha","roles":["inference"],"tenant_id":"tenant-alpha","key_ref":"env:IZWI_TEST_PROCESS_SCOPED_A"},
                {"principal_id":"ops","roles":["metrics","admin"],"key_ref":"env:IZWI_TEST_PROCESS_SCOPED_B"}]}"#,
    )
    .expect("manifest write");

    let port = {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe listener binds");
        probe.local_addr().expect("probe addr").port()
    };
    let approval = format!(
        "{}|chat|{}|mock-chat-v1|1",
        worker.endpoint(),
        model.dir_name()
    );

    let mut gateway = Command::new(env!("CARGO_BIN_EXE_izwi-server"))
        .args([
            "--role",
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--public-model",
            model.dir_name(),
            "--gateway-worker-approval",
            &approval,
        ])
        .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
        .env(
            "IZWI_GATEWAY_PRINCIPAL_KEYS_MANIFEST",
            manifest_path.display().to_string(),
        )
        .env(
            "IZWI_DB_PATH",
            storage.path().join("gateway.sqlite3").display().to_string(),
        )
        .env("IZWI_TEST_PROCESS_SCOPED_A", SCOPED_INFERENCE_KEY)
        .env("IZWI_TEST_PROCESS_SCOPED_B", SCOPED_OPS_KEY)
        .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
        .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
        .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
        .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gateway binary spawns");
    if let Some(stderr) = gateway.stderr.take() {
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for _ in BufReader::new(stderr).lines() {}
        });
    }

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);
    let readyz = loop {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => break response,
            _ if Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(100)).await,
            other => panic!(
                "gateway never became ready: {:?}",
                other.map(|r| r.status())
            ),
        }
    };
    assert_eq!(readyz.status(), 200);

    // A scoped inference principal executes remotely like the root key does.
    let scoped = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {SCOPED_INFERENCE_KEY}"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("scoped chat request reaches the gateway");
    assert_eq!(
        scoped.status(),
        200,
        "scoped inference key must be admitted: {}",
        scoped.text().await.unwrap_or_default()
    );
    let body: serde_json::Value = scoped.json().await.expect("JSON completion body");
    assert_eq!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("message content present"),
        "gateway-process-ds05",
        "scoped principal request must execute through the worker"
    );

    // The shared root key remains valid as the bootstrap root principal.
    let root = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {GATEWAY_API_KEY}"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("root-key chat request reaches the gateway");
    assert_eq!(root.status(), 200, "root key must stay backward compatible");

    // A scoped principal without the inference role is forbidden, and unknown
    // credentials are rejected at the perimeter exactly as before.
    let forbidden = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {SCOPED_OPS_KEY}"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("ops-key chat request reaches the gateway");
    assert_eq!(
        forbidden.status(),
        403,
        "a metrics/admin-only principal must not run inference"
    );
    let rejected = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", "Bearer ds05-unknown-key-000000")
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("unknown-key chat request reaches the gateway");
    assert_eq!(rejected.status(), 401);

    // A scoped metrics principal opens the internal metrics side-channel even
    // though no dedicated metrics credential was configured.
    let metrics = client
        .get(format!("{base}/internal/metrics"))
        .header("Authorization", format!("Bearer {SCOPED_OPS_KEY}"))
        .send()
        .await
        .expect("ops-key metrics request reaches the gateway");
    assert_eq!(
        metrics.status(),
        200,
        "a metrics-role scoped principal must open the metrics endpoint"
    );

    let _ = gateway.kill();
    let _ = gateway.wait();
}

/// Process-level DS3.6 leg: with IZWI_GATEWAY_REALTIME=on the gateway
/// exposes /v1/realtime/ws, authenticates the public client, selects the
/// approved speech_to_text worker, and relays an izwi-realtime-v1 session
/// end to end. With the flag off (the T01 gateway above), the route 404s.
#[tokio::test]
async fn gateway_realtime_relay_relays_v1_sessions_through_the_real_binary() {
    use futures::{SinkExt, StreamExt};
    use izwi_serving_protocol::{
        decode_realtime_audio_frame, encode_realtime_audio_frame, AttemptId, CallerId,
        GatewayAttestedCallerContext, InvocationEventKind, ModelGeneration, PermittedAction,
        PolicyRevision, RealtimeAudioCodec, RealtimeAudioSpec, RealtimeClientFrame,
        RealtimeServerFrame, RealtimeSessionAdmit, RealtimeStageInput, RequestId, ServiceClass,
        SessionId, TaskKind, TenantId, PROTOCOL_V1, REALTIME_SUBPROTOCOL,
    };
    use std::collections::BTreeSet;
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, http::HeaderValue, Message};

    let model = ModelVariant::Qwen354BGguf;
    let model_alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    // The gateway always boots its chat route, so one chat worker is
    // approved alongside the realtime ASR worker under test.
    let chat_worker = MockWorker::spawn(MockWorkerConfig {
        worker_id: izwi_serving_protocol::WorkerId::try_from("mock-chat-worker").expect("identity"),
        node_id: izwi_serving_protocol::NodeId::try_from("mock-chat-node").expect("identity"),
        public_model: model_alias.clone(),
        ..MockWorkerConfig::default()
    })
    .await
    .expect("mock chat worker binds");
    let worker = MockWorker::spawn(MockWorkerConfig {
        worker_id: izwi_serving_protocol::WorkerId::try_from("mock-asr-worker").expect("identity"),
        node_id: izwi_serving_protocol::NodeId::try_from("mock-asr-node").expect("identity"),
        deployment_id: izwi_serving_protocol::DeploymentId::try_from("mock-asr-v1")
            .expect("static identity"),
        public_model: model_alias,
        realtime: Some(izwi_serving_client::mock::MockRealtimeKnobs {
            delta_per_frame: "partial ".into(),
            final_text: "gateway relayed transcript".into(),
            push_cadence: Duration::from_millis(1),
            disconnect_after_frames: None,
        }),
        ..MockWorkerConfig::default()
    })
    .await
    .expect("mock realtime worker binds");

    let port = {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe listener binds");
        probe.local_addr().expect("probe addr").port()
    };
    let approval = format!(
        "{}|speech_to_text|{}|mock-asr-v1|1",
        worker.endpoint(),
        model.dir_name()
    );

    let mut gateway = Command::new(env!("CARGO_BIN_EXE_izwi-server"))
        .args([
            "--role",
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--public-model",
            model.dir_name(),
            "--gateway-worker-approval",
            &format!(
                "{}|chat|{}|mock-chat-v1|1",
                chat_worker.endpoint(),
                model.dir_name()
            ),
            "--gateway-worker-approval",
            &approval,
            "--gateway-realtime",
            "on",
        ])
        .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
        .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
        .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
        .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
        .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gateway binary spawns");
    let stderr_capture = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let stderr_store = std::sync::Arc::clone(&stderr_capture);
    if let Some(stderr) = gateway.stderr.take() {
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(mut collected) = stderr_store.lock() {
                    collected.push_str(&line);
                    collected.push('\n');
                }
            }
        });
    }

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut ready = false;
    while Instant::now() < deadline {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => {
                ready = true;
                break;
            }
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
    if !ready {
        let _ = gateway.kill();
        let _ = gateway.wait();
        panic!(
            "gateway never became ready; stderr:\n{}",
            stderr_capture
                .lock()
                .map(|buf| buf.clone())
                .unwrap_or_default()
        );
    }

    // Public client: the perimeter API key, the v1 subprotocol, and the same
    // admit shape a direct worker client sends.
    let mut request = format!("ws://127.0.0.1:{port}/v1/realtime/ws")
        .into_client_request()
        .expect("websocket request");
    let headers = request.headers_mut();
    headers.insert(
        "sec-websocket-protocol",
        HeaderValue::from_static(REALTIME_SUBPROTOCOL),
    );
    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {GATEWAY_API_KEY}")).unwrap(),
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(request)
        .await
        .expect("gateway upgrades the realtime session");

    let admit = RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: SessionId::try_from("gateway-session-1").expect("identity"),
        request_id: RequestId::try_from("gateway-request-1").expect("identity"),
        attempt_id: AttemptId::try_from("gateway-attempt-1").expect("identity"),
        expected_worker_incarnation: izwi_serving_protocol::IncarnationId::try_from("ignored")
            .expect("identity"),
        deployment_id: izwi_serving_protocol::DeploymentId::try_from("mock-asr-v1")
            .expect("identity"),
        expected_model_generation: ModelGeneration::new(1).unwrap(),
        caller: GatewayAttestedCallerContext {
            tenant_id: TenantId::try_from("spoofed-tenant").expect("identity"),
            caller_id: CallerId::try_from("spoofed-caller").expect("identity"),
            policy_revision: PolicyRevision::try_from("spoofed-policy").expect("identity"),
            permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
            allowed_data_regions: vec![],
        },
        task: TaskKind::SpeechToText,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: 60_000,
        input: RealtimeStageInput::AudioStream {
            spec: RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate: 16_000,
                channels: 1,
            },
            language: None,
        },
    };
    ws.send(Message::Text(
        serde_json::to_string(&RealtimeClientFrame::Admit {
            admit: Box::new(admit),
        })
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();

    async fn next_server_frame(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) -> RealtimeServerFrame {
        loop {
            let message = tokio::time::timeout(Duration::from_secs(10), ws.next())
                .await
                .expect("frame arrives")
                .expect("stream open")
                .expect("no transport error");
            match message {
                Message::Text(text) => {
                    return serde_json::from_str::<RealtimeServerFrame>(&text)
                        .unwrap_or_else(|error| panic!("server frame decodes: {error}: {text}"))
                }
                Message::Close(frame) => panic!("unexpected close: {frame:?}"),
                _ => continue,
            }
        }
    }

    let admitted = next_server_frame(&mut ws).await;
    let RealtimeServerFrame::Admitted { session_id, .. } = &admitted else {
        panic!("expected admitted frame, got {admitted:?}");
    };
    assert_eq!(session_id.as_str(), "gateway-session-1");

    let accepted = next_server_frame(&mut ws).await;
    let RealtimeServerFrame::Event { event } = &accepted else {
        panic!("expected accepted event");
    };
    assert!(matches!(event.event, InvocationEventKind::Accepted { .. }));

    for sequence in 1..=2u32 {
        ws.send(Message::Binary(
            encode_realtime_audio_frame(sequence, false, &[0i16.to_le_bytes(); 8].concat())
                .unwrap()
                .into(),
        ))
        .await
        .unwrap();
        let frame = next_server_frame(&mut ws).await;
        let RealtimeServerFrame::Event { event } = frame else {
            panic!("expected delta event, got {frame:?}");
        };
        assert!(matches!(event.event, InvocationEventKind::TextDelta { .. }));
    }

    ws.send(Message::Text(
        serde_json::to_string(&RealtimeClientFrame::Finish)
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();
    let final_delta = next_server_frame(&mut ws).await;
    let RealtimeServerFrame::Event { event } = final_delta else {
        panic!("expected final delta");
    };
    let InvocationEventKind::TextDelta { text, .. } = event.event else {
        panic!("expected final transcript delta, got {:?}", event.event);
    };
    assert_eq!(text, "gateway relayed transcript");
    let completed = next_server_frame(&mut ws).await;
    let RealtimeServerFrame::Event { event } = completed else {
        panic!("expected completed");
    };
    assert!(matches!(event.event, InvocationEventKind::Completed { .. }));

    // One terminal outcome, then a clean close from the worker path.
    let close = loop {
        let message = tokio::time::timeout(Duration::from_secs(10), ws.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Message::Close(frame) = message {
            break frame.expect("close frame");
        }
    };
    assert_eq!(u16::from(close.code), 1000);

    // Decoding sanity on the audio path the relay forwards (header passes
    // through verbatim).
    let probe = encode_realtime_audio_frame(9, false, &[1, 2, 3, 4]).unwrap();
    let (_header, payload) = decode_realtime_audio_frame(&probe).unwrap();
    assert_eq!(payload, &[1, 2, 3, 4]);

    let _ = gateway.kill();
    let _ = gateway.wait();
}

/// Process-level TTS-stream leg: with IZWI_GATEWAY_REALTIME=on and an
/// approved text_to_speech worker, the gateway resolves the admit's task to
/// the TTS pool, forwards Input text frames downstream, and passes worker
/// audio frames back to the client verbatim.
#[tokio::test]
async fn gateway_realtime_relay_relays_tts_stage_sessions_through_the_real_binary() {
    use futures::{SinkExt, StreamExt};
    use izwi_serving_protocol::{
        decode_realtime_audio_frame, AttemptId, CallerId, GatewayAttestedCallerContext,
        InvocationEventKind, ModelGeneration, PermittedAction, PolicyRevision,
        RealtimeClientFrame, RealtimeServerFrame, RealtimeSessionAdmit, RealtimeStageInput,
        RequestId, ServiceClass, SessionId, TaskKind, TenantId, PROTOCOL_V1, REALTIME_SUBPROTOCOL,
    };
    use std::collections::BTreeSet;
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, http::HeaderValue, Message};

    let model = ModelVariant::Qwen354BGguf;
    let model_alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    let chat_worker = MockWorker::spawn(MockWorkerConfig {
        worker_id: izwi_serving_protocol::WorkerId::try_from("mock-chat-worker").expect("identity"),
        node_id: izwi_serving_protocol::NodeId::try_from("mock-chat-node").expect("identity"),
        public_model: model_alias.clone(),
        ..MockWorkerConfig::default()
    })
    .await
    .expect("mock chat worker binds");
    let tts_worker = MockWorker::spawn(MockWorkerConfig {
        worker_id: izwi_serving_protocol::WorkerId::try_from("mock-tts-worker").expect("identity"),
        node_id: izwi_serving_protocol::NodeId::try_from("mock-tts-node").expect("identity"),
        deployment_id: izwi_serving_protocol::DeploymentId::try_from("mock-tts-v1")
            .expect("static identity"),
        public_model: model_alias,
        realtime_tts: Some(izwi_serving_client::mock::MockTtsRealtimeKnobs {
            chunk_bytes: 32,
            chunk_count: 2,
            push_cadence: Duration::from_millis(1),
            output_sample_rate: 24_000,
            disconnect_after_frames: None,
        }),
        ..MockWorkerConfig::default()
    })
    .await
    .expect("mock tts worker binds");

    let port = {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe listener binds");
        probe.local_addr().expect("probe addr").port()
    };
    let mut gateway = Command::new(env!("CARGO_BIN_EXE_izwi-server"))
        .args([
            "--role",
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--public-model",
            model.dir_name(),
            "--gateway-worker-approval",
            &format!(
                "{}|chat|{}|mock-chat-v1|1",
                chat_worker.endpoint(),
                model.dir_name()
            ),
            "--gateway-worker-approval",
            &format!(
                "{}|text_to_speech|{}|mock-tts-v1|1",
                tts_worker.endpoint(),
                model.dir_name()
            ),
            "--gateway-realtime",
            "on",
        ])
        .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
        .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
        .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
        .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
        .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gateway binary spawns");
    let stderr_capture = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let stderr_store = std::sync::Arc::clone(&stderr_capture);
    if let Some(stderr) = gateway.stderr.take() {
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(mut collected) = stderr_store.lock() {
                    collected.push_str(&line);
                    collected.push('\n');
                }
            }
        });
    }

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut ready = false;
    while Instant::now() < deadline {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => {
                ready = true;
                break;
            }
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
    if !ready {
        let _ = gateway.kill();
        let _ = gateway.wait();
        panic!(
            "gateway never became ready; stderr:\n{}",
            stderr_capture
                .lock()
                .map(|buf| buf.clone())
                .unwrap_or_default()
        );
    }

    let mut request = format!("ws://127.0.0.1:{port}/v1/realtime/ws")
        .into_client_request()
        .expect("websocket request");
    let headers = request.headers_mut();
    headers.insert(
        "sec-websocket-protocol",
        HeaderValue::from_static(REALTIME_SUBPROTOCOL),
    );
    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {GATEWAY_API_KEY}")).unwrap(),
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(request)
        .await
        .expect("gateway upgrades the realtime session");

    let admit = RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: SessionId::try_from("gateway-tts-session-1").expect("identity"),
        request_id: RequestId::try_from("gateway-tts-request-1").expect("identity"),
        attempt_id: AttemptId::try_from("gateway-tts-attempt-1").expect("identity"),
        expected_worker_incarnation: izwi_serving_protocol::IncarnationId::try_from("ignored")
            .expect("identity"),
        deployment_id: izwi_serving_protocol::DeploymentId::try_from("mock-tts-v1")
            .expect("identity"),
        expected_model_generation: ModelGeneration::new(1).unwrap(),
        caller: GatewayAttestedCallerContext {
            tenant_id: TenantId::try_from("spoofed-tenant").expect("identity"),
            caller_id: CallerId::try_from("spoofed-caller").expect("identity"),
            policy_revision: PolicyRevision::try_from("spoofed-policy").expect("identity"),
            permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
            allowed_data_regions: vec![],
        },
        task: TaskKind::TextToSpeech,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: 60_000,
        input: RealtimeStageInput::TextStream,
    };
    ws.send(Message::Text(
        serde_json::to_string(&RealtimeClientFrame::Admit {
            admit: Box::new(admit),
        })
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();

    async fn next_tts_frame(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) -> Result<RealtimeServerFrame, (u32, bool, Vec<u8>)> {
        loop {
            let message = tokio::time::timeout(Duration::from_secs(10), ws.next())
                .await
                .expect("frame arrives")
                .expect("stream open")
                .expect("no transport error");
            match message {
                Message::Text(text) => {
                    return Ok(serde_json::from_str::<RealtimeServerFrame>(&text)
                        .unwrap_or_else(|error| panic!("server frame decodes: {error}: {text}")))
                }
                Message::Binary(data) => {
                    let (header, payload) =
                        decode_realtime_audio_frame(&data).expect("worker audio frame decodes");
                    return Err((header.sequence, header.is_final, payload.to_vec()));
                }
                Message::Close(frame) => panic!("unexpected close: {frame:?}"),
                _ => continue,
            }
        }
    }

    let admitted = next_tts_frame(&mut ws).await.unwrap();
    let RealtimeServerFrame::Admitted { output_audio, .. } = &admitted else {
        panic!("expected admitted frame, got {admitted:?}");
    };
    let spec = output_audio.expect("TTS admission announces the output spec");
    assert_eq!(spec.sample_rate, 24_000);

    let accepted = next_tts_frame(&mut ws).await.unwrap();
    let RealtimeServerFrame::Event { event } = &accepted else {
        panic!("expected accepted event");
    };
    assert!(matches!(event.event, InvocationEventKind::Accepted { .. }));

    ws.send(Message::Text(
        serde_json::to_string(&RealtimeClientFrame::Input {
            text: "Hello ".into(),
        })
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();
    ws.send(Message::Text(
        serde_json::to_string(&RealtimeClientFrame::Input {
            text: "world".into(),
        })
        .unwrap()
        .into(),
    ))
    .await
    .unwrap();
    ws.send(Message::Text(
        serde_json::to_string(&RealtimeClientFrame::Finish)
            .unwrap()
            .into(),
    ))
    .await
    .unwrap();

    // Two payload frames then the zero-payload final frame, forwarded
    // verbatim by the relay, then the Completed terminal.
    let mut audio = Vec::new();
    loop {
        match next_tts_frame(&mut ws).await {
            Err((sequence, is_final, payload)) => audio.push((sequence, is_final, payload)),
            Ok(RealtimeServerFrame::Event { event }) => match event.event {
                InvocationEventKind::Completed { .. } => break,
                other => panic!("expected completed, got {other:?}"),
            },
            Ok(frame) => panic!("unexpected frame: {frame:?}"),
        }
    }
    assert_eq!(audio.len(), 3, "two payload frames plus the final frame");
    assert_eq!(audio[0].0, 1);
    assert!(!audio[0].1);
    assert_eq!(audio[0].2.len(), 32);
    assert_eq!(audio[1].0, 2);
    assert!(!audio[1].1);
    assert_eq!(audio[2].0, 3);
    assert!(audio[2].1);
    assert!(audio[2].2.is_empty());

    let close = loop {
        let message = tokio::time::timeout(Duration::from_secs(10), ws.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Message::Close(frame) = message {
            break frame.expect("close frame");
        }
    };
    assert_eq!(u16::from(close.code), 1000);

    let _ = gateway.kill();
    let _ = gateway.wait();
}

/// Builds one public transcription-realtime client audio frame (ITRW:
/// magic, version 1, kind 1 client PCM16, sample rate at 8..12, frame
/// sequence at 12..16, payload).
fn itrw_public_frame(frame_seq: u32, sample_rate: u32, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(16 + payload.len());
    frame.extend_from_slice(b"ITRW");
    frame.push(1);
    frame.push(1);
    frame.extend_from_slice(&0_u16.to_le_bytes());
    frame.extend_from_slice(&sample_rate.to_le_bytes());
    frame.extend_from_slice(&frame_seq.to_le_bytes());
    frame.extend_from_slice(payload);
    frame
}

type PublicWsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Next text frame from the public socket, or None on close.
async fn next_public_text(ws: &mut PublicWsStream) -> Option<String> {
    use futures::StreamExt;
    use tokio_tungstenite::tungstenite::Message;
    loop {
        let message = tokio::time::timeout(Duration::from_secs(10), ws.next())
            .await
            .expect("frame arrives")
            .expect("stream open")
            .expect("no transport error");
        match message {
            Message::Text(text) => return Some(text.to_string()),
            Message::Close(_) => return None,
            _ => continue,
        }
    }
}

/// Process-level DS3.4 leg (public legacy v2 client): a client of the
/// single-node `/v1/speech-to-text/realtime/ws` surface repoints at the
/// gateway alias unchanged — no subprotocol, legacy JSON session lifecycle,
/// ITRW audio frames. The gateway translates onto the worker subprotocol and
/// maps the worker transcript back onto the public vocabulary.
#[tokio::test]
async fn gateway_realtime_translate_serves_public_v2_transcription_clients() {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, http::HeaderValue, Message};

    let model = ModelVariant::Qwen354BGguf;
    let model_alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    let chat_worker = MockWorker::spawn(MockWorkerConfig {
        worker_id: izwi_serving_protocol::WorkerId::try_from("mock-chat-worker").expect("identity"),
        node_id: izwi_serving_protocol::NodeId::try_from("mock-chat-node").expect("identity"),
        public_model: model_alias.clone(),
        ..MockWorkerConfig::default()
    })
    .await
    .expect("mock chat worker binds");
    let worker = MockWorker::spawn(MockWorkerConfig {
        worker_id: izwi_serving_protocol::WorkerId::try_from("mock-asr-worker").expect("identity"),
        node_id: izwi_serving_protocol::NodeId::try_from("mock-asr-node").expect("identity"),
        deployment_id: izwi_serving_protocol::DeploymentId::try_from("mock-asr-v1")
            .expect("static identity"),
        public_model: model_alias,
        realtime: Some(izwi_serving_client::mock::MockRealtimeKnobs {
            delta_per_frame: "partial ".into(),
            final_text: "gateway translated transcript".into(),
            push_cadence: Duration::from_millis(1),
            disconnect_after_frames: None,
        }),
        ..MockWorkerConfig::default()
    })
    .await
    .expect("mock realtime worker binds");

    let port = {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe listener binds");
        probe.local_addr().expect("probe addr").port()
    };

    let mut gateway = Command::new(env!("CARGO_BIN_EXE_izwi-server"))
        .args([
            "--role",
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--public-model",
            model.dir_name(),
            "--gateway-worker-approval",
            &format!(
                "{}|chat|{}|mock-chat-v1|1",
                chat_worker.endpoint(),
                model.dir_name()
            ),
            "--gateway-worker-approval",
            &format!(
                "{}|speech_to_text|{}|mock-asr-v1|1",
                worker.endpoint(),
                model.dir_name()
            ),
            "--gateway-realtime",
            "on",
        ])
        .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
        .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
        .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
        .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
        .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gateway binary spawns");
    let stderr_capture = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let stderr_store = std::sync::Arc::clone(&stderr_capture);
    if let Some(stderr) = gateway.stderr.take() {
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(mut collected) = stderr_store.lock() {
                    collected.push_str(&line);
                    collected.push('\n');
                }
            }
        });
    }

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut ready = false;
    while Instant::now() < deadline {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => {
                ready = true;
                break;
            }
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
    if !ready {
        let _ = gateway.kill();
        let _ = gateway.wait();
        panic!(
            "gateway never became ready; stderr:\n{}",
            stderr_capture
                .lock()
                .map(|buf| buf.clone())
                .unwrap_or_default()
        );
    }

    // Public legacy client: the alias path, the perimeter API key, no
    // subprotocol offer — byte-identical to how it addresses the single-node
    // surface.
    let mut request = format!("ws://127.0.0.1:{port}/v1/speech-to-text/realtime/ws")
        .into_client_request()
        .expect("websocket request");
    request.headers_mut().insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {GATEWAY_API_KEY}")).unwrap(),
    );
    let (mut ws, response) = tokio_tungstenite::connect_async(request)
        .await
        .expect("gateway upgrades the translated session");
    assert!(
        response.headers().get("sec-websocket-protocol").is_none(),
        "the public surface selects no subprotocol"
    );

    ws.send(Message::Text(
        serde_json::json!({ "type": "session_start" })
            .to_string()
            .into(),
    ))
    .await
    .expect("session_start sends");

    let ready_message = next_public_text(&mut ws)
        .await
        .expect("session_ready arrives");
    let ready: serde_json::Value = serde_json::from_str(&ready_message).expect("JSON ready");
    assert_eq!(ready["type"], "session_ready");
    assert_eq!(ready["protocol"], "transcription_realtime_v2");
    let started = next_public_text(&mut ws).await.expect("session_started");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&started).expect("JSON started")["type"],
        "session_started"
    );

    // Two audio frames; the worker answers one incremental delta per frame.
    let payload = [0i16.to_le_bytes(); 8].concat();
    for sequence in 1..=2u32 {
        ws.send(Message::Binary(
            itrw_public_frame(sequence, 16_000, &payload).into(),
        ))
        .await
        .expect("audio frame sends");
    }

    ws.send(Message::Text(
        serde_json::json!({ "type": "ping", "timestamp_ms": 7 })
            .to_string()
            .into(),
    ))
    .await
    .expect("ping sends");

    ws.send(Message::Text(
        serde_json::json!({ "type": "session_stop" })
            .to_string()
            .into(),
    ))
    .await
    .expect("session_stop sends");

    // The pump answers pings locally, emits the accumulating partials, then
    // the final transcript and session_done when the worker completes.
    let mut partials: Vec<String> = Vec::new();
    let mut pong_timestamp: Option<i64> = None;
    let mut final_text: Option<String> = None;
    loop {
        let Some(text) = next_public_text(&mut ws).await else {
            panic!("socket closed before session_done");
        };
        let value: serde_json::Value = serde_json::from_str(&text).expect("JSON event");
        match value["type"].as_str() {
            Some("transcript_partial") => {
                if value["is_final"] == serde_json::json!(true) {
                    final_text = Some(value["text"].as_str().expect("final text").to_string());
                } else {
                    partials.push(value["text"].as_str().expect("partial text").to_string());
                }
            }
            Some("pong") => {
                pong_timestamp = value["timestamp_ms"].as_i64();
            }
            Some("session_done") => break,
            Some("session_ready" | "session_started") => {}
            other => panic!("unexpected public event: {other:?}: {value}"),
        }
    }
    assert_eq!(
        partials,
        vec!["partial ", "partial partial "],
        "deltas accumulate into the replaceable partial hypothesis"
    );
    assert_eq!(
        final_text.as_deref(),
        Some("gateway translated transcript"),
        "the terminal delta carries the full final text"
    );
    assert_eq!(pong_timestamp, Some(7), "pings are answered locally");

    let close = loop {
        let message = tokio::time::timeout(Duration::from_secs(10), ws.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Message::Close(frame) = message {
            break frame.expect("close frame");
        }
    };
    assert_eq!(u16::from(close.code), 1000);

    let _ = gateway.kill();
    let _ = gateway.wait();
}

/// Process-level DS3.4 leg (public typed v3 client): the typed
/// `transcription_realtime` envelope mode at `/v1/realtime/ws` without the
/// worker subprotocol. The gateway emits the legacy announcement, the typed
/// session ladder, per-frame ingress events with audio-gap detection, and
/// the final/closing/closed ladder with strict envelope continuity.
#[tokio::test]
async fn gateway_realtime_translate_serves_public_v3_typed_clients() {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, http::HeaderValue, Message};

    let model = ModelVariant::Qwen354BGguf;
    let model_alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    let chat_worker = MockWorker::spawn(MockWorkerConfig {
        worker_id: izwi_serving_protocol::WorkerId::try_from("mock-chat-worker").expect("identity"),
        node_id: izwi_serving_protocol::NodeId::try_from("mock-chat-node").expect("identity"),
        public_model: model_alias.clone(),
        ..MockWorkerConfig::default()
    })
    .await
    .expect("mock chat worker binds");
    let worker = MockWorker::spawn(MockWorkerConfig {
        worker_id: izwi_serving_protocol::WorkerId::try_from("mock-asr-worker").expect("identity"),
        node_id: izwi_serving_protocol::NodeId::try_from("mock-asr-node").expect("identity"),
        deployment_id: izwi_serving_protocol::DeploymentId::try_from("mock-asr-v1")
            .expect("static identity"),
        public_model: model_alias,
        realtime: Some(izwi_serving_client::mock::MockRealtimeKnobs {
            delta_per_frame: "partial ".into(),
            final_text: "gateway translated transcript".into(),
            push_cadence: Duration::from_millis(1),
            disconnect_after_frames: None,
        }),
        ..MockWorkerConfig::default()
    })
    .await
    .expect("mock realtime worker binds");

    let port = {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe listener binds");
        probe.local_addr().expect("probe addr").port()
    };

    let mut gateway = Command::new(env!("CARGO_BIN_EXE_izwi-server"))
        .args([
            "--role",
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--public-model",
            model.dir_name(),
            "--gateway-worker-approval",
            &format!(
                "{}|chat|{}|mock-chat-v1|1",
                chat_worker.endpoint(),
                model.dir_name()
            ),
            "--gateway-worker-approval",
            &format!(
                "{}|speech_to_text|{}|mock-asr-v1|1",
                worker.endpoint(),
                model.dir_name()
            ),
            "--gateway-realtime",
            "on",
        ])
        .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
        .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
        .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
        .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
        .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gateway binary spawns");
    let stderr_capture = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let stderr_store = std::sync::Arc::clone(&stderr_capture);
    if let Some(stderr) = gateway.stderr.take() {
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(mut collected) = stderr_store.lock() {
                    collected.push_str(&line);
                    collected.push('\n');
                }
            }
        });
    }

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut ready = false;
    while Instant::now() < deadline {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => {
                ready = true;
                break;
            }
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
    if !ready {
        let _ = gateway.kill();
        let _ = gateway.wait();
        panic!(
            "gateway never became ready; stderr:\n{}",
            stderr_capture
                .lock()
                .map(|buf| buf.clone())
                .unwrap_or_default()
        );
    }

    let mut request = format!("ws://127.0.0.1:{port}/v1/realtime/ws")
        .into_client_request()
        .expect("websocket request");
    request.headers_mut().insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {GATEWAY_API_KEY}")).unwrap(),
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(request)
        .await
        .expect("gateway upgrades the translated session");

    // The legacy announcement still precedes negotiation, then the typed
    // session_start upgrades the session to v3 envelopes.
    ws.send(Message::Text(
        serde_json::json!({
            "type": "session_start",
            "protocol": "transcription_realtime",
            "version": 3
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("session_start sends");

    let ready_message = next_public_text(&mut ws)
        .await
        .expect("legacy announcement arrives");
    let ready: serde_json::Value = serde_json::from_str(&ready_message).expect("JSON ready");
    assert_eq!(ready["type"], "session_ready");

    // Three audio frames (1, 2, 4): each accepted, and the skip surfaces as
    // an AudioGap with the missing span. Then the stop drains the final
    // transcript and the closing ladder.
    let payload = [0i16.to_le_bytes(); 8].concat();
    for sequence in [1u32, 2, 4] {
        ws.send(Message::Binary(
            itrw_public_frame(sequence, 16_000, &payload).into(),
        ))
        .await
        .expect("audio frame sends");
    }

    ws.send(Message::Text(
        serde_json::json!({ "type": "session_stop" })
            .to_string()
            .into(),
    ))
    .await
    .expect("session_stop sends");

    let mut envelopes: Vec<serde_json::Value> = Vec::new();
    loop {
        let Some(text) = next_public_text(&mut ws).await else {
            panic!("socket closed before the typed close ladder");
        };
        let envelope: serde_json::Value = serde_json::from_str(&text).expect("typed envelope");
        assert_eq!(envelope["protocol"], "transcription_realtime");
        assert_eq!(envelope["version"], 3);
        envelopes.push(envelope);
        if envelopes.last().expect("envelope")["type"] == "closed" {
            break;
        }
    }

    // Strict continuity: sequences count 0,1,2.. within epoch 0 and event
    // ids strictly increase from 1.
    let mut previous_sequence: Option<u64> = None;
    let mut previous_event_id: Option<u64> = None;
    for envelope in &envelopes {
        let sequence = envelope["sequence"].as_u64().expect("sequence");
        let event_id = envelope["event_id"].as_u64().expect("event id");
        assert_eq!(envelope["connection_epoch"], 0);
        match previous_sequence {
            None => assert_eq!(sequence, 0, "epoch starts at sequence zero"),
            Some(previous) => assert_eq!(sequence, previous + 1, "sequences are contiguous"),
        }
        if let Some(previous) = previous_event_id {
            assert!(event_id > previous, "event ids increase");
        }
        previous_sequence = Some(sequence);
        previous_event_id = Some(event_id);
    }

    let event_type =
        |envelope: &serde_json::Value| envelope["type"].as_str().expect("event type").to_string();
    let types: Vec<String> = envelopes.iter().map(event_type).collect();
    assert_eq!(
        types.first().map(String::as_str),
        Some("session_ready"),
        "the typed session opens with SessionReady"
    );
    assert_eq!(types.get(1).map(String::as_str), Some("session_started"));
    assert_eq!(
        envelopes[0]["data"]["accepted_version"], 3,
        "admission announces the accepted version"
    );

    let mut accepted_sequences: Vec<u64> = Vec::new();
    let mut gaps: Vec<(u64, u64, u64)> = Vec::new();
    let mut partials: Vec<String> = Vec::new();
    let mut final_text: Option<String> = None;
    let mut saw_closing = false;
    let mut saw_closed = false;
    for envelope in &envelopes {
        match event_type(envelope).as_str() {
            "audio_accepted" => accepted_sequences.push(
                envelope["data"]["frame_sequence"]
                    .as_u64()
                    .expect("frame sequence"),
            ),
            "audio_gap" => gaps.push((
                envelope["data"]["expected_frame_sequence"]
                    .as_u64()
                    .expect("expected"),
                envelope["data"]["received_frame_sequence"]
                    .as_u64()
                    .expect("received"),
                envelope["data"]["missing_frames"]
                    .as_u64()
                    .expect("missing"),
            )),
            "transcript_partial" => partials.push(
                envelope["data"]["text"]
                    .as_str()
                    .expect("partial text")
                    .to_string(),
            ),
            "transcript_final" => {
                final_text = Some(
                    envelope["data"]["text"]
                        .as_str()
                        .expect("final text")
                        .to_string(),
                )
            }
            "closing" => saw_closing = true,
            "closed" => saw_closed = true,
            _ => {}
        }
    }

    assert_eq!(
        accepted_sequences,
        vec![1, 2, 4],
        "every accepted frame echoes the client sequence"
    );
    assert_eq!(
        gaps,
        vec![(3, 4, 1)],
        "the skipped frame 3 is reported as one missing frame"
    );
    assert_eq!(
        partials,
        vec!["partial ", "partial partial ", "partial partial partial "],
        "deltas accumulate into replaceable partial hypotheses"
    );
    assert_eq!(
        final_text.as_deref(),
        Some("gateway translated transcript"),
        "the terminal delta carries the full final text"
    );
    assert!(saw_closing, "Closing precedes Closed");
    assert!(saw_closed, "Closed terminates the session");

    let close = loop {
        let message = tokio::time::timeout(Duration::from_secs(10), ws.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Message::Close(frame) = message {
            break frame.expect("close frame");
        }
    };
    assert_eq!(u16::from(close.code), 1000);

    let _ = gateway.kill();
    let _ = gateway.wait();
}
