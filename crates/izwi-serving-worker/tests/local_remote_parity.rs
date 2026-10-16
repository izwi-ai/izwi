//! Phase 1.3 local/remote parity: one chat request through the local
//! server's API (`--role local`, in process, over a `RuntimeService` with the
//! tiny LFM2 fixture loaded) and through the gateway to a real worker
//! process on the same backend must give the same answer: greedy and seeded
//! sampling, stop strings, logprobs, JSON mode, the output budget, and the
//! D8 overflow 400.
//!
//! The CPU leg always runs. The Metal leg runs with `--features metal` on a
//! host with a GPU; `IZWI_REQUIRE_METAL_PARITY=1` turns its skip into a
//! failure. CUDA is not run here.

mod common;

use axum::{body::Body, http::Request, Router};
#[cfg(feature = "metal")]
use common::metal_leg_device;
use common::{id, single_worker_dispatcher, write_tiny_lfm_fixture};
use izwi_core::{
    backends::RuntimeDeviceAssignment, EngineConfig, ModelVariant, RuntimeService,
    ServeRuntimeConfig,
};
use izwi_hooks::EnterpriseHooks;
use izwi_server::worker_registry::BackendPolicy;
use izwi_server::{
    create_gateway_router, create_local_router, GatewayPerimeterConfig, GatewayState,
    RemoteChatDispatchConfig,
};
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::{ServiceBearerToken, ServiceCredentials};
use izwi_test_process::ProcessGuard;
use serde_json::{json, Value};
use std::{net::TcpListener, path::Path, process::Stdio, time::Duration};
use tower::ServiceExt;

const VARIANT: ModelVariant = ModelVariant::Lfm2512BInstructGguf;
/// The fixture's native context. The gateway's output ceiling matches it,
/// so an unset budget means the same on both paths.
const CONTEXT_TOKENS: u32 = 32;
const GATEWAY_API_KEY: &str = "parity-gateway-api-key";
const WORKER_CREDENTIAL: &str = "parity-credential";
const WORKER_BEARER: &str = "parity-secret";
const DEPLOYMENT: &str = "lfm25-cpu-v1";

/// The legs run one at a time: the local server reads its store paths from
/// the process environment, and two decoding backends at once would only
/// slow each other.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The local server and the gateway over one backend.
struct Paths {
    local: Router,
    gateway: Router,
    _worker: ProcessGuard,
    _models: tempfile::TempDir,
    _data: tempfile::TempDir,
}

async fn paths(
    local_assignment: RuntimeDeviceAssignment,
    worker_env: &[(&str, &str)],
    startup_deadline: Duration,
) -> Paths {
    let models = tempfile::tempdir().unwrap();
    write_tiny_lfm_fixture(models.path());
    let data = tempfile::tempdir().unwrap();
    let local = local_server(models.path(), data.path(), local_assignment).await;
    let (gateway, worker) = gateway_to_worker(models.path(), worker_env, startup_deadline).await;
    Paths {
        local,
        gateway,
        _worker: worker,
        _models: models,
        _data: data,
    }
}

/// The local role's API over an in-process runtime configured like the
/// worker's (one CPU thread, one loaded model).
async fn local_server(models: &Path, data: &Path, assignment: RuntimeDeviceAssignment) -> Router {
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
    // The local stores live in this test's temporary directory; `SERIAL`
    // keeps another leg from changing these while the server reads them.
    std::env::set_var("IZWI_DB_PATH", data.join("izwi.sqlite3"));
    std::env::set_var("IZWI_MEDIA_DIR", data.join("media"));
    let serve_config = ServeRuntimeConfig {
        ui_enabled: false,
        models_dir: models.to_path_buf(),
        ..ServeRuntimeConfig::default()
    };
    create_local_router(runtime, &serve_config).expect("local router")
}

/// The gateway pinned to a real worker process serving the same fixture.
async fn gateway_to_worker(
    models: &Path,
    worker_env: &[(&str, &str)],
    startup_deadline: Duration,
) -> (Router, ProcessGuard) {
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_izwi-serving-worker"));
    command
        .env("IZWI_WORKER_BIND", address.to_string())
        .env("IZWI_MODELS_DIR", models)
        .env("IZWI_WORKER_CREDENTIAL_ID", WORKER_CREDENTIAL)
        .env("IZWI_WORKER_BEARER_TOKEN", WORKER_BEARER)
        .env("IZWI_WORKER_ARTIFACT_REVISION", "tiny-lfm-fixture-v1")
        .env("IZWI_WORKER_DEPLOYMENT_ID", DEPLOYMENT)
        .env("IZWI_WORKER_CPU_THREADS", "1")
        .env("IZWI_WORKER_MAX_ACTIVE", "1")
        .env("RUST_LOG", "warn")
        .envs(worker_env.iter().copied())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    let child = ProcessGuard::spawn(&mut command).unwrap();
    let client = WorkerClient::new(
        &format!("http://{address}/"),
        ServiceCredentials {
            credential_id: id(WORKER_CREDENTIAL),
            bearer_token: ServiceBearerToken::new(WORKER_BEARER).unwrap(),
        },
        WorkerClientConfig {
            connect_timeout: Duration::from_millis(100),
            request_timeout: Duration::from_millis(500),
            progress_timeout: Duration::from_secs(10),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();
    tokio::time::timeout(startup_deadline, async {
        while client.descriptor().await.is_err() {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("worker loads and warms the tiny model before its startup deadline");
    let dispatcher = single_worker_dispatcher(
        client,
        RemoteChatDispatchConfig {
            public_model_variant: VARIANT,
            deployment_id: id(DEPLOYMENT),
            policy_revision: id("parity-policy-v1"),
            backend_policy: BackendPolicy::ANY,
            max_queue_wait: Duration::from_secs(5),
            max_output_tokens: CONTEXT_TOKENS,
            max_output_bytes: 64 * 1024,
            slow_consumer_timeout: Duration::from_secs(5),
            prefix_affinity: None,
        },
    )
    .await;
    let gateway = GatewayState::new(
        dispatcher,
        EnterpriseHooks::noop(),
        GatewayPerimeterConfig::new_for_test(GATEWAY_API_KEY, 1024 * 1024).unwrap(),
        30,
        2,
    );
    gateway.mark_ready();
    let router = create_gateway_router(
        gateway,
        &ServeRuntimeConfig {
            ui_enabled: false,
            ..ServeRuntimeConfig::default()
        },
    );
    (router, child)
}

/// One chat completion's status and JSON body.
async fn complete(app: &Router, body: &Value, api_key: Option<&str>) -> (u16, Value) {
    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(api_key) = api_key {
        request = request.header("authorization", format!("Bearer {api_key}"));
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let body = if body["stream"] == json!(true) && status == 200 {
        stream_summary(&text)
    } else {
        serde_json::from_str(&text).unwrap_or_else(|_| panic!("JSON body, got {text}"))
    };
    (status, body)
}

/// A GET's status and JSON body.
async fn get_json(app: &Router, path: &str, api_key: Option<&str>) -> (u16, Value) {
    let mut request = Request::builder().uri(path);
    if let Some(api_key) = api_key {
        request = request.header("authorization", format!("Bearer {api_key}"));
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// What a client reads from an SSE chat stream: the concatenated content,
/// every logprob entry, the finish reason, the usage and any error.
fn stream_summary(sse: &str) -> Value {
    let mut content = String::new();
    let mut logprobs = Vec::new();
    let mut finish_reason = Value::Null;
    let mut usage = Value::Null;
    let mut error = Value::Null;
    for data in sse.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if data == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(data).unwrap();
        if let Some(chunk_error) = chunk.get("error") {
            error = chunk_error.clone();
        }
        if let Some(choice) = chunk["choices"].get(0) {
            if let Some(text) = choice["delta"]["content"].as_str() {
                content.push_str(text);
            }
            if let Some(entries) = choice["logprobs"]["content"].as_array() {
                logprobs.extend(entries.iter().cloned());
            }
            if !choice["finish_reason"].is_null() {
                finish_reason = choice["finish_reason"].clone();
            }
        }
        if !chunk["usage"].is_null() {
            usage = chunk["usage"].clone();
        }
    }
    json!({
        "content": content,
        "logprobs": logprobs,
        "finish_reason": finish_reason,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "error": error,
    })
}

/// The parts of a completion that must match: the answer, its logprobs,
/// the finish reason and the token counts (ids and timings differ).
fn answer(body: &Value) -> Value {
    if body.get("content").is_some() {
        assert!(body["error"].is_null(), "stream error: {body}");
        return body.clone();
    }
    let choice = &body["choices"][0];
    json!({
        "content": choice["message"]["content"],
        "logprobs": choice["logprobs"]["content"].as_array().cloned().unwrap_or_default(),
        "finish_reason": choice["finish_reason"],
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "completion_tokens": body["usage"]["completion_tokens"],
    })
}

/// Sends `request` to both paths, non-streaming and streaming, and asserts
/// they answer the same. Returns the local non-streaming answer.
async fn assert_same_answer(paths: &Paths, case: &str, request: Value) -> Value {
    let mut first = None;
    for stream in [false, true] {
        let mut request = request.clone();
        request["model"] = json!(VARIANT.dir_name());
        request["stream"] = json!(stream);
        if stream {
            request["stream_options"] = json!({"include_usage": true});
        }
        let (local_status, local) = complete(&paths.local, &request, None).await;
        let (remote_status, remote) =
            complete(&paths.gateway, &request, Some(GATEWAY_API_KEY)).await;
        assert_eq!(
            (local_status, remote_status),
            (200, 200),
            "{case} (stream {stream}): local {local}, gateway {remote}"
        );
        assert_eq!(
            answer(&local),
            answer(&remote),
            "{case} (stream {stream}): local and gateway answers differ"
        );
        if request["logprobs"] == json!(true) {
            // One entry per sampled token, stop-string tokens included.
            assert_eq!(
                answer(&local)["logprobs"].as_array().unwrap().len() as u64,
                answer(&local)["completion_tokens"].as_u64().unwrap(),
                "{case} (stream {stream})"
            );
        }
        first.get_or_insert(answer(&local));
    }
    first.unwrap()
}

/// Sends `request` to both paths and asserts the same error status and body.
async fn assert_same_error(paths: &Paths, case: &str, request: Value) -> Value {
    let mut request = request;
    request["model"] = json!(VARIANT.dir_name());
    let (local_status, local) = complete(&paths.local, &request, None).await;
    let (remote_status, remote) = complete(&paths.gateway, &request, Some(GATEWAY_API_KEY)).await;
    assert_eq!(
        (local_status, &local),
        (remote_status, &remote),
        "{case}: local and gateway errors differ"
    );
    local
}

fn user(content: &str) -> Value {
    json!({"role": "user", "content": content})
}

/// Ten complete turns and a new message: more than the 32-token context.
fn long_conversation() -> Value {
    let mut messages = vec![json!({"role": "system", "content": "be brief"})];
    for _ in 0..10 {
        messages.push(user("a"));
        messages.push(json!({"role": "assistant", "content": "b"}));
    }
    messages.push(user("hello"));
    Value::Array(messages)
}

/// A stop string of a few characters from inside `text`, after its first
/// character, that first occurs there.
fn stop_inside(text: &str) -> Option<String> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    (1..chars.len().saturating_sub(3)).find_map(|start| {
        let from = chars[start].0;
        let stop = &text[from..chars[start + 3].0];
        (!stop.trim().is_empty() && text.find(stop) == Some(from)).then(|| stop.to_string())
    })
}

async fn assert_paths_agree(paths: &Paths) {
    let hello = json!([user("hello")]);

    // Each path lists the one model it serves, and 404s any other.
    for (app, api_key) in [
        (&paths.local, None),
        (&paths.gateway, Some(GATEWAY_API_KEY)),
    ] {
        let (status, list) = get_json(app, "/v1/models", api_key).await;
        assert_eq!(status, 200, "{list}");
        assert_eq!(list["data"][0]["id"], VARIANT.dir_name(), "{list}");
        assert_eq!(list["data"][0]["owned_by"], "izwi");
        assert_eq!(list["data"].as_array().unwrap().len(), 1, "{list}");
        let (status, missing) = get_json(app, "/v1/models/gpt-imaginary", api_key).await;
        assert_eq!(status, 404);
        assert_eq!(missing["error"]["code"], "model_not_found");
    }

    let greedy = assert_same_answer(
        paths,
        "greedy",
        json!({"messages": hello, "temperature": 0.0, "max_tokens": 8}),
    )
    .await;
    assert!(
        !greedy["content"].as_str().unwrap().is_empty(),
        "the fixture answers: {greedy}"
    );

    // Hot seeded sampling with logprobs: the same seed samples the same
    // tokens with the same logprobs on both paths.
    let seeded = json!({
        "messages": hello,
        "temperature": 2.0,
        "seed": 7,
        "max_tokens": 12,
        "logprobs": true,
        "top_logprobs": 2,
    });
    let sampled = assert_same_answer(paths, "seeded sampling", seeded.clone()).await;
    assert!(sampled["logprobs"]
        .as_array()
        .unwrap()
        .iter()
        .all(|entry| entry["top_logprobs"].as_array().unwrap().len() == 2));
    let mut other_seed = seeded.clone();
    other_seed["seed"] = json!(8);
    let resampled = assert_same_answer(paths, "seeded sampling, seed 8", other_seed).await;
    assert_ne!(
        sampled["content"], resampled["content"],
        "the seed must decide the sample"
    );

    // A stop string from inside the sampled text ends both runs before it.
    let text = sampled["content"].as_str().unwrap();
    let stop = stop_inside(text).unwrap_or_else(|| panic!("no stop window in {text:?}"));
    let mut stopped = seeded.clone();
    stopped["stop"] = json!(stop);
    let truncated = assert_same_answer(paths, "stop", stopped).await;
    assert_eq!(truncated["finish_reason"], "stop");
    assert_eq!(
        truncated["content"].as_str().unwrap(),
        &text[..text.find(&stop).unwrap()]
    );

    // An unset budget runs to the end of the context on both paths.
    assert_same_answer(
        paths,
        "unset budget",
        json!({"messages": hello, "temperature": 0.0}),
    )
    .await;

    // JSON mode reaches the engine on both paths. The fixture's vocabulary
    // has no JSON token, so the constraint ends generation at once and LFM2
    // rejects the empty answer; a dropped constraint would answer 200. It is
    // a server fault: the loopback local server shows the model's own error,
    // and the gateway the same type and code with a redacted message.
    let mut json_mode = json!({
        "model": VARIANT.dir_name(),
        "messages": hello,
        "temperature": 0.0,
        "max_tokens": 8,
        "response_format": {"type": "json_object"},
    });
    for stream in [false, true] {
        json_mode["stream"] = json!(stream);
        let (local_status, local) = complete(&paths.local, &json_mode, None).await;
        let (remote_status, remote) =
            complete(&paths.gateway, &json_mode, Some(GATEWAY_API_KEY)).await;
        if stream {
            // Both streams start, then fail.
            assert_eq!((local_status, remote_status), (200, 200));
            assert!(
                !local["error"].is_null(),
                "json_object stream: local {local}"
            );
            for field in ["type", "code", "param"] {
                assert_eq!(
                    local["error"][field], remote["error"][field],
                    "json_object stream: {field}"
                );
            }
            let message = |error: &Value| {
                error["error"]["message"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string()
            };
            assert!(
                message(&local).contains("LFM2 generation did not produce a valid answer"),
                "json_object stream: local {local}"
            );
            assert!(
                message(&remote).starts_with("The server had an error")
                    && !message(&remote).contains("LFM2"),
                "json_object stream: gateway {remote}"
            );
        } else {
            assert_eq!(
                (local_status, remote_status),
                (500, 500),
                "json_object: local {local}, gateway {remote}"
            );
        }
    }

    // D8: overflow is the same OpenAI 400 on both paths, streamed or not,
    // unless the request opts into truncation. A stream is admitted before
    // its response starts, so its overflow is a 400 too.
    for stream in [false, true] {
        let overflow = assert_same_error(
            paths,
            &format!("prompt overflow (stream {stream})"),
            json!({"messages": long_conversation(), "temperature": 0.0, "stream": stream}),
        )
        .await;
        assert_eq!(overflow["error"]["code"], "context_length_exceeded");
        assert_eq!(overflow["error"]["param"], "messages");
        let budget = assert_same_error(
            paths,
            &format!("budget overflow (stream {stream})"),
            json!({
                "messages": hello,
                "temperature": 0.0,
                "max_tokens": CONTEXT_TOKENS + 8,
                "stream": stream,
            }),
        )
        .await;
        assert_eq!(budget["error"]["code"], "context_length_exceeded");
    }
    // Tools reach the prompt on both paths, rendered alike.
    let with_tools = assert_same_answer(
        paths,
        "tools",
        json!({
            "messages": hello,
            "temperature": 0.0,
            "max_tokens": 8,
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }),
    )
    .await;
    assert!(
        with_tools["prompt_tokens"].as_u64() > greedy["prompt_tokens"].as_u64(),
        "the tool list is in the prompt: {with_tools} vs {greedy}"
    );
    assert_same_answer(
        paths,
        "auto truncation",
        json!({
            "messages": long_conversation(),
            "temperature": 0.0,
            "max_tokens": 4,
            "truncation": "auto",
        }),
    )
    .await;
}

#[tokio::test]
async fn local_and_gateway_paths_agree_on_cpu() {
    let _serial = SERIAL.lock().await;
    let paths = paths(RuntimeDeviceAssignment::Cpu, &[], Duration::from_secs(30)).await;
    assert_paths_agree(&paths).await;
}

#[cfg(feature = "metal")]
#[tokio::test]
async fn local_and_gateway_paths_agree_on_metal() {
    let Some(device_id) = metal_leg_device("local/remote parity metal leg") else {
        return;
    };
    let _serial = SERIAL.lock().await;
    let paths = paths(
        RuntimeDeviceAssignment::Metal {
            process_local_device_index: 0,
            expected_device_id: device_id.clone(),
        },
        &[
            ("IZWI_BACKEND", "metal"),
            ("IZWI_WORKER_EXPECTED_DEVICE_ID", device_id.as_str()),
            ("IZWI_METAL_DEVICE_ORDINAL", "0"),
        ],
        // Cold Metal starts compile the shader cache.
        Duration::from_secs(120),
    )
    .await;
    assert_paths_agree(&paths).await;
}
