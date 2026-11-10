//! `/v1/batches` through the router and the executor: every route and error
//! shape, input validation, a run with an error file, cancel, resume after
//! a lost executor, expiry and tenant isolation on a runner the tests
//! control; then real lines through the audio routes' own code, speech on
//! the Fish Audio Drama 3 synthetic checkpoint.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::{Method, StatusCode};
use base64::Engine;
use izwi_core::audio::{AudioEncoder, AudioFormat};
use serde_json::{json, Value};

use super::executor::{
    BatchExecutor, ExecutorConfig, LineJob, LineOutcome, LineReader, LineRunner,
};
use super::webhook::tests::{assert_signed, local_policy, Listener};
use crate::api::openai::files::tests::{bytes, json as read_json, TestServer};
use crate::artifact_store::ArtifactTenant;

const TRANSCRIPTIONS: &str = "/v1/audio/transcriptions";
const SPEECH: &str = "/v1/audio/speech";

fn config() -> ExecutorConfig {
    ExecutorConfig {
        lease: Duration::from_millis(300),
        poll: Duration::from_millis(20),
        line_concurrency: 2,
        max_requests: 8,
        max_line_bytes: 64 * 1024,
        capacity_retry: Duration::from_millis(10),
        webhook: None,
    }
}

/// Start an executor on the server's stores that runs lines with `runner`.
fn start(
    server: &TestServer,
    runner: LineRunner,
    config: ExecutorConfig,
) -> tokio::task::JoinHandle<()> {
    let executor = BatchExecutor::new(
        server.state.batch_api_store.clone(),
        server.state.file_store.clone(),
        runner,
        config,
        server.state.batch_api_wake.clone(),
        server.state.stream_shutdown.subscribe(),
    );
    tokio::spawn(executor.run())
}

/// A runner answering each line with `answer`.
fn runner<F, Fut>(answer: F) -> LineRunner
where
    F: Fn(Arc<LineJob>) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = LineOutcome> + Send + 'static,
{
    Arc::new(move |job| Box::pin(answer(job)))
}

fn ok(body: Value) -> LineOutcome {
    LineOutcome {
        status_code: 200,
        body,
    }
}

/// A line of `endpoint` whose body is `body`.
fn line(custom_id: &str, endpoint: &str, body: Value) -> String {
    json!({"custom_id": custom_id, "method": "POST", "url": endpoint, "body": body}).to_string()
}

/// An input file of one transcription line per id.
fn input(ids: &[&str]) -> String {
    ids.iter()
        .map(|id| {
            line(
                id,
                TRANSCRIPTIONS,
                json!({"model": "m", "audio_base64": "AAAA"}),
            ) + "\n"
        })
        .collect()
}

/// Upload `input` as `tenant` and create a batch of `endpoint` over it.
async fn create(server: &TestServer, tenant: &str, endpoint: &str, input: &str) -> Value {
    let file = server
        .upload_ok(tenant, "batch", "input.jsonl", input.as_bytes())
        .await;
    let response = server
        .post_json(
            "/v1/batches",
            tenant,
            json!({"input_file_id": file, "endpoint": endpoint, "completion_window": "24h"}),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    read_json(response).await
}

async fn get(server: &TestServer, tenant: &str, id: &str) -> Value {
    let response = server
        .call(Method::GET, &format!("/v1/batches/{id}"), tenant)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    read_json(response).await
}

/// The batch once `done` holds for it.
async fn wait_for(
    server: &TestServer,
    tenant: &str,
    id: &str,
    done: impl Fn(&Value) -> bool,
) -> Value {
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    loop {
        let batch = get(server, tenant, id).await;
        if done(&batch) {
            return batch;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the batch never got there: {batch}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_status(server: &TestServer, tenant: &str, id: &str, status: &str) -> Value {
    wait_for(server, tenant, id, |batch| batch["status"] == status).await
}

/// The lines of `tenant`'s JSONL file `id`.
async fn file_lines(server: &TestServer, tenant: &str, id: &Value) -> Vec<Value> {
    let content = server
        .content(tenant, id.as_str().expect("a file id"))
        .await;
    String::from_utf8(content)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("a JSON line"))
        .collect()
}

fn custom_ids(lines: &[Value]) -> Vec<&str> {
    lines
        .iter()
        .map(|line| line["custom_id"].as_str().unwrap())
        .collect()
}

async fn result_rows(server: &TestServer) -> i64 {
    use sea_orm::ConnectionTrait;
    let db = server.state.batch_runtime_store.connection().await.unwrap();
    db.query_one_raw(
        crate::db::raw::statement(db, "SELECT COUNT(*) FROM api_batch_requests", vec![]).unwrap(),
    )
    .await
    .unwrap()
    .unwrap()
    .try_get_by_index::<i64>(0)
    .unwrap()
}

#[tokio::test]
async fn batches_are_created_read_listed_and_cancelled() {
    let server = TestServer::new("batches-routes");
    let input_file = server
        .upload_ok("a", "batch", "input.jsonl", input(&["one"]).as_bytes())
        .await;
    let audio_file = server
        .upload_ok("a", "user_data", "clip.wav", b"RIFF")
        .await;
    let request = |patch: Value| {
        let mut body = json!({
            "input_file_id": input_file,
            "endpoint": TRANSCRIPTIONS,
            "completion_window": "24h",
        });
        body.as_object_mut()
            .unwrap()
            .extend(patch.as_object().unwrap().clone());
        body
    };

    // What OpenAI refuses when a batch is created.
    for (patch, param, says) in [
        (
            json!({"endpoint": "/v1/chat/completions"}),
            "endpoint",
            "/v1/audio/transcriptions, /v1/audio/translations, /v1/audio/speech",
        ),
        (
            json!({"completion_window": "1h"}),
            "completion_window",
            "24h",
        ),
        (
            json!({"input_file_id": "file-missing"}),
            "input_file_id",
            "file-missing",
        ),
        (
            json!({"input_file_id": audio_file}),
            "input_file_id",
            "user_data",
        ),
        (json!({"metadata": {"k": 1}}), "metadata", "string pairs"),
        (json!({"metadata": ["k"]}), "metadata", "string pairs"),
    ] {
        let response = server.post_json("/v1/batches", "a", request(patch)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{param}");
        let error = read_json(response).await;
        assert_eq!(error["error"]["param"], param);
        assert_eq!(error["error"]["type"], "invalid_request_error");
        let message = error["error"]["message"].as_str().unwrap();
        assert!(message.contains(says), "{message}");
    }
    let missing_field = server
        .post_json("/v1/batches", "a", json!({"endpoint": TRANSCRIPTIONS}))
        .await;
    assert_eq!(missing_field.status(), StatusCode::BAD_REQUEST);

    // A new batch, as OpenAI shapes it.
    let response = server
        .post_json(
            "/v1/batches",
            "a",
            request(json!({"metadata": {"job": "nightly"}})),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let batch = read_json(response).await;
    let id = batch["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("batch_"));
    let created_at = batch["created_at"].as_i64().unwrap();
    assert_eq!(
        batch,
        json!({
            "id": id,
            "object": "batch",
            "endpoint": TRANSCRIPTIONS,
            "errors": null,
            "input_file_id": input_file,
            "completion_window": "24h",
            "status": "validating",
            "output_file_id": null,
            "error_file_id": null,
            "created_at": created_at,
            "in_progress_at": null,
            "expires_at": created_at + 24 * 60 * 60,
            "finalizing_at": null,
            "completed_at": null,
            "failed_at": null,
            "expired_at": null,
            "cancelling_at": null,
            "cancelled_at": null,
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
            "metadata": {"job": "nightly"},
        })
    );
    assert_eq!(get(&server, "a", &id).await, batch);

    // Lists page newest first.
    let second = create(&server, "a", SPEECH, &input(&["two"])).await;
    let third = create(&server, "a", SPEECH, &input(&["three"])).await;
    let page = read_json(server.call(Method::GET, "/v1/batches?limit=2", "a").await).await;
    assert_eq!(page["object"], "list");
    assert_eq!(page["has_more"], true);
    assert_eq!(page["data"].as_array().unwrap().len(), 2);
    assert_eq!(page["first_id"], third["id"]);
    assert_eq!(page["last_id"], second["id"]);
    let rest = read_json(
        server
            .call(
                Method::GET,
                &format!("/v1/batches?after={}", second["id"].as_str().unwrap()),
                "a",
            )
            .await,
    )
    .await;
    assert_eq!(rest["has_more"], false);
    assert_eq!(rest["data"].as_array().unwrap().len(), 1);
    assert_eq!(rest["data"][0]["id"], id.as_str());
    let unknown_cursor = server
        .call(Method::GET, "/v1/batches?after=batch_unknown", "a")
        .await;
    assert_eq!(unknown_cursor.status(), StatusCode::BAD_REQUEST);

    // Another principal sees none of it, and cannot use the input file.
    for (method, path) in [
        (Method::GET, format!("/v1/batches/{id}")),
        (Method::POST, format!("/v1/batches/{id}/cancel")),
    ] {
        let response = server.call(method, &path, "b").await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        let error = read_json(response).await;
        assert!(error["error"]["message"].as_str().unwrap().contains(&id));
    }
    let other = read_json(server.call(Method::GET, "/v1/batches", "b").await).await;
    assert_eq!(other["data"], json!([]));
    let stolen = server
        .post_json("/v1/batches", "b", request(json!({})))
        .await;
    assert_eq!(stolen.status(), StatusCode::BAD_REQUEST);
    assert_eq!(get(&server, "a", &id).await["status"], "validating");

    // Cancel: `cancelling` at once, and again without complaint.
    for _ in 0..2 {
        let response = server
            .call(Method::POST, &format!("/v1/batches/{id}/cancel"), "a")
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let cancelling = read_json(response).await;
        assert_eq!(cancelling["status"], "cancelling");
        assert!(cancelling["cancelling_at"].is_i64());
    }
    // With an executor the batch ends `cancelled`, and a batch that has
    // ended cannot be cancelled.
    let executor = start(&server, runner(|_| async { ok(json!({})) }), config());
    let cancelled = wait_for_status(&server, "a", &id, "cancelled").await;
    assert!(cancelled["cancelled_at"].is_i64());
    assert_eq!(cancelled["output_file_id"], Value::Null);
    let response = server
        .call(Method::POST, &format!("/v1/batches/{id}/cancel"), "a")
        .await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert!(read_json(response).await["error"]["message"]
        .as_str()
        .unwrap()
        .contains("cancelled"));
    executor.abort();
}

#[tokio::test]
async fn an_input_file_that_breaks_the_line_format_fails_the_batch() {
    let server = TestServer::new("batches-validation");
    let body = json!({"model": "m", "audio_base64": "AAAA"});
    let lines = [
        line("good", TRANSCRIPTIONS, body.clone()),
        "{not json".to_string(),
        json!({"method": "POST", "url": TRANSCRIPTIONS, "body": body}).to_string(),
        line("good", TRANSCRIPTIONS, body.clone()),
        json!({"custom_id": "get", "method": "GET", "url": TRANSCRIPTIONS, "body": body})
            .to_string(),
        line("elsewhere", SPEECH, body.clone()),
        json!({"custom_id": "bodiless", "method": "POST", "url": TRANSCRIPTIONS}).to_string(),
        line(
            "streamed",
            TRANSCRIPTIONS,
            json!({"model": "m", "audio_base64": "AAAA", "stream": true}),
        ),
        // Blank lines are not requests, and keep their line numbers.
        String::new(),
        line(
            "long",
            TRANSCRIPTIONS,
            json!({"audio_base64": "A".repeat(70_000)}),
        ),
    ];
    let mut wide = config();
    wide.max_requests = 100;
    let executor = start(&server, runner(|_| async { ok(json!({})) }), wide);
    let batch = create(&server, "a", TRANSCRIPTIONS, &lines.join("\n")).await;
    let failed = wait_for_status(&server, "a", batch["id"].as_str().unwrap(), "failed").await;
    assert!(failed["failed_at"].is_i64());
    assert_eq!(failed["in_progress_at"], Value::Null);
    assert_eq!(failed["output_file_id"], Value::Null);
    assert_eq!(failed["error_file_id"], Value::Null);
    assert_eq!(failed["errors"]["object"], "list");
    let errors: Vec<(String, i64, Value)> = failed["errors"]["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|error| {
            assert!(error["message"]
                .as_str()
                .is_some_and(|text| !text.is_empty()));
            (
                error["code"].as_str().unwrap().to_string(),
                error["line"].as_i64().unwrap(),
                error["param"].clone(),
            )
        })
        .collect();
    assert_eq!(
        errors,
        [
            ("invalid_json_line", 2, Value::Null),
            ("missing_custom_id", 3, json!("custom_id")),
            ("duplicate_custom_id", 4, json!("custom_id")),
            ("invalid_method", 5, json!("method")),
            ("invalid_url", 6, json!("url")),
            ("invalid_body", 7, json!("body")),
            ("streaming_not_supported", 8, json!("body")),
            ("line_too_long", 10, Value::Null),
        ]
        .map(|(code, line, param)| (code.to_string(), line, param))
    );
    executor.abort();

    // More requests than the limit, and none at all.
    let executor = start(&server, runner(|_| async { ok(json!({})) }), config());
    let ids: Vec<String> = (0..9).map(|index| format!("line-{index}")).collect();
    let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
    for (content, code) in [
        (input(&ids), "too_many_requests"),
        ("\n \n".to_string(), "empty_file"),
    ] {
        let batch = create(&server, "a", TRANSCRIPTIONS, &content).await;
        let failed = wait_for_status(&server, "a", batch["id"].as_str().unwrap(), "failed").await;
        assert_eq!(failed["errors"]["data"][0]["code"], code);
        assert_eq!(failed["errors"]["data"][0]["line"], Value::Null);
    }
    executor.abort();
    let _ = executor.await;

    // An input file deleted before its batch was validated fails the batch.
    let unvalidated = create(&server, "a", TRANSCRIPTIONS, &input(&["one"])).await;
    let deleted = server
        .call(
            Method::DELETE,
            &format!(
                "/v1/files/{}",
                unvalidated["input_file_id"].as_str().unwrap()
            ),
            "a",
        )
        .await;
    assert_eq!(deleted.status(), StatusCode::OK);
    let executor = start(&server, stalling_runner(&["one"], Arc::default()), config());
    let failed = wait_for_status(&server, "a", unvalidated["id"].as_str().unwrap(), "failed").await;
    assert_eq!(failed["errors"]["data"][0]["code"], "input_file_not_found");
    assert_eq!(failed["errors"]["data"][0]["param"], "input_file_id");

    // One deleted while its batch runs fails the batch once the executor
    // next opens the file; the lines that finished are still written out.
    let running = create(&server, "a", TRANSCRIPTIONS, &input(&["one", "two"])).await;
    let id = running["id"].as_str().unwrap();
    wait_for(&server, "a", id, |batch| {
        batch["request_counts"]["completed"] == 1
    })
    .await;
    executor.abort();
    let _ = executor.await;
    let input_file = running["input_file_id"].as_str().unwrap();
    server
        .call(Method::DELETE, &format!("/v1/files/{input_file}"), "a")
        .await;
    let executor = start(&server, runner(|_| async { ok(json!({})) }), config());
    let failed = wait_for_status(&server, "a", id, "failed").await;
    assert_eq!(failed["errors"]["data"][0]["code"], "input_file_not_found");
    let output = file_lines(&server, "a", &failed["output_file_id"]).await;
    assert_eq!(custom_ids(&output), ["one"]);
    executor.abort();
}

#[tokio::test]
async fn a_batch_runs_every_line_and_writes_output_and_error_files() {
    let server = TestServer::new("batches-run");
    let refused = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let most_in_flight = Arc::new(AtomicUsize::new(0));
    let (refusals, flying, peak) = (refused.clone(), in_flight.clone(), most_in_flight.clone());
    let executor = start(
        &server,
        runner(move |job| {
            let (refusals, flying, peak) = (refusals.clone(), flying.clone(), peak.clone());
            async move {
                peak.fetch_max(flying.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(5)).await;
                flying.fetch_sub(1, Ordering::SeqCst);
                assert_eq!(job.endpoint, super::BatchEndpoint::Transcriptions);
                assert!(job.request_id.starts_with("batch_req_"));
                match job.custom_id.as_str() {
                    "bad" => LineOutcome {
                        status_code: 400,
                        body: json!({"error": {
                            "message": "No such model",
                            "type": "invalid_request_error",
                            "param": "model",
                            "code": "invalid_request",
                        }}),
                    },
                    // The server is at capacity twice, then serves it.
                    "busy" if refusals.fetch_add(1, Ordering::SeqCst) < 2 => LineOutcome {
                        status_code: 429,
                        body: json!({"error": {"message": "at capacity"}}),
                    },
                    custom_id => ok(json!({"text": custom_id, "model": job.body["model"]})),
                }
            }
        }),
        config(),
    );
    let batch = create(
        &server,
        "a",
        TRANSCRIPTIONS,
        &input(&["one", "bad", "busy", "four", "five"]),
    )
    .await;
    let id = batch["id"].as_str().unwrap();
    let done = wait_for_status(&server, "a", id, "completed").await;
    assert_eq!(
        done["request_counts"],
        json!({"total": 5, "completed": 4, "failed": 1})
    );
    for at in ["in_progress_at", "finalizing_at", "completed_at"] {
        assert!(done[at].is_i64(), "{at}");
    }
    assert_eq!(done["errors"], Value::Null);
    assert_eq!(refused.load(Ordering::SeqCst), 3);
    // Never more lines at once than the limit.
    assert_eq!(most_in_flight.load(Ordering::SeqCst), 2);

    // The output file: one line per request served, in input order.
    let output = file_lines(&server, "a", &done["output_file_id"]).await;
    assert_eq!(custom_ids(&output), ["one", "busy", "four", "five"]);
    let first = &output[0];
    let request_id = first["id"].as_str().unwrap();
    assert!(request_id.starts_with("batch_req_"));
    assert_eq!(
        *first,
        json!({
            "id": request_id,
            "custom_id": "one",
            "response": {
                "status_code": 200,
                "request_id": request_id,
                "body": {"text": "one", "model": "m"},
            },
            "error": null,
        })
    );
    // The error file: the route's error, as response and as error.
    let errors = file_lines(&server, "a", &done["error_file_id"]).await;
    assert_eq!(custom_ids(&errors), ["bad"]);
    assert_eq!(errors[0]["response"]["status_code"], 400);
    assert_eq!(
        errors[0]["response"]["body"]["error"]["message"],
        "No such model"
    );
    assert_eq!(
        errors[0]["error"],
        json!({"code": "invalid_request", "message": "No such model"})
    );

    // Both are `batch_output` files that expire, and the finished-line rows
    // are gone now that the files hold them.
    for file in [&done["output_file_id"], &done["error_file_id"]] {
        let file = read_json(
            server
                .call(
                    Method::GET,
                    &format!("/v1/files/{}", file.as_str().unwrap()),
                    "a",
                )
                .await,
        )
        .await;
        assert_eq!(file["purpose"], "batch_output");
        assert!(file["expires_at"].is_i64());
        assert!(file["filename"].as_str().unwrap().starts_with(id));
    }
    assert_eq!(result_rows(&server).await, 0);

    let metrics = bytes(
        server
            .call(Method::GET, "/v1/metrics/prometheus", "a")
            .await,
    )
    .await;
    let metrics = String::from_utf8(metrics).unwrap();
    for sample in [
        "izwi_batch_runtime_api_batches{status=\"completed\"} 1",
        "izwi_batch_runtime_api_batch_requests{outcome=\"completed\"} 4",
        "izwi_batch_runtime_api_batch_requests{outcome=\"failed\"} 1",
    ] {
        assert!(metrics.contains(sample), "{sample} in {metrics}");
    }
    executor.abort();
}

/// A runner that answers the lines named in `quick` and never answers the
/// others; `started` holds every line it was given.
fn stalling_runner(quick: &'static [&'static str], started: Arc<Mutex<Vec<String>>>) -> LineRunner {
    runner(move |job| {
        let started = started.clone();
        async move {
            started.lock().unwrap().push(job.custom_id.clone());
            if !quick.contains(&job.custom_id.as_str()) {
                std::future::pending::<()>().await;
            }
            ok(json!({"text": job.custom_id}))
        }
    })
}

#[tokio::test]
async fn cancelling_a_batch_starts_no_new_lines_and_keeps_finished_results() {
    let server = TestServer::new("batches-cancel");
    let started = Arc::new(Mutex::new(Vec::new()));
    let executor = start(
        &server,
        stalling_runner(&["one"], started.clone()),
        config(),
    );
    let batch = create(
        &server,
        "a",
        TRANSCRIPTIONS,
        &input(&["one", "two", "three", "four", "five"]),
    )
    .await;
    let id = batch["id"].as_str().unwrap();
    // One line is done and two are in flight, which is the limit.
    wait_for(&server, "a", id, |batch| {
        batch["request_counts"]["completed"] == 1 && started.lock().unwrap().len() == 3
    })
    .await;

    let response = server
        .call(Method::POST, &format!("/v1/batches/{id}/cancel"), "a")
        .await;
    assert_eq!(read_json(response).await["status"], "cancelling");
    let cancelled = wait_for_status(&server, "a", id, "cancelled").await;
    assert!(cancelled["cancelling_at"].is_i64());
    assert!(cancelled["cancelled_at"].is_i64());
    assert_eq!(
        cancelled["request_counts"],
        json!({"total": 5, "completed": 1, "failed": 0})
    );
    // The finished line is in the output file; nothing new was started.
    let output = file_lines(&server, "a", &cancelled["output_file_id"]).await;
    assert_eq!(custom_ids(&output), ["one"]);
    assert_eq!(cancelled["error_file_id"], Value::Null);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(*started.lock().unwrap(), ["one", "two", "three"]);
    executor.abort();
}

#[tokio::test]
async fn a_new_executor_resumes_a_batch_without_redoing_finished_lines() {
    let server = TestServer::new("batches-resume");
    let first_run = Arc::new(Mutex::new(Vec::new()));
    let executor = start(
        &server,
        stalling_runner(&["one", "two"], first_run.clone()),
        config(),
    );
    let batch = create(
        &server,
        "a",
        TRANSCRIPTIONS,
        &input(&["one", "two", "three", "four"]),
    )
    .await;
    let id = batch["id"].as_str().unwrap();
    wait_for(&server, "a", id, |batch| {
        batch["request_counts"]["completed"] == 2
    })
    .await;

    // The executor dies with two lines in flight. Its lease lapses, and the
    // next executor runs only the lines without a result.
    executor.abort();
    let _ = executor.await;
    assert_eq!(get(&server, "a", id).await["status"], "in_progress");
    let second_run = Arc::new(Mutex::new(Vec::new()));
    let executor = start(
        &server,
        stalling_runner(&["one", "two", "three", "four"], second_run.clone()),
        config(),
    );
    let done = wait_for_status(&server, "a", id, "completed").await;
    assert_eq!(
        done["request_counts"],
        json!({"total": 4, "completed": 4, "failed": 0})
    );
    let rerun: HashSet<String> = second_run.lock().unwrap().iter().cloned().collect();
    assert_eq!(
        rerun,
        HashSet::from(["three".to_string(), "four".to_string()])
    );
    let output = file_lines(&server, "a", &done["output_file_id"]).await;
    assert_eq!(custom_ids(&output), ["one", "two", "three", "four"]);
    executor.abort();

    // A graceful stop gives the lease back, so the next executor does not
    // wait for it to lapse.
    let mut slow = config();
    slow.lease = Duration::from_secs(3600);
    let batch = create(&server, "a", TRANSCRIPTIONS, &input(&["one", "two"])).await;
    let id = batch["id"].as_str().unwrap();
    let executor = start(
        &server,
        stalling_runner(&["one"], Arc::default()),
        slow.clone(),
    );
    wait_for(&server, "a", id, |batch| {
        batch["request_counts"]["completed"] == 1
    })
    .await;
    server.state.stream_shutdown.send_replace(true);
    executor.await.expect("the executor stops on shutdown");
    server.state.stream_shutdown.send_replace(false);
    let executor = start(
        &server,
        stalling_runner(&["one", "two"], Arc::default()),
        slow,
    );
    wait_for_status(&server, "a", id, "completed").await;
    executor.abort();
}

#[tokio::test]
async fn a_batch_past_its_window_expires_with_what_finished() {
    let server = TestServer::new("batches-expiry");
    let executor = start(&server, stalling_runner(&["one"], Arc::default()), config());
    let batch = create(
        &server,
        "a",
        TRANSCRIPTIONS,
        &input(&["one", "two", "three"]),
    )
    .await;
    let id = batch["id"].as_str().unwrap();
    wait_for(&server, "a", id, |batch| {
        batch["request_counts"]["completed"] == 1
    })
    .await;
    server
        .state
        .batch_api_store
        .set_expires_at_for_test(id, 1)
        .await
        .unwrap();
    let expired = wait_for_status(&server, "a", id, "expired").await;
    assert!(expired["expired_at"].is_i64());
    assert_eq!(expired["completed_at"], Value::Null);
    assert_eq!(
        expired["request_counts"],
        json!({"total": 3, "completed": 1, "failed": 0})
    );
    let output = file_lines(&server, "a", &expired["output_file_id"]).await;
    assert_eq!(custom_ids(&output), ["one"]);
    executor.abort();
    let _ = executor.await;

    // A batch nothing ran within its window expires without files.
    let batch = create(&server, "a", TRANSCRIPTIONS, &input(&["one"])).await;
    let id = batch["id"].as_str().unwrap();
    server
        .state
        .batch_api_store
        .set_expires_at_for_test(id, 1)
        .await
        .unwrap();
    let executor = start(&server, runner(|_| async { ok(json!({})) }), config());
    let expired = wait_for_status(&server, "a", id, "expired").await;
    assert_eq!(expired["output_file_id"], Value::Null);
    assert_eq!(expired["in_progress_at"], Value::Null);
    executor.abort();
}

#[tokio::test]
async fn input_lines_are_read_one_at_a_time_within_the_line_limit() {
    let content = b"first\n\n  \r\nsecond is much too long\nthird\r\nlast".to_vec();
    let mut lines = LineReader::new(Box::pin(std::io::Cursor::new(content)), 8);
    let mut read = Vec::new();
    while let Some((index, line)) = lines.next().await.unwrap() {
        read.push((
            index,
            line.map(|line| String::from_utf8(line).unwrap().trim().to_string()),
        ));
    }
    assert_eq!(
        read,
        [
            (0, Some("first".to_string())),
            (3, None),
            (4, Some("third".to_string())),
            (5, Some("last".to_string())),
        ]
    );
}

/// Transcription and translation lines through the routes' own code, up to
/// the model: none is on disk here, so each line ends at one of the route's
/// checks and the batch completes with an error file alone.
#[tokio::test]
async fn transcription_lines_run_the_routes_own_checks() {
    let server = TestServer::new("batches-asr");
    let wav = AudioEncoder::new(16_000, 1)
        .encode(&[0.0; 1600], AudioFormat::Wav)
        .unwrap();
    let audio = server.upload_ok("a", "user_data", "clip.wav", &wav).await;
    let others_audio = server.upload_ok("b", "user_data", "clip.wav", &wav).await;
    let not_audio = server.upload_ok("a", "batch", "notes.jsonl", b"{}").await;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&wav);
    let lines = [
        (
            "unknown-model",
            json!({"model": "no-such-model", "file_id": audio}),
        ),
        (
            "unknown-model-inline",
            json!({"model": "no-such-model", "audio_base64": encoded}),
        ),
        ("no-audio", json!({"model": "no-such-model"})),
        (
            "two-audios",
            json!({"model": "m", "file_id": audio, "audio_base64": encoded}),
        ),
        (
            "missing-file",
            json!({"model": "m", "file_id": "file-missing"}),
        ),
        (
            "others-file",
            json!({"model": "m", "file_id": others_audio}),
        ),
        ("wrong-purpose", json!({"model": "m", "file_id": not_audio})),
        (
            "bad-format",
            json!({"model": "Parakeet-TDT-0.6B-v3", "file_id": audio, "response_format": "yaml"}),
        ),
    ];
    let content: String = lines
        .iter()
        .map(|(id, body)| line(id, TRANSCRIPTIONS, body.clone()) + "\n")
        .collect();
    let mut real = config();
    real.max_requests = 100;
    let executor = start(
        &server,
        super::runner::audio_line_runner(server.state.clone()),
        real.clone(),
    );
    let batch = create(&server, "a", TRANSCRIPTIONS, &content).await;
    let done = wait_for_status(&server, "a", batch["id"].as_str().unwrap(), "completed").await;
    assert_eq!(
        done["request_counts"],
        json!({"total": 8, "completed": 0, "failed": 8})
    );
    assert_eq!(done["output_file_id"], Value::Null);
    let errors = file_lines(&server, "a", &done["error_file_id"]).await;
    let found = |custom_id: &str| {
        errors
            .iter()
            .find(|line| line["custom_id"] == custom_id)
            .unwrap_or_else(|| panic!("no line {custom_id}"))
    };
    for (custom_id, says) in [
        ("unknown-model", "no-such-model"),
        ("unknown-model-inline", "no-such-model"),
        ("no-audio", "exactly one of `file_id` or `audio_base64`"),
        ("two-audios", "exactly one of `file_id` or `audio_base64`"),
        ("missing-file", "No such File object: file-missing"),
        ("others-file", "No such File object"),
        ("wrong-purpose", "purpose `batch`"),
        ("bad-format", "yaml"),
    ] {
        let line = found(custom_id);
        assert_eq!(line["response"]["status_code"], 400, "{custom_id}");
        let message = line["error"]["message"].as_str().unwrap();
        assert!(message.contains(says), "{custom_id}: {message}");
        assert_eq!(
            line["response"]["body"]["error"]["message"],
            line["error"]["message"]
        );
    }

    // A translation line is the translation route's request.
    let batch = create(
        &server,
        "a",
        "/v1/audio/translations",
        &line(
            "untranslatable",
            "/v1/audio/translations",
            json!({"model": "Parakeet-TDT-0.6B-v3", "file_id": audio}),
        ),
    )
    .await;
    let done = wait_for_status(&server, "a", batch["id"].as_str().unwrap(), "completed").await;
    let errors = file_lines(&server, "a", &done["error_file_id"]).await;
    assert!(errors[0]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does not translate speech"));
    executor.abort();
}

/// Speech lines on the Fish Audio Drama 3 synthetic checkpoint: the audio
/// file a line writes is the audio `/v1/audio/speech` answers the same body
/// with, and a line the route refuses lands in the error file.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn speech_lines_write_the_speech_routes_audio_to_files() {
    use izwi_core::runtime_models::architectures::fish_drama3::test_support::write_lifecycle_synthetic_model;
    use izwi_core::ModelVariant;

    let _load = crate::api::fish_drama3_speech_tests::LOADS.lock().await;
    let root = std::env::temp_dir().join(format!("izwi-batches-speech-{}", uuid::Uuid::new_v4()));
    let models_dir = root.join("models");
    write_lifecycle_synthetic_model(
        &models_dir.join(ModelVariant::FishAudioDrama3.dir_name()),
        5,
    );
    let server = TestServer::with_models_dir(root, models_dir);

    // A seed whose audio ends on its own: the synthetic model's random
    // weights run into the frame budget for some.
    let body = |seed: u64| {
        json!({
            "model": "FishAudio-Drama-3",
            "input": "hello world.",
            "voice": "alloy",
            "temperature": 2.0,
            "max_tokens": 160,
            "seed": seed,
            // A line is batch work whatever tier it names.
            "service_tier": "priority",
        })
    };
    let mut expected = None;
    for seed in 0..24 {
        let response = server.post_json(SPEECH, "a", body(seed)).await;
        if response.status() == StatusCode::OK {
            expected = Some((seed, bytes(response).await));
            break;
        }
    }
    let (seed, expected) = expected.expect("a seed whose audio ends on eos");

    let mut unknown_model = body(seed);
    unknown_model["model"] = json!("no-such-model");
    let content = [
        line("spoken", SPEECH, body(seed)),
        line("refused", SPEECH, unknown_model),
    ]
    .join("\n");
    let executor = start(
        &server,
        super::runner::audio_line_runner(server.state.clone()),
        config(),
    );
    let batch = create(&server, "a", SPEECH, &content).await;
    let done = wait_for_status(&server, "a", batch["id"].as_str().unwrap(), "completed").await;
    assert_eq!(
        done["request_counts"],
        json!({"total": 2, "completed": 1, "failed": 1})
    );

    let output = file_lines(&server, "a", &done["output_file_id"]).await;
    assert_eq!(custom_ids(&output), ["spoken"]);
    let spoken = &output[0]["response"]["body"];
    assert_eq!(spoken["format"], "wav");
    assert_eq!(spoken["bytes"], expected.len());
    let audio_file = spoken["audio_file_id"].as_str().unwrap();
    assert_eq!(server.content("a", audio_file).await, expected);
    let file = read_json(
        server
            .call(Method::GET, &format!("/v1/files/{audio_file}"), "a")
            .await,
    )
    .await;
    assert_eq!(file["filename"], "spoken.wav");
    assert_eq!(file["purpose"], "batch_output");

    let errors = file_lines(&server, "a", &done["error_file_id"]).await;
    assert_eq!(custom_ids(&errors), ["refused"]);
    assert_eq!(errors[0]["response"]["status_code"], 400);
    assert!(errors[0]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("no-such-model"));
    executor.abort();
}

/// A batch of `tenant` over a new one-line input file, announced to
/// `webhook_url`. It is written to the store as the route writes it: the
/// route itself refuses a webhook while the test process has no secret.
async fn create_with_webhook(server: &TestServer, tenant: &str, webhook_url: &str) -> String {
    let template = create(server, tenant, TRANSCRIPTIONS, &input(&["one"])).await;
    let store = &server.state.batch_api_store;
    let template = store
        .load(template["id"].as_str().unwrap())
        .await
        .unwrap()
        .expect("the batch");
    let batch = store
        .create(
            &ArtifactTenant::parse(template.tenant_scope).unwrap(),
            &template.endpoint,
            &template.input_file_id,
            Some(&json!({"webhook_url": webhook_url})),
        )
        .await
        .unwrap();
    batch.id
}

#[tokio::test]
async fn a_batch_that_ends_posts_a_signed_event_to_its_webhook() {
    static ACCEPTS: [StatusCode; 1] = [StatusCode::OK];
    let server = TestServer::new("batches-webhook");
    let listener = Listener::start(&ACCEPTS).await;
    let id = create_with_webhook(&server, "a", &listener.url).await;
    let executor = start(
        &server,
        runner(|_| async { ok(json!({"text": "hi"})) }),
        ExecutorConfig {
            webhook: Some(local_policy("batch-secret")),
            ..config()
        },
    );
    wait_for_status(&server, "a", &id, "completed").await;

    // The delivery runs apart from the batch, so it lands a little later.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let delivery = loop {
        if let Some(delivery) = listener.deliveries().into_iter().next() {
            break delivery;
        }
        assert!(std::time::Instant::now() < deadline, "no delivery");
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert_signed(&delivery, b"batch-secret");
    let event: Value = serde_json::from_str(&delivery.body).unwrap();
    assert_eq!(event["id"], delivery.id.as_str());
    assert!(delivery.id.starts_with("evt_"));
    assert_eq!(event["object"], "event");
    assert_eq!(event["type"], "batch.completed");
    assert!(event["created_at"].is_i64());
    assert_eq!(event["data"], json!({"id": id}));
    // A batch without a webhook posts nothing.
    assert_eq!(listener.deliveries().len(), 1);
    executor.abort();
}

#[tokio::test]
async fn a_webhook_that_keeps_failing_leaves_the_batch_as_it_ended() {
    static REFUSES: [StatusCode; 1] = [StatusCode::INTERNAL_SERVER_ERROR];
    let server = TestServer::new("batches-webhook-fails");
    let listener = Listener::start(&REFUSES).await;
    let id = create_with_webhook(&server, "a", &listener.url).await;
    let executor = start(
        &server,
        runner(|_| async { ok(json!({"text": "hi"})) }),
        ExecutorConfig {
            webhook: Some(local_policy("batch-secret")),
            ..config()
        },
    );
    let completed = wait_for_status(&server, "a", &id, "completed").await;
    assert_eq!(
        completed["request_counts"],
        json!({"total": 1, "completed": 1, "failed": 0})
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while listener.deliveries().len() < 3 {
        assert!(std::time::Instant::now() < deadline, "three attempts");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(listener.deliveries().len(), 3, "no fourth attempt");
    assert_eq!(get(&server, "a", &id).await["status"], "completed");
    executor.abort();
}

#[tokio::test]
async fn a_batch_naming_a_webhook_is_refused_while_webhooks_are_off() {
    let server = TestServer::new("batches-webhook-off");
    let file = server
        .upload_ok("a", "batch", "input.jsonl", input(&["one"]).as_bytes())
        .await;
    let response = server
        .post_json(
            "/v1/batches",
            "a",
            json!({
                "input_file_id": file,
                "endpoint": TRANSCRIPTIONS,
                "completion_window": "24h",
                "metadata": {"webhook_url": "https://hooks.example.com/izwi"},
            }),
        )
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = read_json(response).await;
    assert_eq!(error["error"]["param"], "metadata.webhook_url");
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("IZWI_BATCH_WEBHOOK_SECRET"));
}
