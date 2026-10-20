use super::*;
use crate::batch_runtime::{
    store::{NewJobStage, NewRuntimeArtifact, NewRuntimeJob},
    types::{
        RuntimeArtifactKind, RuntimeArtifactRole, RuntimeJobKind, RuntimeJobStatus,
        RuntimeStageStatus,
    },
};
use crate::state::AppState;
use crate::test_support::env_lock;
use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header, Method, Request, StatusCode},
    response::Response,
    routing::get,
};
use base64::Engine;
use izwi_core::{
    audio::{AudioEncoder, AudioFormat},
    RuntimeService,
};
use izwi_hooks::{
    AuthProvider, AuthorizationDecision, AuthorizationRequest, EnterpriseHooks, HookError,
    HookResult, PolicyEngine, Principal, RequestEnvelope,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::Service;

#[tokio::test]
async fn enabled_wildcard_cors_adds_allow_origin_header() {
    let app = apply_cors(
        Router::new().route("/hello", get(|| async { "ok" })),
        &ServeRuntimeConfig {
            cors_enabled: true,
            cors_origins: vec!["*".to_string()],
            ..ServeRuntimeConfig::default()
        },
    );

    let response = send_request(
        app,
        Request::builder()
            .method(Method::GET)
            .uri("/hello")
            .header(header::ORIGIN, "http://localhost:3000")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;

    assert_eq!(
        response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
        Some(&HeaderValue::from_static("*"))
    );
}

#[tokio::test]
async fn disabled_cors_emits_no_allow_origin_header_for_browser_origin() {
    let app = apply_cors(
        Router::new().route("/hello", get(|| async { "ok" })),
        &ServeRuntimeConfig {
            cors_enabled: false,
            ..ServeRuntimeConfig::default()
        },
    );

    let response = send_request(
        app,
        Request::builder()
            .method(Method::GET)
            .uri("/hello")
            .header(header::ORIGIN, "http://localhost:3000")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;

    assert!(response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .is_none());
}

#[tokio::test]
async fn disabled_cors_still_allows_desktop_origin() {
    let app = apply_cors(
        Router::new().route("/hello", get(|| async { "ok" })),
        &ServeRuntimeConfig {
            cors_enabled: false,
            ..ServeRuntimeConfig::default()
        },
    );

    let response = send_request(
        app,
        Request::builder()
            .method(Method::GET)
            .uri("/hello")
            .header(header::ORIGIN, "tauri://localhost")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;

    assert_eq!(
        response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
        Some(&HeaderValue::from_static("tauri://localhost"))
    );
}

#[tokio::test]
async fn enabled_custom_cors_keeps_desktop_allowlist() {
    let app = apply_cors(
        Router::new().route("/hello", get(|| async { "ok" })),
        &ServeRuntimeConfig {
            cors_enabled: true,
            cors_origins: vec!["http://localhost:3000".to_string()],
            ..ServeRuntimeConfig::default()
        },
    );

    let browser_response = send_request(
        app.clone(),
        Request::builder()
            .method(Method::GET)
            .uri("/hello")
            .header(header::ORIGIN, "http://localhost:3000")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;

    assert_eq!(
        browser_response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
        Some(&HeaderValue::from_static("http://localhost:3000"))
    );

    let desktop_response = send_request(
        app,
        Request::builder()
            .method(Method::GET)
            .uri("/hello")
            .header(header::ORIGIN, "tauri://localhost")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;

    assert_eq!(
        desktop_response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
        Some(&HeaderValue::from_static("tauri://localhost"))
    );
}

#[tokio::test]
async fn enabled_ui_serves_index_as_fallback() {
    let ui_dir = test_ui_dir("enabled_ui_serves_index_as_fallback");
    std::fs::create_dir_all(&ui_dir).expect("ui dir should be created");
    std::fs::write(ui_dir.join("index.html"), "<html>ui</html>")
        .expect("index should be written");

    let app = apply_ui_fallback(
        Router::new().route("/hello", get(|| async { "ok" })),
        &ServeRuntimeConfig {
            ui_enabled: true,
            ui_dir: ui_dir.clone(),
            ..ServeRuntimeConfig::default()
        },
    );

    let response = send_request(
        app,
        Request::builder()
            .uri("/missing")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should read");
    assert_eq!(body, "<html>ui</html>");

    std::fs::remove_dir_all(ui_dir).expect("ui dir should be removed");
}

#[tokio::test]
async fn disabled_ui_returns_not_found_for_unknown_routes() {
    let app = apply_ui_fallback(
        Router::new().route("/hello", get(|| async { "ok" })),
        &ServeRuntimeConfig {
            ui_enabled: false,
            ..ServeRuntimeConfig::default()
        },
    );

    let response = send_request(
        app,
        Request::builder()
            .uri("/missing")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// PIPE-7: cancelling a running diarization stops its pipeline (the
/// job's cancellation fires), and the record reports the cancel.
#[tokio::test]
async fn diarization_cancel_stops_the_running_pipeline() {
    let (state, _temp_dir) = test_state("diarization_cancel_stops_the_running_pipeline", false);
    state.lifecycle.mark_ready();
    let mut pending = crate::diarization_store::tests::sample_record();
    pending.processing_status = crate::diarization_store::DiarizationProcessingStatus::Processing;
    let record = state
        .diarization_store
        .create_record(pending)
        .await
        .expect("record should be created");
    let job = state.diarization_jobs.register(&record.id);
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &serve_config);

    let response = send_request(
        app,
        build_request(
            Method::POST,
            &format!("/v1/diarizations/{}/cancel", record.id),
            None,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_json(response).await;
    assert_eq!(body["processing_status"], "failed");
    assert_eq!(body["processing_error"], "Cancelled by user.");
    assert!(job.cancellation().is_cancelled());
}

#[tokio::test]
async fn canonical_history_routes_still_resolve() {
    let (app, temp_dir) = test_api_app("canonical_history_routes_still_resolve", true);

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/diarizations",
        None,
        StatusCode::OK,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::PATCH,
        "/v1/diarizations/missing",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/diarizations/missing/audio",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/diarizations/missing/reruns",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/diarizations/missing/summary/regenerate",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/text-to-speech",
        None,
        StatusCode::OK,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/text-to-speech/missing/audio",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/voice-designs",
        None,
        StatusCode::OK,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/voice-designs/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/voice-clones",
        None,
        StatusCode::OK,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/voice-clones/missing/audio",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    for removed_path in [
        "/v1/text-to-speech-generations",
        "/v1/voice-design-generations",
        "/v1/voice-clone-generations",
    ] {
        assert_route_status(
            app.clone(),
            Method::GET,
            removed_path,
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
    }

    drop(temp_dir);
}

#[tokio::test]
async fn unified_transcription_job_list_pages_across_job_kinds() {
    let (app, _temp_dir) = test_api_app(
        "unified_transcription_job_list_pages_across_job_kinds",
        true,
    );
    let audio_body = tiny_audio_json_body();
    let saa_body = tiny_audio_json_body_with_fields(
        r#""model_id":"Granite-Speech-4.1-2B-Plus","min_speakers":2"#,
    );
    let mut created_ids = Vec::new();
    for (uri, body) in [
        (
            "/v1/speech-to-text/jobs?job_kind=transcription",
            audio_body.as_str(),
        ),
        (
            "/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
            saa_body.as_str(),
        ),
        ("/v1/diarizations", audio_body.as_str()),
        (
            "/v1/speech-to-text/jobs?job_kind=transcription",
            audio_body.as_str(),
        ),
    ] {
        let response =
            send_request(app.clone(), build_request(Method::POST, uri, Some(body))).await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let payload = read_json(response).await;
        created_ids.push(
            payload
                .get("id")
                .and_then(|value| value.as_str())
                .expect("created job id")
                .to_string(),
        );
    }

    // Walk the merged list one row at a time. Before the merged cursor this
    // returned `has_more: true` with `next_cursor: null` after the first page.
    let mut seen_ids = Vec::new();
    let mut cursor: Option<String> = None;
    for _ in 0..10 {
        let uri = match &cursor {
            Some(cursor) => format!("/v1/speech-to-text/jobs?job_kind=all&limit=1&cursor={cursor}"),
            None => "/v1/speech-to-text/jobs?job_kind=all&limit=1".to_string(),
        };
        let response = send_request(app.clone(), build_request(Method::GET, &uri, None)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let payload = read_json(response).await;
        let records = payload["records"].as_array().expect("records array");
        assert!(records.len() <= 1);
        seen_ids.extend(
            records
                .iter()
                .filter_map(|record| record["id"].as_str().map(str::to_string)),
        );
        let has_more = payload["pagination"]["has_more"].as_bool().unwrap();
        let next_cursor = payload["pagination"]["next_cursor"]
            .as_str()
            .map(str::to_string);
        assert_eq!(
            has_more,
            next_cursor.is_some(),
            "has_more must come with a cursor"
        );
        if !has_more {
            break;
        }
        cursor = next_cursor;
    }

    let mut expected = created_ids.clone();
    expected.sort();
    let mut actual = seen_ids.clone();
    actual.sort();
    assert_eq!(
        actual, expected,
        "every job appears exactly once across pages"
    );
}

#[tokio::test]
async fn unified_transcription_list_filters_and_totals() {
    let (app, _temp_dir) = test_api_app("unified_transcription_list_filters_and_totals", true);
    let audio_body = tiny_audio_json_body();
    let saa_body = tiny_audio_json_body_with_fields(
        r#""model_id":"Granite-Speech-4.1-2B-Plus","min_speakers":2"#,
    );
    for (uri, body) in [
        (
            "/v1/speech-to-text/jobs?job_kind=transcription",
            audio_body.as_str(),
        ),
        (
            "/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
            saa_body.as_str(),
        ),
        ("/v1/diarizations", audio_body.as_str()),
    ] {
        let response =
            send_request(app.clone(), build_request(Method::POST, uri, Some(body))).await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }
    let total = |uri: &'static str| {
        let app = app.clone();
        async move {
            let response = send_request(app, build_request(Method::GET, uri, None)).await;
            assert_eq!(response.status(), StatusCode::OK, "{uri}");
            let body = read_json(response).await;
            let records = body["records"].as_array().map(Vec::len).unwrap_or(0);
            (body["total"].as_u64().unwrap_or(u64::MAX), records)
        }
    };

    assert_eq!(total("/v1/speech-to-text/jobs").await, (3, 3));
    assert_eq!(
        total("/v1/speech-to-text/jobs?with_speakers=true").await,
        (2, 2)
    );
    assert_eq!(
        total("/v1/speech-to-text/jobs?job_kind=transcription&with_speakers=true").await,
        (0, 0)
    );
    // Jobs can fail in the background here (no model is installed), so check
    // that every job matches exactly one status rather than which one.
    let mut by_status = 0;
    for uri in [
        "/v1/speech-to-text/jobs?status=pending",
        "/v1/speech-to-text/jobs?status=processing",
        "/v1/speech-to-text/jobs?status=ready",
        "/v1/speech-to-text/jobs?status=failed",
    ] {
        by_status += total(uri).await.0;
    }
    assert_eq!(by_status, 3);
    assert_eq!(
        total("/v1/speech-to-text/jobs?status=pending,processing,ready,failed").await,
        (3, 3)
    );
    assert_eq!(
        total("/v1/speech-to-text/jobs?q=no-such-text").await,
        (0, 0)
    );
    let bad = send_request(
        app.clone(),
        build_request(Method::GET, "/v1/speech-to-text/jobs?status=done", None),
    )
    .await;
    assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn deleting_a_transcription_cancels_its_durable_job() {
    let (state, _temp_dir) = test_state("deleting_a_transcription_cancels_its_durable_job", false);
    state.lifecycle.mark_ready();
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &serve_config);

    let audio_body = tiny_audio_json_body();
    let created = send_request(
        app.clone(),
        build_request(
            Method::POST,
            "/v1/speech-to-text/jobs?job_kind=transcription",
            Some(audio_body.as_str()),
        ),
    )
    .await;
    assert_eq!(created.status(), StatusCode::ACCEPTED);
    let record_id = read_json(created).await["id"]
        .as_str()
        .expect("record id")
        .to_string();
    let job = state
        .batch_runtime_store
        .get_active_job_for_route_record(
            RuntimeJobKind::AsrTranscription,
            "transcription",
            &record_id,
        )
        .await
        .expect("job lookup")
        .expect("accepted transcription has an active durable job");

    let deleted = send_request(
        app.clone(),
        build_request(
            Method::DELETE,
            &format!("/v1/speech-to-text/jobs/{record_id}?job_kind=transcription"),
            None,
        ),
    )
    .await;
    assert_eq!(deleted.status(), StatusCode::OK);

    let job = state
        .batch_runtime_store
        .get_job(&job.id)
        .await
        .expect("job read")
        .expect("job row is kept");
    assert_eq!(job.status, RuntimeJobStatus::Cancelled);
}

#[tokio::test]
async fn unified_transcription_job_read_routes_support_both_job_kinds() {
    let (app, temp_dir) = test_api_app(
        "unified_transcription_job_read_routes_support_both_job_kinds",
        true,
    );
    let audio_body = tiny_audio_json_body();

    let transcription_create = send_request(
        app.clone(),
        build_request(
            Method::POST,
            "/v1/speech-to-text/jobs?job_kind=transcription",
            Some(audio_body.as_str()),
        ),
    )
    .await;
    assert_eq!(transcription_create.status(), StatusCode::ACCEPTED);
    let transcription_body = read_json(transcription_create).await;
    let transcription_id = transcription_body
        .get("id")
        .and_then(|value| value.as_str())
        .expect("transcription id should exist")
        .to_string();

    let saa_body = tiny_audio_json_body_with_fields(
        r#""model_id":"Granite-Speech-4.1-2B-Plus","min_speakers":2"#,
    );
    let saa_create = send_request(
        app.clone(),
        build_request(
            Method::POST,
            "/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
            Some(saa_body.as_str()),
        ),
    )
    .await;
    assert_eq!(saa_create.status(), StatusCode::ACCEPTED);
    let saa_body = read_json(saa_create).await;
    assert_eq!(
        saa_body
            .get("transcription_mode")
            .and_then(|value| value.as_str()),
        Some("speaker_attributed_asr")
    );
    assert_eq!(
        saa_body.get("model_id").and_then(|value| value.as_str()),
        Some("Granite-Speech-4.1-2B-Plus")
    );
    let saa_id = saa_body
        .get("id")
        .and_then(|value| value.as_str())
        .expect("SAA id should exist")
        .to_string();

    let diarization_create = send_request(
        app.clone(),
        build_request(Method::POST, "/v1/diarizations", Some(audio_body.as_str())),
    )
    .await;
    assert_eq!(diarization_create.status(), StatusCode::ACCEPTED);
    let diarization_body = read_json(diarization_create).await;
    let diarization_id = diarization_body
        .get("id")
        .and_then(|value| value.as_str())
        .expect("diarization id should exist")
        .to_string();

    let unified_list = send_request(
        app.clone(),
        build_request(Method::GET, "/v1/speech-to-text/jobs?job_kind=all", None),
    )
    .await;
    assert_eq!(unified_list.status(), StatusCode::OK);
    let list_payload = read_json(unified_list).await;
    let records = list_payload
        .get("records")
        .and_then(|value| value.as_array())
        .expect("records array should exist");
    let kinds = records
        .iter()
        .filter_map(|record| record.get("kind").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(
        kinds.contains(&"transcription"),
        "unified list should include transcription records"
    );
    assert!(
        kinds.contains(&"diarization"),
        "unified list should include diarization records"
    );
    assert!(
        kinds.contains(&"speaker_attributed_asr"),
        "unified list should include SAA records"
    );

    let transcription_list = send_request(
        app.clone(),
        build_request(
            Method::GET,
            "/v1/speech-to-text/jobs?job_kind=transcription",
            None,
        ),
    )
    .await;
    assert_eq!(transcription_list.status(), StatusCode::OK);
    let transcription_list_payload = read_json(transcription_list).await;
    let transcription_records = transcription_list_payload
        .get("records")
        .and_then(|value| value.as_array())
        .expect("transcription records array should exist");
    assert!(transcription_records.iter().all(|record| {
        record.get("kind").and_then(|value| value.as_str()) == Some("transcription")
    }));

    let saa_list = send_request(
        app.clone(),
        build_request(
            Method::GET,
            "/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
            None,
        ),
    )
    .await;
    assert_eq!(saa_list.status(), StatusCode::OK);
    let saa_list_payload = read_json(saa_list).await;
    let saa_records = saa_list_payload
        .get("records")
        .and_then(|value| value.as_array())
        .expect("SAA records array should exist");
    assert!(saa_records.iter().all(|record| {
        record.get("kind").and_then(|value| value.as_str()) == Some("speaker_attributed_asr")
    }));

    let unified_list_with_limit = send_request(
        app.clone(),
        build_request(Method::GET, "/v1/speech-to-text/jobs?limit=25", None),
    )
    .await;
    assert_eq!(unified_list_with_limit.status(), StatusCode::OK);
    let list_with_limit_payload = read_json(unified_list_with_limit).await;
    assert_eq!(
        list_with_limit_payload
            .get("pagination")
            .and_then(|value| value.get("limit"))
            .and_then(|value| value.as_u64()),
        Some(25)
    );

    let unified_transcription_record = send_request(
        app.clone(),
        build_request(
            Method::GET,
            format!(
                "/v1/speech-to-text/jobs/{}?job_kind=transcription",
                transcription_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(unified_transcription_record.status(), StatusCode::OK);
    let transcription_payload = read_json(unified_transcription_record).await;
    assert_eq!(
        transcription_payload
            .get("kind")
            .and_then(|value| value.as_str()),
        Some("transcription")
    );

    let unified_saa_record = send_request(
        app.clone(),
        build_request(
            Method::GET,
            format!(
                "/v1/speech-to-text/jobs/{}?job_kind=speaker_attributed_asr",
                saa_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(unified_saa_record.status(), StatusCode::OK);
    let saa_payload = read_json(unified_saa_record).await;
    assert_eq!(
        saa_payload.get("kind").and_then(|value| value.as_str()),
        Some("speaker_attributed_asr")
    );

    let saa_as_transcription = send_request(
        app.clone(),
        build_request(
            Method::GET,
            format!("/v1/speech-to-text/jobs/{}?job_kind=transcription", saa_id).as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(saa_as_transcription.status(), StatusCode::NOT_FOUND);

    let unified_diarization_record = send_request(
        app.clone(),
        build_request(
            Method::GET,
            format!(
                "/v1/speech-to-text/jobs/{}?job_kind=diarization",
                diarization_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(unified_diarization_record.status(), StatusCode::OK);
    let diarization_payload = read_json(unified_diarization_record).await;
    assert_eq!(
        diarization_payload
            .get("kind")
            .and_then(|value| value.as_str()),
        Some("diarization")
    );

    let transcription_audio = send_request(
        app.clone(),
        build_request(
            Method::GET,
            format!(
                "/v1/speech-to-text/jobs/{}/audio?job_kind=transcription",
                transcription_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(transcription_audio.status(), StatusCode::OK);

    let saa_audio = send_request(
        app.clone(),
        build_request(
            Method::GET,
            format!(
                "/v1/speech-to-text/jobs/{}/audio?job_kind=speaker_attributed_asr",
                saa_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(saa_audio.status(), StatusCode::OK);

    let diarization_audio = send_request(
        app.clone(),
        build_request(
            Method::GET,
            format!(
                "/v1/speech-to-text/jobs/{}/audio?job_kind=diarization",
                diarization_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(diarization_audio.status(), StatusCode::OK);

    for removed_path in [
        "/v1/transcriptions/jobs?job_kind=all",
        "/v1/transcriptions/jobs/missing/audio?job_kind=transcription",
    ] {
        assert_route_status(
            app.clone(),
            Method::GET,
            removed_path,
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
    }

    let canonical_diarization_get = send_request(
        app,
        build_request(
            Method::GET,
            format!("/v1/diarizations/{}", diarization_id).as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(canonical_diarization_get.status(), StatusCode::OK);

    drop(temp_dir);
}

#[tokio::test]
async fn unified_transcription_job_mutation_routes_support_both_job_kinds() {
    let (app, temp_dir) = test_api_app(
        "unified_transcription_job_mutation_routes_support_both_job_kinds",
        true,
    );
    let audio_body = tiny_audio_json_body();

    let transcription_create = send_request(
        app.clone(),
        build_request(
            Method::POST,
            "/v1/speech-to-text/jobs?job_kind=transcription",
            Some(audio_body.as_str()),
        ),
    )
    .await;
    assert_eq!(transcription_create.status(), StatusCode::ACCEPTED);
    let transcription_id = read_json(transcription_create)
        .await
        .get("id")
        .and_then(|value| value.as_str())
        .expect("transcription id should exist")
        .to_string();

    let saa_body =
        tiny_audio_json_body_with_fields(r#""model_id":"Granite-Speech-4.1-2B-Plus""#);
    let saa_create = send_request(
        app.clone(),
        build_request(
            Method::POST,
            "/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
            Some(saa_body.as_str()),
        ),
    )
    .await;
    assert_eq!(saa_create.status(), StatusCode::ACCEPTED);
    let saa_id = read_json(saa_create)
        .await
        .get("id")
        .and_then(|value| value.as_str())
        .expect("SAA id should exist")
        .to_string();

    let diarization_create = send_request(
        app.clone(),
        build_request(
            Method::POST,
            "/v1/speech-to-text/jobs?job_kind=diarization",
            Some(audio_body.as_str()),
        ),
    )
    .await;
    assert_eq!(diarization_create.status(), StatusCode::ACCEPTED);
    let diarization_id = read_json(diarization_create)
        .await
        .get("id")
        .and_then(|value| value.as_str())
        .expect("diarization id should exist")
        .to_string();

    let update_diarization = send_request(
        app.clone(),
        build_request(
            Method::PATCH,
            format!(
                "/v1/speech-to-text/jobs/{}?job_kind=diarization",
                diarization_id
            )
            .as_str(),
            Some("{\"speaker_name_overrides\":{}}"),
        ),
    )
    .await;
    assert_eq!(update_diarization.status(), StatusCode::OK);

    let rerun_diarization = send_request(
        app.clone(),
        build_request(
            Method::POST,
            format!(
                "/v1/speech-to-text/jobs/{}/reruns?job_kind=diarization",
                diarization_id
            )
            .as_str(),
            Some("{}"),
        ),
    )
    .await;
    assert_eq!(rerun_diarization.status(), StatusCode::ACCEPTED);

    let cancel_diarization = send_request(
        app.clone(),
        build_request(
            Method::POST,
            format!(
                "/v1/speech-to-text/jobs/{}/cancel?job_kind=diarization",
                diarization_id
            )
            .as_str(),
            Some("{}"),
        ),
    )
    .await;
    assert_eq!(cancel_diarization.status(), StatusCode::OK);

    let regen_transcription_summary = send_request(
        app.clone(),
        build_request(
            Method::POST,
            format!(
                "/v1/speech-to-text/jobs/{}/summary/regenerate?job_kind=transcription",
                transcription_id
            )
            .as_str(),
            Some("{}"),
        ),
    )
    .await;
    assert_eq!(regen_transcription_summary.status(), StatusCode::OK);

    let regen_saa_summary = send_request(
        app.clone(),
        build_request(
            Method::POST,
            format!(
                "/v1/speech-to-text/jobs/{}/summary/regenerate?job_kind=speaker_attributed_asr",
                saa_id
            )
            .as_str(),
            Some("{}"),
        ),
    )
    .await;
    assert_eq!(regen_saa_summary.status(), StatusCode::OK);

    let regen_diarization_summary = send_request(
        app.clone(),
        build_request(
            Method::POST,
            format!(
                "/v1/speech-to-text/jobs/{}/summary/regenerate?job_kind=diarization",
                diarization_id
            )
            .as_str(),
            Some("{}"),
        ),
    )
    .await;
    assert_eq!(regen_diarization_summary.status(), StatusCode::OK);

    let delete_transcription = send_request(
        app.clone(),
        build_request(
            Method::DELETE,
            format!(
                "/v1/speech-to-text/jobs/{}?job_kind=transcription",
                transcription_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(delete_transcription.status(), StatusCode::OK);

    let delete_saa = send_request(
        app.clone(),
        build_request(
            Method::DELETE,
            format!(
                "/v1/speech-to-text/jobs/{}?job_kind=speaker_attributed_asr",
                saa_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(delete_saa.status(), StatusCode::OK);

    let delete_diarization = send_request(
        app.clone(),
        build_request(
            Method::DELETE,
            format!(
                "/v1/speech-to-text/jobs/{}?job_kind=diarization",
                diarization_id
            )
            .as_str(),
            None,
        ),
    )
    .await;
    assert_eq!(delete_diarization.status(), StatusCode::OK);

    for removed_path in [
        "/v1/transcriptions/jobs/missing/reruns?job_kind=diarization",
        "/v1/transcriptions/jobs/missing/cancel?job_kind=diarization",
        "/v1/transcriptions/jobs/missing/summary/regenerate?job_kind=transcription",
    ] {
        assert_route_status(
            app.clone(),
            Method::POST,
            removed_path,
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;
    }

    let canonical_diarization_create = send_request(
        app,
        build_request(
            Method::POST,
            "/v1/diarizations",
            Some(tiny_audio_json_body().as_str()),
        ),
    )
    .await;
    assert_eq!(canonical_diarization_create.status(), StatusCode::ACCEPTED);

    drop(temp_dir);
}

#[tokio::test]
async fn diarization_create_rejects_pipeline_models_it_cannot_run() {
    let (app, temp_dir) = test_api_app(
        "diarization_create_rejects_pipeline_models_it_cannot_run",
        true,
    );
    for (field, expected) in [
        (r#""asr_model":"bogus""#, "Unknown diarization ASR model id"),
        (
            r#""asr_model":"LFM2.5-Audio-1.5B-GGUF""#,
            "cannot transcribe in the diarization pipeline",
        ),
        (
            r#""aligner_model":"bogus""#,
            "is not a forced aligner model",
        ),
        (
            r#""aligner_model":"Parakeet-TDT-0.6B-v3""#,
            "is not a forced aligner model",
        ),
    ] {
        let body = tiny_audio_json_body_with_fields(field);
        let response = send_request(
            app.clone(),
            build_request(Method::POST, "/v1/diarizations", Some(body.as_str())),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
        let payload = read_json(response).await;
        let message = payload["error"]["message"]
            .as_str()
            .expect("error message should exist");
        assert!(message.contains(expected), "{field}: {message}");
    }

    let accepted = send_request(
        app,
        build_request(
            Method::POST,
            "/v1/diarizations",
            Some(
                tiny_audio_json_body_with_fields(
                    r#""asr_model":"parakeet-tdt-0.6b-v3","aligner_model":"Qwen3-ForcedAligner-0.6B""#,
                )
                .as_str(),
            ),
        ),
    )
    .await;
    assert_eq!(accepted.status(), StatusCode::ACCEPTED);
    let record = read_json(accepted).await;
    assert_eq!(record["asr_model_id"], "Parakeet-TDT-0.6B-v3");
    assert_eq!(record["aligner_model_id"], "Qwen3-ForcedAligner-0.6B");

    drop(temp_dir);
}

#[tokio::test]
async fn granite_speech_is_rejected_for_diarization_jobs() {
    let (app, temp_dir) = test_api_app("granite_speech_is_rejected_for_diarization_jobs", true);
    let body = tiny_audio_json_body_with_fields(r#""model_id":"Granite-Speech-4.1-2B-Plus""#);

    let response = send_request(
        app,
        build_request(
            Method::POST,
            "/v1/speech-to-text/jobs?job_kind=diarization",
            Some(body.as_str()),
        ),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let payload = read_json(response).await;
    let message = payload["error"]["message"]
        .as_str()
        .expect("error message should exist");
    assert!(message.contains("speaker-attributed ASR"));
    assert!(message.contains("job_kind=speaker_attributed_asr"));

    drop(temp_dir);
}

#[tokio::test]
async fn canonical_history_mutation_routes_still_resolve() {
    let (app, temp_dir) = test_api_app("canonical_history_mutation_routes_still_resolve", true);

    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/diarizations",
        Some("{}"),
        StatusCode::BAD_REQUEST,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::PUT,
        "/v1/diarizations/missing",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/diarizations/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/diarizations/missing/reruns",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/text-to-speech",
        Some("{}"),
        StatusCode::BAD_REQUEST,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/text-to-speech/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/voice-designs",
        Some("{}"),
        StatusCode::BAD_REQUEST,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/voice-designs/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/voice-clones",
        Some("{}"),
        StatusCode::BAD_REQUEST,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/voice-clones/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    for removed_path in [
        "/v1/text-to-speech-generations",
        "/v1/voice-design-generations",
        "/v1/voice-clone-generations",
    ] {
        assert_route_status(
            app.clone(),
            Method::POST,
            removed_path,
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;
    }

    drop(temp_dir);
}

#[tokio::test]
async fn api_namespace_does_not_fall_back_to_ui_when_enabled() {
    let (app, temp_dir) =
        test_api_app("api_namespace_does_not_fall_back_to_ui_when_enabled", true);

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/not-a-route",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/not-a-route",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::PUT,
        "/v1/not-a-route",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app,
        Method::DELETE,
        "/v1/not-a-route",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    drop(temp_dir);
}

#[tokio::test]
async fn legacy_history_routes_return_not_found() {
    let (app, temp_dir) = test_api_app("legacy_history_routes_return_not_found", true);

    for (method, path, body) in [
        (Method::POST, "/v1/audio/diarize", Some("{}")),
        (Method::POST, "/v1/audio/diarizations", Some("{}")),
        (Method::GET, "/v1/transcriptions", None),
        (Method::POST, "/v1/transcriptions", Some("{}")),
        (Method::GET, "/v1/transcriptions/missing", None),
        (Method::DELETE, "/v1/transcriptions/missing", None),
        (Method::GET, "/v1/transcriptions/missing/audio", None),
        (
            Method::POST,
            "/v1/transcriptions/missing/summary/regenerate",
            Some("{}"),
        ),
    ] {
        assert_route_status(app.clone(), method, path, body, StatusCode::NOT_FOUND).await;
    }

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/transcription/records",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/transcription/records",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/transcription/records/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/transcription/records/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/transcription/records/missing/audio",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/diarization/records",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/diarization/records",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::PATCH,
        "/v1/diarization/records/missing",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::PUT,
        "/v1/diarization/records/missing",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/diarization/records/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/diarization/records/missing/audio",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/diarization/records/missing/rerun",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/voice-design/records",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/voice-design/records",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/voice-design/records/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/voice-design/records/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/voice-cloning/records",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::POST,
        "/v1/voice-cloning/records",
        Some("{}"),
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/voice-cloning/records/missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    assert_route_status(
        app,
        Method::GET,
        "/v1/voice-cloning/records/missing/audio",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    drop(temp_dir);
}

#[tokio::test]
async fn internal_metrics_route_is_available_for_bench_compat() {
    let (app, temp_dir) = test_api_app("internal_metrics_route_compat", false);

    assert_route_status(
        app.clone(),
        Method::GET,
        "/internal/metrics",
        None,
        StatusCode::OK,
    )
    .await;
    assert_route_status(app, Method::GET, "/v1/metrics", None, StatusCode::OK).await;

    drop(temp_dir);
}

#[tokio::test]
async fn text_only_tts_idempotency_replays_and_conflicts_without_duplicate_work() {
    use crate::speech_history_store::SpeechRouteKind;

    let (state, _root) = test_state("text_only_tts_idempotency", false);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let body = serde_json::json!({
        "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "text": "Atomic speech request"
    })
    .to_string();

    let first = send_request(
        app.clone(),
        build_request_with_idempotency(&body, "tts-request-1"),
    )
    .await;
    assert_eq!(first.status(), StatusCode::ACCEPTED);
    let first = read_json(first).await;
    let first_id = first["id"].as_str().unwrap().to_string();
    assert_eq!(first["processing_status"], "pending");

    // Replays must bypass fresh admission: the original request itself
    // fills this one-job capacity limit.
    state
        .batch_runtime_store
        .set_tts_admission_limits_for_test(Some((1, 32)));
    let replay = send_request(
        app.clone(),
        build_request_with_idempotency(&body, "tts-request-1"),
    )
    .await;
    state
        .batch_runtime_store
        .set_tts_admission_limits_for_test(None);
    assert_eq!(replay.status(), StatusCode::ACCEPTED);
    assert_eq!(
        read_json(replay).await["id"].as_str(),
        Some(first_id.as_str())
    );

    let changed = serde_json::json!({
        "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "text": "Different speech request"
    })
    .to_string();
    let conflict = send_request(
        app,
        build_request_with_idempotency(&changed, "tts-request-1"),
    )
    .await;
    assert_eq!(conflict.status(), StatusCode::CONFLICT);

    let jobs = state
        .batch_runtime_store
        .list_active_jobs_by_kind(RuntimeJobKind::TtsSpeech)
        .await
        .unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].route_record_id.as_deref(), Some(first_id.as_str()));
    let stages = state
        .batch_runtime_store
        .list_stages_for_job(&jobs[0].id)
        .await
        .unwrap();
    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].status, RuntimeStageStatus::Queued);
    let (records, _) = state
        .speech_history_store
        .list_records_page(SpeechRouteKind::TextToSpeech, 10, None)
        .await
        .unwrap();
    assert_eq!(records.len(), 1);
}

#[tokio::test]
async fn text_only_tts_accepts_maximum_bounded_input_with_idempotency() {
    use crate::batch_runtime::store::MAX_DURABLE_TTS_TEXT_BYTES;

    let (state, _root) = test_state("bounded_tts_idempotency", false);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state, &config);
    let body = serde_json::json!({
        "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "text": "x".repeat(MAX_DURABLE_TTS_TEXT_BYTES)
    })
    .to_string();

    let first = send_request(
        app.clone(),
        build_request_with_idempotency(&body, "maximum-text"),
    )
    .await;
    assert_eq!(first.status(), StatusCode::ACCEPTED);
    let first_id = read_json(first).await["id"].as_str().unwrap().to_string();

    let replay = send_request(app, build_request_with_idempotency(&body, "maximum-text")).await;
    assert_eq!(replay.status(), StatusCode::ACCEPTED);
    assert_eq!(
        read_json(replay).await["id"].as_str(),
        Some(first_id.as_str())
    );
}

/// EDGE-F14: a malformed JSON body on a local JSON route is an OpenAI
/// error object, not axum's plain-text rejection.
#[tokio::test]
async fn malformed_json_bodies_are_openai_errors_on_local_routes() {
    let (state, _root) = test_state("malformed_json", false);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state, &config);
    for path in [
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/audio/speech",
        "/v1/chat/threads",
    ] {
        let response = send_request(
            app.clone(),
            build_request(Method::POST, path, Some("{\"model\": ")),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/json",
            "{path}"
        );
        let body = read_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error", "{path}");
        assert_eq!(body["error"]["code"], "invalid_value", "{path}");
    }
}

/// TPIPE-14: an over-limit speech input is refused with 413 on both
/// speech routes before a model is loaded or a record is created.
#[tokio::test]
async fn over_limit_speech_input_is_payload_too_large_on_both_routes() {
    // `test_state` holds the env lock; nothing sets the text limit.
    let (state, _root) = test_state("tts_input_limit", false);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state, &config);
    let text = "x".repeat(1024 * 1024 + 1);
    let openai = serde_json::json!({
        "model": "Kokoro-82M",
        "input": text,
        "voice": "af_heart"
    })
    .to_string();
    let response = send_request(
        app.clone(),
        build_request(Method::POST, "/v1/audio/speech", Some(&openai)),
    )
    .await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let first_party = serde_json::json!({
        "model_id": "Kokoro-82M",
        "text": text,
        "stream": true
    })
    .to_string();
    let response = send_request(
        app,
        build_request(Method::POST, "/v1/text-to-speech", Some(&first_party)),
    )
    .await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

fn keyed_request(path: &str, body: &str, key: &str, tenant: &str) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .header("idempotency-key", key)
        .header("x-test-tenant", tenant)
        .body(Body::from(body.to_string()))
        .expect("request should build")
}

async fn count_rows(state: &AppState, sql: &str) -> i64 {
    use sea_orm::ConnectionTrait;
    let db = state.batch_runtime_store.connection().await.unwrap();
    db.query_one_raw(crate::db::raw::statement(db, sql, vec![]).unwrap())
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<i64>(0)
        .unwrap()
}

/// DUR-F3: a keyed transcription retry replays the original job instead
/// of creating a second one, a different body is a conflict, and keys
/// are scoped per tenant. Nothing is written to the legacy table.
#[tokio::test]
async fn keyed_transcription_replays_conflicts_and_is_tenant_scoped() {
    let mut hooks = EnterpriseHooks::noop();
    hooks.auth = Arc::new(HeaderTenantAuth);
    let (state, _root) = test_state_with_hooks("keyed_transcription", false, hooks);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let path = "/v1/speech-to-text/jobs?job_kind=transcription";
    let body = tiny_audio_json_body();

    let first = send_request(app.clone(), keyed_request(path, &body, "asr-1", "a")).await;
    assert_eq!(first.status(), StatusCode::ACCEPTED);
    let first_id = read_json(first).await["id"].as_str().unwrap().to_string();

    let retry = send_request(app.clone(), keyed_request(path, &body, "asr-1", "a")).await;
    assert_eq!(retry.status(), StatusCode::ACCEPTED);
    assert_eq!(
        read_json(retry).await["id"].as_str(),
        Some(first_id.as_str())
    );

    let changed = tiny_audio_json_body_with_fields(r#""language":"French""#);
    let conflict = send_request(app.clone(), keyed_request(path, &changed, "asr-1", "a")).await;
    assert_eq!(conflict.status(), StatusCode::CONFLICT);

    let other = send_request(app.clone(), keyed_request(path, &body, "asr-1", "b")).await;
    assert_eq!(other.status(), StatusCode::ACCEPTED);
    assert_ne!(
        read_json(other).await["id"].as_str(),
        Some(first_id.as_str())
    );

    assert_eq!(
        count_rows(
            &state,
            "SELECT COUNT(*) FROM runtime_jobs WHERE job_kind = 'asr_transcription'"
        )
        .await,
        2
    );
    assert_eq!(
        count_rows(&state, "SELECT COUNT(*) FROM transcription_records").await,
        2
    );
    assert_eq!(
        count_rows(&state, "SELECT COUNT(*) FROM idempotency_keys").await,
        0
    );
}

/// DUR-F3: reference-voice TTS takes the same keyed path as text-only TTS.
#[tokio::test]
async fn keyed_reference_tts_replays_without_duplicate_work() {
    let mut hooks = EnterpriseHooks::noop();
    hooks.auth = Arc::new(HeaderTenantAuth);
    let (state, _root) = test_state_with_hooks("keyed_reference_tts", false, hooks);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let samples = (0..12_000)
        .map(|index| ((index as f32) * 0.05).sin() * 0.2)
        .collect::<Vec<_>>();
    let wav = AudioEncoder::new(24_000, 1)
        .encode(&samples, AudioFormat::Wav)
        .expect("reference wav should encode");
    let reference = base64::engine::general_purpose::STANDARD.encode(wav);
    let body = serde_json::json!({
        "model_id": "Qwen3-TTS-12Hz-1.7B-Base",
        "text": "Clone this voice",
        "reference_audio": reference,
        "reference_text": "reference words"
    })
    .to_string();

    let first = send_request(
        app.clone(),
        keyed_request("/v1/text-to-speech", &body, "ref-1", "a"),
    )
    .await;
    assert_eq!(first.status(), StatusCode::ACCEPTED);
    let first_id = read_json(first).await["id"].as_str().unwrap().to_string();

    let retry = send_request(
        app.clone(),
        keyed_request("/v1/text-to-speech", &body, "ref-1", "a"),
    )
    .await;
    assert_eq!(retry.status(), StatusCode::ACCEPTED);
    assert_eq!(
        read_json(retry).await["id"].as_str(),
        Some(first_id.as_str())
    );

    let changed = serde_json::json!({
        "model_id": "Qwen3-TTS-12Hz-1.7B-Base",
        "text": "Clone this voice",
        "reference_audio": reference,
        "reference_text": "other reference words"
    })
    .to_string();
    let conflict = send_request(
        app,
        keyed_request("/v1/text-to-speech", &changed, "ref-1", "a"),
    )
    .await;
    assert_eq!(conflict.status(), StatusCode::CONFLICT);

    assert_eq!(
        count_rows(
            &state,
            "SELECT COUNT(*) FROM runtime_jobs WHERE job_kind = 'tts_speech'"
        )
        .await,
        1
    );
    assert_eq!(
        count_rows(&state, "SELECT COUNT(*) FROM speech_history_records").await,
        1
    );
}

#[tokio::test]
async fn keyed_streaming_tts_rejects_before_creating_records() {
    use crate::speech_history_store::SpeechRouteKind;

    let (state, _root) = test_state("keyed_tts_unsupported_shapes", false);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let body = serde_json::json!({
        "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "text": "stream me",
        "stream": true
    })
    .to_string();
    let response = send_request(
        app,
        build_request_with_idempotency(&body, "unsupported-key"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let (records, _) = state
        .speech_history_store
        .list_records_page(SpeechRouteKind::TextToSpeech, 10, None)
        .await
        .unwrap();
    assert!(records.is_empty());
}

#[tokio::test]
async fn text_only_tts_admission_rejection_leaves_no_failed_placeholder() {
    use crate::speech_history_store::SpeechRouteKind;

    let (state, _root) = test_state("atomic_tts_admission", false);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let body = serde_json::json!({
        "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "text": "first"
    })
    .to_string();
    let accepted = send_request(
        app.clone(),
        build_request(Method::POST, "/v1/text-to-speech", Some(&body)),
    )
    .await;
    assert_eq!(accepted.status(), StatusCode::ACCEPTED);

    state
        .batch_runtime_store
        .set_tts_admission_limits_for_test(Some((1, 32)));
    let rejected = send_request(
        app,
        build_request(
            Method::POST,
            "/v1/text-to-speech",
            Some(
                &serde_json::json!({
                    "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "text": "second"
                })
                .to_string(),
            ),
        ),
    )
    .await;
    state
        .batch_runtime_store
        .set_tts_admission_limits_for_test(None);
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    let (records, _) = state
        .speech_history_store
        .list_records_page(SpeechRouteKind::TextToSpeech, 10, None)
        .await
        .unwrap();
    assert_eq!(records.len(), 1);
}

async fn speech_replay_fixture(
    state: &AppState,
    tenant: Option<[u8; 32]>,
    opaque: bool,
) -> (String, String, Option<String>) {
    use crate::artifact_store::{
        ArtifactRetention, ArtifactTenant, ArtifactWrite, AttemptArtifactWrite,
    };
    use crate::batch_runtime::store::{sha256_hex, NewStageOutputArtifact};
    use crate::speech_history_store::{
        NewSpeechHistoryRecord, SpeechHistoryProcessingStatus, SpeechRouteKind,
    };
    let samples = vec![0.1_f32, -0.2, 0.3];
    let record = state
        .speech_history_store
        .create_record(NewSpeechHistoryRecord {
            stop_reason: None,
            route_kind: SpeechRouteKind::TextToSpeech,
            processing_status: SpeechHistoryProcessingStatus::Ready,
            processing_error: None,
            model_id: Some("FishAudio-S2-Pro".into()),
            speaker: None,
            language: None,
            saved_voice_id: None,
            speed: None,
            input_text: "hello".into(),
            voice_description: None,
            reference_text: None,
            generation_time_ms: 12.0,
            audio_duration_secs: Some(3.0 / 44100.0),
            rtf: Some(0.5),
            tokens_generated: Some(3),
            audio_mime_type: "audio/wav".into(),
            audio_filename: Some("test.wav".into()),
            audio_bytes: AudioEncoder::new(44100, 1)
                .encode(&samples, AudioFormat::Wav)
                .unwrap(),
        })
        .await
        .unwrap();
    let job = state.batch_runtime_store.create_job(NewRuntimeJob {
        job_kind: RuntimeJobKind::TtsSpeech, status: RuntimeJobStatus::Queued, priority: 0,
        model_id: Some("FishAudio-S2-Pro".into()), capability: Some("tts".into()),
        route_record_kind: Some("text_to_speech".into()), route_record_id: Some(record.id.clone()),
        input_media_asset_id: None, input_text_asset_id: None,
        request_json: serde_json::json!({"tenant_key":tenant,"route_kind":"text_to_speech","model_id":"FishAudio-S2-Pro","input_text":"hello","request":{}}),
        model_snapshot_json: serde_json::json!({"version":1}), retry_policy_json: serde_json::json!({}),
        max_attempts: 1, idempotency_key: None, correlation_id: None,
    }).await.unwrap();
    state
        .batch_runtime_store
        .create_stage(NewJobStage {
            job_id: job.id.clone(),
            sequence: 0,
            stage_kind: "tts_synthesize".into(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".into()),
            model_id: job.model_id.clone(),
            max_attempts: 1,
            input_artifact_ids: vec![],
        })
        .await
        .unwrap();
    let claimed = state
        .batch_runtime_store
        .claim_next_stage("test-worker", 60000)
        .await
        .unwrap()
        .unwrap();
    let bytes: Vec<u8> = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect();
    let metadata = serde_json::json!({"version":1,"sequence":0,"segment":0,"sample_offset":0,"sample_count":3,"sample_rate":44100});
    let media_asset_id = if opaque {
        let artifact = state
            .artifact_store
            .put_attempt_artifact(
                &ArtifactTenant::from_scheduling_key(tenant),
                &claimed.lease().unwrap(),
                AttemptArtifactWrite {
                    publication_key: "speech-pcm/00000000000000000000".into(),
                    artifact_kind: RuntimeArtifactKind::Audio,
                    artifact_role: RuntimeArtifactRole::OutputIntermediate,
                    metadata_json: metadata,
                    runtime_retention_policy: "test".into(),
                    object: ArtifactWrite {
                        content_type: "audio/pcm-f32le".into(),
                        filename: Some("chunk.f32le".into()),
                        bytes,
                        retention: ArtifactRetention::Job,
                    },
                },
                serde_json::json!({"publication_started":true}),
            )
            .await
            .unwrap();
        assert!(artifact.storage_key.is_none());
        artifact.media_asset_id
    } else {
        let key = state
            .media_ingest
            .persist_generated_audio(
                "legacy-replay-test".into(),
                Some("chunk.pcm"),
                "audio/pcm-f32le",
                &bytes,
                "speech_replay",
            )
            .await
            .unwrap();
        let artifact = state
            .batch_runtime_store
            .publish_stage_output_artifact(
                &claimed.lease().unwrap(),
                NewStageOutputArtifact {
                    publication_key: "speech-pcm/00000000000000000000".into(),
                    artifact_kind: RuntimeArtifactKind::Audio,
                    artifact_role: RuntimeArtifactRole::OutputIntermediate,
                    media_asset_id: None,
                    text_asset_id: None,
                    storage_key: Some(key),
                    content_type: Some("audio/pcm-f32le".into()),
                    filename: Some("chunk.pcm".into()),
                    size_bytes: Some(bytes.len() as u64),
                    sha256: Some(sha256_hex(&bytes)),
                    metadata_json: metadata,
                    retention_policy: "test".into(),
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert!(artifact.media_asset_id.is_none());
        None
    };
    state
        .batch_runtime_store
        .complete_stage(&claimed.lease().unwrap(), vec![])
        .await
        .unwrap()
        .unwrap();
    (record.id, job.id, media_asset_id)
}

#[tokio::test]
async fn durable_speech_replay_preserves_pcm_stats_and_terminal_cancel() {
    let (state, _root) = test_state("durable_speech_replay", false);
    state.lifecycle.mark_ready();
    let (id, job_id, _) = speech_replay_fixture(&state, None, true).await;
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let path = format!("/v1/text-to-speech/{id}/events");
    let response = send_request(app.clone(), build_request(Method::GET, &path, None)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let events: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events[0]["durable"], true);
    assert_eq!(events[1]["sample_rate"], 44100);
    assert_eq!(events[2]["sequence"], 0);
    let streamed = base64::engine::general_purpose::STANDARD
        .decode(events[2]["audio_base64"].as_str().unwrap())
        .unwrap();
    assert_eq!(
        streamed,
        AudioEncoder::new(44100, 1)
            .encode(&[0.1, -0.2, 0.3], AudioFormat::RawI16)
            .unwrap()
    );
    assert_eq!(events[3]["event"], "final");
    assert_eq!(events[3]["generation_time_ms"], 12.0);
    assert_eq!(events[3]["tokens_generated"], 3);
    assert_eq!(events[4]["event"], "done");
    let response = send_request(
        app.clone(),
        build_request(Method::GET, &format!("{path}?after_sequence=0"), None),
    )
    .await;
    drop(response); // Merely dropping a subscriber never cancels a durable job.
    assert_eq!(
        state
            .batch_runtime_store
            .get_job(&job_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        RuntimeJobStatus::Completed
    );
    assert_route_status(
        app.clone(),
        Method::GET,
        &format!("{path}?after_sequence=99"),
        None,
        StatusCode::BAD_REQUEST,
    )
    .await;
    assert_route_status(
        app,
        Method::POST,
        &format!("/v1/text-to-speech/{id}/cancel"),
        Some("{}"),
        StatusCode::OK,
    )
    .await;
}

#[tokio::test]
async fn deleting_durable_speech_removes_opaque_replay_reference() {
    let (state, _root) = test_state("durable_speech_delete_replay", false);
    state.lifecycle.mark_ready();
    let (id, job_id, artifact_id) = speech_replay_fixture(&state, None, true).await;
    let artifact_id = crate::artifact_store::ArtifactId::parse(artifact_id.unwrap()).unwrap();
    let tenant = crate::artifact_store::ArtifactTenant::from_scheduling_key(None);
    assert!(state
        .artifact_store
        .stat(&tenant, &artifact_id)
        .await
        .is_ok());
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };

    assert_route_status(
        create_router(state.clone(), &config),
        Method::DELETE,
        &format!("/v1/text-to-speech/{id}"),
        None,
        StatusCode::OK,
    )
    .await;
    assert!(matches!(
        state.artifact_store.stat(&tenant, &artifact_id).await,
        Err(crate::artifact_store::ArtifactStoreError::NotFound)
    ));
    assert!(state
        .batch_runtime_store
        .speech_pcm_after(&job_id, None, 1)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn failed_opaque_replay_delete_retains_reference_for_retry() {
    let (state, root) = test_state("durable_speech_delete_retry", false);
    state.lifecycle.mark_ready();
    let (id, job_id, artifact_id) = speech_replay_fixture(&state, None, true).await;
    let artifact_id = crate::artifact_store::ArtifactId::parse(artifact_id.unwrap()).unwrap();
    let tenant = crate::artifact_store::ArtifactTenant::from_scheduling_key(None);
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);

    // Turn the provider root into a regular file so deletion fails with an
    // error other than NotFound while leaving all provider bytes recoverable.
    let media_root = root.0.join("media");
    let media_backup = root.0.join("media-delete-retry-backup");
    std::fs::rename(&media_root, &media_backup).unwrap();
    std::fs::write(&media_root, b"provider unavailable").unwrap();
    assert_route_status(
        app.clone(),
        Method::DELETE,
        &format!("/v1/text-to-speech/{id}"),
        None,
        StatusCode::INTERNAL_SERVER_ERROR,
    )
    .await;
    std::fs::remove_file(&media_root).unwrap();
    std::fs::rename(&media_backup, &media_root).unwrap();

    assert!(matches!(
        state.artifact_store.stat(&tenant, &artifact_id).await,
        Err(crate::artifact_store::ArtifactStoreError::NotFound)
    ));
    assert_eq!(
        state
            .batch_runtime_store
            .speech_pcm_after(&job_id, None, 1)
            .await
            .unwrap()
            .len(),
        1
    );

    assert_route_status(
        app,
        Method::DELETE,
        &format!("/v1/text-to-speech/{id}"),
        None,
        StatusCode::OK,
    )
    .await;
    assert!(state
        .batch_runtime_store
        .speech_pcm_after(&job_id, None, 1)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn legacy_raw_key_speech_replay_remains_readable_and_deletable() {
    let (state, _root) = test_state("legacy_speech_replay", false);
    state.lifecycle.mark_ready();
    let (id, job_id, artifact_id) = speech_replay_fixture(&state, None, false).await;
    assert!(artifact_id.is_none());
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let response = send_request(
        app.clone(),
        build_request(
            Method::GET,
            &format!("/v1/text-to-speech/{id}/events"),
            None,
        ),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 65_536)
        .await
        .unwrap();
    assert!(String::from_utf8(body.to_vec())
        .unwrap()
        .contains("\"event\":\"chunk\""));

    assert_route_status(
        app,
        Method::DELETE,
        &format!("/v1/text-to-speech/{id}"),
        None,
        StatusCode::OK,
    )
    .await;
    assert!(state
        .batch_runtime_store
        .speech_pcm_after(&job_id, None, 1)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn malformed_speech_sequence_fails_closed_without_repeating_audio() {
    use sea_orm::ConnectionTrait;

    let (state, _root) = test_state("malformed_speech_sequence", false);
    state.lifecycle.mark_ready();
    let (id, job_id, _) = speech_replay_fixture(&state, None, false).await;
    let artifact = state
        .batch_runtime_store
        .speech_pcm_after(&job_id, None, 1)
        .await
        .unwrap()
        .pop()
        .unwrap();
    let db = state.batch_runtime_store.connection().await.unwrap();
    db.execute_raw(
        crate::db::raw::statement(
            db,
            "UPDATE runtime_artifacts SET metadata_json = ?1 WHERE id = ?2",
            vec![
                serde_json::json!({"version":1,"sequence":9,"segment":0,"sample_offset":0,"sample_count":3,"sample_rate":44100}).to_string().into(),
                artifact.id.into(),
            ],
        )
        .unwrap(),
    )
    .await
    .unwrap();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let response = send_request(
        create_router(state, &config),
        build_request(
            Method::GET,
            &format!("/v1/text-to-speech/{id}/events"),
            None,
        ),
    )
    .await;
    let body = axum::body::to_bytes(response.into_body(), 65_536)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("\"event\":\"error\""));
    assert!(body.contains("missing or reordered sequence"));
    assert!(!body.contains("\"event\":\"chunk\""));
}

#[tokio::test]
async fn durable_speech_listener_disconnect_does_not_cancel_running_job() {
    use crate::speech_history_store::{SpeechHistoryProcessingStatus, SpeechRouteKind};
    use futures::StreamExt;
    let (state, _root) = test_state("durable_speech_disconnect", false);
    state.lifecycle.mark_ready();
    let (id, job_id, _) = speech_replay_fixture(&state, None, true).await;
    // Stand the fixture's completed job back up as running, as a worker
    // still executing it would leave it.
    let db = state.batch_runtime_store.connection().await.unwrap();
    sea_orm::ConnectionTrait::execute_raw(
        db,
        crate::db::raw::statement(
            db,
            "UPDATE runtime_jobs SET status = 'running', finished_at = NULL WHERE id = ?1",
            vec![job_id.clone().into()],
        )
        .unwrap(),
    )
    .await
    .unwrap();
    state
        .speech_history_store
        .update_processing_status(
            SpeechRouteKind::TextToSpeech,
            id.clone(),
            SpeechHistoryProcessingStatus::Processing,
            None,
        )
        .await
        .unwrap();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let response = send_request(
        create_router(state.clone(), &config),
        build_request(
            Method::GET,
            &format!("/v1/text-to-speech/{id}/events"),
            None,
        ),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    assert!(
        String::from_utf8(body.next().await.unwrap().unwrap().to_vec())
            .unwrap()
            .contains("created")
    );
    drop(body);
    assert_eq!(
        state
            .batch_runtime_store
            .get_job(&job_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        RuntimeJobStatus::Running
    );
    assert_eq!(
        state
            .speech_history_store
            .get_record(SpeechRouteKind::TextToSpeech, id)
            .await
            .unwrap()
            .unwrap()
            .processing_status,
        SpeechHistoryProcessingStatus::Processing
    );
}

#[tokio::test]
async fn durable_speech_replay_rejects_other_tenant() {
    let (state, _root) = test_state("durable_speech_other_tenant", false);
    state.lifecycle.mark_ready();
    let (id, _, _) = speech_replay_fixture(&state, Some([7; 32]), true).await;
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    assert_route_status(
        create_router(state, &config),
        Method::GET,
        &format!("/v1/text-to-speech/{id}/events"),
        None,
        StatusCode::FORBIDDEN,
    )
    .await;
}

#[tokio::test]
async fn runtime_job_routes_return_trace_and_support_cancel_retry() {
    let (state, temp_dir) = test_state("runtime_job_routes_return_trace", false);
    state.lifecycle.mark_ready();
    let job = state
        .batch_runtime_store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("Qwen3-TTS-0.6B".to_string()),
            capability: Some("tts".to_string()),
            route_record_kind: None,
            route_record_id: None,
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: serde_json::json!({"text": "hello"}),
            model_snapshot_json: serde_json::json!({}),
            retry_policy_json: serde_json::json!({"max_attempts": 1}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .expect("runtime job");
    let stage = state
        .batch_runtime_store
        .create_stage(NewJobStage {
            job_id: job.id.clone(),
            sequence: 0,
            stage_kind: "tts_synthesize".to_string(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".to_string()),
            model_id: job.model_id.clone(),
            max_attempts: 1,
            input_artifact_ids: vec![],
        })
        .await
        .expect("runtime stage");
    let artifact = state
        .batch_runtime_store
        .create_artifact(NewRuntimeArtifact {
            job_id: job.id.clone(),
            stage_id: Some(stage.id.clone()),
            artifact_kind: RuntimeArtifactKind::Metadata,
            artifact_role: RuntimeArtifactRole::Debug,
            media_asset_id: None,
            text_asset_id: None,
            storage_key: Some("runtime/jobs/test/trace.json".to_string()),
            content_type: Some("application/json".to_string()),
            filename: Some("trace.json".to_string()),
            size_bytes: Some(2),
            sha256: None,
            metadata_json: serde_json::json!({"source": "test"}),
            retention_policy: "debug".to_string(),
        })
        .await
        .expect("runtime artifact");
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state, &serve_config);

    let trace = send_request(
        app.clone(),
        build_request(Method::GET, &format!("/v1/jobs/{}", job.id), None),
    )
    .await;
    assert_eq!(trace.status(), StatusCode::OK);
    let trace_body = read_json(trace).await;
    assert_eq!(trace_body["job"]["id"].as_str(), Some(job.id.as_str()));
    assert_eq!(
        trace_body["stages"][0]["id"].as_str(),
        Some(stage.id.as_str())
    );
    assert_eq!(
        trace_body["artifacts"][0]["id"].as_str(),
        Some(artifact.id.as_str())
    );

    let artifacts = send_request(
        app.clone(),
        build_request(Method::GET, &format!("/v1/jobs/{}/artifacts", job.id), None),
    )
    .await;
    assert_eq!(artifacts.status(), StatusCode::OK);
    let artifacts_body = read_json(artifacts).await;
    assert_eq!(
        artifacts_body["artifacts"][0]["id"].as_str(),
        Some(artifact.id.as_str())
    );

    let cancel = send_request(
        app.clone(),
        build_request(Method::POST, &format!("/v1/jobs/{}/cancel", job.id), None),
    )
    .await;
    assert_eq!(cancel.status(), StatusCode::OK);
    let cancel_body = read_json(cancel).await;
    assert_eq!(cancel_body["job"]["status"].as_str(), Some("cancelled"));
    assert_eq!(
        cancel_body["stages"][0]["status"].as_str(),
        Some("cancelled")
    );

    let retry = send_request(
        app,
        build_request(Method::POST, &format!("/v1/jobs/{}/retry", job.id), None),
    )
    .await;
    assert_eq!(retry.status(), StatusCode::OK);
    let retry_body = read_json(retry).await;
    assert_eq!(retry_body["job"]["status"].as_str(), Some("queued"));
    assert_eq!(retry_body["stages"][0]["status"].as_str(), Some("retrying"));

    drop(temp_dir);
}

#[tokio::test]
async fn runtime_job_routes_hide_jobs_owned_by_another_tenant() {
    let mut principal = Principal::local_anonymous();
    principal.id = "tenant-a-user".to_string();
    principal.tenant_id = Some("tenant-a".to_string());
    let owner_key =
        crate::api::request_context::RequestContext::new("test".to_string(), principal.clone())
            .tenant_key()
            .expect("authenticated tenant has a key");
    let mut hooks = EnterpriseHooks::noop();
    hooks.auth = Arc::new(FixedPrincipalAuth(principal));
    let (owner_state, owner_dir) = test_state_with_hooks("runtime_job_tenant", false, hooks);
    owner_state.lifecycle.mark_ready();
    let job = owner_state
        .batch_runtime_store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("Qwen3-TTS-0.6B".to_string()),
            capability: Some("tts".to_string()),
            route_record_kind: None,
            route_record_id: None,
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: serde_json::json!({"tenant_key": owner_key, "text": "secret"}),
            model_snapshot_json: serde_json::json!({}),
            retry_policy_json: serde_json::json!({"max_attempts": 1}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .expect("runtime job");
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };

    // The same store seen by the anonymous principal: the job is missing.
    let mut anonymous_state = owner_state.clone();
    anonymous_state.enterprise_hooks = EnterpriseHooks::noop();
    let anonymous = create_router(anonymous_state, &serve_config);
    for (method, suffix) in [
        (Method::GET, ""),
        (Method::GET, "/artifacts"),
        (Method::POST, "/cancel"),
        (Method::POST, "/retry"),
    ] {
        let path = format!("/v1/jobs/{}{suffix}", job.id);
        assert_route_status(
            anonymous.clone(),
            method,
            &path,
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
    }
    let unchanged = owner_state
        .batch_runtime_store
        .get_job(&job.id)
        .await
        .expect("job lookup")
        .expect("job exists");
    assert_eq!(unchanged.status, RuntimeJobStatus::Queued);

    let owner = create_router(owner_state, &serve_config);
    let trace = send_request(
        owner,
        build_request(Method::GET, &format!("/v1/jobs/{}", job.id), None),
    )
    .await;
    assert_eq!(trace.status(), StatusCode::OK);
    assert_eq!(
        read_json(trace).await["job"]["request_json"]["text"],
        "secret"
    );

    drop(owner_dir);
}

#[tokio::test]
async fn media_upload_registers_runtime_assets_and_canonical_derivative() {
    let (app, temp_dir) = test_api_app("media_upload_runtime_assets", false);
    let upload_body = media_upload_audio_json_body();

    let upload = send_request(
        app.clone(),
        build_request(Method::POST, "/v1/media", Some(upload_body.as_str())),
    )
    .await;
    assert_eq!(upload.status(), StatusCode::OK);
    let upload_body = read_json(upload).await;
    assert!(upload_body["media_asset_id"].as_str().is_some());
    assert!(upload_body["canonical_media_asset_id"].as_str().is_some());
    let canonical_path = upload_body["canonical_path"]
        .as_str()
        .expect("canonical path should be returned");
    assert!(
        canonical_path.contains("/canonical/"),
        "canonical path should use canonical namespace: {canonical_path}"
    );
    let expected_canonical_url = format!("/v1/media/{canonical_path}");
    assert_eq!(
        upload_body["canonical_url"].as_str(),
        Some(expected_canonical_url.as_str())
    );

    let canonical = send_request(
        app,
        build_request(
            Method::GET,
            upload_body["canonical_url"]
                .as_str()
                .expect("canonical url should be returned"),
            None,
        ),
    )
    .await;
    assert_eq!(canonical.status(), StatusCode::OK);
    assert_eq!(
        canonical
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("audio/wav")
    );

    drop(temp_dir);
}

#[tokio::test]
async fn root_and_v1_probe_routes_are_available() {
    let (app, temp_dir) = test_api_app("probe_routes_available", false);

    assert_route_status(app.clone(), Method::GET, "/livez", None, StatusCode::OK).await;
    assert_route_status(app.clone(), Method::GET, "/readyz", None, StatusCode::OK).await;
    assert_route_status(app.clone(), Method::GET, "/v1/live", None, StatusCode::OK).await;
    assert_route_status(app, Method::GET, "/v1/ready", None, StatusCode::OK).await;

    drop(temp_dir);
}

#[tokio::test]
async fn text_to_speech_list_reports_total_and_validates_filters() {
    let (app, _temp_dir) = test_api_app(
        "text_to_speech_list_reports_total_and_validates_filters",
        false,
    );
    let ok = send_request(
        app.clone(),
        build_request(
            Method::GET,
            "/v1/text-to-speech?q=hello&voice_kind=saved&status=ready,failed&created_after=0",
            None,
        ),
    )
    .await;
    assert_eq!(ok.status(), StatusCode::OK);
    let body = read_json(ok).await;
    assert_eq!(body["total"].as_u64(), Some(0));
    assert_eq!(body["records"].as_array().map(Vec::len), Some(0));

    for bad in [
        "/v1/text-to-speech?voice_kind=robot",
        "/v1/text-to-speech?status=done",
    ] {
        let response = send_request(app.clone(), build_request(Method::GET, bad, None)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{bad}");
    }
}

/// A thread is titled from its first exchange. With no loaded model to
/// write one, the title is the first message's opening words; a thread the
/// user renamed keeps its name.
#[tokio::test]
async fn chat_thread_title_route_titles_untitled_threads_only() {
    let (state, _root) = test_state("chat_thread_title_route", false);
    state.lifecycle.mark_ready();
    let thread = state.chat_store.create_thread(None, None).await.unwrap();
    let pending = state
        .chat_store
        .prepare_user_message(thread.id.clone(), "  Name one\nplanet  ".into(), None)
        .await
        .unwrap();
    state
        .chat_store
        .append_turn_with_system_prompt(
            pending,
            "Thinking about planets</think>Mars.".into(),
            "Qwen3.5-0.8B".into(),
            1,
            1.0,
            None,
            None,
        )
        .await
        .unwrap();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let title_path = format!("/v1/chat/threads/{}/title", thread.id);

    let response = send_request(
        app.clone(),
        build_request(
            Method::POST,
            &title_path,
            Some(r#"{"model":"Qwen3.5-0.8B"}"#),
        ),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(read_json(response).await["title"], "Name one planet");

    state
        .chat_store
        .update_thread_settings(thread.id.clone(), Some("Planets".into()), None)
        .await
        .unwrap();
    let response = send_request(
        app.clone(),
        build_request(Method::POST, &title_path, Some("{}")),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(read_json(response).await["title"], "Planets");

    let response = send_request(
        app.clone(),
        build_request(
            Method::POST,
            &title_path,
            Some(r#"{"model":"not-a-model"}"#),
        ),
    )
    .await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a titled thread is never regenerated"
    );
    let response = send_request(
        app,
        build_request(Method::POST, "/v1/chat/threads/missing/title", Some("{}")),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn chat_message_routes_delete_and_refuse_the_wrong_targets() {
    let (state, _root) = test_state("chat_message_routes", false);
    state.lifecycle.mark_ready();
    let thread = state.chat_store.create_thread(None, None).await.unwrap();
    let pending = state
        .chat_store
        .prepare_user_message(thread.id.clone(), "Hi".into(), None)
        .await
        .unwrap();
    let (user, reply) = state
        .chat_store
        .append_turn_with_system_prompt(
            pending,
            "Hello".into(),
            "Qwen3.5-0.8B".into(),
            1,
            1.0,
            None,
            None,
        )
        .await
        .unwrap();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let message_path = |id: &str| format!("/v1/chat/threads/{}/messages/{id}", thread.id);

    let cases = [
        (
            Method::POST,
            format!("{}/regenerate", message_path(&user.id)),
            r#"{"model":"Qwen3.5-0.8B"}"#,
            StatusCode::BAD_REQUEST,
        ),
        (
            Method::POST,
            format!("{}/regenerate", message_path("missing")),
            r#"{"model":"Qwen3.5-0.8B"}"#,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::PATCH,
            message_path(&reply.id),
            r#"{"model":"Qwen3.5-0.8B","content":"Edited"}"#,
            StatusCode::BAD_REQUEST,
        ),
        (
            Method::PATCH,
            message_path(&user.id),
            r#"{"model":"Qwen3.5-0.8B","content":"  "}"#,
            StatusCode::BAD_REQUEST,
        ),
    ];
    for (method, path, body, status) in cases {
        let response = send_request(app.clone(), build_request(method, &path, Some(body))).await;
        assert_eq!(response.status(), status, "{path}");
    }

    let response = send_request(
        app.clone(),
        build_request(Method::DELETE, &message_path(&reply.id), None),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(read_json(response).await["deleted"], true);
    let response = send_request(
        app.clone(),
        build_request(Method::DELETE, &message_path(&reply.id), None),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let messages = state
        .chat_store
        .list_messages(thread.id.clone())
        .await
        .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].id, user.id);
}

#[tokio::test]
async fn chat_thread_list_supports_paging_and_search() {
    let (app, _temp_dir) = test_api_app("chat_thread_list_supports_paging_and_search", false);
    for title in ["First topic", "Second topic", "Unrelated"] {
        let body = format!(r#"{{"title":"{title}"}}"#);
        let response = send_request(
            app.clone(),
            build_request(Method::POST, "/v1/chat/threads", Some(body.as_str())),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let unpaged = read_json(
        send_request(
            app.clone(),
            build_request(Method::GET, "/v1/chat/threads", None),
        )
        .await,
    )
    .await;
    assert_eq!(unpaged["threads"].as_array().map(Vec::len), Some(3));
    assert!(unpaged.get("pagination").is_none());
    assert_eq!(unpaged["total_count"].as_u64(), Some(3));

    let page = read_json(
        send_request(
            app.clone(),
            build_request(Method::GET, "/v1/chat/threads?limit=2", None),
        )
        .await,
    )
    .await;
    assert_eq!(page["threads"].as_array().map(Vec::len), Some(2));
    assert_eq!(page["pagination"]["has_more"].as_bool(), Some(true));
    let cursor = page["pagination"]["next_cursor"].as_str().expect("cursor");
    let rest = read_json(
        send_request(
            app.clone(),
            build_request(
                Method::GET,
                &format!("/v1/chat/threads?limit=2&cursor={cursor}"),
                None,
            ),
        )
        .await,
    )
    .await;
    assert_eq!(rest["threads"].as_array().map(Vec::len), Some(1));
    assert_eq!(rest["pagination"]["has_more"].as_bool(), Some(false));

    let searched = read_json(
        send_request(app, build_request(Method::GET, "/v1/chat/threads?q=topic", None)).await,
    )
    .await;
    assert_eq!(searched["total_count"].as_u64(), Some(2));
}

#[tokio::test]
async fn health_reports_system_memory_and_reserved_bytes() {
    let (app, _temp_dir) = test_api_app("health_reports_system_memory_and_reserved_bytes", false);
    let response = send_request(app, build_request(Method::GET, "/v1/health", None)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = read_json(response).await;
    let memory = &body["runtime"]["memory"];

    assert!(memory["reserved_bytes"].as_u64().is_some());
    match (
        memory["system_total_bytes"].as_u64(),
        memory["system_available_bytes"].as_u64(),
    ) {
        (Some(total), Some(available)) => {
            assert!(total > 0);
            assert!(available <= total);
        }
        (None, None) => {
            // The OS probe is unavailable on this platform; both stay null.
            assert!(memory["system_total_bytes"].is_null());
        }
        other => panic!("memory totals must be reported together: {other:?}"),
    }
}

#[tokio::test]
async fn admin_models_expose_names_tasks_sizes_and_image_input() {
    let (app, _temp_dir) = test_api_app(
        "admin_models_expose_names_tasks_sizes_and_image_input",
        false,
    );
    let response = send_request(app, build_request(Method::GET, "/v1/admin/models", None)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = read_json(response).await;
    let models = body["models"].as_array().expect("models array");
    let find = |variant: &str| {
        models
            .iter()
            .find(|model| model["variant"].as_str() == Some(variant))
            .unwrap_or_else(|| panic!("{variant} should be listed"))
    };
    let modalities = |model: &serde_json::Value| -> Vec<String> {
        model["modalities"]
            .as_array()
            .expect("modalities array")
            .iter()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect()
    };

    let vision_chat = find("Qwen3.5-4B");
    assert_eq!(vision_chat["task"].as_str(), Some("chat"));
    assert!(modalities(vision_chat).contains(&"image_input".to_string()));
    assert!(vision_chat["display_name"]
        .as_str()
        .is_some_and(|name| !name.is_empty() && name != "Qwen3.5-4B"));

    let text_chat = find("LFM2.5-1.2B-Instruct-GGUF");
    assert!(!modalities(text_chat).contains(&"image_input".to_string()));

    let tts = find("Kokoro-82M");
    assert_eq!(tts["task"].as_str(), Some("tts"));
    assert!(tts["estimated_size_bytes"]
        .as_u64()
        .is_some_and(|size| size > 0));
    assert!(tts["memory_required_gb"]
        .as_f64()
        .is_some_and(|gb| gb > 0.0));
}

#[tokio::test]
async fn admin_models_expose_voice_app_capabilities() {
    let (app, temp_dir) = test_api_app("admin_models_expose_voice_app_capabilities", false);

    let response =
        send_request(app, build_request(Method::GET, "/v1/admin/models", None)).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_json(response).await;
    let models = body["models"]
        .as_array()
        .expect("models should be an array");
    let find_model = |variant: &str| {
        models
            .iter()
            .find(|model| model["variant"].as_str() == Some(variant))
            .unwrap_or_else(|| panic!("{variant} should be listed"))
    };

    let tts_model = find_model("Kokoro-82M");
    assert_eq!(
        tts_model["route_capabilities"]["openai_audio_speech"].as_bool(),
        Some(true)
    );
    assert_eq!(
        tts_model["route_capabilities"]["voice_realtime_modular_tts"].as_bool(),
        Some(true)
    );
    assert!(tts_model["modalities"]
        .as_array()
        .expect("modalities should be an array")
        .iter()
        .any(|modality| modality.as_str() == Some("audio_output")));

    let fish_s2_model = find_model("FishAudio-S2-Pro");
    assert_eq!(fish_s2_model["enabled"].as_bool(), Some(true));
    assert_eq!(
        fish_s2_model["route_capabilities"]["openai_audio_speech"].as_bool(),
        Some(true)
    );
    assert_eq!(
        fish_s2_model["speech_capabilities"]["supports_reference_voice"].as_bool(),
        Some(true)
    );

    let aligner_model = find_model("Qwen3-ForcedAligner-0.6B");
    assert_eq!(
        aligner_model["route_capabilities"]["forced_alignment"].as_bool(),
        Some(true)
    );
    assert!(aligner_model["modalities"]
        .as_array()
        .expect("modalities should be an array")
        .iter()
        .any(|modality| modality.as_str() == Some("timestamps")));

    let diarization_model = find_model("diar_streaming_sortformer_4spk-v2.1");
    assert_eq!(
        diarization_model["route_capabilities"]["diarization_records"].as_bool(),
        Some(true)
    );
    assert_eq!(
        diarization_model["route_capabilities"]["speech_to_text_jobs"].as_bool(),
        Some(true)
    );

    let granite_model = find_model("Granite-Speech-4.1-2B-Plus");
    assert_eq!(
        granite_model["route_capabilities"]["diarization_records"].as_bool(),
        Some(false)
    );
    assert_eq!(
        granite_model["route_capabilities"]["speech_to_text_jobs"].as_bool(),
        Some(true)
    );
    let granite_modalities = granite_model["modalities"]
        .as_array()
        .expect("Granite modalities should be an array");
    assert!(granite_modalities
        .iter()
        .any(|modality| modality.as_str() == Some("speaker_labels")));
    assert!(!granite_modalities
        .iter()
        .any(|modality| modality.as_str() == Some("timestamps")));

    let voxtral_model = find_model("Voxtral-Mini-4B-Realtime-2602");
    assert_eq!(
        voxtral_model["route_capabilities"]["openai_audio_transcriptions"].as_bool(),
        Some(true)
    );
    assert_eq!(
        voxtral_model["route_capabilities"]["speech_to_text_jobs"].as_bool(),
        Some(true)
    );
    assert_eq!(
        voxtral_model["route_capabilities"]["speech_to_text_realtime"].as_bool(),
        Some(false)
    );
    assert_eq!(
        voxtral_model["route_capabilities"]["voice_realtime_modular_asr"].as_bool(),
        Some(false)
    );
    assert_eq!(
        voxtral_model["route_capabilities"]["voice_realtime_unified"].as_bool(),
        Some(false)
    );
    let voxtral_modalities = voxtral_model["modalities"]
        .as_array()
        .expect("Voxtral modalities should be an array");
    assert!(voxtral_modalities
        .iter()
        .any(|modality| modality.as_str() == Some("audio_input")));
    assert!(voxtral_modalities
        .iter()
        .any(|modality| modality.as_str() == Some("text_output")));
    assert!(!voxtral_modalities
        .iter()
        .any(|modality| modality.as_str() == Some("audio_output")));

    drop(temp_dir);
}

#[tokio::test]
async fn voice_session_rest_lifecycle_routes_work() {
    let (app, temp_dir) = test_api_app("voice_session_rest_lifecycle_routes_work", false);

    let create = send_request(
        app.clone(),
        build_request(Method::POST, "/v1/voice/sessions", Some("{}")),
    )
    .await;
    assert_eq!(create.status(), StatusCode::OK);
    let created = read_json(create).await;
    let session_id = created["session"]["id"]
        .as_str()
        .expect("created session id")
        .to_string();
    assert_eq!(created["turns"].as_array().expect("turns").len(), 0);

    let turns = send_request(
        app.clone(),
        build_request(
            Method::GET,
            &format!("/v1/voice/sessions/{session_id}/turns"),
            None,
        ),
    )
    .await;
    assert_eq!(turns.status(), StatusCode::OK);
    assert_eq!(
        read_json(turns).await.as_array().expect("turn list").len(),
        0
    );

    let patch = send_request(
        app.clone(),
        build_request(
            Method::PATCH,
            &format!("/v1/voice/sessions/{session_id}"),
            Some(r#"{"system_prompt":"Keep the exchange concise."}"#),
        ),
    )
    .await;
    assert_eq!(patch.status(), StatusCode::OK);
    let patched = read_json(patch).await;
    assert_eq!(
        patched["session"]["system_prompt"].as_str(),
        Some("Keep the exchange concise.")
    );

    let end = send_request(
        app.clone(),
        build_request(
            Method::POST,
            &format!("/v1/voice/sessions/{session_id}/end"),
            Some("{}"),
        ),
    )
    .await;
    assert_eq!(end.status(), StatusCode::OK);
    assert!(read_json(end).await["session"]["ended_at"].is_number());

    let export = send_request(
        app.clone(),
        build_request(
            Method::GET,
            &format!("/v1/voice/sessions/{session_id}/export?format=text"),
            None,
        ),
    )
    .await;
    assert_eq!(export.status(), StatusCode::OK);
    let body = axum::body::to_bytes(export.into_body(), usize::MAX)
        .await
        .expect("export body");
    assert!(std::str::from_utf8(&body)
        .expect("text export")
        .contains(&session_id));

    let delete = send_request(
        app.clone(),
        build_request(
            Method::DELETE,
            &format!("/v1/voice/sessions/{session_id}"),
            None,
        ),
    )
    .await;
    assert_eq!(delete.status(), StatusCode::OK);

    assert_route_status(
        app,
        Method::GET,
        &format!("/v1/voice/sessions/{session_id}"),
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    drop(temp_dir);
}

#[tokio::test]
async fn openapi_json_route_returns_valid_scaffold() {
    let (app, temp_dir) = test_api_app("openapi_json_route_returns_valid_scaffold", false);

    let response = send_request(app, build_request(Method::GET, "/openapi.json", None)).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE),
        Some(&HeaderValue::from_static("application/json"))
    );

    let json = read_json(response).await;
    assert!(json["openapi"].as_str().is_some());
    assert_eq!(json["info"]["title"], "Izwi API");
    assert_eq!(json["info"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(json["paths"].get("/livez").is_some());
    assert!(json["paths"].get("/readyz").is_some());

    drop(temp_dir);
}

#[tokio::test]
async fn scalar_docs_route_returns_html_before_ui_fallback() {
    let (app, temp_dir) = test_api_app("scalar_docs_route_returns_html", true);

    let response = send_request(app, build_request(Method::GET, "/docs", None)).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert!(response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/html")));

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should read");
    let html = String::from_utf8(body.to_vec()).expect("html should be utf-8");
    assert!(html.contains("/openapi.json"));
    assert!(html.contains("/docs/scalar.js"));
    assert!(html.contains("\"disabled\":true"));
    assert!(!html.contains("<html>ui</html>"));

    drop(temp_dir);
}

#[tokio::test]
async fn scalar_js_asset_route_returns_javascript() {
    let (app, temp_dir) = test_api_app("scalar_js_asset_route_returns_javascript", false);

    let response = send_request(app, build_request(Method::GET, "/docs/scalar.js", None)).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE),
        Some(&HeaderValue::from_static("application/javascript"))
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should read");
    assert!(!body.is_empty());

    drop(temp_dir);
}

#[tokio::test]
async fn dynamic_routes_cover_axum_upgrade_surface() {
    let (app, temp_dir) = test_api_app("dynamic_routes_cover_axum_upgrade_surface", false);

    let cases = [
        (
            Method::GET,
            "/v1/media/nested/example.png",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/media",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        ),
        (
            Method::DELETE,
            "/v1/media/nested/example.png",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::GET,
            "/v1/voice/sessions/missing-session/turns",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::PATCH,
            "/v1/voice/sessions/missing-session",
            Some(r#"{"system_prompt":"updated"}"#),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/voice/sessions/missing-session/end",
            Some("{}"),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::GET,
            "/v1/voice/sessions/missing-session/export",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::DELETE,
            "/v1/voice/sessions/missing-session",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::GET,
            "/v1/agent/sessions/missing-session",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/agent/sessions/missing-session/turns",
            Some(r#"{"input":"hello"}"#),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::GET,
            "/v1/responses/missing-response",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::DELETE,
            "/v1/responses/missing-response",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/responses/missing-response/cancel",
            Some("{}"),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/audio/align",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        ),
        (
            Method::GET,
            "/v1/responses/missing-response/input_items",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::DELETE,
            "/v1/studio/projects/missing-project/pronunciations/missing-pronunciation",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/studio/projects/missing-project/snapshots/missing-snapshot/restore",
            Some("{}"),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::PATCH,
            "/v1/studio/projects/missing-project/render-jobs/missing-job",
            Some(r#"{"status":"cancelled"}"#),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::GET,
            "/v1/studio/projects/missing-project/segments/missing-segment/audio",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/studio/projects/missing-project/segments/missing-segment/split",
            Some(r#"{"before_text":"before","after_text":"after"}"#),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/studio/projects/missing-project/segments/missing-segment/merge-next",
            Some("{}"),
            StatusCode::NOT_FOUND,
        ),
        (
            Method::POST,
            "/v1/studio/projects/missing-project/segments/missing-segment/render",
            Some("{}"),
            StatusCode::NOT_FOUND,
        ),
    ];

    for (method, path, body, expected_status) in cases {
        assert_route_status(app.clone(), method, path, body, expected_status).await;
    }

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/voice/realtime/ws",
        None,
        StatusCode::BAD_REQUEST,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/speech-to-text/realtime/ws",
        None,
        StatusCode::BAD_REQUEST,
    )
    .await;
    assert_route_status(
        app,
        Method::GET,
        "/v1/transcription/realtime/ws",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;

    drop(temp_dir);
}

#[tokio::test]
async fn readyz_returns_unavailable_when_lifecycle_is_draining() {
    let (state, temp_dir) = test_state("readyz_draining", false);
    state.lifecycle.mark_ready();
    state.lifecycle.mark_draining();
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state, &serve_config);

    assert_route_status(
        app,
        Method::GET,
        "/readyz",
        None,
        StatusCode::SERVICE_UNAVAILABLE,
    )
    .await;

    drop(temp_dir);
}

#[tokio::test]
async fn enterprise_policy_can_deny_requests_at_http_boundary() {
    let mut hooks = EnterpriseHooks::noop();
    hooks.policy = Arc::new(DenyAllPolicy);

    let (state, temp_dir) = test_state_with_hooks("enterprise_policy_denied", false, hooks);
    state.lifecycle.mark_ready();
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state, &serve_config);

    assert_route_status(app, Method::GET, "/livez", None, StatusCode::FORBIDDEN).await;

    drop(temp_dir);
}

#[tokio::test]
async fn cors_preflight_is_answered_before_authentication() {
    let mut hooks = EnterpriseHooks::noop();
    hooks.auth = Arc::new(DenyAllAuth);

    let (state, temp_dir) = test_state_with_hooks("cors_preflight_before_auth", false, hooks);
    state.lifecycle.mark_ready();
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state, &serve_config);

    let preflight = send_request(
        app.clone(),
        Request::builder()
            .method(Method::OPTIONS)
            .uri("/v1/chat/completions")
            .header(header::ORIGIN, "tauri://localhost")
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "authorization")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;
    assert_eq!(preflight.status(), StatusCode::OK);
    assert_eq!(
        preflight.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
        Some(&HeaderValue::from_static("tauri://localhost"))
    );

    // The actual request still goes through authentication, and its
    // rejection carries CORS headers the browser can read.
    let rejected = send_request(
        app,
        Request::builder()
            .method(Method::GET)
            .uri("/v1/admin/models")
            .header(header::ORIGIN, "tauri://localhost")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;
    assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        rejected.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
        Some(&HeaderValue::from_static("tauri://localhost"))
    );

    drop(temp_dir);
}

#[tokio::test]
async fn local_api_key_guards_api_routes_but_not_probes() {
    const KEY: &str = "router-test-api-key-123456";
    let hooks = crate::local_security::with_api_key(
        EnterpriseHooks::noop(),
        Some(crate::local_security::LocalApiKey::new(KEY).unwrap()),
    );
    let (state, temp_dir) = test_state_with_hooks("local_api_key_guard", false, hooks);
    state.lifecycle.mark_ready();
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state, &serve_config);

    assert_route_status(
        app.clone(),
        Method::GET,
        "/v1/admin/models",
        None,
        StatusCode::UNAUTHORIZED,
    )
    .await;
    assert_route_status(
        app.clone(),
        Method::DELETE,
        "/v1/admin/models/Qwen3-TTS-12Hz-0.6B-Base",
        None,
        StatusCode::UNAUTHORIZED,
    )
    .await;
    assert_route_status(app.clone(), Method::GET, "/livez", None, StatusCode::OK).await;

    let authorized = send_request(
        app,
        Request::builder()
            .method(Method::GET)
            .uri("/v1/admin/models")
            .header(header::AUTHORIZATION, format!("Bearer {KEY}"))
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;
    assert_eq!(authorized.status(), StatusCode::OK);

    drop(temp_dir);
}

#[tokio::test]
async fn local_router_rejects_cross_site_writes_and_rebound_hosts() {
    let (app, temp_dir) = test_api_app("local_router_csrf_guard", false);
    let load = |headers: &[(header::HeaderName, &str)]| {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri("/v1/admin/models/Qwen3-TTS-12Hz-0.6B-Base/unload");
        for (name, value) in headers {
            builder = builder.header(name.clone(), *value);
        }
        builder.body(Body::empty()).expect("request should build")
    };

    let cross_site = send_request(
        app.clone(),
        load(&[
            (header::HOST, "127.0.0.1:8080"),
            (header::ORIGIN, "https://attacker.example"),
        ]),
    )
    .await;
    assert_eq!(cross_site.status(), StatusCode::FORBIDDEN);

    let form = send_request(app.clone(), load(&[(header::CONTENT_TYPE, "text/plain")])).await;
    assert_eq!(form.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    let rebound = send_request(
        app.clone(),
        Request::builder()
            .uri("/v1/admin/models")
            .header(header::HOST, "attacker.example:8080")
            .body(Body::empty())
            .expect("request should build"),
    )
    .await;
    assert_eq!(rebound.status(), StatusCode::FORBIDDEN);

    let same_origin = send_request(
        app,
        load(&[
            (header::HOST, "localhost:8080"),
            (header::ORIGIN, "http://localhost:8080"),
            (header::CONTENT_TYPE, "application/json"),
        ]),
    )
    .await;
    assert_ne!(same_origin.status(), StatusCode::FORBIDDEN);
    assert_ne!(same_origin.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    drop(temp_dir);
}

/// DUR-F13: Studio store refusals are client errors (400), not storage
/// faults (500).
#[tokio::test]
async fn studio_segment_refusals_are_bad_requests() {
    let (state, _root) = test_state("studio_segment_refusals", false);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let created = send_request(
        app.clone(),
        build_request(
            Method::POST,
            "/v1/studio/projects",
            Some(
                &serde_json::json!({
                    "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "speaker": "Vivian",
                    "source_text": "One sentence. Another sentence."
                })
                .to_string(),
            ),
        ),
    )
    .await;
    assert_eq!(created.status(), StatusCode::OK);
    let project = read_json(created).await;
    let project_id = project["id"].as_str().unwrap().to_string();
    let segment_ids = project["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|segment| segment["id"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert!(!segment_ids.is_empty());

    let reorder = send_request(
        app.clone(),
        build_request(
            Method::PATCH,
            &format!("/v1/studio/projects/{project_id}/segments/reorder"),
            Some(r#"{"ordered_segment_ids":["not-a-segment"]}"#),
        ),
    )
    .await;
    assert_eq!(reorder.status(), StatusCode::BAD_REQUEST);
    let delete_all = send_request(
        app,
        build_request(
            Method::POST,
            &format!("/v1/studio/projects/{project_id}/segments/bulk-delete"),
            Some(&serde_json::json!({ "segment_ids": segment_ids }).to_string()),
        ),
    )
    .await;
    assert_eq!(delete_all.status(), StatusCode::BAD_REQUEST);
    assert!(read_json(delete_all).await["error"]["message"]
        .as_str()
        .unwrap()
        .contains("at least one segment"));
}

/// The server owns Studio rendering: clients queue and cancel, the
/// worker sets every other status, and export refuses audio that doesn't
/// match the script.
#[tokio::test]
async fn studio_render_jobs_are_server_owned_and_export_refuses_unrendered_audio() {
    let (state, _root) = test_state("studio_render_contract", false);
    state.lifecycle.mark_ready();
    let config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ..ServeRuntimeConfig::default()
    };
    let app = create_router(state.clone(), &config);
    let request = |method: Method, path: String, body: Option<String>| {
        let app = app.clone();
        async move { send_request(app, build_request(method, &path, body.as_deref())).await }
    };

    let created = request(
        Method::POST,
        "/v1/studio/projects".to_string(),
        Some(
            serde_json::json!({
                "model_id": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "speaker": "Vivian",
                "source_text": "One sentence. Another sentence."
            })
            .to_string(),
        ),
    )
    .await;
    assert_eq!(created.status(), StatusCode::OK);
    let project = read_json(created).await;
    let project_id = project["id"].as_str().unwrap().to_string();
    let segment_id = project["segments"][0]["id"].as_str().unwrap().to_string();
    assert_eq!(project["segments"][0]["render_state"], "not_rendered");

    // Reading metadata doesn't create it.
    let meta = request(
        Method::GET,
        format!("/v1/studio/projects/{project_id}/meta"),
        None,
    )
    .await;
    assert_eq!(meta.status(), StatusCode::OK);
    assert_eq!(read_json(meta).await["default_export_format"], "wav");
    assert!(state
        .studio_store
        .get_project_meta(project_id.clone())
        .await
        .unwrap()
        .is_none());

    let queued = request(
        Method::POST,
        format!("/v1/studio/projects/{project_id}/render-jobs"),
        Some("{}".to_string()),
    )
    .await;
    assert_eq!(queued.status(), StatusCode::OK);
    let job = read_json(queued).await;
    let job_id = job["id"].as_str().unwrap().to_string();
    assert_eq!(job["status"], "queued");

    let project = read_json(
        request(
            Method::GET,
            format!("/v1/studio/projects/{project_id}"),
            None,
        )
        .await,
    )
    .await;
    assert_eq!(project["segments"][0]["render_state"], "queued");
    assert_eq!(project["active_render_job"]["id"], job_id.as_str());

    let again = request(
        Method::POST,
        format!("/v1/studio/projects/{project_id}/render-jobs"),
        Some("{}".to_string()),
    )
    .await;
    assert_eq!(again.status(), StatusCode::BAD_REQUEST);

    let completed = request(
        Method::PATCH,
        format!("/v1/studio/projects/{project_id}/render-jobs/{job_id}"),
        Some(r#"{"status":"completed"}"#.to_string()),
    )
    .await;
    assert_eq!(completed.status(), StatusCode::BAD_REQUEST);
    let cancelled = request(
        Method::PATCH,
        format!("/v1/studio/projects/{project_id}/render-jobs/{job_id}"),
        Some(r#"{"status":"cancelled"}"#.to_string()),
    )
    .await;
    assert_eq!(cancelled.status(), StatusCode::OK);
    assert_eq!(read_json(cancelled).await["status"], "cancelled");

    let audio = request(
        Method::GET,
        format!("/v1/studio/projects/{project_id}/segments/{segment_id}/audio"),
        None,
    )
    .await;
    assert_eq!(audio.status(), StatusCode::NOT_FOUND);

    let export = request(
        Method::GET,
        format!("/v1/studio/projects/{project_id}/audio"),
        None,
    )
    .await;
    assert_eq!(export.status(), StatusCode::BAD_REQUEST);
    assert!(read_json(export).await["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not rendered: 1"));
    let long_pause = request(
        Method::GET,
        format!("/v1/studio/projects/{project_id}/audio?pause_ms=9000"),
        None,
    )
    .await;
    assert_eq!(long_pause.status(), StatusCode::BAD_REQUEST);

    let both = request(
        Method::PATCH,
        format!("/v1/studio/projects/{project_id}/segments/{segment_id}"),
        Some(r#"{"speaker":"Ryan","use_project_voice":true}"#.to_string()),
    )
    .await;
    assert_eq!(both.status(), StatusCode::BAD_REQUEST);
}

async fn send_request(mut app: Router, request: Request<Body>) -> Response {
    app.as_service::<Body>()
        .call(request)
        .await
        .expect("request should succeed")
}

fn test_ui_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be monotonic")
        .as_nanos();

    std::env::temp_dir().join(format!("izwi-router-{name}-{nanos}"))
}

async fn assert_route_status(
    app: Router,
    method: Method,
    path: &str,
    body: Option<&str>,
    expected_status: StatusCode,
) {
    let response = send_request(app, build_request(method, path, body)).await;
    assert_eq!(
        response.status(),
        expected_status,
        "unexpected status for {path}"
    );
}

async fn read_json(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should be readable");
    serde_json::from_slice(bytes.as_ref()).expect("response should be valid json")
}

fn build_request(method: Method, path: &str, body: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(path);
    let payload = body.unwrap_or_default();

    if body.is_some() {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
    }

    builder
        .body(if body.is_some() {
            Body::from(payload.to_owned())
        } else {
            Body::empty()
        })
        .expect("request should build")
}

fn build_request_with_idempotency(body: &str, key: &str) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/text-to-speech")
        .header(header::CONTENT_TYPE, "application/json")
        .header("idempotency-key", key)
        .body(Body::from(body.to_string()))
        .expect("request should build")
}

fn tiny_audio_json_body() -> String {
    tiny_audio_json_body_with_fields("")
}

fn media_upload_audio_json_body() -> String {
    let wav = AudioEncoder::new(22_050, 1)
        .encode(&[0.0, 0.2, -0.2, 0.0], AudioFormat::Wav)
        .expect("tiny wav should encode");
    serde_json::json!({
        "data_base64": base64::engine::general_purpose::STANDARD.encode(wav),
        "content_type": "audio/wav",
        "filename": "clip.wav",
        "namespace": "phase-1f"
    })
    .to_string()
}

fn tiny_audio_json_body_with_fields(extra_fields: &str) -> String {
    let wav = AudioEncoder::new(16_000, 1)
        .encode(&[0.0], AudioFormat::Wav)
        .expect("tiny wav should encode");
    let extra_fields = extra_fields.trim();
    let suffix = if extra_fields.is_empty() {
        String::new()
    } else {
        format!(",{extra_fields}")
    };
    format!(
        r#"{{"audio_base64":"{}"{}}}"#,
        base64::engine::general_purpose::STANDARD.encode(wav),
        suffix
    )
}

fn test_api_app(name: &str, ui_enabled: bool) -> (Router, TempDirGuard) {
    let (state, temp_dir) = test_state(name, ui_enabled);
    state.lifecycle.mark_ready();
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled,
        ..ServeRuntimeConfig::default()
    };
    (create_router(state, &serve_config), temp_dir)
}

fn test_state(name: &str, ui_enabled: bool) -> (AppState, TempDirGuard) {
    test_state_with_hooks(name, ui_enabled, EnterpriseHooks::noop())
}

fn test_state_with_hooks(
    name: &str,
    ui_enabled: bool,
    enterprise_hooks: EnterpriseHooks,
) -> (AppState, TempDirGuard) {
    let temp_dir = test_ui_dir(name);
    let ui_dir = temp_dir.join("ui");
    let models_dir = temp_dir.join("models");
    std::fs::create_dir_all(&models_dir).expect("models dir should be created");
    if ui_enabled {
        std::fs::create_dir_all(&ui_dir).expect("ui dir should be created");
        std::fs::write(ui_dir.join("index.html"), "<html>ui</html>")
            .expect("index should be written");
    }

    let db_path = temp_dir.join("izwi.sqlite3");
    let media_dir = temp_dir.join("media");

    let _guard = env_lock();
    std::env::set_var("IZWI_DB_PATH", &db_path);
    std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled,
        ui_dir,
        models_dir,
        ..ServeRuntimeConfig::default()
    };
    let runtime =
        with_suppressed_panic_hook(|| RuntimeService::new(serve_config.engine_config()))
            .expect("runtime should initialize");
    let state = AppState::with_enterprise_hooks(runtime, &serve_config, enterprise_hooks)
        .expect("app state should initialize");

    std::env::remove_var("IZWI_DB_PATH");
    std::env::remove_var("IZWI_MEDIA_DIR");

    (state, TempDirGuard(temp_dir))
}

fn with_suppressed_panic_hook<T>(f: impl FnOnce() -> T) -> T {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = f();
    std::panic::set_hook(default_hook);
    result
}

struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct FixedPrincipalAuth(Principal);

#[async_trait]
impl AuthProvider for FixedPrincipalAuth {
    async fn authenticate(&self, _request: &RequestEnvelope) -> HookResult<Principal> {
        Ok(self.0.clone())
    }
}

/// Authenticates each request as the tenant its `x-test-tenant` names.
struct HeaderTenantAuth;

#[async_trait]
impl AuthProvider for HeaderTenantAuth {
    async fn authenticate(&self, request: &RequestEnvelope) -> HookResult<Principal> {
        let tenant = request
            .headers
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case("x-test-tenant"))
            .map(|header| header.value.clone())
            .unwrap_or_else(|| "default".to_string());
        let mut principal = Principal::local_anonymous();
        principal.id = format!("{tenant}-user");
        principal.tenant_id = Some(tenant);
        Ok(principal)
    }
}

struct DenyAllAuth;

#[async_trait]
impl AuthProvider for DenyAllAuth {
    async fn authenticate(&self, _request: &RequestEnvelope) -> HookResult<Principal> {
        Err(HookError::Denied("test credential required".to_string()))
    }
}

struct DenyAllPolicy;

#[async_trait]
impl PolicyEngine for DenyAllPolicy {
    async fn authorize(
        &self,
        _request: &AuthorizationRequest,
    ) -> HookResult<AuthorizationDecision> {
        Ok(AuthorizationDecision::deny("blocked by test policy"))
    }
}
