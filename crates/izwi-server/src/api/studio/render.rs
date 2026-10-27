//! The Studio render worker. It renders queued paragraphs one at a time,
//! in the order they were queued, whether or not anyone has Studio open.

use std::time::Duration;

use izwi_core::{parse_tts_model_variant, ModelStatus};
use izwi_hooks::Principal;
use tracing::{info, warn};

use super::handlers::{release_studio_audio, validate_project_voice_state};
use crate::api::request_context::RequestContext;
use crate::api::speech_history::{synthesize_record, CreateSpeechHistoryRecordRequest};
use crate::speech_history_store::SpeechRouteKind;
use crate::state::AppState;
use crate::studio_project_store::{segment_render_input, StudioRenderClaim};

/// How long an idle worker waits before looking at the queue again when
/// nothing wakes it.
const IDLE_RECHECK: Duration = Duration::from_secs(30);

/// Start the worker. It stops between paragraphs once shutdown begins.
pub fn spawn_render_worker(state: AppState) -> tokio::task::JoinHandle<()> {
    tokio::spawn(run(state))
}

async fn run(state: AppState) {
    match state.studio_store.requeue_interrupted_renders().await {
        Ok(0) => {}
        Ok(count) => info!(count, "Requeued Studio renders interrupted by a restart"),
        Err(err) => warn!(error = %err, "Failed to requeue interrupted Studio renders"),
    }
    if let Err(err) = state.studio_store.stamp_unfingerprinted_renders().await {
        warn!(error = %err, "Failed to stamp Studio render fingerprints");
    }

    let mut shutdown = state.stream_shutdown.subscribe();
    while !*shutdown.borrow() {
        match state.studio_store.claim_next_render().await {
            Ok(Some(claim)) => {
                render_claim(&state, &claim).await;
                continue;
            }
            Ok(None) => {}
            Err(err) => warn!(error = %err, "Failed to claim a Studio render"),
        }
        tokio::select! {
            _ = state.studio_render_wake.notified() => {}
            _ = shutdown.changed() => {}
            _ = tokio::time::sleep(IDLE_RECHECK) => {}
        }
    }
}

/// Why a paragraph didn't render, and whether the rest of its job would
/// fail the same way.
enum RenderFailure {
    Paragraph(String),
    Job(String),
}

async fn render_claim(state: &AppState, claim: &StudioRenderClaim) {
    let failure = match render_segment(state, claim).await {
        Ok(()) => return,
        Err(failure) => failure,
    };
    let (message, whole_job) = match failure {
        RenderFailure::Paragraph(message) => (message, false),
        RenderFailure::Job(message) => (message, true),
    };
    if let Err(err) = state.studio_store.fail_render(claim, &message).await {
        warn!(error = %err, "Failed to record a Studio render failure");
    }
    if whole_job {
        if let Err(err) = state.studio_store.fail_queued_in_job(claim, &message).await {
            warn!(error = %err, "Failed to record a Studio job failure");
        }
    }
}

async fn render_segment(state: &AppState, claim: &StudioRenderClaim) -> Result<(), RenderFailure> {
    let store_failure = |err: anyhow::Error| RenderFailure::Paragraph(err.to_string());
    let Some(project) = state
        .studio_store
        .get_project(claim.project_id.clone())
        .await
        .map_err(store_failure)?
    else {
        return Ok(());
    };
    let Some(segment) = project
        .segments
        .iter()
        .find(|segment| segment.id == claim.segment_id)
    else {
        return Ok(());
    };
    let pronunciations = state
        .studio_store
        .list_project_pronunciations(claim.project_id.clone())
        .await
        .map_err(store_failure)?;
    let (text, settings, fingerprint) = segment_render_input(&project, segment, &pronunciations);

    let model_id = settings.model_id.clone().ok_or_else(|| {
        RenderFailure::Job("This project has no model. Pick a voice in its settings.".to_string())
    })?;
    let variant = parse_tts_model_variant(&model_id)
        .map_err(|err| RenderFailure::Job(format!("Unsupported TTS model: {err}")))?;
    let available = state
        .runtime
        .model_manager()
        .get_model_info(variant)
        .await
        .is_some_and(|info| {
            info.enabled
                && matches!(
                    info.status,
                    ModelStatus::Downloaded | ModelStatus::Loading | ModelStatus::Ready
                )
        });
    if !available {
        return Err(RenderFailure::Job(format!(
            "{} isn't downloaded. Download it in Models, then render again.",
            variant.display_name()
        )));
    }
    validate_project_voice_state(
        state,
        &model_id,
        settings.voice_mode,
        settings.speaker.as_deref(),
        settings.saved_voice_id.as_deref(),
    )
    .await
    .map_err(|err| RenderFailure::Paragraph(err.message))?;

    let request = CreateSpeechHistoryRecordRequest {
        model_id: Some(model_id),
        text: Some(text),
        speaker: settings.speaker,
        language: None,
        voice_description: None,
        reference_audio: None,
        reference_text: None,
        reference_language: None,
        saved_voice_id: settings.saved_voice_id,
        temperature: None,
        // A model without speed control rejects a speed: the project may
        // keep one from an earlier model.
        speed: settings
            .speed
            .filter(|_| {
                variant
                    .speech_capabilities()
                    .is_some_and(|caps| caps.supports_speed_control)
            })
            .map(|value| value as f32),
        max_tokens: Some(0),
        max_output_tokens: None,
        top_k: None,
        stream: Some(false),
        sampling: Default::default(),
    };
    let ctx = RequestContext::new(
        uuid::Uuid::new_v4().to_string(),
        Principal::local_anonymous(),
    );
    let record = synthesize_record(state, &ctx, request, SpeechRouteKind::Studio)
        .await
        .map_err(|err| RenderFailure::Paragraph(err.message))?;

    match state
        .studio_store
        .complete_render(claim, record.id.clone(), fingerprint)
        .await
    {
        Ok(released) => {
            release_studio_audio(state, released).await;
            Ok(())
        }
        Err(err) => {
            release_studio_audio(state, vec![record.id]).await;
            Err(store_failure(err))
        }
    }
}
