//! Transcript summaries for transcription, speaker-attributed and diarization
//! records: one generation path, so the three record kinds cannot drift.
//!
//! The summary model runs with thinking off and only its answer is kept
//! (`RuntimeService::auxiliary_generate`). A summary either lands as text, or
//! fails with a reason: it never stays pending once its task has ended, and a
//! queue that cannot take it is a failure too.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use izwi_core::{
    ChatMessage, ChatRole, GenerationParams, ModelVariant, RuntimeRequestContext, RuntimeService,
    WorkloadClass,
};

use crate::state::AppState;

/// The model that summarizes when a record names none.
pub(crate) const DEFAULT_SUMMARY_MODEL: &str = "Qwen3.5-4B";

const SUMMARY_MAX_TOKENS: usize = 384;
const MAX_SUMMARY_ERROR_CHARS: usize = 320;

const TRANSCRIPT_SYSTEM_PROMPT: &str = "You summarize transcripts for fast review. Return only the final summary text, with no markdown, bullet markers or code fences. Keep the summary concise while covering key topics, decisions, action items, and unresolved questions when present.";

const DIARIZED_SYSTEM_PROMPT: &str = "You summarize diarized transcripts for fast review. Return only the final summary text, with no markdown, bullet markers or code fences. Keep it concise while preserving major speaker contributions, decisions, action items, and unresolved questions.";

/// What is being summarized: the prompt differs, nothing else does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SummaryKind {
    Transcript,
    DiarizedTranscript,
}

#[derive(Debug, Clone)]
pub(crate) struct SummaryJob {
    pub kind: SummaryKind,
    pub variant: ModelVariant,
    pub transcript: String,
    pub correlation_id: Option<String>,
    pub timeout_secs: u64,
}

/// Generate a summary in the background and hand the outcome to `persist`.
/// `persist` runs exactly once, with the summary text or a failure reason.
pub(crate) fn spawn_summary_generation<P, F>(state: AppState, job: SummaryJob, persist: P)
where
    P: FnOnce(Result<String, String>) -> F + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    tokio::spawn(async move {
        let result = match state
            .acquire_owned_workload_permit(WorkloadClass::Background)
            .await
        {
            Ok(permit) => {
                let result =
                    generate_summary(state.runtime.clone(), &job, permit.runtime_context()).await;
                drop(permit);
                result
            }
            Err(_) => Err("Summary generation unavailable: the request queue closed.".to_string()),
        };
        persist(result.map_err(|err| truncate_summary_error(&err))).await;
    });
}

async fn generate_summary(
    runtime: Arc<RuntimeService>,
    job: &SummaryJob,
    runtime_context: RuntimeRequestContext,
) -> Result<String, String> {
    if job.transcript.trim().is_empty() {
        return Err("Summary generation failed: the transcript is empty.".to_string());
    }
    let timeout_secs = job.timeout_secs.max(1);
    let generation = tokio::time::timeout(
        Duration::from_secs(timeout_secs),
        runtime.auxiliary_generate(
            job.variant,
            summary_messages(job.kind, &job.transcript),
            summary_params(),
            job.correlation_id.as_deref(),
            runtime_context,
        ),
    )
    .await
    .map_err(|_| format!("Summary generation timed out after {timeout_secs} seconds."))?
    .map_err(|err| format!("Summary generation failed: {err}"))?;
    clean_summary(&generation.text)
        .ok_or_else(|| "Summary generation returned empty text.".to_string())
}

fn summary_params() -> GenerationParams {
    GenerationParams {
        max_tokens: SUMMARY_MAX_TOKENS,
        temperature: 0.2,
        top_p: 0.9,
        ..Default::default()
    }
}

pub(crate) fn summary_messages(kind: SummaryKind, transcript: &str) -> Vec<ChatMessage> {
    let (system, instruction) = match kind {
        SummaryKind::Transcript => (
            TRANSCRIPT_SYSTEM_PROMPT,
            "Summarize the following transcript.",
        ),
        SummaryKind::DiarizedTranscript => (
            DIARIZED_SYSTEM_PROMPT,
            "Summarize the following diarized transcript.",
        ),
    };
    vec![
        ChatMessage {
            role: ChatRole::System,
            content: system.to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: format!("{instruction}\n\nTranscript:\n{transcript}"),
        },
    ]
}

/// One paragraph of plain text: code fences dropped, whitespace collapsed.
/// Reasoning was already split off by `auxiliary_generate`.
pub(crate) fn clean_summary(raw: &str) -> Option<String> {
    let text = raw
        .lines()
        .filter(|line| !line.trim_start().starts_with("```"))
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ");
    (!text.is_empty()).then_some(text)
}

pub(crate) fn truncate_summary_error(raw: &str) -> String {
    raw.chars().take(MAX_SUMMARY_ERROR_CHARS).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summaries_are_one_plain_paragraph() {
        assert_eq!(
            clean_summary("```text\nAaron is worried\n  about the exam.\n```").as_deref(),
            Some("Aaron is worried about the exam.")
        );
        assert_eq!(clean_summary("```\n```"), None);
        assert_eq!(clean_summary("   \n "), None);
    }

    #[test]
    fn both_kinds_send_the_transcript_after_their_own_prompt() {
        let transcript = "SPEAKER_00: hello";
        for (kind, system) in [
            (SummaryKind::Transcript, TRANSCRIPT_SYSTEM_PROMPT),
            (SummaryKind::DiarizedTranscript, DIARIZED_SYSTEM_PROMPT),
        ] {
            let messages = summary_messages(kind, transcript);
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[0].role, ChatRole::System);
            assert_eq!(messages[0].content, system);
            assert_eq!(messages[1].role, ChatRole::User);
            assert!(messages[1].content.ends_with(transcript));
        }
    }

    #[test]
    fn summary_errors_are_bounded() {
        let long = "x".repeat(1_000);
        assert_eq!(truncate_summary_error(&long).len(), MAX_SUMMARY_ERROR_CHARS);
    }
}
