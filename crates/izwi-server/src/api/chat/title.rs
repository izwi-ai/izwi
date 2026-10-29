//! Chat thread titles, written by the server after the first exchange.
//!
//! The title is generated with thinking off (`RuntimeService::auxiliary_generate`),
//! so a reasoning model's thoughts never become the title. A model that cannot
//! turn thinking off (LFM2.5 Thinking) hands the job to its non-thinking
//! sibling when that is loaded. Otherwise, or when generation fails, the title
//! is the first message's opening words. A title never loads a model, and never
//! replaces a title the user gave the thread.

use axum::{
    extract::{Extension, Path, State},
    Json,
};
use serde::Deserialize;
use tracing::warn;

use izwi_core::chat_reasoning::split_reasoning;
use izwi_core::{
    ChatMessage, ChatRole, GenerationParams, ModelResidencyState, ModelVariant, WorkloadClass,
};

use super::handlers::{get_thread_or_not_found, map_store_error};
use crate::api::request_context::RequestContext;
use crate::app::chat::parse_chat_model;
use crate::chat_store::{ChatThreadMessage, ChatThreadSummary, DEFAULT_THREAD_TITLE};
use crate::error::{ApiError, ApiJson};
use crate::state::AppState;

const TITLE_MAX_TOKENS: usize = 32;
const TITLE_MAX_CHARS: usize = 80;
/// How much of the exchange the title model reads.
const USER_CONTEXT_CHARS: usize = 800;
const ANSWER_CONTEXT_CHARS: usize = 700;

const TITLE_SYSTEM_PROMPT: &str = "You write titles for chat conversations. Reply with only the title: at most 8 words naming the topic, with no quotes and no trailing punctuation.";

#[derive(Debug, Default, Deserialize)]
pub struct GenerateThreadTitleRequest {
    /// The model that answered. Defaults to the first reply's model.
    #[serde(default)]
    pub model: Option<String>,
}

/// `POST /v1/chat/threads/{thread_id}/title`: title a thread still called
/// "New chat" from its first exchange, and return the thread.
pub async fn generate_thread_title(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(req): ApiJson<GenerateThreadTitleRequest>,
) -> Result<Json<ChatThreadSummary>, ApiError> {
    let thread = get_thread_or_not_found(&state, &thread_id).await?;
    if thread.title != DEFAULT_THREAD_TITLE {
        return Ok(Json(thread));
    }
    // A stopped turn may still be committing: read the thread after it has,
    // without holding the lock while the title generates.
    let messages = {
        let _turn = state.chat_store.lock_turn(&thread_id).await;
        state
            .chat_store
            .list_messages(thread_id.clone())
            .await
            .map_err(map_store_error)?
    };
    let Some((user, reply)) = first_exchange(&messages) else {
        return Ok(Json(thread));
    };
    let variant = match req.model.as_deref() {
        Some(model) => Some(parse_chat_model(model)?),
        // A stored id may name a model that no longer chats (LFM3-Audio, or
        // one removed from the catalog): the title then falls back.
        None => reply
            .and_then(|reply| reply.model_id.as_deref())
            .or(thread.model_id.as_deref())
            .and_then(|model| parse_chat_model(model).ok()),
    };
    let answer = reply.map(|reply| stored_answer(&reply.content));

    let generated = match variant {
        Some(variant) => {
            generate_title(
                &state,
                variant,
                &user.content,
                answer.as_deref().unwrap_or(""),
                &ctx.correlation_id,
            )
            .await
        }
        None => None,
    };
    let Some(title) = generated.or_else(|| fallback_title(&user.content)) else {
        return Ok(Json(thread));
    };
    let thread = state
        .chat_store
        .set_generated_title(thread_id, title)
        .await
        .map_err(map_store_error)?;
    thread
        .map(Json)
        .ok_or_else(|| ApiError::not_found("Thread not found"))
}

/// The first user message and the reply that follows it, if any.
fn first_exchange(
    messages: &[ChatThreadMessage],
) -> Option<(&ChatThreadMessage, Option<&ChatThreadMessage>)> {
    let user_index = messages.iter().position(|message| message.role == "user")?;
    let reply = messages[user_index + 1..]
        .iter()
        .find(|message| message.role == "assistant");
    Some((&messages[user_index], reply))
}

/// A stored reply's answer, without its reasoning. Replies are stored raw: a
/// model that writes its own opener stores `<think>r</think>a`, and one whose
/// prompt opened the block stores `r</think>a`.
fn stored_answer(content: &str) -> String {
    let opened_in_prompt = !content.contains("<think>") && content.contains("</think>");
    split_reasoning(content, opened_in_prompt)
        .1
        .trim()
        .to_string()
}

async fn generate_title(
    state: &AppState,
    variant: ModelVariant,
    user: &str,
    answer: &str,
    correlation_id: &str,
) -> Option<String> {
    let variant = title_model(state, variant).await?;
    let permit = state
        .acquire_workload_permit(WorkloadClass::Interactive)
        .await;
    let result = state
        .runtime
        .auxiliary_generate(
            variant,
            title_messages(user, answer),
            GenerationParams {
                max_tokens: TITLE_MAX_TOKENS,
                temperature: 0.2,
                top_p: 0.9,
                ..Default::default()
            },
            Some(correlation_id),
            permit.runtime_context(),
        )
        .await;
    match result {
        Ok(generation) => normalize_generated_title(&generation.text),
        Err(err) => {
            warn!(model = %variant, error = %err, "chat title generation failed");
            None
        }
    }
}

/// The model that writes the title: the one that answered, or for a model
/// that always reasons, its non-thinking sibling. Only a loaded model is used.
async fn title_model(state: &AppState, variant: ModelVariant) -> Option<ModelVariant> {
    let always_reasons = variant
        .chat_capabilities()
        .is_some_and(|caps| caps.supports_thinking && !caps.thinking_toggleable);
    let candidate = if always_reasons {
        variant.non_thinking_sibling()?
    } else {
        variant
    };
    let loaded = state
        .runtime
        .model_lifecycle_snapshot(candidate)
        .await
        .is_some_and(|snapshot| snapshot.residency_state == ModelResidencyState::Ready);
    loaded.then_some(candidate)
}

fn title_messages(user: &str, answer: &str) -> Vec<ChatMessage> {
    let mut exchange = format!("User: {}", clip(user, USER_CONTEXT_CHARS));
    if !answer.is_empty() {
        exchange.push_str(&format!(
            "\nAssistant: {}",
            clip(answer, ANSWER_CONTEXT_CHARS)
        ));
    }
    vec![
        ChatMessage {
            role: ChatRole::System,
            content: TITLE_SYSTEM_PROMPT.to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: format!("Write a title for this conversation.\n\n{exchange}"),
        },
    ]
}

/// One line of plain text: no "Title:" label, quotes, markdown or trailing
/// punctuation. `None` when nothing usable is left.
fn normalize_generated_title(raw: &str) -> Option<String> {
    let line = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("```"))?;
    let mut title = collapse_whitespace(&line.replace("**", "").replace('`', ""));
    if let Some(rest) = strip_prefix_ignore_case(&title, "title") {
        let rest = rest.trim_start();
        if let Some(rest) = rest.strip_prefix(':').or_else(|| rest.strip_prefix('-')) {
            title = rest.trim().to_string();
        }
    }
    let title = title
        .trim_start_matches(['#', '*', '"', '\'', '“', '‘'])
        .trim_end_matches(['"', '\'', '”', '’', '*', '.', '!', ':', ';', ','])
        .trim();
    let lower = title.to_ascii_lowercase();
    if title.is_empty() || lower.starts_with("user:") || lower.starts_with("assistant:") {
        return None;
    }
    Some(clip(title, TITLE_MAX_CHARS))
}

/// The first message's opening words, when it has any (an image-only
/// message keeps "New chat").
fn fallback_title(user: &str) -> Option<String> {
    let text = collapse_whitespace(user);
    (!text.is_empty()).then(|| clip(&text, TITLE_MAX_CHARS))
}

fn strip_prefix_ignore_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &text[prefix.len()..])
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// At most `max_chars`, cut at a word boundary with an ellipsis when cut.
fn clip(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    let cut = match cut.rfind(' ') {
        Some(space) if space > 0 => &cut[..space],
        _ => cut.as_str(),
    };
    format!("{}…", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> ChatThreadMessage {
        ChatThreadMessage {
            id: format!("{role}-{content}"),
            thread_id: "thread".to_string(),
            role: role.to_string(),
            content: content.to_string(),
            content_parts: None,
            created_at: 0,
            tokens_generated: None,
            generation_time_ms: None,
            model_id: None,
            finish_reason: None,
            reasoning_ms: None,
        }
    }

    #[test]
    fn titles_are_one_clean_line() {
        for (raw, title) in [
            ("Winter houseplant care", "Winter houseplant care"),
            ("Title: Winter houseplant care.", "Winter houseplant care"),
            (
                "TITLE - \"Winter houseplant care\"",
                "Winter houseplant care",
            ),
            (
                "**Winter houseplant care**\n\nExtra line",
                "Winter houseplant care",
            ),
            ("```\nWinter houseplant care\n```", "Winter houseplant care"),
            ("“Naming the planets”", "Naming the planets"),
        ] {
            assert_eq!(
                normalize_generated_title(raw).as_deref(),
                Some(title),
                "{raw}"
            );
        }
        for raw in [
            "",
            "  \n ",
            "```\n```",
            "User: hello",
            "\"\"",
            "Assistant: hi",
        ] {
            assert_eq!(normalize_generated_title(raw), None, "{raw}");
        }
    }

    #[test]
    fn long_titles_are_cut_at_a_word() {
        let title = normalize_generated_title(&"word ".repeat(40)).expect("title");
        assert!(title.chars().count() <= TITLE_MAX_CHARS);
        assert!(title.ends_with("word…"));
    }

    #[test]
    fn the_title_model_reads_the_answer_not_the_reasoning() {
        assert_eq!(
            stored_answer("Let me think about plants</think>\n\nWater less in winter."),
            "Water less in winter."
        );
        assert_eq!(
            stored_answer("<think>Okay</think>Water less in winter."),
            "Water less in winter."
        );
        assert_eq!(
            stored_answer("Water less in winter."),
            "Water less in winter."
        );
        // A reply cut off while reasoning has no answer.
        assert_eq!(stored_answer("<think>Okay, the user"), "");
    }

    #[test]
    fn the_first_exchange_is_the_first_user_message_and_its_reply() {
        let messages = [
            message("system", "Be brief"),
            message("user", "first"),
            message("assistant", "reply"),
            message("user", "second"),
        ];
        let (user, reply) = first_exchange(&messages).expect("exchange");
        assert_eq!(user.content, "first");
        assert_eq!(reply.map(|reply| reply.content.as_str()), Some("reply"));

        let unanswered = [message("user", "only")];
        let (user, reply) = first_exchange(&unanswered).expect("exchange");
        assert_eq!(user.content, "only");
        assert!(reply.is_none());
        assert!(first_exchange(&[]).is_none());
    }

    #[test]
    fn the_prompt_carries_a_clipped_exchange() {
        let messages = title_messages(&"u".repeat(5_000), "Water less in winter.");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[0].content, TITLE_SYSTEM_PROMPT);
        assert!(messages[1].content.len() < 1_600);
        assert!(messages[1]
            .content
            .ends_with("\nAssistant: Water less in winter."));
        assert!(!title_messages("Hi", "")
            .pop()
            .unwrap()
            .content
            .contains("Assistant:"));
    }

    #[test]
    fn the_fallback_is_the_first_message_opening_words() {
        assert_eq!(
            fallback_title("  Name one\nplanet  ").as_deref(),
            Some("Name one planet")
        );
        assert_eq!(fallback_title(" \n "), None);
        let long = fallback_title(&"plant ".repeat(30)).expect("title");
        assert!(long.chars().count() <= TITLE_MAX_CHARS);
    }
}
