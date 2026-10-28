//! Chat completions through LFM3-Audio: a conversation whose user turns may
//! carry audio clips, answered in text by the model's speech-to-speech reply
//! row (the Conversation workload). The row prepares the prompt, so the
//! chat budget rule applies once its length, audio included, is known.

use izwi_model_catalog::{CHAT_AUDIO_PLACEHOLDER, CHAT_MAX_AUDIO_CLIPS};

use crate::engine::{GenerationParams, ResourceVector, TaskType};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMediaKind, ChatMessage, ChatRequestConfig, ChatRole};
use crate::runtime::request::ChatRuntimeRequest;
use crate::runtime::service::{
    retained_chat_preparation_input_bytes, AdmittedEngineRequest, RuntimeService,
};
use crate::runtime::types::RuntimeRequestContext;

impl RuntimeService {
    pub(super) async fn build_lfm3_audio_chat_request(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        mut params: GenerationParams,
        chat_config: ChatRequestConfig,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        streaming: bool,
        max_tokens_ceiling: Option<usize>,
    ) -> Result<AdmittedEngineRequest> {
        if max_tokens_ceiling.is_some() {
            return Err(Error::InvalidInput(format!(
                "{variant} answers chat only on a local server"
            )));
        }
        validate_audio_chat(&messages, &chat_config)?;
        let correlation_id = correlation_id.map(ToOwned::to_owned);
        let input_bytes = retained_chat_preparation_input_bytes(
            &messages,
            messages.capacity(),
            &chat_config,
            &params,
            correlation_id.as_ref(),
        )?;
        self.prepare_engine_request_blocking(
            variant,
            TaskType::SpeechToSpeech,
            streaming,
            runtime_context,
            input_bytes,
            ResourceVector::zero(),
            move |_registry| {
                params.max_tokens = params.max_tokens.max(1);
                Ok(ChatRuntimeRequest::from_messages(
                    variant,
                    messages,
                    params,
                    chat_config,
                    Vec::new(),
                    correlation_id,
                    runtime_context,
                )?
                .into_lfm3_audio_engine_request())
            },
        )
        .await
    }
}

/// An LFM3-Audio chat hears audio clips only, at most
/// [`CHAT_MAX_AUDIO_CLIPS`], each where a user turn has its placeholder.
fn validate_audio_chat(messages: &[ChatMessage], config: &ChatRequestConfig) -> Result<()> {
    if messages.is_empty() {
        return Err(Error::InvalidInput(
            "Chat request missing messages".to_string(),
        ));
    }
    if config
        .media_inputs
        .iter()
        .any(|input| input.kind != ChatMediaKind::Audio)
    {
        return Err(Error::InvalidInput(
            "LFM3-Audio chat hears audio clips, not images or videos".to_string(),
        ));
    }
    let clips = config.media_inputs.len();
    if clips > CHAT_MAX_AUDIO_CLIPS {
        return Err(Error::InvalidInput(format!(
            "A chat request can carry at most {CHAT_MAX_AUDIO_CLIPS} audio clips, not {clips}"
        )));
    }
    let mut placeholders = 0usize;
    for message in messages {
        let count = message.content.matches(CHAT_AUDIO_PLACEHOLDER).count();
        if count > 0 && message.role != ChatRole::User {
            return Err(Error::InvalidInput(format!(
                "Only user messages can carry audio; a {} message has {CHAT_AUDIO_PLACEHOLDER}",
                message.role.as_prompt_role()
            )));
        }
        placeholders += count;
    }
    if placeholders != clips {
        return Err(Error::InvalidInput(format!(
            "The messages place {placeholders} audio clips ({CHAT_AUDIO_PLACEHOLDER}) \
             but the request carries {clips}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::shared::chat::ChatMediaInput;

    fn message(role: ChatRole, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
        }
    }

    fn clips(count: usize) -> ChatRequestConfig {
        ChatRequestConfig {
            media_inputs: (0..count)
                .map(|_| ChatMediaInput {
                    kind: ChatMediaKind::Audio,
                    source: "data:audio/wav;base64,AAAA".to_string(),
                })
                .collect(),
            ..ChatRequestConfig::default()
        }
    }

    #[test]
    fn audio_chat_places_every_clip_in_a_user_turn() {
        let two = format!("{CHAT_AUDIO_PLACEHOLDER} and {CHAT_AUDIO_PLACEHOLDER}");
        let messages = [
            message(ChatRole::System, "Be brief."),
            message(ChatRole::User, &two),
        ];
        validate_audio_chat(&messages, &clips(2)).unwrap();
        validate_audio_chat(&[message(ChatRole::User, "hi")], &clips(0)).unwrap();

        let error = validate_audio_chat(&messages, &clips(1)).unwrap_err();
        assert!(error.to_string().contains("place 2 audio clips"), "{error}");
        let error = validate_audio_chat(
            &[message(ChatRole::Assistant, CHAT_AUDIO_PLACEHOLDER)],
            &clips(1),
        )
        .unwrap_err();
        assert!(error.to_string().contains("Only user messages"), "{error}");
    }

    #[test]
    fn audio_chat_refuses_other_media_and_too_many_clips() {
        let mut config = clips(1);
        config.media_inputs[0].kind = ChatMediaKind::Image;
        let error =
            validate_audio_chat(&[message(ChatRole::User, CHAT_AUDIO_PLACEHOLDER)], &config)
                .unwrap_err();
        assert!(error.to_string().contains("not images"), "{error}");

        let many = CHAT_AUDIO_PLACEHOLDER.repeat(CHAT_MAX_AUDIO_CLIPS + 1);
        let error = validate_audio_chat(
            &[message(ChatRole::User, &many)],
            &clips(CHAT_MAX_AUDIO_CLIPS + 1),
        )
        .unwrap_err();
        assert!(error.to_string().contains("at most"), "{error}");
    }
}
