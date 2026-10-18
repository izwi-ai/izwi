//! Chat request resolution shared by every entry path (the local server, the
//! gateway and the worker). The accelerator-free part (output budgets,
//! sampling profiles, capability checks) lives in `izwi_model_catalog::chat`
//! so the gateway resolves a request exactly as the engine does; this module
//! turns its result into engine generation parameters.

use crate::engine::GenerationParams;
pub use izwi_model_catalog::chat::{
    chat_supports_json_object, chat_supports_tools, resolve_chat_max_tokens, resolve_chat_sampling,
    ChatGenerationControls, ChatSamplingRequest, ResolvedChatSampling, CHAT_MAX_TOKENS_UNSET,
};

impl GenerationParams {
    /// Generation parameters with a chat request's resolved sampling, output
    /// budget and controls; every other field keeps its default. Every chat
    /// entry path (local server, gateway, worker) builds its request here.
    pub fn for_chat(
        sampling: ResolvedChatSampling,
        max_tokens: usize,
        controls: ChatGenerationControls,
    ) -> Self {
        Self {
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            top_k: sampling.top_k,
            repetition_penalty: sampling.repetition_penalty,
            presence_penalty: sampling.presence_penalty,
            seed: controls.seed,
            max_tokens,
            stop_sequences: controls.stop,
            logprobs: controls.logprobs.is_some(),
            top_logprobs: controls.logprobs.unwrap_or(0),
            constrain_json_object: controls.constrain_json_object,
            skip_special_tokens: !controls.keep_special_tokens,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling(
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        presence_penalty: f32,
    ) -> ResolvedChatSampling {
        ResolvedChatSampling {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
        }
    }

    #[test]
    fn generation_params_carry_every_chat_control() {
        let params = GenerationParams::for_chat(
            sampling(0.2, 0.3, 7, 1.2, -0.4),
            16,
            ChatGenerationControls {
                seed: Some(9),
                stop: vec!["END".into()],
                logprobs: Some(2),
                constrain_json_object: true,
                keep_special_tokens: true,
            },
        );
        assert_eq!(
            (params.temperature, params.top_p, params.top_k),
            (0.2, 0.3, 7)
        );
        assert_eq!(
            (params.repetition_penalty, params.presence_penalty),
            (1.2, -0.4)
        );
        assert_eq!(params.seed, Some(9));
        assert_eq!(params.stop_sequences, vec!["END".to_string()]);
        assert!(params.logprobs);
        assert_eq!(params.top_logprobs, 2);
        assert!(params.constrain_json_object);
        assert!(!params.skip_special_tokens);
        assert_eq!(params.max_tokens, 16);

        let plain = GenerationParams::for_chat(
            sampling(0.2, 0.3, 7, 1.2, -0.4),
            16,
            ChatGenerationControls::default(),
        );
        assert!(!plain.logprobs && plain.top_logprobs == 0 && !plain.constrain_json_object);
        assert!(plain.seed.is_none() && plain.stop_sequences.is_empty());
        assert!(plain.skip_special_tokens);
    }

    /// The catalog resolves profile-less models to the engine default
    /// without linking the engine; it must be this engine's default.
    #[test]
    fn catalog_engine_default_is_the_generation_default() {
        let defaults = GenerationParams::default();
        assert_eq!(
            ResolvedChatSampling::ENGINE_DEFAULT,
            sampling(
                defaults.temperature,
                defaults.top_p,
                defaults.top_k,
                defaults.repetition_penalty,
                defaults.presence_penalty,
            )
        );
    }
}
