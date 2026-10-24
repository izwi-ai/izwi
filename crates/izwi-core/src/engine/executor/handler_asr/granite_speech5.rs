//! Granite Speech 5.0 ASR: chunk loop and truncation recovery. A chunk that
//! stopped on its decode budget is re-decoded through the 5.0 model, with a
//! smaller budget when it looped and a larger one when it was cut off. The
//! loop detection and trimming are the text helpers the 4.1 route uses.

use super::granite::{
    granite_chunk_loop_signal, granite_chunk_truncated, granite_decode_stop_reason,
    granite_truncation_retry_max_tokens, trim_granite_looping_chunk_text,
    with_granite_loop_recovery_diagnostics, with_granite_truncation_retry_diagnostics,
};
use super::*;
use crate::models::architectures::granite_speech5::asr::granite_speech5_auto_max_tokens_for_duration;
use crate::models::registry::NativeAsrTranscription;

/// Budget for the re-decode of a looping chunk: three quarters of the
/// original, no more than the chunk's automatic budget, and at least 24
/// tokens (or one fewer than the original).
fn loop_recovery_max_tokens(chunk_seconds: f32, original_max_tokens: usize) -> usize {
    if original_max_tokens <= 1 {
        return original_max_tokens;
    }
    let auto_for_chunk = granite_speech5_auto_max_tokens_for_duration(chunk_seconds);
    (original_max_tokens.saturating_mul(3) / 4)
        .min(auto_for_chunk)
        .max(24.min(original_max_tokens - 1))
        .min(original_max_tokens - 1)
        .max(1)
}

/// What a chunk's recovery may spend: the request's automatic-budget flag
/// and ceiling, and the invocation context the chunk decodes in.
#[derive(Debug, Clone, Copy)]
pub(super) struct ChunkRecoveryLimits {
    pub(super) auto_max_tokens: bool,
    pub(super) request_ceiling: usize,
    /// The effective context and the audio token rate, when both are known.
    pub(super) invocation_context: Option<(usize, f32)>,
}

/// Recover a chunk that stopped on its decode budget. `redecode` runs the
/// same chunk again with the given options.
pub(super) fn recover_granite_speech5_chunk(
    mut details: NativeAsrTranscription,
    chunk_seconds: f32,
    base_options: &NativeAsrGenerationOptions,
    limits: ChunkRecoveryLimits,
    mut redecode: impl FnMut(NativeAsrGenerationOptions) -> Result<NativeAsrTranscription>,
) -> Result<NativeAsrTranscription> {
    let original_max_tokens = base_options.max_new_tokens;
    let Some(original_loop) =
        granite_chunk_loop_signal(&details.text, details.diagnostics.as_ref())
    else {
        return retry_truncated_chunk(details, chunk_seconds, base_options, limits, redecode);
    };
    let retry_max_tokens = loop_recovery_max_tokens(chunk_seconds, original_max_tokens);
    if retry_max_tokens >= original_max_tokens {
        let trim = trim_granite_looping_chunk_text(&mut details.text, &original_loop);
        details.diagnostics = with_granite_loop_recovery_diagnostics(
            details.diagnostics,
            &original_loop,
            None,
            original_max_tokens,
            retry_max_tokens,
            false,
            if trim.is_some() {
                "not_retried_budget_floor_trimmed_original"
            } else {
                "not_retried_budget_floor"
            },
            trim.as_ref(),
        );
        return Ok(details);
    }

    let mut retry_options = base_options.clone();
    retry_options.max_new_tokens = retry_max_tokens;
    let mut retry = redecode(retry_options)?;
    let retry_loop = granite_chunk_loop_signal(&retry.text, retry.diagnostics.as_ref());
    let retry_trim = retry_loop
        .as_ref()
        .and_then(|loop_signal| trim_granite_looping_chunk_text(&mut retry.text, loop_signal));
    let use_retry = !retry.text.trim().is_empty()
        && (retry_trim.is_some()
            || retry_loop
                .as_ref()
                .is_none_or(|loop_signal| loop_signal.score() < original_loop.score()));
    if use_retry {
        retry.diagnostics = with_granite_loop_recovery_diagnostics(
            retry.diagnostics,
            &original_loop,
            retry_loop.as_ref(),
            original_max_tokens,
            retry_max_tokens,
            true,
            if retry_trim.is_some() {
                "retry_selected_trimmed"
            } else {
                "retry_selected"
            },
            retry_trim.as_ref(),
        );
        return Ok(retry);
    }
    let original_trim = trim_granite_looping_chunk_text(&mut details.text, &original_loop);
    details.diagnostics = with_granite_loop_recovery_diagnostics(
        details.diagnostics,
        &original_loop,
        retry_loop.as_ref(),
        original_max_tokens,
        retry_max_tokens,
        false,
        if original_trim.is_some() {
            "retry_not_better_trimmed_original"
        } else {
            "retry_not_better"
        },
        original_trim.as_ref(),
    );
    Ok(details)
}

/// A chunk that spent its whole automatic budget without looping was cut
/// off mid-transcript. Re-decode it once at twice the budget, within the
/// request ceiling and the decode room the invocation leaves.
fn retry_truncated_chunk(
    mut details: NativeAsrTranscription,
    chunk_seconds: f32,
    base_options: &NativeAsrGenerationOptions,
    limits: ChunkRecoveryLimits,
    mut redecode: impl FnMut(NativeAsrGenerationOptions) -> Result<NativeAsrTranscription>,
) -> Result<NativeAsrTranscription> {
    if !granite_chunk_truncated(limits.auto_max_tokens, details.diagnostics.as_ref()) {
        return Ok(details);
    }
    let original_max_tokens = base_options.max_new_tokens;
    let prompt_tokens = details
        .diagnostics
        .as_ref()
        .and_then(|value| value.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let decode_room = limits.invocation_context.map(|(context, rate)| {
        let planned = crate::runtime::invocation_decode_room(context, rate, chunk_seconds);
        prompt_tokens.map_or(planned, |prompt| {
            planned.min(context.saturating_sub(prompt))
        })
    });
    let Some(retry_max_tokens) = granite_truncation_retry_max_tokens(
        original_max_tokens,
        limits.request_ceiling,
        decode_room,
    ) else {
        details.diagnostics = with_granite_truncation_retry_diagnostics(
            details.diagnostics,
            original_max_tokens,
            None,
            None,
            false,
            "not_retried_budget_ceiling",
        );
        return Ok(details);
    };
    let mut retry_options = base_options.clone();
    retry_options.max_new_tokens = retry_max_tokens;
    let mut retry = redecode(retry_options)?;
    let retry_stop_reason =
        granite_decode_stop_reason(retry.diagnostics.as_ref()).map(str::to_owned);
    let retry_loop = granite_chunk_loop_signal(&retry.text, retry.diagnostics.as_ref());
    let retry_trim = retry_loop
        .as_ref()
        .and_then(|loop_signal| trim_granite_looping_chunk_text(&mut retry.text, loop_signal));
    if !retry.text.trim().is_empty() && (retry_loop.is_none() || retry_trim.is_some()) {
        retry.diagnostics = with_granite_truncation_retry_diagnostics(
            retry.diagnostics,
            original_max_tokens,
            Some(retry_max_tokens),
            retry_stop_reason.as_deref(),
            true,
            if retry_trim.is_some() {
                "retry_selected_trimmed"
            } else {
                "retry_selected"
            },
        );
        return Ok(retry);
    }
    details.diagnostics = with_granite_truncation_retry_diagnostics(
        details.diagnostics,
        original_max_tokens,
        Some(retry_max_tokens),
        retry_stop_reason.as_deref(),
        false,
        "retry_not_better",
    );
    Ok(details)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcription(text: &str, stop_reason: &str, prompt_tokens: u64) -> NativeAsrTranscription {
        NativeAsrTranscription {
            text: text.to_string(),
            language: None,
            diagnostics: Some(json!({
                "prompt_tokens": prompt_tokens,
                "decode": { "stop_reason": stop_reason },
            })),
        }
    }

    fn options(max_new_tokens: usize) -> NativeAsrGenerationOptions {
        NativeAsrGenerationOptions {
            max_new_tokens,
            ..NativeAsrGenerationOptions::default()
        }
    }

    const LIMITS: ChunkRecoveryLimits = ChunkRecoveryLimits {
        auto_max_tokens: true,
        request_ceiling: 2048,
        invocation_context: None,
    };

    #[test]
    fn a_looping_chunk_is_redecoded_shorter_and_trimmed() {
        let looping = "we went to the shop and then we went to the shop and then we went to the shop and then we went to the shop and then";
        let mut budgets = Vec::new();
        let recovered = recover_granite_speech5_chunk(
            transcription(looping, "max_tokens", 40),
            10.0,
            &options(300),
            LIMITS,
            |retry| {
                budgets.push(retry.max_new_tokens);
                Ok(transcription("we went to the shop.", "stop_token", 40))
            },
        )
        .unwrap();
        assert_eq!(budgets, vec![225]);
        assert_eq!(recovered.text, "we went to the shop.");
        assert_eq!(
            recovered.diagnostics.unwrap()["chunk_loop_recovery"]["decision"],
            "retry_selected"
        );
    }

    #[test]
    fn a_truncated_chunk_is_redecoded_once_at_twice_the_budget() {
        let mut budgets = Vec::new();
        let recovered = recover_granite_speech5_chunk(
            transcription("the first half of a long", "max_tokens", 40),
            30.0,
            &options(300),
            LIMITS,
            |retry| {
                budgets.push(retry.max_new_tokens);
                Ok(transcription(
                    "the first half of a long sentence, and its end.",
                    "stop_token",
                    40,
                ))
            },
        )
        .unwrap();
        assert_eq!(budgets, vec![600]);
        assert!(recovered.text.ends_with("its end."));

        // A caller's explicit max_tokens is a cap: no retry.
        let capped = recover_granite_speech5_chunk(
            transcription("cut off", "max_tokens", 40),
            30.0,
            &options(300),
            ChunkRecoveryLimits {
                auto_max_tokens: false,
                ..LIMITS
            },
            |_| panic!("an explicit budget must not be retried"),
        )
        .unwrap();
        assert_eq!(capped.text, "cut off");

        // A finished chunk is returned as decoded.
        let finished = recover_granite_speech5_chunk(
            transcription("done.", "stop_token", 40),
            30.0,
            &options(300),
            LIMITS,
            |_| panic!("a finished chunk must not be retried"),
        )
        .unwrap();
        assert_eq!(finished.text, "done.");
    }

    #[test]
    fn loop_budget_stays_below_the_original_and_within_the_chunk_budget() {
        assert_eq!(loop_recovery_max_tokens(10.0, 300), 225);
        assert_eq!(loop_recovery_max_tokens(60.0, 2048), 600);
        assert_eq!(loop_recovery_max_tokens(10.0, 20), 19);
        assert_eq!(loop_recovery_max_tokens(10.0, 1), 1);
    }
}
