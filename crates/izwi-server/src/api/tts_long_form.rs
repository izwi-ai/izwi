use std::sync::Arc;

use izwi_core::{GenerationRequest, GenerationResult, ModelVariant, RuntimeService};

const DEFAULT_CHUNK_MAX_CHARS: usize = 480;
const CHUNK_MAX_CHARS_MIN: usize = 80;
const CHUNK_MAX_CHARS_MAX: usize = 4000;

fn chunk_max_chars() -> usize {
    std::env::var("IZWI_TTS_LONG_FORM_CHUNK_MAX_CHARS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .map(|value| value.clamp(CHUNK_MAX_CHARS_MIN, CHUNK_MAX_CHARS_MAX))
        .unwrap_or(DEFAULT_CHUNK_MAX_CHARS)
}

fn should_enable_chunking(variant: ModelVariant, requested_max_tokens: usize) -> bool {
    let Some(model_max_frames) = variant.tts_max_output_frames_hint() else {
        return false;
    };
    requested_max_tokens == 0 || requested_max_tokens >= model_max_frames
}

fn is_sentence_break(ch: char) -> bool {
    matches!(
        ch,
        '.' | '!' | '?' | ';' | ':' | '。' | '！' | '？' | '；' | '：' | '\n'
    )
}

fn push_trimmed(units: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        units.push(trimmed.to_string());
    }
    current.clear();
}

fn split_sentence_units(text: &str) -> Vec<String> {
    let mut units = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if is_sentence_break(ch) {
            push_trimmed(&mut units, &mut current);
        }
    }
    push_trimmed(&mut units, &mut current);

    if units.is_empty() {
        vec![text.trim().to_string()]
    } else {
        units
    }
}

fn split_overlong_unit(unit: &str, max_chars: usize) -> Vec<String> {
    let char_len = unit.chars().count();
    if char_len <= max_chars {
        return vec![unit.trim().to_string()];
    }

    let words: Vec<&str> = unit.split_whitespace().collect();
    if words.len() > 1 {
        let mut out = Vec::new();
        let mut current = String::new();
        let mut current_chars = 0usize;

        for word in words {
            let word_chars = word.chars().count();
            let sep_chars = if current.is_empty() { 0 } else { 1 };
            if current_chars + sep_chars + word_chars <= max_chars {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(word);
                current_chars += sep_chars + word_chars;
                continue;
            }

            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
                current_chars = 0;
            }

            if word_chars <= max_chars {
                current.push_str(word);
                current_chars = word_chars;
                continue;
            }

            // Single unbroken token longer than chunk limit (e.g. CJK block or URL).
            let mut token_chunk = String::new();
            let mut token_chars = 0usize;
            for ch in word.chars() {
                token_chunk.push(ch);
                token_chars += 1;
                if token_chars >= max_chars {
                    out.push(token_chunk.clone());
                    token_chunk.clear();
                    token_chars = 0;
                }
            }
            if !token_chunk.is_empty() {
                out.push(token_chunk);
            }
        }

        if !current.is_empty() {
            out.push(current);
        }
        return out;
    }

    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;
    for ch in unit.chars() {
        current.push(ch);
        current_chars += 1;
        if current_chars >= max_chars {
            out.push(current.clone());
            current.clear();
            current_chars = 0;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

pub fn split_tts_text_for_long_form(
    variant: ModelVariant,
    requested_max_tokens: usize,
    text: &str,
) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if !should_enable_chunking(variant, requested_max_tokens) {
        return vec![trimmed.to_string()];
    }

    let max_chars = chunk_max_chars();
    let mut chunks = Vec::new();
    for sentence in split_sentence_units(trimmed) {
        chunks.extend(split_overlong_unit(sentence.as_str(), max_chars));
    }

    if chunks.is_empty() {
        vec![trimmed.to_string()]
    } else {
        chunks
    }
}

pub fn expand_generation_requests_for_long_form(
    base_request: &GenerationRequest,
    variant: ModelVariant,
) -> Vec<GenerationRequest> {
    let chunks = split_tts_text_for_long_form(
        variant,
        base_request.config.options.max_tokens,
        base_request.text.as_str(),
    );

    if chunks.len() <= 1 {
        return vec![base_request.clone()];
    }

    chunks
        .into_iter()
        .enumerate()
        .map(|(idx, text)| {
            let mut req = base_request.clone();
            req.id = format!("{}:{}", base_request.id, idx + 1);
            req.text = text;
            req
        })
        .collect()
}

pub async fn generate_long_form_tts(
    runtime: &Arc<RuntimeService>,
    variant: ModelVariant,
    request: GenerationRequest,
) -> Result<GenerationResult, izwi_core::Error> {
    let planned_requests = expand_generation_requests_for_long_form(&request, variant);
    if planned_requests.len() == 1 {
        return runtime
            .generate(planned_requests.into_iter().next().unwrap())
            .await;
    }

    let mut merged_samples: Vec<f32> = Vec::new();
    let mut sample_rate: Option<u32> = None;
    let mut total_tokens = 0usize;
    let mut total_time_ms = 0f32;

    for chunk_request in planned_requests {
        let output = runtime.generate(chunk_request).await?;
        if let Some(existing_rate) = sample_rate {
            if existing_rate != output.sample_rate {
                return Err(izwi_core::Error::InferenceError(format!(
                    "Long-form TTS sample-rate mismatch: {existing_rate} vs {}",
                    output.sample_rate
                )));
            }
        } else {
            sample_rate = Some(output.sample_rate);
        }
        merged_samples.extend_from_slice(&output.samples);
        total_tokens = total_tokens.saturating_add(output.total_tokens);
        total_time_ms += output.total_time_ms;
    }

    let sample_rate = sample_rate.ok_or_else(|| {
        izwi_core::Error::InferenceError("Long-form TTS produced no chunks".to_string())
    })?;

    Ok(GenerationResult {
        request_id: request.id,
        samples: merged_samples,
        sample_rate,
        total_tokens,
        total_time_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_auto_chunking_splits_sentences() {
        let chunks = split_tts_text_for_long_form(
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            0,
            "Hello world. This is sentence two! Final line?",
        );
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn explicit_small_token_budget_disables_long_form_split() {
        let chunks = split_tts_text_for_long_form(
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            256,
            "Sentence one. Sentence two.",
        );
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn non_qwen_variant_stays_single_chunk() {
        let chunks = split_tts_text_for_long_form(
            ModelVariant::Kokoro82M,
            0,
            "Sentence one. Sentence two.",
        );
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn qwen_voice_design_auto_chunking_splits_sentences() {
        let chunks = split_tts_text_for_long_form(
            ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
            0,
            "Voice design sentence one. Voice design sentence two.",
        );
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn qwen_base_auto_chunking_splits_sentences() {
        let chunks = split_tts_text_for_long_form(
            ModelVariant::Qwen3Tts12Hz06BBase,
            0,
            "Voice cloning sentence one. Voice cloning sentence two.",
        );
        assert_eq!(chunks.len(), 2);
    }
}
