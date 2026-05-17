use std::sync::Arc;

use izwi_core::backends::BackendKind;
use izwi_core::catalog::ModelFamily;
use izwi_core::{ChatMessage, ChatRole, GenerationParams, ModelVariant, RuntimeService};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SummaryKind {
    Diarization,
    Transcription,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SummaryStrategy {
    SinglePass,
    Hierarchical,
}

impl SummaryStrategy {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::SinglePass => "single_pass",
            Self::Hierarchical => "hierarchical",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedSummary {
    pub(crate) text: String,
    pub(crate) strategy: SummaryStrategy,
    pub(crate) prompt_tokens: usize,
    pub(crate) chunk_count: usize,
}

impl SummaryKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Diarization => "diarization",
            Self::Transcription => "transcription",
        }
    }

    fn input_label(self) -> &'static str {
        match self {
            Self::Diarization => "diarized transcript",
            Self::Transcription => "transcript",
        }
    }

    fn single_pass_instruction(self) -> &'static str {
        match self {
            Self::Diarization => "Summarize the following diarized transcript.",
            Self::Transcription => "Summarize the following transcript.",
        }
    }

    fn chunk_instruction(self) -> &'static str {
        match self {
            Self::Diarization => {
                "Summarize this diarized transcript chunk. Preserve speaker contributions, decisions, action items, unresolved questions, and useful time anchors."
            }
            Self::Transcription => {
                "Summarize this transcript chunk. Preserve key topics, decisions, action items, and unresolved questions."
            }
        }
    }

    fn synthesis_instruction(self) -> &'static str {
        match self {
            Self::Diarization => {
                "Combine these diarized chunk summaries into one concise final summary. Preserve major speaker contributions, decisions, action items, unresolved questions, and useful time anchors."
            }
            Self::Transcription => {
                "Combine these transcript chunk summaries into one concise final summary. Preserve key topics, decisions, action items, and unresolved questions."
            }
        }
    }
}

pub(crate) fn should_use_cuda_hierarchical_summary(
    runtime: &RuntimeService,
    variant: ModelVariant,
) -> bool {
    runtime.backend_context().backend_kind == BackendKind::Cuda
        && variant.family() == ModelFamily::Qwen35Chat
}

pub(crate) fn summary_prompt_fits_cuda_prefill_budget(
    prompt_tokens: usize,
    max_output_tokens: usize,
    safe_prefill_tokens: usize,
) -> bool {
    prompt_tokens.saturating_add(max_output_tokens.max(1)) <= safe_prefill_tokens.max(1)
}

pub(crate) fn summary_chunk_char_budget(safe_prefill_tokens: usize) -> usize {
    safe_prefill_tokens.saturating_mul(3).clamp(1_024, 24_000)
}

pub(crate) fn single_pass_summary_messages(
    kind: SummaryKind,
    system_prompt: &str,
    input: &str,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: ChatRole::System,
            content: system_prompt.to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: format!(
                "{}\n\n{}:\n{}",
                kind.single_pass_instruction(),
                title_case_label(kind.input_label()),
                input
            ),
        },
    ]
}

pub(crate) fn chunk_summary_messages(
    kind: SummaryKind,
    system_prompt: &str,
    chunk: &str,
    chunk_index: usize,
    chunk_count: usize,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: ChatRole::System,
            content: system_prompt.to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: format!(
                "{}\n\nChunk {} of {}.\n\n{} chunk:\n{}",
                kind.chunk_instruction(),
                chunk_index,
                chunk_count,
                title_case_label(kind.input_label()),
                chunk
            ),
        },
    ]
}

pub(crate) fn synthesis_summary_messages(
    kind: SummaryKind,
    system_prompt: &str,
    chunk_summaries: &[String],
) -> Vec<ChatMessage> {
    let summaries = chunk_summaries
        .iter()
        .enumerate()
        .map(|(idx, summary)| format!("Chunk {}: {}", idx + 1, summary))
        .collect::<Vec<_>>()
        .join("\n\n");

    vec![
        ChatMessage {
            role: ChatRole::System,
            content: system_prompt.to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: format!(
                "{}\n\nChunk summaries:\n{}",
                kind.synthesis_instruction(),
                summaries
            ),
        },
    ]
}

pub(crate) fn split_summary_input_by_line_budget(input: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(1);
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in input.lines() {
        if line.chars().count() > max_chars {
            flush_current_chunk(&mut chunks, &mut current);
            split_long_line_into_chunks(line, max_chars, &mut chunks);
            continue;
        }

        let separator_len = usize::from(!current.is_empty());
        let next_len = current
            .chars()
            .count()
            .saturating_add(separator_len)
            .saturating_add(line.chars().count());
        if !current.is_empty() && next_len > max_chars {
            flush_current_chunk(&mut chunks, &mut current);
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    flush_current_chunk(&mut chunks, &mut current);
    chunks
}

pub(crate) async fn generate_summary_with_cuda_planner(
    runtime: Arc<RuntimeService>,
    variant: ModelVariant,
    kind: SummaryKind,
    system_prompt: &str,
    input: &str,
    params: GenerationParams,
    correlation_id: Option<&str>,
    sanitize: fn(&str) -> Option<String>,
) -> Result<PlannedSummary, String> {
    let safe_prefill_tokens = runtime.qwen35_cuda_safe_prefill_tokens();
    let single_pass_messages = single_pass_summary_messages(kind, system_prompt, input);
    let single_prompt_tokens =
        count_summary_prompt_tokens(&runtime, variant, &single_pass_messages, &params).await?;

    if summary_prompt_fits_cuda_prefill_budget(
        single_prompt_tokens,
        params.max_tokens,
        safe_prefill_tokens,
    ) {
        tracing::info!(
            target: "izwi.summary",
            summary_kind = kind.as_str(),
            summary_strategy = SummaryStrategy::SinglePass.as_str(),
            prompt_tokens = single_prompt_tokens,
            safe_prefill_tokens,
            correlation_id = correlation_id.unwrap_or_default(),
            "CUDA summary prompt fits single-pass prefill budget"
        );
        let text = generate_summary_text(
            &runtime,
            variant,
            single_pass_messages,
            params,
            correlation_id,
            sanitize,
        )
        .await?;
        return Ok(PlannedSummary {
            text,
            strategy: SummaryStrategy::SinglePass,
            prompt_tokens: single_prompt_tokens,
            chunk_count: 1,
        });
    }

    let chunks = plan_summary_chunks(
        &runtime,
        variant,
        kind,
        system_prompt,
        input,
        &params,
        safe_prefill_tokens,
    )
    .await?;
    tracing::info!(
        target: "izwi.summary",
        summary_kind = kind.as_str(),
        summary_strategy = SummaryStrategy::Hierarchical.as_str(),
        prompt_tokens = single_prompt_tokens,
        safe_prefill_tokens,
        chunk_count = chunks.len(),
        correlation_id = correlation_id.unwrap_or_default(),
        "CUDA summary prompt exceeds single-pass budget; using hierarchical summary"
    );

    let mut chunk_summaries = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.iter().enumerate() {
        let messages = chunk_summary_messages(kind, system_prompt, chunk, idx + 1, chunks.len());
        let summary = generate_summary_text(
            &runtime,
            variant,
            messages,
            params.clone(),
            correlation_id,
            sanitize,
        )
        .await?;
        chunk_summaries.push(summary);
    }

    let synthesis_messages = synthesis_summary_messages(kind, system_prompt, &chunk_summaries);
    let synthesis_prompt_tokens =
        count_summary_prompt_tokens(&runtime, variant, &synthesis_messages, &params).await?;
    if !summary_prompt_fits_cuda_prefill_budget(
        synthesis_prompt_tokens,
        params.max_tokens,
        safe_prefill_tokens,
    ) {
        return Err(format!(
            "Summary generation failed: final synthesis prompt exceeds CUDA safe prefill budget: prompt_tokens={}, max_tokens={}, safe_prefill_tokens={}, chunk_count={}",
            synthesis_prompt_tokens,
            params.max_tokens.max(1),
            safe_prefill_tokens,
            chunk_summaries.len()
        ));
    }

    let text = generate_summary_text(
        &runtime,
        variant,
        synthesis_messages,
        params,
        correlation_id,
        sanitize,
    )
    .await?;
    Ok(PlannedSummary {
        text,
        strategy: SummaryStrategy::Hierarchical,
        prompt_tokens: single_prompt_tokens,
        chunk_count: chunks.len(),
    })
}

async fn plan_summary_chunks(
    runtime: &RuntimeService,
    variant: ModelVariant,
    kind: SummaryKind,
    system_prompt: &str,
    input: &str,
    params: &GenerationParams,
    safe_prefill_tokens: usize,
) -> Result<Vec<String>, String> {
    let mut char_budget = summary_chunk_char_budget(safe_prefill_tokens);
    for _ in 0..8 {
        let chunks = split_summary_input_by_line_budget(input, char_budget);
        if chunks.is_empty() {
            return Ok(chunks);
        }

        let mut all_fit = true;
        for (idx, chunk) in chunks.iter().enumerate() {
            let messages =
                chunk_summary_messages(kind, system_prompt, chunk, idx + 1, chunks.len());
            let prompt_tokens =
                count_summary_prompt_tokens(runtime, variant, &messages, params).await?;
            if !summary_prompt_fits_cuda_prefill_budget(
                prompt_tokens,
                params.max_tokens,
                safe_prefill_tokens,
            ) {
                all_fit = false;
                break;
            }
        }
        if all_fit {
            return Ok(chunks);
        }
        char_budget /= 2;
        if char_budget < 512 {
            break;
        }
    }

    Err(format!(
        "Summary generation failed: unable to split {} under CUDA safe prefill budget {}",
        kind.input_label(),
        safe_prefill_tokens
    ))
}

async fn count_summary_prompt_tokens(
    runtime: &RuntimeService,
    variant: ModelVariant,
    messages: &[ChatMessage],
    params: &GenerationParams,
) -> Result<usize, String> {
    runtime
        .chat_prompt_token_count_with_generation_params(variant, messages, params)
        .await
        .map_err(|err| format!("Summary generation failed: prompt tokenization failed: {err}"))
}

async fn generate_summary_text(
    runtime: &RuntimeService,
    variant: ModelVariant,
    messages: Vec<ChatMessage>,
    params: GenerationParams,
    correlation_id: Option<&str>,
    sanitize: fn(&str) -> Option<String>,
) -> Result<String, String> {
    let generation = runtime
        .chat_generate_with_generation_params_and_correlation(
            variant,
            messages,
            params,
            correlation_id,
        )
        .await
        .map_err(|err| format!("Summary generation failed: {err}"))?;

    sanitize(generation.text.as_str())
        .ok_or_else(|| "Summary generation returned empty text".to_string())
}

fn flush_current_chunk(chunks: &mut Vec<String>, current: &mut String) {
    if current.trim().is_empty() {
        current.clear();
        return;
    }
    chunks.push(current.trim().to_string());
    current.clear();
}

fn split_long_line_into_chunks(line: &str, max_chars: usize, chunks: &mut Vec<String>) {
    let chars = line.chars().collect::<Vec<_>>();
    for segment in chars.chunks(max_chars) {
        let chunk = segment.iter().collect::<String>();
        if !chunk.trim().is_empty() {
            chunks.push(chunk.trim().to_string());
        }
    }
}

fn title_case_label(label: &str) -> &'static str {
    match label {
        "diarized transcript" => "Transcript",
        "transcript" => "Transcript",
        _ => "Input",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        chunk_summary_messages, single_pass_summary_messages, split_summary_input_by_line_budget,
        summary_chunk_char_budget, summary_prompt_fits_cuda_prefill_budget, SummaryKind,
    };

    #[test]
    fn summary_prompt_budget_reserves_output_tokens() {
        assert!(summary_prompt_fits_cuda_prefill_budget(7000, 1000, 8192));
        assert!(!summary_prompt_fits_cuda_prefill_budget(8000, 384, 8192));
    }

    #[test]
    fn summary_chunk_char_budget_is_bounded() {
        assert_eq!(summary_chunk_char_budget(1), 1024);
        assert_eq!(summary_chunk_char_budget(4096), 12_288);
        assert_eq!(summary_chunk_char_budget(262_144), 24_000);
    }

    #[test]
    fn split_summary_input_keeps_line_boundaries_where_possible() {
        let chunks = split_summary_input_by_line_budget("a\nbb\nccc\ndddd", 6);
        assert_eq!(chunks, vec!["a\nbb", "ccc", "dddd"]);
    }

    #[test]
    fn split_summary_input_wraps_single_long_lines() {
        let chunks = split_summary_input_by_line_budget("abcdef", 2);
        assert_eq!(chunks, vec!["ab", "cd", "ef"]);
    }

    #[test]
    fn summary_messages_keep_existing_single_pass_instruction() {
        let messages = single_pass_summary_messages(SummaryKind::Diarization, "system", "hello");
        assert_eq!(messages[0].content, "system");
        assert!(messages[1]
            .content
            .starts_with("Summarize the following diarized transcript."));
        assert!(messages[1].content.contains("Transcript:\nhello"));
    }

    #[test]
    fn chunk_summary_messages_include_chunk_ordinals() {
        let messages = chunk_summary_messages(SummaryKind::Transcription, "system", "hello", 2, 3);
        assert!(messages[1].content.contains("Chunk 2 of 3."));
        assert!(messages[1].content.contains("Transcript chunk:\nhello"));
    }
}
