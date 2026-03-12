use serde::Deserialize;

use crate::state::AppState;
use crate::voice_observation_store::CandidateObservation;
use izwi_core::{parse_chat_model_variant, ChatMessage, ChatRole};

const EXTRACTION_SYSTEM_PROMPT: &str = "Extract durable user memory from a single conversation turn. Return strict JSON only. Output a JSON array of objects with keys `category`, `summary`, and `confidence`. Only include stable user preferences, personal facts, recurring constraints, or long-lived goals that would help future conversations. Do not include temporary requests, assistant-only facts, or speculative guesses. If nothing is worth remembering, return []. Limit to at most 5 observations.";

#[derive(Debug, Deserialize)]
struct ExtractedObservation {
    category: Option<String>,
    summary: Option<String>,
    confidence: Option<f32>,
}

pub async fn extract_observation_candidates(
    state: &AppState,
    model_id: &str,
    correlation_id: &str,
    user_text: &str,
    assistant_text: &str,
) -> Result<Vec<CandidateObservation>, String> {
    let variant = parse_chat_model_variant(Some(model_id))
        .map_err(|err| format!("Invalid memory extraction model: {err}"))?;
    let response = state
        .runtime
        .chat_generate_with_correlation(
            variant,
            vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: EXTRACTION_SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: format!(
                        "User utterance:\n{}\n\nAssistant response:\n{}",
                        user_text.trim(),
                        assistant_text.trim()
                    ),
                },
            ],
            512,
            Some(correlation_id),
        )
        .await
        .map_err(|err| format!("Observation extraction failed: {err}"))?;

    let Some(raw_array) = extract_json_array(response.text.as_str()) else {
        return Ok(Vec::new());
    };
    let parsed = serde_json::from_str::<Vec<ExtractedObservation>>(raw_array)
        .map_err(|err| format!("Observation extraction returned invalid JSON: {err}"))?;

    Ok(parsed
        .into_iter()
        .filter_map(|item| {
            let summary = item.summary?.trim().to_string();
            if summary.is_empty() {
                return None;
            }
            Some(CandidateObservation {
                category: item
                    .category
                    .unwrap_or_else(|| "general".to_string())
                    .trim()
                    .to_string(),
                summary,
                confidence: item.confidence.unwrap_or(0.5),
            })
        })
        .take(5)
        .collect())
}

fn extract_json_array(raw: &str) -> Option<&str> {
    let start = raw.find('[')?;
    let end = raw.rfind(']')?;
    if end < start {
        return None;
    }
    raw.get(start..=end)
}

#[cfg(test)]
mod tests {
    use super::extract_json_array;

    #[test]
    fn extracts_json_array_from_fenced_output() {
        let raw = "```json\n[{\"category\":\"preference\",\"summary\":\"Prefers concise answers\",\"confidence\":0.9}]\n```";
        assert_eq!(
            extract_json_array(raw),
            Some("[{\"category\":\"preference\",\"summary\":\"Prefers concise answers\",\"confidence\":0.9}]")
        );
    }
}
