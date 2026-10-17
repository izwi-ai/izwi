use super::*;
use crate::api::openai::chat::dto::OpenAiChatChunk;

fn reasoning_mode_for(body: serde_json::Value) -> Result<Option<bool>, ApiError> {
    let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    let ctx = RequestContext::new(
        "reasoning".into(),
        izwi_hooks::Principal::local_anonymous(),
    );
    prepare_execution_request(&req, &ctx).map(|(_, format)| format.reasoning)
}

#[test]
fn reasoning_format_is_opt_in_and_knows_who_opens_the_block() {
    let user = serde_json::json!([{"role": "user", "content": "hi"}]);
    // Default: reasoning stays in `content`.
    assert_eq!(
        reasoning_mode_for(serde_json::json!({"model": "LFM2.5-1.2B-Thinking-GGUF", "messages": user}))
            .unwrap(),
        None
    );
    // LFM2.5 Thinking writes its own `<think>`.
    assert_eq!(
        reasoning_mode_for(serde_json::json!({
            "model": "LFM2.5-1.2B-Thinking-GGUF", "messages": user, "reasoning_format": "parsed"
        }))
        .unwrap(),
        Some(false)
    );
    // Qwen3.8 opens the block in its prompt unless thinking is off.
    assert_eq!(
        reasoning_mode_for(serde_json::json!({
            "model": "Qwen3.8-27B-FP8", "messages": user, "reasoning_format": "parsed"
        }))
        .unwrap(),
        Some(true)
    );
    assert_eq!(
        reasoning_mode_for(serde_json::json!({
            "model": "Qwen3.8-27B-FP8", "messages": user, "reasoning_format": "parsed",
            "enable_thinking": false
        }))
        .unwrap(),
        Some(false)
    );
    assert!(reasoning_mode_for(serde_json::json!({
        "model": "LFM2.5-1.2B-Thinking-GGUF", "messages": user, "reasoning_format": "deepseek"
    }))
    .is_err());
}

#[tokio::test]
async fn parsed_reasoning_fills_reasoning_content_and_keeps_tool_calls() {
    let generation = |text: &str| ChatGeneration {
        latency_breakdown: None,
        finish_reason: None,
        text: text.to_string(),
        prompt_tokens: 1,
        tokens_generated: 1,
        generation_time_ms: 0.0,
        cached_prompt_tokens: None,
        logprobs: Vec::new(),
    };
    let body = |response: Response| async move {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
    };
    let format = |reasoning, tool_calls| AnswerFormat {
        compat_profile: OpenAiCompatibilityProfile::Strict,
        reasoning,
        tool_calls,
    };
    let text = "<think>check the tool</think>\n\n<|tool_call_start|>[now()]<|tool_call_end|>";
    let parsed = body(render_completion_response(
        ModelVariant::Lfm2512BThinkingGguf,
        generation(text),
        format(Some(false), Some(ToolCallFormat::LfmPythonic)),
    ))
    .await;
    let message = &parsed["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "check the tool");
    assert!(message.get("content").is_none());
    assert_eq!(message["tool_calls"][0]["function"]["name"], "now");
    assert_eq!(parsed["choices"][0]["finish_reason"], "tool_calls");

    let raw = body(render_completion_response(
        ModelVariant::Lfm2512BThinkingGguf,
        generation("<think>r</think>A"),
        format(None, None),
    ))
    .await;
    assert!(raw["choices"][0]["message"].get("reasoning_content").is_none());
    assert_eq!(raw["choices"][0]["message"]["content"], "<think>r</think>A");
}

#[test]
fn budget_termination_is_length_without_overriding_tool_calls() {
    use izwi_core::engine::OutputFinishReason::{MaxTokens, StopToken};
    assert_eq!(finish_reason(false, Some(MaxTokens)), "length");
    assert_eq!(finish_reason(false, Some(StopToken)), "stop");
    assert_eq!(finish_reason(true, Some(MaxTokens)), "tool_calls");
}

#[test]
fn json_and_sse_preserve_independent_timing_and_omit_absent_extensions() {
    let timing: izwi_core::engine::LatencyBreakdown =
        serde_json::from_value(serde_json::json!({
            "queue_wait_ms": 10.0, "prefill_ms": 20.0, "decode_ms": 30.0,
            "decode_wall_ms": 50.0, "decode_tokens": 3, "post_first_token_ms": 50.0,
            "ttft_ms": 30.0, "total_ms": 90.0, "prefill_steps": 1, "decode_steps": 2
        }))
        .unwrap();
    let json = OpenAiChatCompletionResponse {
        id: "timing".into(),
        object: "chat.completion",
        created: 0,
        model: "fixture".into(),
        choices: vec![],
        usage: OpenAiUsage {
            prompt_tokens: 5,
            completion_tokens: 4,
            total_tokens: 9,
            prompt_tokens_details: OpenAiPromptTokensDetails { cached_tokens: 0 },
        },
        izwi_generation_time_ms: Some(90.0),
        izwi_timing: Some(timing.clone()),
    };
    let mut sse = OpenAiChatChunk {
        id: "timing".into(),
        object: "chat.completion.chunk",
        created: 0,
        model: "fixture".into(),
        choices: vec![],
        usage: None,
        izwi_generation_time_ms: Some(90.0),
        izwi_timing: Some(timing),
    };
    let json = serde_json::to_value(json).unwrap();
    let chunk = serde_json::to_value(&sse).unwrap();
    assert_eq!(json["izwi_timing"], chunk["izwi_timing"]);
    assert_eq!(json["izwi_generation_time_ms"], 90.0);
    assert_eq!(chunk["izwi_timing"]["decode_wall_ms"], 50.0);
    assert_eq!(chunk["izwi_timing"]["decode_tokens"], 3);
    sse.izwi_generation_time_ms = None;
    sse.izwi_timing = None;
    let strict = serde_json::to_value(sse).unwrap();
    assert!(strict.get("izwi_timing").is_none());
    assert!(strict.get("izwi_generation_time_ms").is_none());
}

#[test]
fn logprob_requests_validate_openai_bounds() {
    let req = |logprobs: Option<bool>, top_logprobs: Option<u8>| ChatCompletionRequest {
        model: "m".into(),
        messages: vec![],
        max_tokens: None,
        max_completion_tokens: None,
        stream: None,
        stream_options: None,
        n: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        logprobs,
        top_logprobs,
        response_format: None,
        stop: None,
        tools: None,
        tool_choice: None,
        enable_thinking: None,
        reasoning_effort: None,
        preserve_thinking: None,
        chat_template_kwargs: None,
        reasoning_format: None,
        truncation: None,
        skip_special_tokens: None,
    };
    assert!(validate_chat_request_compatibility(&req(Some(true), Some(20)), profile()).is_ok());
    assert!(validate_chat_request_compatibility(&req(Some(true), None), profile()).is_ok());
    let too_many = validate_chat_request_compatibility(&req(Some(true), Some(21)), profile());
    assert!(too_many.is_err(), "top_logprobs above 20 must be rejected");
    let missing_flag = validate_chat_request_compatibility(&req(None, Some(3)), profile());
    assert!(
        missing_flag.is_err(),
        "top_logprobs without logprobs must be rejected"
    );
}

fn profile() -> OpenAiCompatibilityProfile {
    OpenAiCompatibilityProfile::Relaxed
}

#[test]
fn response_format_types_are_validated() {
    let req = |kind: &str| ChatCompletionRequest {
        model: "m".into(),
        messages: vec![],
        max_tokens: None,
        max_completion_tokens: None,
        stream: None,
        stream_options: None,
        n: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        logprobs: None,
        top_logprobs: None,
        response_format: Some(OpenAiResponseFormat { kind: kind.into() }),
        stop: None,
        tools: None,
        tool_choice: None,
        enable_thinking: None,
        reasoning_effort: None,
        preserve_thinking: None,
        chat_template_kwargs: None,
        reasoning_format: None,
        truncation: None,
        skip_special_tokens: None,
    };
    assert!(validate_chat_request_compatibility(&req("json_object"), profile()).is_ok());
    assert!(
        validate_chat_request_compatibility(&req("json_schema"), profile()).is_err(),
        "json_schema must be rejected with a documented error"
    );
    assert!(
        validate_chat_request_compatibility(&req("yaml"), profile()).is_err(),
        "unknown types must be rejected"
    );
}

#[test]
fn response_format_deserialization_ignores_json_schema_payload_keys() {
    let format: OpenAiResponseFormat = serde_json::from_value(serde_json::json!({
        "type": "json_object",
        "json_schema": { "name": "strict-output", "strict": true }
    }))
    .expect("the json_schema payload key is ignored, not a parse error");
    assert_eq!(format.kind, "json_object");
}

#[test]
fn json_object_is_rejected_on_models_without_a_grammar_aware_sampler() {
    let format = OpenAiResponseFormat {
        kind: "json_object".into(),
    };
    assert!(ensure_response_format_supported(ModelVariant::Qwen314BGguf, &format).is_ok());
    assert!(
        ensure_response_format_supported(ModelVariant::Qwen3827BFp8, &format).is_err(),
        "qwen3.8 has its own sampler without the grammar seam"
    );
    assert!(
        ensure_response_format_supported(ModelVariant::Qwen36Moe35BA3BFp8, &format).is_ok(),
        "qwen3.6-moe wires the DS9.2 grammar into its decode states"
    );
}

#[test]
fn logprob_entries_render_openai_shape() {
    let entries = vec![izwi_core::engine::TokenLogprob {
        token: " hello".to_string(),
        logprob: -0.25,
        bytes: b" hello".to_vec(),
        top_logprobs: vec![
            izwi_core::engine::TopTokenLogprob {
                token: " hello".to_string(),
                logprob: -0.25,
                bytes: b" hello".to_vec(),
            },
            izwi_core::engine::TopTokenLogprob {
                token: " hi".to_string(),
                logprob: -1.5,
                bytes: b" hi".to_vec(),
            },
        ],
    }];
    let rendered = serde_json::to_value(openai_logprobs(&entries).unwrap()).unwrap();
    let first = &rendered["content"][0];
    assert_eq!(first["token"], " hello");
    assert_eq!(first["logprob"], -0.25);
    assert_eq!(
        first["bytes"],
        serde_json::json!([32, 104, 101, 108, 108, 111])
    );
    assert_eq!(first["top_logprobs"].as_array().unwrap().len(), 2);
    assert_eq!(first["top_logprobs"][1]["token"], " hi");
    assert!(openai_logprobs(&[]).is_none());
}

#[test]
fn usage_reports_openai_shaped_cached_tokens() {
    let measured = serde_json::to_value(OpenAiUsage {
        prompt_tokens: 10,
        completion_tokens: 4,
        total_tokens: 14,
        prompt_tokens_details: OpenAiPromptTokensDetails { cached_tokens: 7 },
    })
    .unwrap();
    assert_eq!(measured["prompt_tokens"], 10);
    assert_eq!(measured["prompt_tokens_details"]["cached_tokens"], 7);

    let unmeasured = serde_json::to_value(OpenAiUsage {
        prompt_tokens: 10,
        completion_tokens: 4,
        total_tokens: 14,
        prompt_tokens_details: OpenAiPromptTokensDetails { cached_tokens: 0 },
    })
    .unwrap();
    assert_eq!(
        unmeasured["prompt_tokens_details"]["cached_tokens"], 0,
        "unmeasured cache must render as an explicit zero, OpenAI parity"
    );
}
