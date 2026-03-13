use izwi_core::{ChatMediaInput, ChatMediaKind, ModelVariant};
use serde_json::Value;

pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
pub const VIDEO_PLACEHOLDER: &str = "<|vision_start|><|video_pad|><|vision_end|>";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlattenedMultimodalContent {
    pub display_text: String,
    pub runtime_text: String,
    pub media_inputs: Vec<ChatMediaInput>,
}

impl FlattenedMultimodalContent {
    pub fn has_media(&self) -> bool {
        !self.media_inputs.is_empty()
    }
}

pub fn flatten_thread_content(
    raw_content: &str,
    content_parts: Option<&[Value]>,
) -> Result<FlattenedMultimodalContent, String> {
    let raw_trimmed = raw_content.trim().to_string();
    let Some(parts) = content_parts else {
        return Ok(FlattenedMultimodalContent {
            display_text: raw_trimmed.clone(),
            runtime_text: raw_trimmed,
            media_inputs: Vec::new(),
        });
    };
    if parts.is_empty() {
        return Ok(FlattenedMultimodalContent {
            display_text: raw_trimmed.clone(),
            runtime_text: raw_trimmed,
            media_inputs: Vec::new(),
        });
    }

    let mut flattened = flatten_content_parts(parts)?;
    if flattened.runtime_text.trim().is_empty() && !raw_trimmed.is_empty() {
        flattened.display_text = raw_trimmed.clone();
        flattened.runtime_text = raw_trimmed;
    } else if flattened.display_text.trim().is_empty() && !raw_trimmed.is_empty() {
        flattened.display_text = raw_trimmed;
    }
    Ok(flattened)
}

pub fn flatten_content_parts(parts: &[Value]) -> Result<FlattenedMultimodalContent, String> {
    let mut out = FlattenedMultimodalContent::default();
    for part in parts {
        if content_part_is_image(part) {
            let source = extract_media_source(part, ChatMediaKind::Image).ok_or_else(|| {
                "Multimodal content part is missing a usable image source".to_string()
            })?;
            out.runtime_text.push_str(IMAGE_PLACEHOLDER);
            out.media_inputs.push(ChatMediaInput {
                kind: ChatMediaKind::Image,
                source,
            });
            continue;
        }
        if content_part_is_video(part) {
            let source = extract_media_source(part, ChatMediaKind::Video).ok_or_else(|| {
                "Multimodal content part is missing a usable video source".to_string()
            })?;
            out.runtime_text.push_str(VIDEO_PLACEHOLDER);
            out.media_inputs.push(ChatMediaInput {
                kind: ChatMediaKind::Video,
                source,
            });
            continue;
        }
        if let Some(text) = resolve_text_part(part) {
            out.display_text.push_str(&text);
            out.runtime_text.push_str(&text);
        }
    }
    Ok(out)
}

pub fn content_part_is_image(part: &Value) -> bool {
    let Some(map) = part.as_object() else {
        return false;
    };
    if matches!(
        map.get("type")
            .or_else(|| map.get("kind"))
            .and_then(|value| value.as_str()),
        Some("image") | Some("image_url") | Some("input_image")
    ) {
        return true;
    }
    map.get("image").is_some_and(|value| !value.is_null())
        || map.get("image_url").is_some_and(|value| !value.is_null())
        || map.get("input_image").is_some_and(|value| !value.is_null())
}

pub fn content_part_is_video(part: &Value) -> bool {
    let Some(map) = part.as_object() else {
        return false;
    };
    if matches!(
        map.get("type")
            .or_else(|| map.get("kind"))
            .and_then(|value| value.as_str()),
        Some("video") | Some("video_url") | Some("input_video")
    ) {
        return true;
    }
    map.get("video").is_some_and(|value| !value.is_null())
        || map.get("video_url").is_some_and(|value| !value.is_null())
        || map.get("input_video").is_some_and(|value| !value.is_null())
}

pub fn validate_media_inputs_for_variant(
    variant: ModelVariant,
    media_inputs: &[ChatMediaInput],
) -> Result<(), String> {
    if media_inputs.is_empty() {
        return Ok(());
    }
    if media_inputs
        .iter()
        .any(|input| input.kind == ChatMediaKind::Video)
    {
        return Err("Qwen3.5 video inputs are not implemented yet".to_string());
    }
    if variant.is_qwen35_chat_gguf() {
        return Ok(());
    }
    Err(format!(
        "Multimodal chat input is currently supported only for Qwen3.5 GGUF models, not {}",
        variant.dir_name()
    ))
}

fn resolve_text_part(value: &Value) -> Option<String> {
    match value {
        Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Object(map) => map
            .get("text")
            .or_else(|| map.get("input_text"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        _ => None,
    }
}

fn extract_media_source(part: &Value, kind: ChatMediaKind) -> Option<String> {
    let map = part.as_object()?;
    let direct_keys: &[&str] = match kind {
        ChatMediaKind::Image => &["image_url", "input_image", "image"],
        ChatMediaKind::Video => &["video", "video_url", "input_video"],
    };
    for key in direct_keys {
        if let Some(source) = map
            .get(*key)
            .and_then(|value| resolve_media_source(value, 3))
        {
            return Some(source);
        }
    }
    None
}

fn resolve_media_source(value: &Value, max_depth: usize) -> Option<String> {
    if max_depth == 0 {
        return None;
    }
    match value {
        Value::String(raw) => {
            let source = raw.trim();
            if source.is_empty() {
                None
            } else {
                Some(source.to_string())
            }
        }
        Value::Object(map) => {
            if let Some(source) = map
                .get("url")
                .and_then(|nested| resolve_media_source(nested, max_depth - 1))
            {
                return Some(source);
            }
            for key in [
                "src",
                "uri",
                "path",
                "file",
                "image_url",
                "video_url",
                "input_image",
                "input_video",
            ] {
                if let Some(source) = map
                    .get(key)
                    .and_then(|nested| resolve_media_source(nested, max_depth - 1))
                {
                    return Some(source);
                }
            }
            if let Some(data_url) = map
                .get("b64_json")
                .and_then(|v| v.as_str())
                .and_then(|b64| data_url_from_base64_field(b64, map))
            {
                return Some(data_url);
            }
            if let Some(data) = map.get("data").and_then(|v| v.as_str()) {
                let data = data.trim();
                if data.starts_with("data:")
                    || data.starts_with("http://")
                    || data.starts_with("https://")
                    || data.starts_with("file://")
                {
                    return Some(data.to_string());
                }
                let is_base64 = map
                    .get("encoding")
                    .and_then(|v| v.as_str())
                    .is_some_and(|encoding| encoding.eq_ignore_ascii_case("base64"));
                if is_base64 {
                    return data_url_from_base64_field(data, map);
                }
            }
            None
        }
        _ => None,
    }
}

fn data_url_from_base64_field(
    b64: &str,
    map: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    let payload = b64.trim();
    if payload.is_empty() {
        return None;
    }
    let mime = map
        .get("mime_type")
        .or_else(|| map.get("media_type"))
        .or_else(|| map.get("content_type"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("application/octet-stream");
    Some(format!("data:{mime};base64,{payload}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flatten_content_parts_collects_images_and_runtime_placeholders() {
        let flattened = flatten_content_parts(&[
            json!({"type":"text","text":"Look "}),
            json!({"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}),
            json!({"type":"text","text":" now"}),
        ])
        .expect("flatten");

        assert_eq!(flattened.display_text, "Look  now");
        assert_eq!(
            flattened.runtime_text,
            format!("Look {IMAGE_PLACEHOLDER} now")
        );
        assert_eq!(
            flattened.media_inputs,
            vec![ChatMediaInput {
                kind: ChatMediaKind::Image,
                source: "https://example.com/cat.png".to_string(),
            }]
        );
    }

    #[test]
    fn validate_media_inputs_rejects_non_qwen35_variants() {
        let err = validate_media_inputs_for_variant(
            ModelVariant::Qwen34BGguf,
            &[ChatMediaInput {
                kind: ChatMediaKind::Image,
                source: "https://example.com/cat.png".to_string(),
            }],
        )
        .expect_err("non-qwen35 multimodal should fail");

        assert!(err.contains("currently supported only for Qwen3.5"));
    }

    #[test]
    fn flatten_thread_content_uses_raw_summary_for_attachment_only_messages() {
        let flattened = flatten_thread_content(
            "Attached image: cat.png",
            Some(&[json!({
                "type":"input_image",
                "input_image":{"url":"https://example.com/cat.png","name":"cat.png"}
            })]),
        )
        .expect("flatten attachment-only thread message");

        assert_eq!(flattened.display_text, "Attached image: cat.png");
        assert_eq!(flattened.runtime_text, IMAGE_PLACEHOLDER);
        assert_eq!(
            flattened.media_inputs,
            vec![ChatMediaInput {
                kind: ChatMediaKind::Image,
                source: "https://example.com/cat.png".to_string(),
            }]
        );
    }

    #[test]
    fn validate_media_inputs_rejects_video_inputs() {
        let err = validate_media_inputs_for_variant(
            ModelVariant::Qwen354BGguf,
            &[ChatMediaInput {
                kind: ChatMediaKind::Video,
                source: "https://example.com/demo.mp4".to_string(),
            }],
        )
        .expect_err("video inputs should fail");

        assert!(err.contains("video inputs are not implemented"));
    }
}
