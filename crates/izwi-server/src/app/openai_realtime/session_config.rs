//! The GA Realtime session object: what `session.update` may change, how a
//! partial update merges (LiveKit sends instructions-only and tools-only
//! updates), and the full session echoed in `session.created` and
//! `session.updated`.
//!
//! The echo stays strictly OpenAI-shaped: Pipecat validates it with pydantic
//! (24 kHz PCM, a string `tool_choice`, transcription objects carrying
//! `language` and `prompt`), so nothing Izwi-specific goes in it.

use serde_json::{json, Map, Value};

use super::events::RealtimeError;

/// OpenAI's documented `server_vad` defaults (D3).
pub(crate) const DEFAULT_VAD_THRESHOLD: f32 = 0.5;
pub(crate) const DEFAULT_PREFIX_PADDING_MS: u32 = 300;
pub(crate) const DEFAULT_SILENCE_DURATION_MS: u32 = 500;
/// The most metadata pairs a response carries (OpenAI's limit).
const MAX_METADATA_PAIRS: usize = 16;

/// An audio format on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioFormat {
    /// `audio/pcm` at 24 kHz, 16-bit little-endian mono.
    Pcm24k,
    /// `audio/pcmu`: G.711 μ-law at 8 kHz.
    Pcmu,
    /// `audio/pcma`: G.711 A-law at 8 kHz.
    Pcma,
}

impl AudioFormat {
    pub(crate) const PCM_RATE: u32 = 24_000;

    fn parse(value: &Value, param: &str) -> Result<Self, RealtimeError> {
        let invalid = |message: String| {
            RealtimeError::invalid("invalid_value", message).with_param(param.to_string())
        };
        // The beta strings, for clients that still send them in a GA shape.
        if let Some(name) = value.as_str() {
            return match name {
                "pcm16" | "audio/pcm" => Ok(Self::Pcm24k),
                "g711_ulaw" | "audio/pcmu" => Ok(Self::Pcmu),
                "g711_alaw" | "audio/pcma" => Ok(Self::Pcma),
                other => Err(invalid(format!("Unsupported audio format '{other}'."))),
            };
        }
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("The audio format needs a 'type'.".to_string()))?;
        match kind {
            "audio/pcm" => match value.get("rate").and_then(Value::as_u64) {
                None | Some(24_000) => Ok(Self::Pcm24k),
                Some(rate) => Err(invalid(format!(
                    "Only a 24000 Hz sample rate is supported for audio/pcm, not {rate}."
                ))),
            },
            "audio/pcmu" => Ok(Self::Pcmu),
            "audio/pcma" => Ok(Self::Pcma),
            other => Err(invalid(format!("Unsupported audio format type '{other}'."))),
        }
    }

    pub(crate) fn to_json(self) -> Value {
        match self {
            Self::Pcm24k => json!({"type": "audio/pcm", "rate": Self::PCM_RATE}),
            Self::Pcmu => json!({"type": "audio/pcmu"}),
            Self::Pcma => json!({"type": "audio/pcma"}),
        }
    }

    /// The sample rate of audio in this format.
    pub(crate) const fn sample_rate(self) -> u32 {
        match self {
            Self::Pcm24k => Self::PCM_RATE,
            Self::Pcmu | Self::Pcma => 8_000,
        }
    }
}

/// `semantic_vad` eagerness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Eagerness {
    Low,
    Medium,
    High,
    Auto,
}

impl Eagerness {
    fn parse(value: Option<&Value>) -> Result<Self, RealtimeError> {
        match value.and_then(Value::as_str).unwrap_or("auto") {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "auto" => Ok(Self::Auto),
            other => Err(RealtimeError::invalid(
                "invalid_value",
                format!("Invalid eagerness '{other}'."),
            )
            .with_param("session.audio.input.turn_detection.eagerness")),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Auto => "auto",
        }
    }

    /// The silence that ends a turn. Izwi has no semantic end-of-turn model
    /// (D14): eagerness picks the trailing-silence window instead.
    pub(crate) const fn silence_ms(self) -> u32 {
        match self {
            Self::High => 350,
            Self::Medium | Self::Auto => 800,
            Self::Low => 1_500,
        }
    }
}

/// `audio.input.turn_detection`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TurnDetection {
    ServerVad {
        threshold: f32,
        prefix_padding_ms: u32,
        silence_duration_ms: u32,
        idle_timeout_ms: Option<u32>,
        create_response: bool,
        interrupt_response: bool,
    },
    SemanticVad {
        eagerness: Eagerness,
        create_response: bool,
        interrupt_response: bool,
    },
}

impl Default for TurnDetection {
    fn default() -> Self {
        Self::ServerVad {
            threshold: DEFAULT_VAD_THRESHOLD,
            prefix_padding_ms: DEFAULT_PREFIX_PADDING_MS,
            silence_duration_ms: DEFAULT_SILENCE_DURATION_MS,
            idle_timeout_ms: None,
            create_response: true,
            interrupt_response: true,
        }
    }
}

fn bool_field(object: &Map<String, Value>, key: &str, default: bool) -> bool {
    object.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn ms_field(
    object: &Map<String, Value>,
    key: &str,
    default: u32,
    range: (u32, u32),
) -> Result<u32, RealtimeError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|ms| ms.is_finite() && *ms >= 0.0)
            .map(|ms| (ms.round() as u32).clamp(range.0, range.1))
            .ok_or_else(|| {
                RealtimeError::invalid("invalid_type", format!("'{key}' must be a number."))
                    .with_param(format!("session.audio.input.turn_detection.{key}"))
            }),
    }
}

impl TurnDetection {
    /// `None` for `null` (manual turns).
    fn parse(value: &Value) -> Result<Option<Self>, RealtimeError> {
        let Value::Object(object) = value else {
            return if value.is_null() {
                Ok(None)
            } else {
                Err(RealtimeError::invalid(
                    "invalid_type",
                    "'turn_detection' must be an object or null.",
                )
                .with_param("session.audio.input.turn_detection"))
            };
        };
        let create_response = bool_field(object, "create_response", true);
        let interrupt_response = bool_field(object, "interrupt_response", true);
        match object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("server_vad")
        {
            "server_vad" => {
                let threshold = match object.get("threshold") {
                    None | Some(Value::Null) => DEFAULT_VAD_THRESHOLD,
                    Some(value) => value
                        .as_f64()
                        .filter(|threshold| (0.0..=1.0).contains(threshold))
                        .map(|threshold| threshold as f32)
                        .ok_or_else(|| {
                            RealtimeError::invalid(
                                "invalid_value",
                                "'threshold' must be a number from 0 to 1.",
                            )
                            .with_param("session.audio.input.turn_detection.threshold")
                        })?,
                };
                let idle_timeout_ms = match object.get("idle_timeout_ms") {
                    None | Some(Value::Null) => None,
                    Some(_) => Some(ms_field(object, "idle_timeout_ms", 0, (5_000, 30_000))?),
                };
                Ok(Some(Self::ServerVad {
                    threshold,
                    prefix_padding_ms: ms_field(
                        object,
                        "prefix_padding_ms",
                        DEFAULT_PREFIX_PADDING_MS,
                        (0, 2_000),
                    )?,
                    silence_duration_ms: ms_field(
                        object,
                        "silence_duration_ms",
                        DEFAULT_SILENCE_DURATION_MS,
                        (50, 10_000),
                    )?,
                    idle_timeout_ms,
                    create_response,
                    interrupt_response,
                }))
            }
            "semantic_vad" => Ok(Some(Self::SemanticVad {
                eagerness: Eagerness::parse(object.get("eagerness"))?,
                create_response,
                interrupt_response,
            })),
            other => Err(RealtimeError::invalid(
                "invalid_value",
                format!("Invalid turn detection type '{other}'."),
            )
            .with_param("session.audio.input.turn_detection.type")),
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        match self {
            Self::ServerVad {
                threshold,
                prefix_padding_ms,
                silence_duration_ms,
                idle_timeout_ms,
                create_response,
                interrupt_response,
            } => json!({
                "type": "server_vad",
                "threshold": (*threshold as f64 * 1000.0).round() / 1000.0,
                "prefix_padding_ms": prefix_padding_ms,
                "silence_duration_ms": silence_duration_ms,
                "idle_timeout_ms": idle_timeout_ms,
                "create_response": create_response,
                "interrupt_response": interrupt_response,
            }),
            Self::SemanticVad {
                eagerness,
                create_response,
                interrupt_response,
            } => json!({
                "type": "semantic_vad",
                "eagerness": eagerness.as_str(),
                "create_response": create_response,
                "interrupt_response": interrupt_response,
            }),
        }
    }

    pub(crate) fn create_response(&self) -> bool {
        match self {
            Self::ServerVad {
                create_response, ..
            }
            | Self::SemanticVad {
                create_response, ..
            } => *create_response,
        }
    }

    pub(crate) fn interrupt_response(&self) -> bool {
        match self {
            Self::ServerVad {
                interrupt_response, ..
            }
            | Self::SemanticVad {
                interrupt_response, ..
            } => *interrupt_response,
        }
    }

    /// The VAD knobs: start threshold, pre-roll and the silence that ends a
    /// turn.
    pub(crate) fn vad(&self) -> (f32, u32, u32) {
        match self {
            Self::ServerVad {
                threshold,
                prefix_padding_ms,
                silence_duration_ms,
                ..
            } => (*threshold, *prefix_padding_ms, *silence_duration_ms),
            Self::SemanticVad { eagerness, .. } => (
                DEFAULT_VAD_THRESHOLD,
                DEFAULT_PREFIX_PADDING_MS,
                eagerness.silence_ms(),
            ),
        }
    }

    pub(crate) fn idle_timeout_ms(&self) -> Option<u32> {
        match self {
            Self::ServerVad {
                idle_timeout_ms, ..
            } => *idle_timeout_ms,
            Self::SemanticVad { .. } => None,
        }
    }
}

/// `audio.input.transcription`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct TranscriptionConfig {
    pub(crate) model: Option<String>,
    pub(crate) language: Option<String>,
    pub(crate) prompt: Option<String>,
}

impl TranscriptionConfig {
    fn parse(value: &Value) -> Result<Option<Self>, RealtimeError> {
        match value {
            Value::Null => Ok(None),
            Value::Object(object) => {
                let text = |key: &str| {
                    object
                        .get(key)
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|text| !text.is_empty())
                        .map(str::to_string)
                };
                Ok(Some(Self {
                    model: text("model"),
                    language: text("language").or_else(|| {
                        object
                            .get("languages")
                            .and_then(Value::as_array)
                            .and_then(|languages| languages.first())
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    }),
                    prompt: text("prompt"),
                }))
            }
            _ => Err(RealtimeError::invalid(
                "invalid_type",
                "'transcription' must be an object or null.",
            )
            .with_param("session.audio.input.transcription")),
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "model": self.model,
            "language": self.language,
            "prompt": self.prompt,
        })
    }
}

/// `audio.output.voice`: a voice name, or a custom voice by id (an Izwi
/// saved voice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Voice {
    Named(String),
    Custom(String),
}

impl Voice {
    fn parse(value: &Value) -> Result<Self, RealtimeError> {
        match value {
            Value::String(name) if !name.trim().is_empty() => {
                Ok(Self::Named(name.trim().to_string()))
            }
            Value::Object(object) => object
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.trim().is_empty())
                .map(|id| Self::Custom(id.trim().to_string()))
                .ok_or_else(|| {
                    RealtimeError::invalid("invalid_value", "A custom voice needs an 'id'.")
                        .with_param("session.audio.output.voice")
                }),
            _ => Err(RealtimeError::invalid(
                "invalid_type",
                "'voice' must be a voice name or {\"id\": ...}.",
            )
            .with_param("session.audio.output.voice")),
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        match self {
            Self::Named(name) => json!(name),
            Self::Custom(id) => json!({"id": id}),
        }
    }
}

/// `output_modalities`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Modality {
    Audio,
    Text,
}

impl Modality {
    pub(crate) fn parse(value: &Value, param: &str) -> Result<Self, RealtimeError> {
        let values: Vec<&str> = value
            .as_array()
            .map(|items| items.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        match values.as_slice() {
            ["audio"] => Ok(Self::Audio),
            ["text"] => Ok(Self::Text),
            // The beta's both-at-once means audio with its transcript.
            ["text", "audio"] | ["audio", "text"] => Ok(Self::Audio),
            _ => Err(RealtimeError::invalid(
                "invalid_value",
                "'output_modalities' must be [\"audio\"] or [\"text\"].",
            )
            .with_param(param.to_string())),
        }
    }

    pub(crate) fn to_json(self) -> Value {
        match self {
            Self::Audio => json!(["audio"]),
            Self::Text => json!(["text"]),
        }
    }
}

/// `max_output_tokens`: a number, or `"inf"` (the profile's default).
pub(crate) fn parse_max_output_tokens(
    value: &Value,
    param: &str,
) -> Result<Option<usize>, RealtimeError> {
    match value {
        Value::String(text) if text == "inf" => Ok(None),
        Value::Null => Ok(None),
        value => value
            .as_u64()
            .filter(|tokens| *tokens >= 1)
            .map(|tokens| Some(tokens.min(32_768) as usize))
            .ok_or_else(|| {
                RealtimeError::invalid(
                    "invalid_value",
                    "'max_output_tokens' must be a positive integer or \"inf\".",
                )
                .with_param(param.to_string())
            }),
    }
}

/// The tools of a session or response, validated: function tools, and MCP
/// servers whose tools the server runs itself.
pub(crate) fn parse_tools(value: &Value, param: &str) -> Result<Vec<Value>, RealtimeError> {
    let Some(tools) = value.as_array() else {
        return Err(
            RealtimeError::invalid("invalid_type", "'tools' must be an array.")
                .with_param(param.to_string()),
        );
    };
    tools
        .iter()
        .map(|tool| {
            let kind = tool
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("function");
            if kind == "mcp" {
                return parse_mcp_tool(tool, param);
            }
            if kind != "function" {
                return Err(RealtimeError::invalid(
                    "invalid_value",
                    format!(
                        "Tool type '{kind}' is not supported; only function and mcp tools are."
                    ),
                )
                .with_param(param.to_string()));
            }
            // The chat completions shape nests the function; accept both.
            let function = tool.get("function").unwrap_or(tool);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| {
                    RealtimeError::invalid(
                        "missing_required_parameter",
                        "A function tool needs a 'name'.",
                    )
                    .with_param(param.to_string())
                })?;
            let mut tool = json!({"type": "function", "name": name});
            if let Some(description) = function
                .get("description")
                .filter(|value| value.is_string())
            {
                tool["description"] = description.clone();
            }
            if let Some(parameters) = function.get("parameters").filter(|value| value.is_object()) {
                tool["parameters"] = parameters.clone();
            }
            Ok(tool)
        })
        .collect()
}

/// An MCP server entry: `server_label`, `server_url`, optional `headers` or
/// `authorization` (a bearer token), `allowed_tools`, and
/// `require_approval: "never"` (approvals are not implemented, so a session
/// must say it needs none).
fn parse_mcp_tool(tool: &Value, param: &str) -> Result<Value, RealtimeError> {
    let invalid = |message: &str| {
        Err(RealtimeError::invalid("invalid_value", message.to_string())
            .with_param(param.to_string()))
    };
    let text = |key: &str| {
        tool.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let Some(label) = text("server_label") else {
        return invalid("An mcp tool needs a 'server_label'.");
    };
    if tool
        .get("connector_id")
        .is_some_and(|value| !value.is_null())
    {
        return invalid("MCP connectors are not supported; name the server with 'server_url'.");
    }
    let Some(url) = text("server_url") else {
        return invalid("An mcp tool needs a 'server_url'.");
    };
    if tool.get("require_approval").and_then(Value::as_str) != Some("never") {
        return invalid(
            "MCP tool approvals are not supported: set 'require_approval' to \"never\".",
        );
    }
    let mut headers = serde_json::Map::new();
    if let Some(given) = tool.get("headers").and_then(Value::as_object) {
        for (name, value) in given {
            let Some(value) = value.as_str() else {
                return invalid("MCP 'headers' values must be strings.");
            };
            headers.insert(name.clone(), json!(value));
        }
    }
    if let Some(token) = text("authorization") {
        if headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("authorization"))
        {
            return invalid("Send 'authorization' or an Authorization header, not both.");
        }
        headers.insert(
            "Authorization".to_string(),
            json!(format!("Bearer {token}")),
        );
    }
    let allowed_tools = match tool.get("allowed_tools") {
        None | Some(Value::Null) => Value::Null,
        Some(Value::Array(names)) if names.iter().all(Value::is_string) => json!(names),
        Some(Value::Object(filter))
            if filter
                .get("tool_names")
                .and_then(Value::as_array)
                .is_some_and(|names| names.iter().all(Value::is_string)) =>
        {
            filter["tool_names"].clone()
        }
        Some(_) => return invalid("MCP 'allowed_tools' must be a list of tool names."),
    };
    Ok(json!({
        "type": "mcp",
        "server_label": label,
        "server_url": url,
        "headers": headers,
        "allowed_tools": allowed_tools,
        "require_approval": "never",
    }))
}

/// Whether `tool` is an MCP server entry.
pub(crate) fn is_mcp_tool(tool: &Value) -> bool {
    tool.get("type").and_then(Value::as_str) == Some("mcp")
}

/// A session's tools as it reports them: an MCP server's headers may carry
/// credentials and are left out.
pub(crate) fn tools_echo(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let mut tool = tool.clone();
            if is_mcp_tool(&tool) {
                if let Some(tool) = tool.as_object_mut() {
                    tool.remove("headers");
                }
            }
            tool
        })
        .collect()
}

/// A tool in the chat completions form the chat models' templates take.
pub(crate) fn chat_tool(tool: &Value) -> Value {
    let mut function = serde_json::Map::new();
    function.insert("name".to_string(), tool["name"].clone());
    // A tool without a description has none in the prompt either.
    if let Some(description) = tool.get("description").filter(|value| !value.is_null()) {
        function.insert("description".to_string(), description.clone());
    }
    function.insert(
        "parameters".to_string(),
        tool.get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    );
    json!({"type": "function", "function": function})
}

/// A response's `metadata`: at most 16 string pairs.
pub(crate) fn parse_metadata(value: &Value) -> Result<Value, RealtimeError> {
    match value {
        Value::Null => Ok(Value::Null),
        Value::Object(pairs)
            if pairs.len() <= MAX_METADATA_PAIRS
                && pairs
                    .iter()
                    .all(|(key, value)| key.len() <= 64 && value.as_str().is_some_and(|text| text.len() <= 512)) =>
        {
            Ok(value.clone())
        }
        _ => Err(RealtimeError::invalid(
            "invalid_value",
            "'metadata' must hold at most 16 string pairs (keys up to 64 characters, values up to 512).",
        )
        .with_param("response.metadata")),
    }
}

/// The session's configuration.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SessionConfig {
    pub(crate) id: String,
    /// The `model` the client asked for, echoed back.
    pub(crate) model: String,
    pub(crate) modality: Modality,
    pub(crate) instructions: String,
    pub(crate) input_format: AudioFormat,
    pub(crate) output_format: AudioFormat,
    pub(crate) noise_reduction: Value,
    pub(crate) transcription: Option<TranscriptionConfig>,
    pub(crate) turn_detection: Option<TurnDetection>,
    pub(crate) voice: Voice,
    pub(crate) speed: f64,
    pub(crate) tools: Vec<Value>,
    pub(crate) tool_choice: Value,
    pub(crate) max_output_tokens: Option<usize>,
    pub(crate) truncation: Value,
    pub(crate) tracing: Value,
    pub(crate) include: Value,
    pub(crate) expires_at: u64,
}

/// What a `session.update` changed that the session must act on, and the
/// requests it ignored (logged, never an `error`: D8).
#[derive(Debug, Default, PartialEq)]
pub(crate) struct UpdateOutcome {
    pub(crate) input_changed: bool,
    pub(crate) ignored: Vec<String>,
}

impl SessionConfig {
    pub(crate) fn new(
        id: String,
        model: String,
        voice: String,
        instructions: String,
        expires_at: u64,
    ) -> Self {
        Self {
            id,
            model,
            modality: Modality::Audio,
            instructions,
            input_format: AudioFormat::Pcm24k,
            output_format: AudioFormat::Pcm24k,
            noise_reduction: Value::Null,
            transcription: None,
            turn_detection: Some(TurnDetection::default()),
            voice: Voice::Named(voice),
            speed: 1.0,
            tools: Vec::new(),
            tool_choice: json!("auto"),
            max_output_tokens: None,
            truncation: json!("auto"),
            tracing: Value::Null,
            include: Value::Null,
            expires_at,
        }
    }

    /// Merge one `session.update`. Validation is all-or-nothing: an invalid
    /// field leaves the session as it was. `voice_locked` once the session
    /// has spoken (OpenAI fixes the voice then).
    pub(crate) fn apply_update(
        &mut self,
        update: &Value,
        voice_locked: bool,
    ) -> Result<UpdateOutcome, RealtimeError> {
        let Value::Object(update) = update else {
            return Err(
                RealtimeError::invalid("invalid_type", "'session' must be an object.")
                    .with_param("session"),
            );
        };
        let beta_fields = [
            "input_audio_format",
            "output_audio_format",
            "modalities",
            "input_audio_transcription",
        ];
        if !update.contains_key("type")
            && beta_fields.iter().any(|field| update.contains_key(*field))
        {
            return Err(RealtimeError::invalid(
                "unsupported_beta_session",
                "This server speaks the GA Realtime protocol: nest audio settings under \
                 'session.audio' and set 'session.type' (the beta interface was retired).",
            )
            .with_param("session"));
        }
        match update.get("type").and_then(Value::as_str) {
            None | Some("realtime") => {}
            Some(other) => {
                return Err(RealtimeError::invalid(
                    "invalid_value",
                    format!(
                        "This is a realtime session; it cannot become '{other}'. Connect with \
                         ?intent=transcription for a transcription session."
                    ),
                )
                .with_param("session.type"))
            }
        }

        let mut next = self.clone();
        let mut outcome = UpdateOutcome::default();
        if let Some(model) = update.get("model").and_then(Value::as_str) {
            if model != self.model && !model.trim().is_empty() {
                outcome.ignored.push(format!(
                    "model `{model}` (the session's model is fixed at connect)"
                ));
            }
        }
        if let Some(value) = update.get("output_modalities") {
            next.modality = Modality::parse(value, "session.output_modalities")?;
        }
        if let Some(value) = update.get("instructions") {
            next.instructions = value.as_str().unwrap_or_default().to_string();
        }
        if let Some(value) = update.get("tools") {
            next.tools = parse_tools(value, "session.tools")?;
        }
        if let Some(value) = update.get("tool_choice") {
            next.tool_choice = value.clone();
        }
        if let Some(value) = update.get("max_output_tokens") {
            next.max_output_tokens = parse_max_output_tokens(value, "session.max_output_tokens")?;
        }
        if let Some(value) = update.get("truncation") {
            next.truncation = value.clone();
        }
        if let Some(value) = update.get("tracing") {
            next.tracing = value.clone();
        }
        if let Some(value) = update.get("include") {
            next.include = value.clone();
        }
        for unsupported in ["reasoning", "parallel_tool_calls", "prompt"] {
            if update
                .get(unsupported)
                .is_some_and(|value| !value.is_null())
            {
                outcome.ignored.push(format!("`{unsupported}`"));
            }
        }

        let audio = update.get("audio");
        if let Some(input) = audio.and_then(|audio| audio.get("input")) {
            if let Some(format) = input.get("format") {
                next.input_format = AudioFormat::parse(format, "session.audio.input.format")?;
            }
            if let Some(value) = input.get("noise_reduction") {
                next.noise_reduction = value.clone();
                if !value.is_null() {
                    outcome.ignored.push("`noise_reduction`".to_string());
                }
            }
            if let Some(value) = input.get("transcription") {
                next.transcription = TranscriptionConfig::parse(value)?;
            }
            if let Some(value) = input.get("turn_detection") {
                next.turn_detection = TurnDetection::parse(value)?;
            }
        }
        if let Some(output) = audio.and_then(|audio| audio.get("output")) {
            if let Some(format) = output.get("format") {
                next.output_format = AudioFormat::parse(format, "session.audio.output.format")?;
            }
            if let Some(value) = output.get("voice") {
                let voice = Voice::parse(value)?;
                if voice_locked && voice != self.voice {
                    outcome
                        .ignored
                        .push("a voice change after the session has spoken".to_string());
                } else {
                    next.voice = voice;
                }
            }
            if let Some(value) = output.get("speed") {
                next.speed = value
                    .as_f64()
                    .filter(|speed| (0.25..=1.5).contains(speed))
                    .ok_or_else(|| {
                        RealtimeError::invalid("invalid_value", "'speed' must be from 0.25 to 1.5.")
                            .with_param("session.audio.output.speed")
                    })?;
                if (next.speed - 1.0).abs() > f64::EPSILON {
                    outcome
                        .ignored
                        .push("`speed` (speech is generated at 1.0)".to_string());
                }
            }
        }
        outcome.input_changed =
            next.input_format != self.input_format || next.turn_detection != self.turn_detection;
        *self = next;
        Ok(outcome)
    }

    /// The full session object of `session.created` and `session.updated`.
    pub(crate) fn to_json(&self) -> Value {
        json!({
            "type": "realtime",
            "object": "realtime.session",
            "id": self.id,
            "model": self.model,
            "output_modalities": self.modality.to_json(),
            "instructions": self.instructions,
            "tools": tools_echo(&self.tools),
            "tool_choice": self.tool_choice,
            "max_output_tokens": self.max_output_tokens.map_or_else(|| json!("inf"), |tokens| json!(tokens)),
            "tracing": self.tracing,
            "truncation": self.truncation,
            "prompt": null,
            "include": self.include,
            "expires_at": self.expires_at,
            "audio": {
                "input": {
                    "format": self.input_format.to_json(),
                    "transcription": self.transcription.as_ref().map(TranscriptionConfig::to_json),
                    "noise_reduction": self.noise_reduction,
                    "turn_detection": self.turn_detection.as_ref().map(TurnDetection::to_json),
                },
                "output": {
                    "format": self.output_format.to_json(),
                    "voice": self.voice.to_json(),
                    "speed": self.speed,
                },
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> SessionConfig {
        SessionConfig::new(
            "sess_1".to_string(),
            "gpt-realtime".to_string(),
            "marin".to_string(),
            "Be brief.".to_string(),
            1_000,
        )
    }

    #[test]
    fn defaults_follow_openai_and_echo_a_pipecat_valid_session() {
        let echo = session().to_json();
        assert_eq!(echo["type"], "realtime");
        assert_eq!(echo["object"], "realtime.session");
        assert_eq!(
            echo["audio"]["input"]["format"],
            json!({"type": "audio/pcm", "rate": 24000})
        );
        assert_eq!(
            echo["audio"]["input"]["turn_detection"]["type"],
            "server_vad"
        );
        assert_eq!(
            echo["audio"]["input"]["turn_detection"]["silence_duration_ms"],
            500
        );
        assert_eq!(
            echo["audio"]["input"]["turn_detection"]["prefix_padding_ms"],
            300
        );
        assert_eq!(echo["audio"]["output"]["voice"], "marin");
        assert_eq!(echo["tool_choice"], "auto");
        assert_eq!(echo["max_output_tokens"], "inf");
    }

    #[test]
    fn livekit_style_partial_updates_merge() {
        let mut config = session();
        config
            .apply_update(
                &json!({
                    "type": "realtime",
                    "model": "gpt-realtime",
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "gpt-4o-mini-transcribe"},
                            "turn_detection": {"type": "semantic_vad", "eagerness": "medium",
                                               "create_response": true, "interrupt_response": true},
                        },
                        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin", "speed": 1.0},
                    },
                    "max_output_tokens": "inf",
                    "tool_choice": "auto",
                    "tracing": null,
                }),
                false,
            )
            .expect("full update");
        let outcome = config
            .apply_update(
                &json!({"type": "realtime", "instructions": "Talk like a pirate."}),
                false,
            )
            .expect("instructions");
        assert!(!outcome.input_changed);
        config
            .apply_update(
                &json!({"type": "realtime", "tools": [{"type": "function", "name": "lookup",
                        "description": "Look up", "parameters": {"type": "object", "properties": {}}}]}),
                false,
            )
            .expect("tools");
        assert_eq!(config.instructions, "Talk like a pirate.");
        assert_eq!(config.tools.len(), 1);
        assert!(matches!(
            config.turn_detection,
            Some(TurnDetection::SemanticVad {
                eagerness: Eagerness::Medium,
                ..
            })
        ));
        let echo = config.to_json();
        assert_eq!(
            echo["audio"]["input"]["transcription"],
            json!({"model": "gpt-4o-mini-transcribe", "language": null, "prompt": null})
        );
        // `null` turns VAD off; absent leaves it alone.
        config
            .apply_update(
                &json!({"audio": {"input": {"turn_detection": null}}}),
                false,
            )
            .expect("manual");
        assert_eq!(config.turn_detection, None);
        config
            .apply_update(&json!({"instructions": "x"}), false)
            .expect("no type");
        assert_eq!(config.turn_detection, None);
    }

    #[test]
    fn invalid_updates_change_nothing_and_benign_conflicts_are_ignored() {
        let mut config = session();
        let before = config.clone();
        let err = config
            .apply_update(
                &json!({"instructions": "new", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}),
                false,
            )
            .expect_err("16 kHz");
        assert_eq!(err.param.as_deref(), Some("session.audio.input.format"));
        assert_eq!(config, before, "all or nothing");
        let err = config
            .apply_update(
                &json!({"input_audio_format": "pcm16", "modalities": ["text", "audio"]}),
                false,
            )
            .expect_err("beta shape");
        assert_eq!(err.code.as_deref(), Some("unsupported_beta_session"));
        assert!(config
            .apply_update(
                &json!({"tools": [{"type": "mcp", "server_label": "x"}]}),
                false
            )
            .is_err());

        let outcome = config
            .apply_update(
                &json!({"model": "gpt-realtime-2.1", "audio": {"output": {"voice": "cedar"}}}),
                true,
            )
            .expect("benign");
        assert_eq!(outcome.ignored.len(), 2, "{:?}", outcome.ignored);
        assert_eq!(config.voice, Voice::Named("marin".to_string()));
        assert_eq!(config.model, "gpt-realtime");
    }

    #[test]
    fn eagerness_picks_the_silence_window() {
        assert!(Eagerness::High.silence_ms() < Eagerness::Medium.silence_ms());
        assert!(Eagerness::Medium.silence_ms() < Eagerness::Low.silence_ms());
        assert_eq!(Eagerness::Auto.silence_ms(), Eagerness::Medium.silence_ms());
    }

    #[test]
    fn tools_metadata_and_token_limits_validate() {
        let tools = parse_tools(
            &json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]),
            "tools",
        )
        .expect("chat completions shape");
        assert_eq!(tools[0]["name"], "f");
        assert_eq!(chat_tool(&tools[0])["function"]["name"], "f");
        assert!(parse_metadata(&json!({"topic": "classification"})).is_ok());
        assert!(parse_metadata(&json!({"n": 1})).is_err());
        assert_eq!(
            parse_max_output_tokens(&json!("inf"), "m").expect("inf"),
            None
        );
        assert_eq!(
            parse_max_output_tokens(&json!(64), "m").expect("64"),
            Some(64)
        );
        assert!(parse_max_output_tokens(&json!(0), "m").is_err());
    }

    #[test]
    fn mcp_servers_are_accepted_without_approvals_and_echoed_without_credentials() {
        let tool = json!({
            "type": "mcp", "server_label": "docs", "server_url": "https://mcp.example/mcp",
            "authorization": "secret-token", "allowed_tools": {"tool_names": ["search"]},
            "require_approval": "never",
        });
        let tools = parse_tools(&json!([tool]), "session.tools").unwrap();
        assert!(is_mcp_tool(&tools[0]));
        assert_eq!(tools[0]["headers"]["Authorization"], "Bearer secret-token");
        assert_eq!(tools[0]["allowed_tools"], json!(["search"]));
        let echo = tools_echo(&tools);
        assert!(echo[0].get("headers").is_none());
        assert!(!echo[0].to_string().contains("secret-token"));

        let without = |key: &str| {
            let mut tool = tool.clone();
            tool.as_object_mut().unwrap().remove(key);
            parse_tools(&json!([tool]), "session.tools")
        };
        assert!(without("server_url").is_err());
        assert!(without("server_label").is_err());
        // Omitting it would mean approvals, which are not implemented.
        let err = without("require_approval").unwrap_err();
        assert!(err.message.contains("require_approval"), "{}", err.message);
        let mut both = tool.clone();
        both["headers"] = json!({"authorization": "Bearer other"});
        assert!(parse_tools(&json!([both]), "session.tools").is_err());
        let mut connector = tool.clone();
        connector["connector_id"] = json!("connector_x");
        assert!(parse_tools(&json!([connector]), "session.tools").is_err());
    }
}
