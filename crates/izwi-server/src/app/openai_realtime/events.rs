//! OpenAI Realtime (GA) wire events: client events parsed leniently, server
//! events built with OpenAI-shaped ids, and the `error` event.
//!
//! Parsing follows the GA client event union (`session.update`,
//! `input_audio_buffer.*`, `conversation.item.*`, `response.*`,
//! `output_audio_buffer.clear`). Unknown fields are ignored and `event_id` is
//! optional, since stock clients (openai-python, LiveKit) omit it.

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Map, Value};

/// Ids in OpenAI's style: a prefix, `_`, and 21 base62 characters. Each
/// session draws them from its own random nonce and a counter, so ids never
/// repeat within a session and stay at most 26 characters (LiveKit keeps
/// ids up to 32 on the wire).
#[derive(Debug)]
pub(crate) struct Ids {
    nonce: u64,
    counter: AtomicU64,
}

const BASE62: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

fn base62(mut value: u64, width: usize) -> String {
    let mut out = vec![b'0'; width];
    for slot in out.iter_mut().rev() {
        *slot = BASE62[(value % 62) as usize];
        value /= 62;
    }
    String::from_utf8(out).unwrap_or_default()
}

impl Ids {
    pub(crate) fn new() -> Self {
        let nonce = uuid::Uuid::new_v4().as_u64_pair().0;
        Self::with_nonce(nonce)
    }

    pub(crate) fn with_nonce(nonce: u64) -> Self {
        Self {
            nonce,
            counter: AtomicU64::new(0),
        }
    }

    /// The next id with `prefix` (`event`, `item`, `resp`, `sess`, `conv`,
    /// `call`).
    pub(crate) fn next(&self, prefix: &str) -> String {
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}_{}{}", base62(self.nonce, 11), base62(count, 10))
    }
}

/// An OpenAI Realtime `error`: `invalid_request_error` for the client's
/// mistakes, `server_error` for ours.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealtimeError {
    pub(crate) kind: &'static str,
    pub(crate) code: Option<String>,
    pub(crate) message: String,
    pub(crate) param: Option<String>,
    pub(crate) event_id: Option<String>,
}

impl RealtimeError {
    pub(crate) fn invalid(code: &str, message: impl Into<String>) -> Self {
        Self {
            kind: "invalid_request_error",
            code: Some(code.to_string()),
            message: message.into(),
            param: None,
            event_id: None,
        }
    }

    pub(crate) fn server(message: impl Into<String>) -> Self {
        Self {
            kind: "server_error",
            code: None,
            message: message.into(),
            param: None,
            event_id: None,
        }
    }

    pub(crate) fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    pub(crate) fn with_event_id(mut self, event_id: Option<String>) -> Self {
        if self.event_id.is_none() {
            self.event_id = event_id;
        }
        self
    }

    /// The `error` server event.
    pub(crate) fn to_event(&self, ids: &Ids) -> Value {
        server_event(
            ids,
            "error",
            json!({
                "error": {
                    "type": self.kind,
                    "code": self.code,
                    "message": self.message,
                    "param": self.param,
                    "event_id": self.event_id,
                }
            }),
        )
    }
}

/// A server event: `type`, a fresh `event_id`, then `fields`.
pub(crate) fn server_event(ids: &Ids, event_type: &str, fields: Value) -> Value {
    let mut event = Map::new();
    event.insert("type".to_string(), Value::String(event_type.to_string()));
    event.insert("event_id".to_string(), Value::String(ids.next("event")));
    if let Value::Object(fields) = fields {
        event.extend(fields);
    }
    Value::Object(event)
}

/// One client event.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ClientEvent {
    SessionUpdate {
        session: Value,
    },
    InputAudioAppend {
        audio: String,
    },
    InputAudioCommit,
    InputAudioClear,
    ItemCreate {
        previous_item_id: Option<String>,
        item: Value,
    },
    ItemRetrieve {
        item_id: String,
    },
    ItemTruncate {
        item_id: String,
        content_index: u64,
        audio_end_ms: i64,
    },
    ItemDelete {
        item_id: String,
    },
    ResponseCreate {
        response: Value,
    },
    ResponseCancel {
        response_id: Option<String>,
    },
    OutputAudioBufferClear,
}

/// A parsed client event and the `event_id` the client gave it, if any.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParsedClientEvent {
    pub(crate) event_id: Option<String>,
    pub(crate) event: ClientEvent,
}

fn required_str(object: &Map<String, Value>, key: &str) -> Result<String, RealtimeError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            RealtimeError::invalid(
                "missing_required_parameter",
                format!("Missing required parameter: '{key}'."),
            )
            .with_param(key)
        })
}

fn optional_str(object: &Map<String, Value>, key: &str) -> Option<String> {
    object.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Parse one client message.
pub(crate) fn parse_client_event(text: &str) -> Result<ParsedClientEvent, RealtimeError> {
    let value: Value = serde_json::from_str(text).map_err(|err| {
        RealtimeError::invalid(
            "invalid_json",
            format!("The event is not valid JSON: {err}"),
        )
    })?;
    let Value::Object(object) = value else {
        return Err(RealtimeError::invalid(
            "invalid_event",
            "The event must be a JSON object.",
        ));
    };
    let event_id = optional_str(&object, "event_id");
    let fail = |err: RealtimeError| err.with_event_id(event_id.clone());
    let Some(event_type) = object.get("type").and_then(Value::as_str) else {
        return Err(fail(RealtimeError::invalid(
            "invalid_event",
            "The 'type' field is missing.",
        )));
    };
    let event = match event_type {
        "session.update" => ClientEvent::SessionUpdate {
            session: match object.get("session") {
                Some(session @ Value::Object(_)) => session.clone(),
                _ => {
                    return Err(fail(
                        RealtimeError::invalid(
                            "missing_required_parameter",
                            "Missing required parameter: 'session'.",
                        )
                        .with_param("session"),
                    ))
                }
            },
        },
        "input_audio_buffer.append" => ClientEvent::InputAudioAppend {
            audio: required_str(&object, "audio").map_err(fail)?,
        },
        "input_audio_buffer.commit" => ClientEvent::InputAudioCommit,
        "input_audio_buffer.clear" => ClientEvent::InputAudioClear,
        "conversation.item.create" => ClientEvent::ItemCreate {
            previous_item_id: optional_str(&object, "previous_item_id"),
            item: match object.get("item") {
                Some(item @ Value::Object(_)) => item.clone(),
                _ => {
                    return Err(fail(
                        RealtimeError::invalid(
                            "missing_required_parameter",
                            "Missing required parameter: 'item'.",
                        )
                        .with_param("item"),
                    ))
                }
            },
        },
        "conversation.item.retrieve" => ClientEvent::ItemRetrieve {
            item_id: required_str(&object, "item_id").map_err(fail)?,
        },
        "conversation.item.truncate" => ClientEvent::ItemTruncate {
            item_id: required_str(&object, "item_id").map_err(fail)?,
            content_index: object
                .get("content_index")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            audio_end_ms: object
                .get("audio_end_ms")
                .and_then(|value| {
                    value
                        .as_i64()
                        .or_else(|| value.as_f64().map(|ms| ms as i64))
                })
                .ok_or_else(|| {
                    fail(
                        RealtimeError::invalid(
                            "missing_required_parameter",
                            "Missing required parameter: 'audio_end_ms'.",
                        )
                        .with_param("audio_end_ms"),
                    )
                })?,
        },
        "conversation.item.delete" => ClientEvent::ItemDelete {
            item_id: required_str(&object, "item_id").map_err(fail)?,
        },
        "response.create" => ClientEvent::ResponseCreate {
            response: match object.get("response") {
                None | Some(Value::Null) => json!({}),
                Some(response @ Value::Object(_)) => response.clone(),
                Some(_) => {
                    return Err(fail(
                        RealtimeError::invalid(
                            "invalid_type",
                            "Invalid type for 'response': expected an object.",
                        )
                        .with_param("response"),
                    ))
                }
            },
        },
        "response.cancel" => ClientEvent::ResponseCancel {
            response_id: optional_str(&object, "response_id"),
        },
        "output_audio_buffer.clear" => ClientEvent::OutputAudioBufferClear,
        other => {
            return Err(fail(
                RealtimeError::invalid("invalid_value", format!("Invalid value: '{other}'."))
                    .with_param("type"),
            ))
        }
    };
    Ok(ParsedClientEvent { event_id, event })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_openai_shaped_and_unique() {
        let ids = Ids::with_nonce(42);
        let first = ids.next("item");
        let second = ids.next("item");
        assert!(first.starts_with("item_"));
        assert_eq!(first.len(), "item_".len() + 21);
        assert_ne!(first, second);
        assert!(ids.next("event").len() <= 32);
        assert!(first[5..].bytes().all(|byte| byte.is_ascii_alphanumeric()));
    }

    #[test]
    fn client_events_parse_without_event_ids_and_ignore_unknown_fields() {
        let parsed =
            parse_client_event(r#"{"type":"input_audio_buffer.append","audio":"AAA=","x":1}"#)
                .expect("append");
        assert_eq!(parsed.event_id, None);
        assert_eq!(
            parsed.event,
            ClientEvent::InputAudioAppend {
                audio: "AAA=".to_string()
            }
        );
        let parsed = parse_client_event(
            r#"{"type":"conversation.item.truncate","event_id":"e1","item_id":"item_1","content_index":0,"audio_end_ms":1500}"#,
        )
        .expect("truncate");
        assert_eq!(parsed.event_id.as_deref(), Some("e1"));
        assert_eq!(
            parsed.event,
            ClientEvent::ItemTruncate {
                item_id: "item_1".to_string(),
                content_index: 0,
                audio_end_ms: 1500
            }
        );
        // A bare `response.create` (Pipecat's manual mode) has no `response`.
        assert_eq!(
            parse_client_event(r#"{"type":"response.create"}"#)
                .expect("create")
                .event,
            ClientEvent::ResponseCreate {
                response: json!({})
            }
        );
    }

    #[test]
    fn bad_client_events_are_invalid_request_errors_echoing_the_event_id() {
        let err = parse_client_event(r#"{"type":"scooby.dooby.doo","event_id":"mine"}"#)
            .expect_err("unknown type");
        assert_eq!(err.kind, "invalid_request_error");
        assert_eq!(err.code.as_deref(), Some("invalid_value"));
        assert_eq!(err.param.as_deref(), Some("type"));
        assert_eq!(err.event_id.as_deref(), Some("mine"));
        let err = parse_client_event(r#"{"event_id":"x"}"#).expect_err("no type");
        assert_eq!(err.code.as_deref(), Some("invalid_event"));
        let err =
            parse_client_event(r#"{"type":"conversation.item.delete"}"#).expect_err("no item id");
        assert_eq!(err.param.as_deref(), Some("item_id"));
        let event = err.to_event(&Ids::with_nonce(1));
        assert_eq!(event["type"], "error");
        assert_eq!(event["error"]["type"], "invalid_request_error");
        assert!(event["event_id"]
            .as_str()
            .expect("id")
            .starts_with("event_"));
    }
}
