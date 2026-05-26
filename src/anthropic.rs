//! Anthropic Messages API request/response types and SSE event parser.
//!
//! iter-3 surface: pure data types and a single-line SSE parser. The
//! live HTTP client (reqwest + stream wiring) lands iter-4 once these
//! types are stable and round-trip cleanly under serde.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages>. We model
//! only the streaming-message shape used by `wmd`'s conversation loop;
//! tool-use and citations are deferred.

use serde::{Deserialize, Serialize};

/// Anthropic-API role labels. The Messages API accepts only `user` and
/// `assistant` in the message list (`system` rides on a top-level field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// User-authored turn.
    User,
    /// Assistant-authored turn.
    Assistant,
}

/// One conversation turn. iter-3 keeps content as a plain string; the
/// block-array form (`text` / `tool_use` / `tool_result`) lands once
/// tool-use support is on the roadmap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Role of the speaker.
    pub role: Role,
    /// Textual content of the turn.
    pub content: String,
}

/// Top-level body sent in the `POST /v1/messages` request. iter-3
/// covers the fields `wmd` populates; tool definitions and metadata
/// land in iter-4+.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MessageRequest {
    /// Canonical model id (e.g. `claude-sonnet-4-6`). Resolve via
    /// [`crate::canonical_model`] before constructing.
    pub model: String,
    /// Maximum tokens the response may emit. Anthropic requires this.
    pub max_tokens: u32,
    /// Optional top-level system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Conversation history, oldest first.
    pub messages: Vec<Message>,
    /// Request streamed delta events.
    pub stream: bool,
}

impl MessageRequest {
    /// Build a streaming request for the given model + message list.
    #[must_use]
    pub fn streaming(model: impl Into<String>, max_tokens: u32, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            max_tokens,
            system: None,
            messages,
            stream: true,
        }
    }

    /// Set the top-level system prompt and return `self`.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }
}

/// One server-sent event from the Messages stream.
///
/// We model the events the conversation loop actually consumes; ping
/// and control events are folded into [`Self::Other`] so unknown
/// variants stay forward-compat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// `message_start` — opening event carrying the model id + usage
    /// counters (the body is dropped on the floor in iter-3; iter-4
    /// surfaces usage telemetry).
    MessageStart,
    /// `content_block_start` — a new content block opened. We only
    /// track text blocks for now.
    ContentBlockStart {
        /// Zero-based block index inside the response.
        index: u32,
    },
    /// `content_block_delta` — incremental text delta on an open block.
    TextDelta {
        /// Index of the block this delta extends.
        index: u32,
        /// Newly emitted text fragment.
        text: String,
    },
    /// `content_block_stop` — the named block is complete.
    ContentBlockStop {
        /// Block index that closed.
        index: u32,
    },
    /// `message_delta` — top-level message metadata update (usually
    /// `stop_reason`). Body parsed lazily in iter-4.
    MessageDelta {
        /// Reason the model stopped, when present.
        stop_reason: Option<String>,
    },
    /// `message_stop` — terminal event closing the stream.
    MessageStop,
    /// `error` — the API streamed an error event (vs. a non-2xx HTTP
    /// status, which the transport layer surfaces separately).
    Error {
        /// Anthropic error type id (e.g. `overloaded_error`).
        kind: String,
        /// Human-readable message attached to the event.
        message: String,
    },
    /// `ping` and any event variant we don't recognise. Kept opaque so
    /// new event types don't break the parser.
    Other {
        /// Raw event-type label from the SSE `event:` line.
        event_type: String,
    },
}

/// Errors raised while parsing a single SSE `data:` payload.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// The `data:` payload was not valid JSON.
    #[error("invalid json in sse payload: {0}")]
    Json(#[from] serde_json::Error),
    /// JSON parsed but lacked the required `type` field.
    #[error("sse payload missing 'type' field")]
    MissingType,
    /// A typed field was present but in an unexpected shape.
    #[error("malformed sse event {kind:?}: {reason}")]
    Shape {
        /// The `type` value of the offending event.
        kind: String,
        /// What was wrong with it.
        reason: String,
    },
}

/// Parse a single Anthropic SSE `data:` payload into a [`StreamEvent`].
///
/// Callers strip the leading `data: ` prefix and pass the JSON body.
/// Empty lines, comments, and `event:` lines are handled by the
/// stream-framing layer in iter-4 and are not this function's concern.
///
/// # Errors
/// Returns [`ParseError::Json`] when `data` is not valid JSON,
/// [`ParseError::MissingType`] when the `type` field is absent, and
/// [`ParseError::Shape`] when a required nested field is missing or has
/// the wrong type for the declared event kind.
pub fn parse_sse_event(data: &str) -> Result<StreamEvent, ParseError> {
    let v: serde_json::Value = serde_json::from_str(data)?;
    let kind = v
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or(ParseError::MissingType)?;

    match kind {
        "message_start" => Ok(StreamEvent::MessageStart),
        "content_block_start" => {
            let index = require_index(&v, "content_block_start")?;
            Ok(StreamEvent::ContentBlockStart { index })
        }
        "content_block_delta" => {
            let index = require_index(&v, "content_block_delta")?;
            let delta = v.get("delta").ok_or_else(|| ParseError::Shape {
                kind: "content_block_delta".to_string(),
                reason: "missing 'delta' object".to_string(),
            })?;
            let delta_type = delta
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if delta_type == "text_delta" {
                let text = delta
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| ParseError::Shape {
                        kind: "content_block_delta".to_string(),
                        reason: "text_delta missing 'text' string".to_string(),
                    })?
                    .to_string();
                Ok(StreamEvent::TextDelta { index, text })
            } else {
                Ok(StreamEvent::Other {
                    event_type: format!("content_block_delta:{delta_type}"),
                })
            }
        }
        "content_block_stop" => {
            let index = require_index(&v, "content_block_stop")?;
            Ok(StreamEvent::ContentBlockStop { index })
        }
        "message_delta" => {
            let stop_reason = v
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            Ok(StreamEvent::MessageDelta { stop_reason })
        }
        "message_stop" => Ok(StreamEvent::MessageStop),
        "error" => {
            let err = v.get("error").ok_or_else(|| ParseError::Shape {
                kind: "error".to_string(),
                reason: "missing 'error' object".to_string(),
            })?;
            let kind_str = err
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let message = err
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(StreamEvent::Error {
                kind: kind_str,
                message,
            })
        }
        other => Ok(StreamEvent::Other {
            event_type: other.to_string(),
        }),
    }
}

fn require_index(v: &serde_json::Value, kind: &str) -> Result<u32, ParseError> {
    let raw = v.get("index").and_then(serde_json::Value::as_u64).ok_or_else(|| ParseError::Shape {
        kind: kind.to_string(),
        reason: "missing or non-integer 'index'".to_string(),
    })?;
    u32::try_from(raw).map_err(|_| ParseError::Shape {
        kind: kind.to_string(),
        reason: format!("'index' {raw} exceeds u32"),
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::indexing_slicing,
    reason = "tests"
)]
mod tests {
    use super::*;

    #[test]
    fn message_serializes_with_lowercase_role() {
        let m = Message {
            role: Role::User,
            content: "hello".to_string(),
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"role\":\"user\""));
        assert!(s.contains("\"content\":\"hello\""));
    }

    #[test]
    fn streaming_request_serializes_with_stream_true() {
        let req = MessageRequest::streaming(
            "claude-sonnet-4-6",
            1024,
            vec![Message {
                role: Role::User,
                content: "ping".to_string(),
            }],
        );
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "claude-sonnet-4-6");
        assert_eq!(v["max_tokens"], 1024);
        assert_eq!(v["stream"], true);
        // system is None — must be omitted, not null.
        assert!(v.get("system").is_none());
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "ping");
    }

    #[test]
    fn with_system_sets_top_level_system() {
        let req = MessageRequest::streaming("claude-opus-4-7", 256, vec![])
            .with_system("You are Wintermute.");
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["system"], "You are Wintermute.");
    }

    #[test]
    fn parses_message_start() {
        let raw = r#"{"type":"message_start","message":{"id":"msg_x"}}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(ev, StreamEvent::MessageStart);
    }

    #[test]
    fn parses_content_block_start() {
        let raw = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(ev, StreamEvent::ContentBlockStart { index: 0 });
    }

    #[test]
    fn parses_text_delta() {
        let raw = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(
            ev,
            StreamEvent::TextDelta {
                index: 0,
                text: "Hi".to_string(),
            }
        );
    }

    #[test]
    fn non_text_block_delta_is_other() {
        let raw = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert!(matches!(ev, StreamEvent::Other { ref event_type } if event_type == "content_block_delta:input_json_delta"));
    }

    #[test]
    fn parses_content_block_stop() {
        let raw = r#"{"type":"content_block_stop","index":0}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(ev, StreamEvent::ContentBlockStop { index: 0 });
    }

    #[test]
    fn parses_message_delta_with_stop_reason() {
        let raw =
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(
            ev,
            StreamEvent::MessageDelta {
                stop_reason: Some("end_turn".to_string()),
            }
        );
    }

    #[test]
    fn parses_message_delta_without_stop_reason() {
        let raw = r#"{"type":"message_delta","delta":{}}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(ev, StreamEvent::MessageDelta { stop_reason: None });
    }

    #[test]
    fn parses_message_stop() {
        let raw = r#"{"type":"message_stop"}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(ev, StreamEvent::MessageStop);
    }

    #[test]
    fn parses_error_event() {
        let raw = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(
            ev,
            StreamEvent::Error {
                kind: "overloaded_error".to_string(),
                message: "slow down".to_string(),
            }
        );
    }

    #[test]
    fn unknown_event_type_is_other() {
        let raw = r#"{"type":"ping"}"#;
        let ev = parse_sse_event(raw).unwrap();
        assert_eq!(
            ev,
            StreamEvent::Other {
                event_type: "ping".to_string(),
            }
        );
    }

    #[test]
    fn missing_type_field_errors() {
        let raw = r#"{"index":0}"#;
        assert!(matches!(parse_sse_event(raw), Err(ParseError::MissingType)));
    }

    #[test]
    fn invalid_json_errors() {
        let raw = "not-json";
        assert!(matches!(parse_sse_event(raw), Err(ParseError::Json(_))));
    }

    #[test]
    fn content_block_start_without_index_errors() {
        let raw = r#"{"type":"content_block_start","content_block":{}}"#;
        let err = parse_sse_event(raw).unwrap_err();
        assert!(matches!(err, ParseError::Shape { ref kind, .. } if kind == "content_block_start"));
    }
}
