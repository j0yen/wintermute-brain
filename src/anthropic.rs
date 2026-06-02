//! Anthropic Messages API request/response types and SSE event parser.
//!
//! iter-3 surface: pure data types and a single-line SSE parser. The
//! live HTTP client (reqwest + stream wiring) lands iter-4 once these
//! types are stable and round-trip cleanly under serde.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages>. We model
//! only the streaming-message shape used by `wmd`'s conversation loop;
//! tool-use and citations are deferred.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;
use serde::{Deserialize, Serialize};

/// Cache-control type understood by the Anthropic API.
///
/// Currently only `ephemeral` (5-minute TTL) is supported by the API.
/// Carrying this as a typed enum avoids stringly-typed mistakes in call
/// sites and makes the serialized `{"type":"ephemeral"}` shape explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheType {
    /// Cache the prefix for up to 5 minutes (Anthropic ephemeral TTL).
    Ephemeral,
}

/// `cache_control` object attached to a system or message block.
///
/// Serializes as `{"type":"ephemeral"}` (or the appropriate variant).
/// Adding a breakpoint on the *last stable block* tells Anthropic to
/// cache everything up to and including that block as a reusable prefix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    /// The kind of caching to apply.
    #[serde(rename = "type")]
    pub kind: CacheType,
}

impl CacheControl {
    /// Convenience constructor for the common ephemeral breakpoint.
    #[must_use]
    pub const fn ephemeral() -> Self {
        Self { kind: CacheType::Ephemeral }
    }
}

/// One block in the `system` content-block array form.
///
/// The API accepts `system` as either a plain `String` or an array of
/// these objects.  We model only the `text` block type (the only variant
/// the brain uses); tool-result and image blocks are not needed here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemBlock {
    /// A plain-text block, optionally carrying a cache breakpoint.
    Text {
        /// The text content of this block.
        text: String,
        /// When `Some`, Anthropic caches the prompt prefix up to and
        /// including this block for `cache_control.type` duration.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl SystemBlock {
    /// Build a text block without a cache breakpoint.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into(), cache_control: None }
    }

    /// Build a text block with an ephemeral cache breakpoint.
    #[must_use]
    pub fn text_cached(text: impl Into<String>) -> Self {
        Self::Text { text: text.into(), cache_control: Some(CacheControl::ephemeral()) }
    }
}

/// The value of the top-level `system` field in a [`MessageRequest`].
///
/// The Anthropic API accepts two forms:
/// - A plain `String` — the legacy form; serialised as a JSON string.
/// - An array of [`SystemBlock`] objects — the content-block form used
///   when cache-control breakpoints are needed.
///
/// [`SystemField::Plain`] round-trips identically to the previous
/// `system: Option<String>` surface (the existing serialization tests
/// still pass); [`SystemField::Blocks`] enables prompt caching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemField {
    /// Legacy plain-string form. Serialised as a JSON `String`.
    Plain(String),
    /// Content-block array form. Serialised as a JSON array of objects.
    Blocks(Vec<SystemBlock>),
}

impl Serialize for SystemField {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Plain(text) => text.serialize(s),
            Self::Blocks(blocks) => blocks.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for SystemField {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        match v {
            serde_json::Value::String(s) => Ok(Self::Plain(s)),
            serde_json::Value::Array(_) => {
                let blocks: Vec<SystemBlock> =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(Self::Blocks(blocks))
            }
            other => Err(serde::de::Error::custom(format!(
                "expected string or array for system field, got {other}"
            ))),
        }
    }
}

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
///
/// The `system` field accepts both the legacy plain-string form and the
/// content-block array form with [`CacheControl`] breakpoints
/// (PRD-brain-prompt-cache §2.1).  Use [`Self::with_system`] for the
/// legacy string form and [`Self::with_system_blocks`] for the
/// cache-enabled block form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageRequest {
    /// Canonical model id (e.g. `claude-sonnet-4-6`). Resolve via
    /// [`crate::canonical_model`] before constructing.
    pub model: String,
    /// Maximum tokens the response may emit. Anthropic requires this.
    pub max_tokens: u32,
    /// Optional top-level system prompt — either a plain string or a
    /// content-block array with optional cache-control breakpoints.
    pub system: Option<SystemField>,
    /// Conversation history, oldest first.
    pub messages: Vec<Message>,
    /// Request streamed delta events.
    pub stream: bool,
}

impl Serialize for MessageRequest {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        // Count fields: model + max_tokens + messages + stream + optional system.
        let field_count = 4 + usize::from(self.system.is_some());
        let mut map = s.serialize_map(Some(field_count))?;
        map.serialize_entry("model", &self.model)?;
        map.serialize_entry("max_tokens", &self.max_tokens)?;
        if let Some(ref sys) = self.system {
            map.serialize_entry("system", sys)?;
        }
        map.serialize_entry("messages", &self.messages)?;
        map.serialize_entry("stream", &self.stream)?;
        map.end()
    }
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

    /// Set the top-level system prompt as a plain string and return `self`.
    ///
    /// This produces the legacy wire form (`"system": "<text>"`). For
    /// cache-enabled requests use [`Self::with_system_blocks`] instead.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(SystemField::Plain(system.into()));
        self
    }

    /// Set the top-level system prompt as a content-block array and
    /// return `self`. The last block in `blocks` that carries a
    /// `cache_control` breakpoint tells Anthropic to cache everything
    /// up to that point.
    ///
    /// Use [`SystemBlock::text_cached`] on the last stable block to
    /// mark the caching boundary.
    #[must_use]
    pub fn with_system_blocks(mut self, blocks: Vec<SystemBlock>) -> Self {
        self.system = Some(SystemField::Blocks(blocks));
        self
    }
}

/// Usage counters surfaced by the `message_start` SSE event.
///
/// The cache fields (`cache_read_input_tokens`, `cache_creation_input_tokens`)
/// are absent when no `cache_control` breakpoint is present in the request;
/// they default to zero when omitted so callers can sum unconditionally.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[allow(clippy::struct_field_names, reason = "mirrors the Anthropic API field names exactly")]
pub struct MessageUsage {
    /// Number of tokens billed at the full input rate.
    pub input_tokens: u32,
    /// Tokens read from a previously-warmed cache prefix (billed at ~10%).
    pub cache_read_input_tokens: u32,
    /// Tokens written into the cache this turn (billed at ~125%).
    pub cache_creation_input_tokens: u32,
    /// Tokens in the assistant's reply.
    pub output_tokens: u32,
}

/// One server-sent event from the Messages stream.
///
/// We model the events the conversation loop actually consumes; ping
/// and control events are folded into [`Self::Other`] so unknown
/// variants stay forward-compat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// `message_start` — opening event carrying the model id + usage
    /// counters. Cache fields (`cache_read_input_tokens`,
    /// `cache_creation_input_tokens`) are zero when no breakpoint was
    /// sent (PRD-brain-prompt-cache §2.3).
    MessageStart {
        /// Per-turn token accounting including cache-hit metrics.
        usage: MessageUsage,
    },
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
        "message_start" => {
            let msg = v.get("message");
            let usage_val = msg.and_then(|m| m.get("usage"));
            let usage = parse_message_usage(usage_val);
            Ok(StreamEvent::MessageStart { usage })
        }
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

/// Extract [`MessageUsage`] from a JSON `usage` object, defaulting
/// missing or non-integer fields to `0`. Cache fields are optional (the
/// API omits them when no breakpoint was sent).
fn parse_message_usage(usage: Option<&serde_json::Value>) -> MessageUsage {
    let Some(obj) = usage else {
        return MessageUsage::default();
    };
    let get_u32 = |key: &str| -> u32 {
        obj.get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0)
    };
    MessageUsage {
        input_tokens: get_u32("input_tokens"),
        cache_read_input_tokens: get_u32("cache_read_input_tokens"),
        cache_creation_input_tokens: get_u32("cache_creation_input_tokens"),
        output_tokens: get_u32("output_tokens"),
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

/// Anthropic Messages API base URL used in production.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// `anthropic-version` request header value pinned by the brain.
///
/// Bumping requires re-validating that the SSE event shape we model in
/// [`StreamEvent`] still matches the new API contract.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Errors raised by [`AnthropicClient`] when calling the Messages API.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// Network, TLS, timeout, or other transport-level failure.
    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),
    /// API returned a non-2xx HTTP status; body is captured for logging.
    #[error("anthropic api status {code}: {body}")]
    Status {
        /// HTTP status code.
        code: u16,
        /// Response body, truncated to 4 KiB for the message.
        body: String,
    },
    /// SSE framing layer found a malformed event before `parse_sse_event`
    /// could be called (e.g., event block with no `data:` line).
    #[error("sse framing error: {reason}")]
    Frame {
        /// Human-readable diagnosis.
        reason: String,
    },
    /// A `data:` payload could not be decoded into a [`StreamEvent`].
    #[error("sse parse error: {0}")]
    Parse(#[from] ParseError),
}

/// Live HTTP client for `POST /v1/messages` with SSE streaming responses.
///
/// iter-4 buffers the full SSE response before dispatching events; per
/// PRD §2.2.3 the v1 brain consumes the whole reply before handing to
/// wm-tts. iter-5 will swap [`collect_messages`] for a true streaming
/// surface once sentence-handoff lands.
///
/// [`collect_messages`]: AnthropicClient::collect_messages
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    anthropic_version: String,
}

impl AnthropicClient {
    /// Construct a client with the production base URL and the standard
    /// `anthropic-version` header. The HTTP client is built lazily and
    /// reuses connection pooling across calls.
    ///
    /// # Errors
    /// Returns a [`ClientError::Transport`] if reqwest cannot construct
    /// its default TLS-capable client (rare — happens only when the
    /// rustls feature is mis-built).
    pub fn new(api_key: impl Into<String>) -> Result<Self, ClientError> {
        let http = reqwest::Client::builder().build()?;
        Ok(Self {
            http,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            anthropic_version: ANTHROPIC_VERSION.to_string(),
        })
    }

    /// Override the base URL (used by tests against an in-process mock
    /// server). Production code should leave this untouched.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// POST a streaming Messages request and collect every SSE event
    /// into a `Vec`, in order. Non-2xx responses become
    /// [`ClientError::Status`]; framing or per-event decode failures
    /// become [`ClientError::Frame`] / [`ClientError::Parse`].
    ///
    /// # Errors
    /// See [`ClientError`] variants.
    pub async fn collect_messages(
        &self,
        req: &MessageRequest,
    ) -> Result<Vec<StreamEvent>, ClientError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(req)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let truncated = if body.len() > 4096 {
                body.get(..4096).unwrap_or("").to_string()
            } else {
                body
            };
            return Err(ClientError::Status {
                code: status.as_u16(),
                body: truncated,
            });
        }

        let body = resp.text().await?;
        decode_sse_body(&body)
    }

    /// POST a streaming Messages request and return an asynchronous
    /// [`Stream`] of [`StreamEvent`]s as the API emits them.
    ///
    /// Unlike [`collect_messages`], the response body is consumed
    /// incrementally — events surface as soon as the upstream HTTP
    /// layer flushes the relevant bytes, enabling sentence-boundary
    /// TTS handoff (PRD §1.3.3 Fleet 2). Non-2xx responses are
    /// surfaced eagerly during stream construction.
    ///
    /// # Errors
    /// Returns [`ClientError::Transport`] if the request fails to
    /// send, [`ClientError::Status`] if the server responded non-2xx
    /// (body truncated to 4 KiB), or errors during streaming surface
    /// as individual [`Stream::Item`] values.
    ///
    /// [`collect_messages`]: AnthropicClient::collect_messages
    pub async fn stream_messages(
        &self,
        req: &MessageRequest,
    ) -> Result<EventStream, ClientError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(req)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let truncated = if body.len() > 4096 {
                body.get(..4096).unwrap_or("").to_string()
            } else {
                body
            };
            return Err(ClientError::Status {
                code: status.as_u16(),
                body: truncated,
            });
        }

        Ok(EventStream::from_bytes_stream(resp.bytes_stream()))
    }
}

/// Incremental SSE decoder.
///
/// Bytes arrive in arbitrarily-sized chunks (a single TCP read may
/// split a multi-byte UTF-8 codepoint or land in the middle of an
/// event). The decoder accumulates them, only attempting UTF-8
/// decoding up to the most recent valid prefix, and yields one
/// [`StreamEvent`] per call to [`pop_event`] until the buffered text
/// has no more `\n\n` (or `\r\n\r\n`) delimited blocks.
///
/// [`pop_event`]: SseDecoder::pop_event
#[derive(Debug, Default)]
pub struct SseDecoder {
    text: String,
    incomplete_utf8: Vec<u8>,
}

impl SseDecoder {
    /// Construct an empty decoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one chunk of bytes to the decoder's buffer.
    ///
    /// Bytes that complete a UTF-8 codepoint are folded into the
    /// internal text buffer; a partial codepoint at the tail is held
    /// over until the next chunk arrives.
    pub fn feed(&mut self, chunk: &[u8]) {
        let mut combined = std::mem::take(&mut self.incomplete_utf8);
        combined.extend_from_slice(chunk);
        match std::str::from_utf8(&combined) {
            Ok(s) => self.text.push_str(s),
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                let valid_prefix = combined.get(..valid_up_to).unwrap_or(&[]);
                if let Ok(valid_str) = std::str::from_utf8(valid_prefix) {
                    self.text.push_str(valid_str);
                }
                let suffix = combined.get(valid_up_to..).unwrap_or(&[]);
                self.incomplete_utf8 = suffix.to_vec();
            }
        }
    }

    /// Extract the next decoded event from the buffer, if a complete
    /// event-block is present.
    ///
    /// Returns:
    /// - `Some(Ok(event))` when a complete block was found and parsed.
    /// - `Some(Err(_))` when a complete block was found but the
    ///   payload was malformed. The block is consumed regardless so
    ///   the decoder can make forward progress.
    /// - `None` when the buffer holds no terminated block yet (caller
    ///   should feed more bytes).
    pub fn pop_event(&mut self) -> Option<Result<StreamEvent, ClientError>> {
        loop {
            let (block_len, delim_len) = find_block_boundary(&self.text)?;
            // Build the block string and drop the consumed prefix
            // (block + delimiter) from the buffer in one drain call.
            let drained: String = self.text.drain(..block_len + delim_len).collect();
            let block = drained.get(..block_len).unwrap_or("");
            match parse_event_block(block) {
                Ok(Some(ev)) => return Some(Ok(ev)),
                Ok(None) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Search the buffer for the first SSE block boundary (`\n\n` or
/// `\r\n\r\n`). Returns `(block_byte_len, delim_byte_len)` such that
/// `text[..block_byte_len]` is the block payload and
/// `text[block_byte_len..block_byte_len + delim_byte_len]` is the
/// delimiter.
fn find_block_boundary(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        let b = bytes.get(i).copied()?;
        if b == b'\n' {
            if bytes.get(i + 1).copied() == Some(b'\n') {
                return Some((i, 2));
            }
        } else if b == b'\r'
            && bytes.get(i + 1).copied() == Some(b'\n')
            && bytes.get(i + 2).copied() == Some(b'\r')
            && bytes.get(i + 3).copied() == Some(b'\n')
        {
            return Some((i, 4));
        }
        i += 1;
    }
    None
}

/// Parse one event block (the substring between two boundaries) into
/// at most one [`StreamEvent`].
///
/// Returns `Ok(None)` for comment-only blocks (lines starting with
/// `:`) and `event:`-only keepalives that carry no `data:` payload.
fn parse_event_block(block: &str) -> Result<Option<StreamEvent>, ClientError> {
    let trimmed = block.trim_matches(|c: char| c == '\n' || c == '\r');
    if trimmed.is_empty() {
        return Ok(None);
    }
    let mut data_lines: Vec<&str> = Vec::new();
    for raw_line in trimmed.split('\n') {
        let line = raw_line.trim_end_matches('\r');
        if line.starts_with(':') || line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            let payload = rest.strip_prefix(' ').unwrap_or(rest);
            data_lines.push(payload);
        }
    }
    if data_lines.is_empty() {
        return Ok(None);
    }
    let payload = data_lines.join("\n");
    Ok(Some(parse_sse_event(&payload)?))
}

/// Asynchronous stream of [`StreamEvent`]s produced by
/// [`AnthropicClient::stream_messages`].
///
/// The stream wraps reqwest's `bytes_stream()` and feeds incoming
/// chunks into an [`SseDecoder`]. Each call to `poll_next` first
/// drains any complete event the decoder already holds, then pulls
/// more bytes from upstream.
pub struct EventStream {
    inner: Pin<Box<dyn Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>,
    decoder: SseDecoder,
    upstream_done: bool,
}

impl std::fmt::Debug for EventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream")
            .field("decoder", &self.decoder)
            .field("upstream_done", &self.upstream_done)
            .finish_non_exhaustive()
    }
}

impl EventStream {
    /// Wrap a `Stream` of HTTP body chunks. Visible to tests; production
    /// code reaches this through [`AnthropicClient::stream_messages`].
    pub(crate) fn from_bytes_stream<S>(inner: S) -> Self
    where
        S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            decoder: SseDecoder::new(),
            upstream_done: false,
        }
    }
}

impl Stream for EventStream {
    type Item = Result<StreamEvent, ClientError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(ev) = this.decoder.pop_event() {
                return Poll::Ready(Some(ev));
            }
            if this.upstream_done {
                return Poll::Ready(None);
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => {
                    this.decoder.feed(chunk.as_ref());
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(ClientError::Transport(e))));
                }
                Poll::Ready(None) => {
                    this.upstream_done = true;
                }
            }
        }
    }
}

/// Decode a buffered SSE body into the ordered sequence of
/// [`StreamEvent`]s it carries.
///
/// Events are separated by a blank line; within each event we look for
/// `data:` (or `data: ` with optional leading space) lines and feed
/// their payload to [`parse_sse_event`]. Comment lines (`:` prefix) and
/// non-`data:` fields are ignored.
///
/// # Errors
/// Returns [`ClientError::Frame`] when an event block contains no
/// `data:` line, and [`ClientError::Parse`] when the payload fails
/// `parse_sse_event`.
pub fn decode_sse_body(body: &str) -> Result<Vec<StreamEvent>, ClientError> {
    let mut decoder = SseDecoder::new();
    decoder.feed(body.as_bytes());
    let mut out = Vec::new();
    while let Some(ev) = decoder.pop_event() {
        out.push(ev?);
    }
    Ok(out)
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
        assert_eq!(ev, StreamEvent::MessageStart { usage: MessageUsage::default() });
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

    #[test]
    fn decode_sse_body_single_event() {
        let body = "event: message_start\n\
                    data: {\"type\":\"message_start\"}\n\n";
        let evs = decode_sse_body(body).unwrap();
        assert_eq!(evs, vec![StreamEvent::MessageStart { usage: MessageUsage::default() }]);
    }

    #[test]
    fn decode_sse_body_multi_event_in_order() {
        let body = "event: message_start\n\
                    data: {\"type\":\"message_start\"}\n\n\
                    event: content_block_start\n\
                    data: {\"type\":\"content_block_start\",\"index\":0}\n\n\
                    event: content_block_delta\n\
                    data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
                    event: message_stop\n\
                    data: {\"type\":\"message_stop\"}\n\n";
        let evs = decode_sse_body(body).unwrap();
        assert_eq!(evs.len(), 4);
        assert_eq!(evs[0], StreamEvent::MessageStart { usage: MessageUsage::default() });
        assert_eq!(evs[1], StreamEvent::ContentBlockStart { index: 0 });
        assert_eq!(
            evs[2],
            StreamEvent::TextDelta {
                index: 0,
                text: "hi".to_string(),
            }
        );
        assert_eq!(evs[3], StreamEvent::MessageStop);
    }

    #[test]
    fn decode_sse_body_handles_crlf() {
        let body = "event: message_start\r\n\
                    data: {\"type\":\"message_start\"}\r\n\r\n\
                    event: message_stop\r\n\
                    data: {\"type\":\"message_stop\"}\r\n\r\n";
        let evs = decode_sse_body(body).unwrap();
        assert_eq!(evs, vec![StreamEvent::MessageStart { usage: MessageUsage::default() }, StreamEvent::MessageStop]);
    }

    #[test]
    fn decode_sse_body_skips_comment_and_event_only_blocks() {
        let body = ": ping\n\n\
                    event: ping\n\n\
                    event: message_start\n\
                    data: {\"type\":\"message_start\"}\n\n";
        let evs = decode_sse_body(body).unwrap();
        assert_eq!(evs, vec![StreamEvent::MessageStart { usage: MessageUsage::default() }]);
    }

    #[test]
    fn decode_sse_body_handles_no_space_after_data_colon() {
        let body = "data:{\"type\":\"message_stop\"}\n\n";
        let evs = decode_sse_body(body).unwrap();
        assert_eq!(evs, vec![StreamEvent::MessageStop]);
    }

    #[test]
    fn decode_sse_body_surfaces_parse_error() {
        let body = "data: not-json\n\n";
        let err = decode_sse_body(body).unwrap_err();
        assert!(matches!(err, ClientError::Parse(ParseError::Json(_))));
    }

    #[test]
    fn client_new_succeeds_with_default_settings() {
        let client = AnthropicClient::new("test-key").unwrap();
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
        assert_eq!(client.anthropic_version, ANTHROPIC_VERSION);
        assert_eq!(client.api_key, "test-key");
    }

    #[test]
    fn client_with_base_url_overrides() {
        let client = AnthropicClient::new("k")
            .unwrap()
            .with_base_url("http://localhost:9999");
        assert_eq!(client.base_url, "http://localhost:9999");
    }

    #[test]
    fn sse_decoder_yields_nothing_on_partial_block() {
        let mut d = SseDecoder::new();
        d.feed(b"event: message_start\ndata: {\"type\":\"message_start\"}");
        assert!(d.pop_event().is_none());
        d.feed(b"\n\n");
        let ev = d.pop_event().unwrap().unwrap();
        assert_eq!(ev, StreamEvent::MessageStart { usage: MessageUsage::default() });
        assert!(d.pop_event().is_none());
    }

    #[test]
    fn sse_decoder_splits_across_chunks_byte_by_byte() {
        let body = "event: message_start\n\
                    data: {\"type\":\"message_start\"}\n\n\
                    event: content_block_delta\n\
                    data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
                    event: message_stop\n\
                    data: {\"type\":\"message_stop\"}\n\n";
        let mut d = SseDecoder::new();
        let mut events: Vec<StreamEvent> = Vec::new();
        for byte in body.as_bytes() {
            d.feed(&[*byte]);
            while let Some(ev) = d.pop_event() {
                events.push(ev.unwrap());
            }
        }
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart { usage: MessageUsage::default() },
                StreamEvent::TextDelta {
                    index: 0,
                    text: "hi".to_string()
                },
                StreamEvent::MessageStop,
            ]
        );
    }

    #[test]
    fn sse_decoder_handles_partial_utf8_across_chunks() {
        // "é" is 0xC3 0xA9 in UTF-8 — split it across two feeds and
        // confirm the decoder reassembles it before emitting the event.
        let part1 = b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"caf\xC3";
        let part2 = b"\xA9\"}}\n\n";
        let mut d = SseDecoder::new();
        d.feed(part1);
        assert!(d.pop_event().is_none(), "no complete block yet");
        d.feed(part2);
        let ev = d.pop_event().unwrap().unwrap();
        assert_eq!(
            ev,
            StreamEvent::TextDelta {
                index: 0,
                text: "café".to_string(),
            }
        );
    }

    #[test]
    fn sse_decoder_skips_comment_and_eventonly_keepalives() {
        let mut d = SseDecoder::new();
        d.feed(b": ping\n\nevent: ping\n\ndata: {\"type\":\"message_stop\"}\n\n");
        let ev = d.pop_event().unwrap().unwrap();
        assert_eq!(ev, StreamEvent::MessageStop);
        assert!(d.pop_event().is_none());
    }

    #[test]
    fn sse_decoder_consumes_malformed_block_and_recovers() {
        let mut d = SseDecoder::new();
        d.feed(b"data: not-json\n\ndata: {\"type\":\"message_stop\"}\n\n");
        let first = d.pop_event().unwrap();
        assert!(matches!(first, Err(ClientError::Parse(ParseError::Json(_)))));
        let second = d.pop_event().unwrap().unwrap();
        assert_eq!(second, StreamEvent::MessageStop);
        assert!(d.pop_event().is_none());
    }

    #[test]
    fn sse_decoder_handles_crlf_delimiter() {
        let mut d = SseDecoder::new();
        d.feed(b"event: message_start\r\ndata: {\"type\":\"message_start\"}\r\n\r\nevent: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n");
        let a = d.pop_event().unwrap().unwrap();
        let b = d.pop_event().unwrap().unwrap();
        assert_eq!(a, StreamEvent::MessageStart { usage: MessageUsage::default() });
        assert_eq!(b, StreamEvent::MessageStop);
    }

    // ── PRD-brain-prompt-cache AC1 / AC2 ────────────────────────────────────

    /// AC1: `MessageRequest` can serialize a `system` content-block array
    /// with a `cache_control: {"type":"ephemeral"}` breakpoint on the final
    /// stable block, and the serialized JSON matches Anthropic's documented
    /// form exactly.
    #[test]
    fn system_block_with_cache_control_serializes_correctly() {
        let req = MessageRequest::streaming("claude-sonnet-4-6", 256, vec![])
            .with_system_blocks(vec![
                SystemBlock::text_cached("Stable persona prefix."),
            ]);
        let v = serde_json::to_value(&req).unwrap();
        let system = &v["system"];
        assert!(system.is_array(), "system must be a JSON array when blocks used");
        let blocks = system.as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        let block = &blocks[0];
        assert_eq!(block["type"], "text");
        assert_eq!(block["text"], "Stable persona prefix.");
        assert_eq!(block["cache_control"]["type"], "ephemeral",
            "cache_control must be {{\"type\":\"ephemeral\"}} on the marked block");
    }

    /// AC1 (continued): two blocks — first without breakpoint, last with.
    #[test]
    fn system_blocks_only_last_carries_cache_control() {
        let req = MessageRequest::streaming("claude-sonnet-4-6", 256, vec![])
            .with_system_blocks(vec![
                SystemBlock::text("Volatile recall context."),
                SystemBlock::text_cached("Stable persona."),
            ]);
        let v = serde_json::to_value(&req).unwrap();
        let blocks = v["system"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].get("cache_control").is_none(),
            "first block must not have cache_control");
        assert_eq!(blocks[1]["cache_control"]["type"], "ephemeral");
    }

    /// AC2: when no breakpoint is configured, `system` is serialized as a
    /// plain string (byte-identical to the previous format), and the
    /// existing serialization invariants hold.
    #[test]
    fn plain_system_string_serializes_as_string_not_array() {
        let req = MessageRequest::streaming("claude-sonnet-4-6", 256, vec![])
            .with_system("You are Wintermute.");
        let v = serde_json::to_value(&req).unwrap();
        assert!(v["system"].is_string(),
            "plain system must serialize as a JSON string, not array");
        assert_eq!(v["system"], "You are Wintermute.");
    }

    /// AC2 (continued): `system` is omitted when `None`, not serialized as null.
    #[test]
    fn system_omitted_when_none_with_blocks_api() {
        let req = MessageRequest::streaming("claude-sonnet-4-6", 256, vec![]);
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("system").is_none(),
            "system must be omitted (not null) when unset");
    }

    /// AC6: `cache_read_input_tokens` and `cache_creation_input_tokens` are
    /// parsed from `message_start.message.usage`.
    #[test]
    fn message_start_parses_cache_usage_fields() {
        let raw = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_abc",
                "usage": {
                    "input_tokens": 1000,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 50,
                    "output_tokens": 120
                }
            }
        }"#;
        let ev = parse_sse_event(raw).unwrap();
        let StreamEvent::MessageStart { usage } = ev else {
            panic!("expected MessageStart");
        };
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.cache_read_input_tokens, 800);
        assert_eq!(usage.cache_creation_input_tokens, 50);
        assert_eq!(usage.output_tokens, 120);
    }

    /// AC6 (continued): cache fields default to 0 when absent.
    #[test]
    fn message_start_cache_fields_default_to_zero() {
        let raw = r#"{"type":"message_start","message":{"usage":{"input_tokens":200,"output_tokens":10}}}"#;
        let ev = parse_sse_event(raw).unwrap();
        let StreamEvent::MessageStart { usage } = ev else {
            panic!("expected MessageStart");
        };
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.cache_creation_input_tokens, 0);
    }

    /// AC2 / backward compat: the existing `streaming_request_serializes_with_stream_true`
    /// invariant — system absent means it's omitted, not null — still holds
    /// after the `SystemField` refactor.
    #[test]
    fn streaming_request_no_system_omits_field() {
        let req = MessageRequest::streaming(
            "claude-sonnet-4-6",
            1024,
            vec![Message { role: Role::User, content: "ping".to_string() }],
        );
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("system").is_none(), "system must be absent, not null");
        assert_eq!(v["stream"], true);
    }

    // ── PRD-brain-prompt-cache AC5 ──────────────────────────────────────────

    /// Deterministic token estimate: ~4 characters per token, matching the
    /// rough heuristic Anthropic publishes for English text. Used only by the
    /// AC5 fixture below to keep the simulation self-contained (no tokenizer
    /// dependency, no network).
    fn est_tokens(s: &str) -> u32 {
        // ceil(len / 4); 0 for the empty string.
        let n = u32::try_from(s.chars().count()).unwrap_or(u32::MAX);
        n.div_ceil(4)
    }

    /// A fake Anthropic client that produces a [`MessageUsage`] for a request
    /// the same way the real API bills prompt caching: the bytes of any system
    /// block carrying a `cache_control` breakpoint (plus everything before it)
    /// are charged as `cache_creation_input_tokens` the first time that exact
    /// prefix is seen, and as `cache_read_input_tokens` on every subsequent
    /// turn within the 5-minute TTL window. All non-cached tail blocks and the
    /// message list are billed as full-rate `input_tokens` every turn.
    ///
    /// This mirrors the [Anthropic prompt-caching contract](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching):
    /// a stable prefix terminated by a breakpoint is cached; volatile content
    /// after the breakpoint is not. It is the deterministic in-process stand-in
    /// the PRD's AC5 calls for ("a fake Anthropic client that echoes
    /// `cache_read_input_tokens` per the `cache_control` breakpoints sent").
    #[derive(Default)]
    struct CacheSimClient {
        /// Cached prefix bytes seen so far this session (warm prefixes).
        warm: std::collections::HashSet<String>,
    }

    impl CacheSimClient {
        /// Bill one request, returning the usage the real API would echo.
        fn bill(&mut self, req: &MessageRequest) -> MessageUsage {
            let mut usage = MessageUsage::default();
            // Walk the system blocks. The cached prefix is every block up to
            // and including the LAST block carrying a cache_control breakpoint.
            if let Some(SystemField::Blocks(blocks)) = &req.system {
                // Find the index of the last cache_control-bearing block.
                let last_cached = blocks.iter().rposition(|b| {
                    matches!(b, SystemBlock::Text { cache_control: Some(_), .. })
                });
                let mut cached_prefix = String::new();
                for (i, b) in blocks.iter().enumerate() {
                    let SystemBlock::Text { text, .. } = b;
                    let toks = est_tokens(text);
                    match last_cached {
                        Some(c) if i <= c => {
                            // Part of the cacheable prefix.
                            cached_prefix.push_str(text);
                            cached_prefix.push('\u{1}'); // block separator
                            let _ = toks; // accounted below once prefix known
                        }
                        _ => {
                            // Volatile tail block: full-rate input every turn.
                            usage.input_tokens += toks;
                        }
                    }
                }
                if !cached_prefix.is_empty() {
                    let prefix_toks = est_tokens(&cached_prefix);
                    if self.warm.contains(&cached_prefix) {
                        usage.cache_read_input_tokens += prefix_toks;
                    } else {
                        usage.cache_creation_input_tokens += prefix_toks;
                        self.warm.insert(cached_prefix);
                    }
                }
            } else if let Some(SystemField::Plain(text)) = &req.system {
                // No breakpoint: full-rate every turn (the pre-PRD behaviour).
                usage.input_tokens += est_tokens(text);
            }
            // Message list is always full-rate input here (history caching is
            // out of scope for this fixture; the PRD's ≥60% target is met by
            // the persona/tool prefix alone, which dominates per-turn bytes).
            for m in &req.messages {
                usage.input_tokens += est_tokens(&m.content);
            }
            usage
        }
    }

    /// AC5: over a recorded 50-turn conversation, the prompt-cache hit ratio
    /// `sum(cache_read) / sum(cache_read + cache_creation + input)` is ≥0.60.
    ///
    /// The conversation reuses one large stable persona/tool prefix (carrying
    /// the ephemeral breakpoint) across all 50 turns, with a small volatile
    /// recall block and a short user turn that both differ every turn. The
    /// [`CacheSimClient`] bills the prefix as a cache *write* on turn 1 and a
    /// cache *read* on turns 2..50, exactly as the live API would given the
    /// `cache_control` breakpoints the composition emits. This proves the
    /// breakpoints are placed such that the stable prefix is genuinely cached.
    #[test]
    fn cache_hit_ratio_above_60pct_over_50_turns() {
        // A representative large stable prefix: persona + tool defs + gates.
        // ~2 KB is conservative; the real wmd persona prefix is larger, which
        // only improves the ratio.
        let stable_prefix = "You are Wintermute, a calm companion. \
            Speak plainly, never emit markdown. \
            [tool: recall.search] [tool: recall.save_fact] [tool: clock] \
            [tool: weather] [tool: timer] [tool: reminder] \
            Destructive-intent gate: confirm before any irreversible action. \
            Child-lock: refuse age-inappropriate content. "
            .repeat(8); // inflate to a realistic multi-KB system prefix

        let mut client = CacheSimClient::default();
        let mut sum_read: u64 = 0;
        let mut sum_create: u64 = 0;
        let mut sum_input: u64 = 0;

        for turn in 0..50u32 {
            // Volatile per-turn recall block + user transcript both vary.
            let recall = format!(
                "What you remember about the user (most relevant first):\n1. fact #{turn}"
            );
            let transcript = format!("turn {turn}: what's the weather and what did we say?");
            let req = crate::daemon::compose_request(
                "claude-sonnet-4-6",
                &stable_prefix,
                Some(recall.as_str()),
                None,
                &[],
                &transcript,
            );
            let usage = client.bill(&req);
            sum_read += u64::from(usage.cache_read_input_tokens);
            sum_create += u64::from(usage.cache_creation_input_tokens);
            sum_input += u64::from(usage.input_tokens);
        }

        let denom = sum_read + sum_create + sum_input;
        assert!(denom > 0, "fixture produced no billed tokens");
        let ratio = sum_read as f64 / denom as f64;
        assert!(
            ratio >= 0.60,
            "AC5: cache hit ratio {ratio:.3} must be >= 0.60 \
             (read={sum_read} create={sum_create} input={sum_input})"
        );
        // Sanity: turn 1 must be a cache write, not a read.
        assert!(sum_create > 0, "first turn must create the cache entry");
    }
}
