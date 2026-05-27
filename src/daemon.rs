//! Live agorabus subscribe loop for `wm-brain` / `wmd`.
//!
//! Wires the bus schema from [`crate::bus`] to a real subscribe loop.
//! The daemon subscribes to [`bus::DIALOG_TOPIC_PREFIX`]
//! (`wm.dialog.`) — inbound turn + verbal-confirm envelopes from
//! `wm-dialog` — and opens a separate publish connection (read/write
//! on a subscribed socket would interleave `Reply` lines with the
//! broadcast stream — same pattern as `wintermute-stt/src/daemon.rs`).
//!
//! iter-8 lands the minimum-viable conversation cycle: a `TurnUser`
//! event triggers an Anthropic Messages API call (buffered-streaming via
//! [`crate::anthropic::AnthropicClient::collect_messages`]) and the
//! collected assistant text is published as [`bus::outgoing::REPLY`].
//! API failures publish [`bus::outgoing::ERROR`] with `kind=anthropic`.
//! Recall retrieval, tool routing, destructive-intent gating, and turn
//! memorisation remain iter-9+ work.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::anthropic::{AnthropicClient, ClientError, Message, MessageRequest, Role, StreamEvent};
use crate::bus::{
    self, DecodeError, Emit, ErrorEvent, ReplyEvent, Request, TurnUserEvent, decode_request,
    now_unix_ms, outgoing,
};
use crate::{BrainConfig, canonical_model};

/// Default upper bound on tokens the daemon requests per turn.
///
/// The PRD's "one short paragraph per turn unless asked" target sits
/// comfortably under 1 KiB of tokens; the headroom leaves room for the
/// iter-10 destructive-intent JSON trailer.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// System prompt iter-8 ships. iter-9 grows this with child-lock + tool
/// router instructions; iter-10 adds destructive-intent gating.
pub const DEFAULT_PERSONA: &str = "You are wintermute, a voice-first companion daemon. \
The user hears you spoken aloud, never reads you on a screen. \
Speak naturally and warmly in plain prose. Keep replies to one short \
paragraph per turn unless the user asks for more. Do not use markdown, \
bullet lists, code fences, or emoji — they do not speak well.";

/// Abstraction over the Anthropic Messages API the conversation loop
/// drives. Production impl is [`AnthropicClient`]; tests inject an
/// in-memory fake.
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// Issue one buffered streaming call and return every SSE event in
    /// the order Anthropic emitted them.
    ///
    /// # Errors
    /// Propagates the underlying transport / status / parse failure as
    /// a [`ClientError`].
    async fn collect_messages(
        &self,
        req: &MessageRequest,
    ) -> Result<Vec<StreamEvent>, ClientError>;
}

#[async_trait::async_trait]
impl LlmClient for AnthropicClient {
    async fn collect_messages(
        &self,
        req: &MessageRequest,
    ) -> Result<Vec<StreamEvent>, ClientError> {
        Self::collect_messages(self, req).await
    }
}

/// Build a buffered streaming request for a single user turn. Pure
/// function; iter-9 grows this to splice recall hits and tool
/// definitions onto the prompt.
#[must_use]
pub fn compose_request(model: &str, persona: &str, transcript: &str) -> MessageRequest {
    MessageRequest::streaming(
        canonical_model(model),
        DEFAULT_MAX_TOKENS,
        vec![Message {
            role: Role::User,
            content: transcript.to_string(),
        }],
    )
    .with_system(persona.to_string())
}

/// Concatenate every `text_delta` chunk into a single assistant reply.
///
/// Pure function. Non-text events (`ping`, `message_start`,
/// `content_block_stop`, …) are skipped; iter-9 grows this to track
/// `stop_reason` for the destructive-intent code path.
#[must_use]
pub fn extract_assistant_text(events: &[StreamEvent]) -> String {
    let mut out = String::new();
    for ev in events {
        if let StreamEvent::TextDelta { text, .. } = ev {
            out.push_str(text);
        }
    }
    out
}

/// Publish abstraction so per-request handlers can be tested without
/// an actual agorabus daemon. Production impl is [`AgoraSink`]; tests
/// use an in-memory sink.
#[async_trait::async_trait]
pub trait EventSink: Send {
    /// Publish `data` on `topic`. The dispatch layer treats failures
    /// as fatal for the current request but logs and continues the
    /// outer subscribe loop.
    ///
    /// # Errors
    /// Propagates whatever the underlying transport returns.
    async fn publish(&mut self, topic: &str, data: Value) -> Result<()>;
}

/// Production sink: publishes through an agorabus [`agorabus::Client`].
pub struct AgoraSink {
    pub(crate) inner: agorabus::Client,
}

#[async_trait::async_trait]
impl EventSink for AgoraSink {
    async fn publish(&mut self, topic: &str, data: Value) -> Result<()> {
        let reply = self.inner.publish(topic, data).await?;
        if !reply.ok {
            warn!(
                topic = %topic,
                err = %reply.error.as_deref().unwrap_or("?"),
                "wm-brain: bus rejected publish"
            );
        }
        Ok(())
    }
}

/// Live daemon state.
///
/// iter-8 grows this with the optional [`LlmClient`] handle used for
/// per-turn Anthropic Messages calls and the persona text spliced into
/// every request as a system prompt. iter-9+ adds the recall client +
/// per-turn conversation buffer once retrieval and memorisation land.
pub struct DaemonState {
    /// Resolved runtime config (model defaults, recall socket, …).
    pub config: Mutex<BrainConfig>,
    /// Anthropic Messages client. `None` means the daemon is running
    /// without an API key — turn events are logged and dropped without
    /// publishing.
    pub llm: Option<Arc<dyn LlmClient>>,
    /// System-prompt persona spliced into every request. Defaults to
    /// [`DEFAULT_PERSONA`].
    pub persona: String,
}

impl DaemonState {
    /// Construct a daemon state from an already-validated config. The
    /// resulting state has no LLM client; attach one via
    /// [`Self::with_llm`].
    #[must_use]
    pub fn new(config: BrainConfig) -> Self {
        Self {
            config: Mutex::new(config),
            llm: None,
            persona: DEFAULT_PERSONA.to_string(),
        }
    }

    /// Attach an LLM client to a freshly-built state.
    #[must_use]
    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Override the persona system prompt. Useful for tests and for the
    /// future iter-9 child-lock variant.
    #[must_use]
    pub fn with_persona(mut self, persona: impl Into<String>) -> Self {
        self.persona = persona.into();
        self
    }
}

/// Resolve the outbound topic for a single emit. Pure function; exposed
/// so tests can pin the topic vocabulary without spinning up a bus.
#[must_use]
pub const fn topic_for_emit(emit: &Emit) -> &'static str {
    match emit {
        Emit::Reply(_) => outgoing::REPLY,
        Emit::ReplyDestructive(_) => outgoing::REPLY_DESTRUCTIVE,
        Emit::ToolCall(_) => outgoing::TOOL_CALL,
        Emit::ToolResult(_) => outgoing::TOOL_RESULT,
        Emit::Error(_) => outgoing::ERROR,
    }
}

/// Serialise an emit into the JSON value the agorabus expects.
///
/// # Errors
/// Propagates `serde_json::Error` — every [`Emit`] variant uses
/// `Serialize` impls that don't fail in practice, so a returned error
/// is a programmer bug rather than runtime-recoverable.
pub fn emit_to_value(emit: &Emit) -> Result<Value> {
    Ok(match emit {
        Emit::Reply(r) => serde_json::to_value(r)?,
        Emit::ReplyDestructive(r) => serde_json::to_value(r)?,
        Emit::ToolCall(c) => serde_json::to_value(c)?,
        Emit::ToolResult(r) => serde_json::to_value(r)?,
        Emit::Error(e) => serde_json::to_value(e)?,
    })
}

/// Dispatch one decoded request.
///
/// iter-8 wires the conversation cycle for [`Request::TurnUser`]:
/// compose an Anthropic request from the persona + transcript, call the
/// LLM, and publish the assistant text as [`outgoing::REPLY`]. API
/// failures publish [`outgoing::ERROR`] with `kind=anthropic`.
/// `confirm.granted` / `confirm.denied` stay as logging stubs until
/// iter-10 destructive-intent gating lands.
///
/// # Errors
/// Returns the first publish failure encountered. LLM-side errors are
/// surfaced via published [`outgoing::ERROR`] events rather than as
/// `Err` returns — the outer subscribe loop should keep running.
pub async fn dispatch(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    req: Request,
    now_ms: u64,
) -> Result<()> {
    let model = {
        let cfg = state.config.lock().await;
        cfg.effective_model().to_string()
    };
    match req {
        Request::TurnUser(t) => {
            handle_turn_user(state, publish, &model, &t, now_ms).await?;
        }
        Request::ConfirmGranted(c) => {
            info!(
                intent_id = %c.intent_id,
                ts = c.ts,
                "wm-brain: confirm.granted received (destructive dispatch deferred to iter-10)"
            );
        }
        Request::ConfirmDenied(c) => {
            info!(
                intent_id = %c.intent_id,
                reason = %c.reason,
                ts = c.ts,
                "wm-brain: confirm.denied received"
            );
        }
    }
    Ok(())
}

async fn handle_turn_user(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    model: &str,
    turn: &TurnUserEvent,
    now_ms: u64,
) -> Result<()> {
    let Some(llm) = state.llm.as_ref() else {
        info!(
            transcript = %turn.transcript,
            confidence = turn.confidence,
            ts = turn.ts,
            model = %model,
            "wm-brain: turn.user received but no LLM configured; dropping"
        );
        return Ok(());
    };
    let req = compose_request(model, &state.persona, &turn.transcript);
    match llm.collect_messages(&req).await {
        Ok(events) => {
            let text = extract_assistant_text(&events);
            if text.is_empty() {
                warn!(
                    model = %model,
                    "wm-brain: llm returned no text deltas; emitting empty-reply error"
                );
                publish_error_at(publish, "anthropic", "no text in response", now_ms).await?;
                return Ok(());
            }
            let reply = ReplyEvent { text, ts: now_ms };
            publish
                .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
                .await
                .context("publish reply")?;
        }
        Err(err) => {
            error!(err = %err, model = %model, "wm-brain: anthropic call failed");
            publish_error_at(publish, "anthropic", &format!("{err}"), now_ms).await?;
        }
    }
    Ok(())
}

async fn publish_error_at(
    publish: &mut dyn EventSink,
    kind: &str,
    message: &str,
    ts: u64,
) -> Result<()> {
    let ev = ErrorEvent {
        kind: kind.to_string(),
        message: message.to_string(),
        ts,
    };
    publish
        .publish(outgoing::ERROR, serde_json::to_value(&ev)?)
        .await
}

async fn publish_error(publish: &mut dyn EventSink, kind: &str, message: &str) -> Result<()> {
    publish_error_at(publish, kind, message, now_unix_ms()).await
}

/// Wrap a concrete [`LlmClient`] into an `Arc<dyn LlmClient>` via the
/// implicit unsizing coercion at the return-type boundary. Lets callers
/// build trait objects without an explicit `as` cast.
fn into_dyn_llm<T: LlmClient + 'static>(client: T) -> Arc<dyn LlmClient> {
    Arc::new(client)
}

/// Build an [`AnthropicClient`] from the env var named by `api_key_env`.
///
/// Returns `None` (with a `warn!`) if the var is unset, empty, or if
/// reqwest cannot build a TLS-capable HTTP client. The daemon stays
/// runnable without an LLM — turn events get logged and dropped — so
/// the operator can observe live bus traffic during setup.
#[allow(
    clippy::cognitive_complexity,
    reason = "dispatch shell: each branch is a single guard + warn + return"
)]
fn build_llm_from_env(api_key_env: &str) -> Option<Arc<dyn LlmClient>> {
    let Ok(key) = std::env::var(api_key_env) else {
        warn!(
            env = %api_key_env,
            "wm-brain: api-key env var unset; turn.user events will be dropped silently"
        );
        return None;
    };
    if key.is_empty() {
        warn!(
            env = %api_key_env,
            "wm-brain: api-key env var present but empty; turn.user events will be dropped silently"
        );
        return None;
    }
    match AnthropicClient::new(key) {
        Ok(client) => Some(into_dyn_llm(client)),
        Err(err) => {
            warn!(err = %err, "wm-brain: failed to build anthropic client; running without LLM");
            None
        }
    }
}

/// Run the live daemon: validate config, connect to agorabus, subscribe
/// to the dialog prefix, dispatch each event until the bus closes.
///
/// # Errors
/// Propagates I/O failures from config validation or the agorabus
/// client. A missing agorabus socket is *not* an error: the daemon
/// logs and exits cleanly so the systemd unit restarts it when the bus
/// comes back (same pattern as `wm-stt` / `wm-tts`).
pub async fn run(cfg: BrainConfig) -> Result<()> {
    cfg.validate().context("wm-brain: config validation failed")?;

    let llm = build_llm_from_env(&cfg.api_key_env);
    let state = Arc::new({
        let base = DaemonState::new(cfg);
        match llm {
            Some(client) => base.with_llm(client),
            None => base,
        }
    });

    let sock = agorabus::default_socket_path();
    let Some(mut sub_client) = agorabus::Client::try_connect(&sock).await? else {
        warn!(socket = %sock.display(), "wm-brain: agorabus not reachable; exiting");
        return Ok(());
    };
    sub_client.subscribe(bus::DIALOG_TOPIC_PREFIX).await?;
    info!(
        dialog_prefix = bus::DIALOG_TOPIC_PREFIX,
        "wm-brain: subscribed"
    );

    let pub_client = agorabus::Client::connect(&sock).await?;
    let mut sink = AgoraSink { inner: pub_client };

    while let Some(ev) = sub_client.next_event().await? {
        match decode_request(&ev.topic, &ev.data) {
            Ok(req) => {
                let now = now_unix_ms();
                if let Err(err) = dispatch(state.as_ref(), &mut sink, req, now).await {
                    error!(topic = %ev.topic, err = %err, "wm-brain: dispatch failed");
                    let _ = publish_error(&mut sink, "bus", &format!("dispatch: {err}")).await;
                }
            }
            Err(DecodeError::UnknownTopic(t)) => {
                debug!(topic = %t, "wm-brain: ignoring unknown topic under dialog prefix");
            }
            Err(err) => {
                warn!(topic = %ev.topic, err = %err, "wm-brain: decode failed");
                let _ = publish_error(&mut sink, "bus", &format!("decode: {err}")).await;
            }
        }
    }
    info!("wm-brain: bus closed; daemon exiting");
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::significant_drop_tightening,
    reason = "tests"
)]
mod tests {
    use super::*;
    use crate::bus::{
        ConfirmDeniedEvent, ConfirmGrantedEvent, ErrorEvent as Err, ReplyDestructiveEvent,
        ReplyEvent, ToolCallEvent, ToolResultEvent, TurnUserEvent,
    };
    use serde_json::json;
    use std::sync::Mutex as StdMutex;

    #[derive(Default, Clone)]
    struct MemSink {
        events: Arc<StdMutex<Vec<(String, Value)>>>,
    }

    #[async_trait::async_trait]
    impl EventSink for MemSink {
        async fn publish(&mut self, topic: &str, data: Value) -> Result<()> {
            self.events
                .lock()
                .expect("mem sink poisoned")
                .push((topic.to_string(), data));
            Ok(())
        }
    }

    fn fresh_state() -> Arc<DaemonState> {
        let cfg = BrainConfig::default();
        Arc::new(DaemonState::new(cfg))
    }

    #[derive(Clone)]
    struct FakeLlm {
        response: std::result::Result<Vec<StreamEvent>, &'static str>,
    }

    #[async_trait::async_trait]
    impl LlmClient for FakeLlm {
        async fn collect_messages(
            &self,
            _req: &MessageRequest,
        ) -> std::result::Result<Vec<StreamEvent>, ClientError> {
            match &self.response {
                Ok(events) => Ok(events.clone()),
                Err(msg) => Err(ClientError::Status {
                    code: 500,
                    body: (*msg).to_string(),
                }),
            }
        }
    }

    fn state_with_llm(llm: FakeLlm) -> Arc<DaemonState> {
        let cfg = BrainConfig::default();
        Arc::new(DaemonState::new(cfg).with_llm(into_dyn_llm(llm)))
    }

    fn text_delta(text: &str) -> StreamEvent {
        StreamEvent::TextDelta {
            index: 0,
            text: text.to_string(),
        }
    }

    #[tokio::test]
    async fn dispatch_turn_user_without_llm_drops_silently() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hello".to_string(),
            confidence: 0.9,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 1)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert!(
            events.is_empty(),
            "no LLM configured -> dispatch must not publish"
        );
    }

    #[tokio::test]
    async fn dispatch_confirm_granted_is_silent() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let req = Request::ConfirmGranted(ConfirmGrantedEvent {
            intent_id: "abc".to_string(),
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 1)
            .await
            .expect("dispatch ok");
        assert!(sink.events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn dispatch_confirm_denied_is_silent() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let req = Request::ConfirmDenied(ConfirmDeniedEvent {
            intent_id: "abc".to_string(),
            reason: "timeout".to_string(),
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 1)
            .await
            .expect("dispatch ok");
        assert!(sink.events.lock().unwrap().is_empty());
    }

    #[test]
    fn compose_request_uses_canonical_model_and_includes_persona() {
        let req = compose_request("sonnet", "be terse", "hello there");
        assert_eq!(req.model, "claude-sonnet-4-6");
        assert_eq!(req.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(req.stream);
        assert_eq!(req.system.as_deref(), Some("be terse"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].content, "hello there");
    }

    #[test]
    fn extract_assistant_text_concatenates_deltas_in_order() {
        let events = vec![
            StreamEvent::MessageStart,
            StreamEvent::ContentBlockStart { index: 0 },
            text_delta("Hi "),
            text_delta("there."),
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop,
        ];
        assert_eq!(extract_assistant_text(&events), "Hi there.");
    }

    #[test]
    fn extract_assistant_text_returns_empty_when_no_deltas() {
        let events = vec![
            StreamEvent::MessageStart,
            StreamEvent::MessageDelta { stop_reason: None },
            StreamEvent::MessageStop,
        ];
        assert!(extract_assistant_text(&events).is_empty());
    }

    #[tokio::test]
    async fn dispatch_turn_user_publishes_reply_when_llm_returns_text() {
        let llm = FakeLlm {
            response: Ok(vec![
                StreamEvent::MessageStart,
                text_delta("Hello, "),
                text_delta("Joe."),
                StreamEvent::MessageStop,
            ]),
        };
        let state = state_with_llm(llm);
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hi".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 7777)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY);
        assert_eq!(events[0].1["text"], "Hello, Joe.");
        assert_eq!(events[0].1["ts"], 7777);
    }

    #[tokio::test]
    async fn dispatch_turn_user_publishes_error_on_llm_failure() {
        let llm = FakeLlm {
            response: Err("upstream rate-limited"),
        };
        let state = state_with_llm(llm);
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hi".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 4242)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::ERROR);
        assert_eq!(events[0].1["kind"], "anthropic");
        assert!(
            events[0].1["message"]
                .as_str()
                .unwrap()
                .contains("upstream rate-limited"),
            "error message should surface the upstream body"
        );
        assert_eq!(events[0].1["ts"], 4242);
    }

    #[tokio::test]
    async fn dispatch_turn_user_publishes_error_when_no_text_deltas() {
        let llm = FakeLlm {
            response: Ok(vec![
                StreamEvent::MessageStart,
                StreamEvent::MessageDelta {
                    stop_reason: Some("end_turn".to_string()),
                },
                StreamEvent::MessageStop,
            ]),
        };
        let state = state_with_llm(llm);
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hi".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 9001)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::ERROR);
        assert_eq!(events[0].1["kind"], "anthropic");
        assert_eq!(events[0].1["message"], "no text in response");
        assert_eq!(events[0].1["ts"], 9001);
    }

    #[test]
    fn default_persona_anchors_on_voice_and_avoids_markdown() {
        let p = DEFAULT_PERSONA.to_lowercase();
        assert!(p.contains("voice"), "persona should anchor on the voice surface");
        assert!(p.contains("markdown"), "persona should warn the model off markdown");
    }

    #[test]
    fn topic_for_emit_matches_outgoing_constants() {
        assert_eq!(
            topic_for_emit(&Emit::Reply(ReplyEvent {
                text: String::new(),
                ts: 0,
            })),
            outgoing::REPLY
        );
        assert_eq!(
            topic_for_emit(&Emit::ReplyDestructive(ReplyDestructiveEvent {
                text: String::new(),
                intent_id: String::new(),
                summary: String::new(),
                confirm_keyword: String::new(),
                action: json!({}),
                ts: 0,
            })),
            outgoing::REPLY_DESTRUCTIVE
        );
        assert_eq!(
            topic_for_emit(&Emit::ToolCall(ToolCallEvent {
                tool: String::new(),
                args: json!({}),
                ts: 0,
            })),
            outgoing::TOOL_CALL
        );
        assert_eq!(
            topic_for_emit(&Emit::ToolResult(ToolResultEvent {
                tool: String::new(),
                ok: true,
                body: json!({}),
                ts: 0,
            })),
            outgoing::TOOL_RESULT
        );
        assert_eq!(
            topic_for_emit(&Emit::Error(Err {
                kind: String::new(),
                message: String::new(),
                ts: 0,
            })),
            outgoing::ERROR
        );
    }

    #[test]
    fn emit_to_value_reply_has_expected_fields() {
        let e = Emit::Reply(ReplyEvent {
            text: "ok".to_string(),
            ts: 42,
        });
        let v = emit_to_value(&e).expect("serialises");
        assert_eq!(v["text"], "ok");
        assert_eq!(v["ts"], 42);
    }

    #[test]
    fn emit_to_value_reply_destructive_includes_action() {
        let e = Emit::ReplyDestructive(ReplyDestructiveEvent {
            text: "delete?".to_string(),
            intent_id: "01abc".to_string(),
            summary: "rm /tmp/x".to_string(),
            confirm_keyword: "delete".to_string(),
            action: json!({ "tool": "wm.fs.rm" }),
            ts: 7,
        });
        let v = emit_to_value(&e).expect("serialises");
        assert_eq!(v["intent_id"], "01abc");
        assert_eq!(v["confirm_keyword"], "delete");
        assert_eq!(v["action"]["tool"], "wm.fs.rm");
        assert_eq!(v["ts"], 7);
    }

    #[tokio::test]
    async fn publish_error_writes_to_sink() {
        let mut sink = MemSink::default();
        publish_error(&mut sink, "bus", "decode failed")
            .await
            .expect("publish ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::ERROR);
        assert_eq!(events[0].1["kind"], "bus");
        assert_eq!(events[0].1["message"], "decode failed");
    }
}
