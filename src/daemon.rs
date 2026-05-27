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

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::anthropic::{AnthropicClient, ClientError, Message, MessageRequest, Role, StreamEvent};
use crate::bus::{
    self, DecodeError, Emit, ErrorEvent, ReplyEvent, Request, TurnUserEvent, decode_request,
    now_unix_ms, outgoing,
};
use crate::recall_client::{self, QueryArgs, QueryHit, RecallClient};
use crate::{BrainConfig, PROFILE_SUBJECT, canonical_model};

/// Default upper bound on tokens the daemon requests per turn.
///
/// The PRD's "one short paragraph per turn unless asked" target sits
/// comfortably under 1 KiB of tokens; the headroom leaves room for the
/// iter-10 destructive-intent JSON trailer.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// System prompt iter-8 shipped. iter-9 leaves this base verbatim and
/// layers child-lock + recall-context blocks via [`compose_persona`];
/// iter-10 adds the destructive-intent gating clause.
pub const DEFAULT_PERSONA: &str = "You are wintermute, a voice-first companion daemon. \
The user hears you spoken aloud, never reads you on a screen. \
Speak naturally and warmly in plain prose. Keep replies to one short \
paragraph per turn unless the user asks for more. Do not use markdown, \
bullet lists, code fences, or emoji — they do not speak well.";

/// Child-lock guard appended to the system prompt when
/// [`BrainConfig::child_lock`] is true.
///
/// The voice surface should refuse adult / unsafe-action requests
/// gracefully; the brain does not call out the reason to avoid teaching
/// a younger user how to bypass it.
pub const CHILD_LOCK_GUARD: &str = "Child-lock is active. If the user asks for adult content, \
profanity, instructions for anything dangerous, or any action that would change settings or \
delete data, decline kindly with a short redirect to something age-appropriate. Do not explain \
the lock or how to disable it.";

/// Default per-turn limit for recall hits spliced into the system
/// prompt. Conservative for Fleet 1: keeps the prompt-cache breakpoint
/// stable and well under Anthropic's per-request input budget.
pub const DEFAULT_RECALL_LIMIT: usize = 6;

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
/// function; the caller is responsible for splicing recall context and
/// child-lock guidance into `persona` via [`compose_persona`].
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

/// Assemble the effective system prompt the Anthropic call receives.
///
/// Layers in this order: `base` (persona), child-lock guard when set,
/// then a recall-context block when non-empty. Each layer is separated
/// by a blank line so the model parses them as distinct paragraphs.
#[must_use]
pub fn compose_persona(base: &str, child_lock: bool, recall_context: Option<&str>) -> String {
    let mut out = base.to_string();
    if child_lock {
        out.push_str("\n\n");
        out.push_str(CHILD_LOCK_GUARD);
    }
    if let Some(ctx) = recall_context {
        if !ctx.is_empty() {
            out.push_str("\n\n");
            out.push_str(ctx);
        }
    }
    out
}

/// Render a list of recall hits into the human-readable block we splice
/// onto the system prompt. Returns `None` when the slice is empty so
/// callers can skip the block entirely.
///
/// The format avoids markdown bullets — the persona instructs the model
/// not to emit markdown in its replies, and a numbered list parses
/// cleanly without inviting the model to mirror bullets in output.
#[must_use]
pub fn format_recall_context(hits: &[QueryHit]) -> Option<String> {
    if hits.is_empty() {
        return None;
    }
    let mut out = String::from("What you remember about the user (most relevant first):");
    for (i, hit) in hits.iter().enumerate() {
        let snippet = hit.snippet.trim();
        if snippet.is_empty() {
            continue;
        }
        out.push_str(&format!("\n{}. {}", i + 1, snippet));
    }
    Some(out)
}

/// Abstraction over the per-turn recall query. Production impl is
/// [`LiveRecallSource`] (opens a fresh [`RecallClient`] per call);
/// tests inject a canned source.
#[async_trait::async_trait]
pub trait RecallSource: Send + Sync {
    /// Fetch a ranked slice of relevant memories for `transcript`.
    ///
    /// Returning an empty `Vec` is the universal no-context signal and
    /// must not be reported as an error.
    ///
    /// # Errors
    /// Implementations surface transport / decode failures as
    /// [`RecallSourceError`]; the caller logs them and proceeds with an
    /// empty context (recall outages must not break the conversation).
    async fn fetch(&self, transcript: &str) -> Result<Vec<QueryHit>, RecallSourceError>;
}

/// Errors surfaced by [`RecallSource::fetch`].
#[derive(Debug, thiserror::Error)]
pub enum RecallSourceError {
    /// Underlying recall-daemon client failure.
    #[error("recall client: {0}")]
    Client(#[from] recall_client::ClientError),
}

/// Zero-cost no-op [`RecallSource`]. Returned by [`DaemonState`] when
/// the operator hasn't attached a real source; keeps the live daemon
/// runnable without a recall daemon during early bring-up.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullRecall;

#[async_trait::async_trait]
impl RecallSource for NullRecall {
    async fn fetch(&self, _transcript: &str) -> Result<Vec<QueryHit>, RecallSourceError> {
        Ok(Vec::new())
    }
}

/// Production [`RecallSource`]: opens a fresh recall-daemon socket
/// connection per turn. Fleet 1 trades the per-call connect for
/// simplicity; pooling and connection reuse are Fleet 2 work.
#[derive(Debug, Clone)]
pub struct LiveRecallSource {
    socket: PathBuf,
    profile_subject: String,
    limit: usize,
}

impl LiveRecallSource {
    /// Build a live source pointed at `socket`. Defaults the profile
    /// subject to [`PROFILE_SUBJECT`] and the limit to
    /// [`DEFAULT_RECALL_LIMIT`].
    #[must_use]
    pub fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            profile_subject: PROFILE_SUBJECT.to_string(),
            limit: DEFAULT_RECALL_LIMIT,
        }
    }

    /// Override the recall subject scope for the profile query.
    #[must_use]
    pub fn with_profile_subject(mut self, subject: impl Into<String>) -> Self {
        self.profile_subject = subject.into();
        self
    }

    /// Override the per-turn hit cap.
    #[must_use]
    pub const fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

#[async_trait::async_trait]
impl RecallSource for LiveRecallSource {
    async fn fetch(&self, transcript: &str) -> Result<Vec<QueryHit>, RecallSourceError> {
        let mut client = RecallClient::connect(&self.socket).await?;
        let args = QueryArgs {
            text: transcript.to_string(),
            limit: Some(self.limit),
            hybrid: None,
            project_subject: Some(self.profile_subject.clone()),
        };
        let resp = client.query(&args).await?;
        Ok(resp.ranked_hits)
    }
}

/// Abstraction over the brain's tool router. Fleet 1 ships
/// [`NoToolsRouter`] (always answers "no-tools-registered"); Fleet 2
/// wires real tools (time, weather, recall, fleet2 stubs) behind the
/// same trait.
#[async_trait::async_trait]
pub trait ToolRouter: Send + Sync {
    /// Dispatch one tool call and return the wire body the brain will
    /// publish as `wm.brain.tool.result`.
    ///
    /// `ok=false` with a structured `body` is the canonical
    /// missing-tool response — implementations never panic on unknown
    /// names.
    async fn dispatch(&self, name: &str, args: &Value) -> ToolResultBody;
}

/// Wire body the dispatcher returns; mirrors the `ok` + `body` pair on
/// [`crate::bus::ToolResultEvent`] so the caller can forward it
/// straight onto the bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResultBody {
    /// `true` if the tool returned a value; `false` on error / missing.
    pub ok: bool,
    /// Tool-specific payload (success body OR `{error, ...}`).
    pub body: Value,
}

/// Fleet 1 default [`ToolRouter`]: rejects every call with a stable
/// `no-tools-registered` error body so observers can detect when the
/// model attempts a tool before Fleet 2 wires the real ones.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoToolsRouter;

#[async_trait::async_trait]
impl ToolRouter for NoToolsRouter {
    async fn dispatch(&self, name: &str, _args: &Value) -> ToolResultBody {
        ToolResultBody {
            ok: false,
            body: json!({
                "error": "no-tools-registered",
                "tool": name,
            }),
        }
    }
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
    /// Recall retrieval source. Defaults to [`NullRecall`] so the
    /// daemon is runnable without a live recall daemon; `run()` swaps
    /// in a [`LiveRecallSource`] when the config carries a socket path.
    pub recall: Arc<dyn RecallSource>,
    /// Tool router. Defaults to [`NoToolsRouter`] in Fleet 1; Fleet 2
    /// swaps in a real router with the wm.* tool surface.
    pub tool_router: Arc<dyn ToolRouter>,
    /// System-prompt persona spliced into every request. Defaults to
    /// [`DEFAULT_PERSONA`].
    pub persona: String,
}

impl DaemonState {
    /// Construct a daemon state from an already-validated config. The
    /// resulting state has no LLM client and uses [`NullRecall`] +
    /// [`NoToolsRouter`]; attach real implementations via the
    /// `with_*` builders.
    #[must_use]
    pub fn new(config: BrainConfig) -> Self {
        Self {
            config: Mutex::new(config),
            llm: None,
            recall: Arc::new(NullRecall),
            tool_router: Arc::new(NoToolsRouter),
            persona: DEFAULT_PERSONA.to_string(),
        }
    }

    /// Attach an LLM client to a freshly-built state.
    #[must_use]
    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Swap the recall source. Use to inject a live source or a test
    /// fake.
    #[must_use]
    pub fn with_recall(mut self, recall: Arc<dyn RecallSource>) -> Self {
        self.recall = recall;
        self
    }

    /// Swap the tool router. Fleet 2 calls this with a real router.
    #[must_use]
    pub fn with_tool_router(mut self, router: Arc<dyn ToolRouter>) -> Self {
        self.tool_router = router;
        self
    }

    /// Override the persona system prompt. Useful for tests and for
    /// operator-supplied persona overrides.
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
    let (model, child_lock) = {
        let cfg = state.config.lock().await;
        (cfg.effective_model().to_string(), cfg.child_lock)
    };
    match req {
        Request::TurnUser(t) => {
            handle_turn_user(state, publish, &model, child_lock, &t, now_ms).await?;
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

#[allow(
    clippy::too_many_arguments,
    reason = "per-turn handler: state + sink + model + child_lock + turn + ts; refactoring into \
              a struct would just shuffle the call sites"
)]
async fn handle_turn_user(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    model: &str,
    child_lock: bool,
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
    let hits = match state.recall.fetch(&turn.transcript).await {
        Ok(h) => h,
        Err(err) => {
            warn!(err = %err, "wm-brain: recall fetch failed; proceeding without context");
            Vec::new()
        }
    };
    let context = format_recall_context(&hits);
    let persona = compose_persona(&state.persona, child_lock, context.as_deref());
    let req = compose_request(model, &persona, &turn.transcript);
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
    let recall: Arc<dyn RecallSource> = Arc::new(LiveRecallSource::new(cfg.recall_sock.clone()));
    info!(
        socket = %cfg.recall_sock.display(),
        "wm-brain: recall source attached (live, connects per turn)"
    );
    let state = Arc::new({
        let base = DaemonState::new(cfg).with_recall(recall);
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

    #[derive(Clone)]
    struct CapturingLlm {
        captured: Arc<StdMutex<Vec<MessageRequest>>>,
        response: Vec<StreamEvent>,
    }

    #[async_trait::async_trait]
    impl LlmClient for CapturingLlm {
        async fn collect_messages(
            &self,
            req: &MessageRequest,
        ) -> std::result::Result<Vec<StreamEvent>, ClientError> {
            self.captured
                .lock()
                .expect("capturing llm poisoned")
                .push(req.clone());
            Ok(self.response.clone())
        }
    }

    #[derive(Clone)]
    struct FakeRecall {
        hits: Vec<QueryHit>,
    }

    #[async_trait::async_trait]
    impl RecallSource for FakeRecall {
        async fn fetch(
            &self,
            _transcript: &str,
        ) -> std::result::Result<Vec<QueryHit>, RecallSourceError> {
            Ok(self.hits.clone())
        }
    }

    #[derive(Clone)]
    struct FailingRecall;

    #[async_trait::async_trait]
    impl RecallSource for FailingRecall {
        async fn fetch(
            &self,
            _transcript: &str,
        ) -> std::result::Result<Vec<QueryHit>, RecallSourceError> {
            Err(RecallSourceError::Client(recall_client::ClientError::Remote {
                code: "boom".to_string(),
                message: "simulated".to_string(),
            }))
        }
    }

    fn fake_hit(id: &str, snippet: &str, score: f64) -> QueryHit {
        QueryHit {
            id: id.to_string(),
            kind: "fact".to_string(),
            subject: "wintermute-profile".to_string(),
            path: format!("/tmp/{id}.md"),
            snippet: snippet.to_string(),
            score,
            confidence: score,
        }
    }

    #[test]
    fn compose_persona_without_lock_or_context_returns_base() {
        assert_eq!(compose_persona("base persona", false, None), "base persona");
    }

    #[test]
    fn compose_persona_with_child_lock_appends_guard() {
        let out = compose_persona("base", true, None);
        assert!(out.starts_with("base"));
        assert!(out.contains(CHILD_LOCK_GUARD));
        assert!(out.contains("\n\n"), "blocks separated by blank line");
    }

    #[test]
    fn compose_persona_with_empty_context_does_not_add_block() {
        let out = compose_persona("base", false, Some(""));
        assert_eq!(out, "base");
    }

    #[test]
    fn compose_persona_with_context_appends_block_after_lock() {
        let out = compose_persona("base", true, Some("ctx block"));
        let lock_idx = out.find(CHILD_LOCK_GUARD).expect("guard present");
        let ctx_idx = out.find("ctx block").expect("context present");
        assert!(ctx_idx > lock_idx, "context appended after lock guard");
    }

    #[test]
    fn format_recall_context_empty_returns_none() {
        assert!(format_recall_context(&[]).is_none());
    }

    #[test]
    fn format_recall_context_renders_numbered_snippets() {
        let hits = vec![
            fake_hit("a", "She prefers chamomile tea.", 0.9),
            fake_hit("b", "Daughter is Sara.", 0.8),
        ];
        let out = format_recall_context(&hits).expect("non-empty");
        assert!(out.contains("1. She prefers chamomile tea."));
        assert!(out.contains("2. Daughter is Sara."));
        assert!(!out.contains('*'), "no markdown bullets in voice path");
    }

    #[test]
    fn format_recall_context_skips_blank_snippets() {
        let hits = vec![
            fake_hit("a", "Real one.", 0.9),
            fake_hit("b", "   ", 0.8),
        ];
        let out = format_recall_context(&hits).expect("non-empty");
        assert!(out.contains("1. Real one."));
        assert!(!out.contains("2."), "blank snippet skipped");
    }

    #[tokio::test]
    async fn null_recall_returns_empty() {
        let hits = NullRecall.fetch("anything").await.expect("never errors");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn no_tools_router_returns_stable_error_body() {
        let body = NoToolsRouter
            .dispatch("wm.weather.today", &json!({"loc": "LA"}))
            .await;
        assert!(!body.ok);
        assert_eq!(body.body["error"], "no-tools-registered");
        assert_eq!(body.body["tool"], "wm.weather.today");
    }

    #[tokio::test]
    async fn dispatch_turn_user_splices_recall_context_into_system_prompt() {
        let captured = Arc::new(StdMutex::new(Vec::new()));
        let llm = CapturingLlm {
            captured: captured.clone(),
            response: vec![text_delta("hi there")],
        };
        let cfg = BrainConfig::default();
        let state = Arc::new(
            DaemonState::new(cfg)
                .with_llm(into_dyn_llm(llm))
                .with_recall(Arc::new(FakeRecall {
                    hits: vec![fake_hit("a", "She prefers chamomile tea.", 0.9)],
                })),
        );
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "what should I drink?".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 7)
            .await
            .expect("dispatch ok");
        let calls = captured.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let system = calls[0].system.as_deref().expect("system spliced");
        assert!(system.contains("She prefers chamomile tea."));
        assert!(system.starts_with(DEFAULT_PERSONA));
        assert!(
            !system.contains(CHILD_LOCK_GUARD),
            "child-lock off in default config"
        );
    }

    #[tokio::test]
    async fn dispatch_turn_user_appends_child_lock_when_config_sets_it() {
        let captured = Arc::new(StdMutex::new(Vec::new()));
        let llm = CapturingLlm {
            captured: captured.clone(),
            response: vec![text_delta("ok")],
        };
        let cfg = BrainConfig {
            child_lock: true,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg).with_llm(into_dyn_llm(llm)));
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "tell me a joke".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 11)
            .await
            .expect("dispatch ok");
        let calls = captured.lock().unwrap();
        let system = calls[0].system.as_deref().expect("system spliced");
        assert!(system.contains(CHILD_LOCK_GUARD));
    }

    #[tokio::test]
    async fn dispatch_turn_user_proceeds_when_recall_errors() {
        let captured = Arc::new(StdMutex::new(Vec::new()));
        let llm = CapturingLlm {
            captured: captured.clone(),
            response: vec![text_delta("fallback reply")],
        };
        let cfg = BrainConfig::default();
        let state = Arc::new(
            DaemonState::new(cfg)
                .with_llm(into_dyn_llm(llm))
                .with_recall(Arc::new(FailingRecall)),
        );
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "anything".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 13)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY);
        assert_eq!(events[0].1["text"], "fallback reply");
        let calls = captured.lock().unwrap();
        let system = calls[0].system.as_deref().expect("system");
        assert_eq!(system, DEFAULT_PERSONA, "no context spliced on recall error");
    }

    #[test]
    fn live_recall_source_builder_defaults() {
        let src = LiveRecallSource::new(PathBuf::from("/tmp/x.sock"));
        assert_eq!(src.profile_subject, PROFILE_SUBJECT);
        assert_eq!(src.limit, DEFAULT_RECALL_LIMIT);
        let custom = LiveRecallSource::new(PathBuf::from("/tmp/y.sock"))
            .with_profile_subject("subj-x")
            .with_limit(3);
        assert_eq!(custom.profile_subject, "subj-x");
        assert_eq!(custom.limit, 3);
    }

    #[tokio::test]
    async fn live_recall_source_connect_failure_surfaces_client_error() {
        let src = LiveRecallSource::new(PathBuf::from("/tmp/wm-brain-iter9-nonexistent.sock"));
        let err = src.fetch("hi").await.expect_err("connect should fail");
        let RecallSourceError::Client(recall_client::ClientError::Connect { .. }) = err else {
            panic!("expected Connect error, got {err:?}");
        };
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
