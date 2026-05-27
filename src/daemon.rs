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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::anthropic::{AnthropicClient, ClientError, Message, MessageRequest, Role, StreamEvent};
use crate::bus::{
    self, ConfirmDeniedEvent, ConfirmGrantedEvent, DecodeError, Emit, ErrorEvent, ReplyEvent,
    ReplyDestructiveEvent, Request, ToolCallEvent, ToolResultEvent, TurnUserEvent, decode_request,
    now_unix_ms, outgoing,
};
use crate::recall_client::{self, QueryArgs, QueryHit, RecallClient, TouchArgs};
use crate::{BrainConfig, PROFILE_SUBJECT, canonical_model};

/// Default upper bound on tokens the daemon requests per turn.
///
/// The PRD's "one short paragraph per turn unless asked" target sits
/// comfortably under 1 KiB of tokens; the headroom leaves room for the
/// iter-10 destructive-intent JSON trailer.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// System prompt iter-8 shipped. iter-9 layered child-lock + recall
/// context via [`compose_persona`]. iter-10 appends the destructive
/// gate via [`DESTRUCTIVE_GATE_GUARD`] inside [`compose_persona`].
pub const DEFAULT_PERSONA: &str = "You are wintermute, a voice-first companion daemon. \
The user hears you spoken aloud, never reads you on a screen. \
Speak naturally and warmly in plain prose. Keep replies to one short \
paragraph per turn unless the user asks for more. Do not use markdown, \
bullet lists, code fences, or emoji — they do not speak well.";

/// Destructive-intent gating clause appended to every system prompt by
/// [`compose_persona`]. PRD §2.4.
///
/// The model is instructed to never act destructively in-line; instead
/// it ends a reply with a final fenced JSON block carrying the action,
/// a short summary, and a confirm keyword. [`parse_destructive_intent`]
/// recovers the block and the spoken text; the brain stores the pending
/// intent and waits for `wm.dialog.confirm.granted` before invoking the
/// tool router.
pub const DESTRUCTIVE_GATE_GUARD: &str = "If you intend to take any action that deletes data, \
sends a message, makes a purchase, or otherwise changes anything outside this conversation, do \
not perform the action. Instead, finish your reply with a single fenced JSON block on its own \
line. The block must contain exactly these three fields: `intent` (the tool name you would \
invoke), `summary` (one short sentence describing the change), and `confirm_keyword` (a single \
short keyword the user must say to confirm). You may optionally include an `args` object with \
tool-specific arguments. Everything before the fence is what the user will hear; the system \
will ask for spoken confirmation before any action is taken.";

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
/// the destructive-intent gate (always — PRD §2.4), then a
/// recall-context block when non-empty. Each layer is separated by a
/// blank line so the model parses them as distinct paragraphs.
#[must_use]
pub fn compose_persona(base: &str, child_lock: bool, recall_context: Option<&str>) -> String {
    let mut out = base.to_string();
    if child_lock {
        out.push_str("\n\n");
        out.push_str(CHILD_LOCK_GUARD);
    }
    out.push_str("\n\n");
    out.push_str(DESTRUCTIVE_GATE_GUARD);
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

    /// Mark a slice of memory ids as recalled-this-turn. Called after a
    /// successful LLM response so recall's outcome-feedback subsystem
    /// sees real usage signal. Default impl is a no-op so test sources
    /// that don't care about touch signal don't need to implement it.
    ///
    /// # Errors
    /// Implementations surface transport / decode failures as
    /// [`RecallSourceError`]; the caller logs them and proceeds (touch
    /// failures must not break the conversation).
    async fn touch(&self, _ids: &[&str]) -> Result<(), RecallSourceError> {
        Ok(())
    }
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

    async fn touch(&self, ids: &[&str]) -> Result<(), RecallSourceError> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut client = RecallClient::connect(&self.socket).await?;
        for id in ids {
            let args = TouchArgs { id: (*id).to_string() };
            client.touch(&args).await?;
        }
        Ok(())
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

/// Parsed shape of the model's final-fenced JSON block (PRD §2.4).
///
/// `intent` is the tool name the brain will dispatch on
/// `wm.dialog.confirm.granted`; `args` carries optional tool-specific
/// arguments. `summary` and `confirm_keyword` are echoed verbatim into
/// the [`ReplyDestructiveEvent`] for `wm-dialog`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestructiveIntent {
    /// Tool name the model wants to invoke (e.g. `wm.fs.rm`).
    pub intent: String,
    /// One short sentence describing what the action will do.
    pub summary: String,
    /// Single short keyword the user must speak to confirm.
    pub confirm_keyword: String,
    /// Optional tool arguments — passed straight to the router on grant.
    pub args: Option<Value>,
}

/// Extract a destructive intent from the assistant's final fenced JSON.
///
/// Returns `None` when the text has no trailing fence, the fence
/// payload is not parseable JSON, or any required field is missing.
///
/// On `Some`, the second tuple element is the *spoken* text — everything
/// up to (but not including) the opening fence, with trailing whitespace
/// trimmed. Callers publish that as `wm.brain.reply.destructive.text`
/// for `wm-dialog` to hand to `wm-tts`.
#[must_use]
pub fn parse_destructive_intent(text: &str) -> Option<(DestructiveIntent, String)> {
    const FENCE: &str = "```";
    let close_idx = text.rfind(FENCE)?;
    let before_close = text.get(..close_idx)?;
    let open_idx = before_close.rfind(FENCE)?;
    let inside_start = open_idx.checked_add(FENCE.len())?;
    let inside = text.get(inside_start..close_idx)?;
    // Strip an optional language label (e.g. ```json\n…).
    let json_str = match inside.split_once('\n') {
        Some((label, rest))
            if !label.is_empty()
                && label
                    .trim()
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') =>
        {
            rest
        }
        _ => inside,
    };
    let v: Value = serde_json::from_str(json_str.trim()).ok()?;
    let intent = v.get("intent")?.as_str()?.to_string();
    let summary = v.get("summary")?.as_str()?.to_string();
    let confirm_keyword = v.get("confirm_keyword")?.as_str()?.to_string();
    if intent.is_empty() || summary.is_empty() || confirm_keyword.is_empty() {
        return None;
    }
    let args = v.get("args").cloned();
    let spoken = text.get(..open_idx)?.trim_end().to_string();
    Some((
        DestructiveIntent {
            intent,
            summary,
            confirm_keyword,
            args,
        },
        spoken,
    ))
}

/// In-flight destructive intent awaiting `wm.dialog.confirm.granted` or
/// `wm.dialog.confirm.denied`. Stored on [`DaemonState`] keyed by
/// `intent_id`; removed by either confirm path.
#[derive(Debug, Clone)]
pub struct PendingIntent {
    /// Original parsed intent.
    pub intent: DestructiveIntent,
    /// Unix milliseconds when the destructive reply was published.
    pub published_ts: u64,
}

/// Short cancellation reply the brain emits on
/// `wm.dialog.confirm.denied` for a pending intent.
pub const DESTRUCTIVE_CANCELLATION_REPLY: &str = "Okay, I won't do that.";

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
/// iter-10 adds the destructive-intent pending registry + per-process
/// counter used by [`DaemonState::mint_intent_id`].
pub struct DaemonState {
    /// Resolved runtime config (model defaults, recall socket, …).
    pub config: Mutex<BrainConfig>,
    /// On-disk config path. When set, AC6 pending-model consumption is
    /// persisted via [`BrainConfig::save_to_file`] after the turn that
    /// used it. `None` keeps consume in-memory (tests, headless runs
    /// without an XDG location).
    pub config_path: Option<PathBuf>,
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
    /// Destructive intents awaiting `wm.dialog.confirm.granted` /
    /// `wm.dialog.confirm.denied`, keyed by `intent_id`.
    pub pending: Mutex<HashMap<String, PendingIntent>>,
    /// Monotonic per-process counter spliced into minted `intent_id`s.
    pub intent_counter: AtomicU64,
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
            config_path: None,
            llm: None,
            recall: Arc::new(NullRecall),
            tool_router: Arc::new(NoToolsRouter),
            persona: DEFAULT_PERSONA.to_string(),
            pending: Mutex::new(HashMap::new()),
            intent_counter: AtomicU64::new(0),
        }
    }

    /// Attach the on-disk config path that AC6 pending-model consume
    /// should persist to. `None` (the default) keeps consumption
    /// in-memory.
    #[must_use]
    pub fn with_config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }

    /// Mint a stable id for a destructive intent. Combines the
    /// publish-time millisecond stamp with a per-process counter so two
    /// intents minted in the same millisecond never collide.
    #[must_use]
    pub fn mint_intent_id(&self, now_ms: u64) -> String {
        let seq = self.intent_counter.fetch_add(1, Ordering::SeqCst);
        format!("int-{now_ms}-{seq}")
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
/// iter-10 wires the destructive-intent gate end-to-end:
/// [`Request::TurnUser`] runs the conversation cycle and may stash a
/// [`PendingIntent`]; [`Request::ConfirmGranted`] redeems a pending
/// intent (dispatch via the tool router and publish
/// [`outgoing::TOOL_CALL`] + [`outgoing::TOOL_RESULT`]);
/// [`Request::ConfirmDenied`] drops a pending intent and publishes a
/// short cancellation reply. Unknown confirm ids publish
/// [`outgoing::ERROR`] with `kind=confirm`.
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
    match req {
        Request::TurnUser(t) => {
            // AC6: `wmd --model opus` for the next turn only. Read
            // effective_model + child_lock and, if pending_model was
            // set, clear it now so the *next* turn uses the default.
            // Persistence happens after the turn handler returns so a
            // crash during dispatch doesn't strand an empty pending on
            // disk.
            let (model, child_lock, consumed_pending) = {
                let mut cfg = state.config.lock().await;
                let model = cfg.effective_model().to_string();
                let had_pending = cfg.pending_model.is_some();
                if had_pending {
                    cfg.consume_pending();
                }
                (model, cfg.child_lock, had_pending)
            };
            handle_turn_user(state, publish, &model, child_lock, &t, now_ms).await?;
            if consumed_pending {
                persist_after_pending_consume(state).await;
            }
        }
        Request::ConfirmGranted(c) => {
            handle_confirm_granted(state, publish, &c, now_ms).await?;
        }
        Request::ConfirmDenied(c) => {
            handle_confirm_denied(state, publish, &c, now_ms).await?;
        }
    }
    Ok(())
}

/// Persist the post-consume config when [`DaemonState::config_path`] is
/// set. Save failures are logged and swallowed: the in-memory cfg has
/// already been mutated, so the running daemon keeps the AC6 contract;
/// only the on-disk record lags until the next successful save.
async fn persist_after_pending_consume(state: &DaemonState) {
    let Some(path) = state.config_path.as_ref() else {
        return;
    };
    let cfg_snapshot = { state.config.lock().await.clone() };
    if let Err(err) = cfg_snapshot.save_to_file(path) {
        warn!(
            err = %err,
            path = %path.display(),
            "wm-brain: persisting post-consume config failed; in-memory state still reverted"
        );
    }
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
            touch_recalled_hits(state.recall.as_ref(), &hits).await;
            if let Some((intent, spoken)) = parse_destructive_intent(&text) {
                publish_destructive(state, publish, intent, spoken, now_ms).await?;
            } else {
                let reply = ReplyEvent { text, ts: now_ms };
                publish
                    .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
                    .await
                    .context("publish reply")?;
            }
        }
        Err(err) => {
            error!(err = %err, model = %model, "wm-brain: anthropic call failed");
            publish_error_at(publish, "anthropic", &format!("{err}"), now_ms).await?;
        }
    }
    Ok(())
}

/// Fire-and-log [`RecallSource::touch`] for each id in `hits`. Touch
/// failures must not break the conversation, so transport errors are
/// logged and swallowed (recall outage degrades to "no usage signal",
/// not "dropped reply").
async fn touch_recalled_hits(recall: &dyn RecallSource, hits: &[QueryHit]) {
    if hits.is_empty() {
        return;
    }
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    if let Err(err) = recall.touch(&ids).await {
        warn!(err = %err, "wm-brain: recall touch failed; usage signal dropped");
    }
}

/// Mint an `intent_id`, stash the [`PendingIntent`] under it, and
/// publish [`outgoing::REPLY_DESTRUCTIVE`]. The spoken text is whatever
/// the model emitted before the fenced JSON block; if the model
/// volunteered only the JSON, the summary doubles as the spoken text
/// so `wm-dialog` always has something for `wm-tts` to voice.
async fn publish_destructive(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    intent: DestructiveIntent,
    spoken: String,
    now_ms: u64,
) -> Result<()> {
    let intent_id = state.mint_intent_id(now_ms);
    let action = json!({
        "tool": intent.intent.clone(),
        "args": intent.args.clone().unwrap_or_else(|| json!({})),
    });
    let text = if spoken.trim().is_empty() {
        intent.summary.clone()
    } else {
        spoken
    };
    let summary = intent.summary.clone();
    let confirm_keyword = intent.confirm_keyword.clone();
    {
        let mut pending = state.pending.lock().await;
        pending.insert(
            intent_id.clone(),
            PendingIntent {
                intent,
                published_ts: now_ms,
            },
        );
    }
    let event = ReplyDestructiveEvent {
        text,
        intent_id,
        summary,
        confirm_keyword,
        action,
        ts: now_ms,
    };
    publish
        .publish(outgoing::REPLY_DESTRUCTIVE, serde_json::to_value(&event)?)
        .await
        .context("publish reply.destructive")
}

async fn handle_confirm_granted(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    c: &ConfirmGrantedEvent,
    now_ms: u64,
) -> Result<()> {
    let pending = {
        let mut map = state.pending.lock().await;
        map.remove(&c.intent_id)
    };
    let Some(pi) = pending else {
        warn!(intent_id = %c.intent_id, "wm-brain: confirm.granted for unknown intent_id");
        publish_error_at(
            publish,
            "confirm",
            &format!("unknown intent_id: {}", c.intent_id),
            now_ms,
        )
        .await?;
        return Ok(());
    };
    let tool = pi.intent.intent;
    let args = pi.intent.args.unwrap_or_else(|| json!({}));
    let call = ToolCallEvent {
        tool: tool.clone(),
        args: args.clone(),
        ts: now_ms,
    };
    publish
        .publish(outgoing::TOOL_CALL, serde_json::to_value(&call)?)
        .await
        .context("publish tool.call")?;
    let result = state.tool_router.dispatch(&tool, &args).await;
    let result_event = ToolResultEvent {
        tool,
        ok: result.ok,
        body: result.body,
        ts: now_ms,
    };
    publish
        .publish(outgoing::TOOL_RESULT, serde_json::to_value(&result_event)?)
        .await
        .context("publish tool.result")?;
    Ok(())
}

async fn handle_confirm_denied(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    c: &ConfirmDeniedEvent,
    now_ms: u64,
) -> Result<()> {
    let removed = {
        let mut map = state.pending.lock().await;
        map.remove(&c.intent_id)
    };
    if removed.is_none() {
        debug!(intent_id = %c.intent_id, "wm-brain: confirm.denied for unknown intent_id; dropping");
        return Ok(());
    }
    debug!(intent_id = %c.intent_id, reason = %c.reason, "wm-brain: confirm.denied dropped pending intent");
    let reply = ReplyEvent {
        text: DESTRUCTIVE_CANCELLATION_REPLY.to_string(),
        ts: now_ms,
    };
    publish
        .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
        .await
        .context("publish cancellation reply")
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
pub async fn run(cfg: BrainConfig, config_path: Option<PathBuf>) -> Result<()> {
    cfg.validate().context("wm-brain: config validation failed")?;

    let llm = build_llm_from_env(&cfg.api_key_env);
    let recall: Arc<dyn RecallSource> = Arc::new(LiveRecallSource::new(cfg.recall_sock.clone()));
    info!(
        socket = %cfg.recall_sock.display(),
        "wm-brain: recall source attached (live, connects per turn)"
    );
    let state = Arc::new({
        let mut base = DaemonState::new(cfg).with_recall(recall);
        if let Some(p) = config_path {
            base = base.with_config_path(p);
        }
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
    clippy::await_holding_lock,
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
    async fn dispatch_confirm_granted_unknown_id_publishes_error() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let req = Request::ConfirmGranted(ConfirmGrantedEvent {
            intent_id: "abc".to_string(),
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 1234)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::ERROR);
        assert_eq!(events[0].1["kind"], "confirm");
        assert!(
            events[0].1["message"].as_str().unwrap().contains("abc"),
            "error should name the unknown intent_id"
        );
        assert_eq!(events[0].1["ts"], 1234);
    }

    #[tokio::test]
    async fn dispatch_confirm_denied_unknown_id_is_silent() {
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
        touched: Arc<StdMutex<Vec<Vec<String>>>>,
    }

    impl FakeRecall {
        fn new(hits: Vec<QueryHit>) -> Self {
            Self {
                hits,
                touched: Arc::new(StdMutex::new(Vec::new())),
            }
        }

        fn touched_calls(&self) -> Vec<Vec<String>> {
            self.touched.lock().expect("touched poisoned").clone()
        }
    }

    #[async_trait::async_trait]
    impl RecallSource for FakeRecall {
        async fn fetch(
            &self,
            _transcript: &str,
        ) -> std::result::Result<Vec<QueryHit>, RecallSourceError> {
            Ok(self.hits.clone())
        }

        async fn touch(&self, ids: &[&str]) -> std::result::Result<(), RecallSourceError> {
            self.touched
                .lock()
                .expect("touched poisoned")
                .push(ids.iter().map(|s| (*s).to_string()).collect());
            Ok(())
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

    /// Recall source whose `fetch` returns hits but `touch` always
    /// fails. Used to verify touch-failure tolerance in `handle_turn_user`.
    #[derive(Clone)]
    struct TouchFailingRecall {
        hits: Vec<QueryHit>,
    }

    #[async_trait::async_trait]
    impl RecallSource for TouchFailingRecall {
        async fn fetch(
            &self,
            _transcript: &str,
        ) -> std::result::Result<Vec<QueryHit>, RecallSourceError> {
            Ok(self.hits.clone())
        }

        async fn touch(&self, _ids: &[&str]) -> std::result::Result<(), RecallSourceError> {
            Err(RecallSourceError::Client(recall_client::ClientError::Remote {
                code: "touch_failed".to_string(),
                message: "simulated touch boom".to_string(),
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
    fn compose_persona_without_lock_or_context_appends_destructive_guard() {
        let out = compose_persona("base persona", false, None);
        assert!(out.starts_with("base persona"));
        assert!(out.contains(DESTRUCTIVE_GATE_GUARD));
        assert!(
            !out.contains(CHILD_LOCK_GUARD),
            "child-lock off in this case"
        );
    }

    #[test]
    fn compose_persona_with_child_lock_appends_guard_before_destructive() {
        let out = compose_persona("base", true, None);
        assert!(out.starts_with("base"));
        let lock_idx = out.find(CHILD_LOCK_GUARD).expect("lock guard present");
        let dest_idx = out
            .find(DESTRUCTIVE_GATE_GUARD)
            .expect("destructive guard present");
        assert!(dest_idx > lock_idx, "destructive guard follows lock guard");
        assert!(out.contains("\n\n"), "blocks separated by blank line");
    }

    #[test]
    fn compose_persona_with_empty_context_still_includes_destructive_guard() {
        let out = compose_persona("base", false, Some(""));
        assert!(out.starts_with("base"));
        assert!(out.contains(DESTRUCTIVE_GATE_GUARD));
        assert!(
            !out.contains("ctx"),
            "empty context block must not be appended"
        );
    }

    #[test]
    fn compose_persona_with_context_appends_block_after_destructive_guard() {
        let out = compose_persona("base", true, Some("ctx block"));
        let lock_idx = out.find(CHILD_LOCK_GUARD).expect("lock guard present");
        let dest_idx = out
            .find(DESTRUCTIVE_GATE_GUARD)
            .expect("destructive guard present");
        let ctx_idx = out.find("ctx block").expect("context present");
        assert!(dest_idx > lock_idx, "destructive guard follows lock guard");
        assert!(ctx_idx > dest_idx, "context follows destructive guard");
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
                .with_recall(Arc::new(FakeRecall::new(vec![fake_hit(
                    "a",
                    "She prefers chamomile tea.",
                    0.9,
                )]))),
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
        assert!(
            system.starts_with(DEFAULT_PERSONA),
            "persona base preserved on recall error"
        );
        assert!(
            system.contains(DESTRUCTIVE_GATE_GUARD),
            "destructive gate guard always present"
        );
        assert!(
            !system.contains("She prefers"),
            "no recall context spliced on recall error"
        );
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
    async fn live_recall_source_touch_empty_ids_is_noop_no_connect() {
        let src = LiveRecallSource::new(PathBuf::from(
            "/tmp/wm-brain-iter11-touch-noop-nonexistent.sock",
        ));
        src.touch(&[]).await.expect("empty touch must not connect");
    }

    #[tokio::test]
    async fn live_recall_source_touch_nonempty_attempts_connect() {
        let src = LiveRecallSource::new(PathBuf::from(
            "/tmp/wm-brain-iter11-touch-fail-nonexistent.sock",
        ));
        let err = src.touch(&["m-1"]).await.expect_err("connect should fail");
        let RecallSourceError::Client(recall_client::ClientError::Connect { .. }) = err else {
            panic!("expected Connect error, got {err:?}");
        };
    }

    #[tokio::test]
    async fn dispatch_turn_user_touches_recalled_ids_on_successful_reply() {
        let llm = FakeLlm {
            response: Ok(vec![text_delta("brewing chamomile now")]),
        };
        let recall = FakeRecall::new(vec![
            fake_hit("m-1", "She prefers chamomile tea.", 0.9),
            fake_hit("m-2", "Loose-leaf only.", 0.8),
        ]);
        let recall_handle = recall.clone();
        let state = Arc::new(
            DaemonState::new(BrainConfig::default())
                .with_llm(into_dyn_llm(llm))
                .with_recall(Arc::new(recall)),
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
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY);
        let calls = recall_handle.touched_calls();
        assert_eq!(calls.len(), 1, "exactly one touch call per successful turn");
        assert_eq!(calls[0], vec!["m-1".to_string(), "m-2".to_string()]);
    }

    #[tokio::test]
    async fn dispatch_turn_user_does_not_touch_when_no_recall_hits() {
        let llm = FakeLlm {
            response: Ok(vec![text_delta("ok")]),
        };
        let recall = FakeRecall::new(Vec::new());
        let recall_handle = recall.clone();
        let state = Arc::new(
            DaemonState::new(BrainConfig::default())
                .with_llm(into_dyn_llm(llm))
                .with_recall(Arc::new(recall)),
        );
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hi".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 7)
            .await
            .expect("dispatch ok");
        assert!(
            recall_handle.touched_calls().is_empty(),
            "no hits means no touch call"
        );
    }

    #[tokio::test]
    async fn dispatch_turn_user_does_not_touch_when_text_empty() {
        let llm = FakeLlm {
            response: Ok(vec![
                StreamEvent::MessageStart,
                StreamEvent::MessageDelta { stop_reason: None },
                StreamEvent::MessageStop,
            ]),
        };
        let recall = FakeRecall::new(vec![fake_hit("m-1", "fact", 0.9)]);
        let recall_handle = recall.clone();
        let state = Arc::new(
            DaemonState::new(BrainConfig::default())
                .with_llm(into_dyn_llm(llm))
                .with_recall(Arc::new(recall)),
        );
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hi".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 7)
            .await
            .expect("dispatch ok");
        assert!(
            recall_handle.touched_calls().is_empty(),
            "empty-text errors must not touch — recall didn't actually feed a reply"
        );
    }

    #[tokio::test]
    async fn dispatch_turn_user_does_not_touch_when_llm_errors() {
        let llm = FakeLlm {
            response: Err("upstream 503"),
        };
        let recall = FakeRecall::new(vec![fake_hit("m-1", "fact", 0.9)]);
        let recall_handle = recall.clone();
        let state = Arc::new(
            DaemonState::new(BrainConfig::default())
                .with_llm(into_dyn_llm(llm))
                .with_recall(Arc::new(recall)),
        );
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hi".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 7)
            .await
            .expect("dispatch ok");
        assert!(
            recall_handle.touched_calls().is_empty(),
            "llm transport errors must not touch — no successful reply"
        );
    }

    #[tokio::test]
    async fn dispatch_turn_user_touches_on_destructive_intent_reply() {
        // Destructive intents still represent a successful LLM response
        // that consumed recall context — they should bump recall_count.
        let body = "I'll do that.\n```json\n{\"intent\":\"wm.fs.rm\",\
            \"summary\":\"delete x\",\"confirm_keyword\":\"delete\"}\n```";
        let llm = FakeLlm {
            response: Ok(vec![text_delta(body)]),
        };
        let recall = FakeRecall::new(vec![fake_hit("m-1", "context", 0.9)]);
        let recall_handle = recall.clone();
        let state = Arc::new(
            DaemonState::new(BrainConfig::default())
                .with_llm(into_dyn_llm(llm))
                .with_recall(Arc::new(recall)),
        );
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "delete that".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 7)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY_DESTRUCTIVE);
        let calls = recall_handle.touched_calls();
        assert_eq!(calls.len(), 1, "destructive reply still touches recall");
        assert_eq!(calls[0], vec!["m-1".to_string()]);
    }

    #[tokio::test]
    async fn dispatch_turn_user_swallows_touch_errors() {
        let llm = FakeLlm {
            response: Ok(vec![text_delta("brewing")]),
        };
        let recall = TouchFailingRecall {
            hits: vec![fake_hit("m-1", "fact", 0.9)],
        };
        let state = Arc::new(
            DaemonState::new(BrainConfig::default())
                .with_llm(into_dyn_llm(llm))
                .with_recall(Arc::new(recall)),
        );
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hi".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 7)
            .await
            .expect("dispatch must succeed even when touch fails");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1, "reply still published despite touch failure");
        assert_eq!(events[0].0, outgoing::REPLY);
        assert_eq!(events[0].1["text"], "brewing");
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

    // -- iter-10: destructive-intent gate ------------------------------

    #[test]
    fn parse_destructive_intent_happy_with_json_label() {
        let text = "Sure — about to delete the file.\n```json\n{\"intent\": \"wm.fs.rm\", \"summary\": \"delete /tmp/x\", \"confirm_keyword\": \"delete\"}\n```";
        let (intent, spoken) = parse_destructive_intent(text).expect("parses");
        assert_eq!(intent.intent, "wm.fs.rm");
        assert_eq!(intent.summary, "delete /tmp/x");
        assert_eq!(intent.confirm_keyword, "delete");
        assert!(intent.args.is_none());
        assert_eq!(spoken, "Sure — about to delete the file.");
    }

    #[test]
    fn parse_destructive_intent_happy_without_language_label() {
        let text = "About to send.\n```\n{\"intent\":\"wm.mail.send\",\"summary\":\"send to mom\",\"confirm_keyword\":\"send\"}\n```";
        let (intent, _spoken) = parse_destructive_intent(text).expect("parses");
        assert_eq!(intent.intent, "wm.mail.send");
    }

    #[test]
    fn parse_destructive_intent_carries_optional_args() {
        let text = "```json\n{\"intent\":\"wm.fs.rm\",\"summary\":\"s\",\"confirm_keyword\":\"k\",\"args\":{\"path\":\"/tmp/x\"}}\n```";
        let (intent, _spoken) = parse_destructive_intent(text).expect("parses");
        let args = intent.args.expect("args carried through");
        assert_eq!(args["path"], "/tmp/x");
    }

    #[test]
    fn parse_destructive_intent_returns_none_without_fence() {
        let text = "Just chatting, no JSON.";
        assert!(parse_destructive_intent(text).is_none());
    }

    #[test]
    fn parse_destructive_intent_returns_none_on_malformed_json() {
        let text = "```json\n{not valid json}\n```";
        assert!(parse_destructive_intent(text).is_none());
    }

    #[test]
    fn parse_destructive_intent_returns_none_on_missing_required_field() {
        // intent + summary but no confirm_keyword
        let text = "```json\n{\"intent\":\"x\",\"summary\":\"s\"}\n```";
        assert!(parse_destructive_intent(text).is_none());
    }

    #[test]
    fn parse_destructive_intent_returns_none_on_empty_required_field() {
        let text = "```json\n{\"intent\":\"\",\"summary\":\"s\",\"confirm_keyword\":\"k\"}\n```";
        assert!(parse_destructive_intent(text).is_none());
    }

    #[test]
    fn parse_destructive_intent_uses_final_block_when_multiple_present() {
        let text = "First block:\n```json\n{\"intent\":\"first\",\"summary\":\"s\",\"confirm_keyword\":\"k\"}\n```\nNow the real one:\n```json\n{\"intent\":\"wm.last\",\"summary\":\"final\",\"confirm_keyword\":\"go\"}\n```";
        let (intent, spoken) = parse_destructive_intent(text).expect("parses");
        assert_eq!(intent.intent, "wm.last");
        assert!(spoken.contains("First block"));
        assert!(spoken.contains("Now the real one:"));
    }

    #[test]
    fn mint_intent_id_is_unique_per_call() {
        let cfg = BrainConfig::default();
        let state = DaemonState::new(cfg);
        let a = state.mint_intent_id(1000);
        let b = state.mint_intent_id(1000);
        let c = state.mint_intent_id(2000);
        assert_ne!(a, b, "two ids minted same ms must differ");
        assert!(a.starts_with("int-1000-"));
        assert!(c.starts_with("int-2000-"));
    }

    #[tokio::test]
    async fn dispatch_turn_user_publishes_destructive_when_block_present() {
        let llm = FakeLlm {
            response: Ok(vec![text_delta(
                "About to delete /tmp/x.\n```json\n{\"intent\":\"wm.fs.rm\",\"summary\":\"delete /tmp/x\",\"confirm_keyword\":\"delete\",\"args\":{\"path\":\"/tmp/x\"}}\n```",
            )]),
        };
        let state = state_with_llm(llm);
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "remove /tmp/x".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 5555)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY_DESTRUCTIVE);
        assert_eq!(events[0].1["text"], "About to delete /tmp/x.");
        assert_eq!(events[0].1["summary"], "delete /tmp/x");
        assert_eq!(events[0].1["confirm_keyword"], "delete");
        assert_eq!(events[0].1["action"]["tool"], "wm.fs.rm");
        assert_eq!(events[0].1["action"]["args"]["path"], "/tmp/x");
        assert_eq!(events[0].1["ts"], 5555);
        let intent_id = events[0].1["intent_id"]
            .as_str()
            .expect("intent_id present")
            .to_string();
        assert!(intent_id.starts_with("int-5555-"));
        drop(events);
        // PendingIntent stashed for later confirm.
        let pending = state.pending.lock().await;
        assert!(pending.contains_key(&intent_id));
    }

    #[tokio::test]
    async fn dispatch_turn_user_uses_summary_as_spoken_text_when_only_fence() {
        let llm = FakeLlm {
            response: Ok(vec![text_delta(
                "```json\n{\"intent\":\"wm.fs.rm\",\"summary\":\"delete /tmp/x\",\"confirm_keyword\":\"delete\"}\n```",
            )]),
        };
        let state = state_with_llm(llm);
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "rm".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 1)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY_DESTRUCTIVE);
        assert_eq!(events[0].1["text"], "delete /tmp/x");
    }

    #[tokio::test]
    async fn dispatch_turn_user_passes_through_reply_when_no_block() {
        let llm = FakeLlm {
            response: Ok(vec![text_delta("Just a chat reply.")]),
        };
        let state = state_with_llm(llm);
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hi".to_string(),
            confidence: 1.0,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 9)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY);
        let pending = state.pending.lock().await;
        assert!(pending.is_empty(), "no destructive intent → no pending entry");
    }

    #[derive(Default, Clone)]
    struct RecordingRouter {
        calls: Arc<StdMutex<Vec<(String, Value)>>>,
    }

    #[async_trait::async_trait]
    impl ToolRouter for RecordingRouter {
        async fn dispatch(&self, name: &str, args: &Value) -> ToolResultBody {
            self.calls
                .lock()
                .expect("router poisoned")
                .push((name.to_string(), args.clone()));
            ToolResultBody {
                ok: true,
                body: json!({ "executed": name }),
            }
        }
    }

    #[tokio::test]
    async fn dispatch_confirm_granted_redeems_pending_via_tool_router() {
        let router = RecordingRouter::default();
        let calls = router.calls.clone();
        let cfg = BrainConfig::default();
        let state = Arc::new(DaemonState::new(cfg).with_tool_router(Arc::new(router)));
        let intent_id = state.mint_intent_id(100);
        {
            let mut pending = state.pending.lock().await;
            pending.insert(
                intent_id.clone(),
                PendingIntent {
                    intent: DestructiveIntent {
                        intent: "wm.fs.rm".to_string(),
                        summary: "delete /tmp/x".to_string(),
                        confirm_keyword: "delete".to_string(),
                        args: Some(json!({ "path": "/tmp/x" })),
                    },
                    published_ts: 100,
                },
            );
        }
        let mut sink = MemSink::default();
        let req = Request::ConfirmGranted(ConfirmGrantedEvent {
            intent_id: intent_id.clone(),
            ts: 200,
        });
        dispatch(state.as_ref(), &mut sink, req, 333)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2, "tool.call + tool.result");
        assert_eq!(events[0].0, outgoing::TOOL_CALL);
        assert_eq!(events[0].1["tool"], "wm.fs.rm");
        assert_eq!(events[0].1["args"]["path"], "/tmp/x");
        assert_eq!(events[0].1["ts"], 333);
        assert_eq!(events[1].0, outgoing::TOOL_RESULT);
        assert_eq!(events[1].1["tool"], "wm.fs.rm");
        assert_eq!(events[1].1["ok"], true);
        assert_eq!(events[1].1["body"]["executed"], "wm.fs.rm");
        drop(events);
        let router_calls = calls.lock().unwrap();
        assert_eq!(router_calls.len(), 1);
        assert_eq!(router_calls[0].0, "wm.fs.rm");
        let pending = state.pending.lock().await;
        assert!(
            !pending.contains_key(&intent_id),
            "pending intent removed on grant"
        );
    }

    #[tokio::test]
    async fn dispatch_confirm_denied_with_pending_publishes_cancellation_and_drops() {
        let cfg = BrainConfig::default();
        let state = Arc::new(DaemonState::new(cfg));
        let intent_id = state.mint_intent_id(100);
        {
            let mut pending = state.pending.lock().await;
            pending.insert(
                intent_id.clone(),
                PendingIntent {
                    intent: DestructiveIntent {
                        intent: "wm.fs.rm".to_string(),
                        summary: "s".to_string(),
                        confirm_keyword: "k".to_string(),
                        args: None,
                    },
                    published_ts: 100,
                },
            );
        }
        let mut sink = MemSink::default();
        let req = Request::ConfirmDenied(ConfirmDeniedEvent {
            intent_id: intent_id.clone(),
            reason: "negative_keyword".to_string(),
            ts: 200,
        });
        dispatch(state.as_ref(), &mut sink, req, 444)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY);
        assert_eq!(events[0].1["text"], DESTRUCTIVE_CANCELLATION_REPLY);
        assert_eq!(events[0].1["ts"], 444);
        drop(events);
        let pending = state.pending.lock().await;
        assert!(
            !pending.contains_key(&intent_id),
            "pending intent removed on deny"
        );
    }

    // PRD §4 AC5: 10 scripted destructive prompts each produce a
    // wm.brain.reply.destructive event with valid intent_id +
    // confirm_keyword; none execute via the tool router until a
    // wm.dialog.confirm.granted message arrives.
    #[tokio::test]
    #[allow(clippy::too_many_lines, reason = "table-driven AC5 verification fans the assertion set across 10 prompts inline")]
    async fn ac5_ten_scripted_destructive_prompts() {
        let scripts: [(&str, &str, &str, &str, &str); 10] = [
            (
                "remove /tmp/x",
                "wm.fs.rm",
                "delete /tmp/x",
                "delete",
                "About to delete /tmp/x.",
            ),
            (
                "rename Cargo.toml to Cargo.toml.bak",
                "wm.fs.mv",
                "rename Cargo.toml",
                "rename",
                "About to rename Cargo.toml.",
            ),
            (
                "send mom an email saying I'm late",
                "wm.mail.send",
                "email mom that I'm late",
                "send",
                "Drafting an email to mom.",
            ),
            (
                "delete the message from Bob",
                "wm.mail.delete",
                "remove Bob's note",
                "remove",
                "Removing Bob's message.",
            ),
            (
                "cancel my dentist appointment",
                "wm.cal.cancel",
                "cancel dentist event",
                "cancel",
                "About to cancel the dentist event.",
            ),
            (
                "order another bag of coffee",
                "wm.purchase.place_order",
                "buy 12oz beans",
                "order",
                "Placing the coffee order.",
            ),
            (
                "forget that I prefer green tea",
                "wm.recall.delete",
                "drop tea preference",
                "forget",
                "About to remove that memory.",
            ),
            (
                "submit the registration form",
                "wm.browser.nav.destructive",
                "submit registration form",
                "submit",
                "Submitting the form now.",
            ),
            (
                "skip the next three songs in the queue",
                "wm.music.skip",
                "skip 3 in queue",
                "skip",
                "Skipping the next three.",
            ),
            (
                "shut everything down",
                "wm.fleet2.shell",
                "shutdown laptop",
                "yes",
                "About to shut down.",
            ),
        ];

        let mut minted_ids: Vec<String> = Vec::new();
        for (i, (utterance, tool, summary, keyword, pretext)) in scripts.iter().enumerate() {
            let idx = u64::try_from(i).expect("script index fits in u64");
            // Half deny, half grant — neither path is allowed to invoke
            // the tool router before the dialog confirmation arrives.
            let grant = i % 2 == 0;
            let body = format!(
                "{pretext}\n```json\n{{\"intent\":\"{tool}\",\"summary\":\"{summary}\",\"confirm_keyword\":\"{keyword}\"}}\n```"
            );
            let llm = FakeLlm {
                response: Ok(vec![text_delta(&body)]),
            };
            let router = RecordingRouter::default();
            let router_calls = router.calls.clone();
            let cfg = BrainConfig::default();
            let state = Arc::new(
                DaemonState::new(cfg)
                    .with_llm(into_dyn_llm(llm))
                    .with_tool_router(Arc::new(router)),
            );
            let mut sink = MemSink::default();
            let turn_now = 5_000_u64 + idx;
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: (*utterance).to_string(),
                    confidence: 1.0,
                    ts: 1_000 + idx,
                }),
                turn_now,
            )
            .await
            .expect("turn-user dispatch ok");

            let intent_id = {
                let events = sink.events.lock().unwrap();
                assert_eq!(events.len(), 1, "{utterance}: exactly one destructive event");
                assert_eq!(
                    events[0].0,
                    outgoing::REPLY_DESTRUCTIVE,
                    "{utterance}: must publish destructive topic"
                );
                assert_eq!(
                    events[0].1["confirm_keyword"], *keyword,
                    "{utterance}: confirm_keyword echoed"
                );
                assert_eq!(
                    events[0].1["summary"], *summary,
                    "{utterance}: summary echoed"
                );
                assert_eq!(
                    events[0].1["action"]["tool"], *tool,
                    "{utterance}: action tool routed"
                );
                let id = events[0].1["intent_id"]
                    .as_str()
                    .expect("intent_id present")
                    .to_string();
                assert!(
                    id.starts_with(&format!("int-{turn_now}-")),
                    "{utterance}: intent_id should embed dispatch ts"
                );
                id
            };
            assert!(
                !minted_ids.contains(&intent_id),
                "{utterance}: intent_id must be unique across scripts"
            );
            minted_ids.push(intent_id.clone());

            // AC5 invariant: nothing executed yet.
            {
                let calls = router_calls.lock().unwrap();
                assert!(
                    calls.is_empty(),
                    "{utterance}: tool router must NOT execute pre-confirmation"
                );
            }
            // PendingIntent stashed under the minted id.
            {
                let pending = state.pending.lock().await;
                assert!(
                    pending.contains_key(&intent_id),
                    "{utterance}: pending intent stashed under minted id"
                );
            }

            // Isolate the confirm-step events from the destructive event.
            sink.events.lock().unwrap().clear();
            let confirm_now = 9_000_u64 + idx;
            if grant {
                dispatch(
                    state.as_ref(),
                    &mut sink,
                    Request::ConfirmGranted(ConfirmGrantedEvent {
                        intent_id: intent_id.clone(),
                        ts: 8_000 + idx,
                    }),
                    confirm_now,
                )
                .await
                .expect("confirm.granted dispatch ok");
                let events = sink.events.lock().unwrap();
                assert_eq!(events.len(), 2, "{utterance}: tool.call + tool.result");
                assert_eq!(events[0].0, outgoing::TOOL_CALL);
                assert_eq!(events[0].1["tool"], *tool);
                assert_eq!(events[1].0, outgoing::TOOL_RESULT);
                assert_eq!(events[1].1["ok"], true);
                drop(events);
                let calls = router_calls.lock().unwrap();
                assert_eq!(
                    calls.len(),
                    1,
                    "{utterance}: router executed exactly once on grant"
                );
                assert_eq!(calls[0].0, *tool);
            } else {
                dispatch(
                    state.as_ref(),
                    &mut sink,
                    Request::ConfirmDenied(ConfirmDeniedEvent {
                        intent_id: intent_id.clone(),
                        reason: "negative_keyword".to_string(),
                        ts: 8_000 + idx,
                    }),
                    confirm_now,
                )
                .await
                .expect("confirm.denied dispatch ok");
                let events = sink.events.lock().unwrap();
                assert_eq!(events.len(), 1, "{utterance}: cancellation reply published");
                assert_eq!(events[0].0, outgoing::REPLY);
                assert_eq!(events[0].1["text"], DESTRUCTIVE_CANCELLATION_REPLY);
                drop(events);
                let calls = router_calls.lock().unwrap();
                assert!(
                    calls.is_empty(),
                    "{utterance}: router never invoked when confirmation denied"
                );
            }
            // After either confirm path, pending is cleared.
            let pending = state.pending.lock().await;
            assert!(
                !pending.contains_key(&intent_id),
                "{utterance}: pending cleared on confirm"
            );
        }
        assert_eq!(
            minted_ids.len(),
            10,
            "all 10 scripts produced a destructive event"
        );
    }

    // ----- iter-13: AC6 pending-model consume tests -----
    //
    // Re-uses the existing `CapturingLlm` (above) so the model used per
    // turn is inspectable via `llm.captured.lock().unwrap()`.

    fn capturing_llm_ok() -> CapturingLlm {
        CapturingLlm {
            captured: Arc::new(StdMutex::new(Vec::new())),
            response: vec![
                StreamEvent::MessageStart,
                StreamEvent::TextDelta {
                    index: 0,
                    text: "ok".to_string(),
                },
                StreamEvent::MessageStop,
            ],
        }
    }

    #[tokio::test]
    async fn dispatch_turn_user_consumes_pending_model_in_memory() {
        let cfg = BrainConfig {
            pending_model: Some(crate::SHORT_MODEL_OPUS.to_string()),
            ..BrainConfig::default()
        };
        let llm = capturing_llm_ok();
        let captured = Arc::clone(&llm.captured);
        let state = Arc::new(DaemonState::new(cfg).with_llm(into_dyn_llm(llm)));
        let mut sink = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "hi".to_string(),
                confidence: 1.0,
                ts: 1,
            }),
            1,
        )
        .await
        .expect("dispatch ok");

        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs.len(), 1, "exactly one LLM request");
        assert_eq!(reqs[0].model, canonical_model(crate::SHORT_MODEL_OPUS));
        let cfg_after = state.config.lock().await.clone();
        assert!(
            cfg_after.pending_model.is_none(),
            "in-memory pending_model cleared after AC6 turn"
        );
    }

    #[tokio::test]
    async fn dispatch_turn_user_without_pending_does_not_mutate_cfg() {
        let cfg = BrainConfig::default();
        let llm = capturing_llm_ok();
        let captured = Arc::clone(&llm.captured);
        let state = Arc::new(DaemonState::new(cfg).with_llm(into_dyn_llm(llm)));
        let mut sink = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "hi".to_string(),
                confidence: 1.0,
                ts: 1,
            }),
            1,
        )
        .await
        .expect("dispatch ok");

        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs[0].model, canonical_model(crate::DEFAULT_MODEL_NAME));
        let cfg_after = state.config.lock().await.clone();
        assert!(cfg_after.pending_model.is_none());
        assert_eq!(cfg_after.default_model, BrainConfig::default().default_model);
    }

    #[tokio::test]
    async fn dispatch_turn_user_persists_pending_consume_to_config_path() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let cfg = BrainConfig {
            pending_model: Some(crate::SHORT_MODEL_OPUS.to_string()),
            ..BrainConfig::default()
        };
        cfg.save_to_file(&path).expect("seed save ok");

        let state = Arc::new(
            DaemonState::new(cfg)
                .with_llm(into_dyn_llm(capturing_llm_ok()))
                .with_config_path(path.clone()),
        );
        let mut sink = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "hi".to_string(),
                confidence: 1.0,
                ts: 1,
            }),
            1,
        )
        .await
        .expect("dispatch ok");

        let on_disk = BrainConfig::load_from_file(&path).expect("load post-consume");
        assert!(
            on_disk.pending_model.is_none(),
            "on-disk pending_model cleared after AC6 turn; got {:?}",
            on_disk.pending_model
        );
    }

    #[tokio::test]
    async fn dispatch_turn_user_pending_used_exactly_once() {
        // AC6 end-to-end: pending_model = opus -> first turn uses opus,
        // second turn reverts to the configured default.
        let cfg = BrainConfig {
            pending_model: Some(crate::SHORT_MODEL_OPUS.to_string()),
            ..BrainConfig::default()
        };
        let llm = capturing_llm_ok();
        let captured = Arc::clone(&llm.captured);
        let state = Arc::new(DaemonState::new(cfg).with_llm(into_dyn_llm(llm)));
        let mut sink = MemSink::default();

        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "one".to_string(),
                confidence: 1.0,
                ts: 1,
            }),
            1,
        )
        .await
        .expect("dispatch 1 ok");
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "two".to_string(),
                confidence: 1.0,
                ts: 2,
            }),
            2,
        )
        .await
        .expect("dispatch 2 ok");

        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs.len(), 2, "two turns dispatched");
        assert_eq!(reqs[0].model, canonical_model(crate::SHORT_MODEL_OPUS));
        assert_eq!(reqs[1].model, canonical_model(crate::DEFAULT_MODEL_NAME));
    }
}
