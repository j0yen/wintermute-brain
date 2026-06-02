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

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::almanac::{
    AckClass, PendingAck, classify_ack_response, ack_payload, snooze_payload,
    ALMANAC_ACK_TOPIC, ALMANAC_SNOOZE_TOPIC,
};
use crate::anthropic::{AnthropicClient, ClientError, Message, MessageRequest, Role, StreamEvent};
use crate::bus::{
    self, ConfirmDeniedEvent, ConfirmGrantedEvent, DecodeError, Emit, ErrorEvent, ReplyEvent,
    ReplyDestructiveEvent, Request, ToolCallEvent, ToolResultEvent, TurnUserEvent, decode_request,
    now_unix_ms, outgoing,
};
use crate::degrade::{
    HealthState, RateLimitState, HEALTH_SNAPSHOT_INTERVAL_MS, HEALTH_SNAPSHOT_TOPIC,
    component_for_error_topic, process_error_envelope, snapshot_payload, speak_payload,
    TTS_SPEAK_TOPIC,
};
use crate::history::{History, Turn};
use crate::recall_client::{self, QueryArgs, QueryHit, RecallClient, TouchArgs};
use crate::session::{AdvanceOutcome, CloseReason, SessionTracker};
use crate::repair::{self, Repair};
use crate::writeback::{ExtractorClient, WritebackGuard};
use crate::router::{RouteEvent, RouteTier, apply_routing_policy, PolicyInputs, canned_phrase};
use crate::{BrainConfig, PROFILE_SUBJECT, THREAD_SUBJECT_PREFIX, canonical_model};

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

/// Production [`ExtractorClient`] backed by an [`LlmClient`].
///
/// Issues a single non-streaming extraction call using the writeback model
/// configured in [`BrainConfig::writeback_model`]. Uses a small token budget
/// (512) since the extractor only needs to produce `FACT | …` lines.
pub struct AnthropicExtractor {
    llm: Arc<dyn LlmClient>,
    model: String,
}

impl AnthropicExtractor {
    /// Construct an extractor backed by `llm` and targeting `model`.
    #[must_use]
    pub fn new(llm: Arc<dyn LlmClient>, model: impl Into<String>) -> Self {
        Self { llm, model: model.into() }
    }
}

/// Token budget for the extraction call. Small: we only need `FACT | …` lines.
const EXTRACTION_MAX_TOKENS: u32 = 512;

#[async_trait::async_trait]
impl ExtractorClient for AnthropicExtractor {
    async fn extract(&self, transcript: &str) -> std::result::Result<String, String> {
        use crate::writeback::EXTRACTION_SYSTEM_PROMPT;
        let req = MessageRequest::streaming(
            canonical_model(&self.model),
            EXTRACTION_MAX_TOKENS,
            vec![Message { role: Role::User, content: transcript.to_string() }],
        )
        .with_system(EXTRACTION_SYSTEM_PROMPT.to_string());
        match self.llm.collect_messages(&req).await {
            Ok(events) => Ok(extract_assistant_text(&events)),
            Err(e) => Err(format!("{e}")),
        }
    }
}

/// Build a buffered streaming request for a single user turn. Pure
/// function; the caller is responsible for splicing recall context and
/// child-lock guidance into `persona` via [`compose_persona`].
///
/// `history_msgs` is the flat `[user, assistant, …]` prefix produced by
/// [`History::to_messages`] or [`History::trimmed_messages`]; callers pass
/// an empty slice when history is disabled (`history_turns = 0`).
/// The full message list is `[…history_msgs…, current_user]`, satisfying
/// `messages.len() == history_msgs.len() + 1`.
/// PRD-wmd-turn-history §2.2.
#[must_use]
pub fn compose_request(
    model: &str,
    persona: &str,
    history_msgs: &[Message],
    transcript: &str,
) -> MessageRequest {
    let mut messages = history_msgs.to_vec();
    messages.push(Message {
        role: Role::User,
        content: transcript.to_string(),
    });
    MessageRequest::streaming(canonical_model(model), DEFAULT_MAX_TOKENS, messages)
        .with_system(persona.to_string())
}

/// Assemble the effective system prompt the Anthropic call receives.
///
/// Layers in this order: `base` (persona), child-lock guard when set,
/// the destructive-intent gate (always — PRD §2.4), then a per-turn
/// recall-context block when non-empty, then the session recap context
/// (thread memories) when non-empty (PRD-wmd-session-recap §2.2).
/// Each layer is separated by a blank line so the model parses them as
/// distinct paragraphs.
#[must_use]
pub fn compose_persona(
    base: &str,
    child_lock: bool,
    recall_context: Option<&str>,
    recap_context: Option<&str>,
) -> String {
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
    if let Some(ctx) = recap_context {
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

/// Render a list of thread-memory recall hits into the recap context block
/// spliced onto the system prompt under a distinct "Recent conversations:"
/// label. Returns `None` when the slice is empty.
///
/// The section label distinguishes this block from the per-turn "What you
/// remember about the user" block so the model can tell standing profile
/// recall hits from session-continuity context (PRD-wmd-session-recap §2.2).
#[must_use]
pub fn format_recap_context(hits: &[QueryHit]) -> Option<String> {
    let non_empty: Vec<&str> = hits
        .iter()
        .map(|h| h.snippet.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if non_empty.is_empty() {
        return None;
    }
    let mut out = String::from("Recent conversations (most recent first):");
    for (i, snippet) in non_empty.iter().enumerate() {
        use std::fmt::Write as _;
        let _ = write!(out, "\n{}. {}", i + 1, snippet);
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

    /// Ad-hoc retrieval for the `wm.recall.search` tool call. Differs
    /// from [`Self::fetch`] in two ways: `subject` is caller-controlled
    /// (the tool exposes any subject scope, not just the per-turn
    /// profile lookup) and `limit` is caller-controlled (defaults to the
    /// daemon's own default when `None`).
    ///
    /// Default impl delegates to [`Self::fetch`], dropping the caller's
    /// subject / limit overrides — useful for test fakes that want to
    /// reuse a single canned hit set.
    ///
    /// # Errors
    /// Implementations surface transport / decode failures as
    /// [`RecallSourceError`].
    async fn search(
        &self,
        text: &str,
        _subject: Option<&str>,
        _limit: Option<usize>,
    ) -> Result<Vec<QueryHit>, RecallSourceError> {
        self.fetch(text).await
    }

    /// Persist a single fact body under `subject`. Used by the
    /// `wm.recall.save_fact` tool to write profile facts (PRD §2.3,
    /// AC7). Default impl returns
    /// [`RecallSourceError::Unsupported`] so test fakes that don't care
    /// about the write path can opt out by inheriting the default.
    ///
    /// Returns the new memory id on success.
    ///
    /// # Errors
    /// [`RecallSourceError::Unsupported`] from the default impl;
    /// production implementations surface
    /// [`RecallSourceError::Client`] or [`RecallSourceError::Write`].
    async fn save_fact(
        &self,
        _subject: &str,
        _body: &str,
    ) -> Result<String, RecallSourceError> {
        Err(RecallSourceError::Unsupported("save_fact"))
    }
}

/// Errors surfaced by [`RecallSource::fetch`] and friends.
#[derive(Debug, thiserror::Error)]
pub enum RecallSourceError {
    /// Underlying recall-daemon client failure.
    #[error("recall client: {0}")]
    Client(#[from] recall_client::ClientError),
    /// The recall source does not implement the requested operation
    /// (typically `save_fact` on a test fake or a read-only source).
    #[error("recall source does not support {0}")]
    Unsupported(&'static str),
    /// Subprocess-backed write (e.g. `recall write`) failed. Carries the
    /// exit status as a non-zero code or `-1` if the process never
    /// returned a status, plus a captured stderr fragment for tracing.
    #[error("recall write subprocess failed (code={code}): {message}")]
    Write {
        /// Exit code reported by the subprocess (or `-1` if unavailable).
        code: i32,
        /// Truncated stderr fragment for diagnostics.
        message: String,
    },
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

/// Default kind passed to `recall write` by [`LiveRecallSource::save_fact`].
///
/// Maps the brain's "fact" tool-call vocabulary onto the closest recall
/// built-in kind (PRD §2.3 + `recall write --kind semantic` accepts the
/// four canonical kinds: episodic / semantic / procedural / reflective).
pub const DEFAULT_SAVE_KIND: &str = "semantic";

/// Default subprocess binary name for `recall write`. The OS PATH
/// resolves this to whatever user-installed `recall` is first on PATH;
/// tests can override via [`LiveRecallSource::with_recall_bin`].
pub const DEFAULT_RECALL_BIN: &str = "recall";

/// Truncation cap for captured stderr inside
/// [`RecallSourceError::Write`]. Keeps event payloads bounded.
const STDERR_TRUNCATE_BYTES: usize = 4096;

/// Production [`RecallSource`]: opens a fresh recall-daemon socket
/// connection per turn. Fleet 1 trades the per-call connect for
/// simplicity; pooling and connection reuse are Fleet 2 work.
#[derive(Debug, Clone)]
pub struct LiveRecallSource {
    socket: PathBuf,
    profile_subject: String,
    limit: usize,
    /// Optional recall data root passed as `--root <path>` to the
    /// `recall write` subprocess. `None` lets the binary fall back to
    /// `$RECALL_HOME` or `~/.claude/recall` (recall's own defaults).
    data_root: Option<PathBuf>,
    /// Binary invoked for `save_fact` subprocess writes. Defaults to
    /// [`DEFAULT_RECALL_BIN`].
    recall_bin: PathBuf,
    /// Memory kind passed to `recall write --kind`. Defaults to
    /// [`DEFAULT_SAVE_KIND`].
    save_kind: String,
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
            data_root: None,
            recall_bin: PathBuf::from(DEFAULT_RECALL_BIN),
            save_kind: DEFAULT_SAVE_KIND.to_string(),
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

    /// Pin the recall data root that `recall write` should target via
    /// `--root <path>`. Set this when `$RECALL_HOME` may not be
    /// inherited by the brain process.
    #[must_use]
    pub fn with_data_root(mut self, root: PathBuf) -> Self {
        self.data_root = Some(root);
        self
    }

    /// Override the `recall write` binary path. Mainly for tests that
    /// point at a stub script.
    #[must_use]
    pub fn with_recall_bin(mut self, bin: PathBuf) -> Self {
        self.recall_bin = bin;
        self
    }

    /// Override the memory kind passed to `recall write --kind`.
    #[must_use]
    pub fn with_save_kind(mut self, kind: impl Into<String>) -> Self {
        self.save_kind = kind.into();
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

    async fn search(
        &self,
        text: &str,
        subject: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<QueryHit>, RecallSourceError> {
        let mut client = RecallClient::connect(&self.socket).await?;
        let args = QueryArgs {
            text: text.to_string(),
            limit,
            hybrid: None,
            project_subject: subject.map(str::to_string),
        };
        let resp = client.query(&args).await?;
        Ok(resp.ranked_hits)
    }

    async fn save_fact(
        &self,
        subject: &str,
        body: &str,
    ) -> Result<String, RecallSourceError> {
        let mut cmd = tokio::process::Command::new(&self.recall_bin);
        if let Some(root) = &self.data_root {
            cmd.arg("--root").arg(root);
        }
        cmd.arg("write")
            .arg("--kind")
            .arg(&self.save_kind)
            .arg("--subject")
            .arg(subject)
            .arg("--body")
            .arg(body)
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true);
        let output = cmd.output().await.map_err(|e| RecallSourceError::Write {
            code: -1,
            message: format!("spawn {}: {e}", self.recall_bin.display()),
        })?;
        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            let mut stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            if stderr.len() > STDERR_TRUNCATE_BYTES {
                stderr.truncate(STDERR_TRUNCATE_BYTES);
                stderr.push('…');
            }
            return Err(RecallSourceError::Write { code, message: stderr });
        }
        let id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if id.is_empty() {
            return Err(RecallSourceError::Write {
                code: 0,
                message: "recall write succeeded but emitted no id".to_string(),
            });
        }
        Ok(id)
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

/// Tool names handled by [`RecallToolsRouter`] (PRD §2.3 — Fleet 1
/// recall surface). Anything else falls through to the configured
/// fallback router.
pub const TOOL_RECALL_SEARCH: &str = "wm.recall.search";
/// See [`TOOL_RECALL_SEARCH`].
pub const TOOL_RECALL_SAVE_FACT: &str = "wm.recall.save_fact";

/// Default almanac acknowledgment patience window in milliseconds.
///
/// Used when earshot's dialog-timing config has not been supplied at
/// runtime.  The live daemon should override this via
/// [`DaemonState::with_almanac_patience_ms`] once it loads earshot's
/// config (PRD AC5).  30 seconds is a safe fallback for test fixtures
/// that don't wire earshot config.
pub const DEFAULT_ALMANAC_PATIENCE_MS: u64 = 30_000;

/// Graceful degrade phrase emitted when a repair request arrives but the
/// history buffer is empty (AC3). The user hears this instead of silence or
/// an empty reply.
pub const REPAIR_EMPTY_HISTORY_REPLY: &str = "I haven't said anything yet.";

/// Build the effective phrase sets for repair classification from the daemon
/// config. When a list is empty the built-in defaults are used instead.
///
/// This is called once per `TurnUser` event, before the LLM dispatch check,
/// so the overhead is two `BTreeSet` constructions at most (cheap).
#[must_use]
fn repair_phrase_sets(cfg: &BrainConfig) -> (BTreeSet<String>, BTreeSet<String>) {
    let repeat = if cfg.repair_repeat_phrases.is_empty() {
        repair::default_repeat_phrases()
    } else {
        repair::build_phrase_set(&cfg.repair_repeat_phrases)
    };
    let louder = if cfg.repair_louder_phrases.is_empty() {
        repair::default_louder_phrases()
    } else {
        repair::build_phrase_set(&cfg.repair_louder_phrases)
    };
    (repeat, louder)
}

/// Fleet 1 recall tool router.
///
/// Handles `wm.recall.search` (delegates to [`RecallSource::search`]) and
/// `wm.recall.save_fact` (delegates to [`RecallSource::save_fact`]);
/// every other tool name forwards to `fallback` (defaults to
/// [`NoToolsRouter`]). Closes PRD AC7 by routing model-driven recall
/// reads + writes through the same abstraction the per-turn retrieval
/// already uses.
pub struct RecallToolsRouter {
    recall: Arc<dyn RecallSource>,
    fallback: Arc<dyn ToolRouter>,
    default_save_subject: String,
}

impl RecallToolsRouter {
    /// Build a router that fronts `recall`. Defaults the
    /// save-fact subject to [`PROFILE_SUBJECT`] and the fallback to
    /// [`NoToolsRouter`].
    #[must_use]
    pub fn new(recall: Arc<dyn RecallSource>) -> Self {
        Self {
            recall,
            fallback: Arc::new(NoToolsRouter),
            default_save_subject: PROFILE_SUBJECT.to_string(),
        }
    }

    /// Swap the fallback router used for non-recall tool names.
    #[must_use]
    pub fn with_fallback(mut self, fallback: Arc<dyn ToolRouter>) -> Self {
        self.fallback = fallback;
        self
    }

    /// Override the default subject for [`TOOL_RECALL_SAVE_FACT`] when
    /// the caller omits `subject` from the tool args.
    #[must_use]
    pub fn with_default_save_subject(mut self, subject: impl Into<String>) -> Self {
        self.default_save_subject = subject.into();
        self
    }
}

#[async_trait::async_trait]
impl ToolRouter for RecallToolsRouter {
    async fn dispatch(&self, name: &str, args: &Value) -> ToolResultBody {
        match name {
            TOOL_RECALL_SEARCH => dispatch_recall_search(self.recall.as_ref(), args).await,
            TOOL_RECALL_SAVE_FACT => {
                dispatch_recall_save_fact(self.recall.as_ref(), &self.default_save_subject, args)
                    .await
            }
            _ => self.fallback.dispatch(name, args).await,
        }
    }
}

/// Render one [`QueryHit`] into the wire body the brain returns for
/// `wm.recall.search`. Trimmed to the fields a downstream tool consumer
/// is likely to want; recall internals (`bm25`, `vector_sim`, `recall_count`)
/// stay out of the tool surface.
fn render_hit(hit: &QueryHit) -> Value {
    json!({
        "id": hit.id,
        "kind": hit.kind,
        "subject": hit.subject,
        "snippet": hit.snippet,
        "score": hit.score,
        "confidence": hit.confidence,
    })
}

fn bad_args(tool: &str, reason: &str) -> ToolResultBody {
    ToolResultBody {
        ok: false,
        body: json!({
            "error": "bad_args",
            "tool": tool,
            "message": reason,
        }),
    }
}

fn recall_error(tool: &str, err: &RecallSourceError) -> ToolResultBody {
    ToolResultBody {
        ok: false,
        body: json!({
            "error": "recall_error",
            "tool": tool,
            "message": err.to_string(),
        }),
    }
}

async fn dispatch_recall_search(recall: &dyn RecallSource, args: &Value) -> ToolResultBody {
    let Some(obj) = args.as_object() else {
        return bad_args(TOOL_RECALL_SEARCH, "args must be a JSON object");
    };
    let Some(text) = obj.get("text").and_then(Value::as_str) else {
        return bad_args(TOOL_RECALL_SEARCH, "args.text (string) is required");
    };
    if text.trim().is_empty() {
        return bad_args(TOOL_RECALL_SEARCH, "args.text must not be blank");
    }
    let subject = obj.get("subject").and_then(Value::as_str);
    let limit = match obj.get("limit") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) => {
            let Some(u) = n.as_u64() else {
                return bad_args(TOOL_RECALL_SEARCH, "args.limit must be a non-negative integer");
            };
            let Ok(small) = usize::try_from(u) else {
                return bad_args(TOOL_RECALL_SEARCH, "args.limit exceeds usize::MAX");
            };
            Some(small)
        }
        Some(_) => {
            return bad_args(TOOL_RECALL_SEARCH, "args.limit must be a number");
        }
    };
    match recall.search(text, subject, limit).await {
        Ok(hits) => {
            let body = json!({
                "hits": hits.iter().map(render_hit).collect::<Vec<_>>(),
                "count": hits.len(),
            });
            ToolResultBody { ok: true, body }
        }
        Err(err) => recall_error(TOOL_RECALL_SEARCH, &err),
    }
}

async fn dispatch_recall_save_fact(
    recall: &dyn RecallSource,
    default_subject: &str,
    args: &Value,
) -> ToolResultBody {
    let Some(obj) = args.as_object() else {
        return bad_args(TOOL_RECALL_SAVE_FACT, "args must be a JSON object");
    };
    let Some(body_text) = obj.get("body").and_then(Value::as_str) else {
        return bad_args(TOOL_RECALL_SAVE_FACT, "args.body (string) is required");
    };
    if body_text.trim().is_empty() {
        return bad_args(TOOL_RECALL_SAVE_FACT, "args.body must not be blank");
    }
    let subject_owned;
    let subject = match obj.get("subject") {
        None | Some(Value::Null) => default_subject,
        Some(Value::String(s)) if !s.trim().is_empty() => {
            subject_owned = s.clone();
            subject_owned.as_str()
        }
        Some(_) => {
            return bad_args(TOOL_RECALL_SAVE_FACT, "args.subject must be a non-empty string");
        }
    };
    match recall.save_fact(subject, body_text).await {
        Ok(id) => ToolResultBody {
            ok: true,
            body: json!({
                "id": id,
                "subject": subject,
            }),
        },
        Err(err) => recall_error(TOOL_RECALL_SAVE_FACT, &err),
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
///
/// The client is wrapped in an `Arc<tokio::sync::Mutex<_>>` so a
/// background heartbeat task (spawned in [`run`]) can periodically
/// refresh the daemon's `last_heartbeat_unix_secs` without contending
/// destructively with publish call sites. Publish is the hot path; the
/// lock is held only for the duration of one request+reply round-trip
/// (microseconds), so contention is negligible.
pub struct AgoraSink {
    pub(crate) inner: Arc<Mutex<agorabus::Client>>,
}

#[async_trait::async_trait]
impl EventSink for AgoraSink {
    async fn publish(&mut self, topic: &str, data: Value) -> Result<()> {
        let reply = {
            let mut client = self.inner.lock().await;
            client.publish(topic, data).await?
        };
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

/// Minimal in-daemon session tracker for writeback.
///
/// Tracks the current session so [`writeback`] can fire when the idle gap
/// expires or an explicit close phrase is detected.  The session-boundary
/// PRD will expand this; for now we carry just enough state for writeback
/// (PRD-wmd-memory-writeback §2.2 / AC7).
#[derive(Debug, Clone)]
pub struct WritebackSession {
    /// Stable id minted when the session opened.
    pub session_id: String,
    /// Unix milliseconds of the last turn in this session.
    pub last_turn_ms: u64,
    /// Copy of all turns so far (mirrored from history at write-back time).
    pub turns: Vec<Turn>,
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
    /// The tier-ladder orchestrator. When `Some`, [`Request::TurnUser`]
    /// dispatches through the local-first ladder instead of the single
    /// Anthropic client. PRD-brain-backend-ladder.
    pub ladder: Option<Arc<crate::ladder::LadderClient>>,
    /// Per-session tier floor for conversational stickiness (AC10).
    pub session_floor: crate::ladder::SessionFloor,
    /// Bounded rolling turn history (PRD-wmd-turn-history §2.1).
    ///
    /// Capacity is set from [`BrainConfig::history_turns`] at construction
    /// time. Successful turns push a [`Turn`] here; failures and empty
    /// replies do not (AC3 / PRD §2.1 — only real turns chain).
    /// Cleared on every session boundary (PRD-wmd-session-boundary §2.4).
    pub history: Mutex<History>,
    /// Conversation session boundary tracker (PRD-wmd-session-boundary §2.1).
    ///
    /// Derives session boundaries from idle gaps and explicit close phrases;
    /// emits `wm.brain.session.{start,end}` on transitions. Wrapped in a
    /// `Mutex` so `dispatch` and the graceful-shutdown hook share access.
    pub session_tracker: Mutex<SessionTracker>,
    /// In-flight almanac acknowledgment awaiting the user's reply.
    ///
    /// Set by `handle_almanac_due` (speak-bridge path) when a prompt is
    /// voiced; cleared by the STT handler or the patience-window timeout.
    /// `None` means no prompt is currently awaiting acknowledgment (the
    /// normal idle state).
    pub pending_ack: Mutex<Option<PendingAck>>,
    /// Earshot patience-window duration (milliseconds).
    ///
    /// Sourced from earshot's dialog-timing config at startup; the
    /// almanac ack handler uses this value — it never hard-codes a
    /// literal deadline (PRD AC5).
    pub almanac_patience_ms: u64,
    /// Per-kind rate-limit state for the graceful-degradation aggregator
    /// (PRD-wintermute-companion-degrade §2.3).
    pub degrade_rate: Arc<RateLimitState>,
    /// Per-component health state, updated by the degradation aggregator
    /// and snapshotted every 60 s (PRD-wintermute-companion-degrade §2.4).
    pub degrade_health: Arc<HealthState>,
    /// Current writeback session (idle-gap + turn tracking).
    ///
    /// `None` until the first `turn.user` arrives.  Updated on every
    /// turn and closed (triggering writeback) when the idle gap expires.
    /// PRD-wmd-memory-writeback §2.2.
    pub writeback_session: Mutex<Option<WritebackSession>>,
    /// Idempotence guard: ensures each session writes back at most once.
    pub writeback_guard: WritebackGuard,
    /// Optional extraction client. When `Some`, end-of-session writeback
    /// fires; when `None` writeback is disabled (no LLM configured).
    pub extractor: Option<Arc<dyn ExtractorClient>>,
    /// Session-scoped thread-memory recap context.
    ///
    /// Fetched once at `wm.brain.session.start` from the recall store
    /// (thread subject prefix); held for the life of the session and
    /// spliced into every turn's system prompt under "Recent conversations:".
    /// Cleared on each new session open so context never bleeds.
    /// `None` means no recap context was found (cold store) or recap is
    /// disabled. PRD-wmd-session-recap §2.1 / §2.2.
    pub recap_context: Mutex<Option<String>>,
    /// Monotonic count of thread queries fired (AC3: exactly one per session).
    pub session_thread_query_count: std::sync::atomic::AtomicU64,
}

impl DaemonState {
    /// Construct a daemon state from an already-validated config. The
    /// resulting state has no LLM client and uses [`NullRecall`] +
    /// [`NoToolsRouter`]; attach real implementations via the
    /// `with_*` builders.
    #[must_use]
    pub fn new(config: BrainConfig) -> Self {
        let history_turns = config.history_turns;
        let now = crate::bus::now_unix_ms();
        let session_tracker = SessionTracker::new(
            config.idle_gap_ms,
            config.session_end_phrases.clone(),
        );
        // Compose the persona base once at config-load time so the
        // string is byte-stable across turns (prompt-cache discipline,
        // PRD-hearth-persona-config §2.3).
        let persona_base = config.persona.compose_base(config.user_name.as_deref());
        Self {
            config: Mutex::new(config),
            config_path: None,
            llm: None,
            recall: Arc::new(NullRecall),
            tool_router: Arc::new(NoToolsRouter),
            persona: persona_base,
            pending: Mutex::new(HashMap::new()),
            intent_counter: AtomicU64::new(0),
            ladder: None,
            session_floor: crate::ladder::SessionFloor::new(),
            history: Mutex::new(History::new(history_turns)),
            session_tracker: Mutex::new(session_tracker),
            pending_ack: Mutex::new(None),
            almanac_patience_ms: DEFAULT_ALMANAC_PATIENCE_MS,
            degrade_rate: Arc::new(RateLimitState::new()),
            degrade_health: Arc::new(HealthState::new(now)),
            writeback_session: Mutex::new(None),
            writeback_guard: WritebackGuard::new(),
            extractor: None,
            recap_context: Mutex::new(None),
            session_thread_query_count: AtomicU64::new(0),
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

    /// Attach the tier-ladder orchestrator. When set, `TurnUser` dispatch
    /// goes through the ladder rather than the single Anthropic client.
    #[must_use]
    pub fn with_ladder(mut self, ladder: Arc<crate::ladder::LadderClient>) -> Self {
        self.ladder = Some(ladder);
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

    /// Attach an extraction client for end-of-session writeback.
    ///
    /// When not set, writeback is silently skipped.
    #[must_use]
    pub fn with_extractor(mut self, extractor: Arc<dyn ExtractorClient>) -> Self {
        self.extractor = Some(extractor);
        self
    }

    /// Override the almanac patience-window duration.
    ///
    /// The live daemon calls this with the value loaded from earshot's
    /// dialog-timing config so the acknowledgment FSM never hard-codes
    /// a deadline (PRD AC5).
    #[must_use]
    pub const fn with_almanac_patience_ms(mut self, ms: u64) -> Self {
        self.almanac_patience_ms = ms;
        self
    }

    /// Set a [`PendingAck`] on this state (called by the almanac-due handler
    /// after speaking a prompt).
    ///
    /// Replaces any previously open pending ack — in normal operation
    /// there is at most one open ack per session, but if a second
    /// due-prompt fires before the first is resolved, the new one takes
    /// precedence.
    pub async fn set_pending_ack(&self, ack: PendingAck) {
        *self.pending_ack.lock().await = Some(ack);
    }

    /// Clear the pending ack (called after resolution — done, missed, or
    /// a final missed with no further re-ask).
    pub async fn clear_pending_ack(&self) {
        *self.pending_ack.lock().await = None;
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
            // --- Session boundary (PRD-wmd-session-boundary §2.1 / §2.4) ---
            //
            // Advance the session tracker before any other work. On a new
            // session: emit SESSION_END for the closed session (if any), emit
            // SESSION_START for the fresh session, and clear the history ring
            // so context never bleeds across sessions. On an extended session:
            // no events, no history reset.
            let explicit_close = {
                let tracker = state.session_tracker.lock().await;
                tracker.is_explicit_close(&t.transcript)
            };

            let outcome = {
                let mut tracker = state.session_tracker.lock().await;
                tracker.advance(now_ms)
            };

            match outcome {
                AdvanceOutcome::NewSession { closed, opened } => {
                    if let Some(end_payload) = closed {
                        info!(
                            session_id = %end_payload.session_id,
                            reason = ?end_payload.reason,
                            turn_count = end_payload.turn_count,
                            "wm-brain: session closed"
                        );
                        if let Ok(v) = serde_json::to_value(&end_payload) {
                            if let Err(err) = publish
                                .publish(outgoing::SESSION_END, v)
                                .await
                            {
                                warn!(err = %err, "wm-brain: failed to publish session.end");
                            }
                        }
                    }
                    info!(
                        session_id = %opened.session_id,
                        "wm-brain: session started"
                    );
                    if let Ok(v) = serde_json::to_value(&opened) {
                        if let Err(err) = publish
                            .publish(outgoing::SESSION_START, v)
                            .await
                        {
                            warn!(err = %err, "wm-brain: failed to publish session.start");
                        }
                    }
                    // Clear history on session boundary (PRD §2.4).
                    let max_turns = {
                        let cfg = state.config.lock().await;
                        cfg.history_turns
                    };
                    let mut history = state.history.lock().await;
                    *history = History::new(max_turns);
                    drop(history);
                    // Session-recap: fetch thread memories for the new session.
                    // AC3: query fires exactly once per new session here; per-turn
                    // recall queries are separate and still fire every turn.
                    // AC8: failures log WARN and leave recap_context None.
                    handle_session_start(state, publish, now_ms).await;
                }
                AdvanceOutcome::Extended => {
                    // Nothing to emit; history continues.
                }
            }

            // AC6: `wmd --model opus` for the next turn only. Read
            // effective_model + child_lock and, if pending_model was
            // set, clear it now so the *next* turn uses the default.
            // Persistence happens after the turn handler returns so a
            // crash during dispatch doesn't strand an empty pending on
            // disk.
            let (model, tier, child_lock, consumed_pending) = {
                let mut cfg = state.config.lock().await;
                let model = cfg.effective_model().to_string();
                let tier = cfg.effective_tier();
                let had_pending = cfg.pending_model.is_some() || cfg.pending_tier.is_some();
                if cfg.pending_model.is_some() {
                    cfg.consume_pending();
                }
                if cfg.pending_tier.is_some() {
                    cfg.consume_pending_tier();
                }
                (model, tier, cfg.child_lock, had_pending)
            };
            handle_turn_user(state, publish, &model, &tier, child_lock, &t, now_ms).await?;
            if consumed_pending {
                persist_after_pending_consume(state).await;
            }

            // --- Explicit close (PRD §2.2): close after the reply, not before.
            // The model answered the goodbye turn; now close the session.
            if explicit_close {
                let closed = {
                    let mut tracker = state.session_tracker.lock().await;
                    tracker.close(now_ms, CloseReason::Explicit)
                };
                if let Some(end_payload) = closed {
                    info!(
                        session_id = %end_payload.session_id,
                        reason = ?end_payload.reason,
                        turn_count = end_payload.turn_count,
                        "wm-brain: session closed (explicit phrase)"
                    );
                    if let Ok(v) = serde_json::to_value(&end_payload) {
                        if let Err(err) = publish
                            .publish(outgoing::SESSION_END, v)
                            .await
                        {
                            warn!(err = %err, "wm-brain: failed to publish session.end (explicit)");
                        }
                    }
                }
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

/// Handle a `wm.stt.final` event when an almanac acknowledgment window is open.
///
/// Called by the subscribe loop when the topic is `wm.stt.final` and the
/// daemon has a `PendingAck` set.  The function:
///
/// 1. Checks whether the patience window is still open (`now_ms ≤ asked_ms +
///    patience_ms`).  If the window has already elapsed the timeout path
///    handles it — do nothing here (the timeout fires on its own tick).
/// 2. Classifies the transcript via [`classify_ack_response`].
/// 3. Publishes `wm.almanac.ack` + (for snooze) `wm.almanac.snooze`, updates
///    `pending_ack` accordingly.
/// 4. For **Unrelated** transcripts, leaves `pending_ack` open and does not
///    publish anything (AC3).
///
/// Returns `Ok(true)` when the event was consumed by the ack path (so the
/// caller can skip the regular dialog dispatch), `Ok(false)` when no
/// `PendingAck` is open or the transcript is unrelated (AC7: the caller must
/// then proceed with normal dispatch).
///
/// # Errors
/// Propagates the first publish failure.
pub async fn handle_stt_final_for_ack(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    transcript: &str,
    now_ms: u64,
) -> Result<bool> {
    let patience_ms = state.almanac_patience_ms;

    // Take a snapshot under the lock; if nothing is pending, exit fast (AC7).
    let pending_snapshot = {
        let guard = state.pending_ack.lock().await;
        guard.clone()
    };
    let Some(pending) = pending_snapshot else {
        return Ok(false);
    };

    // Window already elapsed — the timeout path will handle the missed emit.
    // Do not double-emit; leave the pending slot for the caller to clean up.
    if !pending.within_window(now_ms, patience_ms) {
        return Ok(false);
    }

    let class = classify_ack_response(transcript);
    match class {
        AckClass::Done => {
            // Clear the pending ack before publishing so a re-entrant STT
            // event after a slow publish cannot double-emit.
            state.clear_pending_ack().await;
            let payload = ack_payload(&pending.id, "done", now_ms);
            publish.publish(ALMANAC_ACK_TOPIC, payload).await
                .context("publish wm.almanac.ack done")?;
            info!(id = %pending.id, "wm-brain almanac: ack done");
            Ok(true)
        }
        AckClass::Snooze => {
            if pending.snoozes_used >= pending.config.max_snoozes {
                // Exhausted — treat as missed (AC2 second branch).
                state.clear_pending_ack().await;
                let payload = ack_payload(&pending.id, "missed", now_ms);
                publish.publish(ALMANAC_ACK_TOPIC, payload).await
                    .context("publish wm.almanac.ack missed (snooze exhausted)")?;
                info!(id = %pending.id, snoozes_used = pending.snoozes_used, "wm-brain almanac: ack missed (snooze exhausted)");
            } else {
                // Grant the snooze — increment counter and update pending.
                let mut updated = pending.clone();
                updated.snoozes_used += 1;
                updated.asked_ms = now_ms; // reset window from now
                updated.re_asked = false;
                state.set_pending_ack(updated.clone()).await;

                let resume_ts = now_ms.saturating_add(pending.config.snooze_ms);
                let ack_v = ack_payload(&pending.id, "snoozed", now_ms);
                publish.publish(ALMANAC_ACK_TOPIC, ack_v).await
                    .context("publish wm.almanac.ack snoozed")?;
                let snooze_v = snooze_payload(&pending.id, resume_ts, now_ms);
                publish.publish(ALMANAC_SNOOZE_TOPIC, snooze_v).await
                    .context("publish wm.almanac.snooze")?;
                info!(id = %pending.id, snoozes_used = updated.snoozes_used, resume_ts, "wm-brain almanac: snoozed");
            }
            Ok(true)
        }
        AckClass::Unrelated => {
            // AC3: leave pending open, do not emit.
            debug!(id = %pending.id, transcript = %transcript, "wm-brain almanac: unrelated transcript; pending left open");
            Ok(false)
        }
    }
}

/// Tick the almanac patience-window timeout for the current `pending_ack`.
///
/// Called periodically (e.g. on each `wm.stt.final` whose window has elapsed,
/// or from a dedicated timer task).  If the window has elapsed:
///
/// - First elapse: emits `{state:"missed"}`, speaks one gentle re-ask via the
///   speak-bridge reply path, sets `re_asked = true` and resets `asked_ms` to
///   `now_ms` so the window reopens for the re-ask.
/// - Second elapse: emits `{state:"missed"}` with no further re-ask and clears
///   `pending_ack`.
///
/// Returns `Ok(true)` when the timeout fired (caller may want to log/trace),
/// `Ok(false)` when the window is still open or there is no pending ack.
///
/// # Errors
/// Propagates the first publish failure.
pub async fn tick_almanac_timeout(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    now_ms: u64,
) -> Result<bool> {
    let patience_ms = state.almanac_patience_ms;
    let pending_snapshot = {
        let guard = state.pending_ack.lock().await;
        guard.clone()
    };
    let Some(pending) = pending_snapshot else {
        return Ok(false);
    };
    if pending.within_window(now_ms, patience_ms) {
        return Ok(false);
    }
    // Window has elapsed.
    if pending.re_asked {
        // Second elapse — finalise missed with no further re-ask (AC4).
        state.clear_pending_ack().await;
        let payload = ack_payload(&pending.id, "missed", now_ms);
        publish.publish(ALMANAC_ACK_TOPIC, payload).await
            .context("publish wm.almanac.ack missed (final)")?;
        info!(id = %pending.id, "wm-brain almanac: timeout final missed (re_asked already)");
    } else {
        // First elapse — speak the gentle re-ask and reset window (AC4).
        let re_ask_text = "Did you take it? Just say yes or later if you need more time.";
        let reply = bus::ReplyEvent { text: re_ask_text.to_string(), ts: now_ms, loudness: None };
        publish
            .publish(bus::outgoing::REPLY, serde_json::to_value(&reply)?)
            .await
            .context("publish re-ask reply")?;
        let payload = ack_payload(&pending.id, "missed", now_ms);
        publish.publish(ALMANAC_ACK_TOPIC, payload).await
            .context("publish wm.almanac.ack missed (first elapse)")?;
        // Update pending: set re_asked, reset window start.
        let mut updated = pending.clone();
        updated.re_asked = true;
        updated.asked_ms = now_ms;
        state.set_pending_ack(updated).await;
        info!(id = %pending.id, "wm-brain almanac: timeout first elapse; re-ask spoken, window reset");
    }
    Ok(true)
}

/// Set a [`PendingAck`] from an almanac-due event (called by the almanac-due
/// handler after a prompt has been spoken via the speak-bridge path).
///
/// This is a thin wrapper around [`DaemonState::set_pending_ack`] that
/// constructs the [`PendingAck`] from the envelope fields and logs the
/// transition.
pub async fn handle_almanac_due(
    state: &DaemonState,
    id: impl Into<String>,
    category: impl Into<String>,
    config: crate::almanac::AlmanacEntryConfig,
    now_ms: u64,
) {
    let id = id.into();
    let category = category.into();
    let ack = PendingAck::new(id.clone(), category.clone(), now_ms, config);
    state.set_pending_ack(ack).await;
    info!(
        id = %id,
        category = %category,
        asked_ms = now_ms,
        patience_ms = state.almanac_patience_ms,
        "wm-brain almanac: pending ack set"
    );
}

/// Speak an almanac due-entry and arm the acknowledgment window.
///
/// This is the speak-bridge handler for `wm.almanac.due` envelopes
/// (PRD-almanac-speak-bridge).  It:
///
/// 1. Checks the `almanac_speak` config gate; returns `Ok(false)` immediately
///    when disabled (AC3 — publishes nothing).
/// 2. Validates `ev.say` is non-empty; logs WARN and returns `Ok(false)` when
///    blank (AC4 — malformed envelope degrades without panic or reply).
/// 3. Builds `ReplyEvent { text: ev.say, ts: now_ms }` and publishes to
///    `outgoing::REPLY` — the identical call `handle_session_start` (recap
///    path) makes.  Persona wrapping and TTS pacing are the existing
///    reply-path's responsibility (AC6 — no persona string added here).
/// 4. Arms the acknowledgment window by calling [`handle_almanac_due`] with
///    the entry-level config from the envelope (AC2-AC5 — patience window,
///    snooze handling).
///
/// Returns `Ok(true)` when the reply was published and the ack window armed,
/// `Ok(false)` when the gate was off or the envelope was malformed.
///
/// # Errors
/// Propagates the first publish failure (bus transport errors).
pub async fn handle_speak_almanac_due(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    ev: &crate::almanac::AlmanacDueEvent,
    now_ms: u64,
) -> Result<bool> {
    // AC3: gate — disabled speak publishes nothing.
    let almanac_speak = { state.config.lock().await.almanac_speak };
    if !almanac_speak {
        debug!(
            id = %ev.id,
            "wm-brain almanac: speak gate off; dropping due event"
        );
        return Ok(false);
    }

    // AC4: malformed envelope (missing / empty say) → WARN, no panic, no reply.
    let say = ev.say.trim();
    if say.is_empty() {
        warn!(
            id = %ev.id,
            label = %ev.label,
            category = %ev.category,
            "wm-brain almanac: wm.almanac.due envelope has empty say field; dropping"
        );
        return Ok(false);
    }

    // AC1: speak verbatim via the same publish path recap_opener uses.
    let reply = bus::ReplyEvent {
        text: say.to_string(),
        ts: now_ms,
        loudness: None,
    };
    publish
        .publish(
            bus::outgoing::REPLY,
            serde_json::to_value(&reply).context("serialise almanac reply")?,
        )
        .await
        .context("publish almanac reply")?;

    info!(
        id = %ev.id,
        label = %ev.label,
        category = %ev.category,
        "wm-brain almanac: due-entry spoken via reply path"
    );

    // Arm the ack window with a default AlmanacEntryConfig (v0.1 doesn't
    // carry per-entry config on the envelope; use the built-in defaults).
    let ack_config = crate::almanac::AlmanacEntryConfig::default();
    handle_almanac_due(state, &ev.id, &ev.category, ack_config, now_ms).await;

    Ok(true)
}

/// Fetch the most recent thread memories from recall and store them as the
/// session's recap context.
///
/// Called once at `wm.brain.session.start` (PRD-wmd-session-recap §2.1 / AC3).
/// Queries both today's thread subject and the most-recent prior day's thread
/// by using the subject prefix `THREAD_SUBJECT_PREFIX`.  Respects
/// `recap_max_memories`; bounds the hit slice to the most recent N.
///
/// Recall outages are tolerated: a query failure logs WARN and leaves the recap
/// context empty — the session proceeds with no recap (AC8).
///
/// If `recap_opener` is `true` and at least one memory was found, publishes a
/// proactive `wm.brain.reply` before the user's first turn (AC7). The opener is
/// the first snippet, prefixed with "Earlier you mentioned:" — plain and
/// conservative (PRD §2.3 / non-goal 3).
///
/// Returns `true` when an opener was published.
pub async fn handle_session_start(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    now_ms: u64,
) -> bool {
    let (recap_max, recap_opener) = {
        let cfg = state.config.lock().await;
        (cfg.recap_max_memories, cfg.recap_opener)
    };

    // AC3: count the query.
    state.session_thread_query_count.fetch_add(1, Ordering::SeqCst);

    if recap_max == 0 {
        *state.recap_context.lock().await = None;
        return false;
    }

    // Query by thread subject prefix to retrieve committed thread memories.
    let hits = match state
        .recall
        .search(
            // Use the prefix as the free-text query so the daemon retrieves
            // recent thread memories scoped to the thread subject namespace.
            THREAD_SUBJECT_PREFIX,
            Some(THREAD_SUBJECT_PREFIX),
            Some(recap_max),
        )
        .await
    {
        Ok(h) => h,
        Err(err) => {
            warn!(
                err = %err,
                "wm-brain recap: thread query failed; proceeding without recap context (AC8)"
            );
            *state.recap_context.lock().await = None;
            return false;
        }
    };

    // Filter to committed memories (AC5: proposals are excluded).
    // The recall query already returns committed memories; we document the
    // intent here. Subject must start with the thread prefix.
    let thread_hits: Vec<&QueryHit> = hits
        .iter()
        .filter(|h| h.subject.starts_with(THREAD_SUBJECT_PREFIX))
        .take(recap_max)
        .collect();

    let thread_hits_owned: Vec<QueryHit> =
        thread_hits.iter().map(|h| (*h).clone()).collect();
    let recap = format_recap_context(&thread_hits_owned);
    state.recap_context.lock().await.clone_from(&recap);

    if let Some(ref ctx) = recap {
        info!(
            memories = thread_hits.len(),
            "wm-brain recap: session-start thread context fetched"
        );
        // AC7: optional opener — off by default (AC6).
        if recap_opener {
            if let Some(first_snippet) = thread_hits
                .first()
                .map(|h| h.snippet.trim().to_string())
                .filter(|s| !s.is_empty())
            {
                let opener_text = format!("Earlier you mentioned: {first_snippet}");
                let reply = bus::ReplyEvent { text: opener_text, ts: now_ms, loudness: None };
                if let Ok(payload) = serde_json::to_value(&reply) {
                    if let Err(err) = publish.publish(outgoing::REPLY, payload).await {
                        warn!(err = %err, "wm-brain recap: opener publish failed");
                    } else {
                        info!("wm-brain recap: continuity opener published");
                        return true;
                    }
                }
            }
        }
        let _ = ctx;
    } else {
        debug!("wm-brain recap: no thread memories found; cold store");
    }

    false
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

/// Handle a classified repair request (PRD-wmd-repair-affordances §2.2).
///
/// Reads the last assistant turn from `state.history`. On `RepeatLast`, re-
/// publishes its text with a fresh `ts`. On `RepeatLouder`, does the same but
/// adds `loudness = "loud"` to the event. When history is empty, publishes the
/// graceful degrade phrase [`REPAIR_EMPTY_HISTORY_REPLY`] instead.
///
/// The replayed turn is **not** pushed back into history (AC4 — prevents
/// "say that again" × N from filling the ring with duplicates).
///
/// # Errors
/// Propagates publish failures.
async fn handle_repair(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    repair: Repair,
    now_ms: u64,
) -> Result<()> {
    let last_text = {
        let history = state.history.lock().await;
        history.last().map(|t| t.assistant.clone())
    };

    let (text, loudness) = if let Some(t) = last_text {
        let loud = (repair == Repair::RepeatLouder).then(|| "loud".to_string());
        (t, loud)
    } else {
        info!("wm-brain: repair request with empty history; emitting degrade phrase");
        (REPAIR_EMPTY_HISTORY_REPLY.to_string(), None)
    };

    info!(
        repair = ?repair,
        loudness = ?loudness,
        "wm-brain: repair replay"
    );
    let reply = ReplyEvent { text, ts: now_ms, loudness };
    publish
        .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
        .await
        .context("publish repair replay")?;
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "per-turn handler: state + sink + model + child_lock + turn + ts; refactoring into \
              a struct would just shuffle the call sites"
)]
#[allow(
    clippy::too_many_arguments,
    clippy::cognitive_complexity,
    reason = "turn dispatch shell: ladder vs single-client branch, each linear"
)]
async fn handle_turn_user(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    model: &str,
    tier: &str,
    child_lock: bool,
    turn: &TurnUserEvent,
    now_ms: u64,
) -> Result<()> {
    // Advance the writeback session (check idle gap, open new session if needed).
    advance_writeback_session(state, now_ms).await;

    // --- Repair-affordance check (PRD-wmd-repair-affordances §2.1 / §2.2) ---
    //
    // Run BEFORE the LLM dispatch. If the transcript is a verbatim-replay or
    // louder-replay request, handle it locally from history and return — no
    // model call, no token cost, near-zero latency.
    {
        let repair_result = {
            let cfg = state.config.lock().await;
            let (repeat_set, louder_set) = repair_phrase_sets(&cfg);
            repair::classify(&turn.transcript, &repeat_set, &louder_set)
        };
        if repair_result != Repair::None {
            return handle_repair(state, publish, repair_result, now_ms).await;
        }
    }

    // When a ladder is configured it owns dispatch (local-first + climb);
    // otherwise fall back to the single Anthropic client path.
    if state.ladder.is_some() {
        return handle_turn_user_ladder(state, publish, model, tier, child_lock, turn, now_ms).await;
    }
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
    let recap = { state.recap_context.lock().await.clone() };
    let persona = compose_persona(&state.persona, child_lock, context.as_deref(), recap.as_deref());
    // Build the request with history prefix (PRD-wmd-turn-history §2.2).
    let history_msgs = {
        let history = state.history.lock().await;
        history.trimmed_messages(DEFAULT_MAX_TOKENS as usize)
    };
    let req = compose_request(model, &persona, &history_msgs, &turn.transcript);
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
                // Destructive turns store the *spoken* prefix, not the JSON
                // fence, so the companion hears the right text on replay
                // (PRD-wmd-turn-history §2.1 + AC5).
                let assistant_stored = spoken.clone();
                publish_destructive(state, publish, intent, spoken, now_ms).await?;
                // Push the stored spoken text as the assistant turn.
                let mut history = state.history.lock().await;
                history.push(Turn {
                    user: turn.transcript.clone(),
                    assistant: assistant_stored.clone(),
                    ts: now_ms,
                });
                drop(history);
                record_turn_for_writeback(state, &turn.transcript, &assistant_stored, now_ms).await;
            } else {
                let stored = text.clone();
                let reply = ReplyEvent { text, ts: now_ms, loudness: None };
                publish
                    .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
                    .await
                    .context("publish reply")?;
                // Only push after successful publish (AC3 — errors don't
                // pollute history).
                let mut history = state.history.lock().await;
                history.push(Turn {
                    user: turn.transcript.clone(),
                    assistant: stored.clone(),
                    ts: now_ms,
                });
                drop(history);
                record_turn_for_writeback(state, &turn.transcript, &stored, now_ms).await;
            }
        }
        Err(err) => {
            error!(err = %err, model = %model, "wm-brain: anthropic call failed");
            publish_error_at(publish, "anthropic", &format!("{err}"), now_ms).await?;
            // Do NOT push to history on LLM failure (AC3).
        }
    }
    Ok(())
}

/// Dispatch a turn through the tier ladder. Recall context + persona are
/// composed exactly as the single-client path; the ladder picks the tier
/// and climbs as needed. Filler backchannels are published as interim
/// replies before the final answer (AC9). A terminal degrade publishes a
/// typed error rather than going silent (AC5).
#[allow(
    clippy::too_many_arguments,
    clippy::cognitive_complexity,
    reason = "turn outcome shell: publish fillers, then answer/degrade, each linear"
)]
async fn handle_turn_user_ladder(
    state: &DaemonState,
    publish: &mut dyn EventSink,
    model: &str,
    _tier: &str,
    child_lock: bool,
    turn: &TurnUserEvent,
    now_ms: u64,
) -> Result<()> {
    let Some(ladder) = state.ladder.as_ref() else {
        return Ok(());
    };

    // --- Routing policy (PRD-wintermute-brain-routing §2.2) ---
    //
    // Compute the routing decision before the ladder runs. The decision
    // supplies an *advisory* starting tier and a machine-readable reason for
    // the `wm.brain.route` observability envelope.
    let (routing_config, pending_tier_override, api_key_present) = {
        let cfg = state.config.lock().await;
        (
            cfg.routing.clone(),
            cfg.pending_tier.clone(),
            !std::env::var(&cfg.api_key_env).unwrap_or_default().is_empty(),
        )
    };
    // Reachability is currently derived from whether we have an API key and the
    // ladder has an Anthropic client — a lightweight heuristic that avoids a
    // blocking TCP probe on the hot path. A full TTL-cached probe is deferred
    // to a later iteration.
    let online = api_key_present;
    let route_decision = apply_routing_policy(&PolicyInputs {
        transcript: turn.transcript.clone(),
        pending_tier_override,
        api_key_present,
        online,
        prefer: routing_config.prefer,
        command_max_words: routing_config.command_max_words,
    });
    // Advisory starting tier from routing policy; the ladder may escalate further.
    let routing_start_tier = match route_decision.tier {
        RouteTier::Local => crate::TIER_LOCAL_3B,
        RouteTier::Cloud => crate::SHORT_MODEL_SONNET,
        RouteTier::Canned => {
            // Both backends are flagged unavailable by policy.
            // Emit a canned phrase and publish route event.
            let turn_count = state.intent_counter.load(Ordering::Relaxed);
            #[allow(clippy::as_conversions, reason = "small counter index")]
            let phrase_idx: usize = turn_count as usize;
            let phrase = canned_phrase(phrase_idx);
            let reply = ReplyEvent { text: phrase.to_string(), ts: now_ms, loudness: None };
            publish
                .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
                .await
                .context("publish canned reply")?;
            let route_evt = RouteEvent {
                turn_id: now_ms,
                tier: RouteTier::Canned,
                reason: route_decision.reason.as_str().to_string(),
                latency_ms: Some(0),
                model: "canned".to_string(),
                ts: now_ms,
            };
            if let Ok(v) = serde_json::to_value(&route_evt) {
                if let Err(e) = publish.publish(outgoing::ROUTE, v).await {
                    warn!(err = %e, "wm-brain: failed to publish route event (canned)");
                }
            }
            return Ok(());
        }
    };
    // The caller-supplied `tier` is the config default; the routing decision
    // may override it (e.g. command → local-3b even when config default is sonnet).
    let effective_tier = routing_start_tier;

    let hits = match state.recall.fetch(&turn.transcript).await {
        Ok(h) => h,
        Err(err) => {
            warn!(err = %err, "wm-brain: recall fetch failed; proceeding without context");
            Vec::new()
        }
    };
    let context = format_recall_context(&hits);
    let recap = { state.recap_context.lock().await.clone() };
    let persona = compose_persona(&state.persona, child_lock, context.as_deref(), recap.as_deref());
    // Build the request with history prefix (PRD-wmd-turn-history §2.2).
    let history_msgs = {
        let history = state.history.lock().await;
        history.trimmed_messages(DEFAULT_MAX_TOKENS as usize)
    };
    let req = compose_request(model, &persona, &history_msgs, &turn.transcript);

    let start_ms = crate::bus::now_unix_ms();
    let sink = crate::ladder::BufferingSink::default();
    let outcome = ladder
        .run_turn_sticky(&turn.transcript, &req, effective_tier, &sink, &state.session_floor)
        .await;
    let latency_ms = crate::bus::now_unix_ms().saturating_sub(start_ms);

    // Publish any filler backchannels first (AC9), then the answer.
    for filler in sink.take_fillers() {
        let reply = ReplyEvent { text: filler, ts: now_ms, loudness: None };
        publish
            .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
            .await
            .context("publish ladder filler")?;
    }

    match outcome {
        crate::ladder::LadderOutcome::Answer { text, tier: served } => {
            if text.is_empty() {
                warn!(tier = %served, "wm-brain ladder: empty answer; emitting error");
                publish_error_at(publish, "ladder", "no text in response", now_ms).await?;
                // Do NOT push empty-answer turns to history (AC3).
                return Ok(());
            }
            info!(
                tier = %served,
                reason = %route_decision.reason.as_str(),
                latency_ms = latency_ms,
                "wm-brain ladder: turn served"
            );
            // Publish wm.brain.route observability event (PRD §2.5).
            let route_evt = RouteEvent {
                turn_id: now_ms,
                tier: route_decision.tier,
                reason: route_decision.reason.as_str().to_string(),
                latency_ms: Some(latency_ms),
                model: served.clone(),
                ts: now_ms,
            };
            if let Ok(v) = serde_json::to_value(&route_evt) {
                if let Err(e) = publish.publish(outgoing::ROUTE, v).await {
                    warn!(err = %e, "wm-brain: failed to publish route event");
                }
            }
            touch_recalled_hits(state.recall.as_ref(), &hits).await;
            if let Some((intent, spoken)) = parse_destructive_intent(&text) {
                // Store the spoken prefix (AC5 — destructive turns store
                // what the user heard, not the JSON fence).
                let assistant_stored = spoken.clone();
                publish_destructive(state, publish, intent, spoken, now_ms).await?;
                let mut history = state.history.lock().await;
                history.push(Turn {
                    user: turn.transcript.clone(),
                    assistant: assistant_stored.clone(),
                    ts: now_ms,
                });
                drop(history);
                record_turn_for_writeback(state, &turn.transcript, &assistant_stored, now_ms).await;
            } else {
                let stored = text.clone();
                let reply = ReplyEvent { text, ts: now_ms, loudness: None };
                publish
                    .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
                    .await
                    .context("publish ladder reply")?;
                let mut history = state.history.lock().await;
                history.push(Turn {
                    user: turn.transcript.clone(),
                    assistant: stored.clone(),
                    ts: now_ms,
                });
                drop(history);
                record_turn_for_writeback(state, &turn.transcript, &stored, now_ms).await;
            }
        }
        crate::ladder::LadderOutcome::Degraded { reason } => {
            error!(reason = %reason, "wm-brain ladder: turn degraded (no tier could serve)");
            // Publish a canned degrade phrase rather than going silent (AC7).
            let turn_count = state.intent_counter.load(Ordering::Relaxed);
            #[allow(clippy::as_conversions, reason = "small counter index")]
            let phrase_idx: usize = turn_count as usize;
            let phrase = canned_phrase(phrase_idx);
            let reply = ReplyEvent { text: phrase.to_string(), ts: now_ms, loudness: None };
            publish
                .publish(outgoing::REPLY, serde_json::to_value(&reply)?)
                .await
                .context("publish canned degrade reply")?;
            // Publish route event with canned tier.
            let route_evt = RouteEvent {
                turn_id: now_ms,
                tier: RouteTier::Canned,
                reason: crate::router::RouteReason::TotalFailure.as_str().to_string(),
                latency_ms: Some(latency_ms),
                model: "canned".to_string(),
                ts: now_ms,
            };
            if let Ok(v) = serde_json::to_value(&route_evt) {
                if let Err(e) = publish.publish(outgoing::ROUTE, v).await {
                    warn!(err = %e, "wm-brain: failed to publish route event (degraded)");
                }
            }
            // Do NOT push degraded turns to history (AC3).
        }
    }
    Ok(())
}

/// Check the idle gap and, if expired, fire writeback for the old session.
///
/// This is called at the *start* of each `turn.user` handler, before the
/// turn is processed.  If the gap since `last_turn_ms` exceeds
/// `idle_gap_ms`, the old session is finalised (writeback fires) and a
/// new one is opened.  The history ring is NOT cleared here (history
/// scoping is PRD-wmd-session-boundary work); we only track the turns for
/// writeback purposes.
///
/// Returns the `session_id` that should receive the current turn (new or
/// continued).
///
/// PRD-wmd-memory-writeback §2.2 / AC7.
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "session-boundary + spawn shell: two linear branches (gap expired / still active); \
              factoring out the spawn would introduce more complexity, not less"
)]
async fn advance_writeback_session(state: &DaemonState, now_ms: u64) -> String {
    let idle_gap_ms = { state.config.lock().await.idle_gap_ms };
    let mut sess_guard = state.writeback_session.lock().await;

    if let Some(ref current) = *sess_guard {
        let elapsed = now_ms.saturating_sub(current.last_turn_ms);
        if elapsed >= idle_gap_ms {
            // Idle gap expired — fire writeback for the closing session.
            let old_session_id = current.session_id.clone();
            let old_turns = current.turns.clone();
            let (config_snap, date_str) = {
                let cfg = state.config.lock().await;
                let date = chrono_local_date();
                (cfg.clone(), date)
            };
            if let Some(ref extractor) = state.extractor {
                // Spawn best-effort writeback; do not await (avoid blocking the
                // turn path — writeback must never stall the conversation).
                let recall = Arc::clone(&state.recall);
                let extractor_clone: Arc<dyn ExtractorClient> = Arc::clone(extractor);
                let guard_ref: &WritebackGuard = &state.writeback_guard;
                // We must await here to get the `try_claim` done before we
                // drop the session (otherwise the next advance_writeback_session
                // for the same id might see a live `writeback_session` entry
                // that's already been claimed).  The actual I/O is spawned.
                if guard_ref.try_claim(&old_session_id).await {
                    let old_session_id2 = old_session_id.clone();
                    tokio::spawn(async move {
                        // The guard was already claimed above; use a
                        // pre-claimed guard so trigger_writeback doesn't
                        // double-claim.
                        let single_use_guard = WritebackGuard::new();
                        let _ = single_use_guard.try_claim(&old_session_id2).await;
                        // Release and let trigger_writeback re-claim.
                        // Actually we call the inner path directly to avoid
                        // double-claim confusion.
                        trigger_writeback_inner(
                            &old_session_id2,
                            &old_turns,
                            &date_str,
                            &config_snap,
                            extractor_clone.as_ref(),
                            recall.as_ref(),
                        )
                        .await;
                    });
                }
            }
            // Open a new session.
            let new_id = format!("wmd-sess-{now_ms}");
            *sess_guard = Some(WritebackSession {
                session_id: new_id.clone(),
                last_turn_ms: now_ms,
                turns: Vec::new(),
            });
            return new_id;
        }
        // Still within the gap — extend the current session.
        let id = current.session_id.clone();
        if let Some(ref mut s) = *sess_guard {
            s.last_turn_ms = now_ms;
        }
        return id;
    }

    // No current session — open the first one.
    let new_id = format!("wmd-sess-{now_ms}");
    *sess_guard = Some(WritebackSession {
        session_id: new_id.clone(),
        last_turn_ms: now_ms,
        turns: Vec::new(),
    });
    new_id
}

/// Record a completed turn into the current writeback session's turn list.
async fn record_turn_for_writeback(state: &DaemonState, user: &str, assistant: &str, ts: u64) {
    let mut sess_guard = state.writeback_session.lock().await;
    if let Some(ref mut s) = *sess_guard {
        s.turns.push(Turn {
            user: user.to_string(),
            assistant: assistant.to_string(),
            ts,
        });
    }
}

/// Inner writeback driver that bypasses the guard (called only after the
/// guard has already been claimed by the caller).
#[allow(
    clippy::too_many_arguments,
    reason = "writeback context: session_id + turns + date + config + extractor + recall; \
              each is semantically distinct"
)]
async fn trigger_writeback_inner(
    session_id: &str,
    turns: &[Turn],
    date_str: &str,
    config: &BrainConfig,
    extractor: &dyn ExtractorClient,
    recall: &dyn RecallSource,
) {
    use crate::writeback::{MIN_TURNS_FOR_WRITEBACK, parse_extraction_response, render_transcript,
                           recall_subject_for};

    if turns.len() < MIN_TURNS_FOR_WRITEBACK {
        debug!(session_id, "writeback(inner): too few turns; skipping");
        return;
    }
    let transcript = render_transcript(turns);
    let raw = match extractor.extract(&transcript).await {
        Ok(r) => r,
        Err(err) => {
            warn!(session_id, err = %err, "writeback(inner): extraction failed; skipping");
            return;
        }
    };
    let facts = parse_extraction_response(&raw, config.writeback_confidence_floor);
    if facts.is_empty() {
        debug!(session_id, "writeback(inner): no facts extracted");
        return;
    }
    let mut written = 0usize;
    for fact in &facts {
        let subject = recall_subject_for(&fact.subject_hint, date_str);
        match recall.save_fact(&subject, &fact.body).await {
            Ok(id) => {
                debug!(session_id, id = %id, subject = %subject, "writeback(inner): fact written");
                written = written.saturating_add(1);
            }
            Err(err) => {
                warn!(session_id, subject = %subject, err = %err,
                      "writeback(inner): save_fact failed; skipping fact");
            }
        }
    }
    info!(session_id, facts_extracted = facts.len(), facts_written = written,
          "writeback(inner): session writeback complete");
}

/// Get today's date as an ISO-8601 string (`YYYY-MM-DD`).
///
/// Uses the system clock; falls back to `"1970-01-01"` if the clock is
/// somehow unavailable (should never happen in practice).
#[must_use]
fn chrono_local_date() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    // Compute YYYY-MM-DD from Unix timestamp (no chrono dep).
    // Algorithm: count days since epoch, then compute calendar date.
    let days = secs / 86400;
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Convert days-since-Unix-epoch to `(year, month, day)`.
///
/// Uses the proleptic Gregorian calendar algorithm.
/// Algorithm from <https://howardhinnant.github.io/date_algorithms.html> (public domain).
#[allow(
    clippy::similar_names,
    reason = "doe/doy are standard names in the Hinnant date algorithm; renaming would obscure \
              the well-known derivation"
)]
const fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Shift to the civil date epoch (days since 1 Jan 1970 → proleptic
    // Gregorian days since 1 March 0000 using the 400-year cycle).
    let z = days.wrapping_add(719_468);
    let era = z / 146_097;
    let doe = z % 146_097; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
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
        loudness: None,
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

/// Build the tier-ladder orchestrator (PRD-brain-backend-ladder).
///
/// Local tiers serve from `cfg.local_endpoint`; cloud tiers use the
/// (optional) Anthropic client. The brain stays up even with no API key —
/// local tiers serve, cloud tiers degrade. Stakes come from a keyword+rules
/// router; recall liveness drives the recall-down safe posture.
fn build_ladder(
    cfg: &BrainConfig,
    llm: Option<&Arc<dyn LlmClient>>,
) -> Arc<crate::ladder::LadderClient> {
    let ladder = Arc::new(crate::ladder::LadderClient::new(
        crate::default_ladder(),
        Arc::new(crate::ladder::LiveLocalBackend::new(cfg.local_endpoint.clone())),
        llm.cloned(),
        Arc::new(crate::ladder::RouterStakes::new()),
        Arc::new(crate::ladder::SocketRecallLiveness::new(cfg.recall_sock.clone())),
    ));
    info!(
        default_tier = %cfg.resolved_default_tier(),
        local_endpoint = %cfg.local_endpoint,
        has_api_key = llm.is_some(),
        "wm-brain: tier ladder wired (local-first; cloud tiers gated on api key)"
    );
    ladder
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
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "live daemon entry-point: sequential setup + subscribe loop; refactoring into smaller \
              functions would scatter the startup sequence and obscure the linear flow"
)]
pub async fn run(cfg: BrainConfig, config_path: Option<PathBuf>) -> Result<()> {
    cfg.validate().context("wm-brain: config validation failed")?;

    let llm = build_llm_from_env(&cfg.api_key_env);
    let recall: Arc<dyn RecallSource> = Arc::new(LiveRecallSource::new(cfg.recall_sock.clone()));
    info!(
        socket = %cfg.recall_sock.display(),
        "wm-brain: recall source attached (live, connects per turn)"
    );
    let tool_router: Arc<dyn ToolRouter> =
        Arc::new(RecallToolsRouter::new(Arc::clone(&recall)));
    info!(
        tools = TOOL_RECALL_SEARCH,
        tools_extra = TOOL_RECALL_SAVE_FACT,
        "wm-brain: tool router wired with recall surface"
    );

    let ladder_client = build_ladder(&cfg, llm.as_ref());

    let state = Arc::new({
        let mut base = DaemonState::new(cfg)
            .with_recall(recall)
            .with_tool_router(tool_router)
            .with_ladder(ladder_client);
        if let Some(p) = config_path {
            base = base.with_config_path(p);
        }
        // Wire the extraction client for end-of-session writeback when
        // an LLM is available (PRD-wmd-memory-writeback §2.2 / §2.4).
        // `cfg.writeback_model` is moved before this; re-read from state.
        base = match llm {
            Some(client) => {
                let extractor_model = { base.config.lock().await.writeback_model.clone() };
                let extractor: Arc<dyn ExtractorClient> =
                    Arc::new(AnthropicExtractor::new(Arc::clone(&client), extractor_model));
                base.with_llm(client).with_extractor(extractor)
            }
            None => base,
        };
        base
    });

    // `WM_BRAIN_BUS_SOCKET` override mirrors `wm-dialog`'s `WM_DIALOG_BUS_SOCKET`
    // / `wm-stt`'s `WM_STT_BUS_SOCKET` / `wm-tts`'s `WM_TTS_BUS_SOCKET` idiom
    // and lets `tests/bus_smoke.rs` point the daemon at a per-test temp
    // socket without touching $HOME.
    let sock = std::env::var("WM_BRAIN_BUS_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| agorabus::default_socket_path());
    let Some(mut sub_client) = agorabus::Client::try_connect(&sock).await? else {
        warn!(socket = %sock.display(), "wm-brain: agorabus not reachable; exiting");
        return Ok(());
    };
    sub_client
        .announce(
            &format!("wm-brain-{}-sub", std::process::id()),
            std::process::id(),
            "",
            "wm-brain dialog subscribe",
        )
        .await?;
    sub_client.subscribe(bus::DIALOG_TOPIC_PREFIX).await?;
    // Also subscribe to the STT prefix so the almanac acknowledgment window
    // can intercept `wm.stt.final` transcripts when a PendingAck is open
    // (PRD-almanac-acknowledge AC1–AC3).
    sub_client
        .subscribe(crate::almanac::STT_TOPIC_PREFIX)
        .await?;
    // Subscribe to the almanac prefix so `wm.almanac.due` events reach the
    // speak-bridge handler (PRD-almanac-speak-bridge).
    sub_client
        .subscribe(crate::almanac::ALMANAC_TOPIC_PREFIX)
        .await?;
    // Subscribe to component error topics for the graceful-degradation
    // aggregator (PRD-wintermute-companion-degrade §2.2).
    for error_topic in [
        crate::degrade::STT_ERROR_TOPIC,
        crate::degrade::TTS_ERROR_TOPIC,
        crate::degrade::AUDIO_ERROR_TOPIC,
        crate::degrade::BRAIN_ERROR_TOPIC,
    ] {
        sub_client.subscribe(error_topic).await?;
    }
    info!(
        dialog_prefix = bus::DIALOG_TOPIC_PREFIX,
        stt_prefix = crate::almanac::STT_TOPIC_PREFIX,
        almanac_prefix = crate::almanac::ALMANAC_TOPIC_PREFIX,
        "wm-brain: subscribed"
    );

    let mut pub_client = agorabus::Client::connect(&sock).await?;
    pub_client
        .announce(
            &format!("wm-brain-{}", std::process::id()),
            std::process::id(),
            "",
            "wm-brain publish path",
        )
        .await?;
    let pub_arc = Arc::new(Mutex::new(pub_client));
    let mut sink = AgoraSink {
        inner: Arc::clone(&pub_arc),
    };

    // Heartbeat keepalive — the bus daemon prunes peers from its
    // `peers` snapshot when `last_heartbeat_unix_secs` ages past
    // `DEFAULT_HEARTBEAT_TIMEOUT_SECS` (60s). Both the publish-owner
    // session (`wm-brain-{pid}`) and the subscribe-owner session
    // (`wm-brain-{pid}-sub`) need their own ticker, since each
    // connection owns a distinct peer record keyed by session_id. See
    // PRD wintermute-fleet-bus-heartbeat-keepalive §4.
    let hb_interval = std::time::Duration::from_secs(
        agorabus::DEFAULT_HEARTBEAT_TIMEOUT_SECS / 2,
    );
    let pub_hb_arc = Arc::clone(&pub_arc);
    let _pub_hb_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(hb_interval);
        ticker.tick().await; // skip the immediate first tick
        loop {
            ticker.tick().await;
            let mut client = pub_hb_arc.lock().await;
            if let Err(e) = client.heartbeat("wm-brain").await {
                warn!(error = %e, "wm-brain: pub heartbeat failed; bus likely gone");
                return;
            }
        }
    });

    // Split sub_client into halves so the heartbeat ticker shares the
    // wire with the reader loop. Heartbeat replies on this wire are
    // filtered by the `InboundLine::Reply` skip below.
    let (mut sub_write, mut sub_reader) = sub_client.into_halves();
    let _sub_hb_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(hb_interval);
        ticker.tick().await; // skip the immediate first tick
        loop {
            ticker.tick().await;
            if let Err(e) = agorabus::client::send_heartbeat(&mut sub_write, "wm-brain").await {
                warn!(error = %e, "wm-brain: sub heartbeat failed; bus likely gone");
                return;
            }
        }
    });

    // Health snapshot ticker — emits `wm.health.snapshot` every 60 s
    // (PRD-wintermute-companion-degrade §2.4).
    let health_pub_arc = Arc::clone(&pub_arc);
    let health_arc = Arc::clone(&state.degrade_health);
    let _health_ticker_task = tokio::spawn(async move {
        let interval = std::time::Duration::from_millis(HEALTH_SNAPSHOT_INTERVAL_MS);
        let mut ticker = tokio::time::interval(interval);
        ticker.tick().await; // skip immediate first tick
        loop {
            ticker.tick().await;
            let now = now_unix_ms();
            let snap = health_arc.snapshot(now);
            match snapshot_payload(&snap) {
                Ok(payload) => {
                    let mut client = health_pub_arc.lock().await;
                    if let Err(e) = client.publish(HEALTH_SNAPSHOT_TOPIC, payload).await {
                        warn!(error = %e, "wm-brain degrade: health snapshot publish failed");
                    } else {
                        debug!("wm-brain degrade: health snapshot published");
                    }
                }
                Err(e) => {
                    warn!(error = %e, "wm-brain degrade: health snapshot serialise failed");
                }
            }
        }
    });

    // Manual InboundLine reader replaces `sub_client.next_event()`.
    // `next_event` takes `&mut self` on the whole Client, which a
    // spawned task cannot reach after `into_halves`.
    loop {
        let line = match sub_reader.next_line().await {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(err) => {
                error!(error = %err, "wm-brain: subscribe wire read failed");
                break;
            }
        };
        let parsed: agorabus::client::InboundLine = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(err) => {
                warn!(error = %err, line = %line, "wm-brain: undecodable bus line; skipping");
                continue;
            }
        };
        let ev = match parsed {
            agorabus::client::InboundLine::Reply(_) => continue,
            agorabus::client::InboundLine::Event(ev) => ev,
        };
        // Route `wm.stt.final` events to the almanac ack handler first.
        // When a PendingAck is open and the transcript is consumed (Done or
        // Snooze/exhausted), the event is NOT forwarded to the normal dialog
        // dispatch (AC7: no double-processing).  For Unrelated transcripts
        // `handle_stt_final_for_ack` returns false and we fall through to
        // the normal path.
        if ev.topic == crate::almanac::STT_FINAL_TOPIC {
            let transcript = ev
                .data
                .get("transcript")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let now = now_unix_ms();
            match handle_stt_final_for_ack(state.as_ref(), &mut sink, &transcript, now).await {
                Ok(true) => {
                    // Consumed by ack path — also run the timeout check in
                    // case the window has since elapsed (belt-and-suspenders).
                    let _ = tick_almanac_timeout(state.as_ref(), &mut sink, now).await;
                    continue;
                }
                Ok(false) => {
                    // Not consumed; check timeout regardless, then fall through.
                    let _ = tick_almanac_timeout(state.as_ref(), &mut sink, now).await;
                    // If the topic is not under the dialog prefix, skip dialog dispatch.
                    if !ev.topic.starts_with(bus::DIALOG_TOPIC_PREFIX) {
                        continue;
                    }
                }
                Err(err) => {
                    error!(err = %err, "wm-brain: almanac ack handler failed");
                    let _ = publish_error(&mut sink, "almanac", &format!("{err}")).await;
                    continue;
                }
            }
        }

        // Route `wm.almanac.due` to the speak-bridge handler
        // (PRD-almanac-speak-bridge AC1–AC4). Other `wm.almanac.*` topics
        // (ack, snooze) are outbound — we published them ourselves, so any
        // echo from the bus is silently dropped below.
        if ev.topic == crate::almanac::ALMANAC_DUE_TOPIC {
            let now = now_unix_ms();
            let parsed_ev: std::result::Result<crate::almanac::AlmanacDueEvent, _> =
                serde_json::from_value(ev.data.clone());
            match parsed_ev {
                Ok(due_ev) => {
                    match handle_speak_almanac_due(state.as_ref(), &mut sink, &due_ev, now).await {
                        Ok(_) => {}
                        Err(err) => {
                            error!(err = %err, "wm-brain: almanac speak-bridge failed");
                            let _ =
                                publish_error(&mut sink, "almanac", &format!("{err}")).await;
                        }
                    }
                }
                Err(err) => {
                    // AC4: malformed envelope — log WARN, never panic the loop.
                    warn!(
                        err = %err,
                        topic = %ev.topic,
                        "wm-brain: wm.almanac.due parse failed; dropping"
                    );
                }
            }
            continue;
        }

        // Silently skip non-dialog almanac topics (ack/snooze echoes).
        if ev.topic.starts_with(crate::almanac::ALMANAC_TOPIC_PREFIX)
            && !ev.topic.starts_with(bus::DIALOG_TOPIC_PREFIX)
        {
            debug!(topic = %ev.topic, "wm-brain: ignoring non-due almanac topic");
            continue;
        }

        // Route component error topics to the graceful-degradation aggregator
        // (PRD-wintermute-companion-degrade §2.2). If a phrase should be
        // spoken, publish `wm.tts.speak` with priority "system".
        if component_for_error_topic(&ev.topic).is_some() {
            let now = now_unix_ms();
            if let Some(phrase) = process_error_envelope(
                &ev.topic,
                &ev.data,
                &state.degrade_rate,
                &state.degrade_health,
                now,
            ) {
                let payload = speak_payload(&phrase, now);
                if let Err(e) = sink.publish(TTS_SPEAK_TOPIC, payload).await {
                    warn!(error = %e, "wm-brain degrade: tts.speak publish failed");
                }
            }
            // Brain's own errors still continue to dialog decode below only if
            // the topic is also under the dialog prefix (it isn't, so we skip).
            if !ev.topic.starts_with(bus::DIALOG_TOPIC_PREFIX) {
                continue;
            }
        }

        // AC7 (PRD-wmd-session-boundary): ignore self-emitted session.*
        // envelopes that the bus echoes back to our own subscription.
        // These are outbound-only events; re-ingesting them would create a
        // feedback loop (brain hears its own session.start / session.end).
        if ev.topic.starts_with(bus::SESSION_TOPIC_PREFIX) {
            debug!(topic = %ev.topic, "wm-brain: ignoring self-emitted session topic");
            continue;
        }

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
    // Graceful shutdown: close any open session so downstream PRDs
    // (writeback, recap) get their trigger.
    // Best-effort — we don't abort the shutdown on publish failure.
    {
        let closed = {
            let mut tracker = state.session_tracker.lock().await;
            tracker.close(now_unix_ms(), CloseReason::Shutdown)
        };
        if let Some(end_payload) = closed {
            info!(
                session_id = %end_payload.session_id,
                turn_count = end_payload.turn_count,
                "wm-brain: session closed on shutdown"
            );
            if let Ok(v) = serde_json::to_value(&end_payload) {
                let _ = sink.publish(outgoing::SESSION_END, v).await;
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
    clippy::as_conversions,
    clippy::items_after_statements,
    clippy::type_complexity,
    clippy::mutex_integer,
    clippy::doc_markdown,
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

    impl MemSink {
        /// Return only events that are NOT session boundary events
        /// (`wm.brain.session.*`). Used by tests focused on LLM dispatch
        /// behaviour so they don't need to account for the session events
        /// that are always emitted by the session tracker.
        fn non_session_events(&self) -> Vec<(String, Value)> {
            self.events
                .lock()
                .expect("mem sink poisoned")
                .iter()
                .filter(|(topic, _)| !topic.starts_with(bus::SESSION_TOPIC_PREFIX))
                .cloned()
                .collect()
        }

        /// Return only `wm.brain.session.*` events.
        fn session_events(&self) -> Vec<(String, Value)> {
            self.events
                .lock()
                .expect("mem sink poisoned")
                .iter()
                .filter(|(topic, _)| topic.starts_with(bus::SESSION_TOPIC_PREFIX))
                .cloned()
                .collect()
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
        let non_session = sink.non_session_events();
        assert!(
            non_session.is_empty(),
            "no LLM configured -> dispatch must not publish LLM events"
        );
        // The session tracker still emits a SESSION_START on the first turn.
        let sess_events = sink.session_events();
        assert_eq!(sess_events.len(), 1);
        assert_eq!(sess_events[0].0, outgoing::SESSION_START);
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
        // AC4 / baseline: empty history → exactly one user message (single-message behaviour).
        let req = compose_request("sonnet", "be terse", &[], "hello there");
        assert_eq!(req.model, "claude-sonnet-4-6");
        assert_eq!(req.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(req.stream);
        assert_eq!(req.system.as_deref(), Some("be terse"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].content, "hello there");
    }

    #[test]
    fn compose_request_with_history_prepends_prior_pairs() {
        // AC1 (unit): compose_request with two prior turns builds a
        // [user, asst, user, asst, current_user] list (5 messages).
        use crate::history::{History, Turn as HTurn};
        let mut h = History::new(6);
        h.push(HTurn {
            user: "first question".to_string(),
            assistant: "first answer".to_string(),
            ts: 1,
        });
        h.push(HTurn {
            user: "second question".to_string(),
            assistant: "second answer".to_string(),
            ts: 2,
        });
        let history_msgs = h.to_messages();
        let req = compose_request("sonnet", "persona", &history_msgs, "third question");
        assert_eq!(req.messages.len(), 5, "2 prior pairs + current user = 5");
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].content, "first question");
        assert_eq!(req.messages[1].role, Role::Assistant);
        assert_eq!(req.messages[1].content, "first answer");
        assert_eq!(req.messages[2].role, Role::User);
        assert_eq!(req.messages[2].content, "second question");
        assert_eq!(req.messages[3].role, Role::Assistant);
        assert_eq!(req.messages[3].content, "second answer");
        assert_eq!(req.messages[4].role, Role::User);
        assert_eq!(req.messages[4].content, "third question");
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
        let events = sink.non_session_events();
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
        let events = sink.non_session_events();
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
        let events = sink.non_session_events();
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
                loudness: None,
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
            loudness: None,
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
        searches: Arc<StdMutex<Vec<(String, Option<String>, Option<usize>)>>>,
        saved: Arc<StdMutex<Vec<(String, String)>>>,
        next_save_id: Arc<StdMutex<u64>>,
    }

    impl FakeRecall {
        fn new(hits: Vec<QueryHit>) -> Self {
            Self {
                hits,
                touched: Arc::new(StdMutex::new(Vec::new())),
                searches: Arc::new(StdMutex::new(Vec::new())),
                saved: Arc::new(StdMutex::new(Vec::new())),
                next_save_id: Arc::new(StdMutex::new(1)),
            }
        }

        fn touched_calls(&self) -> Vec<Vec<String>> {
            self.touched.lock().expect("touched poisoned").clone()
        }

        fn searches_recorded(&self) -> Vec<(String, Option<String>, Option<usize>)> {
            self.searches.lock().expect("searches poisoned").clone()
        }

        fn saved_recorded(&self) -> Vec<(String, String)> {
            self.saved.lock().expect("saved poisoned").clone()
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

        async fn search(
            &self,
            text: &str,
            subject: Option<&str>,
            limit: Option<usize>,
        ) -> std::result::Result<Vec<QueryHit>, RecallSourceError> {
            self.searches.lock().expect("searches poisoned").push((
                text.to_string(),
                subject.map(str::to_string),
                limit,
            ));
            Ok(self.hits.clone())
        }

        async fn save_fact(
            &self,
            subject: &str,
            body: &str,
        ) -> std::result::Result<String, RecallSourceError> {
            self.saved
                .lock()
                .expect("saved poisoned")
                .push((subject.to_string(), body.to_string()));
            let mut next = self.next_save_id.lock().expect("next_save_id poisoned");
            let id = format!("m-fake-{next}");
            *next = next.saturating_add(1);
            Ok(id)
        }
    }

    /// Recall source whose every `save_fact` call fails with the
    /// canonical default `Unsupported`. Useful to verify
    /// [`RecallToolsRouter`] surfaces non-`Write` errors through the
    /// recall_error path.
    #[derive(Clone, Default)]
    struct ReadOnlyRecall;

    #[async_trait::async_trait]
    impl RecallSource for ReadOnlyRecall {
        async fn fetch(
            &self,
            _transcript: &str,
        ) -> std::result::Result<Vec<QueryHit>, RecallSourceError> {
            Ok(Vec::new())
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
        let out = compose_persona("base persona", false, None, None);
        assert!(out.starts_with("base persona"));
        assert!(out.contains(DESTRUCTIVE_GATE_GUARD));
        assert!(
            !out.contains(CHILD_LOCK_GUARD),
            "child-lock off in this case"
        );
    }

    #[test]
    fn compose_persona_with_child_lock_appends_guard_before_destructive() {
        let out = compose_persona("base", true, None, None);
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
        let out = compose_persona("base", false, Some(""), None);
        assert!(out.starts_with("base"));
        assert!(out.contains(DESTRUCTIVE_GATE_GUARD));
        assert!(
            !out.contains("ctx"),
            "empty context block must not be appended"
        );
    }

    #[test]
    fn compose_persona_with_context_appends_block_after_destructive_guard() {
        let out = compose_persona("base", true, Some("ctx block"), None);
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
    async fn recall_tools_router_search_renders_hits_and_records_invocation() {
        let recall = Arc::new(FakeRecall::new(vec![
            fake_hit("m-1", "She prefers chamomile tea.", 0.91),
            fake_hit("m-2", "Likes her tea hot.", 0.74),
        ]));
        let router = RecallToolsRouter::new(Arc::clone(&recall) as Arc<dyn RecallSource>);
        let body = router
            .dispatch(
                TOOL_RECALL_SEARCH,
                &json!({"text": "tea", "subject": "wintermute-profile", "limit": 5}),
            )
            .await;
        assert!(body.ok);
        let hits = body.body["hits"].as_array().expect("hits array");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0]["id"], "m-1");
        assert_eq!(hits[0]["snippet"], "She prefers chamomile tea.");
        assert_eq!(hits[0]["subject"], "wintermute-profile");
        assert_eq!(body.body["count"], 2);
        let searches = recall.searches_recorded();
        assert_eq!(searches.len(), 1);
        assert_eq!(searches[0].0, "tea");
        assert_eq!(searches[0].1.as_deref(), Some("wintermute-profile"));
        assert_eq!(searches[0].2, Some(5));
    }

    #[tokio::test]
    async fn recall_tools_router_search_rejects_missing_or_blank_text() {
        let recall = Arc::new(FakeRecall::new(Vec::new()));
        let router = RecallToolsRouter::new(Arc::clone(&recall) as Arc<dyn RecallSource>);
        let body = router.dispatch(TOOL_RECALL_SEARCH, &json!({})).await;
        assert!(!body.ok);
        assert_eq!(body.body["error"], "bad_args");
        assert_eq!(body.body["tool"], TOOL_RECALL_SEARCH);
        let blank = router
            .dispatch(TOOL_RECALL_SEARCH, &json!({"text": "   "}))
            .await;
        assert!(!blank.ok);
        assert_eq!(blank.body["error"], "bad_args");
        assert!(recall.searches_recorded().is_empty());
    }

    #[tokio::test]
    async fn recall_tools_router_search_rejects_non_object_args() {
        let recall = Arc::new(FakeRecall::new(Vec::new()));
        let router = RecallToolsRouter::new(Arc::clone(&recall) as Arc<dyn RecallSource>);
        let body = router.dispatch(TOOL_RECALL_SEARCH, &json!(["tea"])).await;
        assert!(!body.ok);
        assert_eq!(body.body["error"], "bad_args");
        let limit_wrong = router
            .dispatch(TOOL_RECALL_SEARCH, &json!({"text": "tea", "limit": "five"}))
            .await;
        assert!(!limit_wrong.ok);
        assert_eq!(limit_wrong.body["error"], "bad_args");
    }

    #[tokio::test]
    async fn recall_tools_router_save_fact_happy_uses_default_subject() {
        let recall = Arc::new(FakeRecall::new(Vec::new()));
        let router = RecallToolsRouter::new(Arc::clone(&recall) as Arc<dyn RecallSource>);
        let body = router
            .dispatch(
                TOOL_RECALL_SAVE_FACT,
                &json!({"body": "prefers chamomile tea"}),
            )
            .await;
        assert!(body.ok);
        assert_eq!(body.body["id"], "m-fake-1");
        assert_eq!(body.body["subject"], PROFILE_SUBJECT);
        let saved = recall.saved_recorded();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].0, PROFILE_SUBJECT);
        assert_eq!(saved[0].1, "prefers chamomile tea");
    }

    #[tokio::test]
    async fn recall_tools_router_save_fact_honours_explicit_subject() {
        let recall = Arc::new(FakeRecall::new(Vec::new()));
        let router = RecallToolsRouter::new(Arc::clone(&recall) as Arc<dyn RecallSource>);
        let body = router
            .dispatch(
                TOOL_RECALL_SAVE_FACT,
                &json!({
                    "body": "daughter's name is Sara",
                    "subject": "wintermute-people",
                }),
            )
            .await;
        assert!(body.ok);
        assert_eq!(body.body["subject"], "wintermute-people");
        let saved = recall.saved_recorded();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].0, "wintermute-people");
    }

    #[tokio::test]
    async fn recall_tools_router_save_fact_rejects_blank_or_missing_body() {
        let recall = Arc::new(FakeRecall::new(Vec::new()));
        let router = RecallToolsRouter::new(Arc::clone(&recall) as Arc<dyn RecallSource>);
        let missing = router
            .dispatch(TOOL_RECALL_SAVE_FACT, &json!({"subject": "x"}))
            .await;
        assert!(!missing.ok);
        assert_eq!(missing.body["error"], "bad_args");
        let blank = router
            .dispatch(TOOL_RECALL_SAVE_FACT, &json!({"body": "   "}))
            .await;
        assert!(!blank.ok);
        assert_eq!(blank.body["error"], "bad_args");
        let bad_subject = router
            .dispatch(
                TOOL_RECALL_SAVE_FACT,
                &json!({"body": "x", "subject": ""}),
            )
            .await;
        assert!(!bad_subject.ok);
        assert_eq!(bad_subject.body["error"], "bad_args");
        assert!(recall.saved_recorded().is_empty());
    }

    #[tokio::test]
    async fn recall_tools_router_save_fact_surfaces_unsupported_error() {
        let router = RecallToolsRouter::new(Arc::new(ReadOnlyRecall) as Arc<dyn RecallSource>);
        let body = router
            .dispatch(TOOL_RECALL_SAVE_FACT, &json!({"body": "x"}))
            .await;
        assert!(!body.ok);
        assert_eq!(body.body["error"], "recall_error");
        assert_eq!(body.body["tool"], TOOL_RECALL_SAVE_FACT);
        let msg = body.body["message"].as_str().expect("message");
        assert!(msg.contains("save_fact"), "msg={msg}");
    }

    #[tokio::test]
    async fn recall_tools_router_unknown_tool_falls_through_to_fallback() {
        let recall = Arc::new(FakeRecall::new(Vec::new()));
        let router = RecallToolsRouter::new(Arc::clone(&recall) as Arc<dyn RecallSource>);
        let body = router.dispatch("wm.weather.today", &json!({})).await;
        assert!(!body.ok);
        assert_eq!(body.body["error"], "no-tools-registered");
        assert_eq!(body.body["tool"], "wm.weather.today");
        assert!(recall.saved_recorded().is_empty());
        assert!(recall.searches_recorded().is_empty());
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
        // The default persona is now composed from PersonaConfig (WarmElder);
        // verify the base appears at the start rather than hard-coding the
        // old DEFAULT_PERSONA const (PRD-hearth-persona-config §2.3).
        let expected_base = BrainConfig::default()
            .persona
            .compose_base(BrainConfig::default().user_name.as_deref());
        assert!(
            system.starts_with(&expected_base),
            "system prompt should start with composed persona base"
        );
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
        let events = sink.non_session_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::REPLY);
        assert_eq!(events[0].1["text"], "fallback reply");
        let calls = captured.lock().unwrap();
        let system = calls[0].system.as_deref().expect("system");
        // The persona base is now composed from PersonaConfig; verify the
        // default composed base is preserved on recall error.
        let expected_base = BrainConfig::default()
            .persona
            .compose_base(BrainConfig::default().user_name.as_deref());
        assert!(
            system.starts_with(&expected_base),
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
        assert!(src.data_root.is_none());
        assert_eq!(src.recall_bin, PathBuf::from(DEFAULT_RECALL_BIN));
        assert_eq!(src.save_kind, DEFAULT_SAVE_KIND);
        let custom = LiveRecallSource::new(PathBuf::from("/tmp/y.sock"))
            .with_profile_subject("subj-x")
            .with_limit(3)
            .with_data_root(PathBuf::from("/tmp/recall-root"))
            .with_recall_bin(PathBuf::from("/usr/local/bin/recall"))
            .with_save_kind("reflective");
        assert_eq!(custom.profile_subject, "subj-x");
        assert_eq!(custom.limit, 3);
        assert_eq!(custom.data_root.as_deref(), Some(std::path::Path::new("/tmp/recall-root")));
        assert_eq!(custom.recall_bin, PathBuf::from("/usr/local/bin/recall"));
        assert_eq!(custom.save_kind, "reflective");
    }

    #[tokio::test]
    async fn live_recall_source_save_fact_surfaces_spawn_failure() {
        let src = LiveRecallSource::new(PathBuf::from("/tmp/wm-brain-iter14-unused.sock"))
            .with_recall_bin(PathBuf::from(
                "/tmp/wm-brain-iter14-nonexistent-binary-xyz",
            ));
        let err = src
            .save_fact("wintermute-profile", "x")
            .await
            .expect_err("spawn should fail");
        match err {
            RecallSourceError::Write { code, message } => {
                assert_eq!(code, -1);
                assert!(message.contains("spawn"), "msg={message}");
            }
            other => panic!("expected Write, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn live_recall_source_save_fact_against_stub_binary_returns_stdout_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stub = dir.path().join("recall-stub.sh");
        std::fs::write(
            &stub,
            "#!/bin/sh\necho m-stub-42\n",
        )
        .expect("write stub");
        let mut perms = std::fs::metadata(&stub).expect("meta").permissions();
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o755);
        std::fs::set_permissions(&stub, perms).expect("chmod");
        let src = LiveRecallSource::new(PathBuf::from("/tmp/wm-brain-iter14-unused-2.sock"))
            .with_recall_bin(stub);
        let id = src
            .save_fact("wintermute-profile", "stub body")
            .await
            .expect("stub should succeed");
        assert_eq!(id, "m-stub-42");
    }

    #[tokio::test]
    async fn live_recall_source_save_fact_propagates_nonzero_exit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stub = dir.path().join("recall-fail.sh");
        std::fs::write(
            &stub,
            "#!/bin/sh\necho 'bad subject' 1>&2\nexit 2\n",
        )
        .expect("write stub");
        let mut perms = std::fs::metadata(&stub).expect("meta").permissions();
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o755);
        std::fs::set_permissions(&stub, perms).expect("chmod");
        let src = LiveRecallSource::new(PathBuf::from("/tmp/wm-brain-iter14-unused-3.sock"))
            .with_recall_bin(stub);
        let err = src
            .save_fact("wintermute-profile", "stub body")
            .await
            .expect_err("nonzero exit should surface");
        match err {
            RecallSourceError::Write { code, message } => {
                assert_eq!(code, 2);
                assert!(message.contains("bad subject"), "msg={message}");
            }
            other => panic!("expected Write, got {other:?}"),
        }
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
        let events = sink.non_session_events();
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
        let events = sink.non_session_events();
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
        let events = sink.non_session_events();
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
        let events = sink.non_session_events();
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
        let events = sink.non_session_events();
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
        let events = sink.non_session_events();
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
                // Filter out session events — this test focuses on the
                // destructive-intent gate, not the session boundary.
                let events = sink.non_session_events();
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

    // --- turn-history integration tests (PRD-wmd-turn-history) ------------

    /// A capturing LLM that returns a distinct text reply for each successive
    /// call. The responses vec is consumed in order; if exhausted it panics
    /// (test design error). Captures each `MessageRequest` for assertion.
    struct SequenceLlm {
        captured: Arc<StdMutex<Vec<MessageRequest>>>,
        responses: Arc<StdMutex<Vec<String>>>,
    }

    impl SequenceLlm {
        fn new(replies: Vec<&str>) -> Self {
            Self {
                captured: Arc::new(StdMutex::new(Vec::new())),
                responses: Arc::new(StdMutex::new(
                    replies.into_iter().map(str::to_string).collect(),
                )),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for SequenceLlm {
        async fn collect_messages(
            &self,
            req: &MessageRequest,
        ) -> std::result::Result<Vec<StreamEvent>, ClientError> {
            self.captured
                .lock()
                .expect("seq llm captured poisoned")
                .push(req.clone());
            let reply = self
                .responses
                .lock()
                .expect("seq llm responses poisoned")
                .remove(0);
            Ok(vec![
                StreamEvent::MessageStart,
                StreamEvent::TextDelta { index: 0, text: reply },
                StreamEvent::MessageStop,
            ])
        }
    }

    // AC1 — multi-turn request shape.
    // Three sequential TurnUser dispatches through a SequenceLlm. On the 3rd
    // turn the request must carry the prior two (user, assistant) pairs
    // followed by the current user transcript: messages.len() == 5,
    // alternating roles, last message is the 3rd user transcript.
    #[tokio::test]
    async fn turn_history_ac1_third_turn_carries_two_prior_pairs() {
        let llm = SequenceLlm::new(vec!["reply one", "reply two", "reply three"]);
        let captured = Arc::clone(&llm.captured);
        let cfg = BrainConfig {
            history_turns: 6,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));

        let turns = ["question one", "question two", "question three"];
        for (i, t) in turns.iter().enumerate() {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: (*t).to_string(),
                    confidence: 1.0,
                    ts: i as u64 + 1,
                }),
                i as u64 + 1,
            )
            .await
            .expect("dispatch ok");
        }

        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs.len(), 3, "three calls to the LLM");

        // Turn 1: just one message (no history yet).
        assert_eq!(reqs[0].messages.len(), 1);
        assert_eq!(reqs[0].messages[0].role, Role::User);
        assert_eq!(reqs[0].messages[0].content, "question one");

        // Turn 2: prior (user, asst) pair + current user = 3 messages.
        assert_eq!(reqs[1].messages.len(), 3);
        assert_eq!(reqs[1].messages[0].role, Role::User);
        assert_eq!(reqs[1].messages[0].content, "question one");
        assert_eq!(reqs[1].messages[1].role, Role::Assistant);
        assert_eq!(reqs[1].messages[1].content, "reply one");
        assert_eq!(reqs[1].messages[2].role, Role::User);
        assert_eq!(reqs[1].messages[2].content, "question two");

        // Turn 3: two prior (user, asst) pairs + current user = 5 messages.
        assert_eq!(reqs[2].messages.len(), 5, "AC1: 5 messages on 3rd turn");
        assert_eq!(reqs[2].messages[0].role, Role::User);
        assert_eq!(reqs[2].messages[0].content, "question one");
        assert_eq!(reqs[2].messages[1].role, Role::Assistant);
        assert_eq!(reqs[2].messages[1].content, "reply one");
        assert_eq!(reqs[2].messages[2].role, Role::User);
        assert_eq!(reqs[2].messages[2].content, "question two");
        assert_eq!(reqs[2].messages[3].role, Role::Assistant);
        assert_eq!(reqs[2].messages[3].content, "reply two");
        assert_eq!(reqs[2].messages[4].role, Role::User);
        assert_eq!(reqs[2].messages[4].content, "question three");
    }

    // AC2 — ring bound.
    // With history_turns=2, after 5 successful turns the 6th turn's request
    // carries at most 2*2+1=5 messages; oldest turns are evicted first.
    #[tokio::test]
    async fn turn_history_ac2_ring_bound_evicts_oldest() {
        let llm = SequenceLlm::new(vec!["r1", "r2", "r3", "r4", "r5", "r6"]);
        let captured = Arc::clone(&llm.captured);
        let cfg = BrainConfig {
            history_turns: 2,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));

        for i in 0u64..6 {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: format!("q{i}"),
                    confidence: 1.0,
                    ts: i,
                }),
                i,
            )
            .await
            .expect("dispatch ok");
        }

        let reqs = captured.lock().unwrap().clone();
        // The 6th request (index 5) must carry at most 2*2+1=5 messages.
        let sixth = &reqs[5];
        assert!(
            sixth.messages.len() <= 5,
            "AC2: ring bound should cap at 5 messages on 6th turn, got {}",
            sixth.messages.len()
        );
        // Last message is always the current user turn.
        assert_eq!(sixth.messages.last().unwrap().role, Role::User);
        assert_eq!(sixth.messages.last().unwrap().content, "q5");
    }

    // AC3 — failures don't pollute history.
    // A turn whose LLM call errors adds nothing to history; the next
    // successful turn's request shows the prior successful turn only.
    #[tokio::test]
    async fn turn_history_ac3_failed_turn_not_added_to_history() {
        // Turn 1: success ("reply one"), Turn 2: error, Turn 3: success ("reply three").
        // Turn 3's request should carry only turn 1's (user, asst) pair — turn 2 is absent.
        struct FailOnSecondCall {
            captured: Arc<StdMutex<Vec<MessageRequest>>>,
            calls: Arc<StdMutex<u32>>,
        }

        #[async_trait::async_trait]
        impl LlmClient for FailOnSecondCall {
            async fn collect_messages(
                &self,
                req: &MessageRequest,
            ) -> std::result::Result<Vec<StreamEvent>, ClientError> {
                let mut calls = self.calls.lock().unwrap();
                *calls += 1;
                self.captured.lock().unwrap().push(req.clone());
                if *calls == 2 {
                    return Err(ClientError::Status {
                        code: 500,
                        body: "simulated failure".to_string(),
                    });
                }
                let reply = if *calls == 1 { "reply one" } else { "reply three" };
                Ok(vec![
                    StreamEvent::MessageStart,
                    StreamEvent::TextDelta { index: 0, text: reply.to_string() },
                    StreamEvent::MessageStop,
                ])
            }
        }

        let captured = Arc::new(StdMutex::new(Vec::new()));
        let llm = FailOnSecondCall {
            captured: Arc::clone(&captured),
            calls: Arc::new(StdMutex::new(0)),
        };
        let cfg = BrainConfig {
            history_turns: 6,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));

        let turns = ["success q1", "fail q2", "success q3"];
        for (i, t) in turns.iter().enumerate() {
            let mut sink = MemSink::default();
            let _ = dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: (*t).to_string(),
                    confidence: 1.0,
                    ts: i as u64,
                }),
                i as u64,
            )
            .await;
        }

        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs.len(), 3, "three LLM calls attempted");

        // Turn 3 request should only carry turn 1's pair (turn 2 failed → not stored).
        let third = &reqs[2];
        assert_eq!(
            third.messages.len(),
            3,
            "AC3: only 1 prior successful pair + current user = 3 msgs; got {}",
            third.messages.len()
        );
        assert_eq!(third.messages[0].role, Role::User);
        assert_eq!(third.messages[0].content, "success q1");
        assert_eq!(third.messages[1].role, Role::Assistant);
        assert_eq!(third.messages[1].content, "reply one");
        assert_eq!(third.messages[2].role, Role::User);
        assert_eq!(third.messages[2].content, "success q3");
    }

    // AC4 — history_turns=0 restores single-message behaviour.
    #[tokio::test]
    async fn turn_history_ac4_disabled_single_message() {
        let llm = SequenceLlm::new(vec!["r1", "r2"]);
        let captured = Arc::clone(&llm.captured);
        let cfg = BrainConfig {
            history_turns: 0,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));

        for i in 0u64..2 {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: format!("q{i}"),
                    confidence: 1.0,
                    ts: i,
                }),
                i,
            )
            .await
            .expect("dispatch ok");
        }

        let reqs = captured.lock().unwrap().clone();
        // Both requests must carry exactly 1 message each.
        assert_eq!(reqs[0].messages.len(), 1, "AC4: first turn single message");
        assert_eq!(reqs[1].messages.len(), 1, "AC4: second turn still single message with history disabled");
        assert_eq!(reqs[1].messages[0].content, "q1");
    }

    // AC5 — destructive intent stores spoken text (not the JSON fence).
    // After a destructive turn the assistant entry in history is the spoken
    // prefix; the next turn's request must carry that spoken text, not the
    // full fence block.
    #[tokio::test]
    async fn turn_history_ac5_destructive_stores_spoken_text() {
        // Turn 1: destructive reply with spoken prefix "I will delete it."
        // Turn 2: ordinary reply — captures the request to check history.
        let destructive_body =
            "I will delete it.\n```json\n{\"intent\":\"wm.fs.rm\",\"summary\":\"delete /tmp/x\",\"confirm_keyword\":\"delete\"}\n```";

        struct TwoTurnLlm {
            captured: Arc<StdMutex<Vec<MessageRequest>>>,
            call_count: Arc<StdMutex<u32>>,
            destructive_body: String,
        }

        #[async_trait::async_trait]
        impl LlmClient for TwoTurnLlm {
            async fn collect_messages(
                &self,
                req: &MessageRequest,
            ) -> std::result::Result<Vec<StreamEvent>, ClientError> {
                let mut count = self.call_count.lock().unwrap();
                *count += 1;
                self.captured.lock().unwrap().push(req.clone());
                let text = if *count == 1 {
                    self.destructive_body.clone()
                } else {
                    "ordinary reply two".to_string()
                };
                Ok(vec![
                    StreamEvent::MessageStart,
                    StreamEvent::TextDelta { index: 0, text },
                    StreamEvent::MessageStop,
                ])
            }
        }

        let captured = Arc::new(StdMutex::new(Vec::new()));
        let llm = TwoTurnLlm {
            captured: Arc::clone(&captured),
            call_count: Arc::new(StdMutex::new(0)),
            destructive_body: destructive_body.to_string(),
        };
        let cfg = BrainConfig {
            history_turns: 6,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));

        // Turn 1: triggers destructive path.
        {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: "delete that file".to_string(),
                    confidence: 1.0,
                    ts: 1,
                }),
                1,
            )
            .await
            .expect("dispatch ok");
        }
        // Turn 2: ordinary turn — check what the LLM receives.
        {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: "what else?".to_string(),
                    confidence: 1.0,
                    ts: 2,
                }),
                2,
            )
            .await
            .expect("dispatch ok");
        }

        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs.len(), 2);
        // Turn 2's request must have 3 messages: [user1, asst1, user2].
        let second = &reqs[1];
        assert_eq!(second.messages.len(), 3, "AC5: prior destructive turn in history = 3 msgs");
        assert_eq!(second.messages[0].content, "delete that file");
        // AC5: the stored assistant text is the spoken prefix, not the JSON fence.
        assert_eq!(
            second.messages[1].content,
            "I will delete it.",
            "AC5: spoken prefix stored, not the JSON fence"
        );
        assert!(
            !second.messages[1].content.contains("```"),
            "AC5: JSON fence must not appear in stored assistant turn"
        );
        assert_eq!(second.messages[2].content, "what else?");
    }

    // AC7 — config round-trip for history_turns (dispatcher-level).
    // After two successful turns the history ring has one entry; after
    // two more the ring has two, capped at history_turns=2.
    #[tokio::test]
    async fn turn_history_ac7_history_fills_from_config_turns_setting() {
        // Build state with history_turns = 2 explicitly.
        let llm = SequenceLlm::new(vec!["r1", "r2", "r3"]);
        let captured = Arc::clone(&llm.captured);
        let cfg = BrainConfig {
            history_turns: 2,
            ..BrainConfig::default()
        };
        assert_eq!(cfg.history_turns, 2, "config reflects the value we set");
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));

        for (i, q) in ["q0", "q1", "q2"].iter().enumerate() {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: (*q).to_string(),
                    confidence: 1.0,
                    ts: i as u64,
                }),
                i as u64,
            )
            .await
            .expect("dispatch ok");
        }
        let reqs = captured.lock().unwrap().clone();
        // Third turn: max 2 pairs = 4 prior + current = 5 messages.
        let third = &reqs[2];
        assert!(
            third.messages.len() <= 5,
            "AC7: history_turns=2 caps at 5 messages, got {}",
            third.messages.len()
        );
        assert_eq!(third.messages.last().unwrap().content, "q2");
    }

    // history_turns=0 leaves no prior turns even after many rounds.
    #[tokio::test]
    async fn turn_history_disabled_never_accumulates() {
        let llm = SequenceLlm::new(vec!["r1", "r2", "r3"]);
        let captured = Arc::clone(&llm.captured);
        let cfg = BrainConfig {
            history_turns: 0,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));

        for i in 0u64..3 {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: format!("q{i}"),
                    confidence: 1.0,
                    ts: i,
                }),
                i,
            )
            .await
            .expect("dispatch ok");
        }
        let reqs = captured.lock().unwrap().clone();
        for (i, req) in reqs.iter().enumerate() {
            assert_eq!(
                req.messages.len(),
                1,
                "history disabled: turn {i} must have exactly 1 message"
            );
        }
    }

    // Empty-text LLM reply must not pollute history (AC3 variant).
    #[tokio::test]
    async fn turn_history_empty_reply_not_stored() {
        struct EmptyThenOkLlm {
            captured: Arc<StdMutex<Vec<MessageRequest>>>,
            call_count: Arc<StdMutex<u32>>,
        }
        #[async_trait::async_trait]
        impl LlmClient for EmptyThenOkLlm {
            async fn collect_messages(
                &self,
                req: &MessageRequest,
            ) -> std::result::Result<Vec<StreamEvent>, ClientError> {
                let mut c = self.call_count.lock().unwrap();
                *c += 1;
                self.captured.lock().unwrap().push(req.clone());
                if *c == 1 {
                    // Return no text deltas (empty reply).
                    Ok(vec![StreamEvent::MessageStart, StreamEvent::MessageStop])
                } else {
                    Ok(vec![
                        StreamEvent::MessageStart,
                        StreamEvent::TextDelta { index: 0, text: "ok reply".to_string() },
                        StreamEvent::MessageStop,
                    ])
                }
            }
        }
        let captured = Arc::new(StdMutex::new(Vec::new()));
        let llm = EmptyThenOkLlm {
            captured: Arc::clone(&captured),
            call_count: Arc::new(StdMutex::new(0)),
        };
        let cfg = BrainConfig { history_turns: 6, ..BrainConfig::default() };
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));

        for (i, q) in ["empty turn", "ok turn"].iter().enumerate() {
            let mut sink = MemSink::default();
            let _ = dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: (*q).to_string(),
                    confidence: 1.0,
                    ts: i as u64,
                }),
                i as u64,
            )
            .await;
        }
        let reqs = captured.lock().unwrap().clone();
        // Second request must have only 1 message (no prior — empty wasn't stored).
        assert_eq!(
            reqs[1].messages.len(),
            1,
            "empty-reply turn must not be stored in history"
        );
        assert_eq!(reqs[1].messages[0].content, "ok turn");
    }

    // History state is visible through DaemonState::history field.
    #[tokio::test]
    async fn turn_history_state_field_reflects_pushes() {
        let llm = FakeLlm { response: Ok(vec![text_delta("answer")]) };
        let cfg = BrainConfig { history_turns: 4, ..BrainConfig::default() };
        let state = Arc::new(DaemonState::new(cfg).with_llm(into_dyn_llm(llm)));

        // Before any turn, history is empty.
        assert!(state.history.lock().await.is_empty());

        let mut sink = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "hello".to_string(),
                confidence: 1.0,
                ts: 1,
            }),
            1,
        )
        .await
        .expect("dispatch ok");

        // After one successful turn, history has one entry.
        let h = state.history.lock().await;
        assert_eq!(h.len(), 1);
        let msgs = h.to_messages();
        assert_eq!(msgs[0].content, "hello");
        assert_eq!(msgs[1].content, "answer");
    }

    // History grows correctly: first turn has no prior context, second turn
    // carries the first pair, third carries both.
    #[tokio::test]
    async fn turn_history_grows_monotonically() {
        let llm = SequenceLlm::new(vec!["a1", "a2", "a3"]);
        let captured = Arc::clone(&llm.captured);
        let cfg = BrainConfig { history_turns: 6, ..BrainConfig::default() };
        let state = Arc::new(DaemonState::new(cfg).with_llm(Arc::new(llm)));
        for (i, q) in ["q1", "q2", "q3"].iter().enumerate() {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: (*q).to_string(),
                    confidence: 1.0,
                    ts: i as u64,
                }),
                i as u64,
            )
            .await
            .expect("dispatch ok");
        }
        let reqs = captured.lock().unwrap().clone();
        assert_eq!(reqs[0].messages.len(), 1, "first turn: no history");
        assert_eq!(reqs[1].messages.len(), 3, "second turn: 1 prior pair + current");
        assert_eq!(reqs[2].messages.len(), 5, "third turn: 2 prior pairs + current");
    }

    // compose_request with a single history pair builds a 3-message list.
    #[test]
    fn compose_request_single_history_pair_is_three_messages() {
        use crate::history::{History, Turn as HTurn};
        let mut h = History::new(4);
        h.push(HTurn { user: "prev user".to_string(), assistant: "prev asst".to_string(), ts: 0 });
        let msgs = h.to_messages();
        let req = compose_request("sonnet", "persona", &msgs, "current");
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].content, "prev user");
        assert_eq!(req.messages[1].role, Role::Assistant);
        assert_eq!(req.messages[1].content, "prev asst");
        assert_eq!(req.messages[2].role, Role::User);
        assert_eq!(req.messages[2].content, "current");
    }

    // DaemonState::new initialises history capacity from config.history_turns.
    #[test]
    fn daemon_state_new_uses_config_history_turns() {
        let cfg = BrainConfig { history_turns: 3, ..BrainConfig::default() };
        let state = DaemonState::new(cfg);
        // Synchronous lock check: history starts empty and has max_turns from config.
        let h = state.history.try_lock().expect("no contention");
        assert!(h.is_empty());
        // Push 4 turns and verify cap at 3.
        drop(h);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("rt");
        rt.block_on(async {
            use crate::history::Turn as HTurn;
            let mut h = state.history.lock().await;
            for i in 0u64..4 {
                h.push(HTurn {
                    user: format!("u{i}"),
                    assistant: format!("a{i}"),
                    ts: i,
                });
            }
            assert_eq!(h.len(), 3, "capped at history_turns=3");
        });
    }

    // --- ladder integration through dispatch() ----------------------------

    struct AlwaysAnswersLocal(String);

    #[async_trait::async_trait]
    impl crate::ladder::LocalBackend for AlwaysAnswersLocal {
        async fn generate(
            &self,
            _model: &str,
            _prompt: &wm_local_llm::Prompt,
            _sink: &dyn wm_local_llm::DeltaSink,
        ) -> wm_local_llm::LocalOutcome {
            wm_local_llm::LocalOutcome::Answer { text: self.0.clone() }
        }
    }

    struct FixedStakes(wm_router::Stakes);
    impl crate::ladder::StakesProvider for FixedStakes {
        fn stakes(&self, _t: &str) -> wm_router::Stakes {
            self.0
        }
    }

    struct RecallUp(bool);
    #[async_trait::async_trait]
    impl crate::ladder::RecallLiveness for RecallUp {
        async fn recall_up(&self) -> bool {
            self.0
        }
    }

    #[tokio::test]
    async fn dispatch_turn_user_through_ladder_publishes_local_reply() {
        // A configured ladder owns dispatch; an ordinary turn served locally
        // publishes a wm.brain.reply and never touches the Anthropic client.
        let ladder = Arc::new(crate::ladder::LadderClient::new(
            crate::default_ladder(),
            Arc::new(AlwaysAnswersLocal(
                "It's a calm, pleasant afternoon here.".to_string(),
            )),
            None, // no API key path
            Arc::new(FixedStakes(wm_router::Stakes::Ordinary)),
            Arc::new(RecallUp(true)),
        ));
        let state = Arc::new(DaemonState::new(BrainConfig::default()).with_ladder(ladder));
        let mut sink = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "tell me about your afternoon".to_string(),
                confidence: 1.0,
                ts: 7,
            }),
            7,
        )
        .await
        .expect("dispatch ok");
        let events = sink.non_session_events();
        // A ladder turn now publishes wm.brain.reply + wm.brain.route (PRD-brain-routing §2.5).
        let reply_events: Vec<_> = events
            .iter()
            .filter(|(t, _)| t == outgoing::REPLY)
            .collect();
        assert_eq!(reply_events.len(), 1, "exactly one reply published");
        assert_eq!(
            reply_events[0].1["text"],
            "It's a calm, pleasant afternoon here."
        );
        // A route event is also published.
        let route_events: Vec<_> = events
            .iter()
            .filter(|(t, _)| t == outgoing::ROUTE)
            .collect();
        assert_eq!(route_events.len(), 1, "exactly one route event published");
    }

    // ── almanac-acknowledge integration tests (PRD-almanac-acknowledge) ────────

    /// Build a state with a specific almanac patience window and a preset
    /// [`PendingAck`] whose `asked_ms` is `0`.
    fn state_with_pending_ack(patience_ms: u64, max_snoozes: u32, snooze_ms: u64) -> Arc<DaemonState> {
        let cfg = BrainConfig::default();
        let state = Arc::new(
            DaemonState::new(cfg).with_almanac_patience_ms(patience_ms),
        );
        let ack_cfg = crate::almanac::AlmanacEntryConfig { max_snoozes, snooze_ms };
        let ack = PendingAck::new("med-1", "medication", 0, ack_cfg);
        // Use block_on trick: in test context we drive futures inline.
        // We must synchronously initialise the pending_ack — use try_lock.
        *state.pending_ack.try_lock().expect("init ack") = Some(ack);
        state
    }

    // AC1: "I took it" → wm.almanac.ack {id, state:"done"}, pending cleared.
    #[tokio::test]
    async fn handle_stt_final_for_ack_done_emits_done_and_clears_pending() {
        let patience_ms = 60_000u64;
        let state = state_with_pending_ack(patience_ms, 2, 300_000);
        let mut sink = MemSink::default();

        let consumed = handle_stt_final_for_ack(state.as_ref(), &mut sink, "I took it", 1_000)
            .await
            .expect("handler ok");

        assert!(consumed, "done transcript should be consumed by ack path");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1, "exactly one ack event published");
        assert_eq!(events[0].0, crate::almanac::ALMANAC_ACK_TOPIC);
        assert_eq!(events[0].1["id"], "med-1");
        assert_eq!(events[0].1["state"], "done");
        assert_eq!(events[0].1["ts"], 1_000_u64);
        drop(events);
        let pending = state.pending_ack.lock().await;
        assert!(pending.is_none(), "pending ack must be cleared on done");
    }

    // AC2 (first branch): snooze below max → wm.almanac.ack snoozed + wm.almanac.snooze.
    #[tokio::test]
    async fn handle_stt_final_for_ack_snooze_below_max_emits_snoozed_and_snooze_event() {
        let patience_ms = 60_000u64;
        let snooze_ms = 5 * 60_000u64;
        let state = state_with_pending_ack(patience_ms, 2, snooze_ms);
        let mut sink = MemSink::default();

        let consumed = handle_stt_final_for_ack(state.as_ref(), &mut sink, "in a minute", 1_000)
            .await
            .expect("handler ok");

        assert!(consumed);
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2, "ack + snooze events");
        // First: ack snoozed
        assert_eq!(events[0].0, crate::almanac::ALMANAC_ACK_TOPIC);
        assert_eq!(events[0].1["state"], "snoozed");
        // Second: snooze resume_ts
        assert_eq!(events[1].0, crate::almanac::ALMANAC_SNOOZE_TOPIC);
        let expected_resume = 1_000u64 + snooze_ms;
        assert_eq!(events[1].1["resume_ts"], expected_resume);
        drop(events);
        // Pending ack still set (snooze_used incremented, window reset).
        let guard = state.pending_ack.lock().await;
        let updated = guard.as_ref().expect("pending must remain after snooze");
        assert_eq!(updated.snoozes_used, 1, "snooze counter incremented");
        assert_eq!(updated.asked_ms, 1_000, "window reset to now_ms");
    }

    // AC2 (second branch): snooze at max → wm.almanac.ack missed.
    #[tokio::test]
    async fn handle_stt_final_for_ack_snooze_at_max_emits_missed() {
        let patience_ms = 60_000u64;
        let state = state_with_pending_ack(patience_ms, 2, 300_000);
        // Exhaust snoozes.
        {
            let mut guard = state.pending_ack.lock().await;
            if let Some(ref mut ack) = *guard {
                ack.snoozes_used = 2;
            }
        }
        let mut sink = MemSink::default();

        let consumed = handle_stt_final_for_ack(state.as_ref(), &mut sink, "in a minute", 1_000)
            .await
            .expect("handler ok");

        assert!(consumed);
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1, "only a missed ack event");
        assert_eq!(events[0].0, crate::almanac::ALMANAC_ACK_TOPIC);
        assert_eq!(events[0].1["state"], "missed");
        drop(events);
        let guard = state.pending_ack.lock().await;
        assert!(guard.is_none(), "pending cleared on missed");
    }

    // AC3: unrelated transcript leaves pending open, returns false.
    #[tokio::test]
    async fn handle_stt_final_for_ack_unrelated_leaves_pending_open() {
        let state = state_with_pending_ack(60_000, 2, 300_000);
        let mut sink = MemSink::default();

        let consumed = handle_stt_final_for_ack(state.as_ref(), &mut sink, "what time is it", 1_000)
            .await
            .expect("handler ok");

        assert!(!consumed, "unrelated must not be consumed");
        assert!(sink.events.lock().unwrap().is_empty(), "no events published for unrelated");
        let guard = state.pending_ack.lock().await;
        assert!(guard.is_some(), "pending must remain open on unrelated");
    }

    // AC7: no pending ack → handler returns false, no events.
    #[tokio::test]
    async fn handle_stt_final_for_ack_no_pending_is_noop() {
        let state = fresh_state(); // no pending ack
        let mut sink = MemSink::default();

        let consumed = handle_stt_final_for_ack(state.as_ref(), &mut sink, "I took it", 1_000)
            .await
            .expect("handler ok");

        assert!(!consumed);
        assert!(sink.events.lock().unwrap().is_empty());
    }

    // Window expired → handler defers to timeout path, returns false.
    #[tokio::test]
    async fn handle_stt_final_for_ack_expired_window_returns_false() {
        let patience_ms = 30_000u64;
        let state = state_with_pending_ack(patience_ms, 2, 300_000);
        // asked_ms = 0, now = 31_001 → window expired.
        let mut sink = MemSink::default();

        let consumed = handle_stt_final_for_ack(state.as_ref(), &mut sink, "I took it", 31_001)
            .await
            .expect("handler ok");

        assert!(!consumed, "expired window must defer to timeout path");
        assert!(sink.events.lock().unwrap().is_empty(), "no events from ack handler when window expired");
    }

    // AC4: timeout first elapse → missed + re-ask reply, window reset.
    #[tokio::test]
    async fn tick_almanac_timeout_first_elapse_speaks_reask_and_emits_missed() {
        let patience_ms = 30_000u64;
        let state = state_with_pending_ack(patience_ms, 2, 300_000);
        // asked_ms = 0, now = 31_001 → window expired.
        let mut sink = MemSink::default();

        let fired = tick_almanac_timeout(state.as_ref(), &mut sink, 31_001)
            .await
            .expect("timeout ok");

        assert!(fired, "timeout should fire on first elapse");
        let events = sink.events.lock().unwrap();
        // Two events: the re-ask reply + the missed ack.
        assert_eq!(events.len(), 2, "re-ask reply + missed ack");
        assert_eq!(events[0].0, outgoing::REPLY, "re-ask via reply topic");
        assert!(events[0].1["text"].as_str().unwrap().contains("Did you take it"), "re-ask text");
        assert_eq!(events[1].0, crate::almanac::ALMANAC_ACK_TOPIC);
        assert_eq!(events[1].1["state"], "missed");
        drop(events);
        let guard = state.pending_ack.lock().await;
        let updated = guard.as_ref().expect("pending must remain after first elapse");
        assert!(updated.re_asked, "re_asked flag set");
        assert_eq!(updated.asked_ms, 31_001, "window reset to now_ms");
    }

    // AC4: timeout second elapse → missed, no re-ask, pending cleared.
    #[tokio::test]
    async fn tick_almanac_timeout_second_elapse_emits_final_missed_and_clears() {
        let patience_ms = 30_000u64;
        let state = state_with_pending_ack(patience_ms, 2, 300_000);
        // Set re_asked = true so we're on the second elapse.
        {
            let mut guard = state.pending_ack.lock().await;
            if let Some(ref mut ack) = *guard {
                ack.re_asked = true;
            }
        }
        let mut sink = MemSink::default();

        let fired = tick_almanac_timeout(state.as_ref(), &mut sink, 31_001)
            .await
            .expect("timeout ok");

        assert!(fired);
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1, "only final missed ack event, no re-ask");
        assert_eq!(events[0].0, crate::almanac::ALMANAC_ACK_TOPIC);
        assert_eq!(events[0].1["state"], "missed");
        drop(events);
        let guard = state.pending_ack.lock().await;
        assert!(guard.is_none(), "pending cleared on second elapse");
    }

    // handle_almanac_due sets the pending ack correctly.
    #[tokio::test]
    async fn handle_almanac_due_sets_pending_ack() {
        let state = fresh_state();
        let cfg = crate::almanac::AlmanacEntryConfig { max_snoozes: 1, snooze_ms: 60_000 };
        handle_almanac_due(state.as_ref(), "rem-99", "water", cfg.clone(), 5_000).await;

        let guard = state.pending_ack.lock().await;
        let ack = guard.as_ref().expect("pending should be set");
        assert_eq!(ack.id, "rem-99");
        assert_eq!(ack.category, "water");
        assert_eq!(ack.asked_ms, 5_000);
        assert_eq!(ack.snoozes_used, 0);
        assert!(!ack.re_asked);
        assert_eq!(ack.config, cfg);
    }

    // AC5: patience window is sourced externally — verify with_almanac_patience_ms builder.
    #[test]
    fn daemon_state_with_almanac_patience_ms_overrides_default() {
        let cfg = BrainConfig::default();
        let state = DaemonState::new(cfg).with_almanac_patience_ms(90_000);
        assert_eq!(state.almanac_patience_ms, 90_000);
        assert_ne!(state.almanac_patience_ms, DEFAULT_ALMANAC_PATIENCE_MS);
    }

    // ── speak-bridge tests (PRD-almanac-speak-bridge) ─────────────────────────

    fn make_due_event(id: &str, say: &str) -> crate::almanac::AlmanacDueEvent {
        crate::almanac::AlmanacDueEvent {
            id: id.to_string(),
            label: "test-label".to_string(),
            say: say.to_string(),
            category: "medication".to_string(),
            fire_ts: 1_000,
        }
    }

    // AC1: wm.almanac.due with say="time for your blue pill" causes exactly one
    // wm.brain.reply with text = ev.say (verbatim), via handle_speak_almanac_due.
    #[tokio::test]
    async fn speak_bridge_publishes_reply_with_verbatim_say() {
        let state = fresh_state(); // almanac_speak defaults to true
        let mut sink = MemSink::default();
        let ev = make_due_event("rem-1", "time for your blue pill");

        let spoke = handle_speak_almanac_due(state.as_ref(), &mut sink, &ev, 5_000)
            .await
            .expect("speak ok");
        assert!(spoke, "should return true when speak fires");

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1, "exactly one reply published");
        assert_eq!(events[0].0, outgoing::REPLY, "published on reply topic");
        assert_eq!(
            events[0].1["text"],
            "time for your blue pill",
            "text is verbatim from ev.say"
        );
        assert_eq!(events[0].1["ts"], 5_000_u64);
    }

    // AC2: after speak, the pending ack is armed via handle_almanac_due.
    #[tokio::test]
    async fn speak_bridge_arms_pending_ack_after_speak() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let ev = make_due_event("rem-arm", "drink water");

        handle_speak_almanac_due(state.as_ref(), &mut sink, &ev, 9_000)
            .await
            .expect("speak ok");

        let guard = state.pending_ack.lock().await;
        let ack = guard.as_ref().expect("pending ack must be set after speak");
        assert_eq!(ack.id, "rem-arm");
        assert_eq!(ack.category, "medication");
        assert_eq!(ack.asked_ms, 9_000, "asked_ms matches now_ms");
        assert_eq!(ack.snoozes_used, 0);
        assert!(!ack.re_asked);
    }

    // AC3: with almanac_speak=false, wm.almanac.due publishes NO reply.
    #[tokio::test]
    async fn speak_bridge_disabled_publishes_nothing() {
        let cfg = BrainConfig {
            almanac_speak: false,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg));
        let mut sink = MemSink::default();
        let ev = make_due_event("rem-off", "time for your blue pill");

        let spoke = handle_speak_almanac_due(state.as_ref(), &mut sink, &ev, 1_234)
            .await
            .expect("call ok even when disabled");
        assert!(!spoke, "almanac_speak=false -> returns false");

        let events = sink.events.lock().unwrap();
        assert!(events.is_empty(), "no reply published when gate is off");
    }

    // AC4: malformed envelope (empty say) logs WARN and publishes nothing.
    #[tokio::test]
    async fn speak_bridge_empty_say_logs_warn_and_publishes_nothing() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let ev = make_due_event("rem-bad", ""); // empty say

        let spoke = handle_speak_almanac_due(state.as_ref(), &mut sink, &ev, 2_000)
            .await
            .expect("call must not Err on malformed envelope");
        assert!(!spoke, "empty say -> returns false, no panic");

        let events = sink.events.lock().unwrap();
        assert!(events.is_empty(), "no reply published for empty say");
    }

    // AC4 (whitespace-only say is also malformed).
    #[tokio::test]
    async fn speak_bridge_whitespace_say_publishes_nothing() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let ev = make_due_event("rem-ws", "   \t  ");

        let spoke = handle_speak_almanac_due(state.as_ref(), &mut sink, &ev, 3_000)
            .await
            .expect("call must not Err");
        assert!(!spoke);
        assert!(sink.events.lock().unwrap().is_empty());
    }

    // AC6: no persona string / phrase bank embedded in the speak-bridge diff.
    // Tested structurally: handle_speak_almanac_due publishes ev.say verbatim
    // without appending or prepending any persona text.
    #[tokio::test]
    async fn speak_bridge_publishes_say_verbatim_no_persona_wrapping() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let ev = make_due_event("rem-v", "this is the exact say field");

        handle_speak_almanac_due(state.as_ref(), &mut sink, &ev, 7_777)
            .await
            .expect("speak ok");

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        // The published text is EXACTLY ev.say — no appended persona clause.
        assert_eq!(events[0].1["text"], "this is the exact say field");
    }

    // --- Session-boundary integration (PRD-wmd-session-boundary) ---------------

    // AC1 (dispatch integration): Two turns 6 minutes apart produce
    // SESSION_START → SESSION_END(reason=idle) → SESSION_START.
    // The second turn's history ring is empty (AC§2.4).
    #[tokio::test]
    async fn session_ac1_gap_emits_end_then_start_and_clears_history() {
        let cfg = BrainConfig {
            idle_gap_ms: 300_000, // 5 min
            history_turns: 6,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg));

        // Turn 1 at t=0 — should open session 1.
        let mut sink = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "first turn".to_string(),
                confidence: 1.0,
                ts: 0,
            }),
            0,
        )
        .await
        .expect("first turn ok");
        let sess_events = sink.session_events();
        assert_eq!(sess_events.len(), 1, "Turn 1 opens a session");
        assert_eq!(sess_events[0].0, outgoing::SESSION_START);
        let sess1_id = sess_events[0].1["session_id"]
            .as_str()
            .expect("session_id")
            .to_string();

        // Turn 2 at t=6min — gap exceeded → SESSION_END + SESSION_START.
        let six_min_ms = 6 * 60_000_u64;
        let mut sink2 = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink2,
            Request::TurnUser(TurnUserEvent {
                transcript: "second turn after gap".to_string(),
                confidence: 1.0,
                ts: six_min_ms,
            }),
            six_min_ms,
        )
        .await
        .expect("second turn ok");
        let sess_events2 = sink2.session_events();
        // Should have SESSION_END (close old) + SESSION_START (open new).
        assert_eq!(sess_events2.len(), 2, "gap triggers end+start");
        assert_eq!(sess_events2[0].0, outgoing::SESSION_END, "end first");
        assert_eq!(sess_events2[0].1["session_id"], sess1_id, "end closes sess1");
        assert_eq!(sess_events2[0].1["reason"], "idle");
        assert_eq!(sess_events2[1].0, outgoing::SESSION_START, "then start");
        let sess2_id = sess_events2[1].1["session_id"]
            .as_str()
            .expect("session2_id")
            .to_string();
        assert_ne!(sess2_id, sess1_id, "new session has different id");

        // History must be empty at the start of session 2 (no LLM so no
        // turns were actually pushed, but verify the ring was reset).
        let history = state.history.lock().await;
        assert!(history.is_empty(), "history cleared on session boundary");
    }

    // AC2 (dispatch integration): Three turns 1 minute apart share one session.
    #[tokio::test]
    async fn session_ac2_turns_within_gap_share_one_session() {
        let cfg = BrainConfig {
            idle_gap_ms: 300_000,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg));
        let min_ms: u64 = 60_000;

        for ts in [0_u64, min_ms, 2 * min_ms] {
            let mut sink = MemSink::default();
            dispatch(
                state.as_ref(),
                &mut sink,
                Request::TurnUser(TurnUserEvent {
                    transcript: "a turn".to_string(),
                    confidence: 1.0,
                    ts,
                }),
                ts,
            )
            .await
            .expect("turn ok");
            if ts == 0 {
                // First turn: start event only.
                let sess = sink.session_events();
                assert_eq!(sess.len(), 1);
                assert_eq!(sess[0].0, outgoing::SESSION_START);
            } else {
                // Subsequent turns within gap: no session events.
                let sess = sink.session_events();
                assert!(sess.is_empty(), "no session event for turn within gap");
            }
        }

        // Verify session is still open with 3 turns.
        let tracker = state.session_tracker.lock().await;
        assert_eq!(tracker.current_turn_count(), 3);
    }

    // AC3 (dispatch integration): "goodbye" phrase closes the session after
    // the reply (reason=explicit). A subsequent turn opens a new session.
    #[tokio::test]
    async fn session_ac3_explicit_close_phrase_closes_after_reply() {
        let cfg = BrainConfig {
            idle_gap_ms: 300_000,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg));

        // Turn 1: ordinary turn (opens session).
        let mut sink1 = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink1,
            Request::TurnUser(TurnUserEvent {
                transcript: "hello".to_string(),
                confidence: 1.0,
                ts: 1000,
            }),
            1000,
        )
        .await
        .expect("turn1 ok");
        assert_eq!(sink1.session_events().len(), 1, "turn 1 opens session");
        let sess1_id = sink1.session_events()[0].1["session_id"]
            .as_str()
            .unwrap()
            .to_string();

        // Turn 2: "goodbye" — should close the session after the reply.
        // (No LLM configured so reply isn't published, but the session close fires.)
        let mut sink2 = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink2,
            Request::TurnUser(TurnUserEvent {
                transcript: "goodbye".to_string(),
                confidence: 1.0,
                ts: 2000,
            }),
            2000,
        )
        .await
        .expect("goodbye turn ok");
        let sess2 = sink2.session_events();
        // No new session opened (still in the same session).
        // SESSION_END(reason=explicit) after the reply.
        assert_eq!(sess2.len(), 1, "goodbye closes the session");
        assert_eq!(sess2[0].0, outgoing::SESSION_END);
        assert_eq!(sess2[0].1["session_id"], sess1_id);
        assert_eq!(sess2[0].1["reason"], "explicit");

        // Turn 3 should open a fresh session.
        let mut sink3 = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink3,
            Request::TurnUser(TurnUserEvent {
                transcript: "good morning".to_string(),
                confidence: 1.0,
                ts: 3000,
            }),
            3000,
        )
        .await
        .expect("turn3 ok");
        let sess3 = sink3.session_events();
        assert_eq!(sess3.len(), 1, "turn 3 opens a new session");
        assert_eq!(sess3[0].0, outgoing::SESSION_START);
        let sess3_id = sess3[0].1["session_id"].as_str().unwrap().to_string();
        assert_ne!(sess3_id, sess1_id, "new session has different id");
    }

    // AC7 (self-emit filter): SESSION_TOPIC_PREFIX is defined and points to
    // the session topic namespace. Tests that the constant is well-formed.
    #[test]
    fn session_ac7_session_topic_prefix_is_correct() {
        assert_eq!(bus::SESSION_TOPIC_PREFIX, "wm.brain.session.");
        assert!(outgoing::SESSION_START.starts_with(bus::SESSION_TOPIC_PREFIX));
        assert!(outgoing::SESSION_END.starts_with(bus::SESSION_TOPIC_PREFIX));
    }

    // ── Session recap tests (PRD-wmd-session-recap) ─────────────────────────

    /// Build a QueryHit with the thread subject for testing recap.
    fn thread_hit(id: &str, snippet: &str) -> QueryHit {
        QueryHit {
            id: id.to_string(),
            kind: "episodic".to_string(),
            subject: format!("{}{}", THREAD_SUBJECT_PREFIX, "2026-05-28"),
            path: format!("/tmp/{id}.md"),
            snippet: snippet.to_string(),
            score: 0.9,
            confidence: 0.9,
        }
    }

    // format_recap_context tests ─────────────────────────────────────────────

    // AC2 — cold store is a no-op: empty hits → None.
    #[test]
    fn format_recap_context_empty_returns_none() {
        assert!(format_recap_context(&[]).is_none());
    }

    // AC1 — thread memories render under "Recent conversations:" label.
    #[test]
    fn format_recap_context_uses_distinct_label() {
        let hits = vec![thread_hit("t1", "User was worried about the appointment.")];
        let out = format_recap_context(&hits).expect("non-empty");
        assert!(
            out.contains("Recent conversations"),
            "distinct label for model to distinguish"
        );
        assert!(out.contains("User was worried about the appointment."));
        assert!(!out.contains("What you remember"), "not the profile label");
    }

    // AC4 — bound respected: only N most recent hits returned.
    #[test]
    fn format_recap_context_renders_at_most_n_snippets() {
        let hits = vec![
            thread_hit("t1", "First memory."),
            thread_hit("t2", "Second memory."),
            thread_hit("t3", "Third memory."),
        ];
        // format_recap_context itself doesn't bound — bounding happens upstream.
        // But with 3 hits it renders all 3.
        let out = format_recap_context(&hits).expect("non-empty");
        assert!(out.contains("First memory."));
        assert!(out.contains("Second memory."));
        assert!(out.contains("Third memory."));
    }

    #[test]
    fn format_recap_context_skips_blank_snippets() {
        let hits = vec![
            thread_hit("t1", "Real memory."),
            thread_hit("t2", "   "), // blank
        ];
        let out = format_recap_context(&hits).expect("non-empty because t1 has content");
        assert!(out.contains("Real memory."));
        // blank snippet is not rendered as an empty numbered item
        assert!(!out.contains("2."), "blank snippet not rendered");
    }

    // handle_session_start unit tests ────────────────────────────────────────

    // AC2 — cold store: no thread memories → recap_context stays None.
    #[tokio::test]
    async fn handle_session_start_cold_store_no_context() {
        // FakeRecall with only profile hits (no thread subject) — simulates cold store.
        let recall = Arc::new(FakeRecall::new(vec![
            fake_hit("p1", "profile fact", 0.9),
        ]));
        let cfg = BrainConfig::default();
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(recall),
        );
        let mut sink = MemSink::default();
        handle_session_start(state.as_ref(), &mut sink, 1000).await;
        // No thread-subject hits → recap_context remains None.
        let ctx = state.recap_context.lock().await.clone();
        assert!(ctx.is_none(), "cold store leaves recap_context None");
        // No opener published.
        assert!(sink.events.lock().unwrap().is_empty());
    }

    // AC1 — recap injects recent thread context.
    #[tokio::test]
    async fn handle_session_start_injects_thread_context() {
        let recall = Arc::new(FakeRecall::new(vec![
            thread_hit("t1", "Yesterday you were worried about the appointment."),
        ]));
        let cfg = BrainConfig {
            recap_max_memories: 5,
            recap_opener: false,
            ..BrainConfig::default()
        };
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(recall),
        );
        let mut sink = MemSink::default();
        handle_session_start(state.as_ref(), &mut sink, 1000).await;
        let ctx = state.recap_context.lock().await.clone();
        assert!(ctx.is_some(), "thread hit → recap_context set");
        let ctx_str = ctx.unwrap();
        assert!(ctx_str.contains("Recent conversations"), "correct label");
        assert!(ctx_str.contains("Yesterday you were worried about the appointment."));
    }

    // AC3 — recap is session-scoped: thread query fires once per session.
    #[tokio::test]
    async fn handle_session_start_fires_thread_query_once() {
        let recall = Arc::new(FakeRecall::new(vec![
            thread_hit("t1", "Some thread memory."),
        ]));
        let cfg = BrainConfig::default();
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(recall),
        );
        let mut sink = MemSink::default();
        // Call handle_session_start once (one session.start).
        handle_session_start(state.as_ref(), &mut sink, 1000).await;
        let count = state.session_thread_query_count.load(Ordering::SeqCst);
        assert_eq!(count, 1, "exactly one thread query per session.start");
    }

    // AC3 continued — per-turn recall query is separate (not the thread query).
    // Verify that multiple turns don't increment session_thread_query_count.
    #[tokio::test]
    async fn per_turn_recall_does_not_increment_thread_query_count() {
        let recall = Arc::new(FakeRecall::new(vec![
            thread_hit("t1", "Some thread memory."),
        ]));
        let cfg = BrainConfig::default();
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(Arc::clone(&recall) as Arc<dyn RecallSource>),
        );
        // Start session (fires one thread query).
        let mut sink = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "first turn".to_string(),
                confidence: 1.0,
                ts: 0,
            }),
            0,
        )
        .await
        .expect("dispatch ok");
        // Session opened: exactly one thread query.
        let count_after_session = state.session_thread_query_count.load(Ordering::SeqCst);
        assert_eq!(count_after_session, 1);
        // Another turn within the same session — no new thread query.
        let mut sink2 = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink2,
            Request::TurnUser(TurnUserEvent {
                transcript: "second turn within session".to_string(),
                confidence: 1.0,
                ts: 1000,
            }),
            1000,
        )
        .await
        .expect("dispatch ok");
        let count_after_second = state.session_thread_query_count.load(Ordering::SeqCst);
        assert_eq!(count_after_second, 1, "per-turn recall does not fire thread query");
    }

    // AC4 — bound respected: recap_max_memories limits injected memories.
    #[tokio::test]
    async fn handle_session_start_respects_max_memories_bound() {
        let hits = vec![
            thread_hit("t1", "Memory one."),
            thread_hit("t2", "Memory two."),
            thread_hit("t3", "Memory three."),
            thread_hit("t4", "Memory four."),
            thread_hit("t5", "Memory five."),
        ];
        let recall = Arc::new(FakeRecall::new(hits));
        let cfg = BrainConfig {
            recap_max_memories: 2,
            ..BrainConfig::default()
        };
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(recall),
        );
        let mut sink = MemSink::default();
        handle_session_start(state.as_ref(), &mut sink, 1000).await;
        let ctx = state.recap_context.lock().await.clone();
        let ctx_str = ctx.expect("context set");
        // The search call is bounded by recap_max_memories=2.
        // With FakeRecall returning all 5 but limit=2 passed to search,
        // we verify the context contains at most 2 numbered items.
        let count_items = ctx_str.matches("Memory").count();
        assert!(
            count_items <= 2,
            "at most recap_max_memories items injected; got {count_items}"
        );
    }

    // AC5 — proposals are not read: only thread-subject memories are surfaced.
    // We simulate this by including both thread-subject and non-thread hits.
    // The recap handler filters by subject prefix.
    #[tokio::test]
    async fn handle_session_start_filters_to_thread_subject_only() {
        let hits = vec![
            // Thread hit (should be included).
            thread_hit("t1", "Thread memory."),
            // Profile hit (not under thread prefix — should be excluded).
            fake_hit("p1", "Profile fact.", 0.9),
        ];
        let recall = Arc::new(FakeRecall::new(hits));
        let cfg = BrainConfig::default();
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(recall),
        );
        let mut sink = MemSink::default();
        handle_session_start(state.as_ref(), &mut sink, 1000).await;
        let ctx = state.recap_context.lock().await.clone();
        let ctx_str = ctx.expect("context set from thread hit");
        assert!(ctx_str.contains("Thread memory."), "thread hit included");
        assert!(!ctx_str.contains("Profile fact."), "profile hit excluded");
    }

    // AC6 — opener off by default: recap_opener=false → no wm.brain.reply before first turn.
    #[tokio::test]
    async fn handle_session_start_opener_off_by_default() {
        let recall = Arc::new(FakeRecall::new(vec![
            thread_hit("t1", "Some thread memory."),
        ]));
        let cfg = BrainConfig {
            recap_opener: false,
            ..BrainConfig::default()
        };
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(recall),
        );
        let mut sink = MemSink::default();
        handle_session_start(state.as_ref(), &mut sink, 1000).await;
        // Context was set but no reply published.
        assert!(state.recap_context.lock().await.is_some(), "context set");
        let events = sink.events.lock().unwrap();
        assert!(events.is_empty(), "no opener published when recap_opener=false (AC6)");
    }

    // AC7 — opener on publishes a continuity greeting.
    #[tokio::test]
    async fn handle_session_start_opener_on_publishes_reply() {
        let recall = Arc::new(FakeRecall::new(vec![
            thread_hit("t1", "you mentioned your daughter visits Sunday"),
        ]));
        let cfg = BrainConfig {
            recap_opener: true,
            ..BrainConfig::default()
        };
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(recall),
        );
        let mut sink = MemSink::default();
        handle_session_start(state.as_ref(), &mut sink, 1000).await;
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1, "exactly one reply published as opener");
        assert_eq!(events[0].0, outgoing::REPLY, "published on reply topic");
        let text = events[0].1["text"].as_str().expect("text field");
        assert!(
            text.contains("you mentioned your daughter visits Sunday"),
            "opener text references thread snippet: {text}"
        );
    }

    // AC8 — recall outage tolerated: query failure → session proceeds with no recap.
    #[tokio::test]
    async fn handle_session_start_outage_leaves_no_recap() {
        let recall = Arc::new(FailingRecall);
        let cfg = BrainConfig::default();
        let state = Arc::new(
            DaemonState::new(cfg).with_recall(recall),
        );
        let mut sink = MemSink::default();
        // Must not panic.
        handle_session_start(state.as_ref(), &mut sink, 1000).await;
        let ctx = state.recap_context.lock().await.clone();
        assert!(ctx.is_none(), "recall outage → recap_context None (AC8)");
        // No events published.
        assert!(sink.events.lock().unwrap().is_empty());
    }

    // recap_context spliced into persona (AC1 integration).
    #[test]
    fn compose_persona_splices_recap_context_after_recall_context() {
        let recall_ctx = "What you remember about the user (most relevant first):\n1. Drinks tea.";
        let recap_ctx = "Recent conversations (most recent first):\n1. Discussed appointment.";
        let out = compose_persona("base", false, Some(recall_ctx), Some(recap_ctx));
        let recall_idx = out.find("What you remember").expect("recall present");
        let recap_idx = out.find("Recent conversations").expect("recap present");
        assert!(
            recap_idx > recall_idx,
            "recap section follows recall context"
        );
        assert!(out.contains("Discussed appointment."));
    }

    // recap_context cleared on new session: confirm dispatch resets it when session opens.
    // Simulate: state has recap_context set from a prior session; new session.start fires;
    // handle_session_start re-queries (NullRecall → no hits → None again).
    #[tokio::test]
    async fn dispatch_clears_recap_context_on_new_session() {
        let cfg = BrainConfig {
            idle_gap_ms: 300_000,
            ..BrainConfig::default()
        };
        let state = Arc::new(DaemonState::new(cfg));
        // Manually set recap_context to simulate a prior session.
        *state.recap_context.lock().await = Some("stale recap".to_string());

        // Fire a turn that opens a new session (first turn ever).
        let mut sink = MemSink::default();
        dispatch(
            state.as_ref(),
            &mut sink,
            Request::TurnUser(TurnUserEvent {
                transcript: "hello".to_string(),
                confidence: 1.0,
                ts: 0,
            }),
            0,
        )
        .await
        .expect("dispatch ok");
        // NullRecall returns no hits → recap_context should be None after the new session.
        let ctx = state.recap_context.lock().await.clone();
        assert!(
            ctx.is_none(),
            "new session with NullRecall clears stale recap_context"
        );
    }
}
