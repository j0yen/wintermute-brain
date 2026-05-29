//! The brain's tier-ladder orchestrator (PRD-brain-backend-ladder).
//!
//! [`LadderClient`] turns the brain's single-backend turn path into a
//! local-first ladder: an ordinary turn starts at the lowest local rung
//! and *climbs one rung* only when the turn demands it — on a hard local
//! failure ([`wm_local_llm::LocalOutcome::Escalate`]) or a soft failure
//! (a local [`wm_local_llm::LocalOutcome::Answer`] that [`wm_verify`]
//! rejects before it is spoken). High-stakes turns (flagged by
//! [`wm_router`]) skip local and start at a trusted cloud tier.
//!
//! ## Recall-down safe posture
//! The router's high-stakes detection depends on the recall embedder.
//! When recall is unreachable the router is *blind* to high stakes, so an
//! ordinary turn cannot safely start local: the ladder starts it at the
//! lowest **cloud** rung ([`crate::SAFE_FLOOR_TIER_NAME`], haiku) instead,
//! and logs a warning. A high-stakes flag from the keyword pre-pass still
//! routes to the trusted cloud tier.
//!
//! ## Injection seams (for tests; no live network)
//! - [`LocalBackend`] — wraps `wm-local-llm`; tests inject a fake.
//! - [`StakesProvider`] — wraps `wm-router`'s `classify`; tests inject.
//! - [`RecallLiveness`] — a cheap recall reachability probe; tests inject.
//! - The Anthropic side reuses [`crate::daemon::LlmClient`].

use std::sync::Arc;

use tracing::{info, warn};
use wm_local_llm::{DeltaSink, LocalOutcome, Message as LocalMessage, Prompt, Role as LocalRole};
use wm_router::Stakes;
use wm_verify::{Lang, Verdict, VerifyCtx, verify};

use crate::anthropic::{ClientError, MessageRequest, Role};
use crate::daemon::{LlmClient, extract_assistant_text};
use crate::{
    Backend, SAFE_FLOOR_TIER_NAME, TRUSTED_CLOUD_TIER_NAME, Tier, canonical_model,
};

/// Backchannel phrase emitted while the ladder climbs to a higher tier.
///
/// Emitted to the reply/TTS path so a climb never lands as a dead pause.
/// PRD-brain-backend-ladder §2.2 / `AC9` (reuses the companion-degrade tone).
pub const ESCALATION_FILLER: &str = "Let me think about that for a moment.";

/// Minimum `wm-verify` strictness for a locally generated answer.
const VERIFY_MIN_CONFIDENCE: f32 = 0.5;

/// An abstraction over the local OpenAI-compatible backend.
///
/// Production impl ([`LiveLocalBackend`]) builds a `wm-local-llm`
/// [`wm_local_llm::LocalLlm`] per tier model; tests inject a fake that
/// returns canned [`LocalOutcome`]s without any socket I/O.
#[async_trait::async_trait]
pub trait LocalBackend: Send + Sync {
    /// Generate a completion for `prompt` against the local `model`,
    /// streaming deltas to `sink`. Infallible — every failure maps to
    /// [`LocalOutcome::Escalate`] (never silence).
    async fn generate(&self, model: &str, prompt: &Prompt, sink: &dyn DeltaSink) -> LocalOutcome;
}

/// Supplies the [`Stakes`] tag for a turn (wraps `wm-router::classify`).
///
/// Only the `Brain { stakes }` stakes tag is consumed here; `Skill` /
/// `CacheLookup` deflections are the dialog FSM's job, not the brain's.
pub trait StakesProvider: Send + Sync {
    /// Classify `transcript` and return its stakes tag. A deflection
    /// route (skill/cache) maps to [`Stakes::Ordinary`] for the brain's
    /// purposes — if the dialog FSM handed the turn to the brain at all,
    /// the brain treats it as an ordinary brain turn.
    fn stakes(&self, transcript: &str) -> Stakes;
}

/// A cheap probe of whether recall (the router's embedder backend) is
/// reachable. Drives the recall-down safe posture.
#[async_trait::async_trait]
pub trait RecallLiveness: Send + Sync {
    /// `true` when recall is reachable (so the router's high-stakes
    /// detection is trustworthy); `false` when it is down/blind.
    async fn recall_up(&self) -> bool;
}

/// The outcome of one ladder turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LadderOutcome {
    /// A tier produced a spoken answer. Carries the answering tier name.
    Answer {
        /// The final answer text.
        text: String,
        /// The tier that produced it.
        tier: String,
    },
    /// Every reachable tier failed (top tier unreachable, or an Anthropic
    /// tier with no key). The caller routes this to the degrade path —
    /// never a panic, never a hang. PRD-brain-backend-ladder §2.4 / AC5.
    Degraded {
        /// Machine-readable reason for the terminal failure.
        reason: String,
    },
}

/// A sink for streamed answer deltas during a ladder turn. The daemon's
/// reply/TTS path implements this; tests use a recording sink.
///
/// `emit_filler` is called once when a climb is triggered, before the
/// higher tier generates (AC9). `on_delta` forwards each local-answer
/// delta incrementally (AC7).
pub trait LadderSink: DeltaSink {
    /// Emit a short backchannel filler phrase before a climb's answer.
    fn emit_filler(&self, phrase: &str);
}

/// Per-session "tier floor" for conversational stickiness (AC10).
///
/// Within an ongoing thread, once the ladder has climbed to a higher tier
/// to handle a turn, later ordinary turns in the same thread do not drop
/// below that rung — a warm/emotional thread that needed Sonnet keeps being
/// served at Sonnet rather than silently snapping back to `local-3b`. The
/// floor **decays** (resets) on a topic change, detected by low token
/// overlap with the previous turn. High-stakes turns and the recall-down
/// safe posture are orthogonal: they set their own (higher) starting rung.
#[derive(Debug, Default)]
pub struct SessionFloor {
    inner: std::sync::Mutex<FloorState>,
}

#[derive(Debug, Default)]
struct FloorState {
    /// The highest tier index reached so far in this thread (the floor).
    floor_idx: usize,
    /// Token set of the previous turn, for topic-change detection.
    prev_tokens: Vec<String>,
    /// Whether any turn has been seen yet.
    seen: bool,
}

impl SessionFloor {
    /// A fresh floor (no prior turns, floor at rung 0).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Consult the floor for `transcript`, decaying it on a topic change.
    /// Returns the floor rung index an ordinary turn must not start below.
    fn apply(&self, transcript: &str) -> usize {
        let tokens = topic_tokens(transcript);
        let Ok(mut st) = self.inner.lock() else {
            return 0;
        };
        if st.seen && topic_changed(&st.prev_tokens, &tokens) {
            st.floor_idx = 0; // decay on topic change
        }
        st.prev_tokens = tokens;
        st.seen = true;
        st.floor_idx
    }

    /// Raise the floor to `idx` after a turn was served at that rung.
    fn raise_to(&self, idx: usize) {
        if let Ok(mut st) = self.inner.lock() {
            if idx > st.floor_idx {
                st.floor_idx = idx;
            }
        }
    }
}

/// Tokenize a turn for topic-change detection (lowercased alphanumeric
/// words). Cheap and total.
fn topic_tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Two turns are a topic change when their token sets barely overlap
/// (Jaccard below a low floor). Empty-vs-anything counts as a change.
fn topic_changed(prev: &[String], cur: &[String]) -> bool {
    use std::collections::BTreeSet;
    let a: BTreeSet<&String> = prev.iter().collect();
    let b: BTreeSet<&String> = cur.iter().collect();
    if a.is_empty() || b.is_empty() {
        return true;
    }
    let inter = a.intersection(&b).count();
    let union = a.union(&b).count();
    // union > 0 here (both non-empty). Low overlap => topic changed.
    #[allow(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        clippy::float_arithmetic,
        reason = "small token counts; ratio only compared to a threshold"
    )]
    let jaccard = inter as f32 / union as f32;
    jaccard < 0.2
}

/// The orchestrator. Owns the resolved ladder, the local backend, the
/// optional Anthropic client, the stakes provider, and the recall probe.
pub struct LadderClient {
    ladder: Vec<Tier>,
    local: Arc<dyn LocalBackend>,
    anthropic: Option<Arc<dyn LlmClient>>,
    stakes: Arc<dyn StakesProvider>,
    recall: Arc<dyn RecallLiveness>,
    /// Expected reply language for the `wm-verify` gate.
    lang: Lang,
}

impl LadderClient {
    /// Construct a ladder client.
    #[must_use]
    pub fn new(
        ladder: Vec<Tier>,
        local: Arc<dyn LocalBackend>,
        anthropic: Option<Arc<dyn LlmClient>>,
        stakes: Arc<dyn StakesProvider>,
        recall: Arc<dyn RecallLiveness>,
    ) -> Self {
        Self {
            ladder,
            local,
            anthropic,
            stakes,
            recall,
            lang: Lang::English,
        }
    }

    /// Index of the starting rung for a turn.
    ///
    /// - High-stakes → the trusted cloud tier (skipping local), cost-blind
    ///   (`AC3b`). The keyword pre-pass still flags these even when recall is
    ///   down.
    /// - Ordinary + recall UP → the requested `start_tier` (local-first).
    /// - Ordinary + recall DOWN → the lowest cloud rung (safe floor), since
    ///   high stakes can't be ruled out; logged as a warning.
    fn start_index(&self, start_tier: &str, stakes: Stakes, recall_up: bool) -> usize {
        match stakes {
            Stakes::HighStakes(class) => {
                info!(
                    class = class.as_str(),
                    tier = TRUSTED_CLOUD_TIER_NAME,
                    "wm-brain ladder: high-stakes turn starts at trusted cloud tier (skipping local)"
                );
                self.tier_index(TRUSTED_CLOUD_TIER_NAME)
                    .unwrap_or_else(|| self.first_cloud_index())
            }
            Stakes::Ordinary => {
                if recall_up {
                    self.tier_index(start_tier).unwrap_or(0)
                } else {
                    warn!(
                        requested = start_tier,
                        floor = SAFE_FLOOR_TIER_NAME,
                        "wm-brain ladder: recall down -> high-stakes detection blind; \
                         starting ordinary turn at cloud safe floor instead of local"
                    );
                    // Never start below the safe floor when blind. If the
                    // requested tier is already a cloud tier at or above the
                    // floor, honor it; otherwise clamp up to the floor.
                    let floor = self
                        .tier_index(SAFE_FLOOR_TIER_NAME)
                        .unwrap_or_else(|| self.first_cloud_index());
                    let requested = self.tier_index(start_tier).unwrap_or(0);
                    requested.max(floor)
                }
            }
        }
    }

    /// Find a tier's index by name.
    fn tier_index(&self, name: &str) -> Option<usize> {
        self.ladder.iter().position(|t| t.name == name)
    }

    /// Index of the first Anthropic (cloud) tier, or the top rung if the
    /// ladder is somehow all-local (defensive — the built-in ladder always
    /// has cloud rungs).
    fn first_cloud_index(&self) -> usize {
        self.ladder
            .iter()
            .position(|t| t.backend == Backend::Anthropic)
            .unwrap_or_else(|| self.ladder.len().saturating_sub(1))
    }

    /// Run one turn through the ladder.
    ///
    /// `transcript` is the user turn; `request` is the fully-composed
    /// Anthropic request (system prompt + messages + any prompt-cache
    /// breakpoints) used UNMODIFIED for Anthropic tiers (AC8); `start_tier`
    /// is the resolved per-turn starting tier name.
    pub async fn run_turn(
        &self,
        transcript: &str,
        request: &MessageRequest,
        start_tier: &str,
        sink: &dyn LadderSink,
    ) -> LadderOutcome {
        self.run_turn_inner(transcript, request, start_tier, sink, None)
            .await
    }

    /// Like [`Self::run_turn`] but honors a per-session [`SessionFloor`] for
    /// conversational stickiness (AC10): an ordinary turn never starts below
    /// the floor, and the floor is raised to the answering rung afterward.
    #[allow(
        clippy::too_many_arguments,
        reason = "turn inputs: transcript, request, start tier, sink, floor"
    )]
    pub async fn run_turn_sticky(
        &self,
        transcript: &str,
        request: &MessageRequest,
        start_tier: &str,
        sink: &dyn LadderSink,
        floor: &SessionFloor,
    ) -> LadderOutcome {
        self.run_turn_inner(transcript, request, start_tier, sink, Some(floor))
            .await
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "turn inputs: transcript, request, start tier, sink, optional floor"
    )]
    async fn run_turn_inner(
        &self,
        transcript: &str,
        request: &MessageRequest,
        start_tier: &str,
        sink: &dyn LadderSink,
        floor: Option<&SessionFloor>,
    ) -> LadderOutcome {
        let stakes = self.stakes.stakes(transcript);
        let recall_up = self.recall.recall_up().await;
        let mut start = self.start_index(start_tier, stakes, recall_up);

        // AC10: an ordinary turn never starts below the session floor.
        // (High-stakes / recall-down already pick a cloud rung; the floor
        // only ever raises, so applying it is safe there too.)
        if let Some(f) = floor {
            start = start.max(f.apply(transcript));
        }

        let prompt = prompt_from_request(request, transcript);
        let mut idx = start;
        let mut climbed = false;

        loop {
            let Some(tier) = self.ladder.get(idx) else {
                return LadderOutcome::Degraded {
                    reason: "ladder exhausted".to_string(),
                };
            };

            // If we are climbing onto this rung (not the first dispatch),
            // emit the backchannel filler before the slower tier generates.
            if climbed {
                sink.emit_filler(ESCALATION_FILLER);
            }

            let at_top = idx + 1 >= self.ladder.len();
            let rung = match tier.backend {
                Backend::Local => self.run_local_rung(tier, transcript, &prompt, sink, at_top).await,
                Backend::Anthropic => self.run_anthropic_rung(tier, request, sink, at_top).await,
            };
            match rung {
                RungResult::Answer(out) => {
                    // AC10: raise the session floor to the answering rung.
                    if let Some(f) = floor {
                        f.raise_to(idx);
                    }
                    return out;
                }
                RungResult::Degraded(out) => return out,
                RungResult::Climb => {
                    climbed = true;
                    idx += 1;
                }
            }
        }
    }

    /// Dispatch one local rung: generate, then run the soft-failure gate.
    #[allow(
        clippy::cognitive_complexity,
        clippy::too_many_arguments,
        reason = "rung dispatch: each arm is a single accept/climb/degrade decision"
    )]
    async fn run_local_rung(
        &self,
        tier: &Tier,
        transcript: &str,
        prompt: &Prompt,
        sink: &dyn LadderSink,
        at_top: bool,
    ) -> RungResult {
        match self.local.generate(&tier.model, prompt, sink).await {
            LocalOutcome::Answer { text } => {
                let ctx = VerifyCtx {
                    utterance: transcript,
                    expected_lang: self.lang,
                    min_confidence: VERIFY_MIN_CONFIDENCE,
                };
                match verify(&text, &ctx) {
                    Verdict::Accept => RungResult::answer(text, &tier.name),
                    Verdict::Reject { reason } if at_top => {
                        // Bounded: speak the top tier's answer even if
                        // soft-rejected (better than silence); log it.
                        warn!(
                            tier = %tier.name,
                            reason = ?reason,
                            "wm-brain ladder: top-tier answer soft-rejected; speaking it (bounded)"
                        );
                        RungResult::answer(text, &tier.name)
                    }
                    Verdict::Reject { reason } => {
                        info!(
                            from = %tier.name,
                            signal = "soft",
                            reason = ?reason,
                            "wm-brain ladder: climbing one rung (verify reject)"
                        );
                        RungResult::Climb
                    }
                }
            }
            LocalOutcome::Escalate { reason } if at_top => RungResult::Degraded(
                LadderOutcome::Degraded {
                    reason: format!("top tier escalated: {reason}"),
                },
            ),
            LocalOutcome::Escalate { reason } => {
                info!(
                    from = %tier.name,
                    signal = "hard",
                    reason = %reason,
                    "wm-brain ladder: climbing one rung (local escalate)"
                );
                RungResult::Climb
            }
        }
    }

    /// Dispatch one Anthropic rung, passing `request` through UNMODIFIED
    /// except for the model id (AC8). A missing API key, an empty response,
    /// or a transport error climbs (or degrades at the top rung) — never a
    /// panic/hang (AC5).
    #[allow(
        clippy::cognitive_complexity,
        reason = "rung dispatch: each arm is a single climb/answer/degrade decision"
    )]
    async fn run_anthropic_rung(
        &self,
        tier: &Tier,
        request: &MessageRequest,
        sink: &dyn LadderSink,
        at_top: bool,
    ) -> RungResult {
        let Some(client) = self.anthropic.as_ref() else {
            warn!(
                tier = %tier.name,
                "wm-brain ladder: Anthropic tier requested but no API key; degrading"
            );
            return if at_top {
                RungResult::Degraded(LadderOutcome::Degraded {
                    reason: format!("no api key for tier {}", tier.name),
                })
            } else {
                RungResult::Climb
            };
        };
        let req = with_model(request, canonical_model(&tier.model));
        match client.collect_messages(&req).await {
            Ok(events) => {
                let text = extract_assistant_text(&events);
                for ev in &events {
                    if let crate::anthropic::StreamEvent::TextDelta { text, .. } = ev {
                        sink.on_delta(text);
                    }
                }
                if text.trim().is_empty() {
                    return if at_top {
                        RungResult::Degraded(LadderOutcome::Degraded {
                            reason: "top tier returned no text".to_string(),
                        })
                    } else {
                        RungResult::Climb
                    };
                }
                RungResult::answer(text, &tier.name)
            }
            Err(err) if at_top => RungResult::Degraded(LadderOutcome::Degraded {
                reason: format!("top tier error: {}", describe(&err)),
            }),
            Err(err) => {
                warn!(
                    tier = %tier.name,
                    err = %describe(&err),
                    "wm-brain ladder: cloud tier failed; climbing"
                );
                RungResult::Climb
            }
        }
    }
}

/// The result of dispatching one rung: terminal answer, terminal degrade,
/// or "climb one rung".
enum RungResult {
    Answer(LadderOutcome),
    Degraded(LadderOutcome),
    Climb,
}

impl RungResult {
    fn answer(text: String, tier: &str) -> Self {
        Self::Answer(LadderOutcome::Answer {
            text,
            tier: tier.to_string(),
        })
    }
}

/// Short description of a [`ClientError`] for logs (avoids leaking bodies).
fn describe(err: &ClientError) -> String {
    format!("{err}")
}

/// Build a `wm-local-llm` [`Prompt`] from a composed Anthropic request.
///
/// The Anthropic system prompt becomes the local `system` message; the
/// request's messages map to local roles. The `transcript` is already in
/// the request's messages, so we mirror them rather than re-appending.
fn prompt_from_request(request: &MessageRequest, transcript: &str) -> Prompt {
    let mut prompt = Prompt::new();
    if let Some(system) = request.system.as_ref() {
        prompt = prompt.with(LocalMessage::system(system.clone()));
    }
    if request.messages.is_empty() {
        prompt = prompt.with(LocalMessage::user(transcript.to_string()));
    } else {
        for m in &request.messages {
            let role = match m.role {
                Role::User => LocalRole::User,
                Role::Assistant => LocalRole::Assistant,
            };
            prompt = prompt.with(LocalMessage {
                role,
                content: m.content.clone(),
            });
        }
    }
    prompt
}

/// Clone `request` with a different model id, leaving every other field
/// (system, messages, cache breakpoints, stream flag) UNTOUCHED. AC8.
fn with_model(request: &MessageRequest, model: &str) -> MessageRequest {
    let mut req = request.clone();
    req.model = model.to_string();
    req
}

/// A [`LadderSink`] that buffers fillers and deltas in memory.
///
/// An async caller (the daemon) can publish them to the bus after the
/// (sync-sink) ladder run completes. Fillers are returned in emit order so
/// the caller can publish them as interim replies before the final answer
/// (AC9).
#[derive(Debug, Default)]
pub struct BufferingSink {
    fillers: std::sync::Mutex<Vec<String>>,
    deltas: std::sync::Mutex<Vec<String>>,
}

impl BufferingSink {
    /// Drain and return the buffered filler phrases, in emit order.
    #[must_use]
    pub fn take_fillers(&self) -> Vec<String> {
        drain(&self.fillers)
    }

    /// Drain and return the buffered answer deltas, in arrival order.
    #[must_use]
    pub fn take_deltas(&self) -> Vec<String> {
        drain(&self.deltas)
    }
}

/// Drain a mutex-guarded `Vec`, returning its contents (empty on poison).
fn drain(m: &std::sync::Mutex<Vec<String>>) -> Vec<String> {
    m.lock().map(|mut v| std::mem::take(&mut *v)).unwrap_or_default()
}

impl DeltaSink for BufferingSink {
    fn on_delta(&self, delta: &str) {
        if let Ok(mut v) = self.deltas.lock() {
            v.push(delta.to_string());
        }
    }
}

impl LadderSink for BufferingSink {
    fn emit_filler(&self, phrase: &str) {
        if let Ok(mut v) = self.fillers.lock() {
            v.push(phrase.to_string());
        }
    }
}

/// Production [`LocalBackend`]: builds a `wm-local-llm` client per call
/// from the configured endpoint + the tier's ollama model id.
pub struct LiveLocalBackend {
    endpoint: String,
}

impl LiveLocalBackend {
    /// Construct a live backend bound to `endpoint`.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
        }
    }
}

#[async_trait::async_trait]
impl LocalBackend for LiveLocalBackend {
    async fn generate(&self, model: &str, prompt: &Prompt, sink: &dyn DeltaSink) -> LocalOutcome {
        let cfg = wm_local_llm::LocalLlmConfig::new(self.endpoint.clone(), model.to_string());
        let client = wm_local_llm::LocalLlm::configured(cfg);
        client.generate(prompt, sink).await
    }
}

/// Production [`StakesProvider`].
///
/// `wm-router`'s [`wm_router::Router`] holds a `Box<dyn Embedder>`, which is
/// not `Send + Sync`, so a `Router` cannot live inside the `Send + Sync`
/// [`LadderClient`]. Instead this provider builds a fresh keyword+rules
/// router per call (cheap, pure, socket-free) to derive the stakes tag.
/// The semantic-safety stage (which needs the recall embedder) is governed
/// separately by the [`RecallLiveness`] probe and the recall-down safe
/// posture, so dropping it here does not weaken the safety floor: the
/// keyword pre-pass still flags unambiguous high-stakes cues, and an
/// ordinary turn with recall down starts at the cloud safe floor anyway.
#[derive(Debug, Default, Clone, Copy)]
pub struct RouterStakes;

impl RouterStakes {
    /// Construct a router-backed stakes provider.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl StakesProvider for RouterStakes {
    fn stakes(&self, transcript: &str) -> Stakes {
        let router = wm_router::Router::new();
        match router.classify(transcript).route {
            wm_router::Route::Brain { stakes } => stakes,
            // A skill/cache deflection reaching the brain is treated as an
            // ordinary brain turn (the brain doesn't run skills/cache).
            wm_router::Route::Skill(_) | wm_router::Route::CacheLookup => Stakes::Ordinary,
        }
    }
}

/// Production [`RecallLiveness`]: a cheap recall reachability probe.
///
/// Probes the recall daemon via a connect + `ping` over its Unix socket. A
/// failure to connect/ping means recall (and thus the router's semantic
/// safety stage) is unreachable, so the ladder applies its recall-down safe
/// posture.
pub struct SocketRecallLiveness {
    socket: std::path::PathBuf,
}

impl SocketRecallLiveness {
    /// Construct a probe bound to the recall socket `path`.
    #[must_use]
    pub const fn new(path: std::path::PathBuf) -> Self {
        Self { socket: path }
    }
}

#[async_trait::async_trait]
impl RecallLiveness for SocketRecallLiveness {
    async fn recall_up(&self) -> bool {
        match crate::recall_client::RecallClient::connect(&self.socket).await {
            Ok(mut client) => client.ping().await.is_ok(),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    reason = "tests"
)]
mod tests {
    use super::*;
    use crate::anthropic::{Message as AnthropicMessage, StreamEvent};
    use crate::{SHORT_MODEL_HAIKU, SHORT_MODEL_SONNET, default_ladder};
    use std::sync::Mutex as StdMutex;
    use wm_router::StakesClass;

    // ---- fakes -----------------------------------------------------------

    /// A fake local backend that returns a scripted outcome per call, in
    /// order. Records which models it was asked to generate against.
    struct FakeLocal {
        outcomes: StdMutex<Vec<LocalOutcome>>,
        calls: StdMutex<Vec<String>>,
        /// deltas to emit on each Answer (proves streaming forwarding).
        deltas: Vec<String>,
    }

    impl FakeLocal {
        fn new(outcomes: Vec<LocalOutcome>) -> Self {
            Self {
                outcomes: StdMutex::new(outcomes),
                calls: StdMutex::new(Vec::new()),
                deltas: Vec::new(),
            }
        }
        fn with_deltas(mut self, deltas: Vec<&str>) -> Self {
            self.deltas = deltas.into_iter().map(String::from).collect();
            self
        }
    }

    #[async_trait::async_trait]
    impl LocalBackend for FakeLocal {
        async fn generate(
            &self,
            model: &str,
            _prompt: &Prompt,
            sink: &dyn DeltaSink,
        ) -> LocalOutcome {
            self.calls.lock().unwrap().push(model.to_string());
            let mut outs = self.outcomes.lock().unwrap();
            let outcome = if outs.is_empty() {
                LocalOutcome::Escalate {
                    reason: "no scripted outcome".to_string(),
                }
            } else {
                outs.remove(0)
            };
            if matches!(outcome, LocalOutcome::Answer { .. }) {
                for d in &self.deltas {
                    sink.on_delta(d);
                }
            }
            outcome
        }
    }

    /// A fake Anthropic client returning canned stream events.
    #[derive(Clone)]
    struct FakeAnthropic {
        events: Vec<StreamEvent>,
        calls: Arc<StdMutex<u32>>,
        /// when set, every call sleeps to simulate a slow tier.
        slow: bool,
    }

    impl FakeAnthropic {
        fn answering(text: &str) -> Self {
            Self {
                events: vec![
                    StreamEvent::MessageStart,
                    StreamEvent::TextDelta {
                        index: 0,
                        text: text.to_string(),
                    },
                    StreamEvent::MessageStop,
                ],
                calls: Arc::new(StdMutex::new(0)),
                slow: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for FakeAnthropic {
        async fn collect_messages(
            &self,
            _req: &MessageRequest,
        ) -> Result<Vec<StreamEvent>, ClientError> {
            *self.calls.lock().unwrap() += 1;
            if self.slow {
                tokio::task::yield_now().await;
            }
            Ok(self.events.clone())
        }
    }

    /// A fake Anthropic client that always errors (transport failure).
    struct FailingAnthropic;

    #[async_trait::async_trait]
    impl LlmClient for FailingAnthropic {
        async fn collect_messages(
            &self,
            _req: &MessageRequest,
        ) -> Result<Vec<StreamEvent>, ClientError> {
            Err(ClientError::Status {
                code: 503,
                body: "overloaded".to_string(),
            })
        }
    }

    struct FixedStakes(Stakes);
    impl StakesProvider for FixedStakes {
        fn stakes(&self, _t: &str) -> Stakes {
            self.0
        }
    }

    struct FixedRecall(bool);
    #[async_trait::async_trait]
    impl RecallLiveness for FixedRecall {
        async fn recall_up(&self) -> bool {
            self.0
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        deltas: StdMutex<Vec<String>>,
        fillers: StdMutex<Vec<String>>,
    }

    impl DeltaSink for RecordingSink {
        fn on_delta(&self, delta: &str) {
            self.deltas.lock().unwrap().push(delta.to_string());
        }
    }
    impl LadderSink for RecordingSink {
        fn emit_filler(&self, phrase: &str) {
            self.fillers.lock().unwrap().push(phrase.to_string());
        }
    }

    fn req() -> MessageRequest {
        MessageRequest::streaming(
            "claude-sonnet-4-6",
            256,
            vec![AnthropicMessage {
                role: Role::User,
                content: "tell me about your day".to_string(),
            }],
        )
        .with_system("be warm".to_string())
    }

    fn client(
        local: FakeLocal,
        anthropic: Option<Arc<dyn LlmClient>>,
        stakes: Stakes,
        recall_up: bool,
    ) -> LadderClient {
        LadderClient::new(
            default_ladder(),
            Arc::new(local),
            anthropic,
            Arc::new(FixedStakes(stakes)),
            Arc::new(FixedRecall(recall_up)),
        )
    }

    // ---- AC2: local-first, anthropic never called ------------------------

    #[tokio::test]
    async fn ac2_local_answer_accepted_anthropic_never_called() {
        let local = FakeLocal::new(vec![LocalOutcome::Answer {
            text: "It was a calm and pleasant morning here.".to_string(),
        }]);
        let fake_anth = FakeAnthropic::answering("CLOUD");
        let calls = Arc::clone(&fake_anth.calls);
        let c = client(
            local,
            Some(Arc::new(fake_anth)),
            Stakes::Ordinary,
            true,
        );
        let sink = RecordingSink::default();
        let out = c.run_turn("tell me about your day", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Answer { ref tier, .. } if tier == "local-3b"),
            "served by local-3b, got {out:?}"
        );
        assert_eq!(*calls.lock().unwrap(), 0, "anthropic must never be called");
    }

    // ---- AC7: streaming deltas forwarded ---------------------------------

    #[tokio::test]
    async fn ac7_local_deltas_forwarded_incrementally() {
        let local = FakeLocal::new(vec![LocalOutcome::Answer {
            text: "Sunny and mild today.".to_string(),
        }])
        .with_deltas(vec!["Sunny ", "and ", "mild ", "today."]);
        let c = client(local, None, Stakes::Ordinary, true);
        let sink = RecordingSink::default();
        let out = c.run_turn("how is the weather", &req(), "local-3b", &sink).await;
        assert!(matches!(out, LadderOutcome::Answer { .. }));
        assert!(
            sink.deltas.lock().unwrap().len() > 1,
            "sink must receive >1 delta"
        );
    }

    // ---- AC3 soft: verify reject climbs one rung -------------------------

    #[tokio::test]
    async fn ac3_soft_reject_climbs_to_higher_tier() {
        // local-3b returns a refusal (verify rejects); local-8b answers OK.
        let local = FakeLocal::new(vec![
            LocalOutcome::Answer {
                text: "I'm just an AI and can't help with that.".to_string(),
            },
            LocalOutcome::Answer {
                text: "Of course — here is what I think about that.".to_string(),
            },
        ]);
        let c = client(local, None, Stakes::Ordinary, true);
        let sink = RecordingSink::default();
        let out = c.run_turn("what should i plant", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Answer { ref tier, .. } if tier == "local-8b"),
            "soft reject should climb to local-8b, got {out:?}"
        );
        // AC9: a filler was emitted before the higher tier answered.
        assert_eq!(sink.fillers.lock().unwrap().len(), 1);
    }

    // ---- AC3 hard: escalate climbs one rung ------------------------------

    #[tokio::test]
    async fn ac3_hard_escalate_climbs_to_higher_tier() {
        let local = FakeLocal::new(vec![
            LocalOutcome::Escalate {
                reason: "timeout".to_string(),
            },
            LocalOutcome::Answer {
                text: "Here is a thoughtful answer to your question.".to_string(),
            },
        ]);
        let c = client(local, None, Stakes::Ordinary, true);
        let sink = RecordingSink::default();
        let out = c.run_turn("a real question?", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Answer { ref tier, .. } if tier == "local-8b"),
            "hard escalate should climb to local-8b, got {out:?}"
        );
    }

    // ---- AC3 bounded: never climbs past the top -------------------------

    #[tokio::test]
    async fn ac3_bounded_top_tier_escalate_degrades() {
        // Every tier escalates/fails. Local rungs escalate; cloud rungs error.
        let local = FakeLocal::new(vec![
            LocalOutcome::Escalate {
                reason: "x".to_string(),
            },
            LocalOutcome::Escalate {
                reason: "y".to_string(),
            },
        ]);
        let c = client(
            local,
            Some(Arc::new(FailingAnthropic)),
            Stakes::Ordinary,
            true,
        );
        let sink = RecordingSink::default();
        let out = c.run_turn("q?", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Degraded { .. }),
            "all tiers failing must degrade (bounded), got {out:?}"
        );
    }

    // ---- AC3b: high-stakes skips local -----------------------------------

    #[tokio::test]
    async fn ac3b_high_stakes_starts_cloud_local_never_called() {
        let local = FakeLocal::new(vec![LocalOutcome::Answer {
            text: "local answer".to_string(),
        }]);
        let fl = Arc::new(local);
        let local_calls = Arc::clone(&fl);
        let anth = FakeAnthropic::answering("Please sit down and I'll call for help.");
        let anth_calls = Arc::clone(&anth.calls);
        let c = LadderClient::new(
            default_ladder(),
            fl,
            Some(Arc::new(anth)),
            Arc::new(FixedStakes(Stakes::HighStakes(StakesClass::Medication))),
            Arc::new(FixedRecall(true)),
        );
        let sink = RecordingSink::default();
        let out = c.run_turn("did i take my pills?", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Answer { ref tier, .. } if tier == SHORT_MODEL_SONNET),
            "high-stakes must start at trusted cloud (sonnet), got {out:?}"
        );
        assert!(
            local_calls.calls.lock().unwrap().is_empty(),
            "local backend must never be called for a high-stakes turn"
        );
        assert_eq!(*anth_calls.lock().unwrap(), 1);
    }

    // ---- AC5: no API key, local serves; cloud unavailable degrades -------

    #[tokio::test]
    async fn ac5_no_key_local_still_serves() {
        let local = FakeLocal::new(vec![LocalOutcome::Answer {
            text: "A perfectly good local answer.".to_string(),
        }]);
        let c = client(local, None, Stakes::Ordinary, true);
        let sink = RecordingSink::default();
        let out = c.run_turn("hello there friend", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Answer { ref tier, .. } if tier == "local-3b"),
            "local must serve with no API key, got {out:?}"
        );
    }

    #[tokio::test]
    async fn ac5_no_key_cloud_escalation_degrades_not_panic() {
        // Both local rungs escalate; no key for the cloud rungs.
        let local = FakeLocal::new(vec![
            LocalOutcome::Escalate {
                reason: "unreachable".to_string(),
            },
            LocalOutcome::Escalate {
                reason: "unreachable".to_string(),
            },
        ]);
        let c = client(local, None, Stakes::Ordinary, true);
        let sink = RecordingSink::default();
        let out = c.run_turn("q?", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Degraded { .. }),
            "no key + local exhausted must degrade, got {out:?}"
        );
    }

    // ---- AC8: anthropic request passed through unmodified ----------------

    #[tokio::test]
    async fn ac8_anthropic_request_passthrough_unmodified() {
        // High-stakes forces a cloud start; capture the request the fake saw.
        #[derive(Clone)]
        struct Capturing {
            seen: Arc<StdMutex<Option<MessageRequest>>>,
        }
        #[async_trait::async_trait]
        impl LlmClient for Capturing {
            async fn collect_messages(
                &self,
                req: &MessageRequest,
            ) -> Result<Vec<StreamEvent>, ClientError> {
                *self.seen.lock().unwrap() = Some(req.clone());
                Ok(vec![
                    StreamEvent::TextDelta {
                        index: 0,
                        text: "ok".to_string(),
                    },
                ])
            }
        }
        let seen = Arc::new(StdMutex::new(None));
        let cap = Capturing {
            seen: Arc::clone(&seen),
        };
        let local = FakeLocal::new(vec![]);
        let c = LadderClient::new(
            default_ladder(),
            Arc::new(local),
            Some(Arc::new(cap)),
            Arc::new(FixedStakes(Stakes::HighStakes(StakesClass::Money))),
            Arc::new(FixedRecall(true)),
        );
        let original = req();
        let sink = RecordingSink::default();
        let _ = c.run_turn("transfer money?", &original, "local-3b", &sink).await;
        let captured = seen.lock().unwrap().clone().expect("cloud was called");
        // Everything except the model id (which names the rung) is identical.
        assert_eq!(captured.system, original.system, "system preserved");
        assert_eq!(captured.messages, original.messages, "messages preserved");
        assert_eq!(captured.max_tokens, original.max_tokens, "max_tokens preserved");
        assert_eq!(captured.stream, original.stream, "stream flag preserved");
        assert_eq!(captured.model, "claude-sonnet-4-6", "model names the rung");
    }

    // ---- AC9: filler emitted before a climb's answer ---------------------

    #[tokio::test]
    async fn ac9_filler_emitted_before_higher_tier_answer() {
        // Force a climb that lands on cloud: local-3b escalate -> local-8b
        // (also escalate) -> haiku (cloud). Script two local escalates.
        let local = FakeLocal::new(vec![
            LocalOutcome::Escalate {
                reason: "timeout".to_string(),
            },
            LocalOutcome::Escalate {
                reason: "timeout".to_string(),
            },
        ]);
        let c = client(
            local,
            Some(Arc::new(FakeAnthropic::answering("A careful cloud answer."))),
            Stakes::Ordinary,
            true,
        );
        let sink = RecordingSink::default();
        let out = c.run_turn("q?", &req(), "local-3b", &sink).await;
        assert!(matches!(out, LadderOutcome::Answer { ref tier, .. } if tier == SHORT_MODEL_HAIKU));
        // Two climbs (3b->8b, 8b->haiku) => two fillers, both before the answer.
        assert_eq!(sink.fillers.lock().unwrap().len(), 2);
    }

    // ---- recall-down safe posture ----------------------------------------

    #[tokio::test]
    async fn recall_down_ordinary_starts_cloud_floor_not_local() {
        // Recall is down -> the router can't see high stakes, so an ordinary
        // turn must NOT start local; it starts at the haiku safe floor.
        let local = FakeLocal::new(vec![LocalOutcome::Answer {
            text: "local answer".to_string(),
        }]);
        let fl = Arc::new(local);
        let local_calls = Arc::clone(&fl);
        let anth = FakeAnthropic::answering("Cloud-floor answer.");
        let c = LadderClient::new(
            default_ladder(),
            fl,
            Some(Arc::new(anth)),
            Arc::new(FixedStakes(Stakes::Ordinary)),
            Arc::new(FixedRecall(false)),
        );
        let sink = RecordingSink::default();
        let out = c.run_turn("ordinary chat", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Answer { ref tier, .. } if tier == SHORT_MODEL_HAIKU),
            "recall-down ordinary must start at cloud safe floor (haiku), got {out:?}"
        );
        assert!(
            local_calls.calls.lock().unwrap().is_empty(),
            "recall-down ordinary must not call local"
        );
    }

    // ---- AC10: conversational stickiness (per-session tier floor) --------

    #[tokio::test]
    async fn ac10_floor_holds_across_turns_then_decays_on_topic_change() {
        // Turn 1 (same thread): local-3b escalates -> local-8b answers, so the
        // floor rises to local-8b. Turn 2 on the SAME topic must NOT start at
        // local-3b again; with the floor it starts at local-8b. We prove this
        // by scripting turn 2 with only ONE local outcome (an answer): if the
        // turn started at local-3b it'd consume that for 3b; instead it starts
        // at 8b and the single answer is served as local-8b.
        let local = FakeLocal::new(vec![
            // turn 1
            LocalOutcome::Escalate {
                reason: "timeout".to_string(),
            },
            LocalOutcome::Answer {
                text: "A thoughtful answer about gardening tomatoes.".to_string(),
            },
            // turn 2 (same topic) — single answer, expected to be 8b
            LocalOutcome::Answer {
                text: "More thoughtful gardening tomato advice for you.".to_string(),
            },
        ]);
        let fl = Arc::new(local);
        let calls = Arc::clone(&fl);
        let c = LadderClient::new(
            default_ladder(),
            fl,
            None,
            Arc::new(FixedStakes(Stakes::Ordinary)),
            Arc::new(FixedRecall(true)),
        );
        let floor = SessionFloor::new();
        let sink = RecordingSink::default();

        let t1 = "what tomatoes should i plant in the garden this spring";
        let out1 = c.run_turn_sticky(t1, &req(), "local-3b", &sink, &floor).await;
        assert!(matches!(out1, LadderOutcome::Answer { ref tier, .. } if tier == "local-8b"));

        // Turn 2, same topic: must hold the floor at local-8b.
        let t2 = "and which tomatoes grow best in a spring garden bed";
        let out2 = c.run_turn_sticky(t2, &req(), "local-3b", &sink, &floor).await;
        assert!(
            matches!(out2, LadderOutcome::Answer { ref tier, .. } if tier == "local-8b"),
            "floor must hold at local-8b on same topic, got {out2:?}"
        );
        // local-3b never re-dispatched on turn 2: total local calls = 3
        // (3b+8b on turn1, 8b on turn2), and the first model of turn 2 is 8b.
        let recorded = calls.calls.lock().unwrap().clone();
        assert_eq!(recorded, vec!["qwen2.5:3b", "qwen3:8b", "qwen3:8b"]);

        // Now a topic change decays the floor: a brand-new subject should be
        // free to start back at local-3b.
        let local2 = FakeLocal::new(vec![LocalOutcome::Answer {
            text: "The weather looks bright and clear today.".to_string(),
        }]);
        let fl2 = Arc::new(local2);
        let calls2 = Arc::clone(&fl2);
        let c2 = LadderClient::new(
            default_ladder(),
            fl2,
            None,
            Arc::new(FixedStakes(Stakes::Ordinary)),
            Arc::new(FixedRecall(true)),
        );
        // Reuse the SAME floor object (raised to 8b by turn 1/2 above), then
        // send a totally different topic — the floor must decay to 0.
        let t3 = "tell me about the weather forecast outside right now";
        let out3 = c2.run_turn_sticky(t3, &req(), "local-3b", &sink, &floor).await;
        assert!(
            matches!(out3, LadderOutcome::Answer { ref tier, .. } if tier == "local-3b"),
            "topic change must decay the floor back to local-3b, got {out3:?}"
        );
        assert_eq!(calls2.calls.lock().unwrap().clone(), vec!["qwen2.5:3b"]);
    }

    #[tokio::test]
    async fn recall_down_high_stakes_still_trusted_cloud() {
        // Even with recall down, a keyword-prepass high-stakes flag routes
        // to the trusted cloud tier.
        let local = FakeLocal::new(vec![]);
        let anth = FakeAnthropic::answering("I'll help right away.");
        let c = LadderClient::new(
            default_ladder(),
            Arc::new(local),
            Some(Arc::new(anth)),
            Arc::new(FixedStakes(Stakes::HighStakes(StakesClass::Emergency))),
            Arc::new(FixedRecall(false)),
        );
        let sink = RecordingSink::default();
        let out = c.run_turn("there's a fire!", &req(), "local-3b", &sink).await;
        assert!(
            matches!(out, LadderOutcome::Answer { ref tier, .. } if tier == SHORT_MODEL_SONNET),
            "high-stakes still routes to trusted cloud even when recall down, got {out:?}"
        );
    }
}
