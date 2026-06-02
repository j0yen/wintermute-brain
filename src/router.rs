//! Turn-type classifier and routing-policy table for the brain.
//!
//! [`classify_turn`] labels a user transcript as **command/control** or
//! **conversational** using cheap, deterministic rules — no model call
//! on the hot path.  [`RoutingPolicy`] implements the six-row policy table
//! from PRD-wintermute-brain-routing §2.2 to pick a starting tier and
//! produce a [`RouteDecision`] that is published as `wm.brain.route` on
//! every turn for observability.

use serde::{Deserialize, Serialize};
use tracing::info;

// ── Turn classification ────────────────────────────────────────────────────────

/// Broad turn-type label: a short command/control phrase, or an open
/// conversational turn.  Ambiguous turns are labelled [`TurnKind::Conversational`]
/// so they go to the better mind when the cloud is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnKind {
    /// Matches command/control patterns (imperative verbs, time/date queries,
    /// playback controls) or is shorter than [`COMMAND_MAX_WORDS_DEFAULT`].
    Command,
    /// Everything else — open questions, statements, anything long or
    /// emotionally open-ended.
    Conversational,
}

/// Default word-count ceiling below which a turn is presumed a command when
/// it has no open-conversation particles.  Configurable via `brain.toml`
/// `[routing] command_max_words`.
pub const COMMAND_MAX_WORDS_DEFAULT: usize = 6;

/// Leading imperative verbs that strongly signal a control/command turn.
const COMMAND_VERBS: &[&str] = &[
    "turn",
    "set",
    "start",
    "stop",
    "pause",
    "resume",
    "play",
    "skip",
    "mute",
    "unmute",
    "louder",
    "quieter",
    "increase",
    "decrease",
    "open",
    "close",
    "lock",
    "unlock",
    "show",
    "hide",
    "enable",
    "disable",
    "remind",
    "timer",
    "alarm",
    "cancel",
];

/// Closed-world time/date/status phrases that are always command-class.
const COMMAND_PATTERNS: &[&str] = &[
    "what time",
    "what's the time",
    "what date",
    "what day",
    "what's today",
    "what's the date",
    "what's the weather",
    "how's the weather",
];

/// Open-conversation particles that signal this is *not* a command, even
/// when the turn is short.
const CONVERSATION_PARTICLES: &[&str] = &[
    "how are",
    "how do",
    "how does",
    "how did",
    "how would",
    "how should",
    "how can",
    "tell me",
    "what do you think",
    "do you",
    "can you",
    "could you",
    "would you",
    "why",
    "explain",
    "describe",
    "talk about",
];

/// Classify a transcript as command or conversational.
///
/// Rules (in order):
/// 1. Exact-prefix match against [`COMMAND_PATTERNS`] → `Command`.
/// 2. First (lowercased) word matches [`COMMAND_VERBS`] → `Command`.
/// 3. Substring match against [`CONVERSATION_PARTICLES`] → `Conversational`.
/// 4. Word count ≤ `command_max_words` → `Command`.
/// 5. Default → `Conversational`.
#[must_use]
pub fn classify_turn(transcript: &str, command_max_words: usize) -> TurnKind {
    let lower = transcript.trim().to_ascii_lowercase();
    let lower = lower.trim_end_matches(|c: char| c.is_ascii_punctuation());

    // Rule 1: closed-world patterns
    for pat in COMMAND_PATTERNS {
        if lower.starts_with(pat) {
            return TurnKind::Command;
        }
    }

    // Rule 2: leading command verb
    let first_word = lower
        .split_ascii_whitespace()
        .next()
        .unwrap_or("");
    if COMMAND_VERBS.contains(&first_word) {
        return TurnKind::Command;
    }

    // Rule 3: conversation particles (marks this as NOT a command)
    for particle in CONVERSATION_PARTICLES {
        if lower.contains(particle) {
            return TurnKind::Conversational;
        }
    }

    // Rule 4: short enough to be a command
    let word_count = lower.split_ascii_whitespace().count();
    if word_count <= command_max_words && word_count > 0 {
        return TurnKind::Command;
    }

    // Rule 5: default — treat as conversational
    TurnKind::Conversational
}

// ── Route decision ─────────────────────────────────────────────────────────────

/// The tier the router recommends for a given turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouteTier {
    /// Start at the local tier (lowest cost, lowest latency).
    Local,
    /// Start at the cloud tier (higher quality, higher latency).
    Cloud,
    /// Both tiers were unavailable; use a canned degrade phrase.
    Canned,
}

/// Machine-readable reason for the routing decision, emitted in the
/// `wm.brain.route` envelope for operator observability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteReason {
    /// Per-turn override was set by `wmd swap-model`.
    Override,
    /// Turn classified as command/control → local low-latency path.
    Command,
    /// Turn conversational, online, API key present → cloud for quality.
    Conversational,
    /// Turn conversational but no API key → forced local.
    NoKey,
    /// Turn conversational but network probe shows offline → forced local.
    Offline,
    /// Cloud backend exceeded its latency budget → fell back to local.
    CloudTimeoutFallback,
    /// Local backend also unavailable → canned degrade phrase.
    TotalFailure,
    /// Deployment-wide `prefer = "local-only"` override.
    PreferLocalOnly,
    /// Deployment-wide `prefer = "cloud-only"` override.
    PreferCloudOnly,
}

impl RouteReason {
    /// Machine-readable string emitted in the `wm.brain.route` envelope.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::Command => "command",
            Self::Conversational => "conversational",
            Self::NoKey => "no_key",
            Self::Offline => "offline",
            Self::CloudTimeoutFallback => "cloud_timeout_fallback",
            Self::TotalFailure => "total_failure",
            Self::PreferLocalOnly => "prefer_local_only",
            Self::PreferCloudOnly => "prefer_cloud_only",
        }
    }
}

/// Complete routing decision for a single turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    /// Recommended starting tier.
    pub tier: RouteTier,
    /// Machine-readable reason.
    pub reason: RouteReason,
    /// Turn classification used to reach the decision.
    pub turn_kind: TurnKind,
}

// ── Deployment-wide preference ─────────────────────────────────────────────────

/// Deployment-wide tier preference stored in `brain.toml [routing] prefer`.
///
/// Overrides classification for the entire deployment (e.g. set
/// `LocalOnly` when no API key is available, `CloudOnly` for maximum
/// quality in a supervised session).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoutePrefer {
    /// Classification-based routing (the default).
    #[default]
    Auto,
    /// Force every turn to the local tier, regardless of key or network.
    LocalOnly,
    /// Force every turn to the cloud tier.
    CloudOnly,
}

impl RoutePrefer {
    /// Parse a string accepted by `wmd route prefer`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "local-only" => Some(Self::LocalOnly),
            "cloud-only" => Some(Self::CloudOnly),
            _ => None,
        }
    }

    /// The canonical string stored in `brain.toml`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::LocalOnly => "local-only",
            Self::CloudOnly => "cloud-only",
        }
    }
}

// ── Routing config ─────────────────────────────────────────────────────────────

/// The `[routing]` section of `brain.toml`.
///
/// All fields are `#[serde(default)]` so existing `brain.toml` files without
/// a `[routing]` section deserialise cleanly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingConfig {
    /// Ollama model for the local tier.  Default: `qwen2.5:3b`.
    #[serde(default = "default_local_model")]
    pub local_model: String,
    /// Short model name for the cloud tier.  Default: `sonnet`.
    #[serde(default = "default_cloud_model")]
    pub cloud_model: String,
    /// Ollama base URL.  Default: `http://localhost:11434`.
    #[serde(default = "default_ollama_endpoint")]
    pub ollama_endpoint: String,
    /// `OLLAMA_KEEP_ALIVE` value forwarded in every chat request.
    /// `-1` keeps the model resident indefinitely.
    #[serde(default = "default_ollama_keep_alive")]
    pub ollama_keep_alive: String,
    /// How long (ms) to wait for the cloud before falling back to local.
    #[serde(default = "default_cloud_timeout_ms")]
    pub cloud_timeout_ms: u64,
    /// How long (s) a reachability probe result is cached.
    #[serde(default = "default_reachability_ttl_s")]
    pub reachability_ttl_s: u64,
    /// Word-count ceiling below which a turn without conversation particles
    /// is classified as a command.
    #[serde(default = "default_command_max_words")]
    pub command_max_words: usize,
    /// Deployment-wide tier preference override.
    #[serde(default)]
    pub prefer: RoutePrefer,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            local_model: default_local_model(),
            cloud_model: default_cloud_model(),
            ollama_endpoint: default_ollama_endpoint(),
            ollama_keep_alive: default_ollama_keep_alive(),
            cloud_timeout_ms: default_cloud_timeout_ms(),
            reachability_ttl_s: default_reachability_ttl_s(),
            command_max_words: default_command_max_words(),
            prefer: RoutePrefer::default(),
        }
    }
}

fn default_local_model() -> String {
    "qwen2.5:3b".to_string()
}
fn default_cloud_model() -> String {
    "sonnet".to_string()
}
fn default_ollama_endpoint() -> String {
    "http://localhost:11434".to_string()
}
fn default_ollama_keep_alive() -> String {
    "-1".to_string()
}
const fn default_cloud_timeout_ms() -> u64 {
    6_000
}
const fn default_reachability_ttl_s() -> u64 {
    30
}
const fn default_command_max_words() -> usize {
    COMMAND_MAX_WORDS_DEFAULT
}

// ── Routing policy ─────────────────────────────────────────────────────────────

/// Inputs to [`apply_routing_policy`].
#[derive(Debug, Clone)]
pub struct PolicyInputs {
    /// Transcript of the user turn.
    pub transcript: String,
    /// Per-turn tier override (from `pending_tier`), if any.
    pub pending_tier_override: Option<String>,
    /// Whether the Anthropic API key is configured and non-empty.
    pub api_key_present: bool,
    /// Whether the Anthropic API host is reachable (cached probe result).
    pub online: bool,
    /// Deployment-wide preference from `[routing] prefer`.
    pub prefer: RoutePrefer,
    /// Word-count ceiling for command classification.
    pub command_max_words: usize,
}

/// Apply the six-row routing policy table (PRD §2.2) to decide a starting
/// tier and reason for this turn.
///
/// The returned [`RouteDecision`] is published as `wm.brain.route` by the
/// caller.  The `tier` field is a *recommendation* to the ladder; the ladder
/// may still escalate to a higher tier if the local backend fails.
#[must_use]
pub fn apply_routing_policy(inputs: &PolicyInputs) -> RouteDecision {
    let turn_kind = classify_turn(&inputs.transcript, inputs.command_max_words);

    // Row 1: per-turn override
    if inputs.pending_tier_override.is_some() {
        info!(
            transcript = %inputs.transcript,
            tier = "local",
            reason = RouteReason::Override.as_str(),
            "brain router: override route"
        );
        return RouteDecision {
            tier: RouteTier::Local,
            reason: RouteReason::Override,
            turn_kind,
        };
    }

    // Deployment-wide preference gates
    match inputs.prefer {
        RoutePrefer::LocalOnly => {
            info!(
                transcript = %inputs.transcript,
                tier = "local",
                reason = RouteReason::PreferLocalOnly.as_str(),
                "brain router: local-only"
            );
            return RouteDecision {
                tier: RouteTier::Local,
                reason: RouteReason::PreferLocalOnly,
                turn_kind,
            };
        }
        RoutePrefer::CloudOnly => {
            info!(
                transcript = %inputs.transcript,
                tier = "cloud",
                reason = RouteReason::PreferCloudOnly.as_str(),
                "brain router: cloud-only"
            );
            return RouteDecision {
                tier: RouteTier::Cloud,
                reason: RouteReason::PreferCloudOnly,
                turn_kind,
            };
        }
        RoutePrefer::Auto => {}
    }

    // Row 2: command → local (low latency)
    if turn_kind == TurnKind::Command {
        info!(
            transcript = %inputs.transcript,
            tier = "local",
            reason = RouteReason::Command.as_str(),
            "brain router: command route"
        );
        return RouteDecision {
            tier: RouteTier::Local,
            reason: RouteReason::Command,
            turn_kind,
        };
    }

    // Row 3: conversational + online + key → cloud
    if inputs.online && inputs.api_key_present {
        info!(
            transcript = %inputs.transcript,
            tier = "cloud",
            reason = RouteReason::Conversational.as_str(),
            "brain router: conversational cloud route"
        );
        return RouteDecision {
            tier: RouteTier::Cloud,
            reason: RouteReason::Conversational,
            turn_kind,
        };
    }

    // Row 4a: conversational, no key → local
    if !inputs.api_key_present {
        info!(
            transcript = %inputs.transcript,
            tier = "local",
            reason = RouteReason::NoKey.as_str(),
            "brain router: no api key → local"
        );
        return RouteDecision {
            tier: RouteTier::Local,
            reason: RouteReason::NoKey,
            turn_kind,
        };
    }

    // Row 4b: conversational, offline → local
    info!(
        transcript = %inputs.transcript,
        tier = "local",
        reason = RouteReason::Offline.as_str(),
        "brain router: offline → local"
    );
    RouteDecision {
        tier: RouteTier::Local,
        reason: RouteReason::Offline,
        turn_kind,
    }
}

/// Map a [`RouteDecision`] to the starting-tier name the ladder should begin
/// at.  Returns the tier name string used in [`crate::BrainConfig::default_tier`].
#[must_use]
pub fn decision_to_tier_name(decision: &RouteDecision, local_model: &str, cloud_model: &str) -> String {
    match decision.tier {
        RouteTier::Local => local_model.to_string(),
        RouteTier::Cloud => cloud_model.to_string(),
        RouteTier::Canned => local_model.to_string(), // canned phrase handled by the ladder degrade path
    }
}

// ── Canned degrade phrases ─────────────────────────────────────────────────────

/// Spoken-safe canned phrases used when both local and cloud backends are
/// unavailable (PRD §2.1 — CannedBackend).
///
/// Phrasing avoids technical jargon: no "servers", "API", or error codes.
pub const CANNED_PHRASES: &[&str] = &[
    "I'm having trouble thinking right now — can you say that again in a moment?",
    "Something went quiet on my end. I'll be back in just a moment.",
    "I need a brief pause to gather my thoughts. Can you try again?",
    "My mind is a bit cloudy right now. Please give me a moment.",
];

/// Pick a canned phrase by index, cycling through the bank.
///
/// The `index` is typically derived from a monotonic counter (e.g.
/// the turn counter) so the user hears varied phrases rather than the same
/// one repeatedly.
#[must_use]
pub fn canned_phrase(index: usize) -> &'static str {
    let n = CANNED_PHRASES.len();
    if n == 0 {
        return "I'm having trouble thinking right now.";
    }
    #[allow(clippy::indexing_slicing, reason = "index is modulo len, always in bounds")]
    CANNED_PHRASES[index % n]
}

// ── Route observability envelope ──────────────────────────────────────────────

/// `wm.brain.route` bus envelope published on every turn (PRD §2.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteEvent {
    /// Stable turn id (if available; the brain doesn't mint turn ids today,
    /// so this mirrors `ts` as a proxy).
    pub turn_id: u64,
    /// The tier that was recommended for this turn.
    pub tier: RouteTier,
    /// Machine-readable reason.
    pub reason: String,
    /// Observed latency of the LLM call in milliseconds (`None` until the
    /// turn completes; the event is published once with the final value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    /// The model that answered (tier name or model id).
    pub model: String,
    /// Unix milliseconds at publish time.
    pub ts: u64,
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

    // ── classify_turn ─────────────────────────────────────────────────────────

    #[test]
    fn command_verb_prefix_classifies_command() {
        assert_eq!(
            classify_turn("turn off the kitchen light", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Command
        );
        assert_eq!(
            classify_turn("mute yourself", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Command
        );
        assert_eq!(
            classify_turn("set a timer for ten minutes", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Command
        );
    }

    #[test]
    fn time_query_classifies_command() {
        assert_eq!(
            classify_turn("what time is it", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Command
        );
        assert_eq!(
            classify_turn("what's the date today", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Command
        );
    }

    #[test]
    fn open_question_classifies_conversational() {
        assert_eq!(
            classify_turn("how are you feeling today?", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Conversational
        );
        assert_eq!(
            classify_turn("tell me about the weather you like", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Conversational
        );
        assert_eq!(
            classify_turn("why do you think the sky is blue?", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Conversational
        );
    }

    #[test]
    fn short_turn_without_particles_classifies_command() {
        // ≤ 6 words, no conversation particles → command
        assert_eq!(
            classify_turn("louder please", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Command
        );
        assert_eq!(
            classify_turn("stop", COMMAND_MAX_WORDS_DEFAULT),
            TurnKind::Command
        );
    }

    #[test]
    fn long_ambiguous_turn_classifies_conversational() {
        // > 6 words, no command verb or pattern, no particles — default
        assert_eq!(
            classify_turn(
                "the sky was so beautiful this morning I had to take a photo of it",
                COMMAND_MAX_WORDS_DEFAULT
            ),
            TurnKind::Conversational
        );
    }

    // ── apply_routing_policy ──────────────────────────────────────────────────

    fn inputs_online_with_key(transcript: &str) -> PolicyInputs {
        PolicyInputs {
            transcript: transcript.to_string(),
            pending_tier_override: None,
            api_key_present: true,
            online: true,
            prefer: RoutePrefer::Auto,
            command_max_words: COMMAND_MAX_WORDS_DEFAULT,
        }
    }

    #[test]
    fn policy_command_routes_local_even_when_online() {
        // AC4: command utterance → local regardless of key/network
        let dec = apply_routing_policy(&inputs_online_with_key(
            "turn off the kitchen light",
        ));
        assert_eq!(dec.tier, RouteTier::Local);
        assert_eq!(dec.reason, RouteReason::Command);
        assert_eq!(dec.turn_kind, TurnKind::Command);
    }

    #[test]
    fn policy_conversational_online_routes_cloud() {
        // AC5: conversational + online + key → cloud
        let dec = apply_routing_policy(&inputs_online_with_key(
            "how are you feeling today?",
        ));
        assert_eq!(dec.tier, RouteTier::Cloud);
        assert_eq!(dec.reason, RouteReason::Conversational);
    }

    #[test]
    fn policy_no_key_routes_local() {
        // AC3: no API key → local regardless of turn type
        let dec = apply_routing_policy(&PolicyInputs {
            transcript: "how are you feeling today?".to_string(),
            pending_tier_override: None,
            api_key_present: false,
            online: false,
            prefer: RoutePrefer::Auto,
            command_max_words: COMMAND_MAX_WORDS_DEFAULT,
        });
        assert_eq!(dec.tier, RouteTier::Local);
        assert_eq!(dec.reason, RouteReason::NoKey);
    }

    #[test]
    fn policy_offline_routes_local() {
        // Offline conversational → local
        let dec = apply_routing_policy(&PolicyInputs {
            transcript: "tell me about your favorite memory".to_string(),
            pending_tier_override: None,
            api_key_present: true,
            online: false,
            prefer: RoutePrefer::Auto,
            command_max_words: COMMAND_MAX_WORDS_DEFAULT,
        });
        assert_eq!(dec.tier, RouteTier::Local);
        assert_eq!(dec.reason, RouteReason::Offline);
    }

    #[test]
    fn policy_override_beats_everything() {
        // Row 1: pending_tier override routes local regardless of classification
        let dec = apply_routing_policy(&PolicyInputs {
            transcript: "how are you feeling today?".to_string(),
            pending_tier_override: Some("local-3b".to_string()),
            api_key_present: true,
            online: true,
            prefer: RoutePrefer::Auto,
            command_max_words: COMMAND_MAX_WORDS_DEFAULT,
        });
        assert_eq!(dec.tier, RouteTier::Local);
        assert_eq!(dec.reason, RouteReason::Override);
    }

    #[test]
    fn policy_local_only_prefer_overrides_all() {
        // Deployment-wide local-only: even conversational + online + key → local
        let dec = apply_routing_policy(&PolicyInputs {
            transcript: "how are you feeling today?".to_string(),
            pending_tier_override: None,
            api_key_present: true,
            online: true,
            prefer: RoutePrefer::LocalOnly,
            command_max_words: COMMAND_MAX_WORDS_DEFAULT,
        });
        assert_eq!(dec.tier, RouteTier::Local);
        assert_eq!(dec.reason, RouteReason::PreferLocalOnly);
    }

    #[test]
    fn policy_cloud_only_prefer_overrides_command() {
        // Deployment-wide cloud-only: even a command goes to cloud
        let dec = apply_routing_policy(&PolicyInputs {
            transcript: "turn off the kitchen light".to_string(),
            pending_tier_override: None,
            api_key_present: true,
            online: true,
            prefer: RoutePrefer::CloudOnly,
            command_max_words: COMMAND_MAX_WORDS_DEFAULT,
        });
        assert_eq!(dec.tier, RouteTier::Cloud);
        assert_eq!(dec.reason, RouteReason::PreferCloudOnly);
    }

    // ── RoutePrefer parsing ────────────────────────────────────────────────────

    #[test]
    fn route_prefer_parses_all_variants() {
        assert_eq!(RoutePrefer::parse("auto"), Some(RoutePrefer::Auto));
        assert_eq!(RoutePrefer::parse("local-only"), Some(RoutePrefer::LocalOnly));
        assert_eq!(RoutePrefer::parse("cloud-only"), Some(RoutePrefer::CloudOnly));
        assert!(RoutePrefer::parse("invalid").is_none());
    }

    #[test]
    fn route_prefer_as_str_round_trips() {
        for p in [RoutePrefer::Auto, RoutePrefer::LocalOnly, RoutePrefer::CloudOnly] {
            let s = p.as_str();
            assert_eq!(RoutePrefer::parse(s), Some(p));
        }
    }

    // ── canned phrases ────────────────────────────────────────────────────────

    #[test]
    fn canned_phrase_cycles_through_bank() {
        let n = CANNED_PHRASES.len();
        assert!(n >= 3, "need at least 3 phrases in the bank");
        // Cycling should wrap without panicking
        let _ = canned_phrase(0);
        let _ = canned_phrase(n - 1);
        let _ = canned_phrase(n);
        let _ = canned_phrase(1_000);
    }

    #[test]
    fn routing_config_defaults_deserialise_from_empty_section() {
        let cfg: RoutingConfig = toml::from_str("").expect("empty section to default");
        assert_eq!(cfg.local_model, "qwen2.5:3b");
        assert_eq!(cfg.cloud_model, "sonnet");
        assert_eq!(cfg.command_max_words, COMMAND_MAX_WORDS_DEFAULT);
        assert_eq!(cfg.prefer, RoutePrefer::Auto);
        assert_eq!(cfg.cloud_timeout_ms, 6_000);
    }

    #[test]
    fn routing_config_round_trips_through_serde() {
        let cfg = RoutingConfig {
            prefer: RoutePrefer::LocalOnly,
            cloud_timeout_ms: 3_000,
            ..RoutingConfig::default()
        };
        let toml_str = toml::to_string(&cfg).expect("serialises");
        let back: RoutingConfig = toml::from_str(&toml_str).expect("round-trips");
        assert_eq!(cfg, back);
    }
}
