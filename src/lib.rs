//! `wintermute-brain` — Claude API conversation loop with recall-backed
//! persistent memory that backs the `wmd` daemon.
//!
//! iter-2 surface: runtime [`BrainConfig`] (with env-var loader),
//! allowed model-name validation, and the [`BrainError`] enum the
//! daemon will raise from config + bus + Anthropic-API paths in later
//! iterations. The agorabus topic schema, the streaming Anthropic
//! client, and the recall-daemon socket bridge land in subsequent
//! iterations per `PRD-wintermute-brain.md` §2.

#![cfg_attr(not(test), forbid(unsafe_code))]

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub mod almanac;
pub mod anthropic;
pub mod bus;
pub mod daemon;
pub mod degrade;
pub mod history;
pub mod ladder;
pub mod persist;
pub mod recall_client;
pub mod session;
pub mod writeback;
pub use persist::default_config_path;

/// Short model name resolved by [`canonical_model`] to the
/// Sonnet 4.6 model id. PRD §1.2 / §2.6.
pub const SHORT_MODEL_SONNET: &str = "sonnet";

/// Short model name resolved by [`canonical_model`] to the
/// Opus 4.8 model id. PRD §1.2 / §2.6.
pub const SHORT_MODEL_OPUS: &str = "opus";

/// Short model name resolved by [`canonical_model`] to the
/// Haiku 4.5 model id. PRD-brain-backend-ladder §2.1 (cheap-cloud floor).
pub const SHORT_MODEL_HAIKU: &str = "haiku";

/// Default chat model when `WM_BRAIN_DEFAULT_MODEL` is unset. PRD §1.2.
pub const DEFAULT_MODEL_NAME: &str = SHORT_MODEL_SONNET;

/// Env var the daemon reads the Anthropic API key from. PRD §2.1.
pub const DEFAULT_API_KEY_ENV: &str = "WM_ANTHROPIC_API_KEY";

/// Default recall-daemon Unix socket path basename, looked up under
/// `$XDG_RUNTIME_DIR` (or `/run/user/<uid>` if that's unset). PRD §2.1.
pub const DEFAULT_RECALL_SOCK_BASENAME: &str = "recall.sock";

/// Recall subject holding her persistent profile facts. PRD §2.7.
pub const PROFILE_SUBJECT: &str = "wintermute-profile";

/// Recall subject prefix for per-day conversation thread memories;
/// `<YYYY-MM-DD>` is appended at write time. PRD §2.7.
pub const THREAD_SUBJECT_PREFIX: &str = "wintermute-thread-";

/// Default config file location relative to `$XDG_CONFIG_HOME`. The
/// daemon persists the per-turn + default-model swaps here. iter-1 log.
pub const DEFAULT_CONFIG_BASENAME: &str = "wintermute/brain.toml";

/// Default number of recent `(user, assistant)` turn pairs retained in the
/// in-memory history ring. `0` disables history (single-message behaviour).
/// PRD-wmd-turn-history §2.3.
pub const DEFAULT_HISTORY_TURNS: usize = 6;

/// Short and canonical model names the daemon accepts on the CLI and
/// in config files. PRD §2.6 promises Sonnet/Opus; the backend-ladder
/// PRD adds the Haiku rung and the local tier names.
pub const ALLOWED_MODEL_NAMES: &[&str] = &[
    SHORT_MODEL_HAIKU,
    SHORT_MODEL_SONNET,
    SHORT_MODEL_OPUS,
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    // Tier names accepted by the ladder switches (swap-model/default-model).
    TIER_LOCAL_3B,
    TIER_LOCAL_8B,
];

/// Built-in tier names, lowest→highest rung. PRD-brain-backend-ladder §2.1.
/// (The cloud rungs `haiku`/`sonnet`/`opus` share their names with the
/// short model ids above, so they are not duplicated here.)
pub const TIER_LOCAL_3B: &str = "local-3b";
/// The 8B local tier name. PRD-brain-backend-ladder §2.1.
pub const TIER_LOCAL_8B: &str = "local-8b";

/// Default ollama model id for the `local-3b` tier. Config-overridable.
pub const DEFAULT_LOCAL_3B_MODEL: &str = "qwen2.5:3b";
/// Default ollama model id for the `local-8b` tier. Config-overridable.
pub const DEFAULT_LOCAL_8B_MODEL: &str = "qwen3:8b";

/// Default OpenAI-compatible endpoint for the local backend (ollama on
/// loopback). PRD-brain-backend-ladder §2.1.
pub const DEFAULT_LOCAL_ENDPOINT: &str = "http://127.0.0.1:11434/v1";

/// Default starting tier name. PRD-brain-backend-ladder §2.1 / AC1.
pub const DEFAULT_TIER_NAME: &str = TIER_LOCAL_3B;

/// Which backend serves a [`Tier`]. PRD-brain-backend-ladder §2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// A local OpenAI-compatible model (ollama via `wm-local-llm`).
    Local,
    /// An Anthropic cloud model (via the existing `AnthropicClient`).
    Anthropic,
}

/// One rung of the brain's tier ladder: a name, the backend that serves
/// it, and the backend-specific model id. PRD-brain-backend-ladder §2.1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tier {
    /// The tier's stable name (e.g. `local-3b`, `haiku`, `sonnet`).
    pub name: String,
    /// Which backend serves this tier.
    pub backend: Backend,
    /// Backend-specific model id: an ollama id (`qwen2.5:3b`) for
    /// [`Backend::Local`], or a short/canonical Anthropic id for
    /// [`Backend::Anthropic`].
    pub model: String,
}

impl Tier {
    /// Construct a tier.
    #[must_use]
    pub fn new(name: impl Into<String>, backend: Backend, model: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            backend,
            model: model.into(),
        }
    }
}

/// Build the default tier ladder (lowest→highest rung).
///
/// Order: `local-3b → local-8b → haiku → sonnet → opus`
/// (PRD-brain-backend-ladder §2.1). The local rungs' endpoints come from
/// `local_endpoint`; the cloud rungs carry short model ids resolved by
/// [`canonical_model`] at request time.
#[must_use]
pub fn default_ladder() -> Vec<Tier> {
    vec![
        Tier::new(TIER_LOCAL_3B, Backend::Local, DEFAULT_LOCAL_3B_MODEL),
        Tier::new(TIER_LOCAL_8B, Backend::Local, DEFAULT_LOCAL_8B_MODEL),
        Tier::new(SHORT_MODEL_HAIKU, Backend::Anthropic, SHORT_MODEL_HAIKU),
        Tier::new(SHORT_MODEL_SONNET, Backend::Anthropic, SHORT_MODEL_SONNET),
        Tier::new(SHORT_MODEL_OPUS, Backend::Anthropic, SHORT_MODEL_OPUS),
    ]
}

/// The lowest *cloud* (Anthropic) tier name in the default ladder.
///
/// Used as the recall-down safe floor for `Ordinary` turns: when recall is
/// unreachable the router's high-stakes detection is blind, so an ordinary
/// turn cannot safely start local. PRD-brain-backend-ladder §2.2.
pub const SAFE_FLOOR_TIER_NAME: &str = SHORT_MODEL_HAIKU;

/// The trusted cloud tier a high-stakes turn starts at (skipping local).
///
/// PRD-brain-backend-ladder §2.2 / `AC3b`.
pub const TRUSTED_CLOUD_TIER_NAME: &str = SHORT_MODEL_SONNET;

/// Runtime configuration for `wmd`.
///
/// `PartialEq` is derived; `Eq` is not (f64 fields prevent it).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainConfig {
    /// Default model id used when no per-turn override is in effect.
    /// Must be in [`ALLOWED_MODEL_NAMES`].
    #[serde(default = "default_model")]
    pub default_model: String,
    /// Per-turn model override, consumed once after `wmd swap-model`
    /// sets it. `None` means "use `default_model`". PRD §2.6.
    #[serde(default)]
    pub pending_model: Option<String>,
    /// Env var name to pull the Anthropic API key from. The daemon
    /// reads this lazily so test runs can avoid leaking key material.
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    /// Recall-daemon Unix socket. PRD §2.1 / §2.7.
    #[serde(default = "default_recall_sock")]
    pub recall_sock: PathBuf,
    /// User display name surfaced in the system prompt. PRD §2.1.
    #[serde(default)]
    pub user_name: Option<String>,
    /// IANA timezone surfaced in the system prompt. PRD §2.1.
    #[serde(default)]
    pub timezone: Option<String>,
    /// When true, the daemon refuses non-`recall.save_fact` /
    /// `recall.search` tool calls — the "child lock" PRD §2.2 mentions.
    #[serde(default)]
    pub child_lock: bool,
    /// Persistent starting tier for the ladder (lowest rung an ordinary
    /// turn begins at). Defaults to [`DEFAULT_TIER_NAME`] (`local-3b`).
    /// PRD-brain-backend-ladder §2.1 / AC1.
    #[serde(default = "default_tier")]
    pub default_tier: String,
    /// Per-turn tier override, consumed once after `wmd swap-model`
    /// sets it — mirrors [`Self::pending_model`] but for tiers.
    /// PRD-brain-backend-ladder §2.3 / AC4.
    #[serde(default)]
    pub pending_tier: Option<String>,
    /// OpenAI-compatible endpoint for the local backend (ollama).
    /// Defaults to [`DEFAULT_LOCAL_ENDPOINT`].
    /// PRD-brain-backend-ladder §2.1.
    #[serde(default = "default_local_endpoint")]
    pub local_endpoint: String,
    /// Number of recent `(user, assistant)` turn pairs retained in the
    /// in-memory rolling history ring. `0` disables history (restores
    /// single-message behaviour). Persisted through `brain.toml`.
    /// PRD-wmd-turn-history §2.3.
    #[serde(default = "default_history_turns")]
    pub history_turns: usize,
    /// When `true` (the default), `wm.almanac.due` envelopes cause wmd to
    /// speak the prompt text via `wm.brain.reply` — the speak-bridge path
    /// described in PRD-almanac-speak-bridge.  Set to `false` (or
    /// `WM_BRAIN_ALMANAC_SPEAK=0`) in tests and developer desks that do
    /// not want live almanac audio.
    #[serde(default = "default_almanac_speak")]
    pub almanac_speak: bool,
    /// When `true`, extracted facts are committed directly to recall
    /// instead of going through the proposal/triage queue.
    /// Defaults to `false` (proposals-by-default — PRD-wmd-memory-writeback §2.3).
    #[serde(default)]
    pub writeback_auto_commit: bool,
    /// Short model name used for the per-session fact-extraction call.
    /// Defaults to `"haiku"` — cheap and sufficient for extraction.
    /// PRD-wmd-memory-writeback §5.
    #[serde(default = "default_writeback_model")]
    pub writeback_model: String,
    /// Minimum confidence a FACT line must carry to be written back.
    /// Lines below this floor are silently dropped.
    /// Defaults to `0.5`. PRD-wmd-memory-writeback §5.
    #[serde(default = "default_writeback_confidence_floor")]
    pub writeback_confidence_floor: f64,
    /// Idle gap in milliseconds after which a new `turn.user` starts a
    /// fresh session. Defaults to `300_000` ms (5 minutes).
    /// PRD-wmd-session-boundary §2.1 / PRD-wmd-memory-writeback §2.2.
    #[serde(default = "default_idle_gap_ms")]
    pub idle_gap_ms: u64,
    /// End-of-conversation phrases that trigger an explicit session close
    /// after the reply is published. Matched case-insensitively with
    /// punctuation stripped. Default: see
    /// [`crate::session::DEFAULT_END_PHRASES`].
    /// PRD-wmd-session-boundary §2.2.
    #[serde(default = "default_session_end_phrases")]
    pub session_end_phrases: Vec<String>,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            pending_model: None,
            api_key_env: default_api_key_env(),
            recall_sock: default_recall_sock(),
            user_name: None,
            timezone: None,
            child_lock: false,
            default_tier: default_tier(),
            pending_tier: None,
            local_endpoint: default_local_endpoint(),
            history_turns: default_history_turns(),
            almanac_speak: default_almanac_speak(),
            writeback_auto_commit: false,
            writeback_model: default_writeback_model(),
            writeback_confidence_floor: default_writeback_confidence_floor(),
            idle_gap_ms: default_idle_gap_ms(),
            session_end_phrases: default_session_end_phrases(),
        }
    }
}

fn default_model() -> String {
    DEFAULT_MODEL_NAME.to_string()
}

fn default_tier() -> String {
    DEFAULT_TIER_NAME.to_string()
}

fn default_local_endpoint() -> String {
    DEFAULT_LOCAL_ENDPOINT.to_string()
}

fn default_history_turns() -> usize {
    DEFAULT_HISTORY_TURNS
}

const fn default_almanac_speak() -> bool {
    true
}

fn default_writeback_model() -> String {
    SHORT_MODEL_HAIKU.to_string()
}

const fn default_writeback_confidence_floor() -> f64 {
    0.5
}

fn default_idle_gap_ms() -> u64 {
    crate::session::DEFAULT_IDLE_GAP_MS
}

fn default_session_end_phrases() -> Vec<String> {
    crate::session::DEFAULT_END_PHRASES
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn default_api_key_env() -> String {
    DEFAULT_API_KEY_ENV.to_string()
}

fn default_recall_sock() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok().filter(|s| !s.is_empty());
    let base = runtime_dir.map_or_else(
        || {
            #[allow(clippy::option_if_let_else, reason = "branch reads more clearly")]
            match nix_uid() {
                Some(uid) => PathBuf::from(format!("/run/user/{uid}")),
                None => PathBuf::from("/tmp"),
            }
        },
        PathBuf::from,
    );
    base.join(DEFAULT_RECALL_SOCK_BASENAME)
}

fn nix_uid() -> Option<u32> {
    std::env::var("UID")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
}

/// Errors raised by config loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum BrainError {
    /// I/O error reading a file from disk.
    #[error("io failure on {path}: {source}")]
    Io {
        /// Path that produced the error.
        path: PathBuf,
        /// Underlying I/O cause.
        #[source]
        source: std::io::Error,
    },
    /// Model name not in [`ALLOWED_MODEL_NAMES`].
    #[error("unknown model {name:?}; allowed: {allowed:?}")]
    UnknownModel {
        /// Offending model name.
        name: String,
        /// Allow-list at the moment of the failure.
        allowed: Vec<String>,
    },
    /// Env var present but unparseable into the expected type.
    #[error("env var {var} has invalid value {value:?}: {reason}")]
    InvalidEnv {
        /// Env var name.
        var: &'static str,
        /// Raw string value seen.
        value: String,
        /// Human-readable parse failure.
        reason: String,
    },
    /// Required runtime resource is missing.
    #[error("missing required resource: {0}")]
    Missing(String),
    /// Config file present but contents could not be deserialised
    /// into a [`BrainConfig`].
    #[error("invalid config at {path}: {reason}")]
    InvalidConfig {
        /// Path to the offending file.
        path: PathBuf,
        /// Human-readable parse failure.
        reason: String,
    },
}

/// Reject model names that are not in [`ALLOWED_MODEL_NAMES`].
///
/// # Errors
/// Returns [`BrainError::UnknownModel`] when `name` is not a member of
/// the allow-list. The error includes a snapshot of the allow-list so
/// the caller can present a useful message.
pub fn validate_model_name(name: &str) -> Result<(), BrainError> {
    if ALLOWED_MODEL_NAMES.contains(&name) {
        Ok(())
    } else {
        Err(BrainError::UnknownModel {
            name: name.to_string(),
            allowed: ALLOWED_MODEL_NAMES.iter().map(|s| (*s).to_string()).collect(),
        })
    }
}

/// Resolve a short or canonical model name to the canonical Anthropic
/// model id used in API requests. iter-2 keeps this trivial; iter-3
/// rewires it once the Anthropic client lands.
#[must_use]
pub fn canonical_model(name: &str) -> &'static str {
    match name {
        SHORT_MODEL_HAIKU | "claude-haiku-4-5" => "claude-haiku-4-5",
        SHORT_MODEL_OPUS | "claude-opus-4-8" => "claude-opus-4-8",
        // Sonnet is the conservative fallback for any unknown / local name
        // (and the explicit sonnet ids): canonical_model only ever feeds an
        // Anthropic request, so a stray string must resolve to a safe cloud
        // id rather than panic.
        _ => "claude-sonnet-4-6",
    }
}

/// Map a legacy `default_model` value to the ladder tier name it should
/// resolve to (back-compat for configs predating the tier ladder).
/// Returns `None` for values that don't name a cloud tier.
#[must_use]
fn legacy_model_to_tier(model: &str) -> Option<&'static str> {
    match model {
        SHORT_MODEL_HAIKU | "claude-haiku-4-5" => Some(SHORT_MODEL_HAIKU),
        SHORT_MODEL_SONNET | "claude-sonnet-4-6" => Some(SHORT_MODEL_SONNET),
        SHORT_MODEL_OPUS | "claude-opus-4-8" => Some(SHORT_MODEL_OPUS),
        _ => None,
    }
}

impl BrainConfig {
    /// Build a config from environment variables, falling back to
    /// defaults for any unset variable.
    ///
    /// Recognised vars:
    /// - `WM_BRAIN_DEFAULT_MODEL` (string; default `sonnet`)
    /// - `WM_BRAIN_API_KEY_ENV` (string; default `WM_ANTHROPIC_API_KEY`)
    /// - `WM_BRAIN_RECALL_SOCK` (path; default `$XDG_RUNTIME_DIR/recall.sock`)
    /// - `WM_USER_NAME` (string; optional)
    /// - `WM_TIMEZONE` (IANA tz string; optional)
    /// - `WM_BRAIN_CHILD_LOCK` (`true`/`false`; default `false`)
    /// - `WM_BRAIN_ALMANAC_SPEAK` (`true`/`false`; default `true`)
    ///
    /// # Errors
    /// Returns [`BrainError::InvalidEnv`] when a var is set but
    /// unparseable, or [`BrainError::UnknownModel`] if the parsed model
    /// name fails [`validate_model_name`].
    pub fn from_env() -> Result<Self, BrainError> {
        let default_model = env_string("WM_BRAIN_DEFAULT_MODEL").unwrap_or_else(default_model);
        validate_model_name(&default_model)?;

        let api_key_env = env_string("WM_BRAIN_API_KEY_ENV").unwrap_or_else(default_api_key_env);

        let recall_sock = env_string("WM_BRAIN_RECALL_SOCK")
            .map_or_else(default_recall_sock, PathBuf::from);

        let user_name = env_string("WM_USER_NAME");
        let timezone = env_string("WM_TIMEZONE");

        let child_lock = match env_string("WM_BRAIN_CHILD_LOCK") {
            Some(raw) => parse_bool_env("WM_BRAIN_CHILD_LOCK", &raw)?,
            None => false,
        };

        let almanac_speak = match env_string("WM_BRAIN_ALMANAC_SPEAK") {
            Some(raw) => parse_bool_env("WM_BRAIN_ALMANAC_SPEAK", &raw)?,
            None => default_almanac_speak(),
        };

        let default_tier = env_string("WM_BRAIN_DEFAULT_TIER").unwrap_or_else(default_tier);
        let local_endpoint =
            env_string("WM_BRAIN_LOCAL_ENDPOINT").unwrap_or_else(default_local_endpoint);

        let writeback_auto_commit = match env_string("WM_BRAIN_WRITEBACK_AUTO_COMMIT") {
            Some(raw) => parse_bool_env("WM_BRAIN_WRITEBACK_AUTO_COMMIT", &raw)?,
            None => false,
        };
        let writeback_model =
            env_string("WM_BRAIN_WRITEBACK_MODEL").unwrap_or_else(default_writeback_model);
        let writeback_confidence_floor =
            match env_string("WM_BRAIN_WRITEBACK_CONFIDENCE_FLOOR") {
                Some(raw) => raw.trim().parse::<f64>().map_err(|e| BrainError::InvalidEnv {
                    var: "WM_BRAIN_WRITEBACK_CONFIDENCE_FLOOR",
                    value: raw.clone(),
                    reason: e.to_string(),
                })?,
                None => default_writeback_confidence_floor(),
            };
        let idle_gap_ms = match env_string("WM_BRAIN_IDLE_GAP_MS") {
            Some(raw) => raw.trim().parse::<u64>().map_err(|e| BrainError::InvalidEnv {
                var: "WM_BRAIN_IDLE_GAP_MS",
                value: raw.clone(),
                reason: e.to_string(),
            })?,
            None => default_idle_gap_ms(),
        };

        Ok(Self {
            default_model,
            pending_model: None,
            api_key_env,
            recall_sock,
            user_name,
            timezone,
            child_lock,
            almanac_speak,
            default_tier,
            pending_tier: None,
            local_endpoint,
            history_turns: default_history_turns(),
            writeback_auto_commit,
            writeback_model,
            writeback_confidence_floor,
            idle_gap_ms,
            session_end_phrases: default_session_end_phrases(),
        })
    }

    /// Validate an already-constructed config. Used after deserialising
    /// from a config file or after CLI-flag overrides.
    ///
    /// # Errors
    /// Forwards from [`validate_model_name`] for both `default_model`
    /// and (if set) `pending_model`.
    pub fn validate(&self) -> Result<(), BrainError> {
        validate_model_name(&self.default_model)?;
        if let Some(p) = &self.pending_model {
            validate_model_name(p)?;
        }
        validate_model_name(&self.default_tier)?;
        if let Some(t) = &self.pending_tier {
            validate_model_name(t)?;
        }
        Ok(())
    }

    /// Resolve the effective model id for the next API turn.
    ///
    /// Consumes `pending_model` semantically: a follow-up call to
    /// [`Self::consume_pending`] should be invoked after the turn
    /// completes. iter-2 separates resolution from consumption so the
    /// daemon can probe the model id without mutating state when
    /// previewing a request.
    #[must_use]
    pub fn effective_model(&self) -> &str {
        self.pending_model
            .as_deref()
            .unwrap_or(self.default_model.as_str())
    }

    /// Clear the per-turn override after a successful API call.
    pub fn consume_pending(&mut self) {
        self.pending_model = None;
    }

    /// Resolve the *persistent* starting-tier name.
    ///
    /// This is simply `default_tier`; back-compat for legacy configs that
    /// only ever set `default_model` (and carry no `default_tier` key) is
    /// applied at file-load time by [`Self::apply_legacy_tier_backcompat`].
    /// PRD-brain-backend-ladder AC1.
    #[must_use]
    pub fn resolved_default_tier(&self) -> String {
        self.default_tier.clone()
    }

    /// Back-compat shim for configs written before the tier ladder: when a
    /// loaded file carried a `default_model` naming a cloud tier but had no
    /// `default_tier` key (so serde filled it with the built-in default),
    /// migrate `default_tier` to the matching cloud tier so the upgraded
    /// brain keeps starting where it used to. `had_default_tier_key` is
    /// true when the raw TOML contained an explicit `default_tier`.
    /// PRD-brain-backend-ladder AC1.
    fn apply_legacy_tier_backcompat(&mut self, had_default_tier_key: bool) {
        if had_default_tier_key {
            return;
        }
        if self.default_tier != DEFAULT_TIER_NAME {
            return;
        }
        if let Some(tier) = legacy_model_to_tier(&self.default_model) {
            self.default_tier = tier.to_string();
        }
    }

    /// Resolve the effective starting-tier name for the next turn, applying
    /// `pending_tier` (one-shot) over [`Self::resolved_default_tier`].
    /// PRD-brain-backend-ladder §2.3.
    #[must_use]
    pub fn effective_tier(&self) -> String {
        self.pending_tier
            .clone()
            .unwrap_or_else(|| self.resolved_default_tier())
    }

    /// Clear the per-turn tier override after a turn consumes it.
    pub fn consume_pending_tier(&mut self) {
        self.pending_tier = None;
    }

    /// Resolve the today's-thread recall subject for an ISO-formatted
    /// date string like `2026-05-26`.
    #[must_use]
    pub fn thread_subject_for(date: &str) -> String {
        format!("{THREAD_SUBJECT_PREFIX}{date}")
    }
}

fn env_string(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|v| !v.is_empty())
}

fn parse_bool_env(var: &'static str, raw: &str) -> Result<bool, BrainError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(BrainError::InvalidEnv {
            var,
            value: raw.to_string(),
            reason: "expected one of: 1/0, true/false, yes/no, on/off".to_string(),
        }),
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

    #[test]
    fn default_config_validates() {
        let cfg = BrainConfig::default();
        cfg.validate().expect("defaults are valid");
        assert_eq!(cfg.default_model, DEFAULT_MODEL_NAME);
        assert_eq!(cfg.api_key_env, DEFAULT_API_KEY_ENV);
        assert!(cfg.pending_model.is_none());
        assert!(!cfg.child_lock);
    }

    #[test]
    fn validate_model_name_accepts_all_allowed() {
        for name in ALLOWED_MODEL_NAMES {
            validate_model_name(name).expect("allow-list entries validate");
        }
    }

    #[test]
    fn validate_model_name_rejects_unknown() {
        // `haiku` is now an ACCEPTED tier (PRD-brain-backend-ladder §2.2);
        // a genuinely unknown name still rejects.
        let err = validate_model_name("gpt-4o").unwrap_err();
        assert!(matches!(err, BrainError::UnknownModel { ref name, .. } if name == "gpt-4o"));
    }

    #[test]
    fn validate_model_name_accepts_haiku_and_tier_names() {
        // PRD-brain-backend-ladder §2.2: haiku must be accepted, as must the
        // local tier names used by swap-model/default-model.
        for name in [SHORT_MODEL_HAIKU, TIER_LOCAL_3B, TIER_LOCAL_8B, "claude-haiku-4-5"] {
            validate_model_name(name).expect("tier name validates");
        }
    }

    #[test]
    fn canonical_model_maps_short_to_long() {
        assert_eq!(canonical_model("sonnet"), "claude-sonnet-4-6");
        // PRD-brain-backend-ladder §2.2: current Opus is 4.8, not 4.7.
        assert_eq!(canonical_model("opus"), "claude-opus-4-8");
        assert_eq!(canonical_model("haiku"), "claude-haiku-4-5");
        assert_eq!(canonical_model("claude-sonnet-4-6"), "claude-sonnet-4-6");
        assert_eq!(canonical_model("claude-opus-4-8"), "claude-opus-4-8");
        assert_eq!(canonical_model("claude-haiku-4-5"), "claude-haiku-4-5");
    }

    #[test]
    fn canonical_model_falls_back_to_sonnet_for_unknown() {
        // Defense in depth — local tier names are never fed to an Anthropic
        // request, but the resolver must not panic on stray strings.
        assert_eq!(canonical_model("local-3b"), "claude-sonnet-4-6");
        assert_eq!(canonical_model(""), "claude-sonnet-4-6");
    }

    #[test]
    fn default_tier_is_local_3b() {
        // AC1: default starting tier is local-3b even though default_model
        // still defaults to sonnet (the cloud rung used when a turn lands
        // on Anthropic). Back-compat for legacy files is tested in persist.
        let cfg = BrainConfig::default();
        assert_eq!(cfg.resolved_default_tier(), TIER_LOCAL_3B);
        assert_eq!(cfg.effective_tier(), TIER_LOCAL_3B);
    }

    #[test]
    fn legacy_tier_backcompat_maps_default_model() {
        // AC1 back-compat: a file carrying only default_model = sonnet (no
        // default_tier key) migrates to the sonnet tier; a file with an
        // explicit default_tier keeps it.
        let mut legacy = BrainConfig {
            default_model: SHORT_MODEL_SONNET.to_string(),
            default_tier: DEFAULT_TIER_NAME.to_string(),
            ..BrainConfig::default()
        };
        legacy.apply_legacy_tier_backcompat(false);
        assert_eq!(legacy.default_tier, SHORT_MODEL_SONNET);

        let mut explicit = BrainConfig {
            default_model: SHORT_MODEL_SONNET.to_string(),
            default_tier: TIER_LOCAL_8B.to_string(),
            ..BrainConfig::default()
        };
        explicit.apply_legacy_tier_backcompat(true);
        assert_eq!(explicit.default_tier, TIER_LOCAL_8B);
    }

    #[test]
    fn pending_tier_overrides_default_then_consumed() {
        // AC4: swap-model sets a one-shot tier override.
        let mut cfg = BrainConfig {
            pending_tier: Some(TIER_LOCAL_8B.to_string()),
            ..BrainConfig::default()
        };
        assert_eq!(cfg.effective_tier(), TIER_LOCAL_8B);
        cfg.consume_pending_tier();
        assert!(cfg.pending_tier.is_none());
        assert_eq!(cfg.effective_tier(), TIER_LOCAL_3B);
    }

    #[test]
    fn default_ladder_is_five_rungs_lowest_to_highest() {
        // AC1: built-in ladder local-3b -> local-8b -> haiku -> sonnet -> opus.
        let ladder = default_ladder();
        let summary: Vec<(&str, Backend, &str)> = ladder
            .iter()
            .map(|t| (t.name.as_str(), t.backend, t.model.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (TIER_LOCAL_3B, Backend::Local, DEFAULT_LOCAL_3B_MODEL),
                (TIER_LOCAL_8B, Backend::Local, DEFAULT_LOCAL_8B_MODEL),
                ("haiku", Backend::Anthropic, "haiku"),
                ("sonnet", Backend::Anthropic, "sonnet"),
                ("opus", Backend::Anthropic, "opus"),
            ]
        );
    }

    #[test]
    fn effective_model_prefers_pending() {
        let mut cfg = BrainConfig::default();
        assert_eq!(cfg.effective_model(), DEFAULT_MODEL_NAME);
        cfg.pending_model = Some(SHORT_MODEL_OPUS.to_string());
        assert_eq!(cfg.effective_model(), SHORT_MODEL_OPUS);
        cfg.consume_pending();
        assert!(cfg.pending_model.is_none());
        assert_eq!(cfg.effective_model(), DEFAULT_MODEL_NAME);
    }

    #[test]
    fn parse_bool_env_accepts_synonyms() {
        for raw in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(parse_bool_env("WM_TEST", raw).expect("parses"));
        }
        for raw in ["0", "false", "no", "off"] {
            assert!(!parse_bool_env("WM_TEST", raw).expect("parses"));
        }
        assert!(matches!(
            parse_bool_env("WM_TEST", "maybe"),
            Err(BrainError::InvalidEnv { .. })
        ));
    }

    #[test]
    fn thread_subject_includes_date() {
        let s = BrainConfig::thread_subject_for("2026-05-26");
        assert_eq!(s, "wintermute-thread-2026-05-26");
        assert!(s.starts_with(THREAD_SUBJECT_PREFIX));
    }

    #[test]
    fn validate_rejects_invalid_pending_model() {
        let cfg = BrainConfig {
            pending_model: Some("gpt-4o".to_string()),
            ..BrainConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(BrainError::UnknownModel { .. })
        ));
    }

    #[test]
    fn round_trip_serde_json() {
        let cfg = BrainConfig {
            pending_model: Some(SHORT_MODEL_OPUS.to_string()),
            user_name: Some("Mom".to_string()),
            ..BrainConfig::default()
        };
        let v = serde_json::to_value(&cfg).expect("serialises");
        let back: BrainConfig = serde_json::from_value(v).expect("round-trips");
        assert_eq!(cfg, back);
    }

    // ── almanac_speak config gate (PRD AC3 / AC5) ────────────────────────────

    #[test]
    fn default_almanac_speak_is_true() {
        // AC5: default BrainConfig has almanac_speak=true.
        let cfg = BrainConfig::default();
        assert!(cfg.almanac_speak, "almanac_speak must default to true");
    }

    #[test]
    fn almanac_speak_round_trips_through_serde() {
        // AC5: almanac_speak round-trips through serde.
        let cfg = BrainConfig {
            almanac_speak: false,
            ..BrainConfig::default()
        };
        let v = serde_json::to_value(&cfg).expect("serialises");
        let back: BrainConfig = serde_json::from_value(v).expect("round-trips");
        assert!(!back.almanac_speak);
        // Also verify true round-trips.
        let cfg2 = BrainConfig::default();
        let v2 = serde_json::to_value(&cfg2).expect("serialises");
        let back2: BrainConfig = serde_json::from_value(v2).expect("round-trips");
        assert!(back2.almanac_speak);
    }

    #[test]
    fn env_override_wm_brain_almanac_speak_false() {
        // AC5: WM_BRAIN_ALMANAC_SPEAK=0 disables speak.
        // We exercise parse_bool_env directly (env mutation in tests is messy).
        let parsed = parse_bool_env("WM_BRAIN_ALMANAC_SPEAK", "0").expect("parses");
        assert!(!parsed);
        let parsed_false = parse_bool_env("WM_BRAIN_ALMANAC_SPEAK", "false").expect("parses");
        assert!(!parsed_false);
        let parsed_true = parse_bool_env("WM_BRAIN_ALMANAC_SPEAK", "1").expect("parses");
        assert!(parsed_true);
    }
}
