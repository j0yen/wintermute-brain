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

pub mod anthropic;
pub mod persist;
pub use persist::default_config_path;

/// Short model name resolved by [`canonical_model`] to the
/// Sonnet 4.6 model id. PRD §1.2 / §2.6.
pub const SHORT_MODEL_SONNET: &str = "sonnet";

/// Short model name resolved by [`canonical_model`] to the
/// Opus 4.7 model id. PRD §1.2 / §2.6.
pub const SHORT_MODEL_OPUS: &str = "opus";

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

/// Short and canonical model names the daemon accepts on the CLI and
/// in config files. PRD §2.6 promises both Sonnet 4.6 and Opus 4.7.
pub const ALLOWED_MODEL_NAMES: &[&str] = &[
    SHORT_MODEL_SONNET,
    SHORT_MODEL_OPUS,
    "claude-sonnet-4-6",
    "claude-opus-4-7",
];

/// Runtime configuration for `wmd`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
        }
    }
}

fn default_model() -> String {
    DEFAULT_MODEL_NAME.to_string()
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
        SHORT_MODEL_SONNET | "claude-sonnet-4-6" => "claude-sonnet-4-6",
        SHORT_MODEL_OPUS | "claude-opus-4-7" => "claude-opus-4-7",
        _ => "claude-sonnet-4-6",
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

        Ok(Self {
            default_model,
            pending_model: None,
            api_key_env,
            recall_sock,
            user_name,
            timezone,
            child_lock,
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
        let err = validate_model_name("haiku").unwrap_err();
        assert!(matches!(err, BrainError::UnknownModel { ref name, .. } if name == "haiku"));
    }

    #[test]
    fn canonical_model_maps_short_to_long() {
        assert_eq!(canonical_model("sonnet"), "claude-sonnet-4-6");
        assert_eq!(canonical_model("opus"), "claude-opus-4-7");
        assert_eq!(canonical_model("claude-sonnet-4-6"), "claude-sonnet-4-6");
        assert_eq!(canonical_model("claude-opus-4-7"), "claude-opus-4-7");
    }

    #[test]
    fn canonical_model_falls_back_to_sonnet_for_unknown() {
        // Defense in depth — validate_model_name should reject these
        // upstream, but the resolver must not panic on stray strings.
        assert_eq!(canonical_model("haiku"), "claude-sonnet-4-6");
        assert_eq!(canonical_model(""), "claude-sonnet-4-6");
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
            pending_model: Some("haiku".to_string()),
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
}
