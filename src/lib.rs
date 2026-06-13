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

// ── PersonaConfig ─────────────────────────────────────────────────────────────

/// Named register presets controlling the tone and vocabulary of the
/// composed base persona.
///
/// `WarmElder` is the calibrated default — plain, warm prose tuned for a
/// non-technical elder listener. `Plain` reproduces the original
/// `DEFAULT_PERSONA` verbatim (no behaviour change). `Brisk` is a terse
/// professional register for developer/power-user sessions.
///
/// PRD-hearth-persona-config §2.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Register {
    /// Warm, patient prose with short sentences and plain words.
    /// Calibrated for a non-technical elder listener (the target user).
    #[default]
    WarmElder,
    /// Mirrors the original hard-coded `DEFAULT_PERSONA` const.
    /// Byte-identical base; zero behaviour change when selected.
    Plain,
    /// Terse, direct prose for developer / power-user sessions.
    Brisk,
}

impl Register {
    /// Compose the register-specific base prose, substituting
    /// `{self_name}` and `{user_name}` tokens.
    ///
    /// When `user_name` is `None` or `addresses_user` is `false`, the
    /// user clause is omitted cleanly — no literal `{user_name}` leaks
    /// into the result.
    #[must_use]
    pub fn compose_base(
        self,
        self_name: &str,
        user_name: Option<&str>,
        addresses_user: bool,
        max_sentences: u8,
    ) -> String {
        match self {
            Self::WarmElder => {
                let user_clause = if addresses_user {
                    user_name
                        .filter(|n| !n.is_empty())
                        .map(|n| format!(" who speaks with {n}"))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                let brevity_clause = if max_sentences > 0 {
                    format!(
                        " Keep replies to {max_sentences} sentence{} or fewer unless asked for more.",
                        if max_sentences == 1 { "" } else { "s" }
                    )
                } else {
                    String::new()
                };
                format!(
                    "You are {self_name}, a kind companion{user_clause} \
aloud — never on a screen. Talk like a warm, patient person: short \
sentences, plain everyday words, one thought at a time. Avoid all \
computer or engineering jargon; if something is wrong, describe it \
the way a caring friend would, not a technician.{brevity_clause} \
No markdown, lists, code, or emoji — they do not speak well."
                )
            }
            Self::Plain => {
                // Byte-identical to the original DEFAULT_PERSONA (AC3).
                // Ignores most persona fields so existing behaviour is
                // fully preserved when this register is selected.
                crate::daemon::DEFAULT_PERSONA.to_string()
            }
            Self::Brisk => {
                let user_clause = if addresses_user {
                    user_name
                        .filter(|n| !n.is_empty())
                        .map(|n| format!(" assisting {n}"))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                let brevity_clause = if max_sentences > 0 {
                    format!(
                        " Limit replies to {max_sentences} sentence{} unless instructed otherwise.",
                        if max_sentences == 1 { "" } else { "s" }
                    )
                } else {
                    String::new()
                };
                format!(
                    "You are {self_name}, a concise voice-first assistant{user_clause}. \
Speak in plain prose — no markdown, bullets, or emoji.{brevity_clause}"
                )
            }
        }
    }
}

fn default_self_name() -> String {
    "wintermute".to_string()
}

fn default_wake_word() -> String {
    "hey wintermute".to_string()
}

const fn default_addresses_user() -> bool {
    true
}

const fn default_max_sentences() -> u8 {
    3
}

/// Configurable persona settings stored in the `[persona]` table of
/// `brain.toml`.
///
/// All fields are `#[serde(default)]` so existing `brain.toml` files
/// without a `[persona]` section deserialise cleanly to
/// [`PersonaConfig::default`].
///
/// PRD-hearth-persona-config §2.1.
/// PRD-hearth-first-contact-greeting §2.3 adds `greeting` + `wake_word`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersonaConfig {
    /// What the companion calls herself out loud.
    #[serde(default = "default_self_name")]
    pub self_name: String,
    /// Named tone/vocabulary preset. Default: [`Register::WarmElder`].
    #[serde(default)]
    pub register: Register,
    /// When `true` (the default), the user's name is woven into the
    /// composed base when `user_name` is set. Set `false` to omit.
    #[serde(default = "default_addresses_user")]
    pub addresses_user: bool,
    /// Soft spoken-reply brevity ceiling. `0` means no ceiling.
    #[serde(default = "default_max_sentences")]
    pub max_sentences: u8,
    /// Free-form text appended verbatim to the composed base persona.
    /// Useful for operator-specific instructions that don't fit a named
    /// register.
    #[serde(default)]
    pub extra: String,
    /// Controls whether (and when) the daemon emits a proactive boot
    /// greeting before the user's first turn. Default:
    /// [`greeting::GreetingMode::Off`] (conservative — no proactive
    /// speech unless the deployment opts in).
    ///
    /// PRD-hearth-first-contact-greeting §2.3.
    #[serde(default)]
    pub greeting: greeting::GreetingMode,
    /// The voice wake word surfaced in the
    /// [`greeting::GreetingKind::FirstEver`] teaching block.
    /// Defaults to `"hey wintermute"`. The deployment should override
    /// this when the wake word differs.
    ///
    /// PRD-hearth-first-contact-greeting §2.2.
    #[serde(default = "default_wake_word")]
    pub wake_word: String,
    /// Words or phrases the companion must never utter in any form.
    /// When non-empty, an avoidance instruction is appended to the composed
    /// base persona. Useful for deployments where tech jargon would break
    /// the illusion (e.g. Jocelyn elder-companion preset).
    ///
    /// PRD-persona-forbidden-vocab §2.1.
    #[serde(default)]
    pub forbidden_terms: Vec<String>,
    /// Controls whether (and when) the daemon speaks a name-introduction
    /// ceremony before the regular first-ever boot greeting.
    ///
    /// Default is [`introduction::IntroductionMode::Off`] (no ceremony).
    /// Set to [`introduction::IntroductionMode::FirstEverBoot`] to fire once
    /// on the companion's very first boot, or
    /// [`introduction::IntroductionMode::Explicit`] to fire on demand via
    /// `wm.persona.introduce`.
    ///
    /// PRD-persona-name-ceremony §2.1.
    #[serde(default)]
    pub introduction: introduction::IntroductionMode,
}

impl Default for PersonaConfig {
    fn default() -> Self {
        Self {
            self_name: default_self_name(),
            register: Register::default(),
            addresses_user: default_addresses_user(),
            max_sentences: default_max_sentences(),
            extra: String::new(),
            greeting: greeting::GreetingMode::default(),
            wake_word: default_wake_word(),
            forbidden_terms: Vec::new(),
            introduction: introduction::IntroductionMode::default(),
        }
    }
}

impl PersonaConfig {
    /// Compose the stable persona base from this config and an optional
    /// `user_name` (pulled from [`BrainConfig::user_name`] by the caller).
    ///
    /// The returned string is the **stable cache prefix** — it must be
    /// byte-stable across consecutive calls with the same config so prompt
    /// caching is effective (PRD §2.3).
    ///
    /// The per-turn recall/recap context is spliced *after* this prefix by
    /// [`crate::daemon::compose_persona`].
    #[must_use]
    pub fn compose_base(&self, user_name: Option<&str>) -> String {
        let mut base = self.register.compose_base(
            &self.self_name,
            user_name,
            self.addresses_user,
            self.max_sentences,
        );
        if !self.extra.is_empty() {
            base.push_str("\n\n");
            base.push_str(&self.extra);
        }
        if !self.forbidden_terms.is_empty() {
            let joined = self.forbidden_terms.join(", ");
            base.push_str("\n\nYou must never use the following words or phrases, in any form: ");
            base.push_str(&joined);
            base.push_str(". When you would naturally say one of these, rephrase with plain, warm language instead.");
        }
        base
    }
}

pub mod almanac;
pub mod anthropic;
pub mod bus;
pub mod cache;
pub mod daemon;
pub mod degrade;
pub mod greeting;
pub mod introduction;
pub mod history;
pub mod ladder;
pub mod persist;
pub mod recall_client;
pub mod repair;
pub mod router;
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
    TIER_LOCAL_GPU,
    TIER_LOCAL_LLM,
];

/// Built-in tier names, lowest→highest rung. PRD-brain-backend-ladder §2.1.
/// (The cloud rungs `haiku`/`sonnet`/`opus` share their names with the
/// short model ids above, so they are not duplicated here.)
pub const TIER_LOCAL_3B: &str = "local-3b";
/// The 8B local tier name. PRD-brain-backend-ladder §2.1.
pub const TIER_LOCAL_8B: &str = "local-8b";
/// The GPU-accelerated local tier (desktop Radeon via llama.cpp Vulkan).
///
/// Sits between `local-8b` and the cloud tiers. Enabled only when
/// `WM_BRAIN_GPU_ENDPOINT` is set; otherwise the tier is absent from the
/// ladder and the brain falls through to cloud exactly as today.
/// PRD-constellation-brain-gpu AC5 / AC6.
pub const TIER_LOCAL_GPU: &str = "local-gpu";

/// The dedicated local-LLM tier (Ryzen 7 5700U node running llama-server).
///
/// Sits between `local-8b` and the cloud tiers, adjacent to (or in place
/// of) `local-gpu`. Enabled only when `WM_BRAIN_LOCAL_LLM_ENDPOINT` is set;
/// otherwise the tier is absent from the ladder and the brain falls through
/// to cloud exactly as today (graceful absence, AC6 pattern).
///
/// This is the privacy/offline/zero-marginal-cost tier — a dedicated
/// qwen2.5-8B node on the fleet. It is honestly *not* the latency brain
/// (~8–10 tok/s on a 5700U APU vs sub-2s from cloud); the Anthropic API
/// stays the latency brain. PRD-constellation-brain-local AC5/AC6.
pub const TIER_LOCAL_LLM: &str = "local-llm";

/// Default ollama model id for the `local-3b` tier. Config-overridable.
pub const DEFAULT_LOCAL_3B_MODEL: &str = "qwen2.5:3b";
/// Default ollama model id for the `local-8b` tier. Config-overridable.
pub const DEFAULT_LOCAL_8B_MODEL: &str = "qwen3:8b";
/// Default model id served by the GPU host's `llama-server`.
///
/// Overridable via `WM_BRAIN_GPU_MODEL`. `Q4_K_M` is recommended for
/// ≥8 GB VRAM; 14B variants are selectable for ≥12 GB VRAM hosts.
/// PRD-constellation-brain-gpu §2.
pub const DEFAULT_LOCAL_GPU_MODEL: &str = "qwen2.5-7b-instruct:Q4_K_M";

/// Default model id served by the `local-llm` node's `llama-server`
/// (the Ryzen 7 5700U APU, no discrete GPU).
///
/// qwen2.5-8B-Instruct Q4_K_M fits in the ~20 GB headroom after the OS
/// and is expected to produce ~8–10 tok/s on this CPU-only box.
/// Overridable via `WM_BRAIN_LOCAL_LLM_MODEL`.
/// PRD-constellation-brain-local §2.
pub const DEFAULT_LOCAL_LLM_MODEL: &str = "qwen2.5-8b-instruct:Q4_K_M";

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
    /// Per-tier OpenAI-compatible endpoint override for local tiers.
    ///
    /// When `Some`, this endpoint is used instead of `BrainConfig::local_endpoint`
    /// for [`Backend::Local`] tiers. Used by the `local-gpu` tier to point at
    /// the desktop's `llama-server` on its own endpoint.
    /// PRD-constellation-brain-gpu §2 (brain ladder integration).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_override: Option<String>,
}

impl Tier {
    /// Construct a tier with no endpoint override (uses the default local endpoint).
    #[must_use]
    pub fn new(name: impl Into<String>, backend: Backend, model: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            backend,
            model: model.into(),
            endpoint_override: None,
        }
    }

    /// Construct a tier with an explicit endpoint override.
    ///
    /// Used for [`TIER_LOCAL_GPU`] which points at the desktop `llama-server`
    /// rather than the local ollama.
    #[must_use]
    pub fn with_endpoint(
        name: impl Into<String>,
        backend: Backend,
        model: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            backend,
            model: model.into(),
            endpoint_override: Some(endpoint.into()),
        }
    }
}

/// Build the default tier ladder (lowest→highest rung).
///
/// Order: `local-3b → local-8b → haiku → sonnet → opus`
/// (PRD-brain-backend-ladder §2.1). The local rungs' endpoints come from
/// `local_endpoint`; the cloud rungs carry short model ids resolved by
/// [`canonical_model`] at request time.
///
/// The `local-gpu` tier is NOT included here; it is injected by
/// [`build_ladder_with_gpu`] when `WM_BRAIN_GPU_ENDPOINT` is set.
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

/// Build the tier ladder, optionally inserting a `local-gpu` rung.
///
/// When `gpu_endpoint` is `Some`, a [`TIER_LOCAL_GPU`] rung is inserted
/// between `local-8b` and the cloud tiers. When `None` (the GPU endpoint
/// env var is unset or empty), the ladder is identical to [`default_ladder`].
///
/// Graceful absence: if the GPU endpoint is unreachable at turn time,
/// [`crate::ladder::LiveLocalBackend`] maps the failure to
/// [`wm_local_llm::LocalOutcome::Escalate`] and the ladder climbs to the
/// next rung (haiku) without any turn failure. PRD-constellation-brain-gpu
/// AC6.
#[must_use]
pub fn ladder_with_gpu(gpu_endpoint: Option<&str>, gpu_model: &str) -> Vec<Tier> {
    let mut tiers = default_ladder();
    if let Some(endpoint) = gpu_endpoint {
        if !endpoint.is_empty() {
            // Insert after local-8b (index 1), before haiku (index 2).
            let insert_pos = tiers
                .iter()
                .position(|t| t.backend == Backend::Anthropic)
                .unwrap_or(tiers.len());
            tiers.insert(
                insert_pos,
                Tier::with_endpoint(TIER_LOCAL_GPU, Backend::Local, gpu_model, endpoint),
            );
        }
    }
    tiers
}

/// Build the tier ladder, optionally inserting a `local-llm` rung.
///
/// When `local_llm_endpoint` is `Some`, a [`TIER_LOCAL_LLM`] rung is inserted
/// between `local-8b` (or `local-gpu` when present) and the cloud tiers.
/// When `None` (the env var is unset or empty), the ladder is identical to
/// its input.
///
/// This function composes with [`ladder_with_gpu`]: call `ladder_with_gpu`
/// first, then pass the result to this function (or compose in either order —
/// the insertion logic finds the first Anthropic rung and inserts before it).
///
/// Graceful absence: if the local-llm endpoint is unreachable at turn time,
/// [`crate::ladder::LiveLocalBackend`] maps the failure to
/// [`wm_local_llm::LocalOutcome::Escalate`] and the ladder climbs to the next
/// rung without any turn failure. PRD-constellation-brain-local AC6.
#[must_use]
pub fn ladder_with_local_llm(
    base: Vec<Tier>,
    local_llm_endpoint: Option<&str>,
    local_llm_model: &str,
) -> Vec<Tier> {
    let mut tiers = base;
    if let Some(endpoint) = local_llm_endpoint {
        if !endpoint.is_empty() {
            // Insert before the first Anthropic (cloud) rung, so local-llm
            // sits after all local rungs (3b, 8b, gpu if present) and before haiku.
            let insert_pos = tiers
                .iter()
                .position(|t| t.backend == Backend::Anthropic)
                .unwrap_or(tiers.len());
            tiers.insert(
                insert_pos,
                Tier::with_endpoint(TIER_LOCAL_LLM, Backend::Local, local_llm_model, endpoint),
            );
        }
    }
    tiers
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
#[allow(
    clippy::struct_excessive_bools,
    reason = "BrainConfig fields are independent feature gates that map naturally to bools; \
              grouping them into bitfields or enums would obscure their meaning and serde shape"
)]
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
    /// Optional OpenAI-compatible endpoint for the GPU-accelerated local tier
    /// (`local-gpu`). When set (or env var `WM_BRAIN_GPU_ENDPOINT` is set),
    /// a `local-gpu` rung is inserted between `local-8b` and the cloud tiers.
    /// When absent, the tier is not present and the ladder falls through to
    /// cloud exactly as today (graceful absence, AC6).
    /// PRD-constellation-brain-gpu §2.
    #[serde(default)]
    pub gpu_endpoint: Option<String>,
    /// Model id served by the GPU host's `llama-server`. Defaults to
    /// [`DEFAULT_LOCAL_GPU_MODEL`]. Override via `WM_BRAIN_GPU_MODEL` or
    /// `brain.toml` `gpu_model = "…"`.
    /// PRD-constellation-brain-gpu §2.
    #[serde(default = "default_gpu_model")]
    pub gpu_model: String,
    /// Optional OpenAI-compatible endpoint for the dedicated `local-llm` tier
    /// (the Ryzen 7 5700U node running llama-server). When set (or env var
    /// `WM_BRAIN_LOCAL_LLM_ENDPOINT` is set), a `local-llm` rung is inserted
    /// between `local-8b` and the cloud tiers (and after `local-gpu` when that
    /// is also present). When absent, the tier is not present and the ladder
    /// falls through to cloud exactly as today (graceful absence).
    ///
    /// This tier is the privacy/offline/cheap-default brain — NOT the latency
    /// brain (~8–10 tok/s on the APU). PRD-constellation-brain-local §2 / AC5/AC6.
    #[serde(default)]
    pub local_llm_endpoint: Option<String>,
    /// Model id served by the `local-llm` node's llama-server. Defaults to
    /// [`DEFAULT_LOCAL_LLM_MODEL`]. Override via `WM_BRAIN_LOCAL_LLM_MODEL`
    /// or `brain.toml` `local_llm_model = "…"`.
    /// PRD-constellation-brain-local §2.
    #[serde(default = "default_local_llm_model")]
    pub local_llm_model: String,
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
    /// Upper bound on thread memories injected at session start.
    /// Most-recent first. `0` disables recap injection entirely.
    /// PRD-wmd-session-recap §2.1.
    #[serde(default = "default_recap_max_memories")]
    pub recap_max_memories: usize,
    /// When `true`, wmd publishes a proactive `wm.brain.reply` before
    /// the user's first turn, opening with a continuity greeting drawn
    /// from the recap context. Default `false` (conservative; see PRD §2.3).
    /// PRD-wmd-session-recap §2.3.
    #[serde(default)]
    pub recap_opener: bool,
    /// Phrases that trigger verbatim replay of the last assistant turn without
    /// a new LLM call. Matched case-insensitively with punctuation stripped.
    /// An empty list uses [`crate::repair::DEFAULT_REPEAT_PHRASES`].
    /// PRD-wmd-repair-affordances §2.1.
    #[serde(default)]
    pub repair_repeat_phrases: Vec<String>,
    /// Phrases that trigger verbatim replay of the last assistant turn with a
    /// `loudness = "loud"` hint on the emitted [`crate::bus::ReplyEvent`].
    /// Matched case-insensitively with punctuation stripped.
    /// An empty list uses [`crate::repair::DEFAULT_LOUDER_PHRASES`].
    /// PRD-wmd-repair-affordances §2.1.
    #[serde(default)]
    pub repair_louder_phrases: Vec<String>,
    /// Configurable persona — tone, name, brevity.  All fields default so
    /// existing `brain.toml` files without a `[persona]` section load fine.
    /// PRD-hearth-persona-config §2.1.
    #[serde(default)]
    pub persona: PersonaConfig,
    /// Routing policy configuration — tier preference, classifier tuning,
    /// Ollama endpoint, cloud timeout.  All fields default so existing
    /// `brain.toml` files without a `[routing]` section load fine.
    /// PRD-wintermute-brain-routing §2.4.
    #[serde(default)]
    pub routing: crate::router::RoutingConfig,
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
            gpu_endpoint: None,
            gpu_model: default_gpu_model(),
            local_llm_endpoint: None,
            local_llm_model: default_local_llm_model(),
            history_turns: default_history_turns(),
            almanac_speak: default_almanac_speak(),
            writeback_auto_commit: false,
            writeback_model: default_writeback_model(),
            writeback_confidence_floor: default_writeback_confidence_floor(),
            idle_gap_ms: default_idle_gap_ms(),
            session_end_phrases: default_session_end_phrases(),
            recap_max_memories: default_recap_max_memories(),
            recap_opener: false,
            repair_repeat_phrases: Vec::new(),
            repair_louder_phrases: Vec::new(),
            persona: PersonaConfig::default(),
            routing: crate::router::RoutingConfig::default(),
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

fn default_gpu_model() -> String {
    DEFAULT_LOCAL_GPU_MODEL.to_string()
}

fn default_local_llm_model() -> String {
    DEFAULT_LOCAL_LLM_MODEL.to_string()
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

/// Default upper bound on thread memories injected at session start.
///
/// PRD-wmd-session-recap §2.1: 5 is conservative for prompt-cache stability.
pub const DEFAULT_RECAP_MAX_MEMORIES: usize = 5;

const fn default_recap_max_memories() -> usize {
    DEFAULT_RECAP_MAX_MEMORIES
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
    #[allow(
        clippy::too_many_lines,
        reason = "env-var mapper: each var is one flat let binding; \
                  extracting sub-fns would require passing error context around"
    )]
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
        // GPU tier — optional; absent means local-gpu is not in the ladder.
        let gpu_endpoint = env_string("WM_BRAIN_GPU_ENDPOINT");
        let gpu_model =
            env_string("WM_BRAIN_GPU_MODEL").unwrap_or_else(default_gpu_model);
        // Local-LLM tier — optional; absent means local-llm is not in the ladder.
        // This is the dedicated Ryzen 7 5700U APU node (constellation-brain-local).
        let local_llm_endpoint = env_string("WM_BRAIN_LOCAL_LLM_ENDPOINT");
        let local_llm_model =
            env_string("WM_BRAIN_LOCAL_LLM_MODEL").unwrap_or_else(default_local_llm_model);

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

        let recap_max_memories = match env_string("WM_BRAIN_RECAP_MAX_MEMORIES") {
            Some(raw) => raw.trim().parse::<usize>().map_err(|e| BrainError::InvalidEnv {
                var: "WM_BRAIN_RECAP_MAX_MEMORIES",
                value: raw.clone(),
                reason: e.to_string(),
            })?,
            None => default_recap_max_memories(),
        };
        let recap_opener = match env_string("WM_BRAIN_RECAP_OPENER") {
            Some(raw) => parse_bool_env("WM_BRAIN_RECAP_OPENER", &raw)?,
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
            almanac_speak,
            default_tier,
            pending_tier: None,
            local_endpoint,
            gpu_endpoint,
            gpu_model,
            local_llm_endpoint,
            local_llm_model,
            history_turns: default_history_turns(),
            writeback_auto_commit,
            writeback_model,
            writeback_confidence_floor,
            idle_gap_ms,
            session_end_phrases: default_session_end_phrases(),
            recap_max_memories,
            recap_opener,
            repair_repeat_phrases: Vec::new(),
            repair_louder_phrases: Vec::new(),
            persona: PersonaConfig::default(),
            routing: crate::router::RoutingConfig::default(),
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

    // ── local-gpu tier (PRD-constellation-brain-gpu) ─────────────────────────

    #[test]
    fn ladder_with_gpu_inserts_local_gpu_between_8b_and_cloud() {
        // AC5: when gpu_endpoint is Some, local-gpu is inserted between
        // local-8b and the cloud tiers.
        let ladder = ladder_with_gpu(Some("http://desktop:8080/v1"), DEFAULT_LOCAL_GPU_MODEL);
        let names: Vec<&str> = ladder.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                TIER_LOCAL_3B,
                TIER_LOCAL_8B,
                TIER_LOCAL_GPU,
                SHORT_MODEL_HAIKU,
                SHORT_MODEL_SONNET,
                SHORT_MODEL_OPUS,
            ],
            "local-gpu must sit between local-8b and haiku"
        );
        // Endpoint override is wired into the gpu tier.
        let gpu = ladder.iter().find(|t| t.name == TIER_LOCAL_GPU).unwrap();
        assert_eq!(
            gpu.endpoint_override.as_deref(),
            Some("http://desktop:8080/v1"),
            "local-gpu tier must carry the GPU endpoint override"
        );
        assert_eq!(gpu.backend, Backend::Local, "local-gpu is a Local backend");
    }

    #[test]
    fn ladder_with_gpu_absent_when_endpoint_none() {
        // AC6: when gpu_endpoint is None the ladder is identical to the
        // five-rung default (graceful absence).
        let without = ladder_with_gpu(None, DEFAULT_LOCAL_GPU_MODEL);
        let default = default_ladder();
        let names_w: Vec<&str> = without.iter().map(|t| t.name.as_str()).collect();
        let names_d: Vec<&str> = default.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names_w, names_d, "no gpu_endpoint → ladder identical to default");
        assert!(
            without.iter().all(|t| t.name != TIER_LOCAL_GPU),
            "local-gpu must not appear when endpoint is absent"
        );
    }

    #[test]
    fn ladder_with_gpu_absent_when_endpoint_empty() {
        // An empty string is treated as absent (consistent with env-var handling).
        let without = ladder_with_gpu(Some(""), DEFAULT_LOCAL_GPU_MODEL);
        assert!(
            without.iter().all(|t| t.name != TIER_LOCAL_GPU),
            "empty gpu_endpoint string → local-gpu absent"
        );
    }

    #[test]
    fn local_gpu_in_allowed_model_names() {
        // AC7: the local-gpu tier name must be accepted by validate_model_name
        // so `wmd swap-model local-gpu` works.
        validate_model_name(TIER_LOCAL_GPU).expect("local-gpu must be in ALLOWED_MODEL_NAMES");
    }

    // ── local-llm tier (PRD-constellation-brain-local) ───────────────────────

    #[test]
    fn ladder_with_local_llm_inserts_before_cloud() {
        // AC5: when local_llm_endpoint is Some, local-llm is inserted between
        // the last local rung and the first cloud rung.
        let base = default_ladder();
        let ladder = ladder_with_local_llm(
            base,
            Some("http://5700u:8080/v1"),
            DEFAULT_LOCAL_LLM_MODEL,
        );
        let names: Vec<&str> = ladder.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                TIER_LOCAL_3B,
                TIER_LOCAL_8B,
                TIER_LOCAL_LLM,
                SHORT_MODEL_HAIKU,
                SHORT_MODEL_SONNET,
                SHORT_MODEL_OPUS,
            ],
            "local-llm must sit between local-8b and haiku"
        );
        let llm = ladder.iter().find(|t| t.name == TIER_LOCAL_LLM).unwrap();
        assert_eq!(
            llm.endpoint_override.as_deref(),
            Some("http://5700u:8080/v1"),
            "local-llm tier must carry the endpoint override"
        );
        assert_eq!(llm.backend, Backend::Local, "local-llm is a Local backend");
        assert_eq!(llm.model, DEFAULT_LOCAL_LLM_MODEL);
    }

    #[test]
    fn ladder_with_local_llm_absent_when_endpoint_none() {
        // AC6: when local_llm_endpoint is None the ladder is unchanged.
        let base = default_ladder();
        let without = ladder_with_local_llm(base, None, DEFAULT_LOCAL_LLM_MODEL);
        let default = default_ladder();
        let names_w: Vec<&str> = without.iter().map(|t| t.name.as_str()).collect();
        let names_d: Vec<&str> = default.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names_w, names_d, "no local_llm_endpoint → ladder unchanged");
        assert!(
            without.iter().all(|t| t.name != TIER_LOCAL_LLM),
            "local-llm must not appear when endpoint is absent"
        );
    }

    #[test]
    fn ladder_with_local_llm_absent_when_endpoint_empty() {
        // An empty string is treated as absent (consistent with env-var handling).
        let base = default_ladder();
        let without = ladder_with_local_llm(base, Some(""), DEFAULT_LOCAL_LLM_MODEL);
        assert!(
            without.iter().all(|t| t.name != TIER_LOCAL_LLM),
            "empty local_llm_endpoint → local-llm absent"
        );
    }

    #[test]
    fn ladder_with_gpu_and_local_llm_both_present() {
        // Both GPU and local-llm endpoints set: local-gpu and local-llm both
        // appear before the cloud tiers, local-gpu first (inserted first, then
        // local-llm inserted at the Anthropic boundary which is now after gpu).
        let base = default_ladder();
        let with_gpu = ladder_with_gpu(Some("http://desktop:8080/v1"), DEFAULT_LOCAL_GPU_MODEL);
        let with_both = ladder_with_local_llm(
            with_gpu,
            Some("http://5700u:8080/v1"),
            DEFAULT_LOCAL_LLM_MODEL,
        );
        let names: Vec<&str> = with_both.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                TIER_LOCAL_3B,
                TIER_LOCAL_8B,
                TIER_LOCAL_GPU,
                TIER_LOCAL_LLM,
                SHORT_MODEL_HAIKU,
                SHORT_MODEL_SONNET,
                SHORT_MODEL_OPUS,
            ],
            "both local-gpu and local-llm sit between local-8b and haiku, gpu first"
        );
    }

    #[test]
    fn local_llm_in_allowed_model_names() {
        // AC7 pattern: the local-llm tier name must be accepted by validate_model_name
        // so `wmd swap-model local-llm` works.
        validate_model_name(TIER_LOCAL_LLM).expect("local-llm must be in ALLOWED_MODEL_NAMES");
    }

    #[test]
    fn brain_config_local_llm_fields_default() {
        // BrainConfig defaults: local_llm_endpoint is None, model is the default.
        let cfg = BrainConfig::default();
        assert!(cfg.local_llm_endpoint.is_none(), "local_llm_endpoint defaults to None");
        assert_eq!(cfg.local_llm_model, DEFAULT_LOCAL_LLM_MODEL);
    }

    #[test]
    fn brain_config_local_llm_round_trips_serde() {
        // local_llm_endpoint and local_llm_model serialise/deserialise cleanly.
        let cfg = BrainConfig {
            local_llm_endpoint: Some("http://5700u:8080/v1".to_string()),
            local_llm_model: "qwen2.5-3b-instruct:Q4_K_M".to_string(),
            ..BrainConfig::default()
        };
        let v = serde_json::to_value(&cfg).expect("serialises");
        let back: BrainConfig = serde_json::from_value(v).expect("round-trips");
        assert_eq!(cfg.local_llm_endpoint, back.local_llm_endpoint);
        assert_eq!(cfg.local_llm_model, back.local_llm_model);
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

    // ── PersonaConfig tests (PRD-hearth-persona-config AC1–AC7) ─────────────

    #[test]
    fn persona_default_deserialises_from_empty_toml() {
        // AC2: a brain.toml without a [persona] section loads to defaults.
        let cfg: BrainConfig = toml::from_str("default_model = \"sonnet\"\n")
            .expect("deserialise");
        assert_eq!(cfg.persona, PersonaConfig::default());
        assert_eq!(cfg.persona.register, Register::WarmElder);
        assert_eq!(cfg.persona.self_name, "wintermute");
        assert!(cfg.persona.addresses_user);
        assert_eq!(cfg.persona.max_sentences, 3);
        assert!(cfg.persona.extra.is_empty());
    }

    #[test]
    fn persona_per_field_override() {
        // AC1: explicit [persona] fields override defaults.
        let toml_str = r#"
[persona]
self_name = "Ada"
register = "brisk"
addresses_user = false
max_sentences = 2
extra = "Always end with a friendly greeting."
"#;
        let cfg: BrainConfig = toml::from_str(toml_str).expect("deserialise");
        assert_eq!(cfg.persona.self_name, "Ada");
        assert_eq!(cfg.persona.register, Register::Brisk);
        assert!(!cfg.persona.addresses_user);
        assert_eq!(cfg.persona.max_sentences, 2);
        assert_eq!(cfg.persona.extra, "Always end with a friendly greeting.");
    }

    #[test]
    fn persona_register_plain_produces_default_persona() {
        // AC3: Plain register → byte-identical to DEFAULT_PERSONA const.
        use crate::daemon::DEFAULT_PERSONA;
        let p = PersonaConfig {
            register: Register::Plain,
            ..PersonaConfig::default()
        };
        let base = p.compose_base(None);
        assert_eq!(base, DEFAULT_PERSONA);
    }

    #[test]
    fn persona_warm_elder_contains_required_phrases() {
        // AC3: WarmElder contains "short sentences" and lacks "daemon"/"API"/"config".
        let p = PersonaConfig {
            register: Register::WarmElder,
            ..PersonaConfig::default()
        };
        let base = p.compose_base(None);
        assert!(base.contains("short sentences"), "WarmElder must mention short sentences");
        assert!(!base.contains("daemon"), "WarmElder must not say 'daemon'");
        assert!(!base.contains("API"), "WarmElder must not say 'API'");
        assert!(!base.contains("config"), "WarmElder must not say 'config'");
    }

    #[test]
    fn persona_name_substitution() {
        // AC4: {self_name}/{user_name} replaced; no literal braces.
        let p = PersonaConfig {
            self_name: "Ada".to_string(),
            register: Register::WarmElder,
            addresses_user: true,
            ..PersonaConfig::default()
        };
        let base = p.compose_base(Some("Mum"));
        assert!(base.contains("Ada"), "self_name substituted");
        assert!(base.contains("Mum"), "user_name substituted");
        assert!(!base.contains("{self_name}"), "no literal placeholder");
        assert!(!base.contains("{user_name}"), "no literal placeholder");
    }

    #[test]
    fn persona_user_clause_omitted_when_unset() {
        // AC4: when user_name is None, user clause omitted cleanly.
        let p = PersonaConfig {
            register: Register::WarmElder,
            ..PersonaConfig::default()
        };
        let base_no_user = p.compose_base(None);
        let base_with_user = p.compose_base(Some("Alice"));
        assert!(base_with_user.contains("Alice"));
        assert!(!base_no_user.contains("{user_name}"));
    }

    #[test]
    fn persona_cache_prefix_stable() {
        // AC5: two compose_base calls with the same config yield identical strings.
        let p = PersonaConfig {
            self_name: "Ada".to_string(),
            register: Register::WarmElder,
            addresses_user: true,
            max_sentences: 2,
            extra: "Be concise.".to_string(),
            ..PersonaConfig::default()
        };
        let a = p.compose_base(Some("Mum"));
        let b = p.compose_base(Some("Mum"));
        assert_eq!(a, b, "compose_base must be byte-stable (cache prefix)");
    }

    #[test]
    fn persona_extra_appended_to_base() {
        // AC1 / §2.1: extra text is appended after the register prose.
        let p = PersonaConfig {
            register: Register::Brisk,
            extra: "Always greet warmly.".to_string(),
            ..PersonaConfig::default()
        };
        let base = p.compose_base(None);
        assert!(base.ends_with("Always greet warmly."));
    }

    #[test]
    fn persona_round_trips_through_serde() {
        // AC1: PersonaConfig serialises and deserialises cleanly.
        let original = PersonaConfig {
            self_name: "Ada".to_string(),
            register: Register::Brisk,
            addresses_user: false,
            max_sentences: 1,
            extra: "Extra note.".to_string(),
            ..PersonaConfig::default()
        };
        let v = serde_json::to_value(&original).expect("serialises");
        let back: PersonaConfig = serde_json::from_value(v).expect("round-trips");
        assert_eq!(original, back);
    }

    // ── PersonaConfig forbidden_terms tests (PRD-persona-forbidden-vocab) ────

    #[test]
    fn forbidden_terms_empty_is_noop() {
        // AC3: empty vec → composed prompt unchanged.
        let p_without = PersonaConfig {
            register: Register::WarmElder,
            forbidden_terms: vec![],
            ..PersonaConfig::default()
        };
        let p_with = PersonaConfig {
            register: Register::WarmElder,
            forbidden_terms: vec![],
            ..PersonaConfig::default()
        };
        assert_eq!(p_without.compose_base(None), p_with.compose_base(None));
        // Confirm no avoidance instruction leaked in.
        assert!(!p_without.compose_base(None).contains("never use"));
    }

    #[test]
    fn forbidden_terms_single_term() {
        // AC2: one term → avoidance instruction appended.
        let p = PersonaConfig {
            register: Register::WarmElder,
            forbidden_terms: vec!["robot".to_string()],
            ..PersonaConfig::default()
        };
        let base = p.compose_base(None);
        assert!(base.contains("robot"), "term present in instruction");
        assert!(
            base.contains("You must never use the following words or phrases"),
            "instruction header present"
        );
        assert!(
            base.contains("rephrase with plain, warm language"),
            "rephrasing guidance present"
        );
    }

    #[test]
    fn forbidden_terms_multiple_terms() {
        // AC2: multiple terms → comma-joined correctly.
        let p = PersonaConfig {
            register: Register::WarmElder,
            forbidden_terms: vec!["AI".to_string(), "algorithm".to_string(), "model".to_string()],
            ..PersonaConfig::default()
        };
        let base = p.compose_base(None);
        assert!(base.contains("AI, algorithm, model"), "terms comma-joined");
    }

    #[test]
    fn forbidden_terms_preserved_casing() {
        // AC1: terms stored as-provided, not lowercased.
        let p = PersonaConfig {
            forbidden_terms: vec!["AI".to_string(), "Neural Network".to_string()],
            ..PersonaConfig::default()
        };
        assert_eq!(p.forbidden_terms[0], "AI");
        assert_eq!(p.forbidden_terms[1], "Neural Network");
        let base = p.compose_base(None);
        assert!(base.contains("AI"), "uppercase preserved");
        assert!(base.contains("Neural Network"), "mixed case preserved");
    }

    #[test]
    fn forbidden_terms_in_full_compose() {
        // AC5/AC6: TOML parse → compose includes forbidden_terms instruction.
        let toml_str = r#"
[persona]
self_name = "Wren"
register = "warm-elder"
forbidden_terms = ["AI", "computer", "algorithm"]
"#;
        let cfg: BrainConfig = toml::from_str(toml_str).expect("deserialise");
        assert_eq!(cfg.persona.self_name, "Wren");
        assert_eq!(cfg.persona.forbidden_terms, vec!["AI", "computer", "algorithm"]);
        let base = cfg.persona.compose_base(None);
        assert!(base.contains("AI, computer, algorithm"), "terms in composed output");
        assert!(base.contains("never use"), "instruction present");
    }

    #[test]
    fn forbidden_terms_jocelyn_preset() {
        // AC4/AC6: Jocelyn example preset deserializes and composes correctly.
        let toml_str = r#"
[persona]
self_name = "Wren"
register = "warm-elder"
forbidden_terms = ["AI", "artificial intelligence", "computer", "algorithm", "model", "machine learning", "neural network", "robot", "device", "technology", "software", "program", "digital assistant"]
"#;
        let cfg: BrainConfig = toml::from_str(toml_str).expect("deserialise");
        assert_eq!(cfg.persona.self_name, "Wren");
        assert_eq!(cfg.persona.forbidden_terms.len(), 13);

        // AC6: Composed prompt with Jocelyn preset is bounded (instruction
        // overhead scales with term count; 13-term Jocelyn list must fit in ≤500 chars).
        let base_plain = PersonaConfig {
            register: Register::WarmElder,
            ..PersonaConfig::default()
        }
        .compose_base(None);
        let base_jocelyn = cfg.persona.compose_base(None);
        let delta = base_jocelyn.len().saturating_sub(base_plain.len());
        assert!(
            delta <= 500,
            "Jocelyn preset adds {delta} chars; must be ≤500 for 13 terms"
        );
    }

    #[test]
    fn forbidden_terms_does_not_duplicate_extra() {
        // AC2: extra is still appended after register prose; forbidden_terms follows after.
        let p = PersonaConfig {
            register: Register::Brisk,
            extra: "Always greet warmly.".to_string(),
            forbidden_terms: vec!["device".to_string()],
            ..PersonaConfig::default()
        };
        let base = p.compose_base(None);
        let extra_pos = base.find("Always greet warmly.").expect("extra present");
        let forbidden_pos = base.find("never use").expect("forbidden instruction present");
        assert!(
            forbidden_pos > extra_pos,
            "forbidden instruction must come after extra"
        );
    }
}
