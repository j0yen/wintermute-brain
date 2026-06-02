//! First-contact greeting module for `wintermute-brain`.
//!
//! Selects and composes the proactive boot greeting the daemon publishes
//! before the user's first turn (or stays silent). Three greeting kinds
//! are defined:
//!
//! | Kind | When |
//! |---|---|
//! | [`GreetingKind::FirstEver`] | no profile facts **and** no prior thread in recall |
//! | [`GreetingKind::Returning`] | prior thread found (existing `recap_opener` path) |
//! | [`GreetingKind::Silent`] | `greeting = "off"` or default |
//!
//! PRD-hearth-first-contact-greeting §2.1–2.4.

use serde::{Deserialize, Serialize};

use crate::PersonaConfig;

// ── GreetingMode ─────────────────────────────────────────────────────────────

/// Controls when the daemon emits a proactive boot greeting.
///
/// Serialised as a TOML/JSON string under `[persona] greeting = "..."`.
/// Default is [`GreetingMode::Off`] (conservative — no proactive speech
/// unless the deployment opts in). Set to `"auto"` to enable
/// first-ever / returning detection.
///
/// PRD-hearth-first-contact-greeting §2.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum GreetingMode {
    /// Silence — no proactive greeting published at boot (the default).
    #[default]
    Off,
    /// Automatic: emit [`GreetingKind::FirstEver`] when recall is empty,
    /// [`GreetingKind::Returning`] when a prior thread is found, and
    /// [`GreetingKind::Silent`] otherwise.
    Auto,
    /// Always emit the first-ever greeting on every boot, regardless of
    /// recall state. Useful for factory-reset / demo deployments.
    FirstEverAlways,
}

// ── GreetingKind ─────────────────────────────────────────────────────────────

/// The greeting variant selected by [`select_greeting_kind`].
///
/// PRD §2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreetingKind {
    /// No prior profile or thread in recall — first time the device is
    /// ever summoned. Contains a teaching block (wake word + examples).
    FirstEver,
    /// A prior conversation thread exists. Brief warm re-greeting; the
    /// teaching block is omitted.
    Returning,
    /// No proactive publish. Emitted when [`GreetingMode::Off`] or when
    /// `auto` mode finds neither first-ever nor returning signals.
    Silent,
}

// ── Recall probe inputs ───────────────────────────────────────────────────────

/// Summary of the recall state the greeting selector uses.
///
/// Production code fills this from two recall queries before calling
/// [`select_greeting_kind`]; tests inject values directly.
///
/// PRD §2.1: zero hits across both `wintermute-profile` **and**
/// `wintermute-thread-*` ⇒ first ever.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecallPresence {
    /// At least one `wintermute-profile` memory exists in recall.
    pub has_profile: bool,
    /// At least one `wintermute-thread-*` memory exists in recall.
    pub has_thread: bool,
}

impl RecallPresence {
    /// Probe is considered "empty" (first-ever) when both flags are false.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        !self.has_profile && !self.has_thread
    }
}

// ── Kind selection ────────────────────────────────────────────────────────────

/// Map a [`GreetingMode`] and [`RecallPresence`] to the [`GreetingKind`] to emit.
///
/// PRD §2.1:
/// - `Off` → always [`GreetingKind::Silent`] (AC4).
/// - `FirstEverAlways` → always [`GreetingKind::FirstEver`].
/// - `Auto` + empty recall → [`GreetingKind::FirstEver`] (AC1 / AC2).
/// - `Auto` + prior thread → [`GreetingKind::Returning`] (AC3).
/// - `Auto` + profile but no thread → [`GreetingKind::Silent`] (conservative).
#[must_use]
pub fn select_greeting_kind(mode: GreetingMode, presence: RecallPresence) -> GreetingKind {
    match mode {
        GreetingMode::Off => GreetingKind::Silent,
        GreetingMode::FirstEverAlways => GreetingKind::FirstEver,
        GreetingMode::Auto => {
            if presence.is_empty() {
                GreetingKind::FirstEver
            } else if presence.has_thread {
                GreetingKind::Returning
            } else {
                // Profile exists but no thread: device has been set up but
                // hasn't had a conversation. Stay silent (conservative).
                GreetingKind::Silent
            }
        }
    }
}

// ── Greeting composition ──────────────────────────────────────────────────────

/// Static example prompts shown in a first-ever greeting (v1 — not a live
/// capability probe). PRD §2.2 §v1.
const FIRST_EVER_EXAMPLES: &[&str] = &["what time it is", "what the weather is like"];

/// Compose the greeting text for `kind` from the persona and wake word.
///
/// - [`GreetingKind::Silent`] → `None` (no publish).
/// - [`GreetingKind::FirstEver`] → warm welcome + wake word + examples.
/// - [`GreetingKind::Returning`] → brief warm re-greeting (no teaching
///   block). Distinct from `FirstEver` (AC3).
///
/// The `user_name` parameter is taken from [`crate::BrainConfig::user_name`].
///
/// PRD §2.2.
#[must_use]
pub fn compose_greeting(
    kind: GreetingKind,
    persona: &PersonaConfig,
    user_name: Option<&str>,
    wake_word: &str,
) -> Option<String> {
    match kind {
        GreetingKind::Silent => None,
        GreetingKind::FirstEver => {
            let address = user_name
                .filter(|n| !n.is_empty())
                .map(|n| format!("{n}, "))
                .unwrap_or_default();
            let examples = FIRST_EVER_EXAMPLES.join(", or ");
            Some(format!(
                "Hello {address}I'm {self_name}. \
You can talk to me out loud whenever you like — just say \"{wake_word}\" first. \
Try asking me {examples}. \
I'm here whenever you need me.",
                self_name = persona.self_name,
            ))
        }
        GreetingKind::Returning => {
            let address = user_name
                .filter(|n| !n.is_empty())
                .map(|n| format!(", {n}"))
                .unwrap_or_default();
            Some(format!(
                "Hello{address}, I'm {self_name}. Good to hear from you again.",
                self_name = persona.self_name,
            ))
        }
    }
}

// ── Greet-once guard ──────────────────────────────────────────────────────────

/// In-process guard that ensures the boot greeting is published **exactly
/// once** per daemon lifetime (PRD §2.4 / AC5).
///
/// Set [`GreetingGuard::mark_greeted`] after the greeting is published;
/// [`GreetingGuard::should_greet`] returns `false` on subsequent calls.
#[derive(Debug, Default)]
pub struct GreetingGuard {
    greeted: bool,
}

impl GreetingGuard {
    /// Construct a fresh (un-greeted) guard.
    #[must_use]
    pub const fn new() -> Self {
        Self { greeted: false }
    }

    /// Returns `true` if the greeting has not yet been published this boot.
    #[must_use]
    pub const fn should_greet(&self) -> bool {
        !self.greeted
    }

    /// Mark the greeting as published. Subsequent calls to
    /// [`Self::should_greet`] return `false`.
    pub const fn mark_greeted(&mut self) {
        self.greeted = true;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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
    use crate::PersonaConfig;

    fn default_persona() -> PersonaConfig {
        PersonaConfig::default()
    }

    fn ada_persona() -> PersonaConfig {
        PersonaConfig {
            self_name: "Ada".to_string(),
            ..PersonaConfig::default()
        }
    }

    // ── AC1 / AC2: first-ever detection on empty recall ───────────────────

    #[test]
    fn first_ever_selected_on_empty_recall() {
        let kind = select_greeting_kind(
            GreetingMode::Auto,
            RecallPresence { has_profile: false, has_thread: false },
        );
        assert_eq!(kind, GreetingKind::FirstEver);
    }

    #[test]
    fn first_ever_text_contains_wake_word_and_examples() {
        let p = default_persona();
        let text = compose_greeting(
            GreetingKind::FirstEver,
            &p,
            Some("Alice"),
            "hey wintermute",
        )
        .expect("should produce text");
        assert!(text.contains("hey wintermute"), "must contain wake word");
        // AC2: at least two example prompts
        assert!(text.contains("time"), "must mention time example");
        assert!(text.contains("weather"), "must mention weather example");
    }

    // ── AC3: returning path (seeded thread) ───────────────────────────────

    #[test]
    fn returning_selected_with_prior_thread() {
        let kind = select_greeting_kind(
            GreetingMode::Auto,
            RecallPresence { has_profile: false, has_thread: true },
        );
        assert_eq!(kind, GreetingKind::Returning);
    }

    #[test]
    fn returning_and_first_ever_texts_differ() {
        let p = default_persona();
        let wake = "hey wintermute";
        let fe = compose_greeting(GreetingKind::FirstEver, &p, None, wake).expect("text");
        let ret = compose_greeting(GreetingKind::Returning, &p, None, wake).expect("text");
        assert_ne!(fe, ret, "first-ever and returning must differ");
        // Returning must NOT contain the teaching block
        assert!(!ret.contains(wake), "returning must not contain the wake word teaching block");
    }

    // ── AC4: off is silent ────────────────────────────────────────────────

    #[test]
    fn off_mode_is_always_silent() {
        for presence in [
            RecallPresence { has_profile: false, has_thread: false },
            RecallPresence { has_profile: true, has_thread: true },
        ] {
            let kind = select_greeting_kind(GreetingMode::Off, presence);
            assert_eq!(kind, GreetingKind::Silent);
        }
        // default config → Off
        let kind = select_greeting_kind(GreetingMode::default(), RecallPresence::default());
        assert_eq!(kind, GreetingKind::Silent);
    }

    #[test]
    fn silent_kind_produces_no_text() {
        let p = default_persona();
        let result = compose_greeting(GreetingKind::Silent, &p, None, "hey wintermute");
        assert!(result.is_none(), "Silent must produce None");
    }

    // ── AC5: greet-once guard ─────────────────────────────────────────────

    #[test]
    fn greet_once_guard_fires_exactly_once() {
        let mut guard = GreetingGuard::new();
        assert!(guard.should_greet(), "first call should greet");
        guard.mark_greeted();
        assert!(!guard.should_greet(), "after mark, should not greet again");
        // Second invocation stays false (AC5)
        assert!(!guard.should_greet(), "stays false");
    }

    // ── AC6: register inheritance (name substitution) ─────────────────────

    #[test]
    fn first_ever_contains_self_name_and_user_name() {
        let p = ada_persona();
        let text = compose_greeting(
            GreetingKind::FirstEver,
            &p,
            Some("Mum"),
            "hey ada",
        )
        .expect("text");
        assert!(text.contains("Ada"), "must contain self_name Ada");
        assert!(text.contains("Mum"), "must contain user_name Mum");
        assert!(!text.contains("daemon"), "must not say 'daemon'");
        assert!(!text.contains("API"), "must not say 'API'");
        assert!(!text.contains("config"), "must not say 'config'");
    }

    #[test]
    fn returning_contains_self_name_and_user_name() {
        let p = ada_persona();
        let text = compose_greeting(
            GreetingKind::Returning,
            &p,
            Some("Mum"),
            "hey ada",
        )
        .expect("text");
        assert!(text.contains("Ada"), "returning must contain self_name Ada");
        assert!(text.contains("Mum"), "returning must contain user_name Mum");
    }

    // ── Mode round-trip: serde ────────────────────────────────────────────

    #[test]
    fn greeting_mode_serde_round_trip() {
        for (s, expected) in [
            ("\"off\"", GreetingMode::Off),
            ("\"auto\"", GreetingMode::Auto),
            ("\"first-ever-always\"", GreetingMode::FirstEverAlways),
        ] {
            let m: GreetingMode = serde_json::from_str(s).expect("deserialise");
            assert_eq!(m, expected);
            let back = serde_json::to_string(&m).expect("serialise");
            let m2: GreetingMode = serde_json::from_str(&back).expect("re-parse");
            assert_eq!(m, m2);
        }
    }

    #[test]
    fn greeting_mode_default_is_off() {
        assert_eq!(GreetingMode::default(), GreetingMode::Off);
    }

    // ── Recall presence helpers ───────────────────────────────────────────

    #[test]
    fn recall_presence_is_empty_both_false() {
        assert!(RecallPresence { has_profile: false, has_thread: false }.is_empty());
        assert!(!RecallPresence { has_profile: true, has_thread: false }.is_empty());
        assert!(!RecallPresence { has_profile: false, has_thread: true }.is_empty());
        assert!(!RecallPresence { has_profile: true, has_thread: true }.is_empty());
    }

    // ── first-ever-always overrides recall state ──────────────────────────

    #[test]
    fn first_ever_always_fires_even_with_full_recall() {
        let kind = select_greeting_kind(
            GreetingMode::FirstEverAlways,
            RecallPresence { has_profile: true, has_thread: true },
        );
        assert_eq!(kind, GreetingKind::FirstEver);
    }

    // ── Composition with no user_name ─────────────────────────────────────

    #[test]
    fn first_ever_without_user_name_has_no_literal_placeholder() {
        let p = default_persona();
        let text = compose_greeting(GreetingKind::FirstEver, &p, None, "hey wintermute")
            .expect("text");
        assert!(!text.contains("{user_name}"), "no literal placeholder");
        assert!(!text.contains("{self_name}"), "no literal placeholder");
    }
}
