//! Name-introduction ceremony for `wintermute-brain`.
//!
//! Fires *before* the regular first-ever boot greeting (or on demand
//! via `wm.persona.introduce`) when the deployment opts in.  The
//! ceremony speaks "Hello. My name is [self_name]. You can call me
//! that any time you need me — just say my name, and I'll be right
//! here.", then waits up to `ack_timeout_secs` for a voiced
//! acknowledgment on `wm.stt.final`.
//!
//! Outcomes:
//! - **Ack** received within the window → transition to the regular
//!   `FirstEver` greeting.
//! - **Timeout** (no speech or no ack match) → speak patience phrase,
//!   defer `FirstEver` until the next wake-word trigger.
//! - **Unrecognized** word → speak reassurance phrase → proceed to
//!   `FirstEver`.
//!
//! PRD-persona-name-ceremony §2.

use serde::{Deserialize, Serialize};

// ── Bus topic ─────────────────────────────────────────────────────────────────

/// Bus topic that triggers the introduction ceremony in [`IntroductionMode::Explicit`].
///
/// Publish `{}` on this topic to re-fire the ceremony on demand:
/// ```text
/// agorabus publish wm.persona.introduce '{}'
/// ```
///
/// PRD-persona-name-ceremony §2.3.
pub const INTRODUCE_TOPIC: &str = "wm.persona.introduce";

// ── IntroductionMode ──────────────────────────────────────────────────────────

/// Controls whether the assistant introduces itself by name before the
/// first-ever-boot greeting.
///
/// All variants are `#[serde(default)]` so existing `brain.toml` files
/// without the field deserialise to [`IntroductionMode::Off`].
///
/// PRD-persona-name-ceremony §2.1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
pub enum IntroductionMode {
    /// No introduction spoken (default). The regular `FirstEver` greeting fires
    /// immediately.
    #[default]
    Off,
    /// Speak the introduction on the daemon's very first boot (tracked by the
    /// same first-boot marker as `GreetingKind::FirstEver`).
    ///
    /// Waits up to `ack_timeout_secs` for a voiced acknowledgment before
    /// transitioning to the regular `FirstEver` greeting.
    ///
    /// TOML example:
    /// ```toml
    /// [persona]
    /// introduction.FirstEverBoot = { ack_timeout_secs = 25 }
    /// ```
    FirstEverBoot {
        /// Maximum wait in seconds for a voice acknowledgment.
        ack_timeout_secs: u64,
    },
    /// Speak the introduction when `wm.persona.introduce` is published on the
    /// bus.  Useful for re-introduction after a long absence.
    ///
    /// TOML example:
    /// ```toml
    /// [persona]
    /// introduction.Explicit = { ack_timeout_secs = 20 }
    /// ```
    Explicit {
        /// Maximum wait in seconds for a voice acknowledgment.
        ack_timeout_secs: u64,
    },
}

// ── IntroductionPhase ─────────────────────────────────────────────────────────

/// In-process phase of the introduction ceremony.
///
/// Stored on `DaemonState` (under a `Mutex`).  Initial state is always
/// [`IntroductionPhase::Idle`]; transitions happen inside the event loop.
///
/// PRD-persona-name-ceremony §2.2.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum IntroductionPhase {
    /// No ceremony in progress.  Default state.
    #[default]
    Idle,
    /// Introduction spoken; waiting for a voice acknowledgment.
    ///
    /// `deadline_ms` is the Unix-millisecond timestamp after which the
    /// patience phrase fires and the ceremony transitions to
    /// [`IntroductionPhase::WaitingForWakeWord`] (timeout path).
    AwaitingAck {
        /// Absolute deadline (Unix ms) for the ack window.
        deadline_ms: u64,
    },
    /// Patience phrase was spoken; `FirstEver` greeting fires on the
    /// next wake-word trigger instead of immediately.
    WaitingForWakeWord,
}

// ── Ack classification ────────────────────────────────────────────────────────

/// Ack-word list used by [`classify_intro_response`].
///
/// Case-insensitive partial-match: any transcript that *contains* one of
/// these tokens is considered an acknowledgment.
pub const ACK_WORDS: &[&str] = &[
    "okay",
    "ok",
    "yes",
    "yeah",
    "sure",
    "alright",
    "all right",
    "fine",
    "good",
    "great",
    "got it",
    "i see",
    "hi",
    "hello",
    "nice to meet you",
    "uh huh",
    "mm hmm",
];

/// Result of classifying a user transcript against the introduction ack tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntroAckClass {
    /// Transcript contains a recognised acknowledgment word/phrase.
    Ack,
    /// Transcript does not match any ack token.
    Unrecognized,
}

/// Classify a transcript as [`IntroAckClass::Ack`] or
/// [`IntroAckClass::Unrecognized`].
///
/// Matching is case-insensitive; the transcript is lowercased before
/// comparison and each token in [`ACK_WORDS`] is checked as a substring.
#[must_use]
pub fn classify_intro_response(transcript: &str) -> IntroAckClass {
    let lower = transcript.to_lowercase();
    for word in ACK_WORDS {
        if lower.contains(word) {
            return IntroAckClass::Ack;
        }
    }
    IntroAckClass::Unrecognized
}

// ── Phrase composers ──────────────────────────────────────────────────────────

/// Compose the introduction speech text.
///
/// Published to `wm.tts.speak` when the ceremony fires.
///
/// PRD-persona-name-ceremony §2 (introduction speech).
#[must_use]
pub fn compose_introduction_text(self_name: &str) -> String {
    format!(
        "Hello. My name is {self_name}. \
You can call me that any time you need me \u{2014} \
just say my name, and I'll be right here."
    )
}

/// Compose the patience phrase spoken on timeout.
///
/// Published when no acknowledgment is received within `ack_timeout_secs`.
/// The `FirstEver` greeting is deferred to the next wake-word trigger.
///
/// PRD-persona-name-ceremony §2 (timeout).
#[must_use]
pub fn compose_patience_text() -> String {
    "Take your time \u{2014} I'm not going anywhere. Just say my name when you're ready."
        .to_string()
}

/// Compose the reassurance phrase spoken on unrecognized input.
///
/// Published when a transcript arrives that doesn't match an ack word.
/// The `FirstEver` greeting then proceeds immediately.
///
/// PRD-persona-name-ceremony §2 (unrecognized).
#[must_use]
pub fn compose_reassurance_text(self_name: &str) -> String {
    format!(
        "I heard you \u{2014} no need to say anything in particular. \
I'm {self_name}, here when you need me."
    )
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

    // ── AC1: IntroductionMode serialises/deserialises cleanly ────────────────

    #[test]
    fn introduction_off_is_default() {
        let mode = IntroductionMode::default();
        assert_eq!(mode, IntroductionMode::Off, "Off must be the default");
    }

    #[test]
    fn introduction_off_deserialises_from_missing_field() {
        // Existing brain.toml files without the field load fine (#[serde(default)]).
        #[derive(serde::Deserialize)]
        struct Wrapper {
            #[serde(default)]
            introduction: IntroductionMode,
        }
        let toml_str = r#""#;
        let w: Wrapper = toml::from_str(toml_str).expect("deserialise empty TOML");
        assert_eq!(w.introduction, IntroductionMode::Off);
    }

    #[test]
    fn introduction_first_ever_boot_round_trips_json() {
        let mode = IntroductionMode::FirstEverBoot { ack_timeout_secs: 25 };
        let v = serde_json::to_value(&mode).expect("serialise");
        let back: IntroductionMode = serde_json::from_value(v).expect("deserialise");
        assert_eq!(mode, back);
    }

    #[test]
    fn introduction_explicit_round_trips_json() {
        let mode = IntroductionMode::Explicit { ack_timeout_secs: 20 };
        let v = serde_json::to_value(&mode).expect("serialise");
        let back: IntroductionMode = serde_json::from_value(v).expect("deserialise");
        assert_eq!(mode, back);
    }

    #[test]
    fn introduction_first_ever_boot_round_trips_toml() {
        // TOML external-tag form:
        // introduction.FirstEverBoot = { ack_timeout_secs = 25 }
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct Wrapper {
            introduction: IntroductionMode,
        }
        let toml_str = "introduction.FirstEverBoot = { ack_timeout_secs = 25 }\n";
        let w: Wrapper = toml::from_str(toml_str).expect("deserialise TOML");
        assert_eq!(w.introduction, IntroductionMode::FirstEverBoot { ack_timeout_secs: 25 });
    }

    // ── AC2 / AC8: introduction phrase contains self_name ────────────────────

    #[test]
    fn introduction_first_ever_boot_speaks_name() {
        let text = compose_introduction_text("Wren");
        assert!(text.contains("Wren"), "intro text must contain self_name");
        assert!(!text.contains("wintermute"), "intro text must not hard-code 'wintermute'");
    }

    #[test]
    fn introduction_text_contains_name_arbitrary() {
        for name in ["Clara", "Ada", "Rosie"] {
            let text = compose_introduction_text(name);
            assert!(text.contains(name), "intro text must contain '{name}'");
        }
    }

    // ── AC3 / AC4 / AC5: ack classification ──────────────────────────────────

    #[test]
    fn classify_ack_recognises_standard_words() {
        for word in ["okay", "ok", "yes", "yeah", "sure", "alright", "hi", "hello"] {
            assert_eq!(
                classify_intro_response(word),
                IntroAckClass::Ack,
                "'{word}' should be Ack"
            );
        }
    }

    #[test]
    fn classify_ack_case_insensitive() {
        assert_eq!(classify_intro_response("OKAY"), IntroAckClass::Ack);
        assert_eq!(classify_intro_response("Yes"), IntroAckClass::Ack);
        assert_eq!(classify_intro_response("NICE TO MEET YOU"), IntroAckClass::Ack);
    }

    #[test]
    fn classify_ack_unrecognized_returns_unrecognized() {
        for phrase in ["what time is it", "play music", "I don't know", ""] {
            assert_eq!(
                classify_intro_response(phrase),
                IntroAckClass::Unrecognized,
                "'{phrase}' should be Unrecognized"
            );
        }
    }

    // ── Phrase composition ────────────────────────────────────────────────────

    #[test]
    fn patience_text_is_non_empty() {
        let text = compose_patience_text();
        assert!(!text.is_empty(), "patience text must not be empty");
    }

    #[test]
    fn reassurance_text_contains_self_name() {
        let text = compose_reassurance_text("Wren");
        assert!(text.contains("Wren"), "reassurance must contain self_name");
    }

    // ── IntroductionPhase default ─────────────────────────────────────────────

    #[test]
    fn introduction_phase_default_is_idle() {
        let phase = IntroductionPhase::default();
        assert_eq!(phase, IntroductionPhase::Idle);
    }
}
