//! Repair-intent matcher for `wmd`.
//!
//! When a user says "say that again" or "louder" immediately after an assistant
//! reply, the brain should replay the last turn from history without incurring
//! an LLM round-trip. This module classifies short utterances as repair
//! requests — verbatim replay or louder-replay — before the normal LLM
//! dispatch path runs (PRD-wmd-repair-affordances §2.1).
//!
//! # Design decisions
//! - Conservative: only short, unambiguous utterances match. A sentence
//!   containing "louder" as an incidental word (e.g. "why is the music louder
//!   at night?") passes through to the LLM unchanged.
//! - Word-count gating: transcripts with more than [`REPAIR_WORD_LIMIT`] words
//!   are never classified as repair, regardless of content.
//! - Config-driven: phrase lists are part of [`crate::BrainConfig`] so
//!   operators can add regional variants without recompiling.

use std::collections::BTreeSet;

/// Maximum word count for a transcript to be considered a repair request.
///
/// Transcripts longer than this threshold are ordinary user turns and are
/// never classified as repair requests, even if they contain a repair keyword.
/// Value `5` is conservative: "say that again louder please" is 5 words and
/// still clearly a repair request; "can you tell me why it's louder at night"
/// is 9 words and should reach the LLM.
pub const REPAIR_WORD_LIMIT: usize = 5;

/// Default phrases (case-insensitive, punctuation-stripped) that trigger a
/// verbatim [`Repair::RepeatLast`] classification.
pub const DEFAULT_REPEAT_PHRASES: &[&str] = &[
    "say that again",
    "what did you say",
    "what did you just say",
    "come again",
    "pardon",
    "say it again",
    "repeat that",
    "can you repeat that",
    "sorry what",
    "what was that",
    "i didnt catch that",
    "i didn't catch that",
];

/// Default phrases (case-insensitive, punctuation-stripped) that trigger a
/// [`Repair::RepeatLouder`] classification (replay + loudness hint).
pub const DEFAULT_LOUDER_PHRASES: &[&str] = &[
    "louder",
    "speak up",
    "say that louder",
    "say that again louder",
    "say it louder",
    "speak louder",
    "louder please",
    "can you speak up",
    "a bit louder",
    "a little louder",
];

/// Repair classification for a user utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repair {
    /// Verbatim replay of the last assistant turn with no loudness hint.
    RepeatLast,
    /// Verbatim replay of the last assistant turn with `loudness = "loud"`.
    RepeatLouder,
    /// Not a repair request; fall through to normal LLM dispatch.
    None,
}

/// Strip ASCII punctuation and lowercase a transcript so phrase matching is
/// punctuation- and case-insensitive.
///
/// This is intentionally simple: it drops `.,!?;:"'-()[]{}` and converts
/// to ASCII-lowercase. Unicode punctuation is preserved (not dropped) so
/// non-ASCII transcripts are not mangled beyond recognition.
#[must_use]
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '-' | '(' | ')' | '[' | ']' | '{' | '}'))
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Count whitespace-delimited words in a string.
#[must_use]
fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Classify a user transcript as a [`Repair`] variant.
///
/// The classification algorithm:
///
/// 1. Strip punctuation and lowercase the transcript.
/// 2. Count words. If `> REPAIR_WORD_LIMIT` → [`Repair::None`] (ambiguity guard).
/// 3. Check if the normalized form matches any `louder_phrases` entry →
///    [`Repair::RepeatLouder`].
/// 4. Check if the normalized form matches any `repeat_phrases` entry →
///    [`Repair::RepeatLast`].
/// 5. Otherwise → [`Repair::None`].
///
/// Louder phrases are checked first so "say that again louder" resolves to
/// `RepeatLouder` rather than `RepeatLast`.
#[must_use]
pub fn classify(
    transcript: &str,
    repeat_phrases: &BTreeSet<String>,
    louder_phrases: &BTreeSet<String>,
) -> Repair {
    let normalized = normalize(transcript.trim());
    // Ambiguity guard: long sentences are ordinary turns, not repair requests.
    if word_count(&normalized) > REPAIR_WORD_LIMIT {
        return Repair::None;
    }
    if louder_phrases.contains(&normalized) {
        return Repair::RepeatLouder;
    }
    if repeat_phrases.contains(&normalized) {
        return Repair::RepeatLast;
    }
    Repair::None
}

/// Build the default `BTreeSet` of normalized repeat phrases.
///
/// Callers that have no configured overrides use this to get the built-in set.
#[must_use]
pub fn default_repeat_phrases() -> BTreeSet<String> {
    DEFAULT_REPEAT_PHRASES
        .iter()
        .map(|s| normalize(s))
        .collect()
}

/// Build the default `BTreeSet` of normalized louder phrases.
///
/// Callers that have no configured overrides use this to get the built-in set.
#[must_use]
pub fn default_louder_phrases() -> BTreeSet<String> {
    DEFAULT_LOUDER_PHRASES
        .iter()
        .map(|s| normalize(s))
        .collect()
}

/// Build a `BTreeSet` of normalized phrases from a user-supplied slice.
///
/// An empty `phrases` slice produces an empty set (callers must then fall back
/// to [`default_repeat_phrases`] / [`default_louder_phrases`] themselves).
#[must_use]
pub fn build_phrase_set(phrases: &[String]) -> BTreeSet<String> {
    phrases.iter().map(|s| normalize(s)).collect()
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

    fn repeat() -> BTreeSet<String> {
        default_repeat_phrases()
    }

    fn louder() -> BTreeSet<String> {
        default_louder_phrases()
    }

    // ── AC1 partial: "say that again" → RepeatLast ──────────────────────────

    #[test]
    fn repeat_phrases_classify_as_repeat_last() {
        for phrase in [
            "say that again",
            "what did you say",
            "come again",
            "pardon",
            "Say That Again",   // case insensitive
            "say that again!",  // with punctuation
            "Come again?",
        ] {
            assert_eq!(
                classify(phrase, &repeat(), &louder()),
                Repair::RepeatLast,
                "phrase {phrase:?} should be RepeatLast"
            );
        }
    }

    // ── AC2 partial: "say that louder" → RepeatLouder ───────────────────────

    #[test]
    fn louder_phrases_classify_as_repeat_louder() {
        for phrase in [
            "louder",
            "speak up",
            "say that again louder",
            "say that louder",
            "Louder!",
            "SPEAK UP",
        ] {
            assert_eq!(
                classify(phrase, &repeat(), &louder()),
                Repair::RepeatLouder,
                "phrase {phrase:?} should be RepeatLouder"
            );
        }
    }

    // ── AC5: long sentences containing keywords are not repair ───────────────

    #[test]
    fn long_sentences_with_keywords_not_repair() {
        for phrase in [
            "could you speak a bit about why the radio is louder at night",
            "what did you say when you were talking about the weather yesterday",
            "say that again but in a different way please",
        ] {
            assert_eq!(
                classify(phrase, &repeat(), &louder()),
                Repair::None,
                "long phrase {phrase:?} should be None"
            );
        }
    }

    // ── AC5: unrelated short phrases are not repair ──────────────────────────

    #[test]
    fn unrelated_short_phrases_not_repair() {
        for phrase in [
            "hello",
            "what time is it",
            "tell me a story",
            "good morning",
        ] {
            assert_eq!(
                classify(phrase, &repeat(), &louder()),
                Repair::None,
                "unrelated phrase {phrase:?} should be None"
            );
        }
    }

    // ── Louder beats repeat when both could match ────────────────────────────

    #[test]
    fn louder_takes_priority_over_repeat() {
        // "say that again louder" is in the louder set; check it's RepeatLouder.
        assert_eq!(
            classify("say that again louder", &repeat(), &louder()),
            Repair::RepeatLouder
        );
    }

    // ── Word count exactly at boundary ──────────────────────────────────────

    #[test]
    fn word_count_boundary_at_limit() {
        // REPAIR_WORD_LIMIT=5 → 5-word phrase can match, 6 cannot.
        // "say that again louder please" is 5 words; add it to the louder set.
        let mut custom_louder = louder();
        custom_louder.insert("say that again louder please".to_string());
        assert_eq!(
            classify("say that again louder please", &repeat(), &custom_louder),
            Repair::RepeatLouder,
            "5 words is within limit"
        );
        // 6 words → never matches even if it's in the set.
        let mut custom_louder2 = louder();
        custom_louder2.insert("say that again louder please do".to_string());
        assert_eq!(
            classify("say that again louder please do", &repeat(), &custom_louder2),
            Repair::None,
            "6 words exceeds limit, always None"
        );
    }

    // ── normalize strips common punctuation ─────────────────────────────────

    #[test]
    fn normalize_strips_punctuation_and_lowercases() {
        assert_eq!(normalize("Come again?"), "come again");
        assert_eq!(normalize("Say That Again!"), "say that again");
        assert_eq!(normalize("What did you say..."), "what did you say");
    }

    // ── build_phrase_set normalizes entries ──────────────────────────────────

    #[test]
    fn build_phrase_set_normalizes() {
        let phrases = vec!["Say That Again!".to_string(), "Come Again?".to_string()];
        let set = build_phrase_set(&phrases);
        assert!(set.contains("say that again"));
        assert!(set.contains("come again"));
    }
}
