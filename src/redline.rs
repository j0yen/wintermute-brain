//! Output-side enforcement for persona forbidden vocabulary.
//!
//! Sits between the generated reply and TTS publish; scans for forbidden
//! terms and either substitutes a safe phrase or (in a future iteration)
//! regenerates with a hardened prompt.
//!
//! PRD-persona-redline §2.

use std::sync::atomic::{AtomicU64, Ordering};

/// Running count of redline enforcements since daemon start.
///
/// Incremented once per enforcement event (not per hit).
pub static REDLINE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A hit: a forbidden term found in a reply.
#[derive(Debug, Clone)]
pub struct Hit {
    /// The forbidden term that matched.
    pub term: String,
    /// Byte range within the original reply string.
    pub byte_range: std::ops::Range<usize>,
}

/// What to do when a forbidden term is detected in a generated reply.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "action")]
pub enum RedlineAction {
    /// No enforcement (the default). Forbidden terms found in replies are
    /// allowed through unchanged. Existing configs without a `redline` key
    /// deserialise to this variant.
    Off,
    /// Replace the reply with a safe phrase when forbidden terms are found.
    ///
    /// `safe_phrase` — if `Some`, the literal phrase emitted instead of the
    /// dirty reply.  If `None` a built-in neutral fallback is used.
    ///
    /// # Future work
    ///
    /// A future iteration may add a `Regenerate` variant that re-issues the
    /// LLM request with a hardened system addendum naming the leaked terms.
    /// For now, `SafePhrase` guarantees the invariant that a dirty reply is
    /// never published, while keeping the wiring simple.
    SafePhrase {
        /// Replacement text spoken instead of the dirty reply.
        safe_phrase: Option<String>,
    },
}

impl Default for RedlineAction {
    fn default() -> Self {
        RedlineAction::Off
    }
}

impl RedlineAction {
    /// Returns `true` when enforcement is active (i.e. not `Off`).
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(self, RedlineAction::Off)
    }
}

/// Built-in fallback phrase used when [`RedlineAction::SafePhrase`] has
/// `safe_phrase = None`.
pub const DEFAULT_SAFE_PHRASE: &str =
    "Let me put that a different way — everything's fine.";

/// Normalise a string for forbidden-term matching.
///
/// - Lowercases for case-insensitive comparison.
/// - Strips full-stops that appear between single letters (normalises
///   "A.I." → "ai", "U.S.A." → "usa") so dotted abbreviations match the
///   plain form in the forbidden list.
fn normalise(s: &str) -> String {
    // Strip dots between single-letter runs first (A.I. → AI, U.S.A. → USA).
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c == '.' {
            // Suppress this dot if the char before and after are both single
            // ASCII letters (already seen as one-letter run on each side).
            let prev_letter = i > 0 && chars[i - 1].is_ascii_alphabetic();
            let next_letter = i + 1 < n && chars[i + 1].is_ascii_alphabetic();
            if prev_letter && next_letter {
                // also check the char after the next letter isn't another letter
                // (we only want single-letter groups, e.g. A.I. not the.room)
                let prev_two_letter = i >= 2 && chars[i - 2].is_ascii_alphabetic();
                let next_two_letter = i + 2 < n && chars[i + 2].is_ascii_alphabetic();
                if !prev_two_letter && !next_two_letter {
                    // single letter on each side → abbreviation dot, skip it
                    i += 1;
                    continue;
                }
            }
        }
        out.push(c);
        i += 1;
    }
    out.to_lowercase()
}

/// Scan `reply` for any forbidden term. Returns all hits (in order of
/// discovery).
///
/// Matching rules:
/// - Case-insensitive (via Unicode lowercase).
/// - Dotted abbreviations in the reply are normalised ("A.I." matches "AI").
/// - Word-boundary checked: the hit must not be a substring of a longer word.
///   The boundary check treats any ASCII alphanumeric character as a
///   continuation character; non-ASCII bytes adjacent to a hit are accepted
///   (conservative — avoids false negatives on accented neighbours).
/// - Multi-word terms are matched as contiguous phrases after normalisation.
#[must_use]
pub fn scan(reply: &str, forbidden: &[String]) -> Vec<Hit> {
    let reply_norm = normalise(reply);
    let reply_bytes = reply_norm.as_bytes();
    let mut hits = Vec::new();

    for term in forbidden {
        let term_norm = normalise(term.as_str());
        if term_norm.is_empty() {
            continue;
        }

        let mut search_from = 0;
        while search_from < reply_norm.len() {
            let Some(pos) = reply_norm[search_from..].find(&term_norm) else {
                break;
            };
            let abs_pos = search_from + pos;
            let abs_end = abs_pos + term_norm.len();

            // Word-boundary check on the normalised string.
            let before_ok = abs_pos == 0
                || !reply_bytes[abs_pos - 1].is_ascii_alphanumeric();
            let after_ok = abs_end >= reply_bytes.len()
                || !reply_bytes[abs_end].is_ascii_alphanumeric();

            if before_ok && after_ok {
                hits.push(Hit {
                    term: term.clone(),
                    // Report byte range in the *original* reply.
                    // Because normalise() may shrink the string (e.g. "A.I."→"ai",
                    // 4 chars → 2), the normalised offsets may differ from
                    // the original offsets.  Rather than trying to map them
                    // back precisely, we do a case-insensitive substring search
                    // on the *original* to pin the original range.
                    byte_range: locate_in_original(reply, term, abs_pos, abs_end),
                });
                // Advance past this match to avoid infinite loops on zero-len
                // terms (guarded above) and to find overlapping terms.
                search_from = abs_pos + term_norm.len().max(1);
            } else {
                search_from = abs_pos + 1;
            }
        }
    }
    hits
}

/// Best-effort reverse-mapping of a normalised hit range back to the
/// original string. Walks the original to find the corresponding span.
///
/// Falls back to `0..0` on any inconsistency (callers only use the term
/// field from a Hit; the range is informational).
fn locate_in_original(original: &str, term: &str, _norm_start: usize, _norm_end: usize) -> std::ops::Range<usize> {
    let orig_lower = original.to_lowercase();
    let term_lower = term.to_lowercase();
    // Remove dots for abbreviation matching in the lowercased original
    // (e.g. find "ai" in "A.I." after lower → "a.i." won't work directly,
    // but the normalised search already confirmed a hit exists; just look
    // for the lowercased term in the lowercased original).
    if let Some(pos) = orig_lower.find(&term_lower) {
        return pos..pos + term_lower.len();
    }
    // Abbreviation case: strip dots and look for normalised term.
    let orig_norm = normalise(original);
    let term_norm = normalise(term);
    if let Some(pos) = orig_norm.find(&term_norm) {
        // pos is in normalised space; for reporting just use it as-is
        return pos..pos + term_norm.len();
    }
    0..0
}

/// Apply redline enforcement: if `reply` contains any forbidden term and
/// `action` is active, return the safe replacement text; otherwise return
/// `None` (caller publishes the original reply unchanged).
///
/// Increments [`REDLINE_COUNTER`] on each enforcement.
#[must_use]
pub fn enforce(reply: &str, forbidden: &[String], action: &RedlineAction) -> Option<String> {
    if !action.is_active() || forbidden.is_empty() {
        return None;
    }
    let hits = scan(reply, forbidden);
    if hits.is_empty() {
        return None;
    }
    REDLINE_COUNTER.fetch_add(1, Ordering::Relaxed);
    match action {
        RedlineAction::Off => None,
        RedlineAction::SafePhrase { safe_phrase } => Some(
            safe_phrase
                .as_deref()
                .unwrap_or(DEFAULT_SAFE_PHRASE)
                .to_string(),
        ),
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

    // ── scan() ───────────────────────────────────────────────────────────────

    #[test]
    fn scan_empty_forbidden_returns_no_hits() {
        let hits = scan("The robot is here.", &[]);
        assert!(hits.is_empty());
    }

    #[test]
    fn scan_no_match_returns_no_hits() {
        let hits = scan("Hello, how are you?", &["robot".to_string()]);
        assert!(hits.is_empty());
    }

    #[test]
    fn scan_exact_match_returns_hit() {
        let hits = scan("I am a robot.", &["robot".to_string()]);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].term, "robot");
    }

    #[test]
    fn scan_case_insensitive() {
        let hits = scan("I am a ROBOT.", &["robot".to_string()]);
        assert_eq!(hits.len(), 1, "case-insensitive match");

        let hits2 = scan("I am a Robot here.", &["ROBOT".to_string()]);
        assert_eq!(hits2.len(), 1, "reversed case");
    }

    #[test]
    fn scan_word_boundary_no_false_positive_substring() {
        // "AI" must not match inside "said" or "rain"
        let hits = scan("She said it was raining.", &["AI".to_string()]);
        assert!(
            hits.is_empty(),
            "AI must not match inside 'said' or 'raining'"
        );
    }

    #[test]
    fn scan_word_boundary_at_start_and_end() {
        // "AI" at the very start and end of the string
        let hits_start = scan("AI is here.", &["AI".to_string()]);
        assert_eq!(hits_start.len(), 1, "AI at start of string");

        let hits_end = scan("That is AI", &["AI".to_string()]);
        assert_eq!(hits_end.len(), 1, "AI at end of string");
    }

    #[test]
    fn scan_multi_word_phrase() {
        let hits = scan("I use neural network models.", &["neural network".to_string()]);
        assert_eq!(hits.len(), 1, "multi-word phrase matched");
        assert_eq!(hits[0].term, "neural network");
    }

    #[test]
    fn scan_multi_word_phrase_no_partial_match() {
        // "neural" alone doesn't fire the "neural network" term
        let hits = scan("The neural response was fast.", &["neural network".to_string()]);
        assert!(hits.is_empty(), "partial multi-word should not fire");
    }

    #[test]
    fn scan_dotted_abbreviation_matches_plain_term() {
        // "A.I." in reply should match forbidden "AI"
        let hits = scan("The A.I. said hello.", &["AI".to_string()]);
        assert_eq!(hits.len(), 1, "A.I. should match AI in forbidden list");
    }

    #[test]
    fn scan_multiple_terms_all_found() {
        let forbidden = vec!["AI".to_string(), "robot".to_string(), "algorithm".to_string()];
        let hits = scan("The AI robot uses an algorithm.", &forbidden);
        assert_eq!(hits.len(), 3, "all three terms found");
        let terms: Vec<&str> = hits.iter().map(|h| h.term.as_str()).collect();
        assert!(terms.contains(&"AI"), "AI found");
        assert!(terms.contains(&"robot"), "robot found");
        assert!(terms.contains(&"algorithm"), "algorithm found");
    }

    #[test]
    fn scan_same_term_twice() {
        let hits = scan("robot or robot?", &["robot".to_string()]);
        assert_eq!(hits.len(), 2, "same term found twice");
    }

    // ── RedlineAction serde ──────────────────────────────────────────────────

    #[test]
    fn redline_action_off_is_default() {
        let action = RedlineAction::default();
        assert_eq!(action, RedlineAction::Off);
        assert!(!action.is_active());
    }

    #[test]
    fn redline_action_off_round_trips_toml() {
        // A [persona] table without a redline key should deserialise to Off.
        let toml_str = r#"
[persona]
self_name = "Wren"
"#;
        let cfg: crate::BrainConfig = toml::from_str(toml_str).expect("deserialise");
        assert_eq!(cfg.persona.redline, RedlineAction::Off);
    }

    #[test]
    fn redline_action_off_explicit_round_trips_toml() {
        // Explicit redline = {action="off"} in [persona]
        let toml_str = r#"
[persona]
self_name = "Wren"
redline = {action = "off"}
"#;
        let cfg: crate::BrainConfig = toml::from_str(toml_str).expect("deserialise");
        assert_eq!(cfg.persona.redline, RedlineAction::Off);
    }

    #[test]
    fn redline_action_safe_phrase_round_trips_toml() {
        let toml_str = r#"
[persona]
self_name = "Jocelyn"
[persona.redline]
action = "safe_phrase"
safe_phrase = "Let me rephrase that."
"#;
        let cfg: crate::BrainConfig = toml::from_str(toml_str).expect("deserialise");
        match &cfg.persona.redline {
            RedlineAction::SafePhrase { safe_phrase } => {
                assert_eq!(safe_phrase.as_deref(), Some("Let me rephrase that."));
            }
            other => panic!("expected SafePhrase, got {other:?}"),
        }
    }

    #[test]
    fn redline_action_safe_phrase_no_phrase_round_trips_toml() {
        let toml_str = r#"
[persona.redline]
action = "safe_phrase"
"#;
        let cfg: crate::BrainConfig = toml::from_str(toml_str).expect("deserialise");
        match &cfg.persona.redline {
            RedlineAction::SafePhrase { safe_phrase } => {
                assert!(safe_phrase.is_none(), "safe_phrase should be None when absent");
            }
            other => panic!("expected SafePhrase, got {other:?}"),
        }
    }

    #[test]
    fn redline_action_round_trips_json() {
        let original = RedlineAction::SafePhrase {
            safe_phrase: Some("I'll rephrase.".to_string()),
        };
        let v = serde_json::to_value(&original).expect("serialises");
        let back: RedlineAction = serde_json::from_value(v).expect("round-trips");
        assert_eq!(original, back);
    }

    // ── enforce() ────────────────────────────────────────────────────────────

    #[test]
    fn enforce_off_returns_none_even_with_hit() {
        let result = enforce(
            "The robot is here.",
            &["robot".to_string()],
            &RedlineAction::Off,
        );
        assert!(result.is_none(), "Off action must never enforce");
    }

    #[test]
    fn enforce_safe_phrase_no_hit_returns_none() {
        let result = enforce(
            "Everything is fine.",
            &["robot".to_string()],
            &RedlineAction::SafePhrase { safe_phrase: None },
        );
        assert!(result.is_none(), "no hit → no enforcement");
    }

    #[test]
    fn enforce_safe_phrase_with_hit_returns_safe_phrase() {
        let result = enforce(
            "The robot said hello.",
            &["robot".to_string()],
            &RedlineAction::SafePhrase {
                safe_phrase: Some("I'll put that differently.".to_string()),
            },
        );
        assert_eq!(result.as_deref(), Some("I'll put that differently."));
    }

    #[test]
    fn enforce_safe_phrase_none_uses_default_phrase() {
        let result = enforce(
            "The AI is speaking.",
            &["AI".to_string()],
            &RedlineAction::SafePhrase { safe_phrase: None },
        );
        assert_eq!(result.as_deref(), Some(DEFAULT_SAFE_PHRASE));
    }

    #[test]
    fn enforce_increments_counter() {
        let before = REDLINE_COUNTER.load(Ordering::Relaxed);
        let _ = enforce(
            "robot robot robot",
            &["robot".to_string()],
            &RedlineAction::SafePhrase { safe_phrase: None },
        );
        let after = REDLINE_COUNTER.load(Ordering::Relaxed);
        assert_eq!(after, before + 1, "counter incremented once per enforcement");
    }

    #[test]
    fn enforce_no_increment_when_off() {
        let before = REDLINE_COUNTER.load(Ordering::Relaxed);
        let _ = enforce(
            "robot is here",
            &["robot".to_string()],
            &RedlineAction::Off,
        );
        let after = REDLINE_COUNTER.load(Ordering::Relaxed);
        assert_eq!(after, before, "counter must not increment for Off action");
    }
}
