//! Output-side enforcement for persona forbidden vocabulary.
//!
//! Sits between the generated reply and TTS publish; scans for forbidden
//! terms and either substitutes a safe phrase or regenerates with a hardened
//! prompt addendum naming the leaked terms.
//!
//! PRD-persona-redline §2, PRD-persona-redline-regenerate §2.

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
    SafePhrase {
        /// Replacement text spoken instead of the dirty reply.
        safe_phrase: Option<String>,
    },
    /// Re-issue the request with a hardened system addendum naming the leaked
    /// terms, and only fall back to the safe phrase if regeneration also leaks.
    ///
    /// `max_attempts` bounds the number of re-issues (minimum 1).
    /// `fallback` is the safe phrase used when all attempts still leak;
    /// `None` uses [`DEFAULT_SAFE_PHRASE`].
    ///
    /// The invariant "a dirty reply is never published" is preserved: if
    /// every attempt still leaks, `fallback` is returned instead.
    Regenerate {
        /// Maximum number of re-issue attempts (minimum 1).
        max_attempts: u8,
        /// Replacement text used when all attempts still leak.
        /// `None` uses [`DEFAULT_SAFE_PHRASE`].
        fallback: Option<String>,
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

/// Build a system-prompt addendum that names the specific forbidden terms
/// leaked in `hits`.
///
/// Returns a non-empty instruction string when `hits` is non-empty, listing
/// every distinct term by name. Returns an empty string when `hits` is empty
/// (no addendum needed).
///
/// Used by the `Regenerate` action path to harden the re-issued request.
#[must_use]
pub fn hardened_addendum(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return String::new();
    }
    // Collect distinct terms while preserving first-seen order.
    let mut seen = std::collections::HashSet::new();
    let mut terms: Vec<&str> = Vec::new();
    for hit in hits {
        if seen.insert(hit.term.as_str()) {
            terms.push(hit.term.as_str());
        }
    }
    let term_list = terms.join(", ");
    format!(
        "Do not use these words in your reply: {term_list}. Rephrase as a warm friend would."
    )
}

/// Apply redline enforcement: if `reply` contains any forbidden term and
/// `action` is active, return the safe replacement text; otherwise return
/// `None` (caller publishes the original reply unchanged).
///
/// For [`RedlineAction::Regenerate`] the caller is expected to handle
/// the regeneration loop itself (this function returns the fallback phrase
/// after incrementing the counter, so the sync signature is preserved for
/// the non-async hot path). Use [`enforce_with_regenerate`] from async
/// contexts to get the full regeneration behaviour.
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
        RedlineAction::Regenerate { fallback, .. } => {
            // Sync path: return the fallback phrase. The async regeneration
            // loop is handled by the caller (daemon.rs) which has access to
            // the model client. This ensures the sync enforce() signature is
            // preserved and the invariant (no dirty reply published) holds even
            // if the caller does not implement the async loop.
            Some(
                fallback
                    .as_deref()
                    .unwrap_or(DEFAULT_SAFE_PHRASE)
                    .to_string(),
            )
        }
    }
}

/// Inner regeneration loop driven by an injected model closure.
///
/// Called by the daemon when the action is `Regenerate` and the initial
/// reply contains hits. The `model_fn` closure receives the hardened
/// addendum string and returns a new candidate reply. This function:
/// 1. Loops up to `max_attempts` times.
/// 2. On a clean result (no hits), returns `Some(clean_reply)`.
/// 3. If all attempts still leak, returns `Some(fallback_phrase)`.
///
/// The invariant — a dirty reply is never returned — is preserved by design.
///
/// `max_attempts` is clamped to a minimum of 1 to prevent infinite loops
/// from misconfigured values.
///
/// # Returns
/// `Some(reply)` — either the clean regenerated reply or the fallback phrase.
/// `None` is never returned (there is always a safe result on exhaustion).
#[must_use]
pub fn enforce_with_regenerate<F>(
    original_reply: &str,
    forbidden: &[String],
    max_attempts: u8,
    fallback: Option<&str>,
    model_fn: &mut F,
) -> Option<String>
where
    F: FnMut(&str) -> String,
{
    let max = max_attempts.max(1);
    let mut hits = scan(original_reply, forbidden);
    for _ in 0..max {
        let addendum = hardened_addendum(&hits);
        let candidate = model_fn(&addendum);
        let new_hits = scan(&candidate, forbidden);
        if new_hits.is_empty() {
            return Some(candidate);
        }
        hits = new_hits;
    }
    // All attempts leaked — return fallback (invariant preserved).
    Some(fallback.unwrap_or(DEFAULT_SAFE_PHRASE).to_string())
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

    // ── hardened_addendum() ──────────────────────────────────────────────────

    #[test]
    fn hardened_addendum_empty_hits_returns_empty() {
        let result = hardened_addendum(&[]);
        assert!(result.is_empty(), "empty hits → empty addendum");
    }

    #[test]
    fn hardened_addendum_names_single_term() {
        let hits = vec![Hit {
            term: "robot".to_string(),
            byte_range: 0..5,
        }];
        let result = hardened_addendum(&hits);
        assert!(result.contains("robot"), "addendum must name 'robot'");
        assert!(!result.is_empty());
    }

    #[test]
    fn hardened_addendum_names_all_distinct_terms() {
        let hits = vec![
            Hit { term: "AI".to_string(), byte_range: 0..2 },
            Hit { term: "robot".to_string(), byte_range: 7..12 },
        ];
        let result = hardened_addendum(&hits);
        assert!(result.contains("AI"), "addendum must name 'AI'");
        assert!(result.contains("robot"), "addendum must name 'robot'");
    }

    #[test]
    fn hardened_addendum_deduplicates_repeated_term() {
        let hits = vec![
            Hit { term: "robot".to_string(), byte_range: 0..5 },
            Hit { term: "robot".to_string(), byte_range: 10..15 },
        ];
        let result = hardened_addendum(&hits);
        // "robot" should appear exactly once in the term list
        let count = result.matches("robot").count();
        assert_eq!(count, 1, "duplicate hits → term listed once");
    }

    // ── Regenerate variant serde ─────────────────────────────────────────────

    #[test]
    fn redline_action_regenerate_round_trips_toml() {
        let toml_str = r#"
[persona.redline]
action = "regenerate"
max_attempts = 2
fallback = "Let me try that again."
"#;
        let cfg: crate::BrainConfig = toml::from_str(toml_str).expect("deserialise");
        match &cfg.persona.redline {
            RedlineAction::Regenerate { max_attempts, fallback } => {
                assert_eq!(*max_attempts, 2);
                assert_eq!(fallback.as_deref(), Some("Let me try that again."));
            }
            other => panic!("expected Regenerate, got {other:?}"),
        }
    }

    #[test]
    fn redline_action_regenerate_no_fallback_round_trips_toml() {
        let toml_str = r#"
[persona.redline]
action = "regenerate"
max_attempts = 1
"#;
        let cfg: crate::BrainConfig = toml::from_str(toml_str).expect("deserialise");
        match &cfg.persona.redline {
            RedlineAction::Regenerate { max_attempts, fallback } => {
                assert_eq!(*max_attempts, 1);
                assert!(fallback.is_none(), "fallback should be None when absent");
            }
            other => panic!("expected Regenerate, got {other:?}"),
        }
    }

    #[test]
    fn redline_action_regenerate_round_trips_json() {
        let original = RedlineAction::Regenerate {
            max_attempts: 3,
            fallback: Some("Fallback phrase.".to_string()),
        };
        let v = serde_json::to_value(&original).expect("serialises");
        let back: RedlineAction = serde_json::from_value(v).expect("round-trips");
        assert_eq!(original, back);
    }

    #[test]
    fn redline_action_regenerate_is_active() {
        let action = RedlineAction::Regenerate {
            max_attempts: 1,
            fallback: None,
        };
        assert!(action.is_active(), "Regenerate must be active");
    }

    // ── enforce() with Regenerate (sync path returns fallback) ───────────────

    #[test]
    fn enforce_regenerate_with_hit_returns_fallback() {
        // The sync enforce() path for Regenerate returns the fallback phrase.
        // The actual regeneration loop is in daemon.rs (async path).
        let result = enforce(
            "The robot said hello.",
            &["robot".to_string()],
            &RedlineAction::Regenerate {
                max_attempts: 2,
                fallback: Some("I'll rephrase that.".to_string()),
            },
        );
        assert_eq!(result.as_deref(), Some("I'll rephrase that."));
    }

    #[test]
    fn enforce_regenerate_no_fallback_uses_default_phrase() {
        let result = enforce(
            "The AI said hello.",
            &["AI".to_string()],
            &RedlineAction::Regenerate {
                max_attempts: 1,
                fallback: None,
            },
        );
        assert_eq!(result.as_deref(), Some(DEFAULT_SAFE_PHRASE));
    }

    #[test]
    fn enforce_regenerate_no_hit_returns_none() {
        let result = enforce(
            "Everything is fine.",
            &["robot".to_string()],
            &RedlineAction::Regenerate {
                max_attempts: 1,
                fallback: Some("fallback".to_string()),
            },
        );
        assert!(result.is_none(), "no hit → no enforcement");
    }

    // ── enforce_with_regenerate() injected-closure tests ─────────────────────

    #[test]
    fn enforce_with_regenerate_succeeds_on_attempt_2_with_max_2() {
        // Closure leaks on attempt 1, clean on attempt 2.
        let mut call_count = 0u8;
        let mut model = |_addendum: &str| -> String {
            call_count += 1;
            if call_count == 1 {
                "The robot is thinking.".to_string() // leaks
            } else {
                "I am thinking.".to_string() // clean
            }
        };
        let forbidden = vec!["robot".to_string()];
        let result = enforce_with_regenerate(
            "The robot said hello.",
            &forbidden,
            2,
            Some("Fallback."),
            &mut model,
        );
        assert_eq!(result.as_deref(), Some("I am thinking."));
        assert_eq!(call_count, 2, "exactly 2 calls made");
    }

    #[test]
    fn enforce_with_regenerate_falls_back_when_max_attempts_1_and_leaks() {
        // With max_attempts=1, one attempt, still leaks → fallback.
        let mut call_count = 0u8;
        let mut model = |_addendum: &str| -> String {
            call_count += 1;
            "The robot is thinking.".to_string() // always leaks
        };
        let forbidden = vec!["robot".to_string()];
        let result = enforce_with_regenerate(
            "The robot said hello.",
            &forbidden,
            1,
            Some("Fallback phrase."),
            &mut model,
        );
        assert_eq!(result.as_deref(), Some("Fallback phrase."));
        assert_eq!(call_count, 1, "exactly 1 call made");
    }

    #[test]
    fn enforce_with_regenerate_always_leaks_returns_fallback() {
        // Always-leaking closure → fallback (invariant preserved).
        let mut call_count = 0u8;
        let mut model = |_addendum: &str| -> String {
            call_count += 1;
            "The AI is speaking.".to_string() // always leaks
        };
        let forbidden = vec!["AI".to_string()];
        let result = enforce_with_regenerate(
            "The AI said hello.",
            &forbidden,
            3,
            Some("Safe phrase."),
            &mut model,
        );
        assert_eq!(result.as_deref(), Some("Safe phrase."));
        assert_eq!(call_count, 3, "exactly max_attempts calls made");
    }

    #[test]
    fn enforce_with_regenerate_no_fallback_uses_default() {
        let mut model = |_addendum: &str| -> String {
            "The robot is here.".to_string()
        };
        let forbidden = vec!["robot".to_string()];
        let result = enforce_with_regenerate(
            "The robot said hello.",
            &forbidden,
            1,
            None,
            &mut model,
        );
        assert_eq!(result.as_deref(), Some(DEFAULT_SAFE_PHRASE));
    }
}
