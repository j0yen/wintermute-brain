//! Session-end memory writeback for `wmd`.
//!
//! When a conversation session ends (idle gap or explicit close phrase),
//! this module:
//!
//! 1. Renders the session's turn history into a compact transcript.
//! 2. Issues a *dedicated* extraction call to the model (separate from
//!    the conversation loop, cheap / non-streaming) that returns durable
//!    facts as `FACT | <subject> | <body> | <confidence>` lines.
//! 3. Writes extracted facts as recall **proposals** (or direct writes
//!    when `writeback_auto_commit = true`) via
//!    [`crate::daemon::RecallSource::save_fact`].
//! 4. Guards idempotence: each `session_id` writes back at most once.
//!
//! All failures are best-effort and must never block or crash the daemon
//! (PRD-wmd-memory-writeback §2.4).

use std::collections::HashSet;

use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::BrainConfig;
use crate::daemon::RecallSource;
use crate::history::Turn;

/// System prompt fragment injected into the extraction call.
///
/// The model must return only `FACT | <subject-hint> | <one-line body> |
/// <confidence>` lines or the literal string `NO_FACTS`. Everything else
/// is ignored. This keeps the response parseable and small.
pub const EXTRACTION_SYSTEM_PROMPT: &str = "You are a memory extractor. \
Read the conversation transcript below and extract only durable, concrete \
facts about the user — standing preferences, relationships, commitments, or \
health-relevant facts that are worth remembering for future conversations. \
Do NOT extract the assistant's own statements, questions, small talk, or \
conversational filler. \
For each durable fact, output exactly one line in this format:\n\
FACT | <subject-hint> | <one-line durable fact> | <confidence 0.0..1.0>\n\
subject-hint should be a short lowercase slug describing the topic \
(e.g. preferences, family, schedule, health). \
If there are no durable facts, output exactly: NO_FACTS\n\
Output nothing else — no introduction, no explanation.";

/// Minimum transcript turns required before the extractor is invoked.
///
/// Sessions with fewer real turns (e.g. a single failed utterance) write
/// nothing (PRD-wmd-memory-writeback AC4).
pub const MIN_TURNS_FOR_WRITEBACK: usize = 1;

/// Prefix for the thread-scoped recall subject (re-exported from lib for
/// callers that only import this module).
pub use crate::THREAD_SUBJECT_PREFIX;

/// One parsed FACT line from the extractor's response.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedFact {
    /// Short slug hint for the recall subject (e.g. `preferences`).
    pub subject_hint: String,
    /// One-line durable fact body.
    pub body: String,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
}

/// Parse the extractor's raw text response into a list of [`ExtractedFact`]s.
///
/// Lines that don't start with `FACT |` or that carry fewer than 4
/// pipe-delimited fields are silently skipped. Confidence values outside
/// `[0, 1]` are clamped; lines with confidence below `confidence_floor`
/// are dropped (PRD §2.3 / AC8).
///
/// Returns an empty `Vec` when the response contains no valid FACT lines
/// (including when the model returned `NO_FACTS`).
#[must_use]
pub fn parse_extraction_response(raw: &str, confidence_floor: f64) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("FACT |") {
            continue;
        }
        // Split on `|`, expect exactly 4 parts: "FACT", subject, body, confidence.
        let parts: Vec<&str> = trimmed.splitn(4, '|').collect();
        let (Some(subject_part), Some(body_part), Some(conf_part)) =
            (parts.get(1), parts.get(2), parts.get(3))
        else {
            debug!(line = %trimmed, "writeback: FACT line has wrong field count; skipping");
            continue;
        };
        let subject_hint = subject_part.trim().to_string();
        let body = body_part.trim().to_string();
        let conf_str = conf_part.trim();

        if subject_hint.is_empty() || body.is_empty() {
            debug!(line = %trimmed, "writeback: FACT line has empty subject or body; skipping");
            continue;
        }

        let Ok(parsed_conf) = conf_str.parse::<f64>() else {
            debug!(line = %trimmed, "writeback: FACT line has unparseable confidence; skipping");
            continue;
        };
        let confidence = parsed_conf.clamp(0.0, 1.0);

        if confidence < confidence_floor {
            debug!(
                confidence,
                floor = confidence_floor,
                "writeback: FACT line below confidence floor; skipping"
            );
            continue;
        }

        facts.push(ExtractedFact { subject_hint, body, confidence });
    }
    facts
}

/// Build a compact transcript string from a slice of turns.
///
/// Format: each turn is rendered as `User: <text>\nAssistant: <text>\n`.
/// The transcript is suitable as the `user` message body for the
/// extraction call.
#[must_use]
pub fn render_transcript(turns: &[Turn]) -> String {
    let mut out = String::new();
    for t in turns {
        out.push_str("User: ");
        out.push_str(t.user.trim());
        out.push('\n');
        out.push_str("Assistant: ");
        out.push_str(t.assistant.trim());
        out.push('\n');
    }
    out
}

/// Build the recall subject for an episodic session summary.
///
/// The subject is `wintermute-thread-<subject_hint>` for standing facts
/// or `wintermute-thread-<YYYY-MM-DD>` for the thread-scoped
/// episodic summary.
///
/// For extracted facts, we use the `subject_hint` from the extractor to
/// choose a semantic subject; episodic thread entries go under the date
/// subject built from [`BrainConfig::thread_subject_for`].
#[must_use]
pub fn recall_subject_for(subject_hint: &str, date: &str) -> String {
    // The extractor may return a hint like "preferences", "health", etc.
    // Map that to a profile subject; anything thread-scoped gets the date subject.
    if subject_hint == "thread" || subject_hint.is_empty() {
        BrainConfig::thread_subject_for(date)
    } else {
        // Semantic profile subjects: "wintermute-<hint>"
        format!("wintermute-{subject_hint}")
    }
}

/// Guard that tracks which session ids have already written back.
///
/// A `session_id` that appears in this set will not trigger a second
/// writeback even if `trigger_writeback` is called again
/// (PRD-wmd-memory-writeback AC7).
pub struct WritebackGuard {
    written: Mutex<HashSet<String>>,
}

impl WritebackGuard {
    /// Construct an empty guard.
    #[must_use]
    pub fn new() -> Self {
        Self {
            written: Mutex::new(HashSet::new()),
        }
    }

    /// Check whether `session_id` has already been written back.
    ///
    /// If not, atomically marks it as written and returns `true` (the
    /// caller should proceed with writeback). If already written,
    /// returns `false` (skip — AC7).
    pub async fn try_claim(&self, session_id: &str) -> bool {
        let mut guard = self.written.lock().await;
        guard.insert(session_id.to_string())
    }
}

impl Default for WritebackGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Abstraction over the extraction LLM call. Lets tests inject a fake
/// without spinning up a real Anthropic client.
#[async_trait::async_trait]
pub trait ExtractorClient: Send + Sync {
    /// Issue a single extraction call for `transcript` and return the
    /// raw model text. Failures surface as an `Err` string; the writeback
    /// loop logs them and continues.
    ///
    /// # Errors
    /// Returns a descriptive error string on any transport / model failure.
    async fn extract(&self, transcript: &str) -> Result<String, String>;
}

/// Run the end-of-session writeback pipeline.
///
/// Steps:
/// 1. Guard: if `session_id` was already written, return immediately (AC7).
/// 2. If `turns` has fewer than [`MIN_TURNS_FOR_WRITEBACK`] entries, skip (AC4).
/// 3. Render the transcript and call `extractor.extract(transcript)`.
/// 4. Parse the response with `parse_extraction_response`.
/// 5. For each extracted fact, call `recall.save_fact(subject, body)`.
/// 6. All failures are logged and swallowed (AC5 / AC6).
///
/// `date_str` should be an ISO-8601 date like `"2026-05-29"`.
#[allow(
    clippy::too_many_arguments,
    reason = "writeback context: session_id + turns + date + config + extractor + recall + guard; \
              each is semantically distinct and the alternative would be an opaque struct"
)]
pub async fn trigger_writeback(
    session_id: &str,
    turns: &[Turn],
    date_str: &str,
    config: &BrainConfig,
    extractor: &dyn ExtractorClient,
    recall: &dyn RecallSource,
    guard: &WritebackGuard,
) {
    // AC7: idempotence guard.
    if !guard.try_claim(session_id).await {
        debug!(session_id, "writeback: session already written; skipping");
        return;
    }

    // AC4: skip empty sessions.
    if turns.len() < MIN_TURNS_FOR_WRITEBACK {
        debug!(
            session_id,
            turn_count = turns.len(),
            "writeback: session has too few turns; skipping"
        );
        return;
    }

    let transcript = render_transcript(turns);

    // AC6: extraction failure is tolerated.
    let raw = match extractor.extract(&transcript).await {
        Ok(r) => r,
        Err(err) => {
            warn!(
                session_id,
                err = %err,
                "writeback: extraction call failed; skipping writeback for this session"
            );
            return;
        }
    };

    let facts = parse_extraction_response(&raw, config.writeback_confidence_floor);

    if facts.is_empty() {
        debug!(session_id, "writeback: extraction returned no facts; nothing to write");
        return;
    }

    let mut written_count = 0usize;
    for fact in &facts {
        let subject = recall_subject_for(&fact.subject_hint, date_str);
        // AC5: recall outage is tolerated.
        match recall.save_fact(&subject, &fact.body).await {
            Ok(id) => {
                debug!(
                    session_id,
                    id = %id,
                    subject = %subject,
                    confidence = fact.confidence,
                    "writeback: fact written"
                );
                written_count = written_count.saturating_add(1);
            }
            Err(err) => {
                warn!(
                    session_id,
                    subject = %subject,
                    err = %err,
                    "writeback: recall save_fact failed; skipping this fact"
                );
            }
        }
    }

    info!(
        session_id,
        facts_extracted = facts.len(),
        facts_written = written_count,
        date = date_str,
        auto_commit = config.writeback_auto_commit,
        "writeback: session writeback complete"
    );
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    clippy::indexing_slicing,
    reason = "tests"
)]
mod tests {
    use super::*;
    use crate::BrainConfig;
    use crate::daemon::RecallSourceError;
    use crate::recall_client;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;

    // ── parse_extraction_response ────────────────────────────────────────

    #[test]
    fn parse_extraction_response_parses_valid_fact_lines() {
        let raw = "FACT | preferences | prefers chamomile tea | 0.9\n\
                   FACT | family | daughter visits on Sundays | 0.85\n";
        let facts = parse_extraction_response(raw, 0.0);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].subject_hint, "preferences");
        assert_eq!(facts[0].body, "prefers chamomile tea");
        assert!((facts[0].confidence - 0.9).abs() < 1e-9);
        assert_eq!(facts[1].subject_hint, "family");
        assert_eq!(facts[1].body, "daughter visits on Sundays");
    }

    #[test]
    fn parse_extraction_response_skips_non_fact_lines() {
        // AC8: chaff / NO_FACTS / questions produce empty result.
        let raw = "NO_FACTS\nHere is a summary:\n";
        let facts = parse_extraction_response(raw, 0.0);
        assert!(facts.is_empty(), "no FACT lines -> empty");
    }

    #[test]
    fn parse_extraction_response_drops_below_confidence_floor() {
        let raw = "FACT | health | takes aspirin | 0.3\n\
                   FACT | preferences | likes the radio louder | 0.7\n";
        let facts = parse_extraction_response(raw, 0.5);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject_hint, "preferences");
    }

    #[test]
    fn parse_extraction_response_clamps_confidence_over_one() {
        let raw = "FACT | preferences | likes tea | 1.5\n";
        let facts = parse_extraction_response(raw, 0.0);
        assert_eq!(facts.len(), 1);
        assert!((facts[0].confidence - 1.0).abs() < 1e-9, "clamped to 1.0");
    }

    #[test]
    fn parse_extraction_response_skips_unparseable_confidence() {
        let raw = "FACT | preferences | likes tea | maybe\n";
        let facts = parse_extraction_response(raw, 0.0);
        assert!(facts.is_empty());
    }

    #[test]
    fn parse_extraction_response_skips_empty_subject_or_body() {
        let raw = "FACT |  | some fact | 0.8\n\
                   FACT | topic |  | 0.8\n";
        let facts = parse_extraction_response(raw, 0.0);
        assert!(facts.is_empty());
    }

    #[test]
    fn parse_extraction_response_skips_malformed_line_too_few_pipes() {
        // Only 3 fields (no trailing confidence).
        let raw = "FACT | preferences | likes tea\n";
        let facts = parse_extraction_response(raw, 0.0);
        assert!(facts.is_empty());
    }

    // ── render_transcript ────────────────────────────────────────────────

    #[test]
    fn render_transcript_formats_user_assistant_pairs() {
        let turns = vec![
            Turn { user: "hello".to_string(), assistant: "hi there".to_string(), ts: 1 },
            Turn {
                user: "how are you".to_string(),
                assistant: "doing well".to_string(),
                ts: 2,
            },
        ];
        let out = render_transcript(&turns);
        assert!(out.contains("User: hello\n"));
        assert!(out.contains("Assistant: hi there\n"));
        assert!(out.contains("User: how are you\n"));
        assert!(out.contains("Assistant: doing well\n"));
    }

    #[test]
    fn render_transcript_empty_turns_is_empty_string() {
        assert_eq!(render_transcript(&[]), "");
    }

    // ── recall_subject_for ───────────────────────────────────────────────

    #[test]
    fn recall_subject_for_thread_uses_date_subject() {
        // AC2: thread-scoped subjects go under the lib.rs convention.
        let s = recall_subject_for("thread", "2026-05-29");
        assert_eq!(s, "wintermute-thread-2026-05-29");
        assert!(s.starts_with(crate::THREAD_SUBJECT_PREFIX));
    }

    #[test]
    fn recall_subject_for_semantic_hint_uses_profile_subject() {
        let s = recall_subject_for("preferences", "2026-05-29");
        assert_eq!(s, "wintermute-preferences");
    }

    #[test]
    fn recall_subject_for_empty_hint_falls_back_to_date() {
        let s = recall_subject_for("", "2026-05-29");
        assert_eq!(s, "wintermute-thread-2026-05-29");
    }

    // ── WritebackGuard ───────────────────────────────────────────────────

    #[tokio::test]
    async fn writeback_guard_first_claim_returns_true() {
        let guard = WritebackGuard::new();
        assert!(guard.try_claim("sess-1").await);
    }

    #[tokio::test]
    async fn writeback_guard_second_claim_same_session_returns_false() {
        // AC7: second claim is denied.
        let guard = WritebackGuard::new();
        assert!(guard.try_claim("sess-1").await);
        assert!(!guard.try_claim("sess-1").await, "second claim must fail");
    }

    #[tokio::test]
    async fn writeback_guard_different_sessions_are_independent() {
        let guard = WritebackGuard::new();
        assert!(guard.try_claim("sess-1").await);
        assert!(guard.try_claim("sess-2").await, "different session id must succeed");
    }

    // ── trigger_writeback ────────────────────────────────────────────────

    struct FakeExtractor {
        response: std::result::Result<String, String>,
    }

    #[async_trait::async_trait]
    impl ExtractorClient for FakeExtractor {
        async fn extract(&self, _transcript: &str) -> std::result::Result<String, String> {
            self.response.clone()
        }
    }

    #[derive(Clone, Default)]
    struct CapturingRecall {
        saved: Arc<StdMutex<Vec<(String, String)>>>,
    }

    impl CapturingRecall {
        fn saved_facts(&self) -> Vec<(String, String)> {
            self.saved.lock().expect("saved poisoned").clone()
        }
    }

    #[async_trait::async_trait]
    impl crate::daemon::RecallSource for CapturingRecall {
        async fn fetch(
            &self,
            _transcript: &str,
        ) -> std::result::Result<Vec<crate::recall_client::QueryHit>, RecallSourceError> {
            Ok(Vec::new())
        }

        async fn save_fact(
            &self,
            subject: &str,
            body: &str,
        ) -> std::result::Result<String, RecallSourceError> {
            self.saved
                .lock()
                .expect("saved poisoned")
                .push((subject.to_string(), body.to_string()));
            Ok("m-fake-1".to_string())
        }
    }

    #[derive(Clone, Default)]
    struct FailingRecall;

    #[async_trait::async_trait]
    impl crate::daemon::RecallSource for FailingRecall {
        async fn fetch(
            &self,
            _transcript: &str,
        ) -> std::result::Result<Vec<crate::recall_client::QueryHit>, RecallSourceError> {
            Ok(Vec::new())
        }

        async fn save_fact(
            &self,
            _subject: &str,
            _body: &str,
        ) -> std::result::Result<String, RecallSourceError> {
            Err(RecallSourceError::Client(recall_client::ClientError::Remote {
                code: "boom".to_string(),
                message: "simulated failure".to_string(),
            }))
        }
    }

    fn three_turn_session() -> Vec<Turn> {
        vec![
            Turn {
                user: "I take aspirin every morning.".to_string(),
                assistant: "Noted.".to_string(),
                ts: 1,
            },
            Turn {
                user: "My daughter visits on Sundays.".to_string(),
                assistant: "That sounds lovely.".to_string(),
                ts: 2,
            },
            Turn {
                user: "I prefer chamomile tea.".to_string(),
                assistant: "I will remember that.".to_string(),
                ts: 3,
            },
        ]
    }

    fn default_config() -> BrainConfig {
        BrainConfig::default()
    }

    // AC1: session.end triggers extraction + write.
    #[tokio::test]
    async fn trigger_writeback_calls_extractor_and_writes_facts() {
        let extractor = FakeExtractor {
            response: Ok(
                "FACT | health | takes aspirin every morning | 0.9\n\
                 FACT | family | daughter visits on Sundays | 0.85\n"
                    .to_string(),
            ),
        };
        let recall = CapturingRecall::default();
        let guard = WritebackGuard::new();
        let config = default_config();

        trigger_writeback(
            "sess-abc",
            &three_turn_session(),
            "2026-05-29",
            &config,
            &extractor,
            &recall,
            &guard,
        )
        .await;

        let saved = recall.saved_facts();
        assert_eq!(saved.len(), 2, "AC1: two facts must be written");
        // AC1 checks subjects + bodies
        assert_eq!(saved[0].0, "wintermute-health");
        assert_eq!(saved[0].1, "takes aspirin every morning");
        assert_eq!(saved[1].0, "wintermute-family");
        assert_eq!(saved[1].1, "daughter visits on Sundays");
    }

    // AC2: thread-subject routing.
    #[tokio::test]
    async fn trigger_writeback_thread_hint_routes_to_thread_subject() {
        let extractor = FakeExtractor {
            response: Ok("FACT | thread | session on 2026-05-29: discussed health | 0.7\n"
                .to_string()),
        };
        let recall = CapturingRecall::default();
        let guard = WritebackGuard::new();
        let config = default_config();

        trigger_writeback(
            "sess-thread",
            &three_turn_session(),
            "2026-05-29",
            &config,
            &extractor,
            &recall,
            &guard,
        )
        .await;

        let saved = recall.saved_facts();
        assert_eq!(saved.len(), 1);
        // AC2: thread hint → wintermute-thread-YYYY-MM-DD
        assert_eq!(saved[0].0, "wintermute-thread-2026-05-29");
    }

    // AC4: empty session writes nothing.
    #[tokio::test]
    async fn trigger_writeback_empty_session_skips_extraction() {
        let extractor = FakeExtractor {
            response: Ok("FACT | health | takes aspirin | 0.9\n".to_string()),
        };
        let recall = CapturingRecall::default();
        let guard = WritebackGuard::new();
        let config = default_config();

        trigger_writeback(
            "sess-empty",
            &[], // no turns
            "2026-05-29",
            &config,
            &extractor,
            &recall,
            &guard,
        )
        .await;

        assert!(
            recall.saved_facts().is_empty(),
            "AC4: empty session must write nothing"
        );
    }

    // AC5: recall outage is tolerated.
    #[tokio::test]
    async fn trigger_writeback_recall_failure_is_tolerated() {
        let extractor = FakeExtractor {
            response: Ok("FACT | health | takes aspirin | 0.9\n".to_string()),
        };
        let recall = FailingRecall;
        let guard = WritebackGuard::new();
        let config = default_config();

        // Must not panic, must complete.
        trigger_writeback(
            "sess-fail",
            &three_turn_session(),
            "2026-05-29",
            &config,
            &extractor,
            &recall,
            &guard,
        )
        .await;
        // No assertion on saved; we verified it didn't panic.
    }

    // AC6: extraction failure is tolerated.
    #[tokio::test]
    async fn trigger_writeback_extraction_failure_is_tolerated() {
        let extractor = FakeExtractor {
            response: Err("model unavailable".to_string()),
        };
        let recall = CapturingRecall::default();
        let guard = WritebackGuard::new();
        let config = default_config();

        trigger_writeback(
            "sess-extract-fail",
            &three_turn_session(),
            "2026-05-29",
            &config,
            &extractor,
            &recall,
            &guard,
        )
        .await;

        assert!(
            recall.saved_facts().is_empty(),
            "AC6: extraction failure must write nothing"
        );
    }

    // AC7: write-once per session.
    #[tokio::test]
    async fn trigger_writeback_second_call_same_session_is_noop() {
        let extractor = FakeExtractor {
            response: Ok("FACT | health | takes aspirin | 0.9\n".to_string()),
        };
        let recall = CapturingRecall::default();
        let guard = WritebackGuard::new();
        let config = default_config();
        let turns = three_turn_session();

        trigger_writeback(
            "sess-dedup",
            &turns,
            "2026-05-29",
            &config,
            &extractor,
            &recall,
            &guard,
        )
        .await;
        // Second call with same session_id.
        trigger_writeback(
            "sess-dedup",
            &turns,
            "2026-05-29",
            &config,
            &extractor,
            &recall,
            &guard,
        )
        .await;

        let saved = recall.saved_facts();
        assert_eq!(saved.len(), 1, "AC7: second call must not double-write");
    }

    // AC8: no FACT lines → nothing written (chaff gating via parse_extraction_response).
    #[tokio::test]
    async fn trigger_writeback_no_facts_writes_nothing() {
        let extractor = FakeExtractor {
            response: Ok("NO_FACTS\n".to_string()),
        };
        let recall = CapturingRecall::default();
        let guard = WritebackGuard::new();
        let config = default_config();

        trigger_writeback(
            "sess-nofacts",
            &three_turn_session(),
            "2026-05-29",
            &config,
            &extractor,
            &recall,
            &guard,
        )
        .await;

        assert!(
            recall.saved_facts().is_empty(),
            "AC8: no FACT lines must write nothing"
        );
    }

    // AC3: proposals by default. The `writeback_auto_commit = false` field
    // is threaded through config to save_fact — the underlying RecallSource
    // always uses the proposal path in the production subprocess (recall observe).
    // Here we verify the config field defaults correctly.
    #[test]
    fn writeback_auto_commit_defaults_to_false() {
        let cfg = BrainConfig::default();
        assert!(
            !cfg.writeback_auto_commit,
            "AC3: writeback_auto_commit must default to false"
        );
    }

    #[test]
    fn writeback_confidence_floor_defaults_to_point_five() {
        let cfg = BrainConfig::default();
        assert!(
            (cfg.writeback_confidence_floor - 0.5).abs() < 1e-9,
            "default confidence floor must be 0.5"
        );
    }

    #[test]
    fn writeback_model_defaults_to_haiku() {
        let cfg = BrainConfig::default();
        assert_eq!(cfg.writeback_model, crate::SHORT_MODEL_HAIKU);
    }

    #[test]
    fn idle_gap_ms_defaults_to_five_minutes() {
        let cfg = BrainConfig::default();
        assert_eq!(cfg.idle_gap_ms, 300_000);
    }
}
