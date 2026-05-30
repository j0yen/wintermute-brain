//! Conversation session boundary tracking for `wmd`.
//!
//! A "session" is a coherent conversational span. Without explicit session
//! boundaries, the turn-history ring mixes unrelated conversations — a user
//! asking "say that again" after a 20-minute gap means their *last* turn, not
//! whatever happened to still be in a fixed ring from an earlier exchange.
//!
//! Session inference ([`SessionTracker`]) runs entirely brain-side. It derives
//! boundaries from two signals:
//!
//! 1. **Gap-based close**: when `now_ms − last_turn_ms > idle_gap_ms`
//!    (default 5 min), the existing session closes and a new one opens.
//! 2. **Explicit close phrases**: a small keyword matcher detects utterances
//!    like "goodbye" or "go to sleep"; the current session closes *after* the
//!    reply is published rather than waiting for the idle gap.
//!
//! Session events are emitted as
//! [`crate::bus::outgoing::SESSION_START`] / [`crate::bus::outgoing::SESSION_END`]
//! on the agorabus. The payloads are plain JSON — see [`SessionStartPayload`]
//! and [`SessionEndPayload`]. The history ring is cleared on each session start
//! so context never leaks across session boundaries.
//!
//! PRD-wmd-session-boundary §2.

use serde::{Deserialize, Serialize};

/// Stable close-reason tokens. Keep in sync with the PRD's JSON schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CloseReason {
    /// Closed because `now_ms − last_turn_ms > idle_gap_ms`.
    Idle,
    /// Closed because the user uttered a recognised end-of-conversation phrase.
    Explicit,
    /// Closed on graceful daemon shutdown.
    Shutdown,
}

/// Live session data retained in [`SessionTracker`].
#[derive(Debug, Clone)]
pub struct Session {
    /// Stable id: `"wmd-sess-{first_turn_ts_ms}"` (minted at session open).
    pub id: String,
    /// Unix milliseconds when the session opened.
    pub started_ms: u64,
    /// Unix milliseconds of the most recent turn.
    pub last_turn_ms: u64,
    /// Number of turns logged to this session so far.
    pub turn_count: u32,
}

/// Outcome returned by [`SessionTracker::advance`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdvanceOutcome {
    /// A fresh session was opened (and an existing one may have been closed
    /// first). The caller must emit SESSION_END (if `closed` is `Some`) then
    /// SESSION_START, then clear the history ring.
    NewSession {
        /// Closed session payload, or `None` when this is the very first turn.
        closed: Option<SessionEndPayload>,
        /// Payload for the newly opened session's SESSION_START event.
        opened: SessionStartPayload,
    },
    /// An existing session was extended: `last_turn_ms` and `turn_count` were
    /// bumped. No bus events to emit; no history reset needed.
    Extended,
}

/// Wire payload for `wm.brain.session.start`. PRD §2.3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionStartPayload {
    /// The newly opened session's id.
    pub session_id: String,
    /// Unix milliseconds when the session opened.
    pub ts: u64,
}

/// Wire payload for `wm.brain.session.end`. PRD §2.3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEndPayload {
    /// The closed session's id.
    pub session_id: String,
    /// Unix milliseconds when the session closed.
    pub ts: u64,
    /// Number of turns that completed within this session.
    pub turn_count: u32,
    /// Why the session closed.
    pub reason: CloseReason,
}

/// Default idle gap (milliseconds) before a turn opens a new session.
/// 5 minutes per PRD §2.1.
pub const DEFAULT_IDLE_GAP_MS: u64 = 300_000;

/// Default end-of-conversation phrases matched case-insensitively
/// (punctuation and leading/trailing whitespace stripped before comparison).
pub const DEFAULT_END_PHRASES: &[&str] = &[
    "goodbye",
    "good bye",
    "that's all",
    "thats all",
    "never mind",
    "nevermind",
    "thanks that's it",
    "thanks thats it",
    "go to sleep",
];

/// Stateful session boundary tracker.
///
/// Call [`SessionTracker::advance`] on every inbound `wm.dialog.turn.user`.
/// The tracker mutates its own state and returns the bus events the caller
/// must emit (plus whether to reset the history ring).
///
/// Call [`SessionTracker::close_for_shutdown`] on graceful daemon exit to emit
/// a `SESSION_END(reason=shutdown)` for any open session.
#[derive(Debug)]
pub struct SessionTracker {
    /// Currently open session, or `None` before the first turn.
    current: Option<Session>,
    /// Gap threshold in milliseconds.
    idle_gap_ms: u64,
    /// Lower-cased, punctuation-stripped end-of-conversation phrases.
    end_phrases: Vec<String>,
}

impl SessionTracker {
    /// Build a tracker with the given idle gap and phrase list.
    #[must_use]
    pub fn new(idle_gap_ms: u64, end_phrases: Vec<String>) -> Self {
        let end_phrases = end_phrases.iter().map(|p| normalize_phrase(p)).collect();
        Self { current: None, idle_gap_ms, end_phrases }
    }

    /// Build a tracker using library defaults (`DEFAULT_IDLE_GAP_MS`,
    /// `DEFAULT_END_PHRASES`).
    #[must_use]
    pub fn default_phrases() -> Self {
        Self::new(
            DEFAULT_IDLE_GAP_MS,
            DEFAULT_END_PHRASES.iter().map(|s| (*s).to_string()).collect(),
        )
    }

    /// Return the current session id, if any. Mainly for tests.
    #[must_use]
    pub fn current_session_id(&self) -> Option<&str> {
        self.current.as_ref().map(|s| s.id.as_str())
    }

    /// Return the turn count of the current session. `0` when no session is
    /// open.
    #[must_use]
    pub fn current_turn_count(&self) -> u32 {
        self.current.as_ref().map_or(0, |s| s.turn_count)
    }

    /// Advance the tracker for one `wm.dialog.turn.user` with timestamp
    /// `now_ms`.
    ///
    /// Returns an [`AdvanceOutcome`] describing what bus events to emit. The
    /// caller is also responsible for checking
    /// [`SessionTracker::take_pending_close`] *after* publishing the reply, to
    /// handle the explicit-close path (where the session closes after the reply
    /// rather than before the next turn).
    pub fn advance(&mut self, now_ms: u64) -> AdvanceOutcome {
        match self.current.as_mut() {
            Some(sess) if now_ms.saturating_sub(sess.last_turn_ms) <= self.idle_gap_ms => {
                // Within gap: extend the existing session.
                sess.last_turn_ms = now_ms;
                sess.turn_count = sess.turn_count.saturating_add(1);
                AdvanceOutcome::Extended
            }
            _ => {
                // Either no session yet, or gap exceeded: close any open session
                // and open a new one.
                let closed = self.current.take().map(|old| SessionEndPayload {
                    session_id: old.id,
                    ts: now_ms,
                    turn_count: old.turn_count,
                    reason: CloseReason::Idle,
                });
                let new_id = format!("wmd-sess-{now_ms}");
                let opened = SessionStartPayload { session_id: new_id.clone(), ts: now_ms };
                self.current = Some(Session {
                    id: new_id,
                    started_ms: now_ms,
                    last_turn_ms: now_ms,
                    turn_count: 1,
                });
                AdvanceOutcome::NewSession { closed, opened }
            }
        }
    }

    /// Detect whether `transcript` is an explicit close phrase.
    ///
    /// Returns `true` when the normalised transcript exactly matches one of the
    /// configured end phrases. The caller should call this *before* dispatching
    /// the turn to the LLM; the session closes *after* the reply is published
    /// (the model gets to say goodbye).
    #[must_use]
    pub fn is_explicit_close(&self, transcript: &str) -> bool {
        let normalised = normalize_phrase(transcript);
        self.end_phrases.iter().any(|p| *p == normalised)
    }

    /// Close the current session with the given reason and return its payload.
    /// Returns `None` if no session is currently open.
    ///
    /// After this call [`SessionTracker::current_session_id`] returns `None`.
    pub fn close(&mut self, now_ms: u64, reason: CloseReason) -> Option<SessionEndPayload> {
        self.current.take().map(|sess| SessionEndPayload {
            session_id: sess.id,
            ts: now_ms,
            turn_count: sess.turn_count,
            reason,
        })
    }
}

/// Strip punctuation and normalise whitespace, then lower-case — used for
/// phrase matching so "Goodbye!" and "goodbye" both hit.
fn normalize_phrase(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
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

    fn tracker_5min() -> SessionTracker {
        SessionTracker::default_phrases()
    }

    // -----------------------------------------------------------------------
    // AC1 — gap opens a new session
    // Two turns 6 minutes apart (idle_gap = 5 min) produce:
    //   session.start → session.end(reason=idle) → session.start
    // -----------------------------------------------------------------------
    #[test]
    fn ac1_gap_opens_new_session() {
        let mut t = tracker_5min();
        let min_ms: u64 = 60_000;

        // Turn 1 at t=0 → new session.
        let out1 = t.advance(0);
        let (_, s1) = match &out1 {
            AdvanceOutcome::NewSession { closed, opened } => (closed.as_ref(), opened.clone()),
            _ => panic!("expected NewSession for first turn"),
        };
        let sess1_id = s1.session_id.clone();
        assert!(
            matches!(&out1, AdvanceOutcome::NewSession { closed: None, .. }),
            "no prior session to close"
        );

        // Turn 2 at t=6min → gap exceeded → close old, open new.
        let out2 = t.advance(6 * min_ms);
        match &out2 {
            AdvanceOutcome::NewSession { closed: Some(end), opened } => {
                assert_eq!(end.session_id, sess1_id);
                assert_eq!(end.reason, CloseReason::Idle);
                assert_eq!(end.turn_count, 1);
                assert_ne!(opened.session_id, sess1_id, "new session id must differ");
            }
            other => panic!("expected NewSession with closed Some(_), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // AC2 — turns within the gap stay in one session
    // Three turns 1 minute apart → Extended, Extended on turns 2 and 3.
    // -----------------------------------------------------------------------
    #[test]
    fn ac2_turns_within_gap_stay_in_one_session() {
        let mut t = tracker_5min();
        let min_ms: u64 = 60_000;

        let out1 = t.advance(0);
        assert!(matches!(out1, AdvanceOutcome::NewSession { closed: None, .. }));
        let sess_id = t.current_session_id().unwrap().to_string();

        let out2 = t.advance(min_ms);
        assert!(matches!(out2, AdvanceOutcome::Extended));
        assert_eq!(t.current_session_id().unwrap(), sess_id);

        let out3 = t.advance(2 * min_ms);
        assert!(matches!(out3, AdvanceOutcome::Extended));
        assert_eq!(t.current_session_id().unwrap(), sess_id);
        assert_eq!(t.current_turn_count(), 3);
    }

    // -----------------------------------------------------------------------
    // AC3 — explicit close phrase detection
    // is_explicit_close must return true for configured phrases (case and
    // punctuation insensitive) and false for other transcripts.
    // -----------------------------------------------------------------------
    #[test]
    fn ac3_explicit_close_phrase_detection() {
        let t = tracker_5min();
        assert!(t.is_explicit_close("goodbye"), "bare phrase");
        assert!(t.is_explicit_close("Goodbye!"), "upper-case + punctuation");
        assert!(t.is_explicit_close("  GOODBYE  "), "surrounding whitespace");
        assert!(t.is_explicit_close("go to sleep"), "multi-word phrase");
        assert!(t.is_explicit_close("Go To Sleep."), "upper-case multi-word");
        assert!(!t.is_explicit_close("hello there"), "non-close phrase");
        assert!(!t.is_explicit_close("good morning"), "prefix mismatch");
    }

    // -----------------------------------------------------------------------
    // AC4 — session.end carries turn_count
    // A session with 4 turns emits turn_count=4 on close.
    // -----------------------------------------------------------------------
    #[test]
    fn ac4_session_end_carries_turn_count() {
        let mut t = tracker_5min();
        let min_ms: u64 = 60_000;

        t.advance(0);
        t.advance(min_ms);
        t.advance(2 * min_ms);
        t.advance(3 * min_ms);
        assert_eq!(t.current_turn_count(), 4);

        // Close explicitly (shutdown path).
        let end = t
            .close(4 * min_ms, CloseReason::Shutdown)
            .expect("session was open");
        assert_eq!(end.turn_count, 4);
        assert_eq!(end.reason, CloseReason::Shutdown);
    }

    // -----------------------------------------------------------------------
    // AC5 — shutdown closes the open session
    // close(reason=Shutdown) on an open tracker returns a payload; on an
    // already-closed tracker returns None.
    // -----------------------------------------------------------------------
    #[test]
    fn ac5_shutdown_closes_open_session() {
        let mut t = tracker_5min();
        t.advance(0);
        let end = t.close(100, CloseReason::Shutdown);
        assert!(end.is_some(), "should produce a payload");
        let end = end.unwrap();
        assert_eq!(end.reason, CloseReason::Shutdown);
        assert!(t.current_session_id().is_none(), "tracker now closed");

        // Second close on already-closed tracker returns None.
        let none = t.close(200, CloseReason::Shutdown);
        assert!(none.is_none());
    }

    // -----------------------------------------------------------------------
    // normalize_phrase — internal helper sanity-check
    // -----------------------------------------------------------------------
    #[test]
    fn normalize_strips_punctuation_and_lowercases() {
        assert_eq!(normalize_phrase("Goodbye!"), "goodbye");
        assert_eq!(normalize_phrase("  Go To Sleep.  "), "go to sleep");
        assert_eq!(normalize_phrase("That's all!"), "thats all");
    }

    // -----------------------------------------------------------------------
    // custom phrases round-trip
    // -----------------------------------------------------------------------
    #[test]
    fn custom_phrases_respected() {
        let t = SessionTracker::new(
            60_000,
            vec!["done for now".to_string(), "signing off".to_string()],
        );
        assert!(t.is_explicit_close("Done for now."));
        assert!(t.is_explicit_close("signing off"));
        assert!(!t.is_explicit_close("goodbye"), "not in this custom list");
    }

    // -----------------------------------------------------------------------
    // session id is minted from first-turn ts
    // -----------------------------------------------------------------------
    #[test]
    fn session_id_contains_ts() {
        let mut t = tracker_5min();
        t.advance(12345);
        assert_eq!(
            t.current_session_id().expect("session open"),
            "wmd-sess-12345"
        );
    }

    // -----------------------------------------------------------------------
    // close returns correct payload
    // -----------------------------------------------------------------------
    #[test]
    fn close_explicit_reason_and_ts() {
        let mut t = tracker_5min();
        t.advance(1000);
        t.advance(2000);
        let end = t.close(3000, CloseReason::Explicit).expect("session open");
        assert_eq!(end.reason, CloseReason::Explicit);
        assert_eq!(end.ts, 3000);
        assert_eq!(end.turn_count, 2);
    }
}
