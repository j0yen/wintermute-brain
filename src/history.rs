//! Bounded in-memory turn-history buffer for `wmd`.
//!
//! [`History`] holds the last `max_turns` completed `(user, assistant)` pairs
//! in a [`VecDeque`] (oldest first). After each successful reply the caller
//! pushes a [`Turn`] via [`History::push`]; on overflow the oldest entry is
//! evicted automatically.
//!
//! [`History::to_messages`] converts the ring into the flat `[user, assistant,
//! …, current_user]` slice that [`crate::daemon::compose_request`] feeds into
//! the Anthropic Messages API (PRD-wmd-turn-history §2.2).
//!
//! A token-budget guard ([`History::trimmed_messages`]) trims the oldest turns
//! first, ensuring the total rendered length stays under a caller-supplied
//! character cap. The cap is applied before the current user turn is prepended,
//! so the current turn is **never** dropped (PRD-wmd-turn-history AC6).

use std::collections::VecDeque;

use tracing::debug;

use crate::anthropic::{Message, Role};

/// One completed conversation turn: what the user said and what the assistant
/// replied (the *spoken* text, not any trailing JSON fence).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    /// The transcript the user submitted.
    pub user: String,
    /// The assistant's spoken reply (destructive turns store the spoken
    /// prefix, not the fenced JSON block — PRD §2.1).
    pub assistant: String,
    /// Unix milliseconds when this turn completed.
    pub ts: u64,
}

/// Bounded rolling buffer of recent conversation turns.
///
/// `max_turns = 0` disables history and [`History::to_messages`] returns an
/// empty slice (the daemon prepends the current user turn itself, restoring the
/// single-message behaviour — PRD AC4).
#[derive(Debug, Clone)]
pub struct History {
    turns: VecDeque<Turn>,
    /// Maximum number of `(user, assistant)` pairs to retain.
    max_turns: usize,
}

impl History {
    /// Construct an empty history with the given capacity.
    #[must_use]
    pub fn new(max_turns: usize) -> Self {
        Self {
            turns: VecDeque::with_capacity(max_turns.min(64)),
            max_turns,
        }
    }

    /// Push a completed turn onto the ring, evicting the oldest if full.
    ///
    /// When `max_turns == 0` the push is a no-op (history disabled).
    pub fn push(&mut self, turn: Turn) {
        if self.max_turns == 0 {
            return;
        }
        while self.turns.len() >= self.max_turns {
            self.turns.pop_front();
        }
        self.turns.push_back(turn);
    }

    /// Number of turns currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// `true` when the ring holds no turns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Convert history into a flat `[user, assistant, …]` message list
    /// (without the current-user turn). Oldest pair is first.
    ///
    /// The caller is responsible for appending the current `Role::User`
    /// message. The returned slice satisfies
    /// `len == 2 * self.len()` and alternates `User / Assistant`.
    #[must_use]
    pub fn to_messages(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.turns.len() * 2);
        for t in &self.turns {
            out.push(Message { role: Role::User, content: t.user.clone() });
            out.push(Message { role: Role::Assistant, content: t.assistant.clone() });
        }
        out
    }

    /// Return `to_messages()` trimmed to fit within `token_budget` estimated
    /// characters (chars/4 heuristic — PRD §2.4).  Oldest pairs are dropped
    /// first; the current-user turn is never included in this slice and must
    /// be appended by the caller **after** the trim.
    ///
    /// When trimming fires a DEBUG log records how many pairs were dropped.
    #[must_use]
    pub fn trimmed_messages(&self, token_budget: usize) -> Vec<Message> {
        // Estimate: 4 chars per token (rough GPT/Claude heuristic).
        let char_budget = token_budget.saturating_mul(4);

        // Walk from oldest to newest accumulating total chars; keep the
        // newest N turns that fit. Because we want *newest*, walk in
        // reverse and collect, then reverse the result.
        let mut kept: VecDeque<&Turn> = VecDeque::new();
        let mut total_chars: usize = 0;

        for turn in self.turns.iter().rev() {
            let pair_chars = turn.user.len().saturating_add(turn.assistant.len());
            if total_chars.saturating_add(pair_chars) > char_budget {
                break;
            }
            total_chars = total_chars.saturating_add(pair_chars);
            kept.push_front(turn);
        }

        let dropped = self.turns.len().saturating_sub(kept.len());
        if dropped > 0 {
            debug!(
                dropped,
                retained = kept.len(),
                char_budget,
                "wm-brain history: token guard trimmed oldest turns"
            );
        }

        let mut out = Vec::with_capacity(kept.len() * 2);
        for t in &kept {
            out.push(Message { role: Role::User, content: t.user.clone() });
            out.push(Message { role: Role::Assistant, content: t.assistant.clone() });
        }
        out
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new(crate::DEFAULT_HISTORY_TURNS)
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

    fn turn(user: &str, assistant: &str) -> Turn {
        Turn { user: user.to_string(), assistant: assistant.to_string(), ts: 0 }
    }

    // AC2 — ring bound: with max_turns=2, after 5 pushes only 2 remain,
    // oldest evicted first.
    #[test]
    fn ring_bound_evicts_oldest() {
        let mut h = History::new(2);
        for i in 0u32..5 {
            h.push(turn(&format!("u{i}"), &format!("a{i}")));
        }
        assert_eq!(h.len(), 2);
        let msgs = h.to_messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].content, "u3");
        assert_eq!(msgs[1].content, "a3");
        assert_eq!(msgs[2].content, "u4");
        assert_eq!(msgs[3].content, "a4");
    }

    // AC4 — max_turns=0 disables history: push is no-op, to_messages is empty.
    #[test]
    fn disabled_history_is_empty() {
        let mut h = History::new(0);
        h.push(turn("hello", "world"));
        assert!(h.is_empty());
        assert!(h.to_messages().is_empty());
    }

    // AC1 partial — to_messages produces alternating user/assistant pairs.
    #[test]
    fn to_messages_alternates_roles() {
        let mut h = History::new(4);
        h.push(turn("q1", "a1"));
        h.push(turn("q2", "a2"));
        let msgs = h.to_messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::User);
        assert_eq!(msgs[3].role, Role::Assistant);
        assert_eq!(msgs[0].content, "q1");
        assert_eq!(msgs[1].content, "a1");
    }

    // AC6 — token guard trims oldest turns, never drops current user turn.
    #[test]
    fn token_guard_trims_oldest() {
        let mut h = History::new(4);
        // 40-char user + 40-char assistant = 80 chars per turn → 20 tokens.
        let long_user = "x".repeat(40);
        let long_asst = "y".repeat(40);
        for _ in 0..4 {
            h.push(turn(&long_user, &long_asst));
        }
        // Budget of 30 tokens = 120 chars; one pair is 80 chars → fits 1 pair.
        let msgs = h.trimmed_messages(30);
        assert_eq!(msgs.len(), 2, "only 1 pair should fit under the budget");
    }

    // Trimmed messages with a very large budget returns all turns.
    #[test]
    fn token_guard_keeps_all_when_budget_generous() {
        let mut h = History::new(4);
        h.push(turn("short", "reply"));
        h.push(turn("short2", "reply2"));
        let msgs = h.trimmed_messages(10_000);
        assert_eq!(msgs.len(), 4);
    }
}
