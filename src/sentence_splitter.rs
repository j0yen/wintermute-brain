//! Incremental sentence splitter for the streaming TTS path.
//!
//! [`SentenceSplitter`] buffers text deltas from an LLM stream and emits
//! complete sentences so `wm-brain` can publish `wm.brain.reply.partial`
//! events as soon as a sentence boundary is detected — reducing TTS latency
//! from full-response time to first-sentence time.
//!
//! ## Splitting rules
//!
//! A split is emitted at `[.!?]` followed by whitespace or end-of-stream,
//! **excluding**:
//! - Decimals: `3.14` — a digit on both sides of `.` suppresses the split.
//! - Initials: `J. S.` — a single uppercase letter before `.` suppresses the
//!   split.
//! - Ellipses: `...` — three or more consecutive dots suppress the split.
//!
//! PRD-fluid-brain-streaming-tts §2 / AC2.

/// Incremental sentence splitter.
///
/// # Example
///
/// ```
/// use wintermute_brain::sentence_splitter::SentenceSplitter;
///
/// let mut sp = SentenceSplitter::default();
/// let sentences = sp.push("Hello world. How are you?");
/// assert_eq!(sentences, vec!["Hello world.", "How are you?"]);
/// ```
#[derive(Debug, Default)]
pub struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    /// Create a new, empty splitter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `delta` to the internal buffer and return any completed sentences.
    ///
    /// A sentence is complete when a terminator character (`[.!?]`) is followed
    /// by whitespace. The returned sentences include their terminal punctuation
    /// and are trimmed of surrounding whitespace.
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        self.drain_sentences()
    }

    /// Flush any remaining text from the buffer as a final sentence.
    ///
    /// Call this after the LLM stream ends to drain incomplete trailing text
    /// (e.g. replies that don't end in punctuation).
    ///
    /// Returns `None` when the buffer is empty.
    pub fn flush(&mut self) -> Option<String> {
        let trimmed = self.buf.trim().to_string();
        self.buf.clear();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }

    /// Scan the buffer for complete sentence boundaries and drain them.
    fn drain_sentences(&mut self) -> Vec<String> {
        let mut result = Vec::new();
        loop {
            match find_sentence_end(&self.buf) {
                Some(end) => {
                    // `end` is the byte index *after* the terminator char.
                    let sentence = self.buf[..end].trim().to_string();
                    // Remove consumed text (including any leading whitespace after).
                    self.buf = self.buf[end..].trim_start().to_string();
                    if !sentence.is_empty() {
                        result.push(sentence);
                    }
                }
                None => break,
            }
        }
        result
    }
}

/// Find the byte index immediately after the first valid sentence-ending
/// terminator in `s`.
///
/// Returns `None` when no valid split point exists (no terminator found,
/// or every candidate is suppressed by decimal/initial/ellipsis rules).
fn find_sentence_end(s: &str) -> Option<usize> {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    // We need at least one char after the terminator (whitespace) to commit.
    // Exception: we never split without a following whitespace here — that is
    // handled by `flush()` at stream end.
    let mut i = 0usize;
    while i < n {
        let ch = chars[i];
        if is_terminator(ch) {
            // Check for suppression rules.
            if is_ellipsis(&chars, i) {
                // Skip past the whole ellipsis run.
                while i < n && chars[i] == '.' {
                    i += 1;
                }
                continue;
            }
            if ch == '.' {
                if is_decimal(&chars, i) || is_initial(&chars, i) {
                    i += 1;
                    continue;
                }
            }
            // Valid terminator found. Check whether the next non-empty
            // position is whitespace or we are at the very end of the buffer.
            let after = i + 1; // index of first char after terminator
            if after >= n {
                // Terminator is at the end of the buffer — wait for more
                // data (we can't tell if this is a real end or if more
                // tokens will follow without a space). Do NOT split yet;
                // `flush()` handles this case.
                return None;
            }
            // The char immediately after the terminator must be whitespace.
            if chars[after].is_whitespace() {
                // Byte index of the end of the terminator char itself.
                let byte_end = byte_offset(&chars, i + 1);
                return Some(byte_end);
            }
        }
        i += 1;
    }
    None
}

/// `true` when `chars[i]` is `.`, `!`, or `?`.
fn is_terminator(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?')
}

/// `true` when `chars[i]` is part of an ellipsis run (`...` = 3+ dots).
fn is_ellipsis(chars: &[char], i: usize) -> bool {
    if chars[i] != '.' {
        return false;
    }
    // Count consecutive dots around position i.
    let mut start = i;
    while start > 0 && chars[start - 1] == '.' {
        start -= 1;
    }
    let mut end = i;
    while end + 1 < chars.len() && chars[end + 1] == '.' {
        end += 1;
    }
    (end - start + 1) >= 3
}

/// `true` when `chars[i]` is a `.` with a digit on both sides
/// (decimal number like `3.14`).
fn is_decimal(chars: &[char], i: usize) -> bool {
    if chars[i] != '.' {
        return false;
    }
    let left = i.checked_sub(1).map(|j| chars[j]).unwrap_or('\0');
    let right = chars.get(i + 1).copied().unwrap_or('\0');
    left.is_ascii_digit() && right.is_ascii_digit()
}

/// `true` when `chars[i]` is a `.` preceded by exactly one uppercase ASCII
/// letter (an abbreviation or initial like `J.`).
fn is_initial(chars: &[char], i: usize) -> bool {
    if chars[i] != '.' {
        return false;
    }
    // Check left: must be a single uppercase letter.
    let left = match i.checked_sub(1) {
        Some(j) => chars[j],
        None => return false,
    };
    if !left.is_ascii_uppercase() {
        return false;
    }
    // Check two-left: if it exists it must NOT be a letter (otherwise this is
    // part of an all-caps word, not a standalone initial).
    let two_left = i.checked_sub(2).map(|j| chars[j]).unwrap_or('\0');
    !two_left.is_alphabetic()
}

/// Convert a char-index position to a byte offset in the original string.
fn byte_offset(chars: &[char], char_idx: usize) -> usize {
    chars[..char_idx].iter().map(|c| c.len_utf8()).sum()
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

    // ── AC2: SentenceSplitter unit tests ──────────────────────────────────────

    #[test]
    fn ac2_two_sentences_as_two_deltas() {
        // "Hello world. How are you?" pushed as two deltas → two sentences.
        let mut sp = SentenceSplitter::new();
        let first = sp.push("Hello world.");
        assert!(first.is_empty(), "no sentence yet — no trailing space");
        let second = sp.push(" How are you?");
        assert_eq!(second, vec!["Hello world.", "How are you?"]);
    }

    #[test]
    fn ac2_two_sentences_single_push() {
        let mut sp = SentenceSplitter::new();
        let sentences = sp.push("Hello world. How are you? ");
        assert_eq!(sentences, vec!["Hello world.", "How are you?"]);
    }

    #[test]
    fn ac2_decimal_not_split() {
        // "Pi is 3.14. Next." → two sentences, decimal not split.
        let mut sp = SentenceSplitter::new();
        let sentences = sp.push("Pi is 3.14. Next. ");
        assert_eq!(sentences, vec!["Pi is 3.14.", "Next."]);
    }

    #[test]
    fn ac2_ellipsis_not_split() {
        // "..." → not split mid-ellipsis.
        let mut sp = SentenceSplitter::new();
        let sentences = sp.push("Wait... okay. ");
        // "Wait..." should NOT be a sentence; "Wait... okay." should be.
        assert_eq!(sentences, vec!["Wait... okay."]);
    }

    #[test]
    fn ac2_ellipsis_three_dots_no_split() {
        let mut sp = SentenceSplitter::new();
        let sentences = sp.push("Hmm... ");
        // Trailing ellipsis with space — none of the dots are valid terminators.
        assert!(sentences.is_empty(), "ellipsis must not split");
    }

    #[test]
    fn ac2_flush_no_punct() {
        // flush() on "no punct" → Some("no punct").
        let mut sp = SentenceSplitter::new();
        let partial = sp.push("no punct");
        assert!(partial.is_empty());
        assert_eq!(sp.flush(), Some("no punct".to_string()));
    }

    #[test]
    fn flush_empty_buffer_returns_none() {
        let mut sp = SentenceSplitter::new();
        assert!(sp.flush().is_none());
    }

    #[test]
    fn flush_after_partial_sentence() {
        let mut sp = SentenceSplitter::new();
        sp.push("Hello world");
        assert_eq!(sp.flush(), Some("Hello world".to_string()));
        // Buffer should now be clear.
        assert!(sp.flush().is_none());
    }

    #[test]
    fn exclamation_and_question_split() {
        let mut sp = SentenceSplitter::new();
        let sentences = sp.push("Wow! Great? Yes. ");
        assert_eq!(sentences, vec!["Wow!", "Great?", "Yes."]);
    }

    #[test]
    fn initial_not_split_j_s() {
        // "J. S. Bach lived long." → one sentence.
        let mut sp = SentenceSplitter::new();
        let sentences = sp.push("J. S. Bach lived long. ");
        assert_eq!(sentences, vec!["J. S. Bach lived long."]);
    }

    #[test]
    fn multi_char_acronym_does_split() {
        // "NASA. Good." — NASA is not a single initial, so NASA. splits.
        let mut sp = SentenceSplitter::new();
        let sentences = sp.push("NASA. Good. ");
        assert_eq!(sentences, vec!["NASA.", "Good."]);
    }

    #[test]
    fn incremental_push_accumulates() {
        let mut sp = SentenceSplitter::new();
        assert!(sp.push("Hello").is_empty());
        assert!(sp.push(" world").is_empty());
        assert!(sp.push(".").is_empty()); // no whitespace after yet
        let sentences = sp.push(" Next sentence. ");
        assert_eq!(sentences, vec!["Hello world.", "Next sentence."]);
    }

    #[test]
    fn push_empty_delta_is_noop() {
        let mut sp = SentenceSplitter::new();
        sp.push("Hello.");
        let r = sp.push("");
        assert!(r.is_empty());
    }

    #[test]
    fn unicode_text_splits_correctly() {
        let mut sp = SentenceSplitter::new();
        let sentences = sp.push("Hola. ¿Cómo estás? ");
        assert_eq!(sentences, vec!["Hola.", "¿Cómo estás?"]);
    }

    #[test]
    fn buffer_cleared_after_flush() {
        let mut sp = SentenceSplitter::new();
        sp.push("Hello");
        sp.flush();
        assert!(sp.flush().is_none(), "flush of empty buffer is None");
    }
}
