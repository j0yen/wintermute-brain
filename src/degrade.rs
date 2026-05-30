//! Graceful-degradation phrase bank, error aggregator, and health snapshot
//! for `wm-brain` (PRD-wintermute-companion-degrade).
//!
//! When any component publishes a `wm.*.error` event with a `kind` field,
//! the aggregator looks up the phrase, checks the per-kind 30-second
//! rate-limit window, and—if allowed—publishes `wm.tts.speak` with
//! `priority: "system"` so the companion tells the user what went wrong
//! rather than going silent.
//!
//! A second ticker (every 60 s) emits a `wm.health.snapshot` with each
//! component's last-known state.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ── Topic constants ────────────────────────────────────────────────────────────

/// TTS speak command — published by the aggregator with `priority:"system"`.
pub const TTS_SPEAK_TOPIC: &str = "wm.tts.speak";

/// Health snapshot published every 60 s.
pub const HEALTH_SNAPSHOT_TOPIC: &str = "wm.health.snapshot";

/// Inbound error topic subscription prefix (covers all four error topics).
pub const ERROR_TOPIC_PREFIX: &str = "wm.";

/// Individual inbound error topics we aggregate.
pub const STT_ERROR_TOPIC: &str = "wm.stt.error";
/// TTS error topic.
pub const TTS_ERROR_TOPIC: &str = "wm.tts.error";
/// Audio error topic.
pub const AUDIO_ERROR_TOPIC: &str = "wm.audio.error";
/// Brain error topic (self-emitted, deduplicated by the rate-limiter).
pub const BRAIN_ERROR_TOPIC: &str = "wm.brain.error";

/// Self-emitted `wm.health.*` topics must appear in the brain's own
/// allow-list so the bus doesn't reject them (PRD §2.5).
pub const HEALTH_TOPIC_PREFIX: &str = "wm.health.";

/// Rate-limit window: the same `kind` will not be spoken again within
/// this many milliseconds.
pub const RATE_LIMIT_MS: u64 = 30_000;

/// Health snapshot interval in milliseconds.
pub const HEALTH_SNAPSHOT_INTERVAL_MS: u64 = 60_000;

// ── Phrase bank ────────────────────────────────────────────────────────────────

/// Look up the spoken phrase for a degrade `kind`.
///
/// Returns an empty string for `"tts_pw_cat_missing"` (we can't speak
/// through the TTS path if TTS itself is broken); returns the generic
/// fallback for any unrecognised kind.
#[must_use]
pub fn degrade_phrase(kind: &str) -> &'static str {
    match kind {
        "brain_unreachable" => "I can't reach my brain right now. Try again in a moment.",
        "brain_api_key_missing" => "I'm not configured to think yet. Could you ask jsy?",
        "stt_window_invalid" => "Sorry, I didn't catch that.",
        "stt_model_missing" => "My ears aren't installed yet.",
        "audio_mic_missing" => "I lost my microphone. Hold on.",
        "audio_aec_missing" => {
            "My echo cancellation isn't working; I might hear myself."
        }
        // TTS path is broken — publishing via TTS is futile; callers
        // should log but not enqueue a speak command.
        "tts_pw_cat_missing" => "",
        "network_down" => "I can't reach the network. I'll wait.",
        "general_error" => "Something went wrong. Let me try again.",
        _ => "Something I haven't seen before just happened.",
    }
}

// ── Per-kind rate-limit state ──────────────────────────────────────────────────

/// Per-kind last-spoken timestamp, guarded by a std `Mutex` for use in sync
/// contexts (the aggregator only mutates it from within an async context
/// after a `.lock().unwrap_or_else(...)` guard).
#[derive(Debug, Default)]
pub struct RateLimitState {
    /// Maps `kind` → last-spoken Unix millisecond timestamp.
    last_spoken: Mutex<HashMap<String, u64>>,
}

impl RateLimitState {
    /// Create a new, empty rate-limit state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check whether a phrase for `kind` may be spoken at `now_ms`, and
    /// record the speech if allowed.
    ///
    /// Returns `true` if the phrase should be spoken (the last-spoken time is
    /// absent or older than [`RATE_LIMIT_MS`]); `false` if it is suppressed.
    #[must_use]
    pub fn check_and_record(&self, kind: &str, now_ms: u64) -> bool {
        let mut map = self
            .last_spoken
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match map.entry(kind.to_string()) {
            // First time this kind has been seen — always allow.
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(now_ms);
                true
            }
            // Already seen — allow only if the rate-limit window has elapsed.
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let last = *e.get();
                if now_ms.saturating_sub(last) >= RATE_LIMIT_MS {
                    *e.get_mut() = now_ms;
                    true
                } else {
                    false
                }
            }
        }
    }
}

// ── Health state ───────────────────────────────────────────────────────────────

/// One component's health entry inside a [`HealthSnapshot`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComponentHealth {
    /// Component name: `"audio"`, `"stt"`, `"tts"`, or `"brain"`.
    pub component: String,
    /// Current state: `"ok"`, `"degraded"`, or `"down"`.
    pub state: String,
    /// Last error `kind` seen (empty if none).
    pub last_error: String,
    /// Unix millisecond timestamp of last state update.
    pub last_seen_ts: u64,
}

impl ComponentHealth {
    /// Build an "ok" entry for `component` at `ts`.
    #[must_use]
    pub fn ok(component: &str, ts: u64) -> Self {
        Self {
            component: component.to_string(),
            state: "ok".to_string(),
            last_error: String::new(),
            last_seen_ts: ts,
        }
    }

    /// Build a "degraded" entry.
    #[must_use]
    pub fn degraded(component: &str, kind: &str, ts: u64) -> Self {
        Self {
            component: component.to_string(),
            state: "degraded".to_string(),
            last_error: kind.to_string(),
            last_seen_ts: ts,
        }
    }
}

/// Full health snapshot published to [`HEALTH_SNAPSHOT_TOPIC`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HealthSnapshot {
    /// One entry per component.
    pub components: Vec<ComponentHealth>,
    /// Unix milliseconds at snapshot time.
    pub ts: u64,
}

/// Mutable health state tracked by the aggregator.
#[derive(Debug)]
pub struct HealthState {
    inner: Mutex<HashMap<String, ComponentHealth>>,
}

impl HealthState {
    /// Create a new health state pre-populated with all four components in
    /// the `"ok"` state.
    #[must_use]
    pub fn new(ts: u64) -> Self {
        let mut map = HashMap::new();
        for component in ["audio", "stt", "tts", "brain"] {
            map.insert(
                component.to_string(),
                ComponentHealth::ok(component, ts),
            );
        }
        Self {
            inner: Mutex::new(map),
        }
    }

    /// Record an error for `component`.
    pub fn record_error(&self, component: &str, kind: &str, ts: u64) {
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let entry = map.entry(component.to_string()).or_insert_with(|| {
            ComponentHealth::ok(component, ts)
        });
        entry.state = "degraded".to_string();
        entry.last_error = kind.to_string();
        entry.last_seen_ts = ts;
    }

    /// Build a snapshot from current state.
    #[must_use]
    pub fn snapshot(&self, ts: u64) -> HealthSnapshot {
        let map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut components: Vec<ComponentHealth> = map.values().cloned().collect();
        // Stable order for deterministic tests.
        components.sort_by(|a, b| a.component.cmp(&b.component));
        HealthSnapshot { components, ts }
    }
}

// ── Aggregator helper ─────────────────────────────────────────────────────────

/// Identify the component name from an inbound error topic.
///
/// Returns `None` for topics that are not one of the four known error topics.
#[must_use]
pub fn component_for_error_topic(topic: &str) -> Option<&'static str> {
    match topic {
        STT_ERROR_TOPIC => Some("stt"),
        TTS_ERROR_TOPIC => Some("tts"),
        AUDIO_ERROR_TOPIC => Some("audio"),
        BRAIN_ERROR_TOPIC => Some("brain"),
        _ => None,
    }
}

/// Build the JSON payload for a `wm.tts.speak` system-priority command.
#[must_use]
pub fn speak_payload(text: &str, ts: u64) -> serde_json::Value {
    serde_json::json!({
        "text": text,
        "priority": "system",
        "ts": ts,
    })
}

/// Build the JSON payload for a `wm.health.snapshot` envelope.
///
/// # Errors
/// Returns a `serde_json::Error` if serialisation fails (should never
/// happen for `HealthSnapshot`).
pub fn snapshot_payload(snapshot: &HealthSnapshot) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::to_value(snapshot)
}

/// Process one inbound error envelope through the aggregator.
///
/// - Identifies the component from the `topic`.
/// - Extracts the `kind` field from `data` (falls back to `"general_error"`).
/// - Updates health state.
/// - Checks the rate limiter.
/// - If allowed, logs and returns the phrase to be spoken (or `None` if
///   the kind maps to an empty phrase or is rate-limited).
///
/// The caller is responsible for publishing the speak command.
#[must_use]
pub fn process_error_envelope(
    topic: &str,
    data: &serde_json::Value,
    rate: &RateLimitState,
    health: &HealthState,
    now_ms: u64,
) -> Option<String> {
    let component = component_for_error_topic(topic)?;
    let kind = data
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("general_error");

    health.record_error(component, kind, now_ms);

    let phrase = degrade_phrase(kind);
    if phrase.is_empty() {
        // e.g. tts_pw_cat_missing — can't speak it; log only.
        warn!(topic, kind, component, "wm-brain degrade: silent error (TTS path broken)");
        return None;
    }

    if !rate.check_and_record(kind, now_ms) {
        debug!(topic, kind, component, "wm-brain degrade: rate-limited; suppressing phrase");
        return None;
    }

    info!(topic, kind, component, phrase, "wm-brain degrade: speaking error phrase");
    Some(phrase.to_string())
}

// ── Tests ──────────────────────────────────────────────────────────────────────

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

    // AC1 / test 1: phrase lookup returns expected strings.
    #[test]
    fn phrase_lookup_known_kinds() {
        assert!(degrade_phrase("brain_unreachable").contains("brain"));
        assert!(degrade_phrase("brain_api_key_missing").contains("configured"));
        assert!(degrade_phrase("stt_window_invalid").contains("catch"));
        assert!(degrade_phrase("stt_model_missing").contains("ears"));
        assert!(degrade_phrase("audio_mic_missing").contains("microphone"));
        assert!(degrade_phrase("audio_aec_missing").contains("echo"));
        assert!(degrade_phrase("tts_pw_cat_missing").is_empty()); // silent
        assert!(degrade_phrase("network_down").contains("network"));
        assert!(degrade_phrase("general_error").contains("wrong"));
    }

    // AC6: unknown kind falls back to generic.
    #[test]
    fn phrase_lookup_unknown_kind_fallback() {
        let phrase = degrade_phrase("unknown_specific");
        assert!(phrase.contains("haven't seen"), "expected generic fallback, got: {phrase:?}");
    }

    // AC4: rate-limiter suppresses repeated events within 30 s.
    #[test]
    fn rate_limit_suppresses_within_window() {
        let rl = RateLimitState::new();
        let t0 = 1_000_000u64;
        assert!(rl.check_and_record("stt_window_invalid", t0), "first: allowed");
        assert!(
            !rl.check_and_record("stt_window_invalid", t0 + 1_000),
            "second within 30s: suppressed"
        );
        assert!(
            !rl.check_and_record("stt_window_invalid", t0 + RATE_LIMIT_MS - 1),
            "just before window: still suppressed"
        );
        // Exactly at the boundary is allowed (saturating_sub == RATE_LIMIT_MS).
        assert!(
            rl.check_and_record("stt_window_invalid", t0 + RATE_LIMIT_MS),
            "after window: allowed again"
        );
    }

    // AC4: 10 identical events in 5 s → only 1 allowed.
    #[test]
    fn rate_limit_ten_events_five_seconds() {
        let rl = RateLimitState::new();
        let base = 2_000_000u64;
        let mut allowed = 0u32;
        for i in 0..10u64 {
            if rl.check_and_record("stt_window_invalid", base + i * 500) {
                allowed += 1;
            }
        }
        assert_eq!(allowed, 1, "only the first should be allowed; got {allowed}");
    }

    // AC5: different error kinds do NOT suppress each other.
    #[test]
    fn rate_limit_different_kinds_independent() {
        let rl = RateLimitState::new();
        let t0 = 1_000_000u64;
        assert!(rl.check_and_record("stt_window_invalid", t0), "stt: first allowed");
        assert!(rl.check_and_record("audio_mic_missing", t0 + 100), "audio: still allowed");
        assert!(!rl.check_and_record("stt_window_invalid", t0 + 200), "stt: suppressed");
        assert!(!rl.check_and_record("audio_mic_missing", t0 + 300), "audio: suppressed");
    }

    // AC3 + AC6: aggregator routes unknown kind to generic phrase.
    #[test]
    fn aggregator_unknown_kind_routes_generic() {
        let rl = RateLimitState::new();
        let health = HealthState::new(0);
        let data = serde_json::json!({ "kind": "unknown_specific" });
        let phrase = process_error_envelope(
            STT_ERROR_TOPIC, &data, &rl, &health, 1_000,
        );
        assert!(phrase.is_some(), "unknown kind should produce a generic phrase");
        assert!(phrase.unwrap().contains("haven't seen"));
    }

    // tts_pw_cat_missing → None (silent).
    #[test]
    fn aggregator_silent_kind_returns_none() {
        let rl = RateLimitState::new();
        let health = HealthState::new(0);
        let data = serde_json::json!({ "kind": "tts_pw_cat_missing" });
        let result = process_error_envelope(
            TTS_ERROR_TOPIC, &data, &rl, &health, 1_000,
        );
        assert!(result.is_none(), "tts_pw_cat_missing must be silent");
    }

    // Unknown topic → None.
    #[test]
    fn aggregator_unknown_topic_returns_none() {
        let rl = RateLimitState::new();
        let health = HealthState::new(0);
        let data = serde_json::json!({ "kind": "general_error" });
        let result = process_error_envelope(
            "wm.unknown.error", &data, &rl, &health, 1_000,
        );
        assert!(result.is_none(), "unknown topic should produce None");
    }

    // component_for_error_topic covers all four.
    #[test]
    fn component_for_error_topic_all_four() {
        assert_eq!(component_for_error_topic(STT_ERROR_TOPIC), Some("stt"));
        assert_eq!(component_for_error_topic(TTS_ERROR_TOPIC), Some("tts"));
        assert_eq!(component_for_error_topic(AUDIO_ERROR_TOPIC), Some("audio"));
        assert_eq!(component_for_error_topic(BRAIN_ERROR_TOPIC), Some("brain"));
        assert_eq!(component_for_error_topic("wm.other.error"), None);
    }

    // AC7: health snapshot covers all four components.
    #[test]
    fn health_snapshot_has_all_four_components() {
        let hs = HealthState::new(1_000);
        let snap = hs.snapshot(2_000);
        let names: Vec<&str> = snap.components.iter().map(|c| c.component.as_str()).collect();
        assert!(names.contains(&"audio"), "audio missing");
        assert!(names.contains(&"stt"), "stt missing");
        assert!(names.contains(&"tts"), "tts missing");
        assert!(names.contains(&"brain"), "brain missing");
        assert_eq!(snap.ts, 2_000);
    }

    // Health state records error and updates component state.
    #[test]
    fn health_state_records_error_and_degrades() {
        let hs = HealthState::new(0);
        hs.record_error("stt", "stt_window_invalid", 5_000);
        let snap = hs.snapshot(5_000);
        let stt = snap.components.iter().find(|c| c.component == "stt").unwrap();
        assert_eq!(stt.state, "degraded");
        assert_eq!(stt.last_error, "stt_window_invalid");
        assert_eq!(stt.last_seen_ts, 5_000);
    }

    // speak_payload shape.
    #[test]
    fn speak_payload_has_system_priority() {
        let p = speak_payload("hello world", 9_999);
        assert_eq!(p["text"], "hello world");
        assert_eq!(p["priority"], "system");
        assert_eq!(p["ts"], 9_999);
    }

    // snapshot_payload round-trips.
    #[test]
    fn snapshot_payload_round_trips() {
        let hs = HealthState::new(1_000);
        hs.record_error("audio", "audio_mic_missing", 2_000);
        let snap = hs.snapshot(3_000);
        let value = snapshot_payload(&snap).expect("serialises");
        let back: HealthSnapshot = serde_json::from_value(value).expect("deserialises");
        assert_eq!(back, snap);
    }

    // Aggregator processes error and updates health state.
    #[test]
    fn aggregator_updates_health_state_on_error() {
        let rl = RateLimitState::new();
        let health = HealthState::new(0);
        let data = serde_json::json!({ "kind": "audio_mic_missing" });
        let _ = process_error_envelope(AUDIO_ERROR_TOPIC, &data, &rl, &health, 1_000);
        let snap = health.snapshot(1_000);
        let audio = snap.components.iter().find(|c| c.component == "audio").unwrap();
        assert_eq!(audio.state, "degraded");
        assert_eq!(audio.last_error, "audio_mic_missing");
    }

    // Rate-limit window constant sanity.
    #[test]
    fn rate_limit_constant_is_thirty_seconds() {
        assert_eq!(RATE_LIMIT_MS, 30_000);
    }

    // Health snapshot interval sanity.
    #[test]
    fn health_snapshot_interval_is_sixty_seconds() {
        assert_eq!(HEALTH_SNAPSHOT_INTERVAL_MS, 60_000);
    }
}
