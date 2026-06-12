//! Agorabus topic + payload schema for `wm-brain` (the `wmd` daemon).
//!
//! Brain subscribes to `wm.dialog.*` events from `wm-dialog` (the
//! per-turn transcript and the operator's verbal confirm/deny on
//! destructive intents) and publishes its own `wm.brain.*` responses
//! per `PRD-wintermute-brain.md` §2.1. Payloads round-trip through
//! `serde_json::Value` because [`agorabus::ServerEvent::data`] is a
//! `Value` — the daemon in [`crate::daemon`] decodes per-topic into
//! the request enums defined here.
//!
//! The conversation loop ([`crate::daemon::dispatch`]) is a logging
//! stub in iter-7b; iter-8+ folds Anthropic streaming and recall
//! retrieval into the dispatch path so [`Emit::Reply`] /
//! [`Emit::ReplyDestructive`] payloads start showing up.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Subscribe prefix capturing all inbound `wm.dialog.*` envelopes from
/// `wm-dialog`. PRD §2.1.
pub const DIALOG_TOPIC_PREFIX: &str = "wm.dialog.";

/// Incoming topics decoded by [`decode_request`].
pub mod incoming {
    /// User turn finalised by `wm-dialog` (post-STT-+-tone routing).
    pub const TURN_USER: &str = "wm.dialog.turn.user";
    /// Verbal confirmation granted for a previously-published
    /// destructive intent.
    pub const CONFIRM_GRANTED: &str = "wm.dialog.confirm.granted";
    /// Verbal confirmation denied for a previously-published
    /// destructive intent.
    pub const CONFIRM_DENIED: &str = "wm.dialog.confirm.denied";
}

/// Outgoing topics published by the brain daemon.
pub mod outgoing {
    /// Normal reply — `wm-dialog` will pipe to `wm-tts`.
    pub const REPLY: &str = "wm.brain.reply";
    /// Destructive reply — `wm-dialog` runs verbal confirmation
    /// before any tool call fires.
    pub const REPLY_DESTRUCTIVE: &str = "wm.brain.reply.destructive";
    /// Tool invocation about to be dispatched by the router.
    pub const TOOL_CALL: &str = "wm.brain.tool.call";
    /// Tool invocation result.
    pub const TOOL_RESULT: &str = "wm.brain.tool.result";
    /// Failure marker; payload carries `kind` + `message`.
    pub const ERROR: &str = "wm.brain.error";
    /// Session opened — payload: `{ session_id, ts }`.
    /// PRD-wmd-session-boundary §2.3.
    pub const SESSION_START: &str = "wm.brain.session.start";
    /// Session closed — payload: `{ session_id, ts, turn_count, reason }`.
    /// PRD-wmd-session-boundary §2.3.
    pub const SESSION_END: &str = "wm.brain.session.end";
    /// Per-turn routing decision — `{ turn_id, tier, reason, latency_ms, model, ts }`.
    /// PRD-wintermute-brain-routing §2.5.
    pub const ROUTE: &str = "wm.brain.route";
    /// Per-turn injected-context digest —
    /// `{ turn_id, recall_hits, persona_tier, history_turns, ts }`.
    /// PRD-build-lucid-mind-brain-context §2.
    pub const CONTEXT: &str = "wm.brain.context";
    /// Semantic-cache lookup result —
    /// `{ hit, similarity, tier, latency_ms }`.
    /// PRD-thrift-turn-cache §2.
    pub const CACHE: &str = "wm.brain.cache";
}

/// Self-emitted `wm.brain.route` topic.  The daemon's subscribe-loop
/// allow-list includes this so the bus does not reject self-published route
/// events (PRD-wintermute-brain-routing §2.5).
pub const ROUTE_TOPIC: &str = "wm.brain.route";

/// The `wm.brain.session.*` topic prefix. The brain subscribes to
/// `wm.dialog.*` inbound; it also publishes to this prefix. The self-emit
/// filter in the daemon's subscribe loop must ignore events on this prefix to
/// prevent feedback loops (PRD-wmd-session-boundary AC7).
pub const SESSION_TOPIC_PREFIX: &str = "wm.brain.session.";

/// Decoded request payloads. Returned by [`decode_request`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Request {
    /// `wm.dialog.turn.user` payload.
    TurnUser(TurnUserEvent),
    /// `wm.dialog.confirm.granted` payload.
    ConfirmGranted(ConfirmGrantedEvent),
    /// `wm.dialog.confirm.denied` payload.
    ConfirmDenied(ConfirmDeniedEvent),
}

/// `wm.dialog.turn.user` payload as seen by `wm-brain`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TurnUserEvent {
    /// Finalised user transcript.
    pub transcript: String,
    /// Confidence the upstream STT reported in `(0.0, 1.0]`.
    pub confidence: f32,
    /// Unix milliseconds when the turn was finalised by `wm-dialog`.
    pub ts: u64,
    /// Cross-daemon turn correlation id, minted by `wm-audio` at wake and
    /// threaded through stt → dialog → brain → tts (PRD lucid-turn-id, AC4).
    ///
    /// Format is `<unix_ms_hex>-<seq_hex>` (see [`resolve_turn_id`]). Optional
    /// for backward compatibility (AC5): pre-PRD `dialog.turn.user` envelopes
    /// carry no `turn_id` and must still deserialize; the brain then falls back
    /// to a freshly-minted, `gen-`-flagged id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

/// `wm.dialog.confirm.granted` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfirmGrantedEvent {
    /// Intent id previously published with the destructive reply.
    pub intent_id: String,
    /// Unix milliseconds at confirmation.
    pub ts: u64,
}

/// `wm.dialog.confirm.denied` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfirmDeniedEvent {
    /// Intent id previously published with the destructive reply.
    pub intent_id: String,
    /// Short reason `wm-dialog` reports (`timeout`, `negative_keyword`, …).
    pub reason: String,
    /// Unix milliseconds at denial.
    pub ts: u64,
}

/// Outbound emit produced by the daemon dispatch. iter-7b's dispatch
/// is a logging stub that produces no emits; the variants are wired so
/// iter-8+ can fill them in without rejigging the topic table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Emit {
    /// `wm.brain.reply` event.
    Reply(ReplyEvent),
    /// `wm.brain.reply.destructive` event.
    ReplyDestructive(ReplyDestructiveEvent),
    /// `wm.brain.tool.call` event.
    ToolCall(ToolCallEvent),
    /// `wm.brain.tool.result` event.
    ToolResult(ToolResultEvent),
    /// `wm.brain.error` event.
    Error(ErrorEvent),
}

/// Outbound `wm.brain.reply` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplyEvent {
    /// Reply text the dialog will hand to TTS.
    pub text: String,
    /// Unix milliseconds at emission.
    pub ts: u64,
    /// Optional loudness hint for `wm-tts` (PRD-wmd-repair-affordances §2.2).
    ///
    /// When present, `wm-tts` may honor it with a volume bump. Absent for all
    /// ordinary replies; only set to `"loud"` by the `RepeatLouder` repair
    /// path. Existing consumers that don't inspect this field are unaffected
    /// (backward-compatible — `#[serde(skip_serializing_if = "Option::is_none")]`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loudness: Option<String>,
    /// Cross-daemon turn correlation id, adopted from the inbound
    /// `wm.dialog.turn.user` (PRD lucid-turn-id, AC4). `wm-tts` copies this
    /// onto `wm.tts.*` in a later tick. Optional for backward compat (AC5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

/// Outbound `wm.brain.reply.destructive` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplyDestructiveEvent {
    /// Spoken text confirming what's about to happen, asking for the
    /// keyword.
    pub text: String,
    /// Stable id linking this reply to the eventual confirm/deny.
    pub intent_id: String,
    /// One-line summary of the action.
    pub summary: String,
    /// Short keyword the user must speak to confirm (PRD §2.4).
    pub confirm_keyword: String,
    /// Opaque action descriptor — typically the tool name + args the
    /// brain intends to invoke on confirm.
    pub action: Value,
    /// Unix milliseconds at emission.
    pub ts: u64,
    /// Cross-daemon turn correlation id (PRD lucid-turn-id, AC4). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

/// Outbound `wm.brain.tool.call` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallEvent {
    /// Tool name from the Fleet 1 router table (PRD §2.3).
    pub tool: String,
    /// Tool-specific argument bag.
    pub args: Value,
    /// Unix milliseconds at emission.
    pub ts: u64,
    /// Cross-daemon turn correlation id (PRD lucid-turn-id, AC4). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

/// Outbound `wm.brain.tool.result` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResultEvent {
    /// Tool name (echoed from the preceding tool.call).
    pub tool: String,
    /// `true` if the tool returned a value; `false` if it errored.
    pub ok: bool,
    /// Tool-specific result body (success payload OR error detail).
    pub body: Value,
    /// Unix milliseconds at emission.
    pub ts: u64,
    /// Cross-daemon turn correlation id (PRD lucid-turn-id, AC4). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

/// Outbound `wm.brain.error` payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorEvent {
    /// Short kind tag (`"bus" | "anthropic" | "recall" | "tool" | "internal"`).
    pub kind: String,
    /// Human-readable detail.
    pub message: String,
    /// Unix milliseconds at emission.
    pub ts: u64,
}

// ── Context digest (PRD-build-lucid-mind-brain-context) ──────────────────────

/// One entry in [`BrainContextEvent::recall_hits`]: the memory id and its
/// one-line subject. The memory body is intentionally omitted to keep the
/// payload privacy-light (AC3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecallHitDigest {
    /// Stable memory id (matches recall's `QueryHit::id`).
    pub id: String,
    /// Subject the memory was filed under (e.g. `wintermute-profile`).
    pub subject: String,
}

/// `wm.brain.context` digest event — published once per turn right after
/// the brain assembles its prompt and alongside `wm.brain.route`.
///
/// The `turn_id` is the same millisecond timestamp used as the key in the
/// `wm.brain.route` event, enabling lucid-mind to join the two events with
/// no extra wiring (AC2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrainContextEvent {
    /// Correlation id shared with `wm.brain.route` (equals `ts` for now;
    /// will be a proper UUID once the brain mints turn ids).
    pub turn_id: u64,
    /// Recall memories that were injected into the prompt this turn.
    /// Contains only `id + subject` — no body text (AC3).
    pub recall_hits: Vec<RecallHitDigest>,
    /// Persona tier tag used for this turn (e.g. `"default"`, `"child_lock"`).
    pub persona_tier: String,
    /// Number of prior dialog turns included in the prompt as history context.
    pub history_turns: usize,
    /// Unix milliseconds at publish time.
    pub ts: u64,
}

/// Errors raised while decoding an inbound payload.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Topic was not one of the known incoming names.
    #[error("unknown topic: {0}")]
    UnknownTopic(String),
    /// JSON decode of the payload failed.
    #[error("payload decode failed: {0}")]
    Json(#[from] serde_json::Error),
}

/// Decode a raw `(topic, data)` pair into a strongly-typed [`Request`].
///
/// # Errors
/// Returns [`DecodeError::UnknownTopic`] for topics outside the known
/// `wm.dialog.{turn.user,confirm.granted,confirm.denied}` set, or
/// [`DecodeError::Json`] when the payload shape doesn't match the
/// declared payload struct.
pub fn decode_request(topic: &str, data: &Value) -> Result<Request, DecodeError> {
    match topic {
        incoming::TURN_USER => Ok(Request::TurnUser(serde_json::from_value(data.clone())?)),
        incoming::CONFIRM_GRANTED => Ok(Request::ConfirmGranted(serde_json::from_value(
            data.clone(),
        )?)),
        incoming::CONFIRM_DENIED => Ok(Request::ConfirmDenied(serde_json::from_value(
            data.clone(),
        )?)),
        other => Err(DecodeError::UnknownTopic(other.to_string())),
    }
}

/// Process-global monotone counter for same-millisecond collision
/// avoidance in synthesized (`gen-`) turn ids.
static GEN_TURN_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Resolve the turn correlation id to attach to outbound `wm.brain.*`
/// envelopes for a given inbound `wm.dialog.turn.user` (PRD lucid-turn-id,
/// AC4 + AC3-brain).
///
/// * If the inbound turn carried a `turn_id` (minted upstream by `wm-audio`
///   at wake and threaded through stt → dialog), it is **adopted verbatim**
///   so the whole spoken turn shares one id end-to-end.
/// * Otherwise (system-injected turns, or pre-PRD envelopes — AC5), a fresh
///   id is minted and **flagged with a `gen-` prefix** so consumers can tell
///   it was synthesized by the brain rather than originating at wake. The
///   minted body matches `wm-audio`'s `<unix_ms_hex>-<seq_hex>` shape.
#[must_use]
pub fn resolve_turn_id(inbound: Option<&str>) -> String {
    match inbound {
        Some(id) if !id.is_empty() => id.to_owned(),
        _ => {
            use std::sync::atomic::Ordering;
            let ms = now_unix_ms();
            let seq = GEN_TURN_SEQ.fetch_add(1, Ordering::Relaxed);
            format!("gen-{ms:013x}-{seq:04x}")
        }
    }
}

/// Wall-clock milliseconds since the Unix epoch. Saturates to
/// `u64::MAX` if the clock is set before 1970 (shouldn't happen).
#[must_use]
pub fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(u64::MAX, |d| {
            u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    reason = "tests"
)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decode_turn_user_minimal() {
        let v = json!({
            "transcript": "what time is it",
            "confidence": 0.92_f32,
            "ts": 1_234_567_890_123_u64,
        });
        let req = decode_request(incoming::TURN_USER, &v).expect("decodes");
        match req {
            Request::TurnUser(t) => {
                assert_eq!(t.transcript, "what time is it");
                assert_eq!(t.confidence, 0.92);
                assert_eq!(t.ts, 1_234_567_890_123);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn decode_confirm_granted() {
        let v = json!({ "intent_id": "abc", "ts": 7_u64 });
        let req = decode_request(incoming::CONFIRM_GRANTED, &v).expect("decodes");
        match req {
            Request::ConfirmGranted(c) => {
                assert_eq!(c.intent_id, "abc");
                assert_eq!(c.ts, 7);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn decode_confirm_denied() {
        let v = json!({ "intent_id": "abc", "reason": "timeout", "ts": 9_u64 });
        let req = decode_request(incoming::CONFIRM_DENIED, &v).expect("decodes");
        match req {
            Request::ConfirmDenied(c) => {
                assert_eq!(c.intent_id, "abc");
                assert_eq!(c.reason, "timeout");
                assert_eq!(c.ts, 9);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn decode_unknown_topic() {
        let v = json!({});
        let err = decode_request("wm.brain.reply", &v).expect_err("unknown topic rejected");
        assert!(matches!(err, DecodeError::UnknownTopic(t) if t == "wm.brain.reply"));
    }

    #[test]
    fn decode_turn_user_missing_field_errors() {
        let v = json!({ "transcript": "hi" });
        let err = decode_request(incoming::TURN_USER, &v).expect_err("missing fields");
        assert!(matches!(err, DecodeError::Json(_)));
    }

    #[test]
    fn decode_confirm_granted_missing_intent_errors() {
        let v = json!({ "ts": 1 });
        let err = decode_request(incoming::CONFIRM_GRANTED, &v).expect_err("missing intent");
        assert!(matches!(err, DecodeError::Json(_)));
    }

    #[test]
    fn outbound_reply_roundtrip() {
        let r = ReplyEvent {
            text: "hello".to_string(),
            ts: 17,
            loudness: None,
            turn_id: None,
        };
        let v = serde_json::to_value(&r).expect("serialises");
        let back: ReplyEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, r);
    }

    #[test]
    fn outbound_reply_loudness_roundtrip() {
        // AC6: loudness field is backward-compatible; when absent it deserialises
        // to None, when present it round-trips unchanged.
        let r_loud = ReplyEvent {
            text: "Hello!".to_string(),
            ts: 42,
            loudness: Some("loud".to_string()),
            turn_id: None,
        };
        let v = serde_json::to_value(&r_loud).expect("serialises");
        // The serialised form includes the loudness key.
        assert_eq!(v.get("loudness").and_then(|x| x.as_str()), Some("loud"));
        let back: ReplyEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, r_loud);

        // A payload without "loudness" deserialises to loudness=None.
        let plain = serde_json::json!({ "text": "hi", "ts": 1_u64 });
        let back_plain: ReplyEvent = serde_json::from_value(plain).expect("round-trips without loudness");
        assert_eq!(back_plain.loudness, None);
    }

    #[test]
    fn outbound_reply_destructive_roundtrip() {
        let d = ReplyDestructiveEvent {
            text: "delete that file?".to_string(),
            intent_id: "01abc".to_string(),
            summary: "delete /tmp/foo".to_string(),
            confirm_keyword: "delete".to_string(),
            action: json!({ "tool": "wm.fs.rm", "args": { "path": "/tmp/foo" } }),
            ts: 100,
            turn_id: None,
        };
        let v = serde_json::to_value(&d).expect("serialises");
        let back: ReplyDestructiveEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, d);
    }

    #[test]
    fn outbound_tool_call_roundtrip() {
        let t = ToolCallEvent {
            tool: "wm.time.now".to_string(),
            args: json!({}),
            ts: 11,
            turn_id: None,
        };
        let v = serde_json::to_value(&t).expect("serialises");
        let back: ToolCallEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, t);
    }

    #[test]
    fn outbound_tool_result_roundtrip() {
        let r = ToolResultEvent {
            tool: "wm.time.now".to_string(),
            ok: true,
            body: json!({ "iso": "2026-05-26T23:00:00Z" }),
            ts: 12,
            turn_id: None,
        };
        let v = serde_json::to_value(&r).expect("serialises");
        let back: ToolResultEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, r);
    }

    #[test]
    fn outbound_error_roundtrip() {
        let e = ErrorEvent {
            kind: "anthropic".to_string(),
            message: "stream interrupted".to_string(),
            ts: 3,
        };
        let v = serde_json::to_value(&e).expect("serialises");
        let back: ErrorEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, e);
    }

    #[test]
    fn now_unix_ms_monotonic() {
        let a = now_unix_ms();
        let b = now_unix_ms();
        assert!(b >= a, "clock should not run backwards within a call");
    }

    #[test]
    fn topic_constants_match_strings() {
        assert_eq!(incoming::TURN_USER, "wm.dialog.turn.user");
        assert_eq!(incoming::CONFIRM_GRANTED, "wm.dialog.confirm.granted");
        assert_eq!(incoming::CONFIRM_DENIED, "wm.dialog.confirm.denied");
        assert_eq!(outgoing::REPLY, "wm.brain.reply");
        assert_eq!(outgoing::REPLY_DESTRUCTIVE, "wm.brain.reply.destructive");
        assert_eq!(outgoing::TOOL_CALL, "wm.brain.tool.call");
        assert_eq!(outgoing::TOOL_RESULT, "wm.brain.tool.result");
        assert_eq!(outgoing::ERROR, "wm.brain.error");
        assert_eq!(outgoing::CONTEXT, "wm.brain.context");
        assert_eq!(DIALOG_TOPIC_PREFIX, "wm.dialog.");
    }

    // ── BrainContextEvent tests (AC3 + AC4) ──────────────────────────────────

    #[test]
    fn brain_context_event_roundtrip() {
        // AC4: serialize → deserialize is identity.
        let evt = BrainContextEvent {
            turn_id: 1_717_000_000_000_u64,
            recall_hits: vec![
                RecallHitDigest { id: "m-1".to_string(), subject: "wintermute-profile".to_string() },
                RecallHitDigest { id: "m-2".to_string(), subject: "tea-preference".to_string() },
            ],
            persona_tier: "default".to_string(),
            history_turns: 3,
            ts: 1_717_000_000_001_u64,
        };
        let v = serde_json::to_value(&evt).expect("serialises");
        let back: BrainContextEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, evt);
    }

    #[test]
    fn brain_context_event_no_body_field() {
        // AC3: the serialised payload must NOT contain any "body", "snippet",
        // or "text" key inside recall_hits entries.
        let evt = BrainContextEvent {
            turn_id: 42_u64,
            recall_hits: vec![
                RecallHitDigest { id: "m-7".to_string(), subject: "some-subject".to_string() },
            ],
            persona_tier: "default".to_string(),
            history_turns: 0,
            ts: 42_u64,
        };
        let v = serde_json::to_value(&evt).expect("serialises");
        let hits = v.get("recall_hits").and_then(|h| h.as_array()).expect("has recall_hits");
        for hit in hits {
            let obj = hit.as_object().expect("hit is object");
            assert!(!obj.contains_key("body"), "body must not leak into digest");
            assert!(!obj.contains_key("snippet"), "snippet must not leak into digest");
            assert!(!obj.contains_key("text"), "text must not leak into digest");
        }
    }

    #[test]
    fn brain_context_event_empty_recall_hits_roundtrip() {
        let evt = BrainContextEvent {
            turn_id: 1_u64,
            recall_hits: vec![],
            persona_tier: "child_lock".to_string(),
            history_turns: 0,
            ts: 1_u64,
        };
        let v = serde_json::to_value(&evt).expect("serialises");
        let back: BrainContextEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, evt);
        assert!(back.recall_hits.is_empty());
    }

    #[test]
    fn recall_hit_digest_contains_only_id_and_subject() {
        let digest = RecallHitDigest {
            id: "m-99".to_string(),
            subject: "persona-override".to_string(),
        };
        let v = serde_json::to_value(&digest).expect("serialises");
        let obj = v.as_object().expect("object");
        assert_eq!(obj.len(), 2, "digest must have exactly 2 fields: id + subject");
        assert!(obj.contains_key("id"));
        assert!(obj.contains_key("subject"));
    }

    #[test]
    fn dialog_prefix_covers_all_incoming() {
        for topic in [
            incoming::TURN_USER,
            incoming::CONFIRM_GRANTED,
            incoming::CONFIRM_DENIED,
        ] {
            assert!(
                topic.starts_with(DIALOG_TOPIC_PREFIX),
                "{topic} not covered by subscribe prefix"
            );
        }
    }

    // ── resolve_turn_id (PRD lucid-turn-id, AC4 + AC3-brain) ────────────────────

    #[test]
    fn resolve_turn_id_adopts_inbound_verbatim() {
        let inbound = "0000018f1c2d3e40-0007";
        assert_eq!(resolve_turn_id(Some(inbound)), inbound);
    }

    #[test]
    fn resolve_turn_id_mints_flagged_when_absent() {
        let a = resolve_turn_id(None);
        assert!(a.starts_with("gen-"), "synthesized id must be gen-flagged: {a}");
        // Empty inbound is treated as absent → also flagged.
        let b = resolve_turn_id(Some(""));
        assert!(b.starts_with("gen-"), "empty inbound must fall back: {b}");
        // Two mints differ (same-ms collision avoidance via the seq counter).
        assert_ne!(resolve_turn_id(None), resolve_turn_id(None));
    }

    #[test]
    fn turn_user_event_deserializes_without_turn_id() {
        // AC5: a pre-PRD dialog.turn.user (no turn_id) must still decode.
        let v = serde_json::json!({ "transcript": "hi", "confidence": 0.9, "ts": 1 });
        let e: TurnUserEvent = serde_json::from_value(v).expect("legacy turn decodes");
        assert!(e.turn_id.is_none(), "absent turn_id maps to None");
    }

    #[test]
    fn reply_event_turn_id_absent_from_json_when_none() {
        // Backward compat: None turn_id must not appear in serialized payload.
        let r = ReplyEvent { text: "x".into(), ts: 1, loudness: None, turn_id: None };
        let v = serde_json::to_value(&r).expect("serialize");
        assert!(v.get("turn_id").is_none(), "None turn_id must be omitted");
    }
}
