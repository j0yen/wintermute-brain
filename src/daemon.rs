//! Live agorabus subscribe loop for `wm-brain` / `wmd`.
//!
//! Wires the bus schema from [`crate::bus`] to a real subscribe loop.
//! The daemon subscribes to [`bus::DIALOG_TOPIC_PREFIX`]
//! (`wm.dialog.`) — inbound turn + verbal-confirm envelopes from
//! `wm-dialog` — and opens a separate publish connection (read/write
//! on a subscribed socket would interleave `Reply` lines with the
//! broadcast stream — same pattern as `wintermute-stt/src/daemon.rs`).
//!
//! iter-7b's [`dispatch`] is a logging stub: requests are decoded and
//! logged but the daemon publishes nothing on the bus. The actual
//! conversation loop (Anthropic streaming via [`crate::anthropic`],
//! recall retrieval via [`crate::recall_client`], turn memorisation,
//! destructive-intent gating) lands in iter-8+. Wiring the subscribe
//! loop now lets the operator confirm at runtime that brain is
//! reachable on the bus and is decoding turns correctly.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::BrainConfig;
use crate::bus::{
    self, DecodeError, Emit, ErrorEvent, Request, decode_request, now_unix_ms, outgoing,
};

/// Publish abstraction so per-request handlers can be tested without
/// an actual agorabus daemon. Production impl is [`AgoraSink`]; tests
/// use an in-memory sink.
#[async_trait::async_trait]
pub trait EventSink: Send {
    /// Publish `data` on `topic`. The dispatch layer treats failures
    /// as fatal for the current request but logs and continues the
    /// outer subscribe loop.
    ///
    /// # Errors
    /// Propagates whatever the underlying transport returns.
    async fn publish(&mut self, topic: &str, data: Value) -> Result<()>;
}

/// Production sink: publishes through an agorabus [`agorabus::Client`].
pub struct AgoraSink {
    pub(crate) inner: agorabus::Client,
}

#[async_trait::async_trait]
impl EventSink for AgoraSink {
    async fn publish(&mut self, topic: &str, data: Value) -> Result<()> {
        let reply = self.inner.publish(topic, data).await?;
        if !reply.ok {
            warn!(
                topic = %topic,
                err = %reply.error.as_deref().unwrap_or("?"),
                "wm-brain: bus rejected publish"
            );
        }
        Ok(())
    }
}

/// Live daemon state.
///
/// iter-7b only owns a [`BrainConfig`] snapshot. Once iter-8 lands the
/// Anthropic streaming + recall integration, this struct grows to hold
/// the per-turn conversation buffer + the [`crate::recall_client::RecallClient`]
/// handle.
pub struct DaemonState {
    /// Resolved runtime config (model defaults, recall socket, …).
    pub config: Mutex<BrainConfig>,
}

impl DaemonState {
    /// Construct a daemon state from an already-validated config.
    #[must_use]
    pub const fn new(config: BrainConfig) -> Self {
        Self {
            config: Mutex::const_new(config),
        }
    }
}

/// Resolve the outbound topic for a single emit. Pure function; exposed
/// so tests can pin the topic vocabulary without spinning up a bus.
#[must_use]
pub const fn topic_for_emit(emit: &Emit) -> &'static str {
    match emit {
        Emit::Reply(_) => outgoing::REPLY,
        Emit::ReplyDestructive(_) => outgoing::REPLY_DESTRUCTIVE,
        Emit::ToolCall(_) => outgoing::TOOL_CALL,
        Emit::ToolResult(_) => outgoing::TOOL_RESULT,
        Emit::Error(_) => outgoing::ERROR,
    }
}

/// Serialise an emit into the JSON value the agorabus expects.
///
/// # Errors
/// Propagates `serde_json::Error` — every [`Emit`] variant uses
/// `Serialize` impls that don't fail in practice, so a returned error
/// is a programmer bug rather than runtime-recoverable.
pub fn emit_to_value(emit: &Emit) -> Result<Value> {
    Ok(match emit {
        Emit::Reply(r) => serde_json::to_value(r)?,
        Emit::ReplyDestructive(r) => serde_json::to_value(r)?,
        Emit::ToolCall(c) => serde_json::to_value(c)?,
        Emit::ToolResult(r) => serde_json::to_value(r)?,
        Emit::Error(e) => serde_json::to_value(e)?,
    })
}

/// Dispatch one decoded request. iter-7b logs the request and returns
/// without publishing — the conversation loop (Anthropic streaming +
/// recall + tool router) lands in iter-8+.
///
/// # Errors
/// Returns the first publish failure encountered. iter-7b never
/// publishes so the function is currently infallible, but the
/// signature is shared with iter-8+ where Anthropic / recall calls
/// can fail mid-handle.
pub async fn dispatch(
    state: &DaemonState,
    _publish: &mut dyn EventSink,
    req: Request,
    _now_ms: u64,
) -> Result<()> {
    let model = {
        let cfg = state.config.lock().await;
        cfg.effective_model().to_string()
    };
    match req {
        Request::TurnUser(t) => {
            info!(
                transcript = %t.transcript,
                confidence = t.confidence,
                ts = t.ts,
                model = %model,
                "wm-brain: turn.user received (conversation loop deferred to iter-8)"
            );
        }
        Request::ConfirmGranted(c) => {
            info!(
                intent_id = %c.intent_id,
                ts = c.ts,
                "wm-brain: confirm.granted received (destructive dispatch deferred to iter-8)"
            );
        }
        Request::ConfirmDenied(c) => {
            info!(
                intent_id = %c.intent_id,
                reason = %c.reason,
                ts = c.ts,
                "wm-brain: confirm.denied received"
            );
        }
    }
    Ok(())
}

async fn publish_error(publish: &mut dyn EventSink, kind: &str, message: &str) -> Result<()> {
    let ev = ErrorEvent {
        kind: kind.to_string(),
        message: message.to_string(),
        ts: now_unix_ms(),
    };
    publish
        .publish(outgoing::ERROR, serde_json::to_value(&ev)?)
        .await
}

/// Run the live daemon: validate config, connect to agorabus, subscribe
/// to the dialog prefix, dispatch each event until the bus closes.
///
/// # Errors
/// Propagates I/O failures from config validation or the agorabus
/// client. A missing agorabus socket is *not* an error: the daemon
/// logs and exits cleanly so the systemd unit restarts it when the bus
/// comes back (same pattern as `wm-stt` / `wm-tts`).
pub async fn run(cfg: BrainConfig) -> Result<()> {
    cfg.validate().context("wm-brain: config validation failed")?;
    let state = Arc::new(DaemonState::new(cfg));

    let sock = agorabus::default_socket_path();
    let Some(mut sub_client) = agorabus::Client::try_connect(&sock).await? else {
        warn!(socket = %sock.display(), "wm-brain: agorabus not reachable; exiting");
        return Ok(());
    };
    sub_client.subscribe(bus::DIALOG_TOPIC_PREFIX).await?;
    info!(
        dialog_prefix = bus::DIALOG_TOPIC_PREFIX,
        "wm-brain: subscribed"
    );

    let pub_client = agorabus::Client::connect(&sock).await?;
    let mut sink = AgoraSink { inner: pub_client };

    while let Some(ev) = sub_client.next_event().await? {
        match decode_request(&ev.topic, &ev.data) {
            Ok(req) => {
                let now = now_unix_ms();
                if let Err(err) = dispatch(state.as_ref(), &mut sink, req, now).await {
                    error!(topic = %ev.topic, err = %err, "wm-brain: dispatch failed");
                    let _ = publish_error(&mut sink, "bus", &format!("dispatch: {err}")).await;
                }
            }
            Err(DecodeError::UnknownTopic(t)) => {
                debug!(topic = %t, "wm-brain: ignoring unknown topic under dialog prefix");
            }
            Err(err) => {
                warn!(topic = %ev.topic, err = %err, "wm-brain: decode failed");
                let _ = publish_error(&mut sink, "bus", &format!("decode: {err}")).await;
            }
        }
    }
    info!("wm-brain: bus closed; daemon exiting");
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::significant_drop_tightening,
    reason = "tests"
)]
mod tests {
    use super::*;
    use crate::bus::{
        ConfirmDeniedEvent, ConfirmGrantedEvent, ErrorEvent as Err, ReplyDestructiveEvent,
        ReplyEvent, ToolCallEvent, ToolResultEvent, TurnUserEvent,
    };
    use serde_json::json;
    use std::sync::Mutex as StdMutex;

    #[derive(Default, Clone)]
    struct MemSink {
        events: Arc<StdMutex<Vec<(String, Value)>>>,
    }

    #[async_trait::async_trait]
    impl EventSink for MemSink {
        async fn publish(&mut self, topic: &str, data: Value) -> Result<()> {
            self.events
                .lock()
                .expect("mem sink poisoned")
                .push((topic.to_string(), data));
            Ok(())
        }
    }

    fn fresh_state() -> Arc<DaemonState> {
        let cfg = BrainConfig::default();
        Arc::new(DaemonState::new(cfg))
    }

    #[tokio::test]
    async fn dispatch_turn_user_is_silent_in_iter7b() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let req = Request::TurnUser(TurnUserEvent {
            transcript: "hello".to_string(),
            confidence: 0.9,
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 1)
            .await
            .expect("dispatch ok");
        let events = sink.events.lock().unwrap();
        assert!(
            events.is_empty(),
            "iter-7b should not publish; conversation loop lands in iter-8"
        );
    }

    #[tokio::test]
    async fn dispatch_confirm_granted_is_silent_in_iter7b() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let req = Request::ConfirmGranted(ConfirmGrantedEvent {
            intent_id: "abc".to_string(),
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 1)
            .await
            .expect("dispatch ok");
        assert!(sink.events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn dispatch_confirm_denied_is_silent_in_iter7b() {
        let state = fresh_state();
        let mut sink = MemSink::default();
        let req = Request::ConfirmDenied(ConfirmDeniedEvent {
            intent_id: "abc".to_string(),
            reason: "timeout".to_string(),
            ts: 1,
        });
        dispatch(state.as_ref(), &mut sink, req, 1)
            .await
            .expect("dispatch ok");
        assert!(sink.events.lock().unwrap().is_empty());
    }

    #[test]
    fn topic_for_emit_matches_outgoing_constants() {
        assert_eq!(
            topic_for_emit(&Emit::Reply(ReplyEvent {
                text: String::new(),
                ts: 0,
            })),
            outgoing::REPLY
        );
        assert_eq!(
            topic_for_emit(&Emit::ReplyDestructive(ReplyDestructiveEvent {
                text: String::new(),
                intent_id: String::new(),
                summary: String::new(),
                confirm_keyword: String::new(),
                action: json!({}),
                ts: 0,
            })),
            outgoing::REPLY_DESTRUCTIVE
        );
        assert_eq!(
            topic_for_emit(&Emit::ToolCall(ToolCallEvent {
                tool: String::new(),
                args: json!({}),
                ts: 0,
            })),
            outgoing::TOOL_CALL
        );
        assert_eq!(
            topic_for_emit(&Emit::ToolResult(ToolResultEvent {
                tool: String::new(),
                ok: true,
                body: json!({}),
                ts: 0,
            })),
            outgoing::TOOL_RESULT
        );
        assert_eq!(
            topic_for_emit(&Emit::Error(Err {
                kind: String::new(),
                message: String::new(),
                ts: 0,
            })),
            outgoing::ERROR
        );
    }

    #[test]
    fn emit_to_value_reply_has_expected_fields() {
        let e = Emit::Reply(ReplyEvent {
            text: "ok".to_string(),
            ts: 42,
        });
        let v = emit_to_value(&e).expect("serialises");
        assert_eq!(v["text"], "ok");
        assert_eq!(v["ts"], 42);
    }

    #[test]
    fn emit_to_value_reply_destructive_includes_action() {
        let e = Emit::ReplyDestructive(ReplyDestructiveEvent {
            text: "delete?".to_string(),
            intent_id: "01abc".to_string(),
            summary: "rm /tmp/x".to_string(),
            confirm_keyword: "delete".to_string(),
            action: json!({ "tool": "wm.fs.rm" }),
            ts: 7,
        });
        let v = emit_to_value(&e).expect("serialises");
        assert_eq!(v["intent_id"], "01abc");
        assert_eq!(v["confirm_keyword"], "delete");
        assert_eq!(v["action"]["tool"], "wm.fs.rm");
        assert_eq!(v["ts"], 7);
    }

    #[tokio::test]
    async fn publish_error_writes_to_sink() {
        let mut sink = MemSink::default();
        publish_error(&mut sink, "bus", "decode failed")
            .await
            .expect("publish ok");
        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, outgoing::ERROR);
        assert_eq!(events[0].1["kind"], "bus");
        assert_eq!(events[0].1["message"], "decode failed");
    }
}
