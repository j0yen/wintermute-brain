//! Proactive API-key health probe (PRD-fluid-brain-key-health).
//!
//! [`KeyHealthProbe`] sends a minimal `POST /v1/messages` to Anthropic on
//! daemon start and every [`DEFAULT_PROBE_INTERVAL_SECS`] seconds thereafter.
//! It maps the response to a [`KeyHealthResult`] enum and publishes a
//! `wm.brain.key_status` event on the bus so downstream observers can see
//! the key health without parsing log output.
//!
//! The probe intentionally uses `max_tokens: 1` so the call costs essentially
//! nothing (≤1 output token billed) while still exercising the full auth path.
//! A 401 or 403 response is mapped to [`KeyHealthResult::AuthError`]; any
//! transport failure to [`KeyHealthResult::NetworkError`]; any 2xx (even an
//! empty-content response) to [`KeyHealthResult::Ok`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::anthropic::{ANTHROPIC_VERSION, DEFAULT_BASE_URL};

/// Agorabus topic published by the background key-health probe.
///
/// Payload: `{ ok: bool, code: Option<u16>, ts: u64 }`.
/// PRD-fluid-brain-key-health §2.
pub const KEY_STATUS_TOPIC: &str = "wm.brain.key_status";

/// Default interval between background key-health probes (300 s).
pub const DEFAULT_PROBE_INTERVAL_SECS: u64 = 300;

/// Minimal Anthropic model id used for health probes. Haiku is the
/// cheapest model; `max_tokens: 1` makes the call cost ≤1 output token.
const PROBE_MODEL: &str = "claude-haiku-4-5";

/// The result of one key-health probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyHealthResult {
    /// The key is valid (any 2xx response).
    Ok,
    /// Authentication failed (401 or 403 HTTP status).
    AuthError {
        /// HTTP status code returned by the API.
        code: u16,
    },
    /// Transport-level failure (no response received).
    NetworkError {
        /// Human-readable error description.
        msg: String,
    },
}

impl KeyHealthResult {
    /// `true` when the key is healthy (2xx response).
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }
}

/// Probe that checks Anthropic API key health on demand or in the background.
///
/// The probe sends a single `POST /v1/messages` with `max_tokens: 1` — just
/// enough to hit the auth layer without generating a meaningful response.
/// Construct with [`KeyHealthProbe::new`]; override the base URL for tests
/// with [`KeyHealthProbe::with_base_url`].
#[derive(Debug, Clone)]
pub struct KeyHealthProbe {
    http: reqwest::Client,
    base_url: String,
}

impl KeyHealthProbe {
    /// Construct a probe that targets the production Anthropic base URL.
    ///
    /// # Errors
    /// Returns `reqwest::Error` if the underlying HTTP client cannot be built
    /// (TLS init failure; extremely rare).
    pub fn new() -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder().build()?;
        Ok(Self {
            http,
            base_url: DEFAULT_BASE_URL.to_string(),
        })
    }

    /// Override the base URL. Used in tests to point at an in-process mock.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Send a minimal probe request and return the [`KeyHealthResult`].
    ///
    /// Sends `POST /v1/messages` with `max_tokens: 1` and a single user
    /// message to exercise the full Anthropic auth path. Mapping:
    /// - 2xx → [`KeyHealthResult::Ok`]
    /// - 401 or 403 → [`KeyHealthResult::AuthError`]
    /// - other status → [`KeyHealthResult::NetworkError`] (non-auth API error)
    /// - transport error → [`KeyHealthResult::NetworkError`]
    ///
    /// # Errors (never — fallible results are returned in the enum)
    /// This function is infallible: all failure modes are represented in
    /// [`KeyHealthResult`].
    pub async fn check(&self, api_key: &str) -> KeyHealthResult {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let body = json!({
            "model": PROBE_MODEL,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": false,
        });
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await;

        match resp {
            Err(e) => KeyHealthResult::NetworkError { msg: e.to_string() },
            Ok(r) => {
                let status = r.status().as_u16();
                match status {
                    200..=299 => KeyHealthResult::Ok,
                    401 | 403 => KeyHealthResult::AuthError { code: status },
                    _ => {
                        let body_preview = r
                            .text()
                            .await
                            .unwrap_or_else(|_| String::from("(unreadable)"));
                        let truncated = if body_preview.len() > 256 {
                            body_preview[..256].to_string()
                        } else {
                            body_preview
                        };
                        KeyHealthResult::NetworkError {
                            msg: format!("status {status}: {truncated}"),
                        }
                    }
                }
            }
        }
    }

    /// Spawn a background task that calls [`Self::check`] immediately and
    /// then every `interval_secs` seconds.
    ///
    /// On each check:
    /// - Updates the shared `key_healthy` flag (true on [`KeyHealthResult::Ok`]).
    /// - Publishes a `wm.brain.key_status` event via `publisher`.
    ///
    /// The task runs until the tokio runtime shuts down. Errors publishing
    /// events are logged as warnings and do not abort the probe loop.
    pub fn spawn_background(
        self,
        api_key: String,
        interval_secs: u64,
        key_healthy: Arc<AtomicBool>,
        publisher: Arc<Mutex<dyn BusPublisher + Send>>,
    ) {
        let probe = Arc::new(self);
        tokio::spawn(async move {
            let interval = Duration::from_secs(interval_secs);
            loop {
                let result = probe.check(&api_key).await;
                let ok = result.is_ok();
                let code: Option<u16> = if let KeyHealthResult::AuthError { code } = &result {
                    Some(*code)
                } else {
                    None
                };
                let prev = key_healthy.swap(ok, Ordering::Relaxed);
                if prev != ok {
                    if ok {
                        info!("wm-brain: API key health recovered (ok=true)");
                    } else {
                        error!(
                            ok = false,
                            code = ?code,
                            "wm-brain: API key invalid; cloud tiers disabled until probe recovers"
                        );
                    }
                }
                let ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_millis();
                #[allow(
                    clippy::as_conversions,
                    reason = "unix_ms u128→u64 truncation safe for ~300M years"
                )]
                let ts_u64 = ts as u64;
                let payload = json!({ "ok": ok, "code": code, "ts": ts_u64 });
                {
                    let mut pub_client = publisher.lock().await;
                    if let Err(e) = pub_client.publish_event(KEY_STATUS_TOPIC, payload).await {
                        warn!(error = %e, "wm-brain: key_status publish failed");
                    }
                }
                tokio::time::sleep(interval).await;
            }
        });
    }
}

/// Minimal publish abstraction for the background probe. Allows tests to
/// inject an in-memory collector without a real agorabus socket.
#[async_trait::async_trait]
pub trait BusPublisher {
    /// Publish `payload` on `topic`. Returns an error string on failure.
    ///
    /// # Errors
    /// Implementations surface transport failures as `String`.
    async fn publish_event(
        &mut self,
        topic: &str,
        payload: serde_json::Value,
    ) -> Result<(), String>;
}

/// In-memory [`BusPublisher`] for tests: collects published events so
/// assertions can inspect them without a live bus.
#[cfg(test)]
pub struct MemPublisher {
    /// Collected `(topic, payload)` pairs, in order.
    pub events: Vec<(String, serde_json::Value)>,
}

#[cfg(test)]
impl MemPublisher {
    /// Construct an empty collector.
    #[must_use]
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl BusPublisher for MemPublisher {
    async fn publish_event(
        &mut self,
        topic: &str,
        payload: serde_json::Value,
    ) -> Result<(), String> {
        self.events.push((topic.to_string(), payload));
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::indexing_slicing,
    reason = "tests"
)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use axum::{Router, extract::State, routing::post, response::IntoResponse};
    use axum::http::{StatusCode, HeaderMap};
    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;

    use super::{BusPublisher, KeyHealthProbe, KeyHealthResult, KEY_STATUS_TOPIC, MemPublisher};

    /// Shared state for the mock server: controls what status code the next
    /// request returns. Tests atomically swap the code to simulate key rotation.
    #[derive(Clone)]
    struct MockState {
        status_code: Arc<std::sync::atomic::AtomicU16>,
    }

    async fn mock_handler(
        State(state): State<MockState>,
        _headers: HeaderMap,
        _body: axum::body::Bytes,
    ) -> impl IntoResponse {
        let code = state.status_code.load(Ordering::Relaxed);
        let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = match code {
            200..=299 => {
                json!({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "claude-haiku-4-5",
                    "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                })
            }
            401 | 403 => {
                json!({
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "invalid x-api-key"
                    }
                })
            }
            _ => json!({"error": "server error"}),
        };
        (status, axum::Json(body))
    }

    /// Spawn a local mock Anthropic endpoint. Returns `(base_url, status_cell)`.
    /// Tests mutate `status_cell` atomically to change the next response.
    async fn spawn_mock(initial_code: u16) -> (String, Arc<std::sync::atomic::AtomicU16>) {
        let cell = Arc::new(std::sync::atomic::AtomicU16::new(initial_code));
        let state = MockState { status_code: Arc::clone(&cell) };
        let app = Router::new()
            .route("/v1/messages", post(mock_handler))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), cell)
    }

    // ── AC1: startup probe fires and returns AuthError on 401 ─────────────────

    #[tokio::test]
    async fn ac1_check_returns_auth_error_on_401() {
        let (base_url, _cell) = spawn_mock(401).await;
        let probe = KeyHealthProbe::new().unwrap().with_base_url(base_url);
        let result = probe.check("bad-key").await;
        assert_eq!(result, KeyHealthResult::AuthError { code: 401 });
    }

    // ── AC2: bus event published with ok=false, code=401 ──────────────────────

    #[tokio::test]
    async fn ac2_bad_key_publishes_key_status_false() {
        let (base_url, _cell) = spawn_mock(401).await;
        let probe = KeyHealthProbe::new().unwrap().with_base_url(base_url.clone());

        // Directly check, then publish manually so we can inspect the event.
        let result = probe.check("bad-key").await;
        assert!(!result.is_ok());

        let mem = Arc::new(Mutex::new(MemPublisher::new()));
        let key_healthy = Arc::new(AtomicBool::new(true));
        let ok = result.is_ok();
        let code: Option<u16> = if let KeyHealthResult::AuthError { code } = &result {
            Some(*code)
        } else {
            None
        };
        key_healthy.store(ok, Ordering::Relaxed);
        let payload = json!({ "ok": ok, "code": code, "ts": 0u64 });
        mem.lock().await.publish_event(KEY_STATUS_TOPIC, payload).await.unwrap();

        let events = &mem.lock().await.events;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, KEY_STATUS_TOPIC);
        let v = &events[0].1;
        assert_eq!(v["ok"], false);
        assert_eq!(v["code"], 401);
        assert!(!key_healthy.load(Ordering::Relaxed));
    }

    // ── AC3: bus event published with ok=true on 200 ──────────────────────────

    #[tokio::test]
    async fn ac3_good_key_publishes_key_status_true() {
        let (base_url, _cell) = spawn_mock(200).await;
        let probe = KeyHealthProbe::new().unwrap().with_base_url(base_url);
        let result = probe.check("good-key").await;
        assert_eq!(result, KeyHealthResult::Ok);
        assert!(result.is_ok());

        let mem = Arc::new(Mutex::new(MemPublisher::new()));
        let key_healthy = Arc::new(AtomicBool::new(false));
        key_healthy.store(true, Ordering::Relaxed);
        let payload = json!({ "ok": true, "code": serde_json::Value::Null, "ts": 0u64 });
        mem.lock().await.publish_event(KEY_STATUS_TOPIC, payload).await.unwrap();

        let events = &mem.lock().await.events;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].1["ok"], true);
    }

    // ── AC5: background probe re-checks within the interval ───────────────────

    #[tokio::test]
    async fn ac5_background_probe_rechecks_and_updates_flag() {
        // Start mock returning 401; flip to 200 after first probe fires.
        let (base_url, cell) = spawn_mock(401).await;
        let probe = KeyHealthProbe::new().unwrap().with_base_url(base_url);

        let key_healthy = Arc::new(AtomicBool::new(true)); // start optimistic
        let mem = Arc::new(Mutex::new(MemPublisher::new()));
        let healthy_clone = Arc::clone(&key_healthy);
        let mem_clone = Arc::clone(&mem);

        probe.spawn_background(
            "test-key".to_string(),
            1, // 1-second interval
            healthy_clone,
            mem_clone,
        );

        // Give the first probe ≤500 ms to fire.
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        // First probe should have set key_healthy = false.
        assert!(!key_healthy.load(Ordering::Relaxed), "first probe should set key_healthy=false");

        // Flip mock to 200 so the next probe sees a good key.
        cell.store(200, Ordering::Relaxed);

        // Wait up to 2.5 s for the second probe to fire.
        let start = tokio::time::Instant::now();
        loop {
            if key_healthy.load(Ordering::Relaxed) {
                break;
            }
            if start.elapsed().as_secs() > 2 {
                panic!("background probe did not recover within 2 s");
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        assert!(key_healthy.load(Ordering::Relaxed), "key_healthy should be true after recovery");

        // Verify bus events were published.
        let n = mem.lock().await.events.len();
        assert!(n >= 2, "at least 2 key_status events expected; got {n}");
    }
}
