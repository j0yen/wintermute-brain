//! Live HTTP integration tests for [`wintermute_brain::anthropic::AnthropicClient`].
//!
//! Spins up a throwaway axum server on an OS-assigned port to stand in
//! for `api.anthropic.com`, points the client at it, and verifies that
//! `POST /v1/messages` headers + body match what the brain promises,
//! plus that SSE responses (success and error) round-trip cleanly.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::indexing_slicing,
    reason = "integration tests"
)]

use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use futures_util::StreamExt;
use wintermute_brain::anthropic::{
    AnthropicClient, ClientError, Message, MessageRequest, Role, StreamEvent,
};

#[derive(Debug, Clone)]
struct Recorded {
    headers: HeaderMap,
    body: serde_json::Value,
}

/// Axum's `Response` isn't `Clone`; we hand the handler the raw
/// `(status, body, content_type)` triple and build a fresh response per
/// request from those primitives.
#[derive(Clone)]
struct MockBehavior {
    recorded: Arc<Mutex<Vec<Recorded>>>,
    status: StatusCode,
    body: String,
    content_type: &'static str,
}

async fn handle_messages(
    State(state): State<MockBehavior>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    state
        .recorded
        .lock()
        .await
        .push(Recorded { headers, body });
    (
        state.status,
        [("content-type", state.content_type)],
        state.body.clone(),
    )
        .into_response()
}

async fn spawn_mock(
    status: StatusCode,
    body: impl Into<String>,
    content_type: &'static str,
) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let recorded: Arc<Mutex<Vec<Recorded>>> = Arc::new(Mutex::new(Vec::new()));
    let state = MockBehavior {
        recorded: Arc::clone(&recorded),
        status,
        body: body.into(),
        content_type,
    };
    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    // Give axum a beat to start accepting.
    tokio::time::sleep(Duration::from_millis(20)).await;
    (format!("http://{addr}"), recorded)
}

fn sample_request() -> MessageRequest {
    MessageRequest::streaming(
        "claude-sonnet-4-6",
        128,
        vec![Message {
            role: Role::User,
            content: "ping".to_string(),
        }],
    )
    .with_system("You are Wintermute.")
}

#[tokio::test]
async fn collect_messages_decodes_full_sse_response() {
    let sse = "event: message_start\n\
               data: {\"type\":\"message_start\"}\n\n\
               event: content_block_start\n\
               data: {\"type\":\"content_block_start\",\"index\":0}\n\n\
               event: content_block_delta\n\
               data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi \"}}\n\n\
               event: content_block_delta\n\
               data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"there\"}}\n\n\
               event: content_block_stop\n\
               data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
               event: message_delta\n\
               data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n\
               event: message_stop\n\
               data: {\"type\":\"message_stop\"}\n\n";

    let (base, recorded) = spawn_mock(StatusCode::OK, sse, "text/event-stream").await;
    let client = AnthropicClient::new("sk-test-123")
        .unwrap()
        .with_base_url(base);
    let req = sample_request();
    let events = client.collect_messages(&req).await.unwrap();

    assert_eq!(events.len(), 7);
    assert_eq!(events[0], StreamEvent::MessageStart { usage: wintermute_brain::anthropic::MessageUsage::default() });
    assert_eq!(events[1], StreamEvent::ContentBlockStart { index: 0 });
    assert_eq!(
        events[2],
        StreamEvent::TextDelta {
            index: 0,
            text: "hi ".to_string(),
        }
    );
    assert_eq!(
        events[3],
        StreamEvent::TextDelta {
            index: 0,
            text: "there".to_string(),
        }
    );
    assert_eq!(events[4], StreamEvent::ContentBlockStop { index: 0 });
    assert_eq!(
        events[5],
        StreamEvent::MessageDelta {
            stop_reason: Some("end_turn".to_string()),
        }
    );
    assert_eq!(events[6], StreamEvent::MessageStop);

    let captured = {
        let guard = recorded.lock().await;
        assert_eq!(guard.len(), 1);
        guard[0].clone()
    };
    assert_eq!(
        captured.headers.get("x-api-key").unwrap().to_str().unwrap(),
        "sk-test-123"
    );
    assert_eq!(
        captured
            .headers
            .get("anthropic-version")
            .unwrap()
            .to_str()
            .unwrap(),
        "2023-06-01"
    );
    assert_eq!(
        captured
            .headers
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "application/json"
    );
    assert_eq!(captured.body["model"], "claude-sonnet-4-6");
    assert_eq!(captured.body["max_tokens"], 128);
    assert_eq!(captured.body["stream"], true);
    assert_eq!(captured.body["system"], "You are Wintermute.");
    assert_eq!(captured.body["messages"][0]["role"], "user");
    assert_eq!(captured.body["messages"][0]["content"], "ping");
}

#[tokio::test]
async fn collect_messages_surfaces_non_2xx_as_status_error() {
    let (base, _recorded) = spawn_mock(
        StatusCode::UNAUTHORIZED,
        r#"{"type":"error","error":{"type":"authentication_error","message":"bad key"}}"#,
        "application/json",
    )
    .await;
    let client = AnthropicClient::new("bad")
        .unwrap()
        .with_base_url(base);
    let err = client.collect_messages(&sample_request()).await.unwrap_err();
    match err {
        ClientError::Status { code, body } => {
            assert_eq!(code, 401);
            assert!(body.contains("authentication_error"));
        }
        other => panic!("expected Status error, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_messages_emits_events_in_order() {
    let sse = "event: message_start\n\
               data: {\"type\":\"message_start\"}\n\n\
               event: content_block_start\n\
               data: {\"type\":\"content_block_start\",\"index\":0}\n\n\
               event: content_block_delta\n\
               data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
               event: message_stop\n\
               data: {\"type\":\"message_stop\"}\n\n";
    let (base, _recorded) = spawn_mock(StatusCode::OK, sse, "text/event-stream").await;
    let client = AnthropicClient::new("k").unwrap().with_base_url(base);

    let mut stream = client
        .stream_messages(&sample_request())
        .await
        .expect("stream construction");
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.expect("stream yields ok"));
    }
    assert_eq!(
        events,
        vec![
            StreamEvent::MessageStart { usage: wintermute_brain::anthropic::MessageUsage::default() },
            StreamEvent::ContentBlockStart { index: 0 },
            StreamEvent::TextDelta {
                index: 0,
                text: "hi".to_string()
            },
            StreamEvent::MessageStop,
        ]
    );
}

#[tokio::test]
async fn stream_messages_surfaces_non_2xx_before_any_item() {
    let (base, _recorded) = spawn_mock(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"type":"error","error":{"type":"rate_limit","message":"slow down"}}"#,
        "application/json",
    )
    .await;
    let client = AnthropicClient::new("k").unwrap().with_base_url(base);
    let err = client
        .stream_messages(&sample_request())
        .await
        .expect_err("non-2xx must surface eagerly");
    match err {
        ClientError::Status { code, body } => {
            assert_eq!(code, 429);
            assert!(body.contains("rate_limit"));
        }
        other => panic!("expected Status, got {other:?}"),
    }
}

#[tokio::test]
async fn collect_messages_skips_ping_keepalive() {
    let sse = ": ping\n\n\
               event: message_start\n\
               data: {\"type\":\"message_start\"}\n\n\
               event: message_stop\n\
               data: {\"type\":\"message_stop\"}\n\n";
    let (base, _recorded) = spawn_mock(StatusCode::OK, sse, "text/event-stream").await;
    let client = AnthropicClient::new("k").unwrap().with_base_url(base);
    let events = client.collect_messages(&sample_request()).await.unwrap();
    assert_eq!(events, vec![StreamEvent::MessageStart { usage: wintermute_brain::anthropic::MessageUsage::default() }, StreamEvent::MessageStop]);
}
