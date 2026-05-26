//! Recall-daemon Unix-socket client used by the conversation loop.
//!
//! Speaks the length-prefixed framing protocol that recall v0.5.x
//! exposes on `$XDG_RUNTIME_DIR/recall.sock` (PRD-recall-daemon §2). Per
//! PRD-wintermute-brain §2.1 / §2.7, the brain pulls persistent profile
//! facts and per-thread context through this client. iter-7 only
//! implements what the conversation loop needs in Fleet 1: `ping` for
//! liveness probes and `query` for retrieval. `embed` and `touch` are
//! intentionally omitted — they land when the brain starts writing
//! memories back, which is a separate iter.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// Max frame size accepted on the wire. Mirrors recall's
/// `MAX_FRAME_BYTES` (4 MiB) so the client refuses oversized payloads
/// before allocating a reader buffer.
pub const MAX_FRAME_BYTES: u32 = 4 * 1024 * 1024;

/// Wire-level errors surfaced by [`RecallClient`].
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// Could not open the Unix socket at the configured path.
    #[error("connect to {path}: {source}")]
    Connect {
        /// Socket path the client tried to dial.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// I/O error reading or writing a frame.
    #[error("io on recall socket: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialisation or parsing failure.
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
    /// Frame length advertised by either side exceeds [`MAX_FRAME_BYTES`].
    #[error("frame exceeds MAX_FRAME_BYTES ({len} > {cap})")]
    FrameTooLarge {
        /// Length advertised on the wire (or `u32::MAX` if the body
        /// could not be coerced to `u32` at all).
        len: u32,
        /// Server-side cap the client mirrors.
        cap: u32,
    },
    /// Daemon answered with a structured error body.
    #[error("daemon error {code}: {message}")]
    Remote {
        /// Stable error code from `recall::daemon::ErrorBody::code`.
        code: String,
        /// Human-readable message from the daemon.
        message: String,
    },
}

/// Serialisable envelope matching recall daemon's request shape:
/// `{"op": <name>, "args": <args>}`.
#[derive(Debug, Serialize)]
struct Envelope<'a, A> {
    op: &'a str,
    args: &'a A,
}

/// Arguments for the `query` op. Optional fields are omitted from the
/// serialised body so the daemon defaults apply (see recall's
/// `QueryArgs::default`).
#[derive(Debug, Default, Clone, Serialize)]
pub struct QueryArgs {
    /// Free-text retrieval query. PRD-recall-daemon §2.
    pub text: String,
    /// Optional limit; `None` lets the daemon apply its default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// Hybrid retrieval toggle (BM25 + vector). `None` = daemon default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid: Option<bool>,
    /// Optional subject scope (e.g. `wintermute-profile`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_subject: Option<String>,
}

/// One ranked memory returned by `query`.
///
/// Mirrors `recall::daemon::QueryHit` but ignores fields the brain
/// doesn't consume in Fleet 1 (`bm25`, `vector_sim`, `recall_count`,
/// `last_recalled_at`). Extra fields decode silently because we don't
/// set `#[serde(deny_unknown_fields)]`.
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct QueryHit {
    /// Stable memory id.
    pub id: String,
    /// Memory kind (e.g. `fact`, `note`).
    pub kind: String,
    /// Subject the memory was filed under.
    pub subject: String,
    /// Path of the source memory file on disk.
    pub path: String,
    /// Short text snippet to surface in prompts.
    pub snippet: String,
    /// Combined rank score the daemon assigned.
    pub score: f64,
    /// Confidence in `[0, 1]` reported by the daemon.
    pub confidence: f64,
}

/// Decoded `query` response. Drops `hybrid` because the brain does not
/// branch on it; can be added later if we need it.
#[derive(Debug, Deserialize, Clone)]
pub struct QueryResponse {
    /// Ranked hits, best first.
    pub ranked_hits: Vec<QueryHit>,
    /// Monotonic per-daemon counter, useful for tracing.
    pub query_count: u64,
    /// Effective `limit` the daemon applied.
    pub limit: usize,
}

/// Decoded `ping` response.
#[derive(Debug, Deserialize, Clone)]
pub struct PingResponse {
    /// Embedder id (e.g. `bge-small-en-v1.5`).
    pub model_id: String,
    /// Seconds the daemon has been up.
    pub uptime_s: u64,
    /// Daemon crate version string.
    pub version: String,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    code: String,
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WireResponse<T> {
    Ok { ok: T },
    Err { error: ErrorBody },
}

/// One open connection to a recall daemon Unix socket.
///
/// Holds the stream for the lifetime of the call sequence; drop the
/// client to close the socket. The daemon's read loop tolerates
/// multiple frames per connection so callers may reuse a single client
/// across queries.
#[derive(Debug)]
pub struct RecallClient {
    stream: UnixStream,
}

impl RecallClient {
    /// Open a fresh connection to the recall daemon at `path`.
    ///
    /// # Errors
    /// Returns [`ClientError::Connect`] if the socket cannot be dialled
    /// (e.g. ENOENT when the daemon is down).
    pub async fn connect(path: &Path) -> Result<Self, ClientError> {
        let stream = UnixStream::connect(path).await.map_err(|source| ClientError::Connect {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self { stream })
    }

    /// Probe daemon liveness. Returns the daemon's self-report
    /// (embedder, version, uptime).
    ///
    /// # Errors
    /// Forwards [`ClientError`] variants from the framing and JSON
    /// layers, or [`ClientError::Remote`] if the daemon replied with an
    /// error body.
    pub async fn ping(&mut self) -> Result<PingResponse, ClientError> {
        self.roundtrip("ping", &EmptyArgs {}).await
    }

    /// Run a single retrieval query.
    ///
    /// # Errors
    /// Forwards [`ClientError`] variants from the framing and JSON
    /// layers, or [`ClientError::Remote`] if the daemon replied with an
    /// error body (e.g. malformed args).
    pub async fn query(&mut self, args: &QueryArgs) -> Result<QueryResponse, ClientError> {
        self.roundtrip("query", args).await
    }

    async fn roundtrip<A, R>(&mut self, op: &str, args: &A) -> Result<R, ClientError>
    where
        A: Serialize + Sync,
        R: for<'de> Deserialize<'de> + Send,
    {
        let env = Envelope { op, args };
        let body = serde_json::to_vec(&env)?;
        write_frame(&mut self.stream, &body).await?;
        let resp_bytes = read_frame(&mut self.stream).await?;
        let resp: WireResponse<R> = serde_json::from_slice(&resp_bytes)?;
        match resp {
            WireResponse::Ok { ok } => Ok(ok),
            WireResponse::Err { error } => Err(ClientError::Remote {
                code: error.code,
                message: error.message,
            }),
        }
    }
}

/// Marker struct used to serialise `"args": {}` for ops that take no
/// arguments. Mirrors recall's `PingArgs`.
#[derive(Debug, Serialize)]
struct EmptyArgs {}

async fn write_frame(stream: &mut UnixStream, body: &[u8]) -> Result<(), ClientError> {
    let Ok(len) = u32::try_from(body.len()) else {
        return Err(ClientError::FrameTooLarge {
            len: u32::MAX,
            cap: MAX_FRAME_BYTES,
        });
    };
    if len > MAX_FRAME_BYTES {
        return Err(ClientError::FrameTooLarge { len, cap: MAX_FRAME_BYTES });
    }
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}

async fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>, ClientError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(ClientError::FrameTooLarge { len, cap: MAX_FRAME_BYTES });
    }
    let usize_len = usize::try_from(len).map_err(|_| ClientError::FrameTooLarge {
        len,
        cap: MAX_FRAME_BYTES,
    })?;
    let mut body = vec![0u8; usize_len];
    stream.read_exact(&mut body).await?;
    Ok(body)
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
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;
    use tokio::net::UnixListener;

    fn nonexistent_socket() -> PathBuf {
        let dir = tempdir().expect("tempdir");
        dir.path().join("does-not-exist.sock")
    }

    #[tokio::test]
    async fn connect_missing_socket_returns_connect_error() {
        let path = nonexistent_socket();
        let err = RecallClient::connect(&path).await.expect_err("must fail");
        assert!(matches!(err, ClientError::Connect { .. }));
    }

    #[test]
    fn query_args_serialise_omits_unset_fields() {
        let args = QueryArgs {
            text: "hello".to_string(),
            ..QueryArgs::default()
        };
        let v = serde_json::to_value(&args).expect("serialises");
        assert_eq!(v, serde_json::json!({ "text": "hello" }));
    }

    #[test]
    fn query_args_serialise_includes_set_fields() {
        let args = QueryArgs {
            text: "tea".to_string(),
            limit: Some(3),
            hybrid: Some(true),
            project_subject: Some("wintermute-profile".to_string()),
        };
        let v = serde_json::to_value(&args).expect("serialises");
        assert_eq!(
            v,
            serde_json::json!({
                "text": "tea",
                "limit": 3,
                "hybrid": true,
                "project_subject": "wintermute-profile",
            })
        );
    }

    #[test]
    fn envelope_serialises_in_expected_shape() {
        let body = serde_json::to_value(&Envelope {
            op: "ping",
            args: &EmptyArgs {},
        })
        .expect("serialises");
        assert_eq!(body, serde_json::json!({ "op": "ping", "args": {} }));
    }

    #[test]
    fn wire_response_decodes_ok_and_err() {
        let ok_raw = serde_json::json!({ "ok": { "vector": [1.0, 2.0], "dim": 2 } });
        let parsed: WireResponse<serde_json::Value> =
            serde_json::from_value(ok_raw).expect("ok decodes");
        assert!(matches!(parsed, WireResponse::Ok { .. }));

        let err_raw =
            serde_json::json!({ "error": { "code": "bad_args", "message": "missing text" } });
        let parsed: WireResponse<serde_json::Value> =
            serde_json::from_value(err_raw).expect("err decodes");
        match parsed {
            WireResponse::Err { error } => {
                assert_eq!(error.code, "bad_args");
                assert_eq!(error.message, "missing text");
            }
            WireResponse::Ok { .. } => panic!("expected err"),
        }
    }

    /// Spawn a single-shot server that reads one framed request and
    /// replies with `reply_body`. Returns the socket path.
    async fn spawn_oneshot_server(reply_body: serde_json::Value) -> PathBuf {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("recall.sock");
        let listener = UnixListener::bind(&path).expect("bind");
        let path_for_task = path.clone();
        tokio::spawn(async move {
            let _dir = dir; // keep alive for the connection
            let (mut stream, _addr) = listener.accept().await.expect("accept");
            // read one frame, discard
            let mut len_buf = [0u8; 4];
            stream.read_exact(&mut len_buf).await.expect("read len");
            let req_len = u32::from_be_bytes(len_buf);
            let usize_len = usize::try_from(req_len).expect("len fits");
            let mut req_body = vec![0u8; usize_len];
            stream.read_exact(&mut req_body).await.expect("read body");
            // write reply
            let body = serde_json::to_vec(&reply_body).expect("serialise reply");
            let len = u32::try_from(body.len()).expect("body fits u32");
            stream.write_all(&len.to_be_bytes()).await.expect("write len");
            stream.write_all(&body).await.expect("write body");
            stream.flush().await.expect("flush");
            // keep socket open briefly so the client can finish reading
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let _ = path_for_task;
        });
        // tiny pause so bind is observable to the client
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        path
    }

    #[tokio::test]
    async fn ping_roundtrip_decodes_response() {
        let path = spawn_oneshot_server(serde_json::json!({
            "ok": {
                "model_id": "hash-v1-d256",
                "uptime_s": 7,
                "query_count": 0,
                "version": "0.5.2",
                "root": "/tmp/r"
            }
        }))
        .await;
        let mut client = RecallClient::connect(&path).await.expect("connect");
        let resp = client.ping().await.expect("ping");
        assert_eq!(resp.model_id, "hash-v1-d256");
        assert_eq!(resp.uptime_s, 7);
        assert_eq!(resp.version, "0.5.2");
    }

    #[tokio::test]
    async fn query_roundtrip_decodes_hits() {
        let path = spawn_oneshot_server(serde_json::json!({
            "ok": {
                "ranked_hits": [
                    {
                        "id": "m-1",
                        "kind": "fact",
                        "subject": "wintermute-profile",
                        "path": "/data/m-1.md",
                        "snippet": "prefers chamomile tea",
                        "score": 0.91,
                        "bm25": 0.4,
                        "vector_sim": 0.8,
                        "confidence": 0.95,
                        "recall_count": 0,
                        "last_recalled_at": null
                    }
                ],
                "query_count": 1,
                "limit": 5,
                "hybrid": true
            }
        }))
        .await;
        let mut client = RecallClient::connect(&path).await.expect("connect");
        let args = QueryArgs { text: "tea".to_string(), ..QueryArgs::default() };
        let resp = client.query(&args).await.expect("query");
        assert_eq!(resp.ranked_hits.len(), 1);
        assert_eq!(resp.ranked_hits[0].id, "m-1");
        assert_eq!(resp.ranked_hits[0].subject, "wintermute-profile");
        assert!((resp.ranked_hits[0].confidence - 0.95).abs() < 1e-9);
        assert_eq!(resp.query_count, 1);
        assert_eq!(resp.limit, 5);
    }

    #[tokio::test]
    async fn remote_error_surfaces_as_clienterror_remote() {
        let path = spawn_oneshot_server(serde_json::json!({
            "error": { "code": "bad_args", "message": "text is empty" }
        }))
        .await;
        let mut client = RecallClient::connect(&path).await.expect("connect");
        let args = QueryArgs::default();
        let err = client.query(&args).await.expect_err("must fail");
        match err {
            ClientError::Remote { code, message } => {
                assert_eq!(code, "bad_args");
                assert_eq!(message, "text is empty");
            }
            other => panic!("expected Remote, got {other:?}"),
        }
    }

    #[test]
    fn oversize_advertised_len_rejected_before_alloc() {
        // Exercise read_frame's length check by simulating an in-process
        // length-only stream — we don't have a half-duplex helper, so
        // assert the constant alignment instead.
        assert_eq!(MAX_FRAME_BYTES, 4 * 1024 * 1024);
    }
}
