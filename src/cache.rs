//! Semantic response cache for the brain turn path.
//!
//! [`SemanticCache`] stores recent (transcript, reply) pairs indexed by
//! embedding vectors so that semantically-similar future transcripts can
//! be served from the cache without hitting the LLM tier ladder.
//!
//! **Design constraints:**
//! - SQLite-backed via `rusqlite`; all operations are synchronous and
//!   must be wrapped in [`tokio::task::spawn_blocking`] at the call site.
//! - Time/date/weather queries bypass the cache via [`cache_unsafe`].
//! - On any embedder failure the cache degrades gracefully: `lookup`
//!   returns `Ok(None)` and `insert` logs a warning but never panics or
//!   blocks the turn.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use tracing::warn;

// ── Default constants ─────────────────────────────────────────────────────────

/// Default maximum number of entries retained in the cache.
pub const DEFAULT_MAX_ENTRIES: usize = 256;

/// Default cosine-similarity threshold for a cache hit.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.92;

// ── Unsafe-pattern guard ──────────────────────────────────────────────────────

/// Patterns whose presence marks a transcript as cache-unsafe.
///
/// Any query that is inherently time-sensitive (clocks, weather, alarms)
/// must not be served from the cache because a stale reply would be
/// actively wrong.
const CACHE_UNSAFE_PATTERNS: &[&str] = &[
    "time",
    "date",
    "weather",
    "timer",
    "alarm",
    "what day",
    "what month",
    "what year",
    "right now",
    "today",
    "tonight",
    "tomorrow",
];

/// Returns `true` when the transcript contains any time/date/weather
/// pattern that makes caching unsafe.
///
/// The check is case-insensitive and operates on the lowercased transcript.
#[must_use]
pub fn cache_unsafe(transcript: &str) -> bool {
    let lower = transcript.to_lowercase();
    CACHE_UNSAFE_PATTERNS.iter().any(|p| lower.contains(p))
}

// ── Embedder trait ────────────────────────────────────────────────────────────

/// Converts a text string into a dense floating-point vector.
///
/// Implementations are expected to be relatively slow (subprocess or
/// network round-trip); call sites must wrap [`SemanticCache`] methods
/// in [`tokio::task::spawn_blocking`].
pub trait Embedder: Send + Sync {
    /// Embed `text` into a floating-point vector.
    ///
    /// # Errors
    /// Returns an error when the embedding backend is unavailable or
    /// when parsing its output fails.  The cache treats all errors as
    /// "embedder not available" and degrades gracefully.
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}

// ── RecallEmbedder ────────────────────────────────────────────────────────────

/// Delegates embedding to the `recall embed` sub-command.
///
/// The binary is resolved via `PATH` so both `~/.local/bin/recall` and
/// any system-wide install are accepted.  The sub-command is expected to
/// write a JSON object of the form `{"embedding": [f32, ...]}` to
/// stdout.
pub struct RecallEmbedder;

impl Embedder for RecallEmbedder {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let output = std::process::Command::new("recall")
            .args(["embed", text])
            .output()
            .context("failed to run `recall embed`")?;

        anyhow::ensure!(
            output.status.success(),
            "recall embed exited with status {}",
            output.status
        );

        let stdout = std::str::from_utf8(&output.stdout)
            .context("recall embed output is not UTF-8")?;

        let val: serde_json::Value =
            serde_json::from_str(stdout).context("recall embed output is not valid JSON")?;

        let arr = val
            .get("embedding")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("recall embed JSON missing `embedding` array"))?;

        let vec: Vec<f32> = arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| {
                        #[allow(clippy::as_conversions, reason = "f64→f32 for embedding storage")]
                        (f as f32)
                    })
                    .ok_or_else(|| anyhow::anyhow!("non-numeric value in embedding array"))
            })
            .collect::<anyhow::Result<Vec<f32>>>()?;

        Ok(vec)
    }
}

// ── CacheHit ──────────────────────────────────────────────────────────────────

/// A successful cache lookup result.
#[derive(Debug, Clone)]
pub struct CacheHit {
    /// The cached reply text to serve.
    pub reply: String,
    /// The tier that originally produced this reply.
    pub tier: String,
    /// Cosine similarity score between the query and the cached embedding.
    pub similarity: f32,
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Compute cosine similarity between two equal-length vectors.
///
/// Returns `0.0` when either vector is all-zeros or when the lengths
/// differ (graceful dimension mismatch handling).
#[must_use]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ── SemanticCache ─────────────────────────────────────────────────────────────

/// SQLite-backed semantic response cache.
///
/// Stores (transcript-embedding, reply, tier) tuples.  Lookups embed the
/// incoming transcript and return the most similar cached reply when the
/// cosine similarity exceeds `similarity_threshold`.
///
/// `rusqlite::Connection` is not `Send + Sync`, so it is wrapped in a
/// `Mutex` to allow sharing via `Arc<SemanticCache>` across Tokio tasks.
/// All methods acquire the mutex for the duration of the SQLite operation.
pub struct SemanticCache {
    db: Mutex<rusqlite::Connection>,
    /// Hard upper bound on the number of cache entries.
    max_entries: usize,
    /// Cosine-similarity threshold for a positive cache hit.
    similarity_threshold: f32,
    /// Embedding backend.
    embedder: Arc<dyn Embedder>,
}

impl SemanticCache {
    /// Open (or create) a semantic cache at `db_path`.
    ///
    /// Creates the `entries` table when it does not yet exist.
    ///
    /// # Errors
    /// Returns an error when the SQLite file cannot be opened or the
    /// schema migration fails.
    pub fn open(
        db_path: &Path,
        embedder: Arc<dyn Embedder>,
        max_entries: usize,
        similarity_threshold: f32,
    ) -> anyhow::Result<Self> {
        let db = rusqlite::Connection::open(db_path)
            .with_context(|| format!("open cache db at {}", db_path.display()))?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS entries (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        INTEGER NOT NULL,
                key_vec   BLOB    NOT NULL,
                reply     TEXT    NOT NULL,
                tier      TEXT    NOT NULL,
                hit_count INTEGER DEFAULT 0
            )",
        )
        .context("create entries table")?;
        Ok(Self {
            db: Mutex::new(db),
            max_entries,
            similarity_threshold,
            embedder,
        })
    }

    /// Returns `true` when `WM_BRAIN_CACHE_DISABLED=1` is set in the
    /// process environment.
    #[must_use]
    pub fn is_disabled() -> bool {
        std::env::var("WM_BRAIN_CACHE_DISABLED")
            .ok()
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// Look up a cached reply for `transcript`.
    ///
    /// Returns `Ok(None)` on a miss or when the embedder fails (never
    /// blocks a turn).
    ///
    /// # Errors
    /// Returns an error only for unexpected SQLite failures (not for
    /// embed failures or cosine misses — those degrade to `None`).
    pub fn lookup(&self, transcript: &str) -> anyhow::Result<Option<CacheHit>> {
        let query_vec = match self.embedder.embed(transcript) {
            Ok(v) => v,
            Err(e) => {
                warn!(err = %e, "semantic-cache: embedder failed during lookup; skipping cache");
                return Ok(None);
            }
        };

        let db = self.db.lock().map_err(|e| anyhow::anyhow!("cache db mutex poisoned: {e}"))?;
        let mut stmt = db
            .prepare("SELECT id, key_vec, reply, tier, hit_count FROM entries")
            .context("prepare SELECT")?;

        struct Row {
            id: i64,
            key_vec: Vec<u8>,
            reply: String,
            tier: String,
        }

        let rows: Vec<Row> = stmt
            .query_map([], |row| {
                Ok(Row {
                    id: row.get(0)?,
                    key_vec: row.get(1)?,
                    reply: row.get(2)?,
                    tier: row.get(3)?,
                })
            })
            .context("query entries")?
            .filter_map(|r| r.ok())
            .collect();

        let mut best_id: Option<i64> = None;
        let mut best_sim: f32 = -1.0;
        let mut best_reply = String::new();
        let mut best_tier = String::new();

        for row in &rows {
            let stored_vec = bytes_to_f32_vec(&row.key_vec);
            let sim = cosine_similarity(&query_vec, &stored_vec);
            if sim > best_sim {
                best_sim = sim;
                best_id = Some(row.id);
                best_reply.clone_from(&row.reply);
                best_tier.clone_from(&row.tier);
            }
        }

        if best_sim >= self.similarity_threshold {
            if let Some(id) = best_id {
                let now_ms = unix_ms_now();
                db.execute(
                    "UPDATE entries SET hit_count = hit_count + 1, ts = ?1 WHERE id = ?2",
                    rusqlite::params![now_ms, id],
                )
                .context("bump hit_count")?;
            }
            Ok(Some(CacheHit {
                reply: best_reply,
                tier: best_tier,
                similarity: best_sim,
            }))
        } else {
            Ok(None)
        }
    }

    /// Insert a (transcript, reply, tier) pair into the cache.
    ///
    /// No-ops when `tier == "cache"` to prevent re-insertion loops.
    /// Silently skips on embedder failures, logging a warning.
    ///
    /// When the table is at `max_entries` capacity, the oldest entry
    /// (smallest `ts`) is evicted before inserting.
    ///
    /// # Errors
    /// Returns an error only for unexpected SQLite failures.
    pub fn insert(&self, transcript: &str, reply: &str, tier: &str) -> anyhow::Result<()> {
        if tier == "cache" {
            return Ok(());
        }

        let key_vec = match self.embedder.embed(transcript) {
            Ok(v) => v,
            Err(e) => {
                warn!(err = %e, "semantic-cache: embedder failed during insert; skipping");
                return Ok(());
            }
        };

        let db = self.db.lock().map_err(|e| anyhow::anyhow!("cache db mutex poisoned: {e}"))?;
        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM entries", [], |row| row.get(0))
            .context("count entries")?;

        #[allow(clippy::as_conversions, reason = "row count fits in usize on any platform")]
        if count as usize >= self.max_entries {
            db.execute(
                "DELETE FROM entries WHERE id = (SELECT id FROM entries ORDER BY ts ASC LIMIT 1)",
                [],
            )
            .context("evict oldest entry")?;
        }

        let key_blob = f32_vec_to_bytes(&key_vec);
        let now_ms = unix_ms_now();

        db.execute(
            "INSERT INTO entries (ts, key_vec, reply, tier) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![now_ms, key_blob, reply, tier],
        )
        .context("insert entry")?;

        Ok(())
    }
}

// ── Serialisation helpers ─────────────────────────────────────────────────────

/// Serialize a `Vec<f32>` as a little-endian byte blob.
fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Deserialize a little-endian byte blob back into a `Vec<f32>`.
///
/// Truncates to the last complete 4-byte chunk if the length is not a
/// multiple of 4.
fn bytes_to_f32_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

// ── Time helper ───────────────────────────────────────────────────────────────

/// Return the current time as Unix milliseconds.
fn unix_ms_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            #[allow(clippy::as_conversions, reason = "milliseconds since epoch fit in i64")]
            (d.as_millis() as i64)
        })
        .unwrap_or(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;

    // ── Test doubles ─────────────────────────────────────────────────────────

    /// Returns a fixed, configurable embedding vector for every input.
    struct MockEmbedder {
        vec: Vec<f32>,
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            Ok(self.vec.clone())
        }
    }

    /// A distinct embedder that returns a slightly rotated vector (for
    /// similarity tests).
    struct NearEmbedder {
        base: Vec<f32>,
        delta: f32,
    }

    impl Embedder for NearEmbedder {
        fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            Ok(self
                .base
                .iter()
                .map(|v| v + self.delta)
                .collect())
        }
    }

    /// Always fails to produce an embedding.
    struct FailEmbedder;

    impl Embedder for FailEmbedder {
        fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            anyhow::bail!("embed unavailable")
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn unit_vec(dim: usize) -> Vec<f32> {
        let v = (1..=dim).map(|i| i as f32).collect::<Vec<_>>();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    fn open_cache(dir: &TempDir, embedder: Arc<dyn Embedder>) -> SemanticCache {
        SemanticCache::open(
            &dir.path().join("cache.db"),
            embedder,
            DEFAULT_MAX_ENTRIES,
            DEFAULT_SIMILARITY_THRESHOLD,
        )
        .unwrap()
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_miss_on_empty_db() {
        let dir = TempDir::new().unwrap();
        let emb = Arc::new(MockEmbedder { vec: unit_vec(4) });
        let cache = open_cache(&dir, emb);
        let hit = cache.lookup("hello").unwrap();
        assert!(hit.is_none(), "empty cache must miss");
    }

    #[test]
    fn test_cache_hit_after_insert() {
        let dir = TempDir::new().unwrap();
        let emb = Arc::new(MockEmbedder { vec: unit_vec(4) });
        let cache = open_cache(&dir, emb);
        cache.insert("hello", "world", "local-3b").unwrap();
        let hit = cache.lookup("hello").unwrap().expect("must hit");
        assert_eq!(hit.reply, "world");
        assert_eq!(hit.tier, "local-3b");
        assert!(hit.similarity > 0.99, "identical vector must have similarity ≈ 1");
    }

    #[test]
    fn test_similar_transcript_hits() {
        let dir = TempDir::new().unwrap();
        let base = unit_vec(16);
        // Insert with base vec
        let insert_emb: Arc<dyn Embedder> = Arc::new(MockEmbedder { vec: base.clone() });
        let cache = SemanticCache::open(
            &dir.path().join("cache.db"),
            Arc::clone(&insert_emb),
            DEFAULT_MAX_ENTRIES,
            0.80, // lower threshold so a slightly-perturbed vector still hits
        )
        .unwrap();
        cache.insert("what is a fox", "a furry mammal", "sonnet").unwrap();

        // Look up with a slightly different vec that should still be similar
        let lookup_emb: Arc<dyn Embedder> = Arc::new(NearEmbedder {
            base: base.clone(),
            delta: 0.001,
        });
        let cache2 = SemanticCache::open(
            &dir.path().join("cache.db"),
            lookup_emb,
            DEFAULT_MAX_ENTRIES,
            0.80,
        )
        .unwrap();
        let hit = cache2.lookup("foxes?").unwrap();
        assert!(hit.is_some(), "similar vector must hit above 0.80 threshold");
    }

    #[test]
    fn test_cache_unsafe_time_patterns() {
        assert!(cache_unsafe("what time is it"), "time query is unsafe");
        assert!(cache_unsafe("What is the DATE today?"), "date query is unsafe");
        assert!(cache_unsafe("set a timer for 5 minutes"), "timer is unsafe");
        assert!(cache_unsafe("what's the weather like?"), "weather is unsafe");
        assert!(!cache_unsafe("tell me a fox story"), "story query is safe");
        assert!(!cache_unsafe("how do planes fly"), "general Q is safe");
    }

    #[test]
    fn test_no_reinsert_of_cache_tier() {
        let dir = TempDir::new().unwrap();
        let emb = Arc::new(MockEmbedder { vec: unit_vec(4) });
        let cache = open_cache(&dir, emb);
        // Inserting with tier="cache" should be a silent no-op
        cache.insert("hello", "reply", "cache").unwrap();
        let db = cache.db.lock().unwrap();
        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM entries", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0, "cache-tier insert must be a no-op");
    }

    #[test]
    fn test_lru_eviction() {
        let dir = TempDir::new().unwrap();
        let emb = Arc::new(MockEmbedder { vec: unit_vec(4) });
        // max_entries=3
        let cache = SemanticCache::open(
            &dir.path().join("cache.db"),
            emb,
            3,
            DEFAULT_SIMILARITY_THRESHOLD,
        )
        .unwrap();
        cache.insert("a", "r1", "local").unwrap();
        cache.insert("b", "r2", "local").unwrap();
        cache.insert("c", "r3", "local").unwrap();
        // This insert should evict one entry
        cache.insert("d", "r4", "local").unwrap();
        let db = cache.db.lock().unwrap();
        let count: i64 = db
            .query_row("SELECT COUNT(*) FROM entries", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 3, "eviction must keep count at max_entries");
    }

    #[test]
    fn test_db_persists_across_reopen() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("cache.db");
        let emb: Arc<dyn Embedder> = Arc::new(MockEmbedder { vec: unit_vec(4) });

        {
            let cache = SemanticCache::open(&db_path, Arc::clone(&emb), DEFAULT_MAX_ENTRIES, DEFAULT_SIMILARITY_THRESHOLD)
                .unwrap();
            cache.insert("persistent", "data", "sonnet").unwrap();
        } // cache is dropped here

        // Reopen from same path
        let cache2 = SemanticCache::open(&db_path, emb, DEFAULT_MAX_ENTRIES, DEFAULT_SIMILARITY_THRESHOLD)
            .unwrap();
        let hit = cache2.lookup("persistent").unwrap().expect("must hit after reopen");
        assert_eq!(hit.reply, "data");
    }

    #[test]
    fn test_miss_when_embedder_fails() {
        let dir = TempDir::new().unwrap();
        let emb = Arc::new(FailEmbedder);
        let cache = SemanticCache::open(
            &dir.path().join("cache.db"),
            emb,
            DEFAULT_MAX_ENTRIES,
            DEFAULT_SIMILARITY_THRESHOLD,
        )
        .unwrap();
        // lookup with FailEmbedder must not error — it degrades to None
        let result = cache.lookup("anything");
        assert!(result.is_ok(), "fail-embedder must not propagate error");
        assert!(result.unwrap().is_none(), "fail-embedder must return None");
    }

    #[test]
    fn test_disabled_check() {
        // WM_BRAIN_CACHE_DISABLED is not set in the test environment
        // so is_disabled() must return false.
        // We cannot unset it here if a parent set it, but the default is false.
        if std::env::var("WM_BRAIN_CACHE_DISABLED").ok().as_deref() != Some("1") {
            assert!(!SemanticCache::is_disabled());
        }
    }
}
