//! On-disk persistence for [`BrainConfig`].
//!
//! `wmd swap-model` and `wmd default-model` mutate the persistent
//! configuration so that the brain can keep AC6 ("`wmd --model opus`
//! for next turn uses Opus exactly once, then reverts") working across
//! daemon restarts.
//!
//! The file lives at `$XDG_CONFIG_HOME/wintermute/brain.toml` (falling
//! back to `~/.config/wintermute/brain.toml`). Writes are atomic via
//! `<path>.tmp` + `fs::rename`. Missing files load as
//! [`BrainConfig::default`] so a fresh wmd boots cleanly.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::{BrainConfig, BrainError, DEFAULT_CONFIG_BASENAME};

/// Resolve the on-disk config path.
///
/// Honours `$XDG_CONFIG_HOME` per the XDG base-dir spec; falls back to
/// `$HOME/.config` otherwise.
///
/// # Errors
/// Returns [`BrainError::Missing`] when neither `XDG_CONFIG_HOME` nor
/// `HOME` is set — the daemon can't decide where to persist state.
pub fn default_config_path() -> Result<PathBuf, BrainError> {
    config_path_for_env(
        std::env::var("XDG_CONFIG_HOME").ok(),
        std::env::var("HOME").ok(),
    )
}

/// Pure-function variant of [`default_config_path`] used by tests; the
/// public API derives its arguments from process env.
fn config_path_for_env(xdg: Option<String>, home: Option<String>) -> Result<PathBuf, BrainError> {
    let base = xdg
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            home.filter(|s| !s.is_empty())
                .map(|h| PathBuf::from(h).join(".config"))
        })
        .ok_or_else(|| {
            BrainError::Missing("XDG_CONFIG_HOME or HOME for default config path".to_string())
        })?;
    Ok(base.join(DEFAULT_CONFIG_BASENAME))
}

impl BrainConfig {
    /// Load a config from `path`. Returns [`BrainConfig::default`]
    /// silently when the file does not exist — a first-run wmd boots
    /// without requiring the operator to seed brain.toml manually.
    ///
    /// # Errors
    /// - [`BrainError::Io`] when the file exists but cannot be read.
    /// - [`BrainError::InvalidConfig`] when the contents fail to
    ///   deserialize as TOML matching the [`BrainConfig`] schema.
    /// - [`BrainError::UnknownModel`] when the deserialized value
    ///   carries a model name outside [`crate::ALLOWED_MODEL_NAMES`].
    pub fn load_from_file(path: &Path) -> Result<Self, BrainError> {
        match fs::read_to_string(path) {
            Ok(raw) => {
                let cfg: Self = toml::from_str(&raw).map_err(|e| BrainError::InvalidConfig {
                    path: path.to_path_buf(),
                    reason: e.to_string(),
                })?;
                cfg.validate()?;
                Ok(cfg)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(BrainError::Io {
                path: path.to_path_buf(),
                source: e,
            }),
        }
    }

    /// Atomically persist this config to `path`. Creates the parent
    /// directory tree if missing and writes via `<path>.tmp` + rename
    /// so a partial write cannot leave a corrupt brain.toml behind.
    ///
    /// # Errors
    /// - [`BrainError::UnknownModel`] when [`Self::validate`] fails;
    ///   we refuse to persist invalid state.
    /// - [`BrainError::InvalidConfig`] when the value fails to
    ///   serialise (should be unreachable for a `BrainConfig`).
    /// - [`BrainError::Io`] for any filesystem-level failure
    ///   (mkdir, create, write, fsync, rename).
    /// - [`BrainError::Missing`] when `path` has no parent directory.
    pub fn save_to_file(&self, path: &Path) -> Result<(), BrainError> {
        self.validate()?;
        let serialized =
            toml::to_string_pretty(self).map_err(|e| BrainError::InvalidConfig {
                path: path.to_path_buf(),
                reason: e.to_string(),
            })?;

        let parent = path.parent().ok_or_else(|| {
            BrainError::Missing(format!("config path {} has no parent dir", path.display()))
        })?;
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| BrainError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        let tmp = tmp_path_for(path);
        // Scope file handle so it closes (and drops the OS lock) before rename.
        {
            let mut f = fs::File::create(&tmp).map_err(|e| BrainError::Io {
                path: tmp.clone(),
                source: e,
            })?;
            f.write_all(serialized.as_bytes())
                .map_err(|e| BrainError::Io {
                    path: tmp.clone(),
                    source: e,
                })?;
            f.sync_all().map_err(|e| BrainError::Io {
                path: tmp.clone(),
                source: e,
            })?;
        }
        fs::rename(&tmp, path).map_err(|e| BrainError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
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
    use crate::{ALLOWED_MODEL_NAMES, DEFAULT_MODEL_NAME, SHORT_MODEL_OPUS};
    use tempfile::TempDir;

    #[test]
    fn load_missing_file_returns_default() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("brain.toml");
        let cfg = BrainConfig::load_from_file(&path).expect("missing file -> default");
        assert_eq!(cfg, BrainConfig::default());
    }

    #[test]
    fn save_then_load_round_trip() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("nested/sub/brain.toml");
        let original = BrainConfig {
            default_model: SHORT_MODEL_OPUS.to_string(),
            pending_model: Some("claude-sonnet-4-6".to_string()),
            user_name: Some("Mom".to_string()),
            timezone: Some("America/Los_Angeles".to_string()),
            child_lock: true,
            ..BrainConfig::default()
        };
        original.save_to_file(&path).expect("save succeeds");
        assert!(path.exists(), "config file written");
        assert!(path.parent().expect("parent").exists(), "parent dirs created");
        let loaded = BrainConfig::load_from_file(&path).expect("load succeeds");
        assert_eq!(loaded, original);
    }

    #[test]
    fn save_is_atomic_via_tmp_rename() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("brain.toml");
        let cfg = BrainConfig::default();
        cfg.save_to_file(&path).expect("save");
        // tmp must not linger after a successful save
        let tmp = tmp_path_for(&path);
        assert!(!tmp.exists(), "tmp file removed by rename");
        assert!(path.exists(), "final file present");
    }

    #[test]
    fn load_rejects_invalid_toml() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("brain.toml");
        fs::write(&path, "this is = not = toml = at = all").expect("write");
        let err = BrainConfig::load_from_file(&path).expect_err("must error");
        assert!(matches!(err, BrainError::InvalidConfig { .. }));
    }

    #[test]
    fn load_rejects_unknown_model_in_file() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("brain.toml");
        fs::write(&path, "default_model = \"haiku\"\n").expect("write");
        let err = BrainConfig::load_from_file(&path).expect_err("must error");
        assert!(matches!(err, BrainError::UnknownModel { ref name, .. } if name == "haiku"));
    }

    #[test]
    fn save_refuses_invalid_config() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("brain.toml");
        let bad = BrainConfig {
            default_model: "haiku".to_string(),
            ..BrainConfig::default()
        };
        let err = bad.save_to_file(&path).expect_err("must error");
        assert!(matches!(err, BrainError::UnknownModel { .. }));
        assert!(!path.exists(), "no file written on validation failure");
    }

    #[test]
    fn loaded_value_round_trips_through_all_allowed_models() {
        let dir = TempDir::new().expect("tempdir");
        for (i, name) in ALLOWED_MODEL_NAMES.iter().enumerate() {
            let path = dir.path().join(format!("brain-{i}.toml"));
            let cfg = BrainConfig {
                default_model: (*name).to_string(),
                ..BrainConfig::default()
            };
            cfg.save_to_file(&path).expect("save");
            let back = BrainConfig::load_from_file(&path).expect("load");
            assert_eq!(back.default_model, *name);
        }
    }

    #[test]
    fn config_path_honours_xdg_when_set() {
        let p = config_path_for_env(Some("/tmp/wm-xdg".to_string()), None)
            .expect("xdg present");
        assert_eq!(p, PathBuf::from("/tmp/wm-xdg/wintermute/brain.toml"));
    }

    #[test]
    fn config_path_falls_back_to_home_dot_config() {
        let p = config_path_for_env(None, Some("/tmp/wm-home".to_string()))
            .expect("home fallback");
        assert_eq!(
            p,
            PathBuf::from("/tmp/wm-home/.config/wintermute/brain.toml")
        );
    }

    #[test]
    fn config_path_errors_when_no_locator_set() {
        let err = config_path_for_env(None, None).expect_err("no locator");
        assert!(matches!(err, BrainError::Missing(_)));
    }

    #[test]
    fn config_path_empty_xdg_falls_through_to_home() {
        let p = config_path_for_env(Some(String::new()), Some("/tmp/wm-home2".to_string()))
            .expect("home fallback over empty xdg");
        assert_eq!(
            p,
            PathBuf::from("/tmp/wm-home2/.config/wintermute/brain.toml")
        );
    }

    #[test]
    fn config_path_xdg_wins_over_home() {
        let p = config_path_for_env(
            Some("/tmp/xdg-wins".to_string()),
            Some("/tmp/home-loses".to_string()),
        )
        .expect("xdg wins");
        assert_eq!(p, PathBuf::from("/tmp/xdg-wins/wintermute/brain.toml"));
    }

    #[test]
    fn pending_model_swap_persists_across_reload() {
        // AC6 acceptance-test seed: swap-model writes pending_model;
        // a fresh load surfaces it; consume_pending + save reverts.
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("brain.toml");

        BrainConfig::default()
            .save_to_file(&path)
            .expect("seed default");

        let mut cfg = BrainConfig::load_from_file(&path).expect("reload default");
        cfg.pending_model = Some(SHORT_MODEL_OPUS.to_string());
        cfg.save_to_file(&path).expect("save with pending");

        let reloaded = BrainConfig::load_from_file(&path).expect("reload after swap");
        assert_eq!(reloaded.effective_model(), SHORT_MODEL_OPUS);

        let mut after_turn = reloaded;
        after_turn.consume_pending();
        after_turn.save_to_file(&path).expect("save after turn");

        let final_cfg = BrainConfig::load_from_file(&path).expect("reload after revert");
        assert_eq!(final_cfg.effective_model(), DEFAULT_MODEL_NAME);
        assert!(final_cfg.pending_model.is_none());
    }
}
