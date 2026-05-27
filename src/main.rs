//! `wmd` CLI entrypoint — the wintermute brain (Claude API loop).
//!
//! `start` resolves a [`BrainConfig`] from disk overlaid with env-var
//! overrides and prints it; the live daemon loop (Anthropic streaming,
//! agorabus subscribe, recall socket) lands in iter-6+. `swap-model`
//! and `default-model` mutate `$XDG_CONFIG_HOME/wintermute/brain.toml`
//! atomically so model changes survive daemon restarts (PRD §2.6 +
//! AC6 of `PRD-wintermute-brain.md`).

#![cfg_attr(not(test), forbid(unsafe_code))]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use wintermute_brain::{
    BrainConfig, BrainError, DEFAULT_API_KEY_ENV, daemon, default_config_path, validate_model_name,
};

#[derive(Parser, Debug)]
#[command(
    name = "wmd",
    version,
    about = "wintermute brain daemon — Claude API loop with persistent memory"
)]
struct Cli {
    /// Override the recall-daemon Unix socket
    /// (defaults to `$XDG_RUNTIME_DIR/recall.sock`).
    #[arg(long, global = true)]
    recall_sock: Option<PathBuf>,

    /// Override the env-var the daemon reads the Anthropic API key
    /// from (defaults to `WM_ANTHROPIC_API_KEY`).
    #[arg(long, global = true, default_value = DEFAULT_API_KEY_ENV)]
    api_key_env: String,

    /// Override the persistent config file path
    /// (defaults to `$XDG_CONFIG_HOME/wintermute/brain.toml`).
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start the daemon (long-running). iter-3 wires the Anthropic
    /// streaming client + agorabus subscribe loop.
    Start,
    /// Use a different model for the next turn only. PRD §2.6.
    SwapModel {
        /// One of: `sonnet`, `opus`, `claude-sonnet-4-6`, `claude-opus-4-7`.
        name: String,
    },
    /// Change the persistent default model. PRD §2.6.
    DefaultModel {
        /// One of: `sonnet`, `opus`, `claude-sonnet-4-6`, `claude-opus-4-7`.
        name: String,
    },
    /// Print the resolved configuration as JSON and exit. Useful for
    /// integration tests + ops debugging.
    Status,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    drop(tracing_subscriber::fmt().with_env_filter(filter).try_init());
}

#[allow(clippy::cognitive_complexity, reason = "subcommand dispatch shell")]
fn main() -> ExitCode {
    init_tracing();
    let cli = Cli::parse();

    let config_path = match resolve_config_path(cli.config.as_deref()) {
        Ok(p) => p,
        Err(err) => {
            error!(error = %err, "wmd: cannot resolve config path");
            return ExitCode::from(1);
        }
    };

    match cli.command {
        Command::Start => match load_effective(&config_path, cli.recall_sock, &cli.api_key_env) {
            Ok(cfg) => run_start(&cfg, &config_path),
            Err(err) => {
                error!(error = %err, "wmd start: config load failed");
                ExitCode::from(1)
            }
        },
        Command::SwapModel { name } => run_swap_model(&config_path, &name),
        Command::DefaultModel { name } => run_default_model(&config_path, &name),
        Command::Status => match load_effective(&config_path, cli.recall_sock, &cli.api_key_env) {
            Ok(cfg) => run_status(&cfg, &config_path),
            Err(err) => {
                error!(error = %err, "wmd status: config load failed");
                ExitCode::from(1)
            }
        },
    }
}

fn resolve_config_path(cli_override: Option<&std::path::Path>) -> Result<PathBuf, BrainError> {
    if let Some(p) = cli_override {
        return Ok(p.to_path_buf());
    }
    if let Ok(v) = std::env::var("WM_BRAIN_CONFIG") {
        if !v.is_empty() {
            return Ok(PathBuf::from(v));
        }
    }
    default_config_path()
}

fn load_effective(
    path: &std::path::Path,
    cli_sock: Option<PathBuf>,
    cli_api_key_env: &str,
) -> Result<BrainConfig, BrainError> {
    let mut cfg = BrainConfig::load_from_file(path)?;
    let env_cfg = BrainConfig::from_env()?;
    // Env overrides on-disk values where the env var was actually set;
    // signal "was set" by comparing the env-derived cfg against defaults.
    let defaults = BrainConfig::default();
    if env_cfg.default_model != defaults.default_model {
        cfg.default_model = env_cfg.default_model;
    }
    if env_cfg.api_key_env != defaults.api_key_env {
        cfg.api_key_env = env_cfg.api_key_env;
    }
    if env_cfg.recall_sock != defaults.recall_sock {
        cfg.recall_sock = env_cfg.recall_sock;
    }
    if env_cfg.user_name.is_some() {
        cfg.user_name = env_cfg.user_name;
    }
    if env_cfg.timezone.is_some() {
        cfg.timezone = env_cfg.timezone;
    }
    if env_cfg.child_lock {
        cfg.child_lock = true;
    }
    if let Some(sock) = cli_sock {
        cfg.recall_sock = sock;
    }
    cfg.api_key_env = cli_api_key_env.to_string();
    cfg.validate()?;
    Ok(cfg)
}

#[allow(clippy::cognitive_complexity, reason = "runtime build + block_on + ExitCode shell")]
fn run_start(cfg: &BrainConfig, config_path: &std::path::Path) -> ExitCode {
    info!(?cfg, "wmd start: config resolved");
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            error!(error = %err, "wmd start: tokio runtime build failed");
            return ExitCode::from(1);
        }
    };
    match runtime.block_on(daemon::run(cfg.clone(), Some(config_path.to_path_buf()))) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!(error = %err, "wmd start: daemon exited with error");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::cognitive_complexity, reason = "validate -> load -> mutate -> save shell")]
fn run_swap_model(path: &std::path::Path, name: &str) -> ExitCode {
    if let Err(err) = validate_model_name(name) {
        error!(error = %err, "wmd swap-model: unknown model");
        return ExitCode::from(1);
    }
    let mut cfg = match BrainConfig::load_from_file(path) {
        Ok(c) => c,
        Err(err) => {
            error!(error = %err, "wmd swap-model: load failed");
            return ExitCode::from(1);
        }
    };
    cfg.pending_model = Some(name.to_string());
    if let Err(err) = cfg.save_to_file(path) {
        error!(error = %err, "wmd swap-model: save failed");
        return ExitCode::from(1);
    }
    info!(model = %name, path = %path.display(), "wmd swap-model: pending_model set");
    ExitCode::SUCCESS
}

#[allow(clippy::cognitive_complexity, reason = "validate -> load -> mutate -> save shell")]
fn run_default_model(path: &std::path::Path, name: &str) -> ExitCode {
    if let Err(err) = validate_model_name(name) {
        error!(error = %err, "wmd default-model: unknown model");
        return ExitCode::from(1);
    }
    let mut cfg = match BrainConfig::load_from_file(path) {
        Ok(c) => c,
        Err(err) => {
            error!(error = %err, "wmd default-model: load failed");
            return ExitCode::from(1);
        }
    };
    cfg.default_model = name.to_string();
    if let Err(err) = cfg.save_to_file(path) {
        error!(error = %err, "wmd default-model: save failed");
        return ExitCode::from(1);
    }
    info!(model = %name, path = %path.display(), "wmd default-model: persisted");
    ExitCode::SUCCESS
}

fn run_status(cfg: &BrainConfig, path: &std::path::Path) -> ExitCode {
    match serde_json::to_string_pretty(cfg) {
        Ok(s) => {
            info!(config = %s, config_path = %path.display(), "wmd status");
            ExitCode::SUCCESS
        }
        Err(err) => {
            error!(error = %err, "wmd status: failed to serialise config");
            ExitCode::from(1)
        }
    }
}
