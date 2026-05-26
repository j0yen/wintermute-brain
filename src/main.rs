//! `wmd` CLI entrypoint — the wintermute brain (Claude API loop).
//!
//! iter-2 wires a clap dispatcher and `tracing` initialisation. The
//! `start` subcommand resolves a [`BrainConfig`] from env vars and
//! prints it (debug-level) then exits 2 — the live daemon loop (the
//! Anthropic streaming client, the agorabus subscribe loop, the
//! recall-daemon socket client) lands in iter-3+. The `swap-model`,
//! `default-model`, and `status` subcommands are also stubs at exit
//! code 2; they will mutate `~/.config/wintermute/brain.toml` once
//! iter-3 lands the config persistence layer.

#![cfg_attr(not(test), forbid(unsafe_code))]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use wintermute_brain::{BrainConfig, DEFAULT_API_KEY_ENV, validate_model_name};

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

fn main() -> ExitCode {
    init_tracing();
    let cli = Cli::parse();

    let mut cfg = match BrainConfig::from_env() {
        Ok(c) => c,
        Err(err) => {
            error!(error = %err, "wmd: invalid configuration from env");
            return ExitCode::from(1);
        }
    };
    if let Some(sock) = cli.recall_sock {
        cfg.recall_sock = sock;
    }
    cfg.api_key_env = cli.api_key_env;
    if let Err(err) = cfg.validate() {
        error!(error = %err, "wmd: config validation failed");
        return ExitCode::from(1);
    }

    match cli.command {
        Command::Start => run_start(&cfg),
        Command::SwapModel { name } => run_swap_model(&name),
        Command::DefaultModel { name } => run_default_model(&name),
        Command::Status => run_status(&cfg),
    }
}

fn run_start(cfg: &BrainConfig) -> ExitCode {
    info!(?cfg, "wmd start: config resolved");
    warn!("wmd start: daemon loop deferred to iter-3");
    ExitCode::from(2)
}

fn run_swap_model(name: &str) -> ExitCode {
    if let Err(err) = validate_model_name(name) {
        error!(error = %err, "wmd swap-model: unknown model");
        return ExitCode::from(1);
    }
    warn!(
        model = %name,
        "wmd swap-model: config persistence + bus publish deferred to iter-3"
    );
    ExitCode::from(2)
}

fn run_default_model(name: &str) -> ExitCode {
    if let Err(err) = validate_model_name(name) {
        error!(error = %err, "wmd default-model: unknown model");
        return ExitCode::from(1);
    }
    warn!(
        model = %name,
        "wmd default-model: config persistence deferred to iter-3"
    );
    ExitCode::from(2)
}

fn run_status(cfg: &BrainConfig) -> ExitCode {
    match serde_json::to_string_pretty(cfg) {
        Ok(s) => {
            info!(config = %s, "wmd status");
            ExitCode::SUCCESS
        }
        Err(err) => {
            error!(error = %err, "wmd status: failed to serialise config");
            ExitCode::from(1)
        }
    }
}
