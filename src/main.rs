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
    BrainConfig, BrainError, DEFAULT_API_KEY_ENV, Register, daemon,
    default_config_path, validate_model_name,
    profile::{PersonaProfile, apply_profile_to_config, diff_profile_vs_config},
    router::RoutePrefer,
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
    /// Use a different tier/model for the next turn only. PRD §2.6 +
    /// brain-backend-ladder §2.3.
    SwapModel {
        /// A tier name (`local-3b`, `local-8b`, `haiku`, `sonnet`, `opus`)
        /// or a legacy model id (`claude-sonnet-4-6`, `claude-opus-4-8`, …).
        name: String,
    },
    /// Change the persistent default tier/model. PRD §2.6 +
    /// brain-backend-ladder §2.3.
    DefaultModel {
        /// A tier name (`local-3b`, `local-8b`, `haiku`, `sonnet`, `opus`)
        /// or a legacy model id (`claude-sonnet-4-6`, `claude-opus-4-8`, …).
        name: String,
    },
    /// Print the resolved configuration as JSON and exit. Useful for
    /// integration tests + ops debugging.
    Status,
    /// Inspect or tune the persona without recompiling.
    /// PRD-hearth-persona-config §2.4.
    Persona {
        #[command(subcommand)]
        cmd: PersonaCommand,
    },
    /// Inspect or change the routing configuration.
    /// PRD-wintermute-brain-routing §2.4.
    Route {
        #[command(subcommand)]
        cmd: RouteCommand,
    },
}

/// Sub-commands for `wmd persona`.
#[derive(Subcommand, Debug)]
enum PersonaCommand {
    /// Print the composed persona base (what the model actually receives).
    Show,
    /// Atomically set the register in `brain.toml` and exit.
    SetRegister {
        /// Register name: `warm-elder`, `plain`, or `brisk`.
        register: String,
    },
    /// Inspect or apply named persona profiles.
    /// PRD-persona-profile §2.
    Profile {
        #[command(subcommand)]
        cmd: ProfileCommand,
    },
}

/// Sub-commands for `wmd persona profile`.
#[derive(Subcommand, Debug)]
enum ProfileCommand {
    /// List all built-in profiles (name + description).
    List,
    /// Print the TOML fragment for a named profile.
    Show {
        /// Profile name, e.g. `jocelyn` or `default`.
        name: String,
    },
    /// Compare a named profile with the live `brain.toml` [persona] section.
    Diff {
        /// Profile name, e.g. `jocelyn` or `default`.
        name: String,
    },
    /// Apply a named profile to `brain.toml`.
    ///
    /// Without `--write`, prints a diff only (dry-run).
    /// With `--write`, backs up `brain.toml` to `brain.toml.bak` and
    /// replaces only the `[persona]` section.
    Apply {
        /// Profile name, e.g. `jocelyn` or `default`.
        name: String,
        /// Write the profile to `brain.toml` (default: dry-run/diff only).
        #[arg(long)]
        write: bool,
    },
}

/// Sub-commands for `wmd route`.
#[derive(Subcommand, Debug)]
enum RouteCommand {
    /// Print the effective routing configuration as JSON and exit.
    Status,
    /// Persist the deployment-wide tier preference in `brain.toml`.
    Prefer {
        /// `auto`, `local-only`, or `cloud-only`.
        preference: String,
    },
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
        Command::Persona { cmd } => match cmd {
            PersonaCommand::Show => {
                match load_effective(&config_path, cli.recall_sock, &cli.api_key_env) {
                    Ok(cfg) => run_persona_show(&cfg),
                    Err(err) => {
                        error!(error = %err, "wmd persona show: config load failed");
                        ExitCode::from(1)
                    }
                }
            }
            PersonaCommand::SetRegister { register } => {
                run_persona_set_register(&config_path, &register)
            }
            PersonaCommand::Profile { cmd } => {
                run_persona_profile_cmd(&config_path, cmd)
            }
        },
        Command::Route { cmd } => match cmd {
            RouteCommand::Status => {
                match load_effective(&config_path, cli.recall_sock, &cli.api_key_env) {
                    Ok(cfg) => run_route_status(&cfg, &config_path),
                    Err(err) => {
                        error!(error = %err, "wmd route status: config load failed");
                        ExitCode::from(1)
                    }
                }
            }
            RouteCommand::Prefer { preference } => {
                run_route_prefer(&config_path, &preference)
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
    // Set both the legacy per-turn model override and the ladder's per-turn
    // tier override to the same name. The ladder consumes pending_tier; the
    // legacy single-client path consumes pending_model. Tier names and model
    // short names both validate, so one CLI serves both. PRD AC4.
    cfg.pending_tier = Some(name.to_string());
    cfg.pending_model = Some(name.to_string());
    if let Err(err) = cfg.save_to_file(path) {
        error!(error = %err, "wmd swap-model: save failed");
        return ExitCode::from(1);
    }
    info!(tier = %name, path = %path.display(), "wmd swap-model: pending tier/model set");
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
    // Persist both the ladder default tier and (for the legacy single-client
    // path) the default model. PRD AC4.
    cfg.default_tier = name.to_string();
    cfg.default_model = name.to_string();
    if let Err(err) = cfg.save_to_file(path) {
        error!(error = %err, "wmd default-model: save failed");
        return ExitCode::from(1);
    }
    info!(tier = %name, path = %path.display(), "wmd default-model: persisted");
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

/// Print the composed persona base (the stable system-prompt prefix).
/// PRD-hearth-persona-config §2.4 / AC6.
fn run_persona_show(cfg: &BrainConfig) -> ExitCode {
    let base = cfg.persona.compose_base(cfg.user_name.as_deref());
    info!(persona = %base, "wmd persona show");
    ExitCode::SUCCESS
}

/// Parse a register name from the CLI and persist it atomically.
/// PRD-hearth-persona-config §2.4 / AC6.
#[allow(clippy::cognitive_complexity, reason = "parse -> load -> mutate -> save shell")]
fn run_persona_set_register(path: &std::path::Path, register: &str) -> ExitCode {
    let reg = parse_register(register);
    let Some(reg) = reg else {
        error!(
            register = %register,
            "wmd persona set-register: unknown register; expected warm-elder, plain, or brisk"
        );
        return ExitCode::from(1);
    };
    let mut cfg = match BrainConfig::load_from_file(path) {
        Ok(c) => c,
        Err(err) => {
            error!(error = %err, "wmd persona set-register: load failed");
            return ExitCode::from(1);
        }
    };
    cfg.persona.register = reg;
    if let Err(err) = cfg.save_to_file(path) {
        error!(error = %err, "wmd persona set-register: save failed");
        return ExitCode::from(1);
    }
    info!(
        register = %register,
        path = %path.display(),
        "wmd persona set-register: persisted"
    );
    ExitCode::SUCCESS
}

/// Print the effective routing configuration as JSON.
/// PRD-wintermute-brain-routing §2.4 / `wmd route status`.
fn run_route_status(cfg: &BrainConfig, path: &std::path::Path) -> ExitCode {
    match serde_json::to_string_pretty(&cfg.routing) {
        Ok(s) => {
            info!(routing = %s, config_path = %path.display(), "wmd route status");
            ExitCode::SUCCESS
        }
        Err(err) => {
            error!(error = %err, "wmd route status: failed to serialise routing config");
            ExitCode::from(1)
        }
    }
}

/// Persist the deployment-wide routing preference atomically.
/// PRD-wintermute-brain-routing §2.4 / `wmd route prefer`.
#[allow(clippy::cognitive_complexity, reason = "parse -> load -> mutate -> save shell")]
fn run_route_prefer(path: &std::path::Path, preference: &str) -> ExitCode {
    let Some(pref) = RoutePrefer::parse(preference) else {
        error!(
            preference = %preference,
            "wmd route prefer: unknown preference; expected auto, local-only, or cloud-only"
        );
        return ExitCode::from(1);
    };
    let mut cfg = match BrainConfig::load_from_file(path) {
        Ok(c) => c,
        Err(err) => {
            error!(error = %err, "wmd route prefer: load failed");
            return ExitCode::from(1);
        }
    };
    cfg.routing.prefer = pref;
    if let Err(err) = cfg.save_to_file(path) {
        error!(error = %err, "wmd route prefer: save failed");
        return ExitCode::from(1);
    }
    info!(
        preference = %preference,
        path = %path.display(),
        "wmd route prefer: persisted"
    );
    ExitCode::SUCCESS
}

/// Dispatch `wmd persona profile <cmd>`.
///
/// PRD-persona-profile §2.
#[allow(clippy::cognitive_complexity, reason = "subcommand dispatch shell")]
fn run_persona_profile_cmd(config_path: &std::path::Path, cmd: ProfileCommand) -> ExitCode {
    match cmd {
        ProfileCommand::List => {
            for p in PersonaProfile::all() {
                info!(name = p.name, description = p.description, "persona profile");
            }
            ExitCode::SUCCESS
        }
        ProfileCommand::Show { name } => {
            let Some(p) = PersonaProfile::builtin(&name) else {
                error!(name = %name, "wmd persona profile show: unknown profile");
                return ExitCode::from(1);
            };
            info!(fragment = %p.to_toml_fragment(), "wmd persona profile show");
            ExitCode::SUCCESS
        }
        ProfileCommand::Diff { name } => {
            let Some(p) = PersonaProfile::builtin(&name) else {
                error!(name = %name, "wmd persona profile diff: unknown profile");
                return ExitCode::from(1);
            };
            match diff_profile_vs_config(&p, config_path) {
                Ok(None) => {
                    info!("wmd persona profile diff: no differences");
                    ExitCode::SUCCESS
                }
                Ok(Some(diff)) => {
                    info!(diff = %diff, "wmd persona profile diff: differences found");
                    ExitCode::from(1)
                }
                Err(err) => {
                    error!(error = %err, "wmd persona profile diff: failed");
                    ExitCode::from(1)
                }
            }
        }
        ProfileCommand::Apply { name, write } => {
            let Some(p) = PersonaProfile::builtin(&name) else {
                error!(name = %name, "wmd persona profile apply: unknown profile");
                return ExitCode::from(1);
            };
            if !write {
                // Dry-run: show diff only.
                match diff_profile_vs_config(&p, config_path) {
                    Ok(None) => {
                        info!("wmd persona profile apply (dry-run): already matches profile");
                    }
                    Ok(Some(diff)) => {
                        info!(diff = %diff, "wmd persona profile apply (dry-run): changes that would be applied");
                    }
                    Err(err) => {
                        error!(error = %err, "wmd persona profile apply (dry-run): diff failed");
                        return ExitCode::from(1);
                    }
                }
                return ExitCode::SUCCESS;
            }
            // --write: apply and persist.
            match apply_profile_to_config(&p, config_path) {
                Ok(()) => {
                    info!(
                        profile = p.name,
                        path = %config_path.display(),
                        "wmd persona profile apply: profile written"
                    );
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    error!(error = %err, "wmd persona profile apply: write failed");
                    ExitCode::from(1)
                }
            }
        }
    }
}

/// Parse a register name string into a [`Register`] variant.
///
/// Accepts the kebab-case names defined by the serde `rename_all` rule.
fn parse_register(s: &str) -> Option<Register> {
    match s.trim().to_ascii_lowercase().as_str() {
        "warm-elder" => Some(Register::WarmElder),
        "plain" => Some(Register::Plain),
        "brisk" => Some(Register::Brisk),
        _ => None,
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use super::*;
    use wintermute_brain::{SHORT_MODEL_SONNET, TIER_LOCAL_8B};

    fn is_success(code: ExitCode) -> bool {
        // ExitCode has no Eq; format it (Debug renders SUCCESS vs the int).
        format!("{code:?}") == format!("{:?}", ExitCode::SUCCESS)
    }

    #[test]
    fn swap_model_accepts_tier_name_and_sets_pending_tier() {
        // AC4: swap-model local-8b sets the next-turn tier.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.toml");
        BrainConfig::default().save_to_file(&path).unwrap();
        let code = run_swap_model(&path, TIER_LOCAL_8B);
        assert!(is_success(code), "swap-model local-8b should succeed");
        let cfg = BrainConfig::load_from_file(&path).unwrap();
        assert_eq!(cfg.pending_tier.as_deref(), Some(TIER_LOCAL_8B));
        assert_eq!(cfg.effective_tier(), TIER_LOCAL_8B);
    }

    #[test]
    fn swap_model_legacy_short_name_still_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.toml");
        BrainConfig::default().save_to_file(&path).unwrap();
        let code = run_swap_model(&path, SHORT_MODEL_SONNET);
        assert!(is_success(code));
        let cfg = BrainConfig::load_from_file(&path).unwrap();
        assert_eq!(cfg.effective_tier(), SHORT_MODEL_SONNET);
    }

    #[test]
    fn default_model_accepts_tier_and_persists() {
        // AC4: default-model opus sets the persistent tier.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.toml");
        BrainConfig::default().save_to_file(&path).unwrap();
        let code = run_default_model(&path, "opus");
        assert!(is_success(code));
        let cfg = BrainConfig::load_from_file(&path).unwrap();
        assert_eq!(cfg.resolved_default_tier(), "opus");
    }

    #[test]
    fn swap_model_unknown_tier_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.toml");
        BrainConfig::default().save_to_file(&path).unwrap();
        let code = run_swap_model(&path, "gpt-4o");
        assert!(!is_success(code), "unknown tier must error");
    }
}
