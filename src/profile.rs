//! Persona profile registry — named presets that compose a complete `[persona]` config.
//!
//! Built-in profiles can be listed, inspected, and applied to a `brain.toml`
//! via the `wmd persona profile` sub-commands.
//!
//! PRD-persona-profile §2.

use std::fmt::Write as FmtWrite;
use std::path::Path;

use anyhow::{Context, Result};

use crate::{BrainConfig, Register, introduction::IntroductionMode};

// ── Built-in forbidden-term lists ─────────────────────────────────────────────

/// Forbidden-term list calibrated for the Jocelyn elder-companion preset.
/// Matches the list validated in `lib.rs::tests::forbidden_terms_jocelyn_preset`.
const JOCELYN_FORBIDDEN: &[&str] = &[
    "AI",
    "artificial intelligence",
    "computer",
    "algorithm",
    "model",
    "machine learning",
    "neural network",
    "robot",
    "device",
    "technology",
    "software",
    "program",
    "digital assistant",
];

// ── PersonaProfile ────────────────────────────────────────────────────────────

/// A complete, named persona identity.
///
/// Profiles bundle all `[persona]` config fields into a single preset that
/// operators can list, inspect, diff against a live config, and apply.
///
/// PRD-persona-profile §2.1.
pub struct PersonaProfile {
    /// Short unique identifier (ASCII, lower-kebab).
    pub name: &'static str,
    /// One-line description shown by `wmd persona profile list`.
    pub description: &'static str,
    /// Tone/vocabulary register.
    pub register: Register,
    /// How the companion refers to herself out loud.
    pub self_name: &'static str,
    /// Voice wake word surfaced in the first-ever-boot teaching block.
    pub wake_word: &'static str,
    /// Spoken-reply brevity ceiling (`0` = no ceiling).
    pub max_sentences: u8,
    /// Whether the user's name is woven into the persona base.
    pub addresses_user: bool,
    /// Words or phrases the companion must never utter.
    pub forbidden_terms: Vec<String>,
    /// Introduction ceremony config.
    pub introduction: ProfileIntroduction,
}

/// Introduction ceremony selector for a [`PersonaProfile`].
///
/// Mirrors [`IntroductionMode`] but uses owned Rust types so profiles can be
/// constructed as static data without lifetime complications.
#[derive(Clone, Debug)]
pub enum ProfileIntroduction {
    /// No introduction ceremony (default).
    Off,
    /// Speak the introduction on the companion's very first boot.
    FirstEverBoot {
        /// Maximum wait in seconds for a voice acknowledgment.
        ack_timeout_secs: u64,
    },
}

impl PersonaProfile {
    /// Return all built-in profiles in declaration order.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![Self::make_jocelyn(), Self::make_default()]
    }

    /// Look up a built-in profile by name (case-insensitive).
    #[must_use]
    pub fn builtin(name: &str) -> Option<Self> {
        let name_lower = name.to_ascii_lowercase();
        Self::all().into_iter().find(|p| p.name == name_lower.as_str())
    }

    /// Emit a valid `[persona]` + `[persona.introduction]` TOML fragment.
    ///
    /// Key names and value formats match the serde attributes on
    /// [`crate::PersonaConfig`] and [`IntroductionMode`] so the fragment
    /// can be pasted directly into `brain.toml` or injected by `apply --write`.
    #[must_use]
    pub fn to_toml_fragment(&self) -> String {
        let mut out = String::new();
        // [persona]
        writeln!(out, "[persona]").ok();
        let register_str = match self.register {
            Register::WarmElder => "warm-elder",
            Register::Plain => "plain",
            Register::Brisk => "brisk",
        };
        writeln!(out, "register = \"{register_str}\"").ok();
        writeln!(out, "self_name = \"{}\"", self.self_name).ok();
        writeln!(out, "wake_word = \"{}\"", self.wake_word).ok();
        writeln!(out, "max_sentences = {}", self.max_sentences).ok();
        writeln!(out, "addresses_user = {}", self.addresses_user).ok();

        // forbidden_terms as inline array
        if self.forbidden_terms.is_empty() {
            writeln!(out, "forbidden_terms = []").ok();
        } else {
            let terms: Vec<String> = self
                .forbidden_terms
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect();
            writeln!(out, "forbidden_terms = [{}]", terms.join(", ")).ok();
        }

        // introduction inline — use TOML dotted key style
        match &self.introduction {
            ProfileIntroduction::Off => {
                writeln!(out, "introduction = \"Off\"").ok();
            }
            ProfileIntroduction::FirstEverBoot { ack_timeout_secs } => {
                writeln!(
                    out,
                    "introduction.FirstEverBoot = {{ ack_timeout_secs = {ack_timeout_secs} }}"
                )
                .ok();
            }
        }

        out
    }

    // ── constructors ──────────────────────────────────────────────────────────

    fn make_jocelyn() -> Self {
        Self {
            name: "jocelyn",
            description: "Elder-companion preset: warm register, self-name Jocelyn, \
                          avoids all tech jargon, speaks her name on first boot.",
            register: Register::WarmElder,
            self_name: "jocelyn",
            wake_word: "hey jocelyn",
            max_sentences: 2,
            addresses_user: true,
            forbidden_terms: JOCELYN_FORBIDDEN
                .iter()
                .map(|&s| s.to_string())
                .collect(),
            introduction: ProfileIntroduction::FirstEverBoot {
                ack_timeout_secs: 25,
            },
        }
    }

    #[allow(clippy::missing_const_for_fn, reason = "Vec::new is const but String fields aren't")]
    fn make_default() -> Self {
        Self {
            name: "default",
            description: "Standard wintermute persona: warm-elder register, \
                          self-name wintermute, no forbidden terms.",
            register: Register::WarmElder,
            self_name: "wintermute",
            wake_word: "hey wintermute",
            max_sentences: 3,
            addresses_user: true,
            forbidden_terms: Vec::new(),
            introduction: ProfileIntroduction::Off,
        }
    }
}

// ── diff / apply helpers ───────────────────────────────────────────────────────

/// Diff a profile against the `[persona]` section of a live `brain.toml`.
///
/// Returns a human-readable diff string.  If the live config matches the
/// profile exactly, returns `None`.
///
/// # Errors
/// Returns an error when the config file cannot be read or parsed.
pub fn diff_profile_vs_config(
    profile: &PersonaProfile,
    config_path: &Path,
) -> Result<Option<String>> {
    let cfg = BrainConfig::load_from_file(config_path)
        .with_context(|| format!("load config {}", config_path.display()))?;
    let live = &cfg.persona;

    let mut diffs: Vec<String> = Vec::new();

    let profile_register_str = match profile.register {
        Register::WarmElder => "warm-elder",
        Register::Plain => "plain",
        Register::Brisk => "brisk",
    };
    let live_register_str = match live.register {
        Register::WarmElder => "warm-elder",
        Register::Plain => "plain",
        Register::Brisk => "brisk",
    };
    if profile_register_str != live_register_str {
        diffs.push(format!("register: {live_register_str:?} → {profile_register_str:?}"));
    }
    if profile.self_name != live.self_name.as_str() {
        diffs.push(format!(
            "self_name: {:?} → {:?}",
            live.self_name, profile.self_name
        ));
    }
    if profile.wake_word != live.wake_word.as_str() {
        diffs.push(format!(
            "wake_word: {:?} → {:?}",
            live.wake_word, profile.wake_word
        ));
    }
    if profile.max_sentences != live.max_sentences {
        diffs.push(format!(
            "max_sentences: {} → {}",
            live.max_sentences, profile.max_sentences
        ));
    }
    if profile.addresses_user != live.addresses_user {
        diffs.push(format!(
            "addresses_user: {} → {}",
            live.addresses_user, profile.addresses_user
        ));
    }
    let live_terms: Vec<&str> = live.forbidden_terms.iter().map(String::as_str).collect();
    let profile_terms: Vec<&str> = profile.forbidden_terms.iter().map(String::as_str).collect();
    if profile_terms != live_terms {
        diffs.push(format!(
            "forbidden_terms: {} terms → {} terms",
            live.forbidden_terms.len(),
            profile.forbidden_terms.len()
        ));
    }

    // Introduction mode comparison (string-based to avoid importing variants)
    let live_intro_str = match &live.introduction {
        IntroductionMode::Off => "Off".to_string(),
        IntroductionMode::FirstEverBoot { ack_timeout_secs } => {
            format!("FirstEverBoot({ack_timeout_secs}s)")
        }
        IntroductionMode::Explicit { ack_timeout_secs } => {
            format!("Explicit({ack_timeout_secs}s)")
        }
    };
    let profile_intro_str = match &profile.introduction {
        ProfileIntroduction::Off => "Off".to_string(),
        ProfileIntroduction::FirstEverBoot { ack_timeout_secs } => {
            format!("FirstEverBoot({ack_timeout_secs}s)")
        }
    };
    if live_intro_str != profile_intro_str {
        diffs.push(format!("introduction: {live_intro_str} → {profile_intro_str}"));
    }

    if diffs.is_empty() {
        Ok(None)
    } else {
        Ok(Some(diffs.join("\n")))
    }
}

/// Apply a profile to `brain.toml`, replacing only the `[persona]` section.
///
/// Backs up the original to `<path>.bak`, then uses [`toml_edit`] to
/// replace only the `[persona]` and `[persona.introduction]` tables while
/// preserving all other tables byte-for-byte.
///
/// # Errors
/// Returns an error when the config file cannot be read, the TOML cannot be
/// parsed or patched, or the file cannot be written.
pub fn apply_profile_to_config(profile: &PersonaProfile, config_path: &Path) -> Result<()> {
    use std::fs;
    use toml_edit::{DocumentMut, Item, Table, value};

    let raw = match fs::read_to_string(config_path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            return Err(e).with_context(|| format!("read {}", config_path.display()));
        }
    };

    // Backup if non-empty
    if !raw.is_empty() {
        let bak = config_path.with_extension("toml.bak");
        fs::write(&bak, raw.as_bytes())
            .with_context(|| format!("write backup {}", bak.display()))?;
    }

    let mut doc: DocumentMut = raw
        .parse()
        .with_context(|| format!("parse TOML in {}", config_path.display()))?;

    // Remove any existing [persona] section (including [persona.introduction]).
    doc.remove("persona");

    // Build the new [persona] table.
    let mut persona_table = Table::new();

    let register_str = match profile.register {
        Register::WarmElder => "warm-elder",
        Register::Plain => "plain",
        Register::Brisk => "brisk",
    };
    persona_table.insert("register", value(register_str));
    persona_table.insert("self_name", value(profile.self_name));
    persona_table.insert("wake_word", value(profile.wake_word));
    persona_table.insert(
        "max_sentences",
        value(i64::from(profile.max_sentences)),
    );
    persona_table.insert("addresses_user", value(profile.addresses_user));

    // forbidden_terms as an array
    let mut terms_array = toml_edit::Array::new();
    for term in &profile.forbidden_terms {
        terms_array.push(term.as_str());
    }
    persona_table.insert("forbidden_terms", Item::Value(toml_edit::Value::Array(terms_array)));

    // introduction inline table
    match &profile.introduction {
        ProfileIntroduction::Off => {
            persona_table.insert("introduction", value("Off"));
        }
        ProfileIntroduction::FirstEverBoot { ack_timeout_secs } => {
            let mut intro = toml_edit::InlineTable::new();
            intro.insert(
                "ack_timeout_secs",
                toml_edit::Value::Integer(toml_edit::Formatted::new(
                    i64::try_from(*ack_timeout_secs).unwrap_or(i64::MAX),
                )),
            );
            let mut first_ever = Table::new();
            first_ever.insert(
                "FirstEverBoot",
                Item::Value(toml_edit::Value::InlineTable(intro)),
            );
            persona_table.insert("introduction", Item::Table(first_ever));
        }
    }

    doc.insert("persona", Item::Table(persona_table));

    let new_raw = doc.to_string();

    // Write atomically via tmp+rename.
    let parent = config_path.parent().ok_or_else(|| {
        anyhow::anyhow!("config path {} has no parent directory", config_path.display())
    })?;
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create dir {}", parent.display()))?;
    }
    let mut tmp_name = config_path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_default();
    tmp_name.push(".tmp");
    let tmp_path = config_path.with_file_name(tmp_name);
    fs::write(&tmp_path, new_raw.as_bytes())
        .with_context(|| format!("write tmp {}", tmp_path.display()))?;
    fs::rename(&tmp_path, config_path)
        .with_context(|| format!("rename to {}", config_path.display()))?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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
    use crate::{BrainConfig, PersonaConfig, Register};

    #[test]
    fn builtin_jocelyn_returns_some_with_correct_fields() {
        let p = PersonaProfile::builtin("jocelyn").expect("jocelyn profile exists");
        assert_eq!(p.name, "jocelyn");
        assert_eq!(p.register as u8, Register::WarmElder as u8);
        assert!(
            !p.forbidden_terms.is_empty(),
            "jocelyn must have non-empty forbidden_terms"
        );
        assert!(
            matches!(p.introduction, ProfileIntroduction::FirstEverBoot { .. }),
            "jocelyn introduction must be FirstEverBoot"
        );
    }

    #[test]
    fn builtin_default_returns_some_with_off_intro() {
        let p = PersonaProfile::builtin("default").expect("default profile exists");
        assert_eq!(p.name, "default");
        assert!(
            matches!(p.introduction, ProfileIntroduction::Off),
            "default introduction must be Off"
        );
        assert!(p.forbidden_terms.is_empty(), "default has no forbidden terms");
    }

    #[test]
    fn builtin_nobody_returns_none() {
        assert!(
            PersonaProfile::builtin("nobody").is_none(),
            "unknown profile must return None"
        );
    }

    #[test]
    fn to_toml_fragment_round_trips_through_persona_config_serde() {
        let p = PersonaProfile::builtin("default").expect("default profile exists");
        let fragment = p.to_toml_fragment();

        // Wrap in a minimal BrainConfig TOML and parse.
        let full_toml = fragment;
        // We need to wrap [persona] in a BrainConfig context.
        let brain_toml = full_toml;
        let cfg: BrainConfig = toml::from_str(&brain_toml).expect("fragment must parse as BrainConfig");
        assert_eq!(cfg.persona.self_name, "wintermute");
        assert_eq!(cfg.persona.register, Register::WarmElder);
        assert!(cfg.persona.forbidden_terms.is_empty());
    }

    #[test]
    fn to_toml_fragment_jocelyn_round_trips() {
        let p = PersonaProfile::builtin("jocelyn").expect("jocelyn profile exists");
        let fragment = p.to_toml_fragment();
        let cfg: BrainConfig = toml::from_str(&fragment).expect("jocelyn fragment must parse");
        assert_eq!(cfg.persona.self_name, "jocelyn");
        assert_eq!(cfg.persona.max_sentences, 2);
        assert_eq!(cfg.persona.forbidden_terms.len(), JOCELYN_FORBIDDEN.len());
        // Introduction must parse to FirstEverBoot
        assert!(
            matches!(cfg.persona.introduction, IntroductionMode::FirstEverBoot { ack_timeout_secs: 25 }),
            "introduction must be FirstEverBoot(25s), got {:?}", cfg.persona.introduction
        );
    }

    #[test]
    fn builtin_case_insensitive_lookup() {
        assert!(PersonaProfile::builtin("Jocelyn").is_some());
        assert!(PersonaProfile::builtin("DEFAULT").is_some());
    }

    #[test]
    fn all_returns_at_least_two_profiles() {
        assert!(PersonaProfile::all().len() >= 2);
    }

    #[test]
    fn jocelyn_forbidden_terms_match_expected_list() {
        let p = PersonaProfile::builtin("jocelyn").expect("jocelyn exists");
        assert_eq!(p.forbidden_terms.len(), 13);
        assert!(p.forbidden_terms.contains(&"AI".to_string()));
        assert!(p.forbidden_terms.contains(&"digital assistant".to_string()));
    }

    #[test]
    fn persona_config_from_profile_fields_composes_cleanly() {
        let p = PersonaProfile::builtin("jocelyn").expect("jocelyn exists");
        let persona = PersonaConfig {
            self_name: p.self_name.to_string(),
            register: p.register,
            addresses_user: p.addresses_user,
            max_sentences: p.max_sentences,
            forbidden_terms: p.forbidden_terms.clone(),
            ..PersonaConfig::default()
        };
        let base = persona.compose_base(None);
        assert!(base.contains("jocelyn"), "self_name woven in");
        assert!(base.contains("never use"), "forbidden instruction present");
    }
}
