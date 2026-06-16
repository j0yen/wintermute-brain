//! Persona floor enforcement — the base strain is a non-lowerable floor.
//!
//! When composing the effective policy for a persona overlay, the base strain's
//! Boundaries and redlines are loaded as a mandatory floor.  Any persona rule
//! that would *weaken* a base Boundary (remove a forbidden term or widen an
//! allow that the base forbids) is rejected at load time with a named
//! `floor-violation: <rule>` error.
//!
//! The effective policy is `base ∪ persona_additions`: a persona can add
//! constraints but cannot remove base ones.
//!
//! Base-forbid always wins over persona-allow; persona-forbid adds to base.
//!
//! PRD-inoculate-immune §2.

use std::collections::HashSet;

/// The base strain floor loaded from `inoculate strain --format json`.
///
/// If `inoculate` is absent or fails, the floor is treated as empty and
/// persona load is allowed through without any floor check (fail-open).
#[derive(Debug, Clone, Default)]
pub struct BaseFloor {
    /// Forbidden terms declared in the base strain Boundaries.
    pub forbidden_terms: Vec<String>,
    /// SHA-256 hex hash of the strain, used as `strain_hash` on the resolved
    /// policy.  Empty string when no base strain was loaded.
    pub strain_hash: String,
}

/// Error returned when a persona rule would weaken the base-strain floor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloorViolation {
    /// Human-readable description of the offending rule.
    pub rule: String,
}

impl std::fmt::Display for FloorViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "floor-violation: {}", self.rule)
    }
}

impl std::error::Error for FloorViolation {}

/// Attempt to load the base strain from `inoculate strain --format json`.
///
/// Returns `Ok(floor)` on success, or a non-fatal `Ok(BaseFloor::default())`
/// when inoculate is absent (ENOENT) or returns a non-zero exit code.
/// Only hard I/O errors (not ENOENT) are propagated.
///
/// # Errors
/// Returns an error only for unexpected I/O failures that are not "inoculate
/// not found" or "inoculate returned non-zero".
pub fn load_base_floor() -> Result<BaseFloor, std::io::Error> {
    use std::process::Command;

    let result = Command::new("inoculate")
        .args(["strain", "--format", "json"])
        .output();

    match result {
        Ok(output) if output.status.success() => {
            let floor = parse_strain_json(&output.stdout);
            Ok(floor)
        }
        Ok(_non_zero) => {
            // inoculate present but returned non-zero — warn and fail-open.
            tracing::warn!(
                "inoculate strain returned non-zero; no base floor applied (fail-open)"
            );
            Ok(BaseFloor::default())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // inoculate not installed — warn and fail-open.
            tracing::warn!(
                "inoculate binary not found; no base floor applied (fail-open)"
            );
            Ok(BaseFloor::default())
        }
        Err(e) => {
            // Unexpected I/O error — propagate.
            Err(e)
        }
    }
}

/// Parse the JSON output of `inoculate strain --format json` into a
/// [`BaseFloor`].
///
/// Expected shape (subset we care about):
/// ```json
/// {
///   "strain_hash": "abc123…",
///   "boundaries": {
///     "forbidden_terms": ["AI", "robot", …]
///   }
/// }
/// ```
///
/// Unknown fields are ignored.  Missing fields are treated as empty.
fn parse_strain_json(bytes: &[u8]) -> BaseFloor {
    let Ok(s) = std::str::from_utf8(bytes) else {
        tracing::warn!("inoculate strain output is not valid UTF-8; no floor applied");
        return BaseFloor::default();
    };

    let Ok(v) = serde_json::from_str::<serde_json::Value>(s) else {
        tracing::warn!("inoculate strain output is not valid JSON; no floor applied");
        return BaseFloor::default();
    };

    let strain_hash = v
        .get("strain_hash")
        .and_then(|h| h.as_str())
        .unwrap_or("")
        .to_string();

    // forbidden_terms lives under boundaries.forbidden_terms.
    let forbidden_terms: Vec<String> = v
        .pointer("/boundaries/forbidden_terms")
        .and_then(|arr| arr.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    BaseFloor {
        forbidden_terms,
        strain_hash,
    }
}

/// Check whether `persona_forbidden_terms` would weaken the base floor's
/// forbidden terms.
///
/// A persona *weakens* the floor when it attempts to remove a term that the
/// base declares forbidden (i.e. the persona omits a term that the base
/// requires).  Adding terms is always fine.
///
/// Returns a list of [`FloorViolation`]s — one for each term the persona
/// would drop relative to the base.  An empty list means the persona is
/// floor-compliant.
#[must_use]
pub fn check_floor_violations(
    base: &BaseFloor,
    persona_forbidden_terms: &[String],
) -> Vec<FloorViolation> {
    if base.forbidden_terms.is_empty() {
        return Vec::new();
    }

    let persona_set: HashSet<&str> = persona_forbidden_terms
        .iter()
        .map(String::as_str)
        .collect();

    base.forbidden_terms
        .iter()
        .filter(|term| !persona_set.contains(term.as_str()))
        .map(|term| FloorViolation {
            rule: format!("persona omits base-forbidden term {term:?}"),
        })
        .collect()
}

/// Compute the effective forbidden-term list: `base ∪ persona_additions`.
///
/// Base terms are always present.  Persona terms are merged in, preserving
/// order of first appearance.
#[must_use]
pub fn effective_forbidden_terms(base: &BaseFloor, persona_terms: &[String]) -> Vec<String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut result: Vec<String> = Vec::new();

    for term in base.forbidden_terms.iter().chain(persona_terms.iter()) {
        if seen.insert(term.as_str()) {
            result.push(term.clone());
        }
    }
    result
}

/// Result of applying the base floor to a persona's forbidden-term list.
#[derive(Debug, Clone)]
pub struct FlooredPolicy {
    /// Effective forbidden terms (base ∪ persona additions).
    pub effective_forbidden_terms: Vec<String>,
    /// Hash of the base strain the floor was computed from.
    /// Empty when no base strain was loaded.
    pub strain_hash: String,
    /// Any violations found (populated by [`apply_floor`] — the caller
    /// decides whether to reject or log).
    pub violations: Vec<FloorViolation>,
}

/// Apply the base floor to a persona's forbidden-term list.
///
/// The returned [`FlooredPolicy`] always contains the effective union;
/// `violations` is non-empty when the persona would have weakened the floor.
/// The caller is responsible for rejecting or accepting based on violations.
#[must_use]
pub fn apply_floor(base: &BaseFloor, persona_terms: &[String]) -> FlooredPolicy {
    let violations = check_floor_violations(base, persona_terms);
    let effective = effective_forbidden_terms(base, persona_terms);
    FlooredPolicy {
        effective_forbidden_terms: effective,
        strain_hash: base.strain_hash.clone(),
        violations,
    }
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

    fn base(terms: &[&str]) -> BaseFloor {
        BaseFloor {
            forbidden_terms: terms.iter().map(|s| s.to_string()).collect(),
            strain_hash: "deadbeef".to_string(),
        }
    }

    // ── check_floor_violations ────────────────────────────────────────────────

    #[test]
    fn no_violations_when_base_empty() {
        // AC2 related: if base has no terms, any persona is compliant.
        let b = base(&[]);
        let violations = check_floor_violations(&b, &[]);
        assert!(violations.is_empty());
    }

    #[test]
    fn no_violations_when_persona_adds_terms() {
        // AC3: persona only adds constraints — loads cleanly.
        let b = base(&["AI", "robot"]);
        let persona_terms: Vec<String> = vec!["AI".to_string(), "robot".to_string(), "computer".to_string()];
        let violations = check_floor_violations(&b, &persona_terms);
        assert!(violations.is_empty(), "adding terms is always fine: {violations:?}");
    }

    #[test]
    fn no_violations_exact_match() {
        let b = base(&["AI", "robot"]);
        let persona_terms: Vec<String> = vec!["AI".to_string(), "robot".to_string()];
        let violations = check_floor_violations(&b, &persona_terms);
        assert!(violations.is_empty(), "exact match should have no violations");
    }

    #[test]
    fn violation_when_persona_omits_base_term() {
        // AC2: persona re-allows (omits) a base-forbidden term → floor-violation.
        let b = base(&["AI", "robot"]);
        // persona only has "AI", drops "robot"
        let persona_terms: Vec<String> = vec!["AI".to_string()];
        let violations = check_floor_violations(&b, &persona_terms);
        assert_eq!(violations.len(), 1);
        assert!(
            violations[0].rule.contains("robot"),
            "violation must name the dropped term: {:?}",
            violations[0]
        );
    }

    #[test]
    fn violation_error_message_format() {
        // AC2: error format is "floor-violation: <rule description>".
        let b = base(&["forbidden_term"]);
        let violations = check_floor_violations(&b, &[]);
        assert_eq!(violations.len(), 1);
        let msg = violations[0].to_string();
        assert!(
            msg.starts_with("floor-violation:"),
            "message must start with floor-violation: got {msg:?}"
        );
        assert!(
            msg.contains("forbidden_term"),
            "message must name the term: {msg:?}"
        );
    }

    #[test]
    fn multiple_violations_when_persona_omits_many() {
        let b = base(&["AI", "robot", "algorithm"]);
        let violations = check_floor_violations(&b, &[]);
        assert_eq!(violations.len(), 3, "one violation per omitted base term");
    }

    // ── effective_forbidden_terms ─────────────────────────────────────────────

    #[test]
    fn effective_is_union_of_base_and_persona() {
        // AC3: effective policy = base ∪ additions.
        let b = base(&["AI", "robot"]);
        let persona: Vec<String> = vec!["computer".to_string(), "robot".to_string()];
        let effective = effective_forbidden_terms(&b, &persona);
        // Order: base first, then persona additions.
        assert!(effective.contains(&"AI".to_string()));
        assert!(effective.contains(&"robot".to_string()));
        assert!(effective.contains(&"computer".to_string()));
    }

    #[test]
    fn effective_deduplicates() {
        let b = base(&["AI"]);
        let persona: Vec<String> = vec!["AI".to_string(), "robot".to_string()];
        let effective = effective_forbidden_terms(&b, &persona);
        let count = effective.iter().filter(|t| t.as_str() == "AI").count();
        assert_eq!(count, 1, "AI must appear exactly once");
        assert_eq!(effective.len(), 2);
    }

    #[test]
    fn effective_base_forbid_beats_persona_allow() {
        // AC5: base-forbid (the base declares a term forbidden) always appears
        // in effective list regardless of what persona sends.
        let b = base(&["AI"]);
        // Persona sends no terms (simulating "persona allows AI by omission").
        let effective = effective_forbidden_terms(&b, &[]);
        assert!(
            effective.contains(&"AI".to_string()),
            "base-forbid must appear in effective list even when persona omits it"
        );
    }

    #[test]
    fn effective_persona_forbid_adds_to_base_allow() {
        // AC5: persona-forbid adds to base.  If base has no term but persona
        // forbids it, the effective list includes it.
        let b = base(&[]);  // base allows everything
        let persona: Vec<String> = vec!["robot".to_string()];
        let effective = effective_forbidden_terms(&b, &persona);
        assert!(
            effective.contains(&"robot".to_string()),
            "persona-forbid must be present in effective list"
        );
    }

    // ── apply_floor ───────────────────────────────────────────────────────────

    #[test]
    fn apply_floor_compliant_profile_no_violations() {
        let b = base(&["AI", "robot"]);
        let persona: Vec<String> = vec!["AI".to_string(), "robot".to_string(), "computer".to_string()];
        let policy = apply_floor(&b, &persona);
        assert!(policy.violations.is_empty(), "compliant profile must have no violations");
        assert_eq!(policy.strain_hash, "deadbeef");
    }

    #[test]
    fn apply_floor_violating_profile_has_violations() {
        let b = base(&["AI", "robot"]);
        let persona: Vec<String> = vec!["AI".to_string()]; // omits "robot"
        let policy = apply_floor(&b, &persona);
        assert!(!policy.violations.is_empty(), "weakening profile must have violations");
    }

    #[test]
    fn apply_floor_carries_strain_hash() {
        // AC6: resolved policy carries base strain_hash.
        let b = base(&["AI"]);
        let persona: Vec<String> = vec!["AI".to_string()];
        let policy = apply_floor(&b, &persona);
        assert_eq!(policy.strain_hash, "deadbeef", "strain_hash must be present on resolved policy");
    }

    #[test]
    fn apply_floor_empty_base_no_violations() {
        // No base floor → any persona is compliant.
        let b = BaseFloor::default();
        let policy = apply_floor(&b, &[]);
        assert!(policy.violations.is_empty());
        assert!(policy.strain_hash.is_empty());
    }

    // ── parse_strain_json ─────────────────────────────────────────────────────

    #[test]
    fn parse_strain_json_well_formed() {
        let json = r#"{
            "strain_hash": "abc123",
            "boundaries": {
                "forbidden_terms": ["AI", "robot"]
            }
        }"#;
        let floor = parse_strain_json(json.as_bytes());
        assert_eq!(floor.strain_hash, "abc123");
        assert_eq!(floor.forbidden_terms, vec!["AI", "robot"]);
    }

    #[test]
    fn parse_strain_json_missing_boundaries() {
        let json = r#"{"strain_hash": "xyz"}"#;
        let floor = parse_strain_json(json.as_bytes());
        assert_eq!(floor.strain_hash, "xyz");
        assert!(floor.forbidden_terms.is_empty());
    }

    #[test]
    fn parse_strain_json_invalid_utf8() {
        let floor = parse_strain_json(&[0xFF, 0xFE]);
        assert!(floor.forbidden_terms.is_empty());
        assert!(floor.strain_hash.is_empty());
    }

    #[test]
    fn parse_strain_json_not_json() {
        let floor = parse_strain_json(b"not json at all");
        assert!(floor.forbidden_terms.is_empty());
        assert!(floor.strain_hash.is_empty());
    }

    // ── FloorViolation Display ────────────────────────────────────────────────

    #[test]
    fn floor_violation_display_starts_with_floor_violation() {
        let v = FloorViolation { rule: "persona omits base-forbidden term \"AI\"".to_string() };
        let s = v.to_string();
        assert!(s.starts_with("floor-violation:"));
    }
}
