//! OpenAI compatibility profile controls.

/// Runtime OpenAI compatibility profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiCompatibilityProfile {
    /// OpenAI wire-level compatibility is prioritized.
    Strict,
    /// Local-runtime extensions and fallbacks are allowed.
    Relaxed,
}

impl OpenAiCompatibilityProfile {
    pub fn is_strict(self) -> bool {
        self == Self::Strict
    }

    pub fn is_relaxed(self) -> bool {
        self == Self::Relaxed
    }
}

/// Resolve the OpenAI compatibility profile from environment.
///
/// `IZWI_OPENAI_COMPAT_PROFILE` accepted values:
/// - `strict` (default)
/// - `relaxed`
pub fn compatibility_profile() -> OpenAiCompatibilityProfile {
    parse_profile(
        std::env::var("IZWI_OPENAI_COMPAT_PROFILE")
            .ok()
            .as_deref(),
    )
}

pub fn strict_guardrail_message(feature: &str) -> String {
    format!(
        "`{feature}` is not supported in strict OpenAI compatibility mode (`IZWI_OPENAI_COMPAT_PROFILE=strict`)."
    )
}

fn parse_profile(raw: Option<&str>) -> OpenAiCompatibilityProfile {
    match raw.map(|value| value.trim().to_ascii_lowercase()).as_deref() {
        Some("relaxed") => OpenAiCompatibilityProfile::Relaxed,
        _ => OpenAiCompatibilityProfile::Strict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_defaults_to_strict() {
        assert_eq!(parse_profile(None), OpenAiCompatibilityProfile::Strict);
    }

    #[test]
    fn profile_accepts_relaxed() {
        assert_eq!(
            parse_profile(Some("relaxed")),
            OpenAiCompatibilityProfile::Relaxed
        );
    }
}
