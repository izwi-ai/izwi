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

/// The OpenAI compatibility profile, `IZWI_OPENAI_COMPAT_PROFILE`: `strict`
/// (the default) or `relaxed`.
pub fn compatibility_profile() -> OpenAiCompatibilityProfile {
    crate::env_config::tuning().openai_compat_profile
}

pub fn strict_guardrail_message(feature: &str) -> String {
    format!(
        "`{feature}` is not supported in strict OpenAI compatibility mode (`IZWI_OPENAI_COMPAT_PROFILE=strict`)."
    )
}
