//! OpenAI compatibility profile controls. The profile type is izwi-gateway's;
//! the local role reads its profile from the server tuning.

pub use izwi_gateway::api::openai::compat::*;

/// The OpenAI compatibility profile, `IZWI_OPENAI_COMPAT_PROFILE`: `strict`
/// (the default) or `relaxed`.
pub fn compatibility_profile() -> OpenAiCompatibilityProfile {
    crate::env_config::tuning().openai_compat_profile
}
