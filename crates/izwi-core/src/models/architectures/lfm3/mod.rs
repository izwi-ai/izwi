//! LiquidAI LFM3: the tri-hybrid trunk (gated ShortConv, sliding-window and
//! global attention) shared by LFM3-1.7B chat and LFM3-Audio-2B.
//!
//! The trunk runs on the LFM2 backbone (`lfm2::backbone`) with its LFM3
//! features switched on: per-head attention output gates, sandwich norms,
//! per-layer sliding windows with their own RoPE base, and zero-tail partial
//! RoPE on the global layers. This module owns what is LFM3's own: the
//! `lfm3.*` GGUF metadata and the chat template.

pub(crate) mod config;
pub(crate) mod prompt;

pub(crate) use config::parse_lfm3_backbone_config;

#[cfg(test)]
mod tests;
