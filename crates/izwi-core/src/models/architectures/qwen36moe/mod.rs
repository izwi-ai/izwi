//! Qwen3.5 MoE text-chat runtime (`Qwen3_5MoeForConditionalGeneration`).
//!
//! The published Qwen3.6-35B-A3B-FP8 checkpoint combines the Qwen3.5 hybrid
//! backbone (Gated DeltaNet linear attention interleaved 3:1 with gated full
//! attention) with a sparse mixture-of-experts feed-forward block (256 routed
//! experts, 8 active per token, plus one always-on shared expert), stored as
//! 128x128 block-scaled FP8 Safetensors weights.
//!
//! Family posture: this module owns its whole text stack. The hybrid trunk
//! ([`text`]), state contract ([`cache`]), MTP draft head ([`mtp`]), and
//! decode/exec core ([`exec`]) began as a fork of the dense `qwen35` family
//! and are deliberately NOT shared with it, so Qwen3.5 and Qwen3.6 changes
//! cannot reach each other. Only narrow checkpoint-ingestion primitives
//! (indexed shard reader, block-FP8 decode, projection materialization) are
//! still borrowed from `qwen38::native`.

pub(crate) mod cache;
pub mod chat;
pub mod exec;
pub(crate) mod gguf;
pub(crate) mod mtp;
pub mod native;
pub(crate) mod native_model;
pub(crate) mod sparse;
pub(crate) mod text;
