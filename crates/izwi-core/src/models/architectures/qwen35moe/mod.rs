//! Qwen3.5 MoE text-chat runtime (`Qwen3_5MoeForConditionalGeneration`).
//!
//! The published Qwen3.5-35B-A3B-FP8 checkpoint combines the Qwen3.5 hybrid
//! backbone (Gated DeltaNet linear attention interleaved 3:1 with gated full
//! attention) with a sparse mixture-of-experts feed-forward block (256 routed
//! experts, 8 active per token, plus one always-on shared expert), stored as
//! 128x128 block-scaled FP8 Safetensors weights.
//!
//! This family deliberately does not reuse the dense `qwen35` or FP8 `qwen38`
//! family loaders: per the dedicated-family posture, each product family owns
//! its loader, decode state, and chat behavior. Only narrow, stable
//! checkpoint-ingestion primitives (indexed shard reader, block-FP8 decode,
//! projection materialization) are shared with `qwen38::native`.

pub mod native;
