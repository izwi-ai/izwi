//! Qwen3.5 MoE text-chat runtime (`Qwen3_5MoeForConditionalGeneration`).
//!
//! The published Qwen3.6-35B-A3B-FP8 checkpoint combines the Qwen3.5 hybrid
//! backbone (Gated DeltaNet linear attention interleaved 3:1 with gated full
//! attention) with a sparse mixture-of-experts feed-forward block (256 routed
//! experts, 8 active per token, plus one always-on shared expert), stored as
//! 128x128 block-scaled FP8 Safetensors weights.
//!
//! Family posture: the sparse-expert feed-forward block ([`sparse`]) is this
//! family's own, and the loaders, chat wrapper, and decode behavior stay
//! family-owned rather than folding into the dense `qwen35` or FP8 `qwen38`
//! families. The hybrid trunk itself is shared: `qwen35::text` parameterizes
//! the feed-forward branch, so both families build the identical mixer /
//! cache / decode machinery through the [`crate::models::architectures::qwen35::text::Qwen35WeightSource`]
//! seam. Only narrow, stable checkpoint-ingestion primitives (indexed shard
//! reader, block-FP8 decode, projection materialization) are shared with
//! `qwen38::native`.

pub mod chat;
pub(crate) mod gguf;
pub mod native;
pub(crate) mod native_model;
pub(crate) mod sparse;
