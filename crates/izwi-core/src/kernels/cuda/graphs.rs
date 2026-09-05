//! Model-owned stable-input CUDA graph regions for decode and verification.
//!
//! Capture spans pure operations (for example residual addition followed by RMS
//! normalization, or the compact FP8 MLP), retaining all intermediates and weight
//! owners. The model-owned mutex serializes graph APIs across workers; destruction
//! binds the context and fences outstanding work before releasing allocations.
//!
//! Do not capture Candle Q8 fast-MMVQ: its private growable quantization workspace
//! cannot be retained through the public API. A successful warmup does not make
//! that raw scratch pointer stable for the graph lifetime.
#[path = "graphs_region.rs"]
mod region;
pub use region::{IslandOutput, TensorIsland};
