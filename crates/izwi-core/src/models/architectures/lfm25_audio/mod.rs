//! Native LFM2.5 Audio GGUF architecture support.

mod backbone;
mod bundle;
mod config;
mod model;
mod tokenizer;

pub use model::Lfm25AudioModel;
