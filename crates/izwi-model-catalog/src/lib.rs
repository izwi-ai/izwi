//! The accelerator-free part of Izwi's model catalog: model identities and
//! their parsing, families, published capabilities and backend support
//! metadata.
//!
//! izwi-core re-exports everything here under its historical paths. Crates
//! that must not link the inference engine (the gateway) depend on this
//! crate instead; check-serving-boundary.sh keeps it free of izwi-core and
//! every accelerator.

mod backend;
pub mod catalog;

pub use backend::BackendKind;
pub use catalog::*;
