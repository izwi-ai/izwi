//! Whisper family implementations.

pub mod asr;
mod model;
mod physical;

#[cfg(test)]
pub(crate) use physical::whisper_physical_state_spec;
pub(crate) use physical::WhisperPhysicalStateSpec;
