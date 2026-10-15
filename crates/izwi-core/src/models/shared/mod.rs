//! Shared model infrastructure used by all architecture implementations.
//!
//! This module holds reusable components that should not depend on any
//! specific model family.

pub mod attention;
pub mod chat;
pub mod config;
pub(crate) mod detokenize;
pub mod grammar;
pub mod memory;
pub mod moe;
pub mod sampling;
pub mod speculative_sampling;
pub(crate) mod speech_duration;
pub(crate) mod state;
pub mod telemetry;
pub(crate) mod text_normalize;
pub mod weights;

/// The Metal device for a model test's Metal leg, held exclusively, or `None`
/// after a skip note written past libtest's output capture.
/// `IZWI_REQUIRE_METAL_PARITY=1` turns the skip into a failure on lanes that
/// own a Metal GPU. Legs that hold the guard never overlap: Metal work on
/// another thread can corrupt a readback here (seen as stray zero token ids
/// in the unchanged single-row sampler while a tiny model decoded through the
/// accelerator KV arena on another thread).
#[cfg(all(test, feature = "metal"))]
pub(crate) fn metal_test_device_or_skip(
    test: &str,
) -> Option<(candle_core::Device, std::sync::MutexGuard<'static, ()>)> {
    use std::io::Write;

    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let Some(device) = crate::backends::metal_device_if_available(0) else {
        assert!(
            std::env::var_os("IZWI_REQUIRE_METAL_PARITY").is_none(),
            "{test}: IZWI_REQUIRE_METAL_PARITY is set but no Metal device is available"
        );
        let _ = writeln!(std::io::stderr(), "{test}: skipped, no Metal device");
        return None;
    };
    let guard = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Some((device, guard))
}
