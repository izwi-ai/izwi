//! Shared paged-KV defaults.

/// Default KV page size used when model-specific config is unavailable.
pub const DEFAULT_KV_PAGE_SIZE: usize = 64;

use std::sync::atomic::{AtomicUsize, Ordering};

/// The page size the runtime resolved from its configuration, published once
/// at startup. Zero until a runtime publishes it.
static CONFIGURED_KV_PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

/// Publish the engine's resolved page size so every model contract, snapshot
/// interval and invocation pool declares the same page grid as the engine.
/// Before this existed, a page size set in the config file reached only the
/// engine while contracts read `IZWI_KV_PAGE_SIZE` (or 64), which silently
/// disabled hybrid prefix reuse whose snapshot interval no longer matched.
pub(crate) fn publish_configured_kv_page_size(tokens: usize) {
    if tokens > 0 {
        CONFIGURED_KV_PAGE_SIZE.store(tokens, Ordering::Relaxed);
    }
}

/// The KV page size: the runtime-published configuration, else
/// `IZWI_KV_PAGE_SIZE`, else [`DEFAULT_KV_PAGE_SIZE`].
pub fn default_kv_page_size() -> usize {
    match CONFIGURED_KV_PAGE_SIZE.load(Ordering::Relaxed) {
        0 => env_kv_page_size(),
        configured => configured,
    }
}

fn env_kv_page_size() -> usize {
    let Ok(raw) = std::env::var("IZWI_KV_PAGE_SIZE") else {
        return DEFAULT_KV_PAGE_SIZE;
    };
    match raw.trim().parse::<usize>() {
        Ok(tokens) if tokens > 0 => tokens,
        _ => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    value = %raw,
                    default = DEFAULT_KV_PAGE_SIZE,
                    "IZWI_KV_PAGE_SIZE is not a positive integer; using the default page size"
                );
            });
            DEFAULT_KV_PAGE_SIZE
        }
    }
}
