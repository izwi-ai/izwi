//! Shared paged-KV defaults.

/// Default KV page size used when model-specific config is unavailable.
pub const DEFAULT_KV_PAGE_SIZE: usize = 64;

/// Resolve default page size, optionally overridden by env.
pub fn default_kv_page_size() -> usize {
    std::env::var("IZWI_KV_PAGE_SIZE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_KV_PAGE_SIZE)
}
