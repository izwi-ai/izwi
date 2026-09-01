# Izwi compatibility patch

This is the published `candle-metal-kernels` 0.11.0 crate, overridden through
the workspace's `[patch.crates-io]` section.

Izwi preserves Candle's existing `raw: None` residency fallback when
`MTLResidencySetDescriptor` is unavailable. Apple introduced that class in
macOS 15, while Izwi supports Metal on macOS 12 and later. Without the runtime
availability check, creating any Candle Metal device panics on macOS 12–14.
