//! Per-model decode counters for the Qwen3.5 admin diagnostics.
//!
//! Every counter is cumulative since load. Dividing one by `decode_tokens`
//! gives the per-token figure that a deploy check compares across commits:
//! blocking host reads, host-built RoPE uploads, conv-ring stacks and the
//! sampler path each fast path is meant to remove or change.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Which sampler produced a decode token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SamplerPath {
    /// Device argmax with scalar readbacks (temperature 0, no penalties).
    Greedy,
    /// Bounded device candidates with one packed readback.
    Device,
    /// Full logits row copied to the host.
    Host,
    /// JSON grammar constraint, sampled on the host.
    Grammar,
}

#[derive(Debug, Default)]
pub(crate) struct Qwen35Telemetry {
    decode_tokens: AtomicU64,
    solo_decode_steps: AtomicU64,
    batched_decode_steps: AtomicU64,
    batched_decode_rows: AtomicU64,
    sampling_host_reads: AtomicU64,
    sampler_greedy: AtomicU64,
    sampler_device: AtomicU64,
    sampler_host: AtomicU64,
    sampler_grammar: AtomicU64,
    rope_host_uploads: AtomicU64,
    conv_ring_stacks: AtomicU64,
}

impl Qwen35Telemetry {
    pub(crate) fn record_solo_decode_step(&self) {
        self.solo_decode_steps.fetch_add(1, Relaxed);
    }

    pub(crate) fn record_batched_decode_step(&self, rows: usize) {
        self.batched_decode_steps.fetch_add(1, Relaxed);
        self.batched_decode_rows.fetch_add(rows as u64, Relaxed);
    }

    /// One sampled token: its sampler path and the blocking device-to-host
    /// reads the sampler performed for it.
    pub(crate) fn record_sample(&self, path: SamplerPath, host_reads: u64) {
        self.decode_tokens.fetch_add(1, Relaxed);
        self.sampling_host_reads.fetch_add(host_reads, Relaxed);
        match path {
            SamplerPath::Greedy => &self.sampler_greedy,
            SamplerPath::Device => &self.sampler_device,
            SamplerPath::Host => &self.sampler_host,
            SamplerPath::Grammar => &self.sampler_grammar,
        }
        .fetch_add(1, Relaxed);
    }

    pub(crate) fn record_rope_host_upload(&self) {
        self.rope_host_uploads.fetch_add(1, Relaxed);
    }

    pub(crate) fn record_conv_ring_stacks(&self, count: usize) {
        self.conv_ring_stacks.fetch_add(count as u64, Relaxed);
    }

    pub(crate) fn snapshot(&self) -> serde_json::Value {
        let load = |counter: &AtomicU64| counter.load(Relaxed);
        serde_json::json!({
            "decode_tokens": load(&self.decode_tokens),
            "solo_decode_steps": load(&self.solo_decode_steps),
            "batched_decode_steps": load(&self.batched_decode_steps),
            "batched_decode_rows": load(&self.batched_decode_rows),
            "sampling_host_reads": load(&self.sampling_host_reads),
            "sampler": {
                "greedy": load(&self.sampler_greedy),
                "device": load(&self.sampler_device),
                "host": load(&self.sampler_host),
                "grammar": load(&self.sampler_grammar),
            },
            "rope_host_uploads": load(&self.rope_host_uploads),
            "conv_ring_stacks": load(&self.conv_ring_stacks),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reports_each_counter() {
        let telemetry = Qwen35Telemetry::default();
        telemetry.record_solo_decode_step();
        telemetry.record_batched_decode_step(3);
        telemetry.record_sample(SamplerPath::Greedy, 2);
        telemetry.record_sample(SamplerPath::Device, 1);
        telemetry.record_sample(SamplerPath::Host, 1);
        telemetry.record_sample(SamplerPath::Grammar, 1);
        telemetry.record_rope_host_upload();
        telemetry.record_conv_ring_stacks(18);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot["decode_tokens"], 4);
        assert_eq!(snapshot["solo_decode_steps"], 1);
        assert_eq!(snapshot["batched_decode_steps"], 1);
        assert_eq!(snapshot["batched_decode_rows"], 3);
        assert_eq!(snapshot["sampling_host_reads"], 5);
        assert_eq!(snapshot["sampler"]["greedy"], 1);
        assert_eq!(snapshot["sampler"]["device"], 1);
        assert_eq!(snapshot["sampler"]["host"], 1);
        assert_eq!(snapshot["sampler"]["grammar"], 1);
        assert_eq!(snapshot["rope_host_uploads"], 1);
        assert_eq!(snapshot["conv_ring_stacks"], 18);
    }
}
