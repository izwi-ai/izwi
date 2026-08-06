//! Focused managed-KV microbenchmark.
//!
//! This example intentionally uses only the public backend arena ABI. It does
//! not load a model, and its timings are synchronized operation latencies, not
//! end-to-end model throughput.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::shape::ShapeWithOneHole;
use candle_core::{DType, Device, Tensor};
use izwi_core::backends::kv::{
    CpuKvArena, KvArena, KvArenaConfig, KvArenaOperationStats, KvLayerConfig, KvPageCopy,
    KvWriteArgs, PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
};
#[cfg(any(feature = "cuda", feature = "metal"))]
use izwi_core::backends::kv::CandleAcceleratorKvArena;
use izwi_core::backends::BackendKind;
use izwi_core::engine::ModelInstanceId;
use izwi_core::kv::{
    CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvGroupId, KvLayerBinding,
    KvSequenceBlockTable, KvSlotRef,
};

const SCHEMA: &str = "izwi.kv-cache-bench.v1";
const GROUP: KvGroupId = KvGroupId::new(1);
const LAYER: KvLayerBinding = KvLayerBinding {
    model_layer: 0,
    physical_layer: 0,
};
const KV_HEADS: usize = 2;
const QUERY_HEADS: usize = 8;
const HEAD_DIM: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Cpu,
    Metal,
    Cuda,
}

impl Backend {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!("unknown backend {value:?}; expected cpu, metal, or cuda")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    fn kind(self) -> BackendKind {
        match self {
            Self::Cpu => BackendKind::Cpu,
            Self::Metal => BackendKind::Metal,
            Self::Cuda => BackendKind::Cuda,
        }
    }
}

#[derive(Debug)]
struct Options {
    backend: Backend,
    page_tokens: u32,
    profile: String,
    iterations: usize,
    warmup: usize,
    long_context: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            page_tokens: 16,
            profile: "ragged".into(),
            iterations: 10,
            warmup: 2,
            long_context: 1024,
        }
    }
}

struct Runtime {
    arena: Arc<dyn KvArena>,
    device: Device,
    dtype: DType,
}

struct Workload {
    prefill_rows: Vec<PagedKvPrefillRow>,
    decode: KvDecodeBatchMetadata,
    query_count: usize,
    used_pages: u32,
    context_summary: String,
}

fn main() -> ExitCode {
    let options = match parse_options() {
        Ok(Some(options)) => options,
        Ok(None) => return ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}\n");
            print_help();
            return ExitCode::from(2);
        }
    };

    match run(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            emit_status(&options, "failed", &message);
            ExitCode::FAILURE
        }
    }
}

fn parse_options() -> Result<Option<Options>, String> {
    let mut options = Options::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            args.next().ok_or_else(|| format!("{flag} requires a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            "--backend" => options.backend = Backend::parse(&value(&mut args, &arg)?)?,
            "--page-tokens" => {
                options.page_tokens = parse_positive(&value(&mut args, &arg)?, &arg)?
            }
            "--profile" => {
                options.profile = value(&mut args, &arg)?;
                if options.profile != "ragged" && options.profile != "long" {
                    return Err("--profile must be ragged or long".into());
                }
            }
            "--iterations" => options.iterations = parse_positive(&value(&mut args, &arg)?, &arg)?,
            "--warmup" => {
                options.warmup = value(&mut args, &arg)?
                    .parse()
                    .map_err(|_| "--warmup must be a non-negative integer".to_string())?
            }
            "--long-context" => {
                options.long_context = parse_positive(&value(&mut args, &arg)?, &arg)?
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    if !matches!(options.page_tokens, 16 | 32) {
        return Err("--page-tokens must be 16 or 32 for the certification matrix".into());
    }
    Ok(Some(options))
}

fn parse_positive<T>(value: &str, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr + PartialOrd + From<u8>,
{
    let parsed = value
        .parse::<T>()
        .map_err(|_| format!("{flag} must be a positive integer"))?;
    if parsed <= T::from(0) {
        return Err(format!("{flag} must be a positive integer"));
    }
    Ok(parsed)
}

fn print_help() {
    println!(
        "Managed-KV synchronized microbenchmark\n\n\
Usage: cargo run -p izwi-core --example kv-cache-bench [FEATURES] -- [OPTIONS]\n\n\
Options:\n  \
  --backend cpu|metal|cuda  Backend to measure (default: cpu)\n  \
  --page-tokens 16|32       Physical KV page size (default: 16)\n  \
  --profile ragged|long     Context shape (default: ragged)\n  \
  --iterations N            Measured iterations (default: 10)\n  \
  --warmup N                Unmeasured iterations (default: 2)\n  \
  --long-context N          Long-profile maximum context (default: 1024)\n  \
  -h, --help                Show this help\n\n\
Output is JSON Lines. Latencies include an explicit arena drain after each\n\
operation, making asynchronous accelerator submissions comparable. They do\n\
not include model loading, scheduling, sampling, or tokenization."
    );
}

fn run(options: &Options) -> Result<(), String> {
    let workload = build_workload(options)?;
    let capacity_pages = workload
        .used_pages
        .checked_add(4)
        .ok_or_else(|| "benchmark page capacity overflow".to_string())?;
    let runtime = match create_runtime(options, capacity_pages)? {
        Some(runtime) => runtime,
        None => return Ok(()),
    };

    seed_arena(&runtime, options, &workload)?;
    let queries = deterministic_tensor(
        workload.query_count * QUERY_HEADS * HEAD_DIM,
        (workload.query_count, QUERY_HEADS, HEAD_DIM),
        &runtime.device,
        runtime.dtype,
    )?;
    let decode_queries = deterministic_tensor(
        workload.decode.sequences.len() * QUERY_HEADS * HEAD_DIM,
        (workload.decode.sequences.len(), QUERY_HEADS, HEAD_DIM),
        &runtime.device,
        runtime.dtype,
    )?;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    benchmark_operation(options, &workload, &runtime, "paged_prefill", || {
        let output = runtime
            .arena
            .paged_prefill(
                LAYER,
                PagedKvPrefillArgs {
                    queries: &queries,
                    rows: &workload.prefill_rows,
                    softmax_scale: scale,
                    softcap: None,
                    window_tokens: None,
                },
            )
            .map_err(|error| error.to_string())?;
        std::hint::black_box(output);
        runtime.arena.drain().map_err(|error| error.to_string())
    })?;
    benchmark_operation(options, &workload, &runtime, "paged_decode", || {
        let output = runtime
            .arena
            .paged_decode(
                LAYER,
                PagedKvDecodeArgs {
                    queries: &decode_queries,
                    batch: &workload.decode,
                    softmax_scale: scale,
                    softcap: None,
                },
            )
            .map_err(|error| error.to_string())?;
        std::hint::black_box(output);
        runtime.arena.drain().map_err(|error| error.to_string())
    })?;

    let mutation_source = block(runtime.arena.id(), 0);
    let mutation_destination = block(runtime.arena.id(), workload.used_pages);
    benchmark_operation(options, &workload, &runtime, "page_copy", || {
        let fence = runtime
            .arena
            .copy_pages(&[KvPageCopy {
                source: mutation_source,
                destination: mutation_destination,
            }])
            .map_err(|error| error.to_string())?;
        fence.wait().map_err(|error| error.to_string())
    })?;
    benchmark_operation(options, &workload, &runtime, "page_zero", || {
        let fence = runtime
            .arena
            .zero_pages(&[mutation_destination])
            .map_err(|error| error.to_string())?;
        fence.wait().map_err(|error| error.to_string())
    })?;

    let write_slots = (0..options.page_tokens)
        .map(|offset| KvSlotRef {
            block: mutation_source,
            offset,
        })
        .collect::<Vec<_>>();
    let lowered = runtime
        .arena
        .lower_slots(&write_slots)
        .map_err(|error| error.to_string())?;
    let write_keys = deterministic_tensor(
        write_slots.len() * KV_HEADS * HEAD_DIM,
        (write_slots.len(), KV_HEADS, HEAD_DIM),
        &runtime.device,
        runtime.dtype,
    )?;
    let write_values = write_keys.clone();
    benchmark_operation(options, &workload, &runtime, "slot_write", || {
        let completion = runtime
            .arena
            .write_slots(
                LAYER,
                KvWriteArgs {
                    keys: &write_keys,
                    values: &write_values,
                    slots: lowered.as_ref(),
                },
            )
            .map_err(|error| error.to_string())?;
        std::hint::black_box(completion);
        runtime.arena.drain().map_err(|error| error.to_string())
    })?;
    Ok(())
}

fn create_runtime(options: &Options, capacity_pages: u32) -> Result<Option<Runtime>, String> {
    let arena_id = KvArenaId {
        model_instance: ModelInstanceId::new(1),
        backend: options.backend.kind(),
        device_ordinal: (options.backend != Backend::Cpu).then_some(0),
        generation: 1,
    };
    let dtype = if options.backend == Backend::Cuda {
        DType::F16
    } else {
        DType::F32
    };
    let config = KvArenaConfig {
        id: arena_id,
        group: GROUP,
        page_tokens: options.page_tokens,
        capacity_pages,
        dtype,
        layers: vec![KvLayerConfig {
            binding: LAYER,
            num_kv_heads: KV_HEADS as u32,
            key_head_dim: HEAD_DIM as u32,
            value_head_dim: HEAD_DIM as u32,
        }],
    };

    match options.backend {
        Backend::Cpu => Ok(Some(Runtime {
            arena: Arc::new(CpuKvArena::new(config).map_err(|error| error.to_string())?),
            device: Device::Cpu,
            dtype,
        })),
        Backend::Metal => create_metal_runtime(options, config),
        Backend::Cuda => create_cuda_runtime(options, config),
    }
}

#[cfg(feature = "metal")]
fn create_metal_runtime(options: &Options, config: KvArenaConfig) -> Result<Option<Runtime>, String> {
    let device = match Device::new_metal(0) {
        Ok(device) => device,
        Err(error) => {
            emit_status(options, "unsupported", &format!("Metal device unavailable: {error}"));
            return Ok(None);
        }
    };
    let arena = CandleAcceleratorKvArena::new_mutation_only(config, device.clone())
        .map_err(|error| error.to_string())?;
    Ok(Some(Runtime { arena: Arc::new(arena), device, dtype: DType::F32 }))
}

#[cfg(not(feature = "metal"))]
fn create_metal_runtime(options: &Options, _config: KvArenaConfig) -> Result<Option<Runtime>, String> {
    emit_status(options, "unsupported", "izwi-core was built without the metal feature");
    Ok(None)
}

#[cfg(feature = "cuda")]
fn create_cuda_runtime(options: &Options, config: KvArenaConfig) -> Result<Option<Runtime>, String> {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(error) => {
            emit_status(options, "unsupported", &format!("CUDA device unavailable: {error}"));
            return Ok(None);
        }
    };
    let arena = CandleAcceleratorKvArena::new_mutation_only(config, device.clone())
        .map_err(|error| error.to_string())?;
    Ok(Some(Runtime { arena: Arc::new(arena), device, dtype: DType::F16 }))
}

#[cfg(not(feature = "cuda"))]
fn create_cuda_runtime(options: &Options, _config: KvArenaConfig) -> Result<Option<Runtime>, String> {
    emit_status(options, "unsupported", "izwi-core was built without the cuda feature");
    Ok(None)
}

fn build_workload(options: &Options) -> Result<Workload, String> {
    let (contexts, query_lens, mut offsets) = if options.profile == "ragged" {
        (
            vec![options.page_tokens - 3, options.page_tokens * 2 + 5, options.page_tokens * 5 - 1],
            vec![4, 8, 16],
            vec![1, options.page_tokens / 4, 0],
        )
    } else {
        let max_context = options.long_context.max(options.page_tokens * 4);
        (vec![max_context / 2 + 3, max_context], vec![16, 32], vec![3, options.page_tokens / 2])
    };

    // Candle FA2's paged path requires zero first-page offsets. Keep page-32
    // CUDA cases eligible while page-16 cases exercise the native fallback.
    if options.backend == Backend::Cuda && options.page_tokens == 32 {
        offsets.fill(0);
    }

    let mut next_page = 0_u32;
    let mut next_query = 0_u32;
    let mut prefill_rows = Vec::with_capacity(contexts.len());
    let mut sequences = Vec::with_capacity(contexts.len());
    for ((context_len, query_len), first_page_offset) in contexts
        .iter()
        .copied()
        .zip(query_lens.iter().copied())
        .zip(offsets.iter().copied())
    {
        if query_len > context_len {
            return Err("query length exceeds context length".into());
        }
        let page_count = (context_len + first_page_offset).div_ceil(options.page_tokens);
        let blocks = (next_page..next_page + page_count)
            .map(|index| CacheBlockRef {
                arena: KvArenaId {
                    model_instance: ModelInstanceId::new(1),
                    backend: options.backend.kind(),
                    device_ordinal: (options.backend != Backend::Cpu).then_some(0),
                    generation: 1,
                },
                group: GROUP,
                index,
                slot_generation: 1,
            })
            .collect::<Vec<_>>();
        prefill_rows.push(PagedKvPrefillRow {
            blocks: blocks.clone(),
            first_page_offset,
            query_start: next_query,
            query_len,
            context_len,
        });
        sequences.push(KvSequenceBlockTable {
            blocks,
            first_page_offset,
            context_len,
        });
        next_page += page_count;
        next_query += query_len;
    }
    Ok(Workload {
        prefill_rows,
        decode: KvDecodeBatchMetadata { sequences },
        query_count: next_query as usize,
        used_pages: next_page,
        context_summary: contexts.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
    })
}

fn seed_arena(runtime: &Runtime, options: &Options, workload: &Workload) -> Result<(), String> {
    let slots = (0..workload.used_pages)
        .flat_map(|index| {
            (0..options.page_tokens).map(move |offset| KvSlotRef {
                block: block(runtime.arena.id(), index),
                offset,
            })
        })
        .collect::<Vec<_>>();
    let lowered = runtime.arena.lower_slots(&slots).map_err(|error| error.to_string())?;
    let keys = deterministic_tensor(
        slots.len() * KV_HEADS * HEAD_DIM,
        (slots.len(), KV_HEADS, HEAD_DIM),
        &runtime.device,
        runtime.dtype,
    )?;
    let values = deterministic_tensor(
        slots.len() * KV_HEADS * HEAD_DIM,
        (slots.len(), KV_HEADS, HEAD_DIM),
        &runtime.device,
        runtime.dtype,
    )?;
    let completion = runtime
        .arena
        .write_slots(LAYER, KvWriteArgs { keys: &keys, values: &values, slots: lowered.as_ref() })
        .map_err(|error| error.to_string())?;
    std::hint::black_box(completion);
    runtime.arena.drain().map_err(|error| error.to_string())
}

fn deterministic_tensor<S: ShapeWithOneHole>(
    len: usize,
    shape: S,
    device: &Device,
    dtype: DType,
) -> Result<Tensor, String> {
    let values = (0..len)
        .map(|index| ((index % 251) as f32 - 125.0) / 64.0)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape, device)
        .and_then(|tensor| tensor.to_dtype(dtype))
        .map_err(|error| error.to_string())
}

fn block(arena: KvArenaId, index: u32) -> CacheBlockRef {
    CacheBlockRef { arena, group: GROUP, index, slot_generation: 1 }
}

fn benchmark_operation(
    options: &Options,
    workload: &Workload,
    runtime: &Runtime,
    operation: &str,
    mut run_once: impl FnMut() -> Result<(), String>,
) -> Result<(), String> {
    for _ in 0..options.warmup {
        run_once()?;
    }
    let before = runtime.arena.operation_stats();
    let mut samples = Vec::with_capacity(options.iterations);
    for _ in 0..options.iterations {
        let start = Instant::now();
        run_once()?;
        samples.push(start.elapsed());
    }
    let dispatches = dispatch_delta(operation, before, runtime.arena.operation_stats());
    if dispatches == 0 && operation != "paged_prefill" {
        return Err(format!(
            "{operation} completed without advancing its arena dispatch counter"
        ));
    }
    emit_measurement(options, workload, operation, &samples, dispatches);
    Ok(())
}

fn dispatch_delta(operation: &str, before: KvArenaOperationStats, after: KvArenaOperationStats) -> u64 {
    match operation {
        "slot_write" => after.slot_write_dispatches - before.slot_write_dispatches,
        "paged_prefill" | "paged_decode" => after.paged_decode_dispatches - before.paged_decode_dispatches,
        "page_zero" => after.page_zero_dispatches - before.page_zero_dispatches,
        "page_copy" => after.page_copy_dispatches - before.page_copy_dispatches,
        _ => 0,
    }
}

fn emit_measurement(
    options: &Options,
    workload: &Workload,
    operation: &str,
    samples: &[Duration],
    dispatches: u64,
) {
    let mut micros = samples.iter().map(|sample| sample.as_secs_f64() * 1_000_000.0).collect::<Vec<_>>();
    micros.sort_by(f64::total_cmp);
    let mean = micros.iter().sum::<f64>() / micros.len() as f64;
    let percentile = |fraction: f64| micros[((micros.len() - 1) as f64 * fraction).ceil() as usize];
    println!(
        "{{\"schema\":\"{SCHEMA}\",\"status\":\"measured\",\"backend\":\"{}\",\"page_tokens\":{},\"profile\":\"{}\",\"contexts\":\"{}\",\"kernel_path\":\"{}\",\"operation\":\"{operation}\",\"iterations\":{},\"warmup\":{},\"dispatches\":{dispatches},\"dispatches_per_iteration\":{:.3},\"mean_us\":{mean:.3},\"p50_us\":{:.3},\"p95_us\":{:.3}}}",
        options.backend.name(), options.page_tokens, options.profile, workload.context_summary,
        kernel_path(options), options.iterations, options.warmup,
        dispatches as f64 / options.iterations as f64, percentile(0.50), percentile(0.95)
    );
}

fn kernel_path(options: &Options) -> &'static str {
    match options.backend {
        Backend::Cpu => "cpu_portable",
        Backend::Metal => "metal_native",
        Backend::Cuda if options.page_tokens == 32 => {
            #[cfg(feature = "flash-attn")]
            {
                "candle_fa2"
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                "cuda_native"
            }
        }
        Backend::Cuda => "cuda_native",
    }
}

fn emit_status(options: &Options, status: &str, reason: &str) {
    let reason = reason.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ");
    println!(
        "{{\"schema\":\"{SCHEMA}\",\"status\":\"{status}\",\"backend\":\"{}\",\"page_tokens\":{},\"profile\":\"{}\",\"reason\":\"{reason}\"}}",
        options.backend.name(), options.page_tokens, options.profile
    );
}
