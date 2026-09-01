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
#[cfg(any(feature = "cuda", feature = "metal"))]
use izwi_core::backends::kv::CandleAcceleratorKvArena;
use izwi_core::backends::kv::{
    CpuKvArena, KvArena, KvArenaConfig, KvArenaOperationStats, KvLayerConfig, KvPageCopy,
    KvWriteArgs, PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
};
use izwi_core::backends::BackendKind;
use izwi_core::engine::ModelInstanceId;
use izwi_core::kv::{
    CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvGroupId, KvLayerBinding,
    KvSequenceBlockTable, KvSlotRef,
};

const SCHEMA: &str = "izwi.kv-cache-bench.v2";
const GROUP: KvGroupId = KvGroupId::new(1);
const LAYER: KvLayerBinding = KvLayerBinding {
    model_layer: 0,
    physical_layer: 0,
};
const DEFAULT_KV_HEADS: usize = 2;
const DEFAULT_QUERY_HEADS: usize = 8;
const DEFAULT_HEAD_DIM: usize = 32;

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
            _ => Err(format!(
                "unknown backend {value:?}; expected cpu, metal, or cuda"
            )),
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
    dtype: DType,
    provider: String,
    first_page_offset: Option<u32>,
    window_tokens: Option<u32>,
    softcap: Option<f32>,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    batch_size: usize,
    context_len: Option<u32>,
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
            dtype: DType::F32,
            provider: "any".into(),
            first_page_offset: None,
            window_tokens: None,
            softcap: None,
            query_heads: DEFAULT_QUERY_HEADS,
            kv_heads: DEFAULT_KV_HEADS,
            head_dim: DEFAULT_HEAD_DIM,
            batch_size: 0,
            context_len: None,
        }
    }
}

type LayerReader = dyn Fn(KvLayerBinding) -> Result<(Tensor, Tensor), String>;

struct Runtime {
    arena: Arc<dyn KvArena>,
    device: Device,
    dtype: DType,
    read_layer: Box<LayerReader>,
}

struct Workload {
    prefill_rows: Vec<PagedKvPrefillRow>,
    decode: KvDecodeBatchMetadata,
    query_count: usize,
    used_pages: u32,
    context_summary: String,
}

#[derive(Clone, Copy)]
struct ErrorMetrics {
    max_abs: f32,
    max_rel: f32,
    tolerance: f32,
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
            args.next()
                .ok_or_else(|| format!("{flag} requires a value"))
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
            "--dtype" => {
                options.dtype = match value(&mut args, &arg)?.as_str() {
                    "f32" => DType::F32,
                    "f16" => DType::F16,
                    "bf16" => DType::BF16,
                    _ => return Err("--dtype must be f32, f16, or bf16".into()),
                }
            }
            "--provider" => {
                options.provider = value(&mut args, &arg)?;
                if !matches!(options.provider.as_str(), "any" | "portable" | "optimized") {
                    return Err("--provider must be any, portable, or optimized".into());
                }
            }
            "--first-page-offset" => {
                options.first_page_offset = Some(value(&mut args, &arg)?.parse().map_err(|_| {
                    "--first-page-offset must be a non-negative integer".to_string()
                })?)
            }
            "--window-tokens" => {
                options.window_tokens = Some(parse_positive(&value(&mut args, &arg)?, &arg)?)
            }
            "--softcap" => {
                let softcap = value(&mut args, &arg)?
                    .parse::<f32>()
                    .map_err(|_| "--softcap must be a positive number".to_string())?;
                if !softcap.is_finite() || softcap <= 0.0 {
                    return Err("--softcap must be a finite positive number".into());
                }
                options.softcap = Some(softcap);
            }
            "--query-heads" => {
                options.query_heads = parse_positive(&value(&mut args, &arg)?, &arg)?
            }
            "--kv-heads" => options.kv_heads = parse_positive(&value(&mut args, &arg)?, &arg)?,
            "--head-dim" => options.head_dim = parse_positive(&value(&mut args, &arg)?, &arg)?,
            "--batch-size" => options.batch_size = parse_positive(&value(&mut args, &arg)?, &arg)?,
            "--context-len" => {
                options.context_len = Some(parse_positive(&value(&mut args, &arg)?, &arg)?)
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    if !matches!(options.page_tokens, 16 | 32 | 64) {
        return Err("--page-tokens must be 16, 32, or 64 for the certification matrix".into());
    }
    if options.query_heads % options.kv_heads != 0 {
        return Err("--query-heads must be divisible by --kv-heads".into());
    }
    if options
        .first_page_offset
        .is_some_and(|offset| offset >= options.page_tokens)
    {
        return Err("--first-page-offset must be smaller than --page-tokens".into());
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
  --page-tokens 16|32|64    Physical KV page size (default: 16)\n  \
  --profile ragged|long     Context shape (default: ragged)\n  \
  --dtype f32|f16|bf16      KV/query storage dtype (default: f32)\n  \
  --provider any|portable|optimized\n                             Provider promotion axis (default: any)\n  \
  --first-page-offset N     Override every row's page offset\n  \
  --window-tokens N         Sliding attention window\n  \
  --softcap N               Logit soft cap\n  \
  --query-heads N           Query head count (default: 8)\n  \
  --kv-heads N              KV head count (default: 2)\n  \
  --head-dim N              Head dimension (default: 32)\n  \
  --batch-size N            Override profile batch size\n  \
  --context-len N           Override every context length\n  \
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
    match options.provider.as_str() {
        "portable" => env::set_var("IZWI_KV_DISABLE_OPTIMIZED_PROVIDER", "1"),
        "optimized" => env::set_var("IZWI_KV_DISABLE_OPTIMIZED_PROVIDER", "0"),
        _ => {}
    }
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
        workload.query_count * options.query_heads * options.head_dim,
        (workload.query_count, options.query_heads, options.head_dim),
        &runtime.device,
        runtime.dtype,
    )?;
    let decode_queries = deterministic_tensor(
        workload.decode.sequences.len() * options.query_heads * options.head_dim,
        (
            workload.decode.sequences.len(),
            options.query_heads,
            options.head_dim,
        ),
        &runtime.device,
        runtime.dtype,
    )?;
    let scale = 1.0 / (options.head_dim as f32).sqrt();
    let (prefill_error, decode_error) = certify_attention(
        options,
        &workload,
        &runtime,
        &queries,
        &decode_queries,
        scale,
    )?;

    benchmark_operation(
        options,
        &workload,
        &runtime,
        "paged_prefill",
        prefill_error,
        || {
            let output = runtime
                .arena
                .paged_prefill(
                    LAYER,
                    PagedKvPrefillArgs {
                        queries: &queries,
                        rows: &workload.prefill_rows,
                        softmax_scale: scale,
                        softcap: options.softcap,
                        window_tokens: options.window_tokens,
                    },
                )
                .map_err(|error| error.to_string())?;
            std::hint::black_box(output);
            runtime.arena.drain().map_err(|error| error.to_string())
        },
    )?;
    benchmark_operation(
        options,
        &workload,
        &runtime,
        "paged_decode",
        decode_error,
        || {
            let output = runtime
                .arena
                .paged_decode(
                    LAYER,
                    PagedKvDecodeArgs {
                        queries: &decode_queries,
                        batch: &workload.decode,
                        softmax_scale: scale,
                        softcap: options.softcap,
                    },
                )
                .map_err(|error| error.to_string())?;
            std::hint::black_box(output);
            runtime.arena.drain().map_err(|error| error.to_string())
        },
    )?;

    certify_mutations(options, &runtime, workload.used_pages)?;

    let mutation_source = block(runtime.arena.id(), 0);
    let mutation_destination = block(runtime.arena.id(), workload.used_pages);
    benchmark_operation(
        options,
        &workload,
        &runtime,
        "page_copy",
        exact_error(),
        || {
            let fence = runtime
                .arena
                .copy_pages(&[KvPageCopy {
                    source: mutation_source,
                    destination: mutation_destination,
                }])
                .map_err(|error| error.to_string())?;
            fence.wait().map_err(|error| error.to_string())
        },
    )?;
    benchmark_operation(
        options,
        &workload,
        &runtime,
        "page_zero",
        exact_error(),
        || {
            let fence = runtime
                .arena
                .zero_pages(&[mutation_destination])
                .map_err(|error| error.to_string())?;
            fence.wait().map_err(|error| error.to_string())
        },
    )?;

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
        write_slots.len() * options.kv_heads * options.head_dim,
        (write_slots.len(), options.kv_heads, options.head_dim),
        &runtime.device,
        runtime.dtype,
    )?;
    let write_values = write_keys.clone();
    benchmark_operation(
        options,
        &workload,
        &runtime,
        "slot_write",
        exact_error(),
        || {
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
        },
    )?;
    Ok(())
}

fn create_runtime(options: &Options, capacity_pages: u32) -> Result<Option<Runtime>, String> {
    let arena_id = KvArenaId {
        model_instance: ModelInstanceId::new(1),
        backend: options.backend.kind(),
        device_ordinal: (options.backend != Backend::Cpu).then_some(0),
        generation: 1,
    };
    let dtype = options.dtype;
    let config = KvArenaConfig {
        id: arena_id,
        group: GROUP,
        page_tokens: options.page_tokens,
        capacity_pages,
        growth: None,
        dtype,
        layers: vec![KvLayerConfig {
            binding: LAYER,
            num_kv_heads: options.kv_heads as u32,
            key_head_dim: options.head_dim as u32,
            value_head_dim: options.head_dim as u32,
        }],
    };

    match options.backend {
        Backend::Cpu => {
            let arena = Arc::new(CpuKvArena::new(config).map_err(|error| error.to_string())?);
            let reader = arena.clone();
            Ok(Some(Runtime {
                arena,
                device: Device::Cpu,
                dtype,
                read_layer: Box::new(move |binding| {
                    reader
                        .layer_tensors(binding)
                        .map_err(|error| error.to_string())
                }),
            }))
        }
        Backend::Metal => create_metal_runtime(options, config),
        Backend::Cuda => create_cuda_runtime(options, config),
    }
}

#[cfg(feature = "metal")]
fn create_metal_runtime(
    options: &Options,
    config: KvArenaConfig,
) -> Result<Option<Runtime>, String> {
    let Some(device) = izwi_core::backends::metal_device_if_available(0) else {
        emit_status(
            options,
            "unsupported",
            "Metal requires macOS 15 or later and an available Metal device",
        );
        return Ok(None);
    };
    let arena = Arc::new(
        CandleAcceleratorKvArena::new_mutation_only(config, device.clone())
            .map_err(|error| error.to_string())?,
    );
    let reader = arena.clone();
    Ok(Some(Runtime {
        arena,
        device,
        dtype: options.dtype,
        read_layer: Box::new(move |binding| {
            reader
                .layer_tensors(binding)
                .map_err(|error| error.to_string())
        }),
    }))
}

#[cfg(not(feature = "metal"))]
fn create_metal_runtime(
    options: &Options,
    _config: KvArenaConfig,
) -> Result<Option<Runtime>, String> {
    emit_status(
        options,
        "unsupported",
        "izwi-core was built without the metal feature",
    );
    Ok(None)
}

#[cfg(feature = "cuda")]
fn create_cuda_runtime(
    options: &Options,
    config: KvArenaConfig,
) -> Result<Option<Runtime>, String> {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(error) => {
            emit_status(
                options,
                "unsupported",
                &format!("CUDA device unavailable: {error}"),
            );
            return Ok(None);
        }
    };
    let arena = Arc::new(
        CandleAcceleratorKvArena::new_mutation_only(config, device.clone())
            .map_err(|error| error.to_string())?,
    );
    let reader = arena.clone();
    Ok(Some(Runtime {
        arena,
        device,
        dtype: options.dtype,
        read_layer: Box::new(move |binding| {
            reader
                .layer_tensors(binding)
                .map_err(|error| error.to_string())
        }),
    }))
}

#[cfg(not(feature = "cuda"))]
fn create_cuda_runtime(
    options: &Options,
    _config: KvArenaConfig,
) -> Result<Option<Runtime>, String> {
    emit_status(
        options,
        "unsupported",
        "izwi-core was built without the cuda feature",
    );
    Ok(None)
}

fn build_workload(options: &Options) -> Result<Workload, String> {
    let (mut contexts, mut query_lens, mut offsets) = if options.profile == "ragged" {
        (
            vec![
                options.page_tokens - 3,
                options.page_tokens * 2 + 5,
                options.page_tokens * 5 - 1,
            ],
            vec![4, 8, 16],
            vec![1, options.page_tokens / 4, 0],
        )
    } else {
        let max_context = options.long_context.max(options.page_tokens * 4);
        (
            vec![max_context / 2 + 3, max_context],
            vec![16, 32],
            vec![3, options.page_tokens / 2],
        )
    };
    if let Some(context_len) = options.context_len {
        contexts.fill(context_len);
        for query_len in &mut query_lens {
            *query_len = (*query_len).min(context_len);
        }
    }
    if options.batch_size > 0 {
        let base_contexts = contexts.clone();
        let base_queries = query_lens.clone();
        let base_offsets = offsets.clone();
        contexts = (0..options.batch_size)
            .map(|index| base_contexts[index % base_contexts.len()])
            .collect();
        query_lens = (0..options.batch_size)
            .map(|index| base_queries[index % base_queries.len()])
            .collect();
        offsets = (0..options.batch_size)
            .map(|index| base_offsets[index % base_offsets.len()])
            .collect();
    }
    if let Some(first_page_offset) = options.first_page_offset {
        offsets.fill(first_page_offset);
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
        context_summary: contexts
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
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
    let lowered = runtime
        .arena
        .lower_slots(&slots)
        .map_err(|error| error.to_string())?;
    let keys = deterministic_tensor(
        slots.len() * options.kv_heads * options.head_dim,
        (slots.len(), options.kv_heads, options.head_dim),
        &runtime.device,
        runtime.dtype,
    )?;
    let values = deterministic_tensor(
        slots.len() * options.kv_heads * options.head_dim,
        (slots.len(), options.kv_heads, options.head_dim),
        &runtime.device,
        runtime.dtype,
    )?;
    let completion = runtime
        .arena
        .write_slots(
            LAYER,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots: lowered.as_ref(),
            },
        )
        .map_err(|error| error.to_string())?;
    std::hint::black_box(completion);
    runtime.arena.drain().map_err(|error| error.to_string())
}

fn certify_attention(
    options: &Options,
    workload: &Workload,
    runtime: &Runtime,
    queries: &Tensor,
    decode_queries: &Tensor,
    scale: f32,
) -> Result<(ErrorMetrics, ErrorMetrics), String> {
    let reference_id = KvArenaId {
        model_instance: ModelInstanceId::new(2),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    };
    let config = KvArenaConfig {
        id: reference_id,
        group: GROUP,
        page_tokens: options.page_tokens,
        capacity_pages: workload.used_pages + 4,
        growth: None,
        dtype: options.dtype,
        layers: vec![KvLayerConfig {
            binding: LAYER,
            num_kv_heads: options.kv_heads as u32,
            key_head_dim: options.head_dim as u32,
            value_head_dim: options.head_dim as u32,
        }],
    };
    let reference_arena = Arc::new(CpuKvArena::new(config).map_err(|error| error.to_string())?);
    let reference = Runtime {
        arena: reference_arena.clone(),
        device: Device::Cpu,
        dtype: options.dtype,
        read_layer: Box::new(move |binding| {
            reference_arena
                .layer_tensors(binding)
                .map_err(|error| error.to_string())
        }),
    };
    let reference_workload = remap_workload(workload, reference_id);
    seed_arena(&reference, options, &reference_workload)?;
    let reference_queries = queries
        .to_device(&Device::Cpu)
        .map_err(|error| error.to_string())?;
    let reference_decode_queries = decode_queries
        .to_device(&Device::Cpu)
        .map_err(|error| error.to_string())?;

    let actual_prefill = runtime
        .arena
        .paged_prefill(
            LAYER,
            PagedKvPrefillArgs {
                queries,
                rows: &workload.prefill_rows,
                softmax_scale: scale,
                softcap: options.softcap,
                window_tokens: options.window_tokens,
            },
        )
        .map_err(|error| error.to_string())?;
    let expected_prefill = reference
        .arena
        .paged_prefill(
            LAYER,
            PagedKvPrefillArgs {
                queries: &reference_queries,
                rows: &reference_workload.prefill_rows,
                softmax_scale: scale,
                softcap: options.softcap,
                window_tokens: options.window_tokens,
            },
        )
        .map_err(|error| error.to_string())?;
    let prefill = compare_tensors(&actual_prefill, &expected_prefill, options.dtype)?;

    let actual_decode = runtime
        .arena
        .paged_decode(
            LAYER,
            PagedKvDecodeArgs {
                queries: decode_queries,
                batch: &workload.decode,
                softmax_scale: scale,
                softcap: options.softcap,
            },
        )
        .map_err(|error| error.to_string())?;
    let expected_decode = reference
        .arena
        .paged_decode(
            LAYER,
            PagedKvDecodeArgs {
                queries: &reference_decode_queries,
                batch: &reference_workload.decode,
                softmax_scale: scale,
                softcap: options.softcap,
            },
        )
        .map_err(|error| error.to_string())?;
    let decode = compare_tensors(&actual_decode, &expected_decode, options.dtype)?;
    runtime.arena.drain().map_err(|error| error.to_string())?;
    Ok((prefill, decode))
}

fn remap_workload(workload: &Workload, arena: KvArenaId) -> Workload {
    let remap_block = |block: CacheBlockRef| CacheBlockRef { arena, ..block };
    Workload {
        prefill_rows: workload
            .prefill_rows
            .iter()
            .map(|row| PagedKvPrefillRow {
                blocks: row.blocks.iter().copied().map(remap_block).collect(),
                first_page_offset: row.first_page_offset,
                query_start: row.query_start,
                query_len: row.query_len,
                context_len: row.context_len,
            })
            .collect(),
        decode: KvDecodeBatchMetadata {
            sequences: workload
                .decode
                .sequences
                .iter()
                .map(|sequence| KvSequenceBlockTable {
                    blocks: sequence.blocks.iter().copied().map(remap_block).collect(),
                    first_page_offset: sequence.first_page_offset,
                    context_len: sequence.context_len,
                })
                .collect(),
        },
        query_count: workload.query_count,
        used_pages: workload.used_pages,
        context_summary: workload.context_summary.clone(),
    }
}

fn compare_tensors(
    actual: &Tensor,
    expected: &Tensor,
    dtype: DType,
) -> Result<ErrorMetrics, String> {
    let actual = actual
        .to_dtype(DType::F32)
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| error.to_string())?;
    let expected = expected
        .to_dtype(DType::F32)
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| error.to_string())?;
    if actual.len() != expected.len() {
        return Err(format!(
            "numerical output length mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (actual, expected) in actual.iter().copied().zip(expected.iter().copied()) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("numerical certification produced a non-finite value".into());
        }
        let abs = (actual - expected).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / expected.abs().max(1e-6));
    }
    let tolerance = match dtype {
        DType::F32 => 1e-4,
        DType::F16 => 1e-2,
        DType::BF16 => 3e-2,
        _ => return Err(format!("unsupported certification dtype {dtype:?}")),
    };
    if max_abs > tolerance && max_rel > tolerance {
        return Err(format!(
            "numerical certification exceeded {dtype:?} tolerance {tolerance}: max_abs={max_abs}, max_rel={max_rel}"
        ));
    }
    Ok(ErrorMetrics {
        max_abs,
        max_rel,
        tolerance,
    })
}

fn exact_error() -> ErrorMetrics {
    ErrorMetrics {
        max_abs: 0.0,
        max_rel: 0.0,
        tolerance: 0.0,
    }
}

fn certify_mutations(
    options: &Options,
    runtime: &Runtime,
    first_free_page: u32,
) -> Result<(), String> {
    let page0 = block(runtime.arena.id(), first_free_page);
    let page1 = block(runtime.arena.id(), first_free_page + 1);
    let page2 = block(runtime.arena.id(), first_free_page + 2);
    let page_values = options.page_tokens as usize * options.kv_heads * options.head_dim;
    write_certification_page(runtime, options, page0, 0.25)?;
    write_certification_page(runtime, options, page1, -0.5)?;
    runtime
        .arena
        .copy_pages(&[
            KvPageCopy {
                source: page0,
                destination: page1,
            },
            KvPageCopy {
                source: page1,
                destination: page0,
            },
        ])
        .map_err(|error| error.to_string())?
        .wait()
        .map_err(|error| error.to_string())?;
    let (keys, values) = (runtime.read_layer)(LAYER)?;
    let keys = tensor_values(&keys)?;
    let values = tensor_values(&values)?;
    assert_page_value(
        &keys,
        first_free_page as usize,
        page_values,
        -0.5,
        "copy-cycle keys",
    )?;
    assert_page_value(
        &values,
        first_free_page as usize,
        page_values,
        -0.5,
        "copy-cycle values",
    )?;
    assert_page_value(
        &keys,
        first_free_page as usize + 1,
        page_values,
        0.25,
        "copy-cycle keys",
    )?;
    assert_page_value(
        &values,
        first_free_page as usize + 1,
        page_values,
        0.25,
        "copy-cycle values",
    )?;

    runtime
        .arena
        .zero_pages(&[page2])
        .map_err(|error| error.to_string())?
        .wait()
        .map_err(|error| error.to_string())?;
    let (keys, values) = (runtime.read_layer)(LAYER)?;
    assert_page_value(
        &tensor_values(&keys)?,
        first_free_page as usize + 2,
        page_values,
        0.0,
        "zero keys",
    )?;
    assert_page_value(
        &tensor_values(&values)?,
        first_free_page as usize + 2,
        page_values,
        0.0,
        "zero values",
    )?;
    Ok(())
}

fn write_certification_page(
    runtime: &Runtime,
    options: &Options,
    page: CacheBlockRef,
    value: f32,
) -> Result<(), String> {
    let slots = (0..options.page_tokens)
        .map(|offset| KvSlotRef {
            block: page,
            offset,
        })
        .collect::<Vec<_>>();
    let lowered = runtime
        .arena
        .lower_slots(&slots)
        .map_err(|error| error.to_string())?;
    let tensor = Tensor::full(
        value,
        (slots.len(), options.kv_heads, options.head_dim),
        &runtime.device,
    )
    .and_then(|tensor| tensor.to_dtype(runtime.dtype))
    .map_err(|error| error.to_string())?;
    let completion = runtime
        .arena
        .write_slots(
            LAYER,
            KvWriteArgs {
                keys: &tensor,
                values: &tensor,
                slots: lowered.as_ref(),
            },
        )
        .map_err(|error| error.to_string())?;
    std::hint::black_box(completion);
    runtime.arena.drain().map_err(|error| error.to_string())
}

fn tensor_values(tensor: &Tensor) -> Result<Vec<f32>, String> {
    tensor
        .to_dtype(DType::F32)
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| error.to_string())
}

fn assert_page_value(
    values: &[f32],
    page: usize,
    page_values: usize,
    expected: f32,
    label: &str,
) -> Result<(), String> {
    let start = page
        .checked_mul(page_values)
        .ok_or_else(|| "page readback overflow".to_string())?;
    let end = start
        .checked_add(page_values)
        .ok_or_else(|| "page readback overflow".to_string())?;
    let page = values
        .get(start..end)
        .ok_or_else(|| format!("{label} page is out of bounds"))?;
    if let Some((index, actual)) = page
        .iter()
        .copied()
        .enumerate()
        .find(|(_, actual)| (*actual - expected).abs() > 1e-6)
    {
        return Err(format!(
            "{label} mismatch at element {index}: expected {expected}, got {actual}"
        ));
    }
    Ok(())
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
    CacheBlockRef {
        arena,
        group: GROUP,
        index,
        slot_generation: 1,
    }
}

fn benchmark_operation(
    options: &Options,
    workload: &Workload,
    runtime: &Runtime,
    operation: &str,
    correctness: ErrorMetrics,
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
    let after = runtime.arena.operation_stats();
    let dispatches = dispatch_delta(operation, before, after);
    if dispatches == 0 {
        return Err(format!(
            "{operation} completed without advancing its arena dispatch counter"
        ));
    }
    if options.provider == "portable"
        && matches!(
            after
                .last_attention_provider
                .map(|provider| provider.name()),
            Some("cuda_flash_attention")
        )
    {
        return Err("portable provider request executed the optimized CUDA provider".into());
    }
    if options.provider == "optimized"
        && matches!(operation, "paged_prefill" | "paged_decode")
        && after
            .last_attention_provider
            .map(|provider| provider.name())
            != Some("cuda_flash_attention")
    {
        return Err(format!(
            "optimized provider request executed {:?}",
            after
                .last_attention_provider
                .map(|provider| provider.name())
        ));
    }
    emit_measurement(
        options,
        workload,
        operation,
        &samples,
        before,
        after,
        correctness,
    );
    Ok(())
}

fn dispatch_delta(
    operation: &str,
    before: KvArenaOperationStats,
    after: KvArenaOperationStats,
) -> u64 {
    match operation {
        "slot_write" => after.slot_write_dispatches - before.slot_write_dispatches,
        "paged_prefill" => after.paged_prefill_dispatches - before.paged_prefill_dispatches,
        "paged_decode" => after.paged_decode_dispatches - before.paged_decode_dispatches,
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
    before: KvArenaOperationStats,
    after: KvArenaOperationStats,
    correctness: ErrorMetrics,
) {
    let mut micros = samples
        .iter()
        .map(|sample| sample.as_secs_f64() * 1_000_000.0)
        .collect::<Vec<_>>();
    micros.sort_by(f64::total_cmp);
    let mean = micros.iter().sum::<f64>() / micros.len() as f64;
    let percentile = |fraction: f64| micros[((micros.len() - 1) as f64 * fraction).ceil() as usize];
    let dispatches = dispatch_delta(operation, before, after);
    let provider = after
        .last_attention_provider
        .map(|provider| format!("\"{}\"", provider.name()))
        .unwrap_or_else(|| "null".into());
    let backing_allocations = optional_u64(after.backing_allocations);
    let workspace_bytes = optional_u64(after.workspace_bytes);
    let workspace_allocations = optional_u64(after.workspace_allocations);
    println!(
        "{{\"schema\":\"{SCHEMA}\",\"status\":\"measured\",\"backend\":\"{}\",\"dtype\":\"{}\",\"page_tokens\":{},\"profile\":\"{}\",\"contexts\":\"{}\",\"requested_context_len\":{},\"requested_provider\":\"{}\",\"observed_provider\":{},\"first_page_offset\":{},\"window_tokens\":{},\"softcap\":{},\"query_heads\":{},\"kv_heads\":{},\"head_dim\":{},\"batch_size\":{},\"operation\":\"{operation}\",\"iterations\":{},\"warmup\":{},\"dispatches\":{dispatches},\"dispatches_per_iteration\":{:.3},\"plan_cache_hits\":{},\"plan_cache_misses\":{},\"plan_cache_evictions\":{},\"device_uploads\":{},\"plan_resident_bytes\":{},\"backing_allocations\":{},\"host_synchronizations\":{},\"workspace_bytes\":{},\"workspace_allocations\":{},\"rss_bytes\":null,\"vram_bytes\":null,\"max_abs_error\":{:.8},\"max_rel_error\":{:.8},\"tolerance\":{:.8},\"mean_us\":{mean:.3},\"p50_us\":{:.3},\"p95_us\":{:.3}}}",
        options.backend.name(), dtype_name(options.dtype), options.page_tokens, options.profile,
        workload.context_summary, optional_u32(options.context_len), options.provider, provider,
        optional_u32(options.first_page_offset), optional_u32(options.window_tokens), optional_f32(options.softcap),
        options.query_heads, options.kv_heads, options.head_dim, workload.decode.sequences.len(),
        options.iterations, options.warmup, dispatches as f64 / options.iterations as f64,
        after.attention_plan_cache_hits - before.attention_plan_cache_hits,
        after.attention_plan_cache_misses - before.attention_plan_cache_misses,
        after.attention_plan_cache_evictions - before.attention_plan_cache_evictions,
        after.attention_plan_device_uploads - before.attention_plan_device_uploads,
        after.attention_plan_resident_bytes, backing_allocations,
        after.host_synchronizations - before.host_synchronizations, workspace_bytes,
        workspace_allocations,
        correctness.max_abs, correctness.max_rel, correctness.tolerance,
        percentile(0.50), percentile(0.95)
    );
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "unsupported",
    }
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

fn optional_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

fn optional_f32(value: Option<f32>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

fn emit_status(options: &Options, status: &str, reason: &str) {
    let reason = reason
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ");
    println!(
        "{{\"schema\":\"{SCHEMA}\",\"status\":\"{status}\",\"backend\":\"{}\",\"dtype\":\"{}\",\"page_tokens\":{},\"profile\":\"{}\",\"requested_provider\":\"{}\",\"reason\":\"{reason}\"}}",
        options.backend.name(), dtype_name(options.dtype), options.page_tokens, options.profile,
        options.provider
    );
}
