//! Metal load footprint harness: loads one model through the lifecycle's
//! registry dispatch, without load admission, and reports what the load
//! actually allocated beside what admission priced for it.
//!
//! One model per process: Candle's Metal pool and lazily created device state
//! outlive a model, so a second load in the same process measures less than a
//! cold one. `scripts/measure-metal-load-footprints.sh` runs the variants one
//! process each and collects the JSON lines.
//!
//! ```text
//! IZWI_MEASURE_MODEL=Qwen3.5-4B cargo test -p izwi-core --release --features metal \
//!     --lib measure_metal_load_footprint -- --ignored --nocapture
//! ```
//!
//! `IZWI_MODELS_DIR` overrides the models directory; `IZWI_MEASURE_OUT`
//! appends the JSON line to a file; `IZWI_MEASURE_DIAG_DIR` writes
//! `footprint`/`vmmap` reports from before the load and after it settles.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::admission::{admission_resource_plan, load_authorization_with_request_floor};
use super::*;
use crate::backends::DeviceSelector;
use crate::config::{EngineConfig, GraniteSpeechOptions};
use crate::models::registry::ModelRegistry;
use crate::models::shared::memory::metal::MetalPoolManager;
use crate::runtime::lifecycle::instantiate::load_into_registry;

/// Metal admission compares the load authorization against one request slot:
/// accelerator backends run one physical batch at a time.
const METAL_REQUEST_SLOTS: usize = 1;

/// The process's physical footprint: Metal buffers and host memory alike.
fn phys_footprint_bytes() -> u64 {
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
    }
    // rusage_info_v0: a 16-byte UUID, then u64 fields; ri_phys_footprint is
    // index 9 counting the UUID as two words.
    let mut info = [0_u64; 32];
    // SAFETY: rusage_info_v0 is 96 bytes; the buffer outlives the call.
    let status = unsafe { proc_pid_rusage(std::process::id() as i32, 0, info.as_mut_ptr().cast()) };
    assert_eq!(status, 0, "proc_pid_rusage failed");
    info[9]
}

/// Ask every malloc zone to return its free pages to the system; the bytes
/// released.
fn malloc_pressure_relief_bytes() -> u64 {
    extern "C" {
        fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
    }
    // SAFETY: a null zone relieves every zone; a zero goal releases all it can.
    unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) as u64 }
}

fn known_unified(amount: ResourceVector) -> u64 {
    match load_domain_bytes(amount, BackendKind::Metal) {
        ResourceAmount::Known(bytes) => bytes,
        ResourceAmount::Unknown => 0,
    }
}

/// High-water marks of two counters, sampled every 200 µs on a side thread
/// until dropped.
struct HighWater {
    marks: [Arc<AtomicU64>; 2],
    done: Arc<AtomicBool>,
    sampler: Option<std::thread::JoinHandle<()>>,
}

impl HighWater {
    fn start(sample: impl Fn() -> [u64; 2] + Send + 'static) -> Self {
        let first = sample();
        let marks = first.map(|value| Arc::new(AtomicU64::new(value)));
        let done = Arc::new(AtomicBool::new(false));
        let sampler = {
            let (marks, done) = (marks.clone(), done.clone());
            std::thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    for (mark, value) in marks.iter().zip(sample()) {
                        mark.fetch_max(value, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_micros(200));
                }
            })
        };
        Self {
            marks,
            done,
            sampler: Some(sampler),
        }
    }

    fn stop(mut self) -> [u64; 2] {
        self.done.store(true, Ordering::Relaxed);
        if let Some(sampler) = self.sampler.take() {
            sampler.join().expect("footprint sampler panicked");
        }
        [0, 1].map(|index| self.marks[index].load(Ordering::Relaxed))
    }
}

/// With `IZWI_MEASURE_DIAG_DIR` set, write this process's `footprint` and
/// `vmmap --summary` reports to `<dir>/<stem>.<stage>.{footprint,vmmap}.txt`,
/// so the memory a load leaves behind can be attributed by region type. Run
/// under `MallocStackLogging=1` and the settled stage also writes the live
/// heap by allocation stack (`malloc_history -allBySize`).
fn write_memory_reports(stem: &str, stage: &str) {
    let Ok(dir) = std::env::var("IZWI_MEASURE_DIAG_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("create IZWI_MEASURE_DIAG_DIR");
    let pid = std::process::id().to_string();
    let mut reports: Vec<(&str, Vec<&str>, &str)> = vec![
        (
            "/usr/bin/footprint",
            vec!["-w", "-f", "bytes", "--pid", &pid],
            "footprint",
        ),
        ("/usr/bin/vmmap", vec!["--summary", &pid], "vmmap"),
    ];
    if stage == "settled" && std::env::var_os("MallocStackLogging").is_some() {
        reports.push((
            "/usr/bin/malloc_history",
            vec![&pid, "-allBySize", "-fullStacks"],
            "malloc_history",
        ));
    }
    for (tool, args, suffix) in reports {
        let output = std::process::Command::new(tool)
            .args(args)
            .output()
            .unwrap_or_else(|error| panic!("run {tool}: {error}"));
        let mut report = output.stdout;
        report.extend_from_slice(&output.stderr);
        std::fs::write(dir.join(format!("{stem}.{stage}.{suffix}.txt")), report)
            .expect("write memory report");
    }
}

#[test]
#[ignore = "loads a real checkpoint on Metal; set IZWI_MEASURE_MODEL (see module docs)"]
fn measure_metal_load_footprint() {
    let Ok(model) = std::env::var("IZWI_MEASURE_MODEL") else {
        eprintln!("measure_metal_load_footprint: IZWI_MEASURE_MODEL is unset; nothing to measure");
        return;
    };
    let Some((_, _lease)) =
        crate::backends::metal_test_device_or_skip("measure_metal_load_footprint")
    else {
        return;
    };
    // A model directory name, or any name the catalog parses.
    let variant = ModelVariant::all()
        .iter()
        .copied()
        .find(|variant| variant.dir_name() == model.trim())
        .unwrap_or_else(|| {
            crate::catalog::parse_model_variant(&model).expect("IZWI_MEASURE_MODEL")
        });
    let models_dir = std::env::var("IZWI_MODELS_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::data_local_dir().unwrap().join("izwi/models"));
    let model_path = models_dir.join(variant.dir_name());
    assert!(
        model_path.is_dir(),
        "{} is not downloaded",
        model_path.display()
    );

    let device = DeviceSelector::detect_with_preference(Some("metal")).expect("metal device");
    let metal = device.device.as_metal_device().expect("metal").clone();
    let allocated = move || metal.metal_device().current_allocated_size() as u64;
    let config = EngineConfig::default();
    let granite = GraniteSpeechOptions::default().with_env_fallback();

    // What admission prices: the plan, and the bytes the Metal pre-load
    // check compares against free memory (the plan's peak, raised to the
    // weights plus one request envelope for ASR).
    let plan = admission_resource_plan(
        variant,
        &model_path,
        BackendKind::Metal,
        &device,
        &config.performance,
        &granite,
    )
    .expect("admission plan");
    let admission_bytes = known_unified(
        load_authorization_with_request_floor(
            variant,
            &plan,
            BackendKind::Metal,
            METAL_REQUEST_SLOTS,
        )
        .expect("load authorization"),
    );
    let stored_bytes = checkpoint_tensor_inventory(&model_path)
        .expect("checkpoint census")
        .map_or(0, |inventory| inventory.total_bytes);

    let registry =
        ModelRegistry::new_with_performance(models_dir, device.clone(), config.performance.clone())
            .with_granite_speech(granite);
    let device_thread = registry.device_thread().expect("device thread");
    let drain = || {
        device_thread
            .run_sync(|| MetalPoolManager::global().clear_all())
            .expect("clear Metal pools");
        device.device.synchronize().expect("synchronize");
    };
    drain();
    write_memory_reports(variant.dir_name(), "before");
    let before = [allocated(), phys_footprint_bytes()];
    let high_water = {
        let allocated = allocated.clone();
        HighWater::start(move || [allocated(), phys_footprint_bytes()])
    };

    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime
        .block_on(load_into_registry(
            &registry,
            variant,
            &model_path,
            None,
            config.kv_page_size,
            &config.kv_cache_dtype,
        ))
        .unwrap_or_else(|error| panic!("{variant} failed to load: {error}"));
    device.device.synchronize().expect("synchronize");
    let load_ms = started.elapsed().as_millis();
    // At the load fence: what the lifecycle sees when it publishes the model,
    // Candle's pooled released buffers included.
    let fence = [allocated(), phys_footprint_bytes()];
    drain();
    let settled = [allocated(), phys_footprint_bytes()];
    let peak = high_water.stop();
    write_memory_reports(variant.dir_name(), "settled");
    // libmalloc keeps freed large blocks (the load's host transients) as
    // dirty "Malloc Large (empty)" regions; ask it to return them, to show
    // how much of the settled footprint is freed memory rather than live.
    let malloc_relief_bytes = malloc_pressure_relief_bytes();
    let relieved = [allocated(), phys_footprint_bytes()];

    let metal_growth = |sample: [u64; 2]| sample[0].saturating_sub(before[0]);
    let footprint_growth = |sample: [u64; 2]| sample[1].saturating_sub(before[1]);
    let record = serde_json::json!({
        "model": variant.to_string(),
        "dir": variant.dir_name(),
        "stored_bytes": stored_bytes,
        "plan_resident_bytes": known_unified(plan.resident_authorization),
        "plan_load_peak_bytes": known_unified(plan.load_authorization),
        "admission_bytes": admission_bytes,
        "metal_peak_bytes": metal_growth(peak),
        "metal_fence_bytes": metal_growth(fence),
        "metal_settled_bytes": metal_growth(settled),
        "footprint_peak_bytes": footprint_growth(peak),
        "footprint_fence_bytes": footprint_growth(fence),
        "footprint_settled_bytes": footprint_growth(settled),
        "malloc_relief_bytes": malloc_relief_bytes,
        "footprint_relieved_bytes": footprint_growth(relieved),
        "load_ms": load_ms,
    });
    let gb = |key: &str| record[key].as_u64().unwrap_or(0) as f64 / 1e9;
    eprintln!(
        "FOOTPRINT {variant}: stored {:.2} GB | admission {:.2}, plan peak {:.2}, plan resident {:.2} | \
         Metal peak {:.2}, fence {:.2}, settled {:.2} | footprint peak {:.2}, settled {:.2}, \
         after malloc relief {:.2} | {load_ms} ms",
        gb("stored_bytes"),
        gb("admission_bytes"),
        gb("plan_load_peak_bytes"),
        gb("plan_resident_bytes"),
        gb("metal_peak_bytes"),
        gb("metal_fence_bytes"),
        gb("metal_settled_bytes"),
        gb("footprint_peak_bytes"),
        gb("footprint_settled_bytes"),
        gb("footprint_relieved_bytes"),
    );
    eprintln!("FOOTPRINT_JSON {record}");
    if let Ok(out) = std::env::var("IZWI_MEASURE_OUT") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out)
            .unwrap_or_else(|error| panic!("open {out}: {error}"));
        writeln!(file, "{record}").expect("write measurement");
    }
    drop(registry);
}
