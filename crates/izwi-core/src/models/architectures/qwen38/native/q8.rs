//! Bounded exact FP8 -> Q8_0 tiles, with one producer and one upload consumer.
use super::cache::{validate_q8, words_bytes, DerivedCache, ABI};
use super::*;
use candle_core::quantized::QStorage;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::sync::{atomic::Ordering, mpsc::sync_channel};
use std::time::Instant;

#[derive(Debug)]
struct TilePlan {
    blocks: usize,
    workers: usize,
}
impl TilePlan {
    fn new(options: &LoadingPerformanceConfig, scale_bytes: usize) -> Result<Self> {
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        let requested = if options.workers == 0 {
            available
        } else {
            options.workers
        };
        let workers = if options.parallel_conversion.enabled() {
            requested.min(available).clamp(1, 64)
        } else {
            1
        };
        // Reserve worker F32 blocks, source-scale copies/hash scratch, bounded
        // channel + producer + consumer tiles and two pinned upload buffers.
        let overhead = scale_bytes
            .checked_add(workers * 256 + 1024)
            .ok_or_else(|| Error::ModelLoadError("Q8 staging budget overflow".into()))?;
        let usable = options
            .max_staging_bytes
            .checked_sub(overhead)
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Q8 staging budget cannot hold scales and conversion scratch".into(),
                )
            })?;
        let blocks = (usable / 8 / 34).min(4 * 1024 * 1024 / 34);
        if blocks == 0 {
            return Err(Error::ModelLoadError(
                "Q8 staging budget cannot hold one conversion tile".into(),
            ));
        }
        Ok(Self { blocks, workers })
    }
}
struct Tile {
    offset: usize,
    words: Vec<u16>,
}

impl IndexedSafetensors {
    pub(super) fn materialize_q8_tiled(
        &self,
        projections: &[(&str, [usize; 2])],
        block_shape: [usize; 2],
        device: &Device,
    ) -> Result<QMatMul> {
        self.check_loading_cancelled()?;
        let _staging = self
            .loading
            .staging
            .lock()
            .map_err(|_| Error::ModelLoadError("Load staging lock poisoned".into()))?;
        if !cfg!(target_endian = "little") {
            return Err(Error::ModelLoadError(
                "Optimized Q8 loader requires little endian".into(),
            ));
        }
        let Some((_, first)) = projections.first() else {
            return Err(Error::ModelLoadError(
                "Native projection group cannot be empty".into(),
            ));
        };
        let mut rows = 0usize;
        let mut scale_bytes = 0usize;
        for (name, shape) in projections {
            if shape[1] != first[1] || shape.contains(&0) || !shape[1].is_multiple_of(32) {
                return Err(Error::ModelLoadError(format!(
                    "Incompatible Q8 projection shape for `{name}`: {shape:?}"
                )));
            }
            let info = self.tensor_info(name)?;
            if info.dtype != SafeDType::F8_E4M3 || info.shape != shape.as_slice() {
                return Err(Error::ModelLoadError(format!(
                    "Q8 projection `{name}` dtype/shape mismatch: {:?} {:?}, expected F8_E4M3 {shape:?}",
                    info.dtype, info.shape
                )));
            }
            rows = rows
                .checked_add(shape[0])
                .ok_or_else(|| Error::ModelLoadError("Q8 row count overflow".into()))?;
            let ss = block_scale_shape(*shape, block_shape)?;
            let scale_info = self.tensor_info(&scale_name_for_weight(name)?)?;
            if scale_info.dtype != SafeDType::BF16 || scale_info.shape != ss.as_slice() {
                return Err(Error::ModelLoadError(format!(
                    "Q8 scale for `{name}` dtype/shape mismatch: {:?} {:?}, expected BF16 {ss:?}",
                    scale_info.dtype, scale_info.shape
                )));
            }
            scale_bytes = scale_bytes.max(
                ss[0]
                    .checked_mul(ss[1])
                    .and_then(|n| n.checked_mul(8))
                    .ok_or_else(|| Error::ModelLoadError("Scale size overflow".into()))?,
            );
        }
        let shape = [rows, first[1]];
        let elements = rows
            .checked_mul(first[1])
            .ok_or_else(|| Error::ModelLoadError("Q8 element count overflow".into()))?;
        let total_bytes = (elements / 32)
            .checked_mul(34)
            .ok_or_else(|| Error::ModelLoadError("Q8 byte count overflow".into()))?;
        let plan = TilePlan::new(&self.options, scale_bytes)?;
        if self.loading.pool.get().is_none() {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(plan.workers)
                .stack_size(256 * 1024)
                .build()
                .map_err(|e| Error::ModelLoadError(format!("Conversion worker pool: {e}")))?;
            let _ = self.loading.pool.set(pool);
        }
        let pool = self
            .loading
            .pool
            .get()
            .ok_or_else(|| Error::ModelLoadError("Conversion pool missing".into()))?;
        let cache = self
            .loading
            .cache
            .get_or_init(|| DerivedCache::new(&self.options));
        // CPU storage below is used by portable tests. Production routing only
        // selects this method for CUDA; CPU/Metal keep their compatibility path.
        #[cfg(feature = "cuda")]
        let mut cuda = match device {
            Device::Cuda(d) => Some(super::upload::Q8Upload::new(
                d,
                elements,
                plan.blocks * 34,
                self.options.pinned_uploads.enabled(),
            )?),
            _ => None,
        };
        let mut cpu = Vec::new();
        if device.is_cpu() {
            cpu.try_reserve_exact(total_bytes / 2)
                .map_err(|e| Error::ModelLoadError(e.to_string()))?;
        }
        if !device.is_cpu() && !device.is_cuda() {
            return Err(Error::ModelLoadError(
                "Tiled Q8 upload requires CUDA or CPU".into(),
            ));
        }
        let (tx, rx) = sync_channel::<Result<Tile>>(1);
        std::thread::scope(|scope| -> Result<()> {
            let producer = std::thread::Builder::new()
                .name("qwen38-weight-conversion".into())
                .stack_size(256 * 1024)
                .spawn_scoped(scope, || {
                    let result = (|| -> Result<()> {
                        let mut output_offset = 0usize;
                        for (name, shape) in projections {
                            let scale_name = scale_name_for_weight(name)?;
                            let scale_shape = block_scale_shape(*shape, block_shape)?;
                            let (scales, scale_digest) = self.with_tensor_view(
                                &scale_name,
                                Some(SafeDType::BF16),
                                Some(&scale_shape),
                                |v| {
                                    // Hash and decode ONE immutable snapshot. A
                                    // mutable source must not publish values
                                    // under a digest observed before mutation.
                                    let snapshot = v.data().to_vec();
                                    Ok((
                                        decode_bf16_le(&snapshot, &scale_name)?,
                                        Sha256::digest(&snapshot),
                                    ))
                                },
                            )?;
                            loading::validate_scales(&scales)?;
                            self.with_tensor_tiles(
                                name,
                                SafeDType::F8_E4M3,
                                shape,
                                plan.blocks * 32,
                                |source_offset, raw| {
                                    self.check_loading_cancelled()?;
                                    // Bounded source snapshot belongs to this
                                    // tile's staging reservation. Hashing and
                                    // conversion consume exactly these bytes,
                                    // even if an unmanaged source changes later.
                                    let snapshot = raw.to_vec();
                                    let raw = snapshot.as_slice();
                                    let words_len = raw.len() / 32 * 17;
                                    let key: [u8; 32] = if cache.is_some() {
                                        let mut h = Sha256::new();
                                        h.update(ABI);
                                        h.update((name.len() as u64).to_le_bytes());
                                        h.update(name.as_bytes());
                                        for n in [
                                            shape[0],
                                            shape[1],
                                            block_shape[0],
                                            block_shape[1],
                                            source_offset,
                                            raw.len(),
                                        ] {
                                            h.update((n as u64).to_le_bytes());
                                        }
                                        h.update(scale_digest);
                                        h.update(raw);
                                        h.finalize().into()
                                    } else {
                                        [0; 32]
                                    };
                                    let hit = cache.as_ref().and_then(|c| c.read(&key, words_len));
                                    let words = if let Some(words) = hit {
                                        self.loading.cache_hits.fetch_add(1, Ordering::Relaxed);
                                        words
                                    } else {
                                        self.loading.cache_misses.fetch_add(1, Ordering::Relaxed);
                                        let start = Instant::now();
                                        let words = convert_tile(
                                            raw,
                                            source_offset,
                                            *shape,
                                            block_shape,
                                            scale_shape,
                                            &scales,
                                            pool,
                                        )?;
                                        self.loading.conversion_us.fetch_add(
                                            start.elapsed().as_micros() as u64,
                                            Ordering::Relaxed,
                                        );
                                        self.loading
                                            .converted_bytes
                                            .fetch_add((words.len() * 2) as u64, Ordering::Relaxed);
                                        if let Some(cache) = &cache {
                                            cache.publish(&key, &words);
                                        }
                                        words
                                    };
                                    let bytes = words.len() * 2;
                                    tx.send(Ok(Tile {
                                        offset: output_offset,
                                        words,
                                    }))
                                    .map_err(|_| {
                                        Error::ModelLoadError("Q8 upload consumer stopped".into())
                                    })?;
                                    output_offset += bytes;
                                    Ok(())
                                },
                            )?;
                        }
                        Ok(())
                    })();
                    if let Err(e) = result {
                        let _ = tx.send(Err(e));
                    }
                    drop(tx);
                })
                .map_err(|e| Error::ModelLoadError(format!("Conversion producer: {e}")))?;
            let consumed = (|| -> Result<()> {
                let mut received = 0;
                for tile in &rx {
                    self.check_loading_cancelled()?;
                    let tile = tile?;
                    validate_q8(&tile.words)?;
                    let bytes = tile.words.len() * 2;
                    if tile.offset != received
                        || received.checked_add(bytes).is_none_or(|n| n > total_bytes)
                    {
                        return Err(Error::ModelLoadError(
                            "Q8 tile upload bounds mismatch".into(),
                        ));
                    }
                    let start = Instant::now();
                    #[cfg(feature = "cuda")]
                    if let Some(cuda) = &mut cuda {
                        cuda.push(tile.offset, words_bytes(&tile.words))?;
                        let counter = if cuda.uses_pinned() {
                            &self.loading.pinned_upload_tiles
                        } else {
                            &self.loading.unpinned_upload_tiles
                        };
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                    if device.is_cpu() {
                        cpu.extend_from_slice(&tile.words);
                    }
                    self.loading
                        .upload_us
                        .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    self.loading
                        .uploaded_bytes
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    received += bytes;
                }
                if received != total_bytes {
                    return Err(Error::ModelLoadError("Incomplete Q8 tile stream".into()));
                }
                Ok(())
            })();
            // Drop receiver BEFORE joining: an upload failure must wake a blocked
            // producer instead of deadlocking load rollback.
            drop(rx);
            producer
                .join()
                .map_err(|_| Error::ModelLoadError("Q8 conversion worker panicked".into()))?;
            consumed
        })?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = cuda {
            let start = Instant::now();
            let storage = cuda.finish()?;
            self.loading
                .upload_us
                .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            return Ok(QMatMul::QTensor(Arc::new(QTensor::new(
                QStorage::Cuda(storage),
                &shape,
            )?)));
        }
        validate_q8(&cpu)?;
        let storage = QStorage::from_data(
            std::borrow::Cow::Borrowed(words_bytes(&cpu)),
            device,
            GgmlDType::Q8_0,
        )?;
        Ok(QMatMul::QTensor(Arc::new(QTensor::new(storage, &shape)?)))
    }
}

fn convert_tile(
    raw: &[u8],
    offset: usize,
    shape: [usize; 2],
    block: [usize; 2],
    scale_shape: [usize; 2],
    scales: &[f32],
    pool: &rayon::ThreadPool,
) -> Result<Vec<u16>> {
    if raw.is_empty() || !raw.len().is_multiple_of(32) {
        return Err(Error::ModelLoadError(
            "Q8 tile must contain complete 32-element blocks".into(),
        ));
    }
    let mut words = vec![0u16; raw.len() / 32 * 17];
    pool.install(|| {
        words
            .par_chunks_mut(17)
            .enumerate()
            .try_for_each(|(i, out)| -> Result<()> {
                let mut values = [0f32; 32];
                let mut amax = 0f32;
                for (j, v) in values.iter_mut().enumerate() {
                    let index = offset + i * 32 + j;
                    let row = index / shape[1];
                    let col = index % shape[1];
                    let value = decode_e4m3fn(raw[i * 32 + j]);
                    *v = value * scales[row / block[0] * scale_shape[1] + col / block[1]];
                    if !value.is_finite() || !v.is_finite() {
                        return Err(Error::ModelLoadError(format!(
                            "Non-finite E4M3FN conversion at [{row},{col}]"
                        )));
                    }
                    amax = amax.max(v.abs());
                }
                // Exactly Candle BlockQ8_0::from_float: derive inverse from F32 d,
                // round ties away from zero, only then store d as F16 (not vice versa).
                let d = amax / 127.;
                let inv = if d != 0. { 1. / d } else { 0. };
                out[0] = f16::from_f32(d).to_bits();
                for j in 0..16 {
                    out[j + 1] = u16::from_le_bytes([
                        (values[j * 2] * inv).round() as i8 as u8,
                        (values[j * 2 + 1] * inv).round() as i8 as u8,
                    ]);
                }
                Ok(())
            })
    })?;
    validate_q8(&words)?;
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{bf16_bytes, write_index, write_safetensors, TestDir};
    use super::*;
    #[test]
    fn exact_rounding_matches_candle_across_tiles_and_scale_boundaries() {
        let shape = [6, 160];
        let block = [3, 48];
        let ss = block_scale_shape(shape, block).unwrap();
        let scales = vec![0., 0.1, 3., 0.00001, 7., 0.333, 1., 2.];
        let raw: Vec<u8> = (0..960)
            .map(|i| {
                let b = (i * 37 % 256) as u8;
                if b & 127 == 127 {
                    0
                } else {
                    b
                }
            })
            .collect();
        let values = dequantize_e4m3fn_blockwise_f32(&raw, shape, &scales, ss, block).unwrap();
        let reference = QTensor::quantize(
            &Tensor::from_vec(values, &shape, &Device::Cpu).unwrap(),
            GgmlDType::Q8_0,
        )
        .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let mut packed = Vec::new();
        for (i, tile) in raw.chunks(96).enumerate() {
            packed.extend(convert_tile(tile, i * 96, shape, block, ss, &scales, &pool).unwrap());
        }
        assert_eq!(words_bytes(&packed), reference.data().unwrap().as_ref());
    }
    #[test]
    fn tiled_cache_hits_match_compatibility_and_source_changes_miss() {
        let d = TestDir::new("tiled-q8");
        let name = "test.weight";
        let scale = "test.weight_scale_inv";
        let data = vec![0x38; 2048];
        let scales = bf16_bytes(&[1.]);
        write_safetensors(
            &d.path().join("a.safetensors"),
            &[
                (name, SafeDType::F8_E4M3, vec![16, 128], &data),
                (scale, SafeDType::BF16, vec![1, 1], &scales),
            ],
        );
        write_index(
            d.path(),
            serde_json::json!({(name):"a.safetensors",(scale):"a.safetensors"}),
        );
        let options = LoadingPerformanceConfig {
            cache_dir: Some(d.path().join("cache")),
            max_staging_bytes: 4096,
            workers: 2,
            ..Default::default()
        };
        let source = IndexedSafetensors::open_with_options(d.path(), &options).unwrap();
        let expected = source
            .materialize_q8_projection(name, [16, 128], [128, 128], &Device::Cpu)
            .unwrap();
        let first = source
            .materialize_q8_tiled(&[(name, [16, 128])], [128, 128], &Device::Cpu)
            .unwrap();
        let second = source
            .materialize_q8_tiled(&[(name, [16, 128])], [128, 128], &Device::Cpu)
            .unwrap();
        for actual in [first, second] {
            let (QMatMul::QTensor(a), QMatMul::QTensor(b)) = (actual, &expected) else {
                panic!()
            };
            assert_eq!(a.data().unwrap(), b.data().unwrap());
        }
        assert!(source.loading.cache_hits.load(Ordering::Relaxed) > 0);
        // New process/load scope hashes source again, including same-length edits.
        drop(source);
        write_safetensors(
            &d.path().join("a.safetensors"),
            &[
                (name, SafeDType::F8_E4M3, vec![16, 128], &vec![0x40; 2048]),
                (scale, SafeDType::BF16, vec![1, 1], &scales),
            ],
        );
        let source = IndexedSafetensors::open_with_options(d.path(), &options).unwrap();
        source
            .materialize_q8_tiled(&[(name, [16, 128])], [128, 128], &Device::Cpu)
            .unwrap();
        assert_eq!(source.loading.cache_hits.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn invalid_source_stops_producer_and_releases_staging() {
        let d = TestDir::new("q8-invalid-source");
        let scale = bf16_bytes(&[1.]);
        let mut bytes = vec![0x38; 128];
        bytes[100] = 0x7f;
        write_safetensors(
            &d.path().join("a.safetensors"),
            &[
                ("x.weight", SafeDType::F8_E4M3, vec![1, 128], &bytes),
                ("x.weight_scale_inv", SafeDType::BF16, vec![1, 1], &scale),
            ],
        );
        write_index(
            d.path(),
            serde_json::json!({"x.weight":"a.safetensors","x.weight_scale_inv":"a.safetensors"}),
        );
        let options = LoadingPerformanceConfig {
            derived_weight_cache: crate::performance::OptimizationMode::Off,
            workers: 1,
            max_staging_bytes: 4096,
            ..Default::default()
        };
        let source = IndexedSafetensors::open_with_options(d.path(), &options).unwrap();
        assert!(source
            .materialize_q8_tiled(&[("x.weight", [1, 128])], [128, 128], &Device::Cpu)
            .is_err());
        assert!(source.loading.staging.try_lock().is_ok());
        source.cancel_loading();
        assert!(matches!(
            source.materialize_q8_tiled(&[("x.weight", [1, 128])], [128, 128], &Device::Cpu),
            Err(Error::Cancelled(_))
        ));
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_tiled_pinned_q8_payload_and_eager_matmul_match_cpu() {
        use candle_core::Module;
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(error) => {
                assert!(
                    std::env::var("IZWI_REQUIRE_CUDA_TEST_DEVICE").as_deref() != Ok("1"),
                    "required CUDA loader device unavailable: {error}"
                );
                eprintln!("Skipping CUDA loader probe: {error}");
                return;
            }
        };
        let dir = TestDir::new("cuda-tiled-q8");
        let weights: Vec<u8> = (0..2048)
            .map(|i| [0x38, 0xb8, 0x40, 0xc0, 0x01, 0x7e][i % 6])
            .collect();
        let scales = bf16_bytes(&[2.5]);
        write_safetensors(
            &dir.path().join("a.safetensors"),
            &[
                ("x.weight", SafeDType::F8_E4M3, vec![16, 128], &weights),
                ("x.weight_scale_inv", SafeDType::BF16, vec![1, 1], &scales),
            ],
        );
        write_index(
            dir.path(),
            serde_json::json!({"x.weight":"a.safetensors","x.weight_scale_inv":"a.safetensors"}),
        );
        let baseline = IndexedSafetensors::open(dir.path())
            .unwrap()
            .materialize_q8_projection("x.weight", [16, 128], [128, 128], &Device::Cpu)
            .unwrap();
        let input: Vec<f32> = (0..256).map(|i| (i as f32 % 17. - 8.) / 32.).collect();
        let cpu_input = Tensor::from_vec(input, (2, 128), &Device::Cpu).unwrap();
        let expected = baseline
            .forward(&cpu_input)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for pinned_uploads in [
            crate::performance::OptimizationMode::Auto,
            crate::performance::OptimizationMode::Off,
        ] {
            let options = LoadingPerformanceConfig {
                pinned_uploads,
                workers: 1,
                max_staging_bytes: 4096,
                cache_dir: Some(dir.path().join("cache")),
                ..Default::default()
            };
            let checkpoint = IndexedSafetensors::open_with_options(dir.path(), &options).unwrap();
            for _ in 0..2 {
                let actual = checkpoint
                    .materialize_q8_projection("x.weight", [16, 128], [128, 128], &device)
                    .unwrap();
                let (QMatMul::QTensor(a), QMatMul::QTensor(b)) = (&actual, &baseline) else {
                    panic!("expected packed Q8 weights");
                };
                assert_eq!(
                    a.data().unwrap().as_ref(),
                    b.data().unwrap().as_ref(),
                    "CUDA upload changed packed bytes"
                );
                let output = actual
                    .forward(&cpu_input.to_device(&device).unwrap())
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                for (a, b) in output.iter().flatten().zip(expected.iter().flatten()) {
                    assert!(
                        (a - b).abs() <= 0.002 + 2e-4 * b.abs(),
                        "CUDA eager matmul mismatch {a} vs {b}"
                    );
                }
            }
            assert!(checkpoint.loading.cache_hits.load(Ordering::Relaxed) > 0);
            if pinned_uploads.enabled() {
                assert!(
                    checkpoint
                        .loading
                        .pinned_upload_tiles
                        .load(Ordering::Relaxed)
                        > 0,
                    "CUDA probe did not exercise pinned transfers"
                );
            } else {
                assert!(
                    checkpoint
                        .loading
                        .unpinned_upload_tiles
                        .load(Ordering::Relaxed)
                        > 0
                );
            }
        }
    }
    #[test]
    fn staging_budget_bounds_tiles_and_rejects_too_small() {
        let o = LoadingPerformanceConfig {
            max_staging_bytes: 4096,
            workers: 2,
            ..Default::default()
        };
        let p = TilePlan::new(&o, 128).unwrap();
        assert!(p.blocks * 34 * 8 + p.workers * 256 + 1024 + 128 <= 4096);
        assert!(TilePlan::new(
            &LoadingPerformanceConfig {
                max_staging_bytes: 1,
                ..o
            },
            0
        )
        .is_err());
    }
}
