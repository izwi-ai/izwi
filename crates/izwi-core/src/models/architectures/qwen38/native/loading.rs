//! Load-owned mappings and typed, pre-initialization checkpoint values.
use super::*;
use crate::performance::LoadingIoStrategy;
use memmap2::Mmap;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Mutex, OnceLock,
};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileIdentity {
    len: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    unix: (u64, u64, i64, i64),
}
impl FileIdentity {
    fn of(file: &File) -> Result<Self> {
        let m = file.metadata()?;
        Ok(Self {
            len: m.len(),
            modified: m.modified().ok(),
            #[cfg(unix)]
            unix: {
                use std::os::unix::fs::MetadataExt;
                (m.dev(), m.ino(), m.ctime(), m.ctime_nsec())
            },
        })
    }
}

#[derive(Debug)]
struct Shard {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    mapping: Mmap,
    header: Arc<safetensors::tensor::Metadata>,
    payload: usize,
}
impl Shard {
    fn unchanged(&self) -> Result<()> {
        // Retained mappings assume no in-place writer, like the compatibility
        // loader. Refuse observable replacement/truncation before and after use.
        // This metadata is never used as authentication for derived bytes: each
        // source tile and its scales are hashed on EVERY cache lookup.
        if FileIdentity::of(&self.file)? != self.identity
            || FileIdentity::of(&File::open(&self.path)?)? != self.identity
        {
            return Err(Error::ModelLoadError(format!(
                "Checkpoint shard changed during load: {}",
                self.path.display()
            )));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct LoadState {
    shards: Mutex<VecDeque<Arc<Shard>>>,
    headers: Mutex<BTreeMap<PathBuf, (FileIdentity, usize, Arc<safetensors::tensor::Metadata>)>>,
    cancelled: AtomicBool,
    pub staging: Mutex<()>,
    pub pool: OnceLock<rayon::ThreadPool>,
    pub cache: OnceLock<Option<super::cache::DerivedCache>>,
    pub discovery_us: AtomicU64,
    pub validation_us: AtomicU64,
    pub conversion_us: AtomicU64,
    pub upload_us: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub converted_bytes: AtomicU64,
    pub uploaded_bytes: AtomicU64,
    pub pinned_upload_tiles: AtomicU64,
    pub unpinned_upload_tiles: AtomicU64,
}
impl LoadState {
    pub fn new(discovery: Duration) -> Self {
        Self {
            shards: Mutex::new(VecDeque::new()),
            headers: Mutex::new(BTreeMap::new()),
            cancelled: AtomicBool::new(false),
            staging: Mutex::new(()),
            pool: OnceLock::new(),
            cache: OnceLock::new(),
            discovery_us: AtomicU64::new(discovery.as_micros() as u64),
            validation_us: AtomicU64::new(0),
            conversion_us: AtomicU64::new(0),
            upload_us: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            converted_bytes: AtomicU64::new(0),
            uploaded_bytes: AtomicU64::new(0),
            pinned_upload_tiles: AtomicU64::new(0),
            unpinned_upload_tiles: AtomicU64::new(0),
        }
    }
}

/// Original compact checkpoint representation. Values have no normalization
/// offset or A_log transform applied; construction owns those transforms.
#[derive(Debug, Clone)]
pub struct RawBlockFp8Projection {
    pub weights: Tensor,
    pub scales: Tensor,
    pub shape: [usize; 2],
    pub scale_shape: [usize; 2],
    pub block_shape: [usize; 2],
}

impl IndexedSafetensors {
    /// Cancel this load scope, including clones. No partially uploaded tensor is
    /// published; a later load uses a new checkpoint and independent token.
    pub fn cancel_loading(&self) {
        self.loading.cancelled.store(true, Ordering::Release);
    }
    pub(super) fn check_loading_cancelled(&self) -> Result<()> {
        if self.loading.cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled(
                "Native checkpoint loading cancelled".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn optimized_on(&self, device: &Device) -> bool {
        self.options.enabled() && device.is_cuda()
    }

    pub fn loading_diagnostics(&self) -> serde_json::Value {
        let s = &self.loading;
        let n = |a: &AtomicU64| a.load(Ordering::Relaxed);
        serde_json::json!({
            "discovery_ms": n(&s.discovery_us) as f64 / 1000.,
            "validation_ms": n(&s.validation_us) as f64 / 1000.,
            "conversion_ms": n(&s.conversion_us) as f64 / 1000.,
            "upload_ms": n(&s.upload_us) as f64 / 1000.,
            "pinned_upload_tiles": n(&s.pinned_upload_tiles), "unpinned_upload_tiles": n(&s.unpinned_upload_tiles),
            "cache_hits": n(&s.cache_hits), "cache_misses": n(&s.cache_misses),
            "converted_bytes": n(&s.converted_bytes), "uploaded_bytes": n(&s.uploaded_bytes),
            "cache_source_authentication": "rehash-source-tiles-and-scales",
            "conversion_upload_timing_scope": "q8-tiles-only",
            "io_strategy_resolved": if self.options.io_strategy==LoadingIoStrategy::Sequential {"bounded-sequential-q8; dense-mmap"} else {"mmap"},
        })
    }

    pub(super) fn retained_tensor_view<T, F>(
        &self,
        name: &str,
        dtype: Option<SafeDType>,
        shape: Option<&[usize]>,
        consume: F,
    ) -> Result<T>
    where
        F: FnOnce(TensorView<'_>) -> Result<T>,
    {
        self.check_loading_cancelled()?;
        let started = Instant::now();
        let path = self.shard_path_for_tensor(name)?;
        let shard = {
            let mut shards = self
                .loading
                .shards
                .lock()
                .map_err(|_| Error::ModelLoadError("Shard cache lock poisoned".into()))?;
            if let Some(i) = shards.iter().position(|s| s.path == path) {
                let s = shards.remove(i).expect("position exists");
                s.unchanged()?;
                shards.push_back(s.clone());
                s
            } else {
                let file = File::open(&path)?;
                let identity = FileIdentity::of(&file)?;
                // SAFETY: checkpoint source files must not be modified in place
                // during loading. We check identity/size/change metadata around
                // use; mappings never escape the load-owned cache/callback.
                let mapping = unsafe { MmapOptions::new().map(&file) }?;
                #[cfg(unix)]
                if self.options.io_strategy == LoadingIoStrategy::Sequential {
                    // Forward access hint avoids an extra full-shard heap copy.
                    let _ = mapping.advise(memmap2::Advice::Sequential);
                }
                let (payload, header) = {
                    let mut headers =
                        self.loading.headers.lock().map_err(|_| {
                            Error::ModelLoadError("Shard header cache poisoned".into())
                        })?;
                    if let Some((prior, payload, header)) = headers.get(&path) {
                        if prior != &identity {
                            return Err(Error::ModelLoadError(format!(
                                "Checkpoint shard changed during load: {}",
                                path.display()
                            )));
                        }
                        (*payload, header.clone())
                    } else {
                        let (length, header) =
                            SafeTensors::read_metadata(&mapping).map_err(|e| {
                                Error::ModelLoadError(format!(
                                    "Invalid Safetensors shard {}: {e}",
                                    path.display()
                                ))
                            })?;
                        let header = Arc::new(header);
                        let payload = length + 8;
                        headers.insert(path.clone(), (identity.clone(), payload, header.clone()));
                        (payload, header)
                    }
                };
                let s = Arc::new(Shard {
                    path,
                    file,
                    identity,
                    mapping,
                    header,
                    payload,
                });
                s.unchanged()?;
                while shards.len() >= 2 {
                    shards.pop_front();
                }
                shards.push_back(s.clone());
                s
            }
        };
        let info = shard.header.info(name).ok_or_else(|| {
            Error::ModelLoadError(format!("Indexed tensor `{name}` missing from shard"))
        })?;
        if dtype.is_some_and(|d| d != info.dtype) {
            return Err(Error::ModelLoadError(format!(
                "Native tensor `{name}` dtype mismatch: found {:?}, expected {dtype:?}",
                info.dtype
            )));
        }
        if shape.is_some_and(|s| s != info.shape) {
            return Err(Error::ModelLoadError(format!(
                "Native tensor `{name}` shape mismatch: found {:?}, expected {shape:?}",
                info.shape
            )));
        }
        let start = shard
            .payload
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| Error::ModelLoadError("Tensor offset overflow".into()))?;
        let end = shard
            .payload
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| Error::ModelLoadError("Tensor offset overflow".into()))?;
        let data = shard
            .mapping
            .get(start..end)
            .ok_or_else(|| Error::ModelLoadError("Tensor range exceeds shard".into()))?;
        let view = TensorView::new(info.dtype, info.shape.clone(), data)
            .map_err(|e| Error::ModelLoadError(e.to_string()))?;
        self.loading
            .validation_us
            .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        let result = consume(view);
        shard.unchanged()?;
        self.check_loading_cancelled()?;
        result
    }

    /// Forward bounded reads for the sequential option, borrowed mapping tiles
    /// otherwise. Headers remain validated/retained in both cases.
    pub(super) fn with_tensor_tiles<F>(
        &self,
        name: &str,
        dtype: SafeDType,
        shape: &[usize],
        tile_bytes: usize,
        mut consume: F,
    ) -> Result<()>
    where
        F: FnMut(usize, &[u8]) -> Result<()>,
    {
        if tile_bytes == 0 {
            return Err(Error::ModelLoadError("Zero source tile size".into()));
        }
        self.with_tensor_view(name, Some(dtype), Some(shape), |view| {
            if self.options.enabled() && self.options.io_strategy == LoadingIoStrategy::Sequential {
                use std::io::{Read, Seek, SeekFrom};
                let path = self.shard_path_for_tensor(name)?;
                let offset = {
                    let headers =
                        self.loading.headers.lock().map_err(|_| {
                            Error::ModelLoadError("Shard header cache poisoned".into())
                        })?;
                    let (_, payload, header) = headers.get(&path).ok_or_else(|| {
                        Error::ModelLoadError("Missing retained shard header".into())
                    })?;
                    let info = header
                        .info(name)
                        .ok_or_else(|| Error::ModelLoadError("Missing retained tensor".into()))?;
                    payload.checked_add(info.data_offsets.0).ok_or_else(|| {
                        Error::ModelLoadError("Sequential source offset overflow".into())
                    })?
                };
                let mut file = File::open(path)?;
                file.seek(SeekFrom::Start(offset as u64))?;
                let mut buffer = vec![0u8; tile_bytes.min(view.data().len())];
                let mut position = 0;
                while position < view.data().len() {
                    self.check_loading_cancelled()?;
                    let length = buffer.len().min(view.data().len() - position);
                    file.read_exact(&mut buffer[..length])?;
                    consume(position, &buffer[..length])?;
                    position += length;
                }
            } else {
                for (i, tile) in view.data().chunks(tile_bytes).enumerate() {
                    self.check_loading_cancelled()?;
                    consume(i * tile_bytes, tile)?;
                }
            }
            Ok(())
        })
    }

    pub fn materialize_block_fp8_raw(
        &self,
        weight_name: &str,
        expected_shape: [usize; 2],
        block_shape: [usize; 2],
        device: &Device,
    ) -> Result<RawBlockFp8Projection> {
        #[cfg(feature = "cuda")]
        super::upload::ensure_context_healthy(device)?;
        let _staging = self
            .loading
            .staging
            .lock()
            .map_err(|_| Error::ModelLoadError("Load staging lock poisoned".into()))?;
        let scale_name = scale_name_for_weight(weight_name)?;
        let scale_shape = block_scale_shape(expected_shape, block_shape)?;
        if self.options.enabled() && device.is_cuda() {
            let scale_bytes = scale_shape[0]
                .checked_mul(scale_shape[1])
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| Error::ModelLoadError("FP8 scale bytes overflow".into()))?;
            if scale_bytes > self.options.max_staging_bytes {
                return Err(Error::ModelLoadError(
                    "Raw FP8 scales exceed host staging budget".into(),
                ));
            }
        }
        let scales = self.with_tensor_view(
            &scale_name,
            Some(SafeDType::BF16),
            Some(&scale_shape),
            |v| decode_bf16_le(v.data(), &scale_name),
        )?;
        validate_scales(&scales)?;
        let weights = self.with_tensor_view(
            weight_name,
            Some(SafeDType::F8_E4M3),
            Some(&expected_shape),
            |v| {
                for (i, &b) in v.data().iter().enumerate() {
                    let value = decode_e4m3fn(b);
                    let scale = scales[(i / expected_shape[1] / block_shape[0]) * scale_shape[1]
                        + i % expected_shape[1] / block_shape[1]];
                    if !value.is_finite() || !(value * scale).is_finite() {
                        return Err(Error::ModelLoadError(format!(
                            "Non-finite block FP8 value in `{weight_name}` at {i}"
                        )));
                    }
                }
                Ok(Tensor::from_slice(v.data(), &expected_shape, device)?)
            },
        )?;
        let scales = Tensor::from_vec(scales, &scale_shape, device)?;
        Ok(RawBlockFp8Projection {
            weights,
            scales,
            shape: expected_shape,
            scale_shape,
            block_shape,
        })
    }
}

pub(super) fn validate_scales(scales: &[f32]) -> Result<()> {
    if scales.iter().any(|s| !s.is_finite() || *s < 0.) {
        return Err(Error::ModelLoadError(
            "Block-FP8 inverse scale is non-finite or negative".into(),
        ));
    }
    Ok(())
}

/// Decode each scalar through F32, but allocate ONLY the requested final host
/// dtype. F32 math tensors stay F32; normalization and -exp(A_log) still happen
/// exactly once in the existing model constructor, after this function.
pub(super) fn materialize_dense_typed(
    view: TensorView<'_>,
    name: &str,
    target: ProjectionMaterialization,
    device: &Device,
    max_staging_bytes: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    super::upload::ensure_context_healthy(device)?;
    let width = match view.dtype() {
        SafeDType::BF16 | SafeDType::F16 => 2,
        SafeDType::F32 => 4,
        d => {
            return Err(Error::ModelLoadError(format!(
                "Unsupported dense dtype {d:?}"
            )))
        }
    };
    let count = view
        .shape()
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| Error::ModelLoadError("Dense shape overflow".into()))?;
    if count.checked_mul(width) != Some(view.data().len()) {
        return Err(Error::ModelLoadError(
            "Dense payload length mismatch".into(),
        ));
    }
    let values = view.data().chunks_exact(width).enumerate().map(|(i, b)| {
        let v = match view.dtype() {
            SafeDType::BF16 => bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
            SafeDType::F16 => f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
            _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        };
        if v.is_finite() {
            Ok(v)
        } else {
            Err(Error::ModelLoadError(format!(
                "Native dense tensor `{name}` contains non-finite value at {i}"
            )))
        }
    });
    let same_dtype = matches!(
        (view.dtype(), target),
        (SafeDType::BF16, ProjectionMaterialization::BF16)
            | (SafeDType::F16, ProjectionMaterialization::F16)
            | (SafeDType::F32, ProjectionMaterialization::F32)
    );
    if same_dtype
        && cfg!(target_endian = "little")
        && (view.data().as_ptr() as usize).is_multiple_of(width)
    {
        for value in values {
            value?;
        }
        // Validate shape, exact length and pointer alignment BEFORE Candle's
        // typed raw-buffer API. This borrows the mapping; no 2.54GB BF16 host
        // copy for the head or embedding is created on CUDA.
        return Ok(Tensor::from_raw_buffer(
            view.data(),
            target.dtype(),
            view.shape(),
            device,
        )?);
    }
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        return super::upload::dense_from_values(
            values,
            view.shape(),
            target,
            device,
            max_staging_bytes,
        );
    }
    let _ = max_staging_bytes;
    Ok(match target {
        ProjectionMaterialization::F32 => {
            Tensor::from_vec(values.collect::<Result<Vec<_>>>()?, view.shape(), device)?
        }
        ProjectionMaterialization::F16 => Tensor::from_vec(
            values
                .map(|v| {
                    v.and_then(|v| {
                        let converted = f16::from_f32(v);
                        if converted.is_finite() {
                            Ok(converted)
                        } else {
                            Err(Error::ModelLoadError(format!(
                                "Dense tensor `{name}` overflows F16"
                            )))
                        }
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            view.shape(),
            device,
        )?,
        ProjectionMaterialization::BF16 => Tensor::from_vec(
            values
                .map(|v| {
                    v.and_then(|v| {
                        let converted = bf16::from_f32(v);
                        if converted.is_finite() {
                            Ok(converted)
                        } else {
                            Err(Error::ModelLoadError(format!(
                                "Dense tensor `{name}` overflows BF16"
                            )))
                        }
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            view.shape(),
            device,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::{bf16_bytes, write_index, write_safetensors, TestDir};
    use super::*;
    #[test]
    fn typed_dense_matches_reference_for_every_dtype_and_alignment() {
        let values = [0., -0., 1. + 1. / 256., -2. - 1. / 256., 0.000003, 127.5];
        for dtype in [SafeDType::BF16, SafeDType::F16, SafeDType::F32] {
            let bytes: Vec<u8> = match dtype {
                SafeDType::BF16 => bf16_bytes(&values),
                SafeDType::F16 => values
                    .iter()
                    .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
                    .collect(),
                _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            for offset in 0..4 {
                let mut storage = vec![0u8; offset];
                storage.extend_from_slice(&bytes);
                for target in [
                    ProjectionMaterialization::BF16,
                    ProjectionMaterialization::F16,
                    ProjectionMaterialization::F32,
                ] {
                    let view = TensorView::new(dtype, vec![2, 3], &storage[offset..]).unwrap();
                    let decoded = decode_dense_f32(view.clone(), "fixture").unwrap();
                    let expected = materialize_f32(decoded, &[2, 3], target, &Device::Cpu).unwrap();
                    let actual =
                        materialize_dense_typed(view, "fixture", target, &Device::Cpu, 1024)
                            .unwrap();
                    assert_eq!(actual.dtype(), target.dtype());
                    let a = actual
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let b = expected
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
        let bytes = f32::NAN.to_le_bytes();
        let v = TensorView::new(SafeDType::F32, vec![1], &bytes).unwrap();
        assert!(materialize_dense_typed(
            v,
            "A_log",
            ProjectionMaterialization::F32,
            &Device::Cpu,
            1024
        )
        .is_err());
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dense_typed_tiles_match_cpu() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                assert!(
                    std::env::var("IZWI_REQUIRE_CUDA_TEST_DEVICE").as_deref() != Ok("1"),
                    "required CUDA dense loader device unavailable: {e}"
                );
                eprintln!("Skipping CUDA dense loader probe: {e}");
                return;
            }
        };
        let mut bytes = vec![0u8];
        bytes.extend((0..32).flat_map(|i| ((i as f32 - 17.) / 128.).to_le_bytes()));
        for target in [
            ProjectionMaterialization::F32,
            ProjectionMaterialization::F16,
            ProjectionMaterialization::BF16,
        ] {
            let view = TensorView::new(SafeDType::F32, vec![4, 8], &bytes[1..]).unwrap();
            let expected =
                materialize_dense_typed(view.clone(), "dense", target, &Device::Cpu, 16).unwrap();
            let actual = materialize_dense_typed(view, "dense", target, &device, 16)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            assert_eq!(
                actual
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap(),
                expected
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap()
            );
        }
    }
    #[test]
    fn typed_dense_rejects_target_overflow() {
        let bytes = 70000f32.to_le_bytes();
        let view = TensorView::new(SafeDType::F32, vec![1], &bytes).unwrap();
        assert!(materialize_dense_typed(
            view,
            "overflow",
            ProjectionMaterialization::F16,
            &Device::Cpu,
            1024
        )
        .is_err());
    }
    #[test]
    fn sequential_tiles_match_mmap_and_cancellation_stops_at_boundary() {
        let dir = TestDir::new("sequential-cancel");
        let bytes = vec![0x38; 256];
        write_safetensors(
            &dir.path().join("a.safetensors"),
            &[("x", SafeDType::F8_E4M3, vec![2, 128], &bytes)],
        );
        write_index(dir.path(), serde_json::json!({"x":"a.safetensors"}));
        for io_strategy in [LoadingIoStrategy::Mmap, LoadingIoStrategy::Sequential] {
            let options = LoadingPerformanceConfig {
                io_strategy,
                ..Default::default()
            };
            let source = IndexedSafetensors::open_with_options(dir.path(), &options).unwrap();
            let mut joined = Vec::new();
            source
                .with_tensor_tiles("x", SafeDType::F8_E4M3, &[2, 128], 32, |offset, data| {
                    assert_eq!(offset, joined.len());
                    joined.extend_from_slice(data);
                    Ok(())
                })
                .unwrap();
            assert_eq!(joined, bytes);
            let mut delivered = 0;
            let error = source
                .with_tensor_tiles("x", SafeDType::F8_E4M3, &[2, 128], 32, |_, _| {
                    delivered += 1;
                    source.cancel_loading();
                    Ok(())
                })
                .unwrap_err();
            assert!(matches!(error, Error::Cancelled(_)));
            assert_eq!(delivered, 1);
            assert!(IndexedSafetensors::open_with_options(dir.path(), &options)
                .unwrap()
                .tensor_info("x")
                .is_ok());
        }
    }
    #[test]
    fn raw_fp8_preserves_bytes_and_scale_geometry_without_initialization() {
        let d = TestDir::new("raw-fp8");
        let weights = [0x38, 0xc0, 0x01, 0xfe];
        let scales = bf16_bytes(&[2., 3.]);
        write_safetensors(
            &d.path().join("a.safetensors"),
            &[
                ("x.weight", SafeDType::F8_E4M3, vec![2, 2], &weights),
                ("x.weight_scale_inv", SafeDType::BF16, vec![2, 1], &scales),
            ],
        );
        write_index(
            d.path(),
            serde_json::json!({"x.weight":"a.safetensors","x.weight_scale_inv":"a.safetensors"}),
        );
        let s = IndexedSafetensors::open(d.path()).unwrap();
        let raw = s
            .materialize_block_fp8_raw("x.weight", [2, 2], [1, 2], &Device::Cpu)
            .unwrap();
        assert_eq!(raw.weights.dtype(), DType::U8);
        assert_eq!(raw.scales.dtype(), DType::F32);
        assert_eq!(
            raw.weights.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
            weights
        );
        assert_eq!(
            raw.scales.to_vec2::<f32>().unwrap(),
            vec![vec![2.], vec![3.]]
        );
        assert_eq!(raw.block_shape, [1, 2]);
        assert_eq!(raw.scale_shape, [2, 1]);
    }
    #[test]
    fn mappings_are_bounded_headers_retained_and_mutation_rejected() {
        let d = TestDir::new("shard-retention");
        let data = 1f32.to_le_bytes();
        // Three index names require matching tensor names within each shard.
        for i in 0..3 {
            write_safetensors(
                &d.path().join(format!("{i}.safetensors")),
                &[(&format!("x{i}"), SafeDType::F32, vec![1], &data)],
            );
        }
        write_index(
            d.path(),
            serde_json::json!({"x0":"0.safetensors","x1":"1.safetensors","x2":"2.safetensors"}),
        );
        let s = IndexedSafetensors::open(d.path()).unwrap();
        for i in 0..3 {
            s.tensor_info(&format!("x{i}")).unwrap();
        }
        assert_eq!(s.loading.shards.lock().unwrap().len(), 2);
        assert_eq!(s.loading.headers.lock().unwrap().len(), 3);
        s.tensor_info("x0").unwrap();
        assert_eq!(s.loading.shards.lock().unwrap().len(), 2);
        // Atomic replacement is detected even if size and original FD stay same.
        write_safetensors(
            &d.path().join("replacement.safetensors"),
            &[("x0", SafeDType::F32, vec![1], &2f32.to_le_bytes())],
        );
        fs::rename(
            d.path().join("replacement.safetensors"),
            d.path().join("0.safetensors"),
        )
        .unwrap();
        assert!(s
            .tensor_info("x0")
            .unwrap_err()
            .to_string()
            .contains("changed during load"));
    }
}

/// An unproved asynchronous completion must never run resource destructors.
#[cfg(any(feature = "cuda", test))]
pub(super) fn release_after_fence<T>(resources: T, complete: bool) {
    if !complete {
        std::mem::forget(resources);
    }
}

#[cfg(test)]
mod lifetime_tests {
    use super::*;
    #[test]
    fn cancellation_frees_only_after_a_proven_fence() {
        struct Owner(Arc<AtomicU64>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let dropped = Arc::new(AtomicU64::new(0));
        release_after_fence(Owner(dropped.clone()), true);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        release_after_fence(Owner(dropped.clone()), false);
        assert_eq!(
            dropped.load(Ordering::Relaxed),
            1,
            "unproven DMA must retain its source and destination owner"
        );
    }
}
