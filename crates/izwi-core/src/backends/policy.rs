use candle_core::DType;

use super::types::BackendKind;

pub fn default_dtype_for_device(kind: BackendKind) -> DType {
    match kind {
        BackendKind::Cpu => DType::F32,
        // Keep existing stability/perf preference on Metal.
        BackendKind::Metal => DType::F32,
        BackendKind::Cuda => DType::BF16,
        BackendKind::Mlx => DType::F32,
    }
}

pub fn kv_dtype_bytes(requested_dtype: &str, is_metal: bool) -> usize {
    let requested = match requested_dtype.trim().to_ascii_lowercase().as_str() {
        "float32" | "f32" => 4,
        "int8" | "i8" | "q8" | "q8_0" => 1,
        _ => 2,
    };

    if is_metal && requested != 1 {
        4
    } else {
        requested
    }
}

pub fn can_parallelize_requests(kind: BackendKind) -> bool {
    !matches!(kind, BackendKind::Metal)
}
