use candle_core::DType;

use super::device::DeviceKind;

pub fn default_dtype_for_device(kind: DeviceKind) -> DType {
    match kind {
        DeviceKind::Cpu => DType::F32,
        // Keep existing stability/perf preference on Metal.
        DeviceKind::Metal => DType::F32,
        DeviceKind::Cuda => DType::BF16,
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

pub fn can_parallelize_requests(kind: DeviceKind) -> bool {
    !matches!(kind, DeviceKind::Metal)
}
