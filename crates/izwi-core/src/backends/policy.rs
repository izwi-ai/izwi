use candle_core::DType;

use super::types::BackendKind;

pub fn default_dtype_for_device(kind: BackendKind) -> DType {
    match kind {
        BackendKind::Cpu => DType::F32,
        // Keep existing stability/perf preference on Metal.
        BackendKind::Metal => DType::F32,
        BackendKind::Cuda => DType::BF16,
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

/// Only CPU rows run on parallel request threads. Metal shares one command
/// queue; CUDA shares one device stream whose graph capture is thread-local.
pub fn can_parallelize_requests(kind: BackendKind) -> bool {
    kind == BackendKind::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_cpu_rows_run_on_parallel_request_threads() {
        assert!(can_parallelize_requests(BackendKind::Cpu));
        assert!(!can_parallelize_requests(BackendKind::Metal));
        assert!(!can_parallelize_requests(BackendKind::Cuda));
    }
}
