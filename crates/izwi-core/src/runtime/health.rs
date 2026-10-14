//! Runtime health: admission poison plus latched fatal device faults.
//!
//! Two independent signals make a runtime stop admitting physical work:
//! - the execution admission poison, set when a panic crosses native entry or
//!   a fatal device fault is latched, which is permanent for this process;
//! - the shared resource authority poison, set on backend-fatal asynchronous
//!   errors such as a Metal command-buffer OOM, which clears once the backend
//!   fully drains.
//!
//! Probes and worker status read [`RuntimeHealth`] instead of static
//! configuration so a bricked runtime is never advertised as ready.

use crate::error::Error;

/// Health of one runtime's physical execution, as seen by probes and status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeHealth {
    /// New physical work is admitted.
    Healthy,
    /// New physical work is rejected until the backend fully drains (for
    /// example after a Metal command-buffer OOM). The process may recover
    /// without a restart.
    Degraded { reason: String },
    /// New physical work is rejected for the life of this process: a panic
    /// crossed native entry or a fatal device fault was latched. Only process
    /// recreation recovers.
    Failed { reason: String },
}

impl RuntimeHealth {
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Healthy => None,
            Self::Degraded { reason } | Self::Failed { reason } => Some(reason),
        }
    }
}

/// CUDA driver errors that leave the context unusable ("sticky" errors): every
/// later call on the context fails the same way, so the process must be
/// recreated. cudarc formats driver errors as `DriverError(CUDA_ERROR_*, ..)`,
/// and Candle and izwi's own kernels carry that text through `Error`.
const STICKY_CUDA_ERRORS: &[&str] = &[
    "CUDA_ERROR_ILLEGAL_ADDRESS",
    "CUDA_ERROR_LAUNCH_FAILED",
    "CUDA_ERROR_ECC_UNCORRECTABLE",
    "CUDA_ERROR_HARDWARE_STACK_ERROR",
    "CUDA_ERROR_ILLEGAL_INSTRUCTION",
    "CUDA_ERROR_MISALIGNED_ADDRESS",
    "CUDA_ERROR_INVALID_ADDRESS_SPACE",
    "CUDA_ERROR_INVALID_PC",
    "CUDA_ERROR_LAUNCH_TIMEOUT",
    "CUDA_ERROR_ASSERT",
];

/// Candle's Metal command pool reports a command buffer that finished with
/// `MTLCommandBufferStatusError` with this prefix.
const METAL_COMMAND_BUFFER_ERROR: &str = "command buffer had following error";

/// Classify an execution error as a fatal device fault. Returns a short label
/// for logs and the poison reason, or `None` for ordinary failures.
///
/// Only errors raised by device execution (`InferenceError`, `ModelError`)
/// qualify, so text echoed from a request can never latch the runtime.
/// Allocation failures stay on their own typed capacity path. CPU execution
/// has no device fault class.
pub(crate) fn fatal_device_fault(error: &Error) -> Option<&'static str> {
    match error {
        Error::InferenceError(message) | Error::ModelError(message) => {
            classify_device_message(message)
        }
        _ => None,
    }
}

/// [`fatal_device_fault`] for an error already rendered into a row output.
/// Rows carry `Error`'s display text, so the same kind gate applies through
/// its prefix.
pub(crate) fn fatal_device_fault_message(message: &str) -> Option<&'static str> {
    if message.starts_with("Inference error: ") || message.starts_with("Model error: ") {
        classify_device_message(message)
    } else {
        None
    }
}

fn classify_device_message(message: &str) -> Option<&'static str> {
    if let Some(code) = STICKY_CUDA_ERRORS
        .iter()
        .find(|code| contains_error_code(message, code))
    {
        return Some(code);
    }
    let normalized = message.to_ascii_lowercase();
    if normalized.contains(METAL_COMMAND_BUFFER_ERROR)
        && !normalized.contains("insufficient memory")
        && !normalized.contains("outofmemory")
        && !normalized.contains("out of memory")
    {
        return Some("metal command buffer error");
    }
    None
}

/// Match a whole `CUDA_ERROR_*` identifier so `CUDA_ERROR_LAUNCH_FAILED` does
/// not also match a longer name that merely shares its prefix.
fn contains_error_code(message: &str, code: &str) -> bool {
    message.match_indices(code).any(|(start, _)| {
        message[start + code.len()..]
            .chars()
            .next()
            .is_none_or(|next| !(next.is_ascii_alphanumeric() || next == '_'))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sticky_cuda_driver_errors_are_fatal_device_faults() {
        for code in STICKY_CUDA_ERRORS {
            let error = Error::InferenceError(format!(
                "DriverError({code}, \"device reported a sticky error\")"
            ));
            assert_eq!(fatal_device_fault(&error), Some(*code), "{code}");
        }
        assert_eq!(
            fatal_device_fault_message(
                "Inference error: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access was encountered\")"
            ),
            Some("CUDA_ERROR_ILLEGAL_ADDRESS")
        );
    }

    #[test]
    fn metal_command_buffer_errors_are_fatal_unless_they_are_oom() {
        let fault = Error::InferenceError(
            "Metal error Command buffer had following error: Caused GPU Address Fault Error (0000000b:kIOGPUCommandBufferCallbackErrorPageFault)".into(),
        );
        assert_eq!(
            fatal_device_fault(&fault),
            Some("metal command buffer error")
        );
        let oom = Error::InferenceError(
            "Metal error Command buffer had following error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".into(),
        );
        assert_eq!(fatal_device_fault(&oom), None);
    }

    #[test]
    fn ordinary_and_recoverable_errors_are_not_device_faults() {
        for error in [
            Error::InferenceError("shape mismatch in matmul".into()),
            Error::InferenceError(
                "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")".into(),
            ),
            Error::InferenceError(
                "DriverError(CUDA_ERROR_INVALID_VALUE, \"invalid argument\")".into(),
            ),
            Error::InferenceError("DriverError(CUDA_ERROR_LAUNCH_FAILED_EXTENDED, \"x\")".into()),
            Error::Overloaded("DriverError(CUDA_ERROR_ILLEGAL_ADDRESS)".into()),
            Error::InvalidInput("prompt mentions CUDA_ERROR_ILLEGAL_ADDRESS".into()),
        ] {
            assert_eq!(fatal_device_fault(&error), None, "{error}");
        }
        assert_eq!(
            fatal_device_fault_message("Invalid input: CUDA_ERROR_ILLEGAL_ADDRESS"),
            None
        );
    }
}
