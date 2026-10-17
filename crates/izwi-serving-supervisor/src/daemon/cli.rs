//! The supervisor command line.

use izwi_serving_protocol::{DeviceId, WorkerId};
use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use super::error::SupervisorError;

const MAX_CLI_ARGUMENTS: usize = 24;
const MAX_CPU_IDS: usize = 1024;
const MAX_DEVICE_DECLARATIONS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CliOptions {
    pub(crate) config: PathBuf,
    pub(crate) cpu_worker_binary: Option<PathBuf>,
    pub(crate) metal_worker_binary: Option<PathBuf>,
    pub(crate) cuda_worker_binary: Option<PathBuf>,
    pub(crate) cpu_ids: Vec<u16>,
    pub(crate) metal_devices: Vec<(DeviceId, u32)>,
    pub(crate) cuda_devices: Vec<(DeviceId, u32, u64)>,
    pub(crate) allocatable_host_memory_bytes: u64,
    pub(crate) validate_only: bool,
    pub(crate) canary_worker_id: Option<WorkerId>,
    pub(crate) rollout_plan: Option<PathBuf>,
    pub(crate) rollout_abort: bool,
    pub(crate) rollout_status: bool,
    pub(crate) clear_quarantine: Option<String>,
}

pub(crate) enum ParseOutcome {
    Run(Box<CliOptions>),
    Help,
}

impl CliOptions {
    pub(crate) fn parse(
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<ParseOutcome, SupervisorError> {
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        if arguments.len() > MAX_CLI_ARGUMENTS {
            return Err(SupervisorError::TooManyArguments);
        }
        if arguments.len() == 1 && arguments[0] == OsStr::new("--help") {
            return Ok(ParseOutcome::Help);
        }
        let mut config = None;
        let mut cpu_worker_binary = None;
        let mut metal_worker_binary = None;
        let mut cuda_worker_binary = None;
        let mut cpu_ids = None;
        let mut metal_devices = Vec::new();
        let mut cuda_devices = Vec::new();
        let mut allocatable_host_memory_bytes = None;
        let mut validate_only = false;
        let mut canary_worker_id = None;
        let mut rollout_plan = None;
        let mut rollout_abort = false;
        let mut rollout_status = false;
        let mut clear_quarantine = None;
        let mut index = 0;
        while index < arguments.len() {
            let name = arguments[index]
                .to_str()
                .ok_or(SupervisorError::NonUtf8OptionName)?;
            if name == "--validate-only" {
                if validate_only {
                    return Err(SupervisorError::DuplicateOption(name.to_string()));
                }
                validate_only = true;
                index += 1;
                continue;
            }
            if name == "--rollout-abort" {
                if rollout_abort {
                    return Err(SupervisorError::DuplicateOption(name.to_string()));
                }
                rollout_abort = true;
                index += 1;
                continue;
            }
            if name == "--rollout-status" {
                if rollout_status {
                    return Err(SupervisorError::DuplicateOption(name.to_string()));
                }
                rollout_status = true;
                index += 1;
                continue;
            }
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| SupervisorError::MissingOptionValue(name.to_string()))?;
            match name {
                "--config" => set_once(&mut config, PathBuf::from(value), name)?,
                "--cpu-worker-binary" => {
                    set_once(&mut cpu_worker_binary, PathBuf::from(value), name)?
                }
                "--metal-worker-binary" => {
                    set_once(&mut metal_worker_binary, PathBuf::from(value), name)?
                }
                "--cuda-worker-binary" => {
                    set_once(&mut cuda_worker_binary, PathBuf::from(value), name)?
                }
                "--metal-devices" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    metal_devices.extend(parse_metal_devices(value)?);
                }
                "--cuda-devices" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    cuda_devices.extend(parse_cuda_devices(value)?);
                }
                "--cpu-ids" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut cpu_ids, parse_cpu_ids(value)?, name)?;
                }
                "--allocatable-host-memory-bytes" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    let parsed = value
                        .parse::<u64>()
                        .ok()
                        .filter(|value| *value > 0)
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut allocatable_host_memory_bytes, parsed, name)?;
                }
                "--canary-worker-id" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    let id = WorkerId::new(value)
                        .map_err(|_| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut canary_worker_id, id, name)?;
                }
                "--rollout-plan" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut rollout_plan, PathBuf::from(value), name)?;
                }
                "--clear-quarantine" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut clear_quarantine, value.to_string(), name)?;
                }
                _ => return Err(SupervisorError::UnknownOption(name.to_string())),
            }
            index += 2;
        }
        let rollout_modes = usize::from(rollout_abort)
            + usize::from(rollout_status)
            + usize::from(rollout_plan.is_some());
        if rollout_modes > 1 {
            return Err(SupervisorError::InvalidOption(
                "--rollout-plan, --rollout-abort, and --rollout-status are mutually exclusive"
                    .to_string(),
            ));
        }
        if (rollout_abort || rollout_status || rollout_plan.is_some())
            && (validate_only || canary_worker_id.is_some())
        {
            return Err(SupervisorError::InvalidOption(
                "--rollout-* modes cannot be combined with --validate-only or --canary-worker-id"
                    .to_string(),
            ));
        }
        if clear_quarantine.is_some()
            && (rollout_modes > 0 || validate_only || canary_worker_id.is_some())
        {
            return Err(SupervisorError::InvalidOption(
                "--clear-quarantine is a command of its own".to_string(),
            ));
        }
        Ok(ParseOutcome::Run(Box::new(Self {
            config: config.ok_or(SupervisorError::MissingOption("--config"))?,
            cpu_worker_binary,
            metal_worker_binary,
            cuda_worker_binary,
            cpu_ids: cpu_ids.ok_or(SupervisorError::MissingOption("--cpu-ids"))?,
            metal_devices,
            cuda_devices,
            allocatable_host_memory_bytes: allocatable_host_memory_bytes.ok_or(
                SupervisorError::MissingOption("--allocatable-host-memory-bytes"),
            )?,
            validate_only,
            canary_worker_id,
            rollout_plan,
            rollout_abort,
            rollout_status,
            clear_quarantine,
        })))
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<(), SupervisorError> {
    if slot.replace(value).is_some() {
        Err(SupervisorError::DuplicateOption(name.to_string()))
    } else {
        Ok(())
    }
}

pub(crate) fn parse_cpu_ids(value: &str) -> Result<Vec<u16>, SupervisorError> {
    if value.is_empty() || value.len() > 16 * 1024 {
        return Err(SupervisorError::InvalidCpuIds);
    }
    let mut seen = BTreeSet::new();
    for item in value.split(',') {
        let id = item
            .parse::<u16>()
            .map_err(|_| SupervisorError::InvalidCpuIds)?;
        if !seen.insert(id) || seen.len() > MAX_CPU_IDS {
            return Err(SupervisorError::InvalidCpuIds);
        }
    }
    Ok(seen.into_iter().collect())
}

/// Declares Metal devices as `device_id@process_local_index` pairs. Devices
/// are declared by the operator (from `system_profiler`/`ioreg`) because the
/// supervisor never probes or initializes accelerators; the worker's own
/// assigned-device selection verifies the real identity at startup.
pub(crate) fn parse_metal_devices(value: &str) -> Result<Vec<(DeviceId, u32)>, SupervisorError> {
    const OPTION: &str = "--metal-devices";
    let invalid = |reason: String| SupervisorError::InvalidDeviceDeclaration {
        option: OPTION,
        reason,
    };
    if value.is_empty() || value.len() > 16 * 1024 {
        return Err(invalid("expected device_id@index".into()));
    }
    let mut seen = BTreeSet::new();
    let mut devices = Vec::new();
    for item in value.split(',') {
        let Some((id, index)) = item.split_once('@') else {
            return Err(invalid(format!("expected device_id@index, got {item:?}")));
        };
        let device_id =
            DeviceId::new(id).map_err(|_| invalid(format!("invalid device id {id:?}")))?;
        let index: u32 = index
            .parse()
            .map_err(|_| invalid(format!("invalid process-local index {index:?}")))?;
        if !seen.insert(device_id.clone()) || seen.len() > MAX_DEVICE_DECLARATIONS {
            return Err(invalid("duplicate or excessive device declarations".into()));
        }
        devices.push((device_id, index));
    }
    Ok(devices)
}

/// Declares CUDA devices as `device_uuid@host_device_index@total_memory_bytes`.
pub(crate) fn parse_cuda_devices(
    value: &str,
) -> Result<Vec<(DeviceId, u32, u64)>, SupervisorError> {
    const OPTION: &str = "--cuda-devices";
    let invalid = |reason: String| SupervisorError::InvalidDeviceDeclaration {
        option: OPTION,
        reason,
    };
    if value.is_empty() || value.len() > 16 * 1024 {
        return Err(invalid(
            "expected uuid@host_index@total_memory_bytes".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut devices = Vec::new();
    for item in value.split(',') {
        let fields: Vec<&str> = item.split('@').collect();
        let [uuid, host_index, total_memory_bytes] = fields.as_slice() else {
            return Err(invalid(format!(
                "expected uuid@host_index@total_memory_bytes, got {item:?}"
            )));
        };
        let device_uuid =
            DeviceId::new(*uuid).map_err(|_| invalid(format!("invalid device uuid {uuid:?}")))?;
        let host_index: u32 = host_index
            .parse()
            .map_err(|_| invalid(format!("invalid host device index {host_index:?}")))?;
        let total_memory_bytes: u64 = total_memory_bytes
            .parse()
            .map_err(|_| invalid(format!("invalid total memory bytes {total_memory_bytes:?}")))?;
        if total_memory_bytes == 0
            || !seen.insert(device_uuid.clone())
            || seen.len() > MAX_DEVICE_DECLARATIONS
        {
            return Err(invalid(
                "duplicate, zero-memory, or excessive device declarations".into(),
            ));
        }
        devices.push((device_uuid, host_index, total_memory_bytes));
    }
    Ok(devices)
}

pub(crate) fn print_usage() {
    eprintln!(
        "Usage: izwi-serving-supervisor \\\n  --config PATH \\\n  --cpu-ids 0,1,... \\\n  --allocatable-host-memory-bytes BYTES \\\n  [--cpu-worker-binary PATH] [--metal-worker-binary PATH] [--cuda-worker-binary PATH] \\\n  [--metal-devices id@index,...] [--cuda-devices uuid@host_index@total_memory_bytes,...]\n\n\
Every binary flavor referenced by the node configuration must be provided. Device\n\
declarations come from the operator (the supervisor never probes or initializes\n\
accelerators); each worker verifies its assigned device identity at startup and\n\
the supervisor admits nothing before that worker reports readiness."
    );
    eprintln!(
        "Optional: --validate-only resolves configuration and service credentials, prints bounded redacted diagnostics, and exits without acquiring locks or launching workers."
    );
    eprintln!(
        "Rollout (DS6): --rollout-plan PATH starts or resumes the rollout declared by the plan (blue-green when\n\
  both generations fit the node, otherwise recreate);\n\
  --rollout-abort restores the pre-rollout approvals and clears state; --rollout-status reports persisted rollout state.\n\
  All three require --config."
    );
    eprintln!(
        "Logs go to stderr as text, or one JSON object per line with IZWI_LOG_FORMAT=json; worker\n\
  stdout and stderr are relayed line by line with the worker id. IZWI_SUPERVISOR_METRICS_TEXTFILE=PATH\n\
  keeps supervisor counters and worker-state gauges there in the Prometheus text format."
    );
    eprintln!(
        "--clear-quarantine WORKER_ID|all asks the running supervisor (same --config) to relaunch a\n\
  quarantined worker at once; quarantined workers are otherwise retried every restart.quarantine_retry_ms."
    );
    #[cfg(unix)]
    eprintln!(
        "Send SIGUSR1 to a running supervisor for one bounded, redacted status snapshot on stderr; SIGUSR2 aborts a running rollout while it is still reversible; SIGHUP clears every quarantine."
    );
}
