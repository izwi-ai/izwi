//! CUDA runtime packaging diagnostics for unified installers.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct CudaRuntimeDiagnostics {
    pub current_binary_cuda_compiled: bool,
    pub private_runtime_active: bool,
    pub private_runtime_path: Option<PathBuf>,
    pub private_runtime_packaged: bool,
    pub runtime_libraries_available: bool,
    pub missing_runtime_libraries: Vec<String>,
    pub driver_available: bool,
    pub device_usable: Option<bool>,
    pub search_paths: Vec<PathBuf>,
    pub notes: Vec<String>,
}

impl CudaRuntimeDiagnostics {
    pub fn detect(binary_name: &str) -> Self {
        let private_runtime_path = resolve_private_cuda_runtime(binary_name);
        let private_runtime_packaged = private_runtime_path.is_some();
        let search_paths = cuda_library_search_paths(private_runtime_path.as_deref());
        let missing_runtime_libraries = missing_runtime_libraries(&search_paths);
        let runtime_libraries_available =
            !RUNTIME_LIBRARIES.is_empty() && missing_runtime_libraries.is_empty();
        let driver_available = cuda_driver_available(&search_paths);
        let current_binary_cuda_compiled = cfg!(feature = "cuda");
        let private_runtime_active = private_cuda_runtime_active();
        let device_usable = cuda_device_usable();

        let mut notes = Vec::new();
        if current_binary_cuda_compiled {
            notes.push("current binary is compiled with CUDA support".to_string());
        } else {
            notes.push("current binary is CPU-safe and not directly CUDA-linked".to_string());
        }
        if private_runtime_packaged {
            notes.push("private CUDA runtime binary is packaged".to_string());
        } else {
            notes.push("private CUDA runtime binary was not found".to_string());
        }
        if !runtime_libraries_available {
            if missing_runtime_libraries.is_empty() {
                notes.push("CUDA runtime libraries are not available".to_string());
            } else {
                notes.push(format!(
                    "missing CUDA runtime libraries: {}",
                    missing_runtime_libraries.join(", ")
                ));
            }
        }
        if !driver_available {
            notes.push(cuda_driver_missing_note());
        }

        Self {
            current_binary_cuda_compiled,
            private_runtime_active,
            private_runtime_path,
            private_runtime_packaged,
            runtime_libraries_available,
            missing_runtime_libraries,
            driver_available,
            device_usable,
            search_paths,
            notes,
        }
    }

    pub fn can_start_private_runtime(&self) -> bool {
        self.private_runtime_packaged && self.runtime_libraries_available && self.driver_available
    }
}

#[derive(Debug, Clone, Copy)]
struct LibraryFamily {
    label: &'static str,
    prefixes: &'static [&'static str],
}

const CUDA_RUNTIME_ACTIVE_ENV: &str = "IZWI_CUDA_RUNTIME_ACTIVE";
const CUDA_RUNTIME_BINARY_ENV: &str = "IZWI_CUDA_RUNTIME_BINARY";

#[cfg(any(target_os = "linux", target_os = "macos"))]
const PATH_LIST_SEPARATOR: char = ':';
#[cfg(target_os = "windows")]
const PATH_LIST_SEPARATOR: char = ';';
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const PATH_LIST_SEPARATOR: char = ':';

#[cfg(target_os = "linux")]
const RUNTIME_LIBRARIES: &[LibraryFamily] = &[
    LibraryFamily {
        label: "cudart",
        prefixes: &["libcudart.so"],
    },
    LibraryFamily {
        label: "cublas",
        prefixes: &["libcublas.so"],
    },
    LibraryFamily {
        label: "cublasLt",
        prefixes: &["libcublaslt.so"],
    },
    LibraryFamily {
        label: "curand",
        prefixes: &["libcurand.so"],
    },
    LibraryFamily {
        label: "nvrtc",
        prefixes: &["libnvrtc.so"],
    },
    LibraryFamily {
        label: "nvrtc-builtins",
        prefixes: &["libnvrtc-builtins.so"],
    },
];

#[cfg(target_os = "windows")]
const RUNTIME_LIBRARIES: &[LibraryFamily] = &[
    LibraryFamily {
        label: "cudart",
        prefixes: &["cudart64_"],
    },
    LibraryFamily {
        label: "cublas",
        prefixes: &["cublas64_"],
    },
    LibraryFamily {
        label: "cublasLt",
        prefixes: &["cublaslt64_"],
    },
    LibraryFamily {
        label: "curand",
        prefixes: &["curand64_"],
    },
    LibraryFamily {
        label: "nvrtc",
        prefixes: &["nvrtc64_"],
    },
    LibraryFamily {
        label: "nvrtc-builtins",
        prefixes: &["nvrtc-builtins64_"],
    },
];

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
const RUNTIME_LIBRARIES: &[LibraryFamily] = &[];

pub fn private_cuda_runtime_active() -> bool {
    std::env::var(CUDA_RUNTIME_ACTIVE_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false)
}

pub fn private_cuda_runtime_env_key() -> &'static str {
    CUDA_RUNTIME_ACTIVE_ENV
}

pub fn private_cuda_runtime_binary_env_key() -> &'static str {
    CUDA_RUNTIME_BINARY_ENV
}

pub fn resolve_private_cuda_runtime(binary_name: &str) -> Option<PathBuf> {
    private_cuda_runtime_candidates(binary_name)
        .into_iter()
        .find(|candidate| candidate.is_file())
}

pub fn private_cuda_runtime_candidates(binary_name: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(path) = std::env::var_os(CUDA_RUNTIME_BINARY_ENV) {
        candidates.push(PathBuf::from(path));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("runtime").join("cuda").join(binary_name));
            candidates.push(
                parent
                    .join("..")
                    .join("lib")
                    .join("izwi")
                    .join("runtime")
                    .join("cuda")
                    .join(binary_name),
            );
        }
    }

    #[cfg(target_os = "linux")]
    candidates.push(
        PathBuf::from("/usr/lib")
            .join("izwi")
            .join("runtime")
            .join("cuda")
            .join(binary_name),
    );

    dedupe_paths(candidates)
}

pub fn cuda_library_search_paths(private_runtime_path: Option<&Path>) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Some(runtime_path) = private_runtime_path {
        if let Some(parent) = runtime_path.parent() {
            paths.push(parent.to_path_buf());
            paths.push(parent.join("lib"));
        }
    }

    #[cfg(target_os = "linux")]
    {
        extend_env_paths(&mut paths, "LD_LIBRARY_PATH");
        extend_cuda_home_paths(&mut paths, "lib64");
        paths.push(PathBuf::from("/usr/local/cuda/lib64"));
        paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
        paths.push(PathBuf::from("/usr/lib/wsl/lib"));
        paths.push(PathBuf::from("/usr/lib64"));
        paths.push(PathBuf::from("/usr/lib"));
    }

    #[cfg(target_os = "windows")]
    {
        extend_env_paths(&mut paths, "PATH");
        extend_cuda_home_paths(&mut paths, "bin");
        if let Some(system_root) = std::env::var_os("SystemRoot") {
            paths.push(PathBuf::from(system_root).join("System32"));
        }
    }

    dedupe_paths(paths)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn prepend_cuda_loader_paths(command: &mut std::process::Command, private_runtime_path: &Path) {
    let Some(runtime_dir) = private_runtime_path.parent() else {
        return;
    };

    #[cfg(target_os = "linux")]
    {
        let lib_dir = runtime_dir.join("lib");
        prepend_env_paths(command, "LD_LIBRARY_PATH", [runtime_dir, lib_dir.as_path()]);
    }

    #[cfg(target_os = "windows")]
    {
        prepend_env_paths(command, "PATH", [runtime_dir]);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn prepend_cuda_loader_paths(
    _command: &mut std::process::Command,
    _private_runtime_path: &Path,
) {
}

fn missing_runtime_libraries(search_paths: &[PathBuf]) -> Vec<String> {
    RUNTIME_LIBRARIES
        .iter()
        .filter(|family| !library_family_present(search_paths, family))
        .map(|family| family.label.to_string())
        .collect()
}

#[cfg(target_os = "linux")]
fn cuda_driver_available(search_paths: &[PathBuf]) -> bool {
    file_with_prefix_present(search_paths, &["libcuda.so.1", "libcuda.so"])
}

#[cfg(target_os = "windows")]
fn cuda_driver_available(search_paths: &[PathBuf]) -> bool {
    file_with_prefix_present(search_paths, &["nvcuda.dll"])
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn cuda_driver_available(_search_paths: &[PathBuf]) -> bool {
    false
}

fn cuda_driver_missing_note() -> String {
    #[cfg(target_os = "windows")]
    {
        return "NVIDIA driver DLL nvcuda.dll was not found".to_string();
    }

    #[cfg(target_os = "linux")]
    {
        return "NVIDIA driver library libcuda.so.1 was not found".to_string();
    }

    #[allow(unreachable_code)]
    "CUDA driver is not available on this platform".to_string()
}

fn library_family_present(search_paths: &[PathBuf], family: &LibraryFamily) -> bool {
    file_with_prefix_present(search_paths, family.prefixes)
}

fn file_with_prefix_present(search_paths: &[PathBuf], prefixes: &[&str]) -> bool {
    let normalized_prefixes = prefixes
        .iter()
        .map(|prefix| prefix.to_ascii_lowercase())
        .collect::<Vec<_>>();

    search_paths.iter().any(|dir| {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };

        entries.flatten().any(|entry| {
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            normalized_prefixes
                .iter()
                .any(|prefix| name.starts_with(prefix))
        })
    })
}

fn extend_env_paths(paths: &mut Vec<PathBuf>, key: &str) {
    let Some(raw) = std::env::var_os(key) else {
        return;
    };

    paths.extend(
        raw.to_string_lossy()
            .split(PATH_LIST_SEPARATOR)
            .filter(|entry| !entry.trim().is_empty())
            .map(PathBuf::from),
    );
}

fn extend_cuda_home_paths(paths: &mut Vec<PathBuf>, child: &str) {
    for key in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(path) = std::env::var_os(key) {
            paths.push(PathBuf::from(path).join(child));
        }
    }
}

fn prepend_env_paths<'a, I>(command: &mut std::process::Command, key: &str, paths: I)
where
    I: IntoIterator<Item = &'a Path>,
{
    let mut combined = paths
        .into_iter()
        .filter(|path| path.exists())
        .map(PathBuf::from)
        .collect::<Vec<_>>();

    if let Some(existing) = std::env::var_os(key) {
        combined.extend(std::env::split_paths(&existing));
    }

    if let Ok(joined) = std::env::join_paths(combined) {
        command.env(key, joined);
    }
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut deduped = Vec::new();

    for path in paths {
        let key = path.to_string_lossy().to_string();
        if seen.insert(key) {
            deduped.push(path);
        }
    }

    deduped
}

#[cfg(feature = "cuda")]
fn cuda_device_usable() -> Option<bool> {
    use super::device::{DeviceKind, DeviceSelector};
    use super::types::BackendPreference;

    Some(
        DeviceSelector::detect_for_preference(BackendPreference::Cuda)
            .map(|profile| profile.kind == DeviceKind::Cuda)
            .unwrap_or(false),
    )
}

#[cfg(not(feature = "cuda"))]
fn cuda_device_usable() -> Option<bool> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env_test_lock;

    #[test]
    fn env_override_is_first_private_runtime_candidate() {
        let _guard = env_test_lock().lock().expect("env lock poisoned");
        std::env::set_var(CUDA_RUNTIME_BINARY_ENV, "/tmp/izwi-cuda/izwi-server");

        let candidates = private_cuda_runtime_candidates("izwi-server");
        assert_eq!(
            candidates.first(),
            Some(&PathBuf::from("/tmp/izwi-cuda/izwi-server"))
        );

        std::env::remove_var(CUDA_RUNTIME_BINARY_ENV);
    }

    #[test]
    fn active_env_accepts_truthy_values() {
        let _guard = env_test_lock().lock().expect("env lock poisoned");
        std::env::set_var(CUDA_RUNTIME_ACTIVE_ENV, "1");
        assert!(private_cuda_runtime_active());

        std::env::set_var(CUDA_RUNTIME_ACTIVE_ENV, "false");
        assert!(!private_cuda_runtime_active());

        std::env::remove_var(CUDA_RUNTIME_ACTIVE_ENV);
    }
}
