//! The system allocator settings a model-serving process starts with.
//!
//! macOS's libmalloc keeps freed large blocks in a large-allocation cache as
//! dirty memory. A model load frees large host transients (a dequantized F32
//! vocabulary table alone can be a gigabyte), so with the cache on they stay
//! in the process footprint after the load, where they count against the
//! live free-memory sample that admits the next load, and under memory
//! pressure the kernel writes them to swap instead of dropping them.
//! Measured on a 16 GB M1 Pro (`measure_metal_load_footprint`), the cache
//! held 0.1-1.4 GB per load; with it off a process settles at its Metal
//! residency plus about 0.1 GB. libmalloc reads its settings once, at process
//! start, so they travel as environment variables: launchers set them on the
//! server or worker they spawn, and a server or worker started directly
//! re-executes itself with them ([`reexec_with_serving_allocator_env`]).
//!
//! A variable the operator already set, to any value, is left alone:
//! `MallocLargeCache=1` keeps the cache on.

use std::ffi::OsString;
use std::process::Command;

/// libmalloc's large-allocation cache switch (macOS).
pub const MALLOC_LARGE_CACHE_ENV: &str = "MallocLargeCache";

/// The allocator variables a model-serving process starts with on this
/// platform, as `(name, value)`: the large-allocation cache off on macOS,
/// nothing elsewhere.
pub fn serving_allocator_env() -> &'static [(&'static str, &'static str)] {
    if cfg!(target_os = "macos") {
        &[(MALLOC_LARGE_CACHE_ENV, "0")]
    } else {
        &[]
    }
}

/// The [`serving_allocator_env`] variables `current` (a lookup in the
/// spawning process's environment) does not already set.
pub fn missing_serving_allocator_env(
    current: impl Fn(&str) -> Option<OsString>,
) -> Vec<(&'static str, &'static str)> {
    serving_allocator_env()
        .iter()
        .copied()
        .filter(|(name, _)| current(name).is_none())
        .collect()
}

/// Sets the [`serving_allocator_env`] variables this process does not set on
/// `command`, a server or worker it is about to spawn. One this process sets
/// is inherited as it is, unless the command clears its environment.
pub fn apply_serving_allocator_env(command: &mut Command) -> &mut Command {
    for (name, value) in missing_serving_allocator_env(|name| std::env::var_os(name)) {
        command.env(name, value);
    }
    command
}

/// Re-executes the current binary, with its arguments, when this process
/// lacks a [`serving_allocator_env`] variable, so a server or worker started
/// without a launcher gets the same allocator settings. Call it first in
/// `main`, before any thread starts. The process ID, open descriptors (a
/// launcher's ownership pipe included) and working directory carry over.
///
/// Returns `Ok` when nothing was missing (always, off macOS). On success it
/// does not return; an `Err` means the exec failed and the caller carries on
/// with the default allocator.
pub fn reexec_with_serving_allocator_env() -> std::io::Result<()> {
    let missing = missing_serving_allocator_env(|name| std::env::var_os(name));
    if missing.is_empty() {
        return Ok(());
    }
    reexec_with(&missing)
}

#[cfg(unix)]
fn reexec_with(missing: &[(&str, &str)]) -> std::io::Result<()> {
    use std::os::unix::process::CommandExt;

    let mut args = std::env::args_os();
    let mut command = Command::new(std::env::current_exe()?);
    if let Some(arg0) = args.next() {
        command.arg0(arg0);
    }
    command.args(args);
    for (name, value) in missing {
        command.env(name, value);
    }
    Err(command.exec())
}

#[cfg(not(unix))]
fn reexec_with(_missing: &[(&str, &str)]) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macos_serving_processes_start_with_the_large_cache_off() {
        let expected: &[(&str, &str)] = if cfg!(target_os = "macos") {
            &[("MallocLargeCache", "0")]
        } else {
            &[]
        };
        assert_eq!(serving_allocator_env(), expected);
        assert_eq!(missing_serving_allocator_env(|_| None), expected);
    }

    #[test]
    fn an_operator_value_is_left_alone() {
        let operator = |name: &str| (name == MALLOC_LARGE_CACHE_ENV).then(|| OsString::from("1"));
        assert!(missing_serving_allocator_env(operator).is_empty());
    }

    #[test]
    fn a_spawned_command_gets_only_the_missing_variables() {
        let mut command = Command::new("true");
        apply_serving_allocator_env(&mut command);
        let set: Vec<_> = command
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(ToOwned::to_owned)))
            .collect();
        let expected: Vec<_> = missing_serving_allocator_env(|name| std::env::var_os(name))
            .into_iter()
            .map(|(name, value)| (OsString::from(name), Some(OsString::from(value))))
            .collect();
        assert_eq!(set, expected);
    }
}
