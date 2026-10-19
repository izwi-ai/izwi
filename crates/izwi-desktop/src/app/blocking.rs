/// Run blocking HTTP work on a dedicated thread with no Tokio runtime context.
///
/// `main` runs inside a Tokio runtime because the Aptabase plugin spawns and
/// flushes on it, and Tauri runs `setup` and synchronous commands on the main
/// thread. `reqwest::blocking` builds and drops its own runtime, which panics
/// ("Cannot drop a runtime in a context where blocking is not allowed") when
/// called from inside another runtime's context, so blocking requests made
/// from those places must go through this helper.
pub fn run_off_runtime<T, F>(work: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    match std::thread::spawn(work).join() {
        Ok(value) => value,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[cfg(test)]
mod tests {
    use super::run_off_runtime;

    #[tokio::test(flavor = "multi_thread")]
    async fn blocking_client_works_inside_a_runtime_context() {
        // Building a reqwest blocking client directly here would panic.
        let built = run_off_runtime(|| reqwest::blocking::Client::builder().build().is_ok());
        assert!(built);
    }
}
