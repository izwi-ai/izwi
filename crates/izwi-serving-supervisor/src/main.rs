//! `izwi-serving-supervisor`: the per-node worker supervisor. The process
//! itself lives in the library's `daemon` module.

#[tokio::main]
async fn main() -> Result<(), izwi_serving_supervisor::daemon::SupervisorError> {
    izwi_serving_supervisor::daemon::run_from_cli().await
}
