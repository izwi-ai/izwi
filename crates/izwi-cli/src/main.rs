//! Izwi CLI - World-class command-line interface for audio inference
//!
//! Inspired by vLLM, SGlang, Ollama, and llama.cpp CLIs
// CLI dispatch and benchmark commands deliberately mirror their complete set
// of command-line options at the boundary.
#![allow(clippy::too_many_arguments)]

use clap::Parser;
use std::process::ExitCode;

mod app;
mod commands;
mod config;
mod error;
mod http;
mod style;
mod utils;

pub use app::cli::{
    AudioFormat, Backend, BenchCommands, Cli, Commands, ConfigCommands, LogFormat, ModelCommands,
    OutputFormat, ServeMode, Shell, TranscriptFormat,
};
use style::Theme;

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let no_color = cli.no_color || std::env::var_os("NO_COLOR").is_some();

    let theme = if no_color {
        console::set_colors_enabled(false);
        console::set_colors_enabled_stderr(false);
        Theme::no_color()
    } else {
        Theme::default()
    };

    if cli.verbose {
        tracing_subscriber::fmt::init();
    }

    match app::run(cli, theme).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            style::print_error(&error);
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod test_support {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("environment lock poisoned")
    }
}
