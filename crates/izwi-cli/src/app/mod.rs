pub mod cli;

use crate::commands;
use crate::error::Result;
use crate::style::Theme;
use izwi_core::{ServeRuntimeConfig, ServeRuntimeConfigOverrides};
use std::path::PathBuf;

use self::cli::{Backend, Cli, Commands, ServeMode};

pub async fn run(cli: Cli, theme: Theme) -> Result<()> {
    let Cli {
        command,
        config,
        server,
        output_format,
        quiet,
        ..
    } = cli;

    match command {
        Commands::Serve {
            mode,
            host,
            port,
            models_dir,
            max_batch_size,
            backend,
            threads,
            max_concurrent,
            timeout,
            log_level,
            dev,
            cors,
            no_ui,
        } => {
            commands::serve::execute(build_serve_args(
                config.as_ref(),
                mode,
                host,
                port,
                models_dir,
                max_batch_size,
                backend,
                threads,
                max_concurrent,
                timeout,
                log_level,
                dev,
                cors,
                no_ui,
            )?)
            .await?;
        }
        Commands::Models { command } => {
            commands::models::execute(command, &server, output_format, quiet).await?;
        }
        Commands::Pull { model, force, yes } => {
            commands::pull::execute(model, force, yes, &server, &theme).await?;
        }
        Commands::Rm { model, yes } => {
            commands::rm::execute(model, yes, &server, &theme).await?;
        }
        Commands::List { local, detailed } => {
            commands::list::execute(local, detailed, &server, output_format).await?;
        }
        Commands::Tts {
            text,
            model,
            speaker,
            output,
            format,
            speed,
            temperature,
            stream,
            play,
        } => {
            commands::tts::execute(
                commands::tts::TtsArgs {
                    text,
                    model,
                    speaker,
                    output,
                    format,
                    speed,
                    temperature,
                    stream,
                    play,
                },
                &server,
                &theme,
            )
            .await?;
        }
        Commands::Transcribe {
            file,
            model,
            language,
            format,
            output,
            word_timestamps,
        } => {
            commands::transcribe::execute(
                commands::transcribe::TranscribeArgs {
                    file,
                    model,
                    language,
                    format,
                    output,
                    word_timestamps,
                },
                &server,
            )
            .await?;
        }
        Commands::Chat {
            model,
            system,
            voice,
        } => {
            commands::chat::execute(
                commands::chat::ChatArgs {
                    model,
                    system,
                    voice,
                },
                &server,
                &theme,
            )
            .await?;
        }
        Commands::Diarize {
            file,
            model,
            num_speakers,
            format,
            output,
            transcribe,
            asr_model,
        } => {
            commands::diarize::execute(
                commands::diarize::DiarizeArgs {
                    file,
                    model,
                    num_speakers,
                    format,
                    output,
                    transcribe,
                    asr_model,
                },
                &server,
            )
            .await?;
        }
        Commands::Align {
            file,
            text,
            model,
            format,
            output,
        } => {
            commands::align::execute(
                commands::align::AlignArgs {
                    file,
                    text,
                    model,
                    format,
                    output,
                },
                &server,
            )
            .await?;
        }
        Commands::Bench { command } => {
            commands::bench::execute(command, &server, &theme).await?;
        }
        Commands::Status { detailed, watch } => {
            commands::status::execute(detailed, watch, &server, &theme).await?;
        }
        Commands::Version { full } => {
            commands::version::execute(full, &theme);
        }
        Commands::Config { command } => {
            commands::config::execute(command, config.as_ref(), &theme).await?;
        }
        Commands::Completions { shell } => {
            commands::completions::execute(shell);
        }
    }

    Ok(())
}

fn build_serve_args(
    config_path: Option<&PathBuf>,
    mode: ServeMode,
    host: Option<String>,
    port: Option<u16>,
    models_dir: Option<std::path::PathBuf>,
    max_batch_size: Option<usize>,
    backend: Option<Backend>,
    threads: Option<usize>,
    max_concurrent: Option<usize>,
    timeout: Option<u64>,
    log_level: String,
    dev: bool,
    cors: bool,
    no_ui: bool,
) -> Result<commands::serve::ServeArgs> {
    let cli_overrides = ServeRuntimeConfigOverrides {
        host,
        port,
        models_dir,
        backend: backend.as_ref().map(Backend::as_preference),
        max_batch_size,
        num_threads: threads,
        max_concurrent_requests: max_concurrent,
        request_timeout_secs: timeout,
        cors_enabled: cors.then_some(true),
        ui_enabled: no_ui.then_some(false),
        ..ServeRuntimeConfigOverrides::default()
    };
    let runtime = resolve_serve_runtime_config(config_path, &cli_overrides)?;

    Ok(commands::serve::ServeArgs {
        mode,
        runtime,
        log_level,
        dev,
    })
}

fn resolve_serve_runtime_config(
    config_path: Option<&PathBuf>,
    cli_overrides: &ServeRuntimeConfigOverrides,
) -> Result<ServeRuntimeConfig> {
    let file_config = crate::config::Config::load(config_path)?;
    let config_overrides = file_config.serve_runtime_overrides();
    let env_overrides = ServeRuntimeConfigOverrides::from_env();

    Ok(ServeRuntimeConfig::from_sources(
        &config_overrides,
        &env_overrides,
        cli_overrides,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::backends::BackendPreference;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("environment lock poisoned")
    }

    fn clear_serve_env() {
        std::env::remove_var(izwi_core::serve_runtime::ENV_HOST);
        std::env::remove_var(izwi_core::serve_runtime::ENV_PORT);
        std::env::remove_var(izwi_core::serve_runtime::ENV_MODELS_DIR);
        std::env::remove_var(izwi_core::serve_runtime::ENV_BACKEND);
        std::env::remove_var(izwi_core::serve_runtime::ENV_MAX_BATCH_SIZE);
        std::env::remove_var(izwi_core::serve_runtime::ENV_NUM_THREADS);
        std::env::remove_var(izwi_core::serve_runtime::ENV_MAX_CONCURRENT);
        std::env::remove_var(izwi_core::serve_runtime::ENV_TIMEOUT);
        std::env::remove_var(izwi_core::serve_runtime::ENV_CORS);
        std::env::remove_var(izwi_core::serve_runtime::ENV_CORS_ORIGINS);
        std::env::remove_var(izwi_core::serve_runtime::ENV_NO_UI);
        std::env::remove_var(izwi_core::serve_runtime::ENV_UI_DIR);
        std::env::remove_var(izwi_core::serve_runtime::LEGACY_ENV_MAX_CONCURRENT[0]);
        std::env::remove_var(izwi_core::serve_runtime::LEGACY_ENV_TIMEOUT[0]);
    }

    #[test]
    fn build_serve_args_resolves_cli_env_then_config() {
        let _guard = env_lock();
        clear_serve_env();

        let dir = tempdir().expect("temp dir should be created");
        let config_path = dir.path().join("config.toml");
        let mut config = crate::config::Config::default();
        config
            .set_value("server.host", "config-host")
            .expect("host should be set");
        config
            .set_value("runtime.max_batch_size", "4")
            .expect("batch size should be set");
        config
            .set_value("ui.enabled", "false")
            .expect("ui.enabled should be set");
        config
            .save(Some(&config_path))
            .expect("config should be saved");

        std::env::set_var(izwi_core::serve_runtime::ENV_MAX_BATCH_SIZE, "5");
        std::env::set_var(izwi_core::serve_runtime::ENV_TIMEOUT, "600");

        let args = build_serve_args(
            Some(&config_path),
            ServeMode::Server,
            Some("cli-host".to_string()),
            None,
            None,
            None,
            Some(Backend::Cuda),
            None,
            None,
            None,
            "info".to_string(),
            false,
            true,
            false,
        )
        .expect("serve args should resolve");

        assert_eq!(args.runtime.host, "cli-host");
        assert_eq!(args.runtime.max_batch_size, 5);
        assert_eq!(args.runtime.request_timeout_secs, 600);
        assert_eq!(args.runtime.backend, BackendPreference::Cuda);
        assert!(args.runtime.cors_enabled);
        assert!(!args.runtime.ui_enabled);
        assert!(matches!(args.mode, ServeMode::Server));
        clear_serve_env();
    }

    #[test]
    fn build_serve_args_honors_legacy_runtime_env_aliases() {
        let _guard = env_lock();
        clear_serve_env();

        std::env::set_var(izwi_core::serve_runtime::LEGACY_ENV_MAX_CONCURRENT[0], "45");
        std::env::set_var(izwi_core::serve_runtime::LEGACY_ENV_TIMEOUT[0], "721");

        let args = build_serve_args(
            None,
            ServeMode::Server,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "warn".to_string(),
            false,
            false,
            false,
        )
        .expect("serve args should resolve");

        assert_eq!(args.runtime.max_concurrent_requests, 45);
        assert_eq!(args.runtime.request_timeout_secs, 721);
        clear_serve_env();
    }
}
