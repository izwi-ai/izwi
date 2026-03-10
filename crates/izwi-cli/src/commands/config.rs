use crate::config::Config;
use crate::error::{CliError, Result};
use crate::style::Theme;
use crate::ConfigCommands;
use console::style;
use std::path::PathBuf;

pub async fn execute(
    command: ConfigCommands,
    config_path_override: Option<&PathBuf>,
    theme: &Theme,
) -> Result<()> {
    let config_path = get_config_path(config_path_override)?;

    match command {
        ConfigCommands::Show => show_config(&config_path, theme).await,
        ConfigCommands::Set { key, value } => set_config(&config_path, &key, &value, theme).await,
        ConfigCommands::Get { key } => get_config(&config_path, &key, theme).await,
        ConfigCommands::Edit => edit_config(&config_path, theme).await,
        ConfigCommands::Reset { yes } => reset_config(&config_path, yes, theme).await,
        ConfigCommands::Path => {
            println!("{}", config_path.display());
            Ok(())
        }
    }
}

fn get_config_path(override_path: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.clone());
    }

    let config_dir = dirs::config_dir()
        .ok_or_else(|| CliError::ConfigError("Could not find config directory".to_string()))?;
    Ok(config_dir.join("izwi").join("config.toml"))
}

fn default_config_contents() -> Result<String> {
    let body = toml::to_string_pretty(&Config::default_template())
        .map_err(|e| CliError::ConfigError(e.to_string()))?;
    Ok(format!("# Izwi Configuration\n\n{body}"))
}

async fn show_config(path: &PathBuf, theme: &Theme) -> Result<()> {
    if !path.exists() {
        theme.info("No configuration file found. Using defaults.");
        println!("\nDefault configuration:");
        println!("{}", default_config_contents()?);
        return Ok(());
    }

    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| CliError::Io(e))?;

    println!("{}", style("Configuration:").bold());
    println!("  Path: {}", path.display());
    println!();
    println!("{}", content);
    Ok(())
}

async fn set_config(path: &PathBuf, key: &str, value: &str, theme: &Theme) -> Result<()> {
    let mut config = Config::load(Some(path)).map_err(|e| CliError::ConfigError(e.to_string()))?;
    config
        .set_value(key, value)
        .map_err(|e| CliError::ConfigError(e.to_string()))?;
    config
        .save(Some(path))
        .map_err(|e| CliError::ConfigError(e.to_string()))?;

    theme.success(&format!("Set {} = {}", key, value));
    Ok(())
}

async fn get_config(path: &PathBuf, key: &str, _theme: &Theme) -> Result<()> {
    let config = Config::load(Some(path)).map_err(|e| CliError::ConfigError(e.to_string()))?;

    if let Some(value) = config.get_value(key) {
        println!("{} = {}", key, value);
        return Ok(());
    }

    if let Some(default) = Config::default_value_for_key(key) {
        println!("{} not set (default {})", key, default);
        return Ok(());
    }

    Err(CliError::InvalidInput(format!(
        "Unsupported config key '{}'",
        key
    )))
}

async fn edit_config(path: &PathBuf, theme: &Theme) -> Result<()> {
    // Ensure file exists
    if !path.exists() {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| CliError::Io(e))?;
        }
        tokio::fs::write(path, default_config_contents()?)
            .await
            .map_err(|e| CliError::Io(e))?;
    }

    // Open in default editor
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let status = tokio::process::Command::new(&editor)
        .arg(path)
        .status()
        .await
        .map_err(|e| CliError::Other(format!("Failed to launch editor: {}", e)))?;

    if !status.success() {
        return Err(CliError::Other("Editor exited with error".to_string()));
    }

    theme.success("Configuration updated");
    Ok(())
}

async fn reset_config(path: &PathBuf, yes: bool, theme: &Theme) -> Result<()> {
    if !yes {
        println!("This will delete your configuration file.");
        let confirm = dialoguer::Confirm::new()
            .with_prompt("Are you sure?")
            .default(false)
            .interact()
            .map_err(|e| CliError::Other(e.to_string()))?;

        if !confirm {
            println!("Cancelled.");
            return Ok(());
        }
    }

    if path.exists() {
        tokio::fs::remove_file(path)
            .await
            .map_err(|e| CliError::Io(e))?;
    }

    theme.success("Configuration reset to defaults");
    Ok(())
}
