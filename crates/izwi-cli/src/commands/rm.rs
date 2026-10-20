use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use console::style;

pub async fn execute(model: String, yes: bool, server: &str, theme: &Theme) -> Result<()> {
    if !yes {
        println!(
            "This will permanently delete model '{}' and all its files.",
            style(&model).red()
        );
        println!(
            "{}",
            style("Warning: This action cannot be undone!").red().bold()
        );

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

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .delete(format!("{}/v1/admin/models/{}", server, model))
        .send()
        .await?;

    if response.status().as_u16() == 404 {
        return Err(CliError::ModelNotFound(model));
    }

    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }

    theme.success(&format!("Model '{}' removed successfully", model));
    Ok(())
}
