use crate::error::{CliError, Result};
use crate::http;
use crate::style::{self, Theme};
use console::style;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

/// The model `izwi chat` uses when none is named and none is loaded: small
/// enough to download and run anywhere.
pub const DEFAULT_CHAT_MODEL: &str = "Qwen3.5-0.8B";

const CHAT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

pub struct ChatArgs {
    pub model: Option<String>,
    pub system: Option<String>,
    /// `Some` sends `enable_thinking`; `None` keeps the model's default.
    pub thinking: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    variant: String,
    status: String,
    #[serde(default)]
    download_progress: Option<f32>,
    #[serde(default)]
    route_capabilities: RouteCapabilities,
}

#[derive(Debug, Default, Deserialize)]
struct RouteCapabilities {
    #[serde(default)]
    openai_chat_completions: bool,
}

#[derive(Debug, Deserialize)]
struct ModelList {
    models: Vec<ModelEntry>,
}

/// One assistant turn: the answer, and the reasoning the server parsed out
/// of it.
#[derive(Debug, Default, PartialEq)]
struct AssistantReply {
    content: String,
    reasoning: Option<String>,
}

pub async fn execute(args: ChatArgs, server: &str, theme: &Theme) -> Result<()> {
    let client = http::client(Some(CHAT_REQUEST_TIMEOUT))?;
    let model = match args.model {
        Some(model) => model,
        None => default_chat_model(&client, server).await?,
    };
    let model = check_chat_model(&client, server, &model).await?;

    theme.print_banner();
    println!("{}", style(format!("Chat mode with '{}'", model)).bold());
    println!("Type your message and press Enter. Use /help for commands, /quit to exit.\n");

    let mut messages = initial_messages(args.system.as_deref());

    loop {
        let input = match dialoguer::Input::<String>::new()
            .with_prompt("You")
            .allow_empty(true)
            .interact_text()
        {
            Ok(input) => input,
            // Ctrl+D ends the session like /quit.
            Err(dialoguer::Error::IO(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                println!();
                break;
            }
            Err(error) => {
                return Err(CliError::Other(format!(
                    "Could not read your message: {error}"
                )))
            }
        };
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "/quit" || input == "/exit" || input == "/q" {
            println!("Goodbye!");
            break;
        }

        if input == "/help" {
            println!("Commands:");
            println!("  /quit, /exit, /q - Exit chat");
            println!("  /help - Show this help");
            println!("  /clear - Clear conversation history");
            continue;
        }

        if input == "/clear" {
            messages = initial_messages(args.system.as_deref());
            println!("Conversation history cleared.");
            continue;
        }

        messages.push(json!({ "role": "user", "content": input }));
        let request = chat_request_body(&model, &messages, args.thinking);

        let waiting = style::spinner("Thinking...");
        let reply = send_chat_request(&client, server, &request).await;
        waiting.finish_and_clear();

        match reply {
            Ok(reply) => {
                if let Some(reasoning) = &reply.reasoning {
                    println!("{}", style(reasoning).dim());
                }
                println!("{}: {}", style("Assistant").green().bold(), reply.content);
                // Only the answer goes back as history: the model's reasoning
                // is not part of the conversation.
                messages.push(json!({ "role": "assistant", "content": reply.content }));
            }
            Err(error) => {
                // The turn did not happen; the user can retry it.
                messages.pop();
                style::print_error(&error);
            }
        }

        println!();
    }

    Ok(())
}

fn initial_messages(system: Option<&str>) -> Vec<Value> {
    system
        .map(str::trim)
        .filter(|system| !system.is_empty())
        .map(|system| vec![json!({ "role": "system", "content": system })])
        .unwrap_or_default()
}

fn chat_request_body(model: &str, messages: &[Value], thinking: Option<bool>) -> Value {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
        // Reasoning comes back in `reasoning_content`, not in the answer.
        "reasoning_format": "parsed",
    });
    if let Some(thinking) = thinking {
        body["enable_thinking"] = json!(thinking);
    }
    body
}

async fn send_chat_request(
    client: &reqwest::Client,
    server: &str,
    body: &Value,
) -> Result<AssistantReply> {
    let response = client
        .post(format!("{}/v1/chat/completions", server))
        .json(body)
        .send()
        .await?;
    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }
    Ok(assistant_reply(&response.json::<Value>().await?))
}

fn assistant_reply(completion: &Value) -> AssistantReply {
    let message = &completion["choices"][0]["message"];
    let text = |field: &str| {
        message[field]
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    };
    AssistantReply {
        content: text("content").unwrap_or_default(),
        reasoning: text("reasoning_content"),
    }
}

/// The chat model the server has loaded, else [`DEFAULT_CHAT_MODEL`].
async fn default_chat_model(client: &reqwest::Client, server: &str) -> Result<String> {
    let response = client
        .get(format!("{}/v1/admin/models", server))
        .send()
        .await?;
    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }
    let list: ModelList = response.json().await?;
    Ok(loaded_chat_model(&list.models).unwrap_or_else(|| DEFAULT_CHAT_MODEL.to_string()))
}

fn loaded_chat_model(models: &[ModelEntry]) -> Option<String> {
    models
        .iter()
        .find(|model| model.status == "ready" && model.route_capabilities.openai_chat_completions)
        .map(|model| model.variant.clone())
}

/// The model's catalog id, once the server confirms it can chat with it now.
async fn check_chat_model(client: &reqwest::Client, server: &str, model: &str) -> Result<String> {
    let response = client
        .get(format!("{}/v1/admin/models/{}", server, model))
        .send()
        .await?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(CliError::ModelNotFound(model.to_string()));
    }
    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }
    let entry: ModelEntry = response.json().await?;
    if !entry.route_capabilities.openai_chat_completions {
        return Err(CliError::InvalidInput(format!(
            "'{}' is not a chat model. Run `izwi list` to see the chat models.",
            entry.variant
        )));
    }
    match entry.status.as_str() {
        "not_downloaded" => Err(CliError::InvalidInput(format!(
            "'{0}' is not downloaded. Download it first: izwi pull {0}",
            entry.variant
        ))),
        "downloading" => Err(CliError::InvalidInput(format!(
            "'{}' is still downloading ({:.0}%). Try again when `izwi models progress {}` shows it downloaded.",
            entry.variant,
            entry.download_progress.unwrap_or(0.0),
            entry.variant
        ))),
        _ => Ok(entry.variant),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(variant: &str, status: &str, chat: bool) -> ModelEntry {
        ModelEntry {
            variant: variant.to_string(),
            status: status.to_string(),
            download_progress: None,
            route_capabilities: RouteCapabilities {
                openai_chat_completions: chat,
            },
        }
    }

    #[test]
    fn the_default_model_is_a_loaded_chat_model_first() {
        let models = vec![
            entry("Kokoro-82M", "ready", false),
            entry("Qwen3.5-4B", "downloaded", true),
            entry("LFM2.5-1.2B-Instruct-GGUF", "ready", true),
        ];
        assert_eq!(
            loaded_chat_model(&models).as_deref(),
            Some("LFM2.5-1.2B-Instruct-GGUF")
        );
        assert_eq!(loaded_chat_model(&models[..2]), None);
        assert!(izwi_core::parse_chat_model_variant(Some(DEFAULT_CHAT_MODEL)).is_ok());
    }

    #[test]
    fn requests_ask_for_parsed_reasoning_and_send_thinking_only_when_set() {
        let messages = initial_messages(Some("Be brief."));
        let body = chat_request_body("Qwen3.5-0.8B", &messages, None);
        assert_eq!(body["reasoning_format"], "parsed");
        assert!(body.get("enable_thinking").is_none());
        assert_eq!(body["messages"][0]["role"], "system");
        let body = chat_request_body("Qwen3.5-0.8B", &[], Some(false));
        assert_eq!(body["enable_thinking"], false);
        assert!(initial_messages(Some("  ")).is_empty());
    }

    #[test]
    fn reasoning_is_kept_apart_from_the_answer() {
        let completion = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "The user greets me.\n",
                    "content": "\nHello!"
                }
            }]
        });
        assert_eq!(
            assistant_reply(&completion),
            AssistantReply {
                content: "Hello!".to_string(),
                reasoning: Some("The user greets me.".to_string()),
            }
        );
        let plain =
            json!({ "choices": [{ "message": { "content": "Hi", "reasoning_content": "" } }] });
        assert_eq!(assistant_reply(&plain).reasoning, None);
    }
}
