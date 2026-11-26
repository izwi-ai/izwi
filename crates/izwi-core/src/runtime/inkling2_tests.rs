//! Inkling 2 through the runtime on the synthetic checkpoint
//! (`inkling2::synthetic`): download check, load admission from the header,
//! state allocation (paged keys and values plus the convolution ring), and
//! chat requests alone and concurrently.

use super::*;
use crate::config::EngineConfig;
use crate::model::ModelVariant;
use crate::models::architectures::inkling2::synthetic::write_checkpoint;
use crate::models::shared::chat::{ChatMessage, ChatRole};

const VARIANT: ModelVariant = ModelVariant::Inkling2Mini;
/// The synthetic model writes random tokens until a turn-ending one or its
/// budget.
const MAX_TOKENS: usize = 12;

struct ModelsDir(std::path::PathBuf);

impl Drop for ModelsDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn models_dir(tag: &str) -> ModelsDir {
    let root = std::env::temp_dir().join(format!(
        "izwi-inkling2-runtime-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    write_checkpoint(&root.join(VARIANT.dir_name()));
    ModelsDir(root)
}

async fn runtime_with(models: &ModelsDir, rows: usize) -> RuntimeService {
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models.0.clone(),
        backend: crate::backends::BackendPreference::Cpu,
        max_batch_size: crate::config::BatchSizePreference::Fixed(
            std::num::NonZeroUsize::new(rows).unwrap(),
        ),
        ..Default::default()
    })
    .expect("runtime");
    Box::pin(runtime.load_model(VARIANT))
        .await
        .expect("load the synthetic checkpoint");
    runtime
}

fn user(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: ChatRole::User,
        content: text.to_string(),
    }]
}

async fn chat(runtime: &RuntimeService, text: &str) -> crate::error::Result<ChatGeneration> {
    // Greedy, so a reply can be compared with its repeat.
    let params = crate::engine::GenerationParams {
        temperature: 0.0,
        max_tokens: MAX_TOKENS,
        ..Default::default()
    };
    Box::pin(runtime.chat_generate_with_generation_params(VARIANT, user(text), params)).await
}

async fn synthetic_chats_run_alone_and_concurrently_alike_body() {
    let models = models_dir("rows");
    let runtime = runtime_with(&models, 3).await;
    let diagnostics = runtime
        .loaded_model_diagnostics()
        .await
        .into_iter()
        .find_map(|loaded| loaded.family_diagnostics)
        .expect("the model's diagnostics");
    assert_eq!(diagnostics["family"], "inkling2_chat");
    assert_eq!(diagnostics["layer_applications"], 16);

    let prompts = [
        "Hello there",
        "What is two plus two?",
        "Name a colour, please.",
    ];
    let mut solo = Vec::new();
    for prompt in prompts {
        let generation = chat(&runtime, prompt).await.expect("chat");
        // The prompt is the effort line and the user message.
        assert!(generation.prompt_tokens > prompt.len());
        assert!(generation.tokens_generated <= MAX_TOKENS);
        solo.push((generation.text, generation.tokens_generated));
    }
    eprintln!("synthetic replies: {solo:?}");
    // The prompts lead to different replies, so the model reads its input.
    assert!(solo.iter().any(|reply| reply != &solo[0]));
    // Greedy decoding repeats exactly.
    let again = chat(&runtime, prompts[0]).await.expect("again");
    assert_eq!((again.text, again.tokens_generated), solo[0]);
    let concurrent =
        futures::future::join_all(prompts.iter().map(|prompt| chat(&runtime, prompt))).await;
    for (result, expected) in concurrent.into_iter().zip(&solo) {
        let generation = result.expect("concurrent chat");
        assert_eq!(&(generation.text, generation.tokens_generated), expected);
    }
    // A long prompt prefills in several spans.
    let long = "word ".repeat(120);
    let generation = chat(&runtime, &long).await.expect("long prompt");
    assert!(generation.prompt_tokens > 600);
    Box::pin(runtime.unload_model(VARIANT))
        .await
        .expect("unload");
    assert!(runtime.loaded_model_diagnostics().await.is_empty());
}

#[test]
fn synthetic_chats_run_alone_and_concurrently_alike() {
    crate::block_on_large_stack(4, synthetic_chats_run_alone_and_concurrently_alike_body);
}
