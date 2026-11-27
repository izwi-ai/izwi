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

/// A PNG of `width x height` pixels as a data URL; `seed` varies its pixels,
/// and every other pixel is half transparent.
fn png_data_url(width: u32, height: u32, seed: u8) -> String {
    use base64::Engine;
    let image = image::RgbaImage::from_fn(width, height, |x, y| {
        let value = (x * 37 + y * 101) as u8 ^ seed;
        image::Rgba([
            value,
            value.wrapping_mul(3),
            value.wrapping_add(90),
            if (x + y) % 2 == 0 { 255 } else { 128 },
        ])
    });
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(image)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

async fn chat_with_images(
    runtime: &RuntimeService,
    text: &str,
    images: &[String],
) -> crate::error::Result<ChatGeneration> {
    let params = crate::engine::GenerationParams {
        temperature: 0.0,
        max_tokens: MAX_TOKENS,
        ..Default::default()
    };
    let chat_config = crate::models::shared::chat::ChatRequestConfig {
        media_inputs: images
            .iter()
            .map(|source| crate::models::shared::chat::ChatMediaInput {
                kind: crate::models::shared::chat::ChatMediaKind::Image,
                source: source.clone(),
            })
            .collect(),
        ..Default::default()
    };
    Box::pin(
        runtime.chat_generate_with_generation_params_and_chat_config_and_correlation(
            VARIANT,
            user(text),
            params,
            chat_config,
            None,
        ),
    )
    .await
}

/// Images reach the model through the runtime: fetched, decoded, cut into
/// cells, encoded, and prefilled in place of their slots.
async fn synthetic_chats_see_their_images_body() {
    use crate::models::shared::chat::CHAT_IMAGE_PLACEHOLDER as IMAGE;
    let models = models_dir("images");
    let runtime = runtime_with(&models, 2).await;
    // The tiny tower's cell is 4 pixels: 10 x 7 pixels are 3 x 2 cells.
    let first = png_data_url(10, 7, 0);
    let second = png_data_url(10, 7, 0x5a);
    let ask = format!("What is in {IMAGE}?");
    let plain = chat(&runtime, "What is in ?").await.expect("text chat");
    let seen = chat_with_images(&runtime, &ask, std::slice::from_ref(&first))
        .await
        .expect("image chat");
    // The text's two messages, the image message's role, type and end, and
    // six slots.
    assert_eq!(seen.prompt_tokens, plain.prompt_tokens + 3 + 3 + 6);
    let again = chat_with_images(&runtime, &ask, std::slice::from_ref(&first))
        .await
        .expect("image chat again");
    assert_eq!(again.text, seen.text);
    // Another image under the same tokens is another reply: the pixels are
    // read, not only counted.
    let other = chat_with_images(&runtime, &ask, std::slice::from_ref(&second))
        .await
        .expect("second image");
    assert_eq!(other.prompt_tokens, seen.prompt_tokens);
    assert_ne!(other.text, seen.text);
    // Two requests with images at once reply as they do alone.
    let both = futures::future::join_all([
        chat_with_images(&runtime, &ask, std::slice::from_ref(&first)),
        chat_with_images(&runtime, &ask, std::slice::from_ref(&second)),
    ])
    .await;
    assert_eq!(both[0].as_ref().expect("concurrent").text, seen.text);
    assert_eq!(both[1].as_ref().expect("concurrent").text, other.text);
    // Two images in one message.
    let two = format!("Compare {IMAGE} and {IMAGE}.");
    let compared = chat_with_images(&runtime, &two, &[first.clone(), second.clone()])
        .await
        .expect("two images");
    assert!(compared.prompt_tokens > seen.prompt_tokens + 6);
    // Images and placeholders that do not pair off, and bytes that are not
    // an image, fail the request and leave the model serving.
    assert!(
        chat_with_images(&runtime, &two, std::slice::from_ref(&first))
            .await
            .is_err()
    );
    let broken = "data:image/png;base64,AAAA".to_string();
    assert!(chat_with_images(&runtime, &ask, &[broken]).await.is_err());
    let after = chat_with_images(&runtime, &ask, std::slice::from_ref(&first))
        .await
        .expect("after failures");
    assert_eq!(after.text, seen.text);
    Box::pin(runtime.unload_model(VARIANT))
        .await
        .expect("unload");
}

#[test]
fn synthetic_chats_see_their_images() {
    crate::block_on_large_stack(4, synthetic_chats_see_their_images_body);
}

#[test]
fn synthetic_chats_run_alone_and_concurrently_alike() {
    crate::block_on_large_stack(4, synthetic_chats_run_alone_and_concurrently_alike_body);
}
