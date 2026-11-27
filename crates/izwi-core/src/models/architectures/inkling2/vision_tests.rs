//! The image path against the reference fixture
//! (`tests/fixtures/inkling2_golden/images.safetensors`): two images, their
//! processed pixels and features as transformers computes them, and the
//! logits of a prompt that carries both.

use candle_core::{Device, Tensor};
use safetensors::SafeTensors;

use super::checkpoint::Inkling2Checkpoint;
use super::config::tests::{fixture, tiny};
use super::tests::{max_error, tiny_trunk};
use super::vision::{image_tokens, preprocess_image, Inkling2Image, Inkling2Vision};

pub(crate) struct GoldenImage {
    /// `[height, width, channels]`.
    pub shape: [usize; 3],
    pub bytes: Vec<u8>,
    pub grid: [usize; 2],
    pub pixels: Vec<f32>,
    /// `[cells, hidden]`.
    pub features: Vec<f32>,
}

pub(crate) struct ImageGolden {
    pub images: Vec<GoldenImage>,
    pub tokens: Vec<u32>,
    pub logits: Vec<f32>,
}

pub(crate) fn image_golden() -> ImageGolden {
    let bytes = std::fs::read(fixture("inkling2_golden").join("images.safetensors")).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    let f32s = |name: &str| {
        file.tensor(name)
            .unwrap()
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|quad| f32::from_le_bytes(*quad))
            .collect::<Vec<_>>()
    };
    let i64s = |name: &str| {
        file.tensor(name)
            .unwrap()
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| i64::from_le_bytes(*bytes) as usize)
            .collect::<Vec<_>>()
    };
    let images = (0..)
        .map_while(|index| {
            let rgba = file.tensor(&format!("image.{index}.rgba")).ok()?;
            let grid = i64s(&format!("image.{index}.grid"));
            Some(GoldenImage {
                shape: rgba.shape().try_into().unwrap(),
                bytes: rgba.data().to_vec(),
                grid: [grid[0], grid[1]],
                pixels: f32s(&format!("image.{index}.pixels")),
                features: f32s(&format!("image.{index}.features")),
            })
        })
        .collect::<Vec<_>>();
    ImageGolden {
        images,
        tokens: i64s("tokens").into_iter().map(|t| t as u32).collect(),
        logits: f32s("logits"),
    }
}

pub(crate) fn tiny_vision(device: &Device) -> Inkling2Vision {
    let config = tiny();
    let checkpoint = Inkling2Checkpoint::open(&fixture("inkling2_golden"), &config).unwrap();
    Inkling2Vision::load(&checkpoint, &config, device).unwrap()
}

fn processed(image: &GoldenImage, cell: usize) -> Inkling2Image {
    let [height, width, channels] = image.shape;
    preprocess_image(&image.bytes, height, width, channels, cell).unwrap()
}

#[test]
fn images_are_cut_into_cells_as_the_reference_does() {
    let golden = image_golden();
    assert_eq!(golden.images.len(), 2);
    // One image has transparency and sides that are not whole cells.
    assert_eq!(golden.images[0].shape, [7, 10, 4]);
    let cell = tiny().vision.unwrap().cell();
    for image in &golden.images {
        let got = processed(image, cell);
        assert_eq!([got.rows, got.columns], image.grid);
        assert_eq!(
            got.tokens(),
            image_tokens(image.shape[0], image.shape[1], cell)
        );
        assert!(max_error(&got.pixels, &image.pixels) <= 1e-6);
    }
    assert!(preprocess_image(&[0; 5], 1, 1, 5, cell).is_err());
    assert!(preprocess_image(&[0; 5], 1, 2, 3, cell).is_err());
}

#[test]
fn transparency_is_composited_over_the_checkerboard() {
    // A fully transparent 16-pixel row: light for 8 pixels, then dark; the
    // row below the image is padding.
    let image = preprocess_image(&[0u8; 16 * 4], 1, 16, 4, 16).unwrap();
    let value = |v: f32| (v / 255.0 - 0.5) / 0.5;
    assert!((image.pixels[0] - value(204.0)).abs() < 1e-6);
    assert!((image.pixels[8 * 3] - value(153.0)).abs() < 1e-6);
    assert!((image.pixels[16 * 3] - value(-1.0)).abs() < 1e-6);
    // An opaque pixel keeps its colour; a half-transparent one is rounded
    // to the nearest integer before scaling.
    let image = preprocess_image(&[10, 20, 30, 255, 100, 100, 100, 128], 1, 2, 4, 2).unwrap();
    assert!((image.pixels[1] - value(20.0)).abs() < 1e-6);
    let mixed = (100 * 128 + 204 * 127 + 127) / 255;
    assert!((image.pixels[3] - value(mixed as f32)).abs() < 1e-6);
}

fn features_match(device: &Device, tolerance: f32) {
    let golden = image_golden();
    let vision = tiny_vision(device);
    let cell = vision.config().cell();
    for image in &golden.images {
        let features = vision.encode(&processed(image, cell)).unwrap();
        let got = features.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let error = max_error(&got, &image.features);
        assert!(error <= tolerance, "image feature error {error}");
    }
    // The fixture would not notice a transposed image if both were square.
    assert_ne!(golden.images[0].grid[0], golden.images[0].grid[1]);
}

#[test]
fn vision_tower_matches_the_reference_features() {
    features_match(&Device::Cpu, 2e-5);
}

#[cfg(feature = "metal")]
#[test]
fn vision_tower_matches_the_reference_features_on_metal() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "vision_tower_matches_the_reference_features_on_metal",
    ) else {
        return;
    };
    features_match(&device, 1e-3);
}

/// Image features take the place of the image tokens' embeddings, after the
/// embedding norm, and the trunk then matches the reference logits.
#[test]
fn a_prompt_with_images_matches_the_reference_logits() {
    let golden = image_golden();
    let device = Device::Cpu;
    let trunk = tiny_trunk(&device);
    let vision = tiny_vision(&device);
    let config = tiny();
    let cell = vision.config().cell();
    let features = golden
        .images
        .iter()
        .map(|image| vision.encode(&processed(image, cell)).unwrap())
        .collect::<Vec<_>>();
    let features = Tensor::cat(&features, 0).unwrap();
    let slots = golden
        .tokens
        .iter()
        .filter(|token| **token == config.image_token_id)
        .count();
    assert_eq!(slots, features.dim(0).unwrap());

    let embeddings = trunk
        .embed_with_images(&golden.tokens, config.image_token_id, &features, 0)
        .unwrap();
    let mut state = trunk.new_state().unwrap();
    let hidden = trunk
        .forward_embeddings(&embeddings, &mut state, None)
        .unwrap();
    let logits = trunk.logits(&hidden).unwrap();
    let got = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let error = max_error(&got, &golden.logits);
    assert!(error <= 2e-3, "logit error with images {error}");

    // Without the features the logits are different ones.
    let plain = trunk.embed(&golden.tokens).unwrap();
    let mut state = trunk.new_state().unwrap();
    let hidden = trunk.forward_embeddings(&plain, &mut state, None).unwrap();
    let plain = trunk.logits(&hidden).unwrap();
    let plain = plain.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(max_error(&plain, &golden.logits) > 1e-1);
}
