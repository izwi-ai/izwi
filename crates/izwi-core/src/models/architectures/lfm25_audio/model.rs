use std::path::Path;
use std::sync::Mutex;

use tracing::info;

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::backbone::QuantizedLfm2Backbone;
use super::bundle::{Lfm25AudioBundle, Lfm25AudioBundleInfo};
use super::config::{
    parse_audio_decoder_config, parse_audio_encoder_config, parse_detokenizer_config,
    parse_main_backbone_config, Lfm25AudioDecoderConfig, Lfm25AudioEncoderConfig,
    Lfm2BackboneConfig,
};
use super::tokenizer::Lfm25TextTokenizer;

pub struct Lfm25AudioModel {
    device: DeviceProfile,
    bundle_info: Lfm25AudioBundleInfo,
    tokenizer: Lfm25TextTokenizer,
    main_config: Lfm2BackboneConfig,
    detokenizer_config: Lfm2BackboneConfig,
    encoder_config: Lfm25AudioEncoderConfig,
    decoder_config: Lfm25AudioDecoderConfig,
    main_backbone: Mutex<QuantizedLfm2Backbone>,
    detokenizer_backbone: Mutex<QuantizedLfm2Backbone>,
}

impl Lfm25AudioModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if !matches!(variant, ModelVariant::Lfm25Audio15BGguf) {
            return Err(Error::ModelLoadError(format!(
                "Unsupported LFM2.5 Audio variant: {variant}"
            )));
        }

        let backend = BackendKind::from(device.kind);
        let bundle = Lfm25AudioBundle::load(model_dir, backend)?;
        let bundle_info = bundle.info();

        let tokenizer = Lfm25TextTokenizer::load(&bundle.main)?;
        let main_config = parse_main_backbone_config(&bundle.main)?;
        let detokenizer_config = parse_detokenizer_config(&bundle.tokenizer)?;
        let encoder_config = parse_audio_encoder_config(&bundle.mmproj)?;
        let decoder_config = parse_audio_decoder_config(&bundle.vocoder)?;

        let main_backbone =
            QuantizedLfm2Backbone::load(&bundle.main, main_config.clone(), &device.device)?;
        let detokenizer_backbone = QuantizedLfm2Backbone::load(
            &bundle.tokenizer,
            detokenizer_config.clone(),
            &device.device,
        )?;

        info!(
            "Loaded LFM2.5 Audio GGUF bundle on {:?} from {}",
            device.kind,
            model_dir.display()
        );

        Ok(Self {
            device,
            bundle_info,
            tokenizer,
            main_config,
            detokenizer_config,
            encoder_config,
            decoder_config,
            main_backbone: Mutex::new(main_backbone),
            detokenizer_backbone: Mutex::new(detokenizer_backbone),
        })
    }

    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub fn bundle_info(&self) -> &Lfm25AudioBundleInfo {
        &self.bundle_info
    }

    pub fn tokenizer(&self) -> &Lfm25TextTokenizer {
        &self.tokenizer
    }

    pub fn main_config(&self) -> &Lfm2BackboneConfig {
        &self.main_config
    }

    pub fn detokenizer_config(&self) -> &Lfm2BackboneConfig {
        &self.detokenizer_config
    }

    pub fn encoder_config(&self) -> &Lfm25AudioEncoderConfig {
        &self.encoder_config
    }

    pub fn decoder_config(&self) -> &Lfm25AudioDecoderConfig {
        &self.decoder_config
    }

    pub fn with_main_backbone<T>(
        &self,
        f: impl FnOnce(&mut QuantizedLfm2Backbone) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.main_backbone.lock().map_err(|_| {
            Error::InferenceError("LFM2.5 Audio backbone mutex poisoned".to_string())
        })?;
        f(&mut guard)
    }

    pub fn with_detokenizer_backbone<T>(
        &self,
        f: impl FnOnce(&mut QuantizedLfm2Backbone) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.detokenizer_backbone.lock().map_err(|_| {
            Error::InferenceError("LFM2.5 Audio detokenizer mutex poisoned".to_string())
        })?;
        f(&mut guard)
    }
}
