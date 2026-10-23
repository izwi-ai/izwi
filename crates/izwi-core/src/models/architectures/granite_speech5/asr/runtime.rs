//! Native Candle runtime for Granite Speech 5.0: checkpoint materialization
//! with load-time precision probes, the audio front end (encoder and
//! projector), and greedy decoding on the lifecycle-owned paged cache.

use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};

use crate::backends::{DTypeSelection, DeviceProfile};
use crate::error::{Error, Result};
use crate::kernels::f16_weight_linear::{load_weight_only_f16, WeightOnlyF16Load};
use crate::kernels::numeric_check::{
    load_with_precision_fallback, run_numeric_self_check, PrecisionLoad, F16_SAFE_MAX_ABS,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::weights::staged;

use super::config::GraniteSpeech5Config;
use super::encoder::ConformerEncoder;
use super::language_model::{argmax_last_token, text_attention_geometry, GraniteLanguageModel};
use super::precision::{self, LmExecutionPlan};
use super::preprocessor::GraniteSpeech5AudioFeatures;
use super::projector::QFormerProjector;
use super::prompt::{GraniteSpeech5Prompt, GraniteSpeech5SpecialTokens};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraniteSpeech5AudioStats {
    pub upload: Duration,
    pub encoder: Duration,
    pub projector: Duration,
    pub encoder_frames: usize,
    pub conformer_context_size: usize,
    pub conformer_layers: usize,
    pub qformer_windows: usize,
    pub qformer_window_size: usize,
    pub qformer_queries_per_window: usize,
    pub qformer_layers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeech5GenerationStats {
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
    pub generated_tokens: usize,
    pub max_new_tokens: usize,
    pub stop_reason: &'static str,
    pub stop_token: Option<u32>,
    /// Tokens read back per stop check (1, or Metal's interval).
    pub stop_check_interval: usize,
    pub prefill: Duration,
    pub decode: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeech5Generation {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub stats: GraniteSpeech5GenerationStats,
}

pub struct GraniteSpeech5Runtime {
    device: Device,
    dtype: DType,
    /// The encoder and projector activations' dtype: F32 when the encoder
    /// loads weight-only F16, else `dtype`.
    audio_dtype: DType,
    encoder_precision: WeightOnlyF16Load,
    lm_precision: PrecisionLoad,
    encoder: ConformerEncoder,
    projector: QFormerProjector,
    language_model: GraniteLanguageModel,
}

impl GraniteSpeech5Runtime {
    pub fn load(
        weights: &Path,
        config: &GraniteSpeech5Config,
        device: &DeviceProfile,
        dtype: DType,
        encoder_precision: &DTypeSelection,
    ) -> Result<Self> {
        // Reject malformed geometry before mapping the 4.6 GB checkpoint.
        text_attention_geometry(&config.text_config)?;
        let rope = config.text_config.rope()?;
        let weights = [weights.to_path_buf()];
        let map = |dtype: DType| {
            unsafe { staged::metal_staged_safetensors(&weights, dtype, &device.device) }.map_err(
                |err| {
                    Error::ModelLoadError(format!(
                        "Failed to map Granite Speech 5.0 safetensors: {err}"
                    ))
                },
            )
        };
        let load_started = Instant::now();
        let vb = map(dtype)?;
        // A weight-only F16 encoder keeps F32 activations: its norms,
        // depthwise convolution, biases and the projector read an F32
        // builder even when the language model loads in a lower dtype.
        let audio_dtype = if encoder_precision.dtype == DType::F16 {
            DType::F32
        } else {
            dtype
        };
        let audio_vb = if audio_dtype == dtype {
            vb.clone()
        } else {
            map(audio_dtype)?
        };

        let encoder_vb = audio_vb.pp("encoder");
        let (encoder, encoder_weight_precision) = load_weight_only_f16(
            "Granite Speech 5.0 encoder",
            encoder_precision,
            &device.device,
            |f16| ConformerEncoder::load(&config.encoder_config, encoder_vb.clone(), f16),
            |encoder| encoder.weight_only_f16_self_check(&config.encoder_config, &encoder_vb),
        )?;
        let projector = QFormerProjector::load(config, audio_vb.pp("projector"))?;

        let plan = LmExecutionPlan::for_device(&device.device, dtype);
        let lm_selection = precision::lm_precision(dtype, plan, device);
        let load_lm = |loaded: DType| {
            if loaded == lm_selection.dtype {
                return GraniteLanguageModel::load(
                    &config.text_config,
                    &rope,
                    config.tie_word_embeddings(),
                    vb.pp("language_model"),
                    plan,
                );
            }
            // The range probe's fallback: every LM weight and activation F32.
            let f32_vb = if dtype == DType::F32 {
                vb.clone()
            } else if audio_dtype == DType::F32 {
                audio_vb.clone()
            } else {
                map(DType::F32)?
            };
            GraniteLanguageModel::load(
                &config.text_config,
                &rope,
                config.tie_word_embeddings(),
                f32_vb.pp("language_model"),
                LmExecutionPlan::BASE_DTYPE,
            )
        };
        let (language_model, lm_precision) = load_with_precision_fallback(
            "Granite Speech 5.0 LM",
            &lm_selection,
            device.kind.is_metal(),
            DType::F32,
            load_lm,
            |lm| {
                run_numeric_self_check(
                    format!("Granite Speech 5.0 LM {:?}", lm_selection.dtype),
                    F16_SAFE_MAX_ABS,
                    || lm.numeric_probe(precision::LM_PROBE_TOKENS),
                )
            },
        )?;

        // Construction has consumed the builders; release the mappings
        // before waiting for queued uploads, casts and fusions.
        drop(vb);
        drop(audio_vb);
        drop(encoder_vb);
        device.device.synchronize().map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to synchronize Granite Speech 5.0 checkpoint materialization: {err}"
            ))
        })?;
        tracing::info!(
            dtype = ?dtype,
            total_ms = load_started.elapsed().as_secs_f64() * 1_000.0,
            "Granite Speech 5.0 checkpoint materialization completed"
        );
        Ok(Self {
            device: device.device.clone(),
            dtype,
            audio_dtype,
            encoder_precision: encoder_weight_precision,
            lm_precision,
            encoder,
            projector,
            language_model,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn kv_dtype(&self) -> DType {
        self.language_model.kv_dtype()
    }

    pub(crate) fn encoder_precision(&self) -> &WeightOnlyF16Load {
        &self.encoder_precision
    }

    pub(crate) fn lm_precision(&self) -> &PrecisionLoad {
        &self.lm_precision
    }

    /// The execution paths the language model resolved, for diagnostics.
    pub(crate) fn execution_diagnostics(&self) -> serde_json::Value {
        let plan = self.language_model.plan();
        serde_json::json!({
            "f16_lm_head": plan.f16_lm_head,
            "f16_qkv": plan.f16_qkv,
            "f16_attention_core": plan.f16_attention_core,
            "f16_mlp": plan.f16_mlp,
            "f16_attention_output": plan.f16_attention_output,
            "qkv_projection_fused": precision::PROJECTION_FUSION,
            "gate_up_projection_fused": precision::PROJECTION_FUSION,
            "residual_branches_prescaled": self.language_model.residual_branches_prescaled(),
            "rope_type": "yarn",
            "rope_attention_scale": self.language_model.rope_attention_scale(),
            "audio_dtype": format!("{:?}", self.audio_dtype),
            "kv_dtype": format!("{:?}", self.kv_dtype()),
        })
    }

    /// Encoder then projector: `(1, windows * queries, text hidden)`.
    pub fn audio_embeddings(
        &self,
        features: &GraniteSpeech5AudioFeatures,
    ) -> Result<(Tensor, GraniteSpeech5AudioStats)> {
        let (frames, dim) = (features.encoder_frames, features.encoder_dim);
        if frames == 0 || dim == 0 {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 audio produced no encoder features".to_string(),
            ));
        }
        let flat = features
            .input_features
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .collect::<Vec<_>>();
        let started = Instant::now();
        let input =
            Tensor::from_vec(flat, (1, frames, dim), &self.device)?.to_dtype(self.audio_dtype)?;
        let upload = started.elapsed();
        let started = Instant::now();
        let encoded = self.encoder.forward(&input)?;
        let encoder = started.elapsed();
        let started = Instant::now();
        let embeddings = self.projector.forward(&encoded)?;
        let projector = started.elapsed();
        Ok((
            embeddings,
            GraniteSpeech5AudioStats {
                upload,
                encoder,
                projector,
                encoder_frames: frames,
                conformer_context_size: self.encoder.context_size(),
                conformer_layers: self.encoder.layer_count(),
                qformer_windows: self.projector.window_count(frames),
                qformer_window_size: self.projector.window_size(),
                qformer_queries_per_window: self.projector.num_queries(),
                qformer_layers: self.projector.layer_count(),
            },
        ))
    }

    /// The prompt's token ids with the audio placeholder expanded to one id
    /// per projected audio frame.
    pub(super) fn expanded_input_ids(
        prompt: &GraniteSpeech5Prompt,
        audio_tokens: usize,
        audio_token_id: u32,
    ) -> Result<Vec<u32>> {
        let position = prompt.audio_token_position;
        if audio_tokens == 0 || position >= prompt.input_ids.len() {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 prompt has no room for the projected audio".into(),
            ));
        }
        let mut ids = Vec::with_capacity(prompt.input_ids.len() + audio_tokens - 1);
        ids.extend_from_slice(&prompt.input_ids[..position]);
        ids.extend(std::iter::repeat_n(audio_token_id, audio_tokens));
        ids.extend_from_slice(&prompt.input_ids[position + 1..]);
        Ok(ids)
    }

    /// Prefill and greedy decode. `decode` renders token ids as text; with
    /// `emit_deltas`, `on_delta` receives the text as it grows. The cache
    /// must start empty and hold the prompt plus `max_new_tokens`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &GraniteSpeech5Prompt,
        special_tokens: &GraniteSpeech5SpecialTokens,
        audio_embeds: &Tensor,
        max_new_tokens: usize,
        extra_stop_token_ids: &[u32],
        stop_sequences: &[String],
        decode: &mut dyn FnMut(&[u32]) -> Result<String>,
        cache: &mut PhysicalPagedKvCache,
        on_delta: &mut dyn FnMut(&str),
        emit_deltas: bool,
    ) -> Result<GraniteSpeech5Generation> {
        let audio_tokens = audio_embeds.dim(1)?;
        let input_ids =
            Self::expanded_input_ids(prompt, audio_tokens, special_tokens.audio_token_id)?;
        let max_steps = max_new_tokens.max(1);
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 invocation cache must begin empty".into(),
            ));
        }
        let required = input_ids.len().checked_add(max_steps).ok_or_else(|| {
            Error::InvalidInput("Granite Speech 5.0 context length overflow".into())
        })?;
        if required > cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Granite Speech 5.0 needs {required} decoder tokens ({} prompt, {} audio, {max_steps} to generate), but the context holds {}; shorten the audio or the keyword list",
                input_ids.len() - audio_tokens,
                audio_tokens,
                cache.capacity_tokens()
            )));
        }

        let prefill_started = Instant::now();
        let rope = self.language_model.rope_cache(required)?;
        let embeds = self.language_model.prompt_embeddings_with_audio(
            &input_ids,
            prompt.audio_token_position,
            audio_embeds,
        )?;
        let mut logits =
            self.language_model
                .forward_embeds(&embeds, 0, cache, Some(&rope), None)?;
        let prefill = prefill_started.elapsed();

        let mut stop_tokens = vec![special_tokens.eos_token_id, special_tokens.pad_token_id];
        stop_tokens.extend_from_slice(extra_stop_token_ids);
        stop_tokens.sort_unstable();
        stop_tokens.dedup();
        let is_stop = |token: &u32| stop_tokens.binary_search(token).is_ok();
        let decode_each_step = emit_deltas || !stop_sequences.is_empty();
        // Without per-step text, Metal reads tokens back once per interval.
        let stop_check_interval =
            if !decode_each_step && precision::chunked_stop_check(&self.device) {
                precision::METAL_STOP_CHECK_INTERVAL.clamp(1, max_steps)
            } else {
                1
            };

        let decode_started = Instant::now();
        let mut generated = Vec::new();
        let mut rendered = String::new();
        // Bytes of `rendered` already streamed: text up to a trailing partial
        // UTF-8 sequence (decoded as U+FFFD until its last byte arrives).
        let mut emitted = 0usize;
        let mut stop_reason = "max_tokens";
        let mut stop_token = None;
        let mut pending = Vec::with_capacity(stop_check_interval);
        for step in 0..max_steps {
            let token = argmax_last_token(&logits)?;
            if step + 1 < max_steps {
                logits = self.language_model.forward_token(
                    &token,
                    input_ids.len() + step,
                    cache,
                    &rope,
                )?;
            }
            pending.push(token);
            if pending.len() < stop_check_interval && step + 1 < max_steps {
                continue;
            }
            let tokens = Tensor::cat(&pending, 1)?.squeeze(0)?.to_vec1::<u32>()?;
            pending.clear();
            let stop_at = tokens.iter().position(is_stop);
            let accepted = &tokens[..stop_at.unwrap_or(tokens.len())];
            generated.extend_from_slice(accepted);
            if decode_each_step && !accepted.is_empty() {
                let mut next = decode(&generated)?;
                let stopped_on_sequence = truncate_at_stop_sequence(&mut next, stop_sequences);
                if emit_deltas {
                    let stable = next.trim_end_matches('\u{FFFD}').len();
                    if stable > emitted && next.is_char_boundary(emitted) {
                        on_delta(&next[emitted..stable]);
                        emitted = stable;
                    }
                }
                rendered = next;
                if stopped_on_sequence {
                    stop_reason = "stop_sequence";
                    break;
                }
            }
            if let Some(index) = stop_at {
                stop_reason = "stop_token";
                stop_token = Some(tokens[index]);
                break;
            }
        }
        if !decode_each_step && !generated.is_empty() {
            rendered = decode(&generated)?;
        }
        if emit_deltas && rendered.len() > emitted && rendered.is_char_boundary(emitted) {
            on_delta(&rendered[emitted..]);
        }
        Ok(GraniteSpeech5Generation {
            text: rendered,
            stats: GraniteSpeech5GenerationStats {
                prompt_tokens: input_ids.len(),
                audio_tokens,
                generated_tokens: generated.len(),
                max_new_tokens,
                stop_reason,
                stop_token,
                stop_check_interval,
                prefill,
                decode: decode_started.elapsed(),
            },
            token_ids: generated,
        })
    }

    #[cfg(test)]
    pub(super) fn language_model(&self) -> &GraniteLanguageModel {
        &self.language_model
    }

    /// Prefill logits at transformers' scale, for the parity fixture.
    #[cfg(test)]
    pub(super) fn prefill_logits(
        &self,
        prompt: &GraniteSpeech5Prompt,
        special_tokens: &GraniteSpeech5SpecialTokens,
        audio_embeds: &Tensor,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let input_ids =
            Self::expanded_input_ids(prompt, audio_embeds.dim(1)?, special_tokens.audio_token_id)?;
        let embeds = self.language_model.prompt_embeddings_with_audio(
            &input_ids,
            prompt.audio_token_position,
            audio_embeds,
        )?;
        let logits = self
            .language_model
            .forward_embeds(&embeds, 0, cache, None, None)?;
        self.language_model.scale_logits(&logits)
    }
}

fn truncate_at_stop_sequence(text: &mut String, stop_sequences: &[String]) -> bool {
    for stop in stop_sequences.iter().filter(|stop| !stop.is_empty()) {
        if let Some(index) = text.find(stop.as_str()) {
            text.truncate(index);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_sequences_truncate_at_their_first_match() {
        let mut text = "hello world. STOP more".to_string();
        assert!(truncate_at_stop_sequence(
            &mut text,
            &["".into(), "STOP".into()]
        ));
        assert_eq!(text, "hello world. ");
        let mut untouched = "no match".to_string();
        assert!(!truncate_at_stop_sequence(&mut untouched, &["STOP".into()]));
    }

    #[test]
    fn audio_placeholder_expands_to_one_id_per_projected_frame() {
        let prompt = GraniteSpeech5Prompt {
            task: super::super::prompt::GraniteSpeech5Task::Transcribe,
            text: String::new(),
            instruction: String::new(),
            prompt_ignored: false,
            keyword_count: 0,
            input_ids: vec![10, 99, 11, 12],
            audio_token_position: 1,
        };
        assert_eq!(
            GraniteSpeech5Runtime::expanded_input_ids(&prompt, 3, 99).unwrap(),
            vec![10, 99, 99, 99, 11, 12]
        );
        assert!(GraniteSpeech5Runtime::expanded_input_ids(&prompt, 0, 99).is_err());
    }
}
