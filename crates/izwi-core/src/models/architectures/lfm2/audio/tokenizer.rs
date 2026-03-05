use std::fs;
use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LfmModality {
    Text = 1,
    AudioIn = 2,
    AudioOut = 3,
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub start_of_text: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub audio_start: u32,
    pub text_end: u32,
}

#[derive(Debug)]
pub struct Lfm2Tokenizer {
    inner: Tokenizer,
    specials: SpecialTokens,
    turn_format: ChatTurnFormat,
}

impl Lfm2Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(path)
            .map_err(|e| Error::ModelLoadError(format!("Failed to load LFM2 tokenizer: {e}")))?;

        let lookup = |tok: &str| -> Result<u32> {
            inner.token_to_id(tok).ok_or_else(|| {
                Error::ModelLoadError(format!("LFM2 tokenizer missing special token '{tok}'"))
            })
        };

        let specials = SpecialTokens {
            start_of_text: lookup("<|startoftext|>")?,
            im_start: lookup("<|im_start|>")?,
            im_end: lookup("<|im_end|>")?,
            audio_start: lookup("<|audio_start|>")?,
            text_end: lookup("<|text_end|>")?,
        };

        let template_path = model_dir.join("chat_template.jinja");
        let template = fs::read_to_string(&template_path).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to read LFM2 chat template {}: {e}",
                template_path.display()
            ))
        })?;
        let turn_format = ChatTurnFormat::from_chat_template(&template)?;

        Ok(Self {
            inner,
            specials,
            turn_format,
        })
    }

    pub fn specials(&self) -> &SpecialTokens {
        &self.specials
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::InferenceError(format!("LFM2 tokenizer encode error: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode_text(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::InferenceError(format!("LFM2 tokenizer decode error: {e}")))
    }

    pub fn render_turn_start(&self, role: &str) -> String {
        self.turn_format.render_turn_start(role)
    }

    pub fn turn_end(&self) -> &str {
        self.turn_format.turn_end()
    }
}

#[derive(Debug, Clone)]
struct ChatTurnFormat {
    turn_start_prefix: String,
    turn_start_suffix: String,
    turn_end: String,
}

impl ChatTurnFormat {
    fn from_chat_template(template: &str) -> Result<Self> {
        if !template.contains("<|im_start|>") {
            return Err(Error::ModelLoadError(
                "LFM2 chat template is missing <|im_start|>".to_string(),
            ));
        }
        if !template.contains("<|im_end|>") {
            return Err(Error::ModelLoadError(
                "LFM2 chat template is missing <|im_end|>".to_string(),
            ));
        }

        let turn_start_suffix = Self::suffix_after_any(
            template,
            &[
                "<|im_start|>\" + message[\"role\"] + \"",
                "<|im_start|>\" + message['role'] + \"",
                "<|im_start|>' + message[\"role\"] + '",
                "<|im_start|>' + message['role'] + '",
                "<|im_start|>assistant",
            ],
        )
        .unwrap_or_else(|| "\n".to_string());

        let turn_end_suffix = Self::suffix_after_any(
            template,
            &[
                "content + \"<|im_end|>",
                "content + '<|im_end|>",
                "<|im_end|>",
            ],
        )
        .unwrap_or_else(|| "\n".to_string());

        Ok(Self {
            turn_start_prefix: "<|im_start|>".to_string(),
            turn_start_suffix,
            turn_end: format!("<|im_end|>{turn_end_suffix}"),
        })
    }

    fn suffix_after_any(template: &str, markers: &[&str]) -> Option<String> {
        markers
            .iter()
            .find_map(|marker| template.find(marker).map(|start| (marker, start)))
            .map(|(marker, start)| {
                let rest = &template[start + marker.len()..];
                if rest.starts_with("\\r\\n") {
                    "\r\n".to_string()
                } else if rest.starts_with("\\n") || rest.starts_with('\n') {
                    "\n".to_string()
                } else if rest.starts_with("\\r") || rest.starts_with('\r') {
                    "\r".to_string()
                } else {
                    String::new()
                }
            })
    }

    fn render_turn_start(&self, role: &str) -> String {
        format!(
            "{}{}{}",
            self.turn_start_prefix, role, self.turn_start_suffix
        )
    }

    fn turn_end(&self) -> &str {
        self.turn_end.as_str()
    }
}

#[derive(Debug, Clone)]
pub struct ChatState {
    pub text: Vec<u32>,
    pub audio_in: Vec<f32>, // flattened [features, total_frames]
    pub audio_in_lens: Vec<usize>,
    pub audio_out: Vec<u32>, // flattened [codebooks, total_frames]
    pub modality_flag: Vec<u32>,
    pub codebooks: usize,
    pub features: usize,
}

impl ChatState {
    pub fn new(tok: &Lfm2Tokenizer, codebooks: usize, features: usize) -> Self {
        Self {
            text: vec![tok.specials.start_of_text],
            audio_in: Vec::new(),
            audio_in_lens: Vec::new(),
            audio_out: Vec::new(),
            modality_flag: vec![LfmModality::Text as u32],
            codebooks,
            features,
        }
    }

    pub fn add_text(&mut self, tok: &Lfm2Tokenizer, text: &str) -> Result<()> {
        let mut ids = tok.encode_text(text)?;
        self.modality_flag
            .extend(std::iter::repeat_n(LfmModality::Text as u32, ids.len()));
        self.text.append(&mut ids);
        Ok(())
    }

    pub fn add_audio_mel(&mut self, mel: &[f32], frames: usize) {
        if frames == 0 {
            return;
        }
        self.audio_in.extend_from_slice(mel);
        self.audio_in_lens.push(frames);
        let emb_len = mel_to_embed_len(frames);
        self.modality_flag
            .extend(std::iter::repeat_n(LfmModality::AudioIn as u32, emb_len));
    }

    pub fn end_turn(&mut self, tok: &Lfm2Tokenizer) -> Result<()> {
        self.add_text(tok, tok.turn_end())
    }

    pub fn new_turn(&mut self, tok: &Lfm2Tokenizer, role: &str) -> Result<()> {
        let turn = tok.render_turn_start(role);
        self.add_text(tok, &turn)
    }

    pub fn to_text_tensor(&self) -> Vec<u32> {
        self.text.clone()
    }

    pub fn to_audio_in_tensor(&self) -> (Vec<f32>, usize) {
        (self.audio_in.clone(), self.audio_in_lens.iter().sum())
    }

    pub fn to_audio_out_tensor(&self) -> Vec<u32> {
        self.audio_out.clone()
    }
}

pub fn mel_to_embed_len(frames: usize) -> usize {
    frames.div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::ChatTurnFormat;

    #[test]
    fn parses_turn_wrappers_from_chat_template() {
        let template = "{{- \"<|im_start|>\" + message[\"role\"] + \"\\n\" -}} {{- content + \"<|im_end|>\\n\" -}}";
        let fmt = ChatTurnFormat::from_chat_template(template).expect("parse chat template");
        assert_eq!(
            fmt.render_turn_start("assistant"),
            "<|im_start|>assistant\n"
        );
        assert_eq!(fmt.turn_end(), "<|im_end|>\n");
    }

    #[test]
    fn rejects_template_missing_required_markers() {
        let err = ChatTurnFormat::from_chat_template("{{- message['content'] -}}")
            .expect_err("missing im markers should fail");
        let msg = err.to_string();
        assert!(msg.contains("<|im_start|>") || msg.contains("<|im_end|>"));
    }
}
