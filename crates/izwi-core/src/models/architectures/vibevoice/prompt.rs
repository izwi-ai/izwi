//! VibeVoice Qwen-tokenizer prompt assembly.

use std::collections::HashMap;
use std::fs;
use std::ops::Range;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

const TTS_SYSTEM_PROMPT: &str = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
const ASR_SYSTEM_PROMPT: &str =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";
const ASR_TRANSCRIPTION_KEYS: &str = "Start time, End time, Speaker ID, Content";

/// The user turn's text after the speech placeholders, exactly as the
/// reference processor builds it (`vibevoice_asr_processor.py`, VibeVoice
/// @16fb2cb1): the duration with two decimals and, when present, the trimmed
/// context ("hotwords") as extra info.
fn vibevoice_asr_user_suffix(audio_seconds: f64, extra_info: Option<&str>) -> String {
    let seconds = if audio_seconds.is_finite() {
        audio_seconds.max(0.0)
    } else {
        0.0
    };
    match extra_info.map(str::trim).filter(|extra| !extra.is_empty()) {
        Some(extra) => format!(
            "This is a {seconds:.2} seconds audio, with extra info: {extra}\n\nPlease transcribe it with these keys: {ASR_TRANSCRIPTION_KEYS}"
        ),
        None => format!(
            "This is a {seconds:.2} seconds audio, please transcribe it with these keys: {ASR_TRANSCRIPTION_KEYS}"
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceSpecialTokens {
    pub endoftext: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub object_ref_start: u32,
    pub object_ref_end: u32,
    pub box_start: u32,
    pub speech_start: u32,
    pub speech_end: u32,
    pub speech_pad: u32,
    pub image_pad: u32,
}

impl Default for VibeVoiceSpecialTokens {
    fn default() -> Self {
        Self {
            endoftext: 151_643,
            im_start: 151_644,
            im_end: 151_645,
            object_ref_start: 151_646,
            object_ref_end: 151_647,
            box_start: 151_648,
            speech_start: 151_652,
            speech_end: 151_653,
            speech_pad: 151_654,
            image_pad: 151_655,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceTtsPrompt {
    pub input_ids: Vec<u32>,
    pub reference_voice_range: Option<Range<usize>>,
    pub text_token_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceAsrPrompt {
    pub input_ids: Vec<u32>,
    pub acoustic_input_range: Range<usize>,
    pub prompt_token_count: usize,
}

pub struct VibeVoicePromptTokenizer {
    tokenizer: Tokenizer,
    specials: VibeVoiceSpecialTokens,
}

impl VibeVoicePromptTokenizer {
    pub fn load(model_dir: &Path, expected_vocab_size: usize) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_path_with_expected_vocab(model_dir, Some(expected_vocab_size))?;
        let specials = read_special_tokens(model_dir).unwrap_or_default();
        Ok(Self {
            tokenizer,
            specials,
        })
    }

    pub fn specials(&self) -> &VibeVoiceSpecialTokens {
        &self.specials
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens)
    }

    pub fn build_tts_prompt(
        &self,
        text: &str,
        speaker: &str,
        reference_frames: usize,
    ) -> Result<VibeVoiceTtsPrompt> {
        let script = parse_tts_script(text, speaker)?;
        let voice_speaker = single_tts_voice_speaker(&script)?;
        let mut input_ids = Vec::new();
        input_ids.extend(self.encode_text(TTS_SYSTEM_PROMPT)?);

        let reference_voice_range = if reference_frames > 0 {
            input_ids.extend(self.encode_text(" Voice input:\n")?);
            input_ids.extend(self.encode_text(&format!(" {voice_speaker}:"))?);
            input_ids.push(self.specials.speech_start);
            let start = input_ids.len();
            input_ids.extend(std::iter::repeat_n(
                self.specials.speech_pad,
                reference_frames,
            ));
            let end = input_ids.len();
            input_ids.push(self.specials.speech_end);
            input_ids.extend(self.encode_text("\n")?);
            Some(start..end)
        } else {
            None
        };

        let text_input = build_tts_text_input(&script);
        let text_tokens = self.encode_text(&text_input)?;
        let text_token_count = text_tokens.len();
        input_ids.extend(text_tokens);
        input_ids.extend(self.encode_text(" Speech output:\n")?);
        input_ids.push(self.specials.speech_start);

        Ok(VibeVoiceTtsPrompt {
            input_ids,
            reference_voice_range,
            text_token_count,
        })
    }

    pub fn build_asr_prompt(
        &self,
        audio_seconds: f64,
        acoustic_frames: usize,
        extra_instruction: Option<&str>,
    ) -> Result<VibeVoiceAsrPrompt> {
        let mut input_ids = Vec::new();
        self.push_chat_segment(&mut input_ids, "system", ASR_SYSTEM_PROMPT)?;
        input_ids.push(self.specials.im_end);
        input_ids.extend(self.encode_text("\n")?);
        self.push_role_header(&mut input_ids, "user")?;

        input_ids.push(self.specials.object_ref_start);
        let start = input_ids.len();
        input_ids.extend(std::iter::repeat_n(
            self.specials.box_start,
            acoustic_frames,
        ));
        let end = input_ids.len();
        input_ids.push(self.specials.object_ref_end);
        // The reference tokenizes the whole user turn as one string, which the
        // special tokens split; "\n" + suffix is a single ordinary span, so
        // encode it in one call to keep merges such as `Ġplease`.
        let suffix = vibevoice_asr_user_suffix(audio_seconds, extra_instruction);
        input_ids.extend(self.encode_text(&format!("\n{suffix}"))?);
        input_ids.push(self.specials.im_end);
        input_ids.extend(self.encode_text("\n")?);
        self.push_role_header(&mut input_ids, "assistant")?;

        Ok(VibeVoiceAsrPrompt {
            prompt_token_count: input_ids.len(),
            input_ids,
            acoustic_input_range: start..end,
        })
    }

    fn push_role_header(&self, ids: &mut Vec<u32>, role: &str) -> Result<()> {
        ids.push(self.specials.im_start);
        ids.extend(self.encode_text(&format!("{role}\n"))?);
        Ok(())
    }

    fn push_chat_segment(&self, ids: &mut Vec<u32>, role: &str, text: &str) -> Result<()> {
        self.push_role_header(ids, role)?;
        ids.extend(self.encode_text(text)?);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VibeVoiceTtsScriptLine {
    speaker_id: usize,
    text: String,
}

fn build_tts_text_input(script: &[VibeVoiceTtsScriptLine]) -> String {
    let mut input = String::from(" Text input:\n");
    for line in script {
        input.push_str(&format!(" Speaker {}: {}\n", line.speaker_id, line.text));
    }
    input
}

fn parse_tts_script(text: &str, default_speaker: &str) -> Result<Vec<VibeVoiceTtsScriptLine>> {
    let text = text.trim();
    if text.is_empty() {
        return Err(Error::InvalidInput(
            "VibeVoice TTS text input cannot be empty".to_string(),
        ));
    }

    let mut parsed = Vec::new();
    let mut unparsed_nonempty_lines = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(script_line) = parse_tts_speaker_line(line)? {
            parsed.push(script_line);
        } else {
            unparsed_nonempty_lines.push(line.to_string());
        }
    }

    if parsed.is_empty() {
        return Ok(vec![VibeVoiceTtsScriptLine {
            speaker_id: sanitize_tts_speaker_id(default_speaker),
            text: normalize_tts_text_line(text),
        }]);
    }
    if !unparsed_nonempty_lines.is_empty() {
        return Err(Error::InvalidInput(
            "VibeVoice TTS script input must put every non-empty line in `Speaker N:` format"
                .to_string(),
        ));
    }

    normalize_tts_script_speakers(&mut parsed);
    if parsed
        .iter()
        .any(|line| line.speaker_id != parsed[0].speaker_id)
    {
        return Err(Error::InvalidInput(
            "VibeVoice TTS currently supports one reference speaker per generation request"
                .to_string(),
        ));
    }
    Ok(parsed)
}

fn parse_tts_speaker_line(line: &str) -> Result<Option<VibeVoiceTtsScriptLine>> {
    let Some(rest) = line.strip_prefix("Speaker ") else {
        return Ok(None);
    };
    let Some((speaker_id, text)) = rest.split_once(':') else {
        return Ok(None);
    };
    let speaker_id = speaker_id.trim().parse::<usize>().map_err(|_| {
        Error::InvalidInput(format!(
            "VibeVoice TTS speaker label must use `Speaker N:` syntax, got `Speaker {speaker_id}`"
        ))
    })?;
    let text = normalize_tts_text_line(text);
    if text.is_empty() {
        return Ok(None);
    }
    Ok(Some(VibeVoiceTtsScriptLine { speaker_id, text }))
}

fn normalize_tts_script_speakers(script: &mut [VibeVoiceTtsScriptLine]) {
    let min_speaker = script.iter().map(|line| line.speaker_id).min().unwrap_or(0);
    if min_speaker > 0 {
        for line in script {
            line.speaker_id = line.speaker_id.saturating_sub(1);
        }
    }
}

fn single_tts_voice_speaker(script: &[VibeVoiceTtsScriptLine]) -> Result<String> {
    script
        .first()
        .map(|line| format!("Speaker {}", line.speaker_id))
        .ok_or_else(|| Error::InvalidInput("VibeVoice TTS text input cannot be empty".to_string()))
}

fn normalize_tts_text_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn sanitize_tts_speaker_id(speaker: &str) -> usize {
    let trimmed = speaker.trim();
    let Some(rest) = trimmed.strip_prefix("Speaker ") else {
        return 0;
    };
    rest.trim().parse::<usize>().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_text_input_uses_target_text_only_for_reference_audio() {
        let script = parse_tts_script("Generate this line.", "Speaker 0").expect("script");
        let block = build_tts_text_input(&script);

        assert_eq!(block, " Text input:\n Speaker 0: Generate this line.\n");
        assert!(!block.contains("Reference transcript."));
    }

    #[test]
    fn tts_text_input_normalizes_one_based_script_speaker_to_zero_based_voice() {
        let script =
            parse_tts_script("Speaker 1: Generate this line.", "Speaker 0").expect("script");
        let block = build_tts_text_input(&script);

        assert_eq!(block, " Text input:\n Speaker 0: Generate this line.\n");
        assert_eq!(single_tts_voice_speaker(&script).unwrap(), "Speaker 0");
    }

    #[test]
    fn tts_text_input_rejects_multiple_speakers_for_single_reference_voice() {
        let err = parse_tts_script("Speaker 1: One.\nSpeaker 2: Two.", "Speaker 0")
            .expect_err("multi speaker error");

        assert!(format!("{err}").contains("one reference speaker"));
    }

    #[test]
    fn tts_script_input_rejects_mixed_speaker_and_plain_lines() {
        let err = parse_tts_script("Speaker 1: One.\nplain continuation", "Speaker 0")
            .expect_err("mixed script error");

        assert!(format!("{err}").contains("Speaker N"));
    }

    #[test]
    fn asr_user_suffix_matches_reference_strings() {
        let plain = "This is a 1.00 seconds audio, please transcribe it with these keys: \
                     Start time, End time, Speaker ID, Content";
        assert_eq!(vibevoice_asr_user_suffix(1.0, None), plain);
        assert_eq!(vibevoice_asr_user_suffix(1.0, Some(" \n\t ")), plain);
        assert_eq!(
            vibevoice_asr_user_suffix(2.5, Some("  Microsoft, VibeVoice\n")),
            "This is a 2.50 seconds audio, with extra info: Microsoft, VibeVoice\n\n\
             Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        );
        // Python's `{:.2f}` rounds the exact binary value, ties to even, as
        // Rust does: 3000/24000 = 0.125 gives 0.12. The duration must be f64:
        // 1080/24000 is 0.04 in f64 but 0.05 once narrowed to f32.
        assert!(vibevoice_asr_user_suffix(3_000.0 / 24_000.0, None).starts_with("This is a 0.12 "));
        assert!(vibevoice_asr_user_suffix(1_080.0 / 24_000.0, None).starts_with("This is a 0.04 "));
        assert!(vibevoice_asr_user_suffix(f64::NAN, None).starts_with("This is a 0.00 "));
    }

    /// Prompt ids from the reference processor (VibeVoice @16fb2cb1) with the
    /// Qwen/Qwen2.5-7B tokenizer (main @d1497293) and vLLM's generation prompt.
    #[test]
    #[ignore = "needs a Qwen2.5 tokenizer: set IZWI_TEST_QWEN25_TOKENIZER to its tokenizer.json or directory"]
    fn asr_prompt_token_ids_match_reference_goldens() {
        const HEADER: [u32; 25] = [
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 429, 1356, 55136, 7699, 1946, 1119,
            1467, 2550, 304, 4718, 3561, 13, 151645, 198, 151644, 872, 198,
        ];
        const KEYS: [u32; 18] = [
            1356, 3114, 432, 448, 1493, 6894, 25, 5145, 882, 11, 3972, 882, 11, 29073, 3034, 11,
            8883, 151645,
        ];
        const TAIL: [u32; 4] = [198, 151644, 77091, 198];
        // 4486 is `Ġplease` and 5100 `ĠMicrosoft`: encoding the suffix in
        // pieces split each into `Ġ` (220) plus the bare word.
        let plain_1s = [
            198, 1986, 374, 264, 220, 16, 13, 15, 15, 6486, 7699, 11, 4486,
        ];
        let hotwords_1s = [
            198, 1986, 374, 264, 220, 16, 13, 15, 15, 6486, 7699, 11, 448, 4960, 3546, 25, 5100,
            11, 647, 23549, 51167, 271, 5501,
        ];
        let plain_2_5s = [
            198, 1986, 374, 264, 220, 17, 13, 20, 15, 6486, 7699, 11, 4486,
        ];
        let expected = |frames: usize, suffix: &[u32]| {
            let mut ids = HEADER.to_vec();
            ids.push(151646);
            ids.extend(std::iter::repeat_n(151648, frames));
            ids.push(151647);
            ids.extend_from_slice(suffix);
            ids.extend_from_slice(&KEYS);
            ids.extend_from_slice(&TAIL);
            ids
        };

        let path = std::env::var_os("IZWI_TEST_QWEN25_TOKENIZER")
            .map(std::path::PathBuf::from)
            .expect("set IZWI_TEST_QWEN25_TOKENIZER to a Qwen2.5 tokenizer.json or its directory");
        let dir = if path.is_file() {
            path.parent().expect("tokenizer directory").to_path_buf()
        } else {
            path
        };
        let tokenizer = VibeVoicePromptTokenizer::load(&dir, 152_064).expect("Qwen2.5 tokenizer");
        for (seconds, frames, extra, suffix, len) in [
            (1.0, 3, None, &plain_1s[..], 65),
            (1.0, 3, Some(" Microsoft, VibeVoice "), &hotwords_1s[..], 75),
            (2.5, 8, None, &plain_2_5s[..], 70),
        ] {
            let prompt = tokenizer.build_asr_prompt(seconds, frames, extra).unwrap();
            assert_eq!(prompt.input_ids, expected(frames, suffix));
            assert_eq!(prompt.input_ids.len(), len);
            assert_eq!(prompt.acoustic_input_range, 26..26 + frames);
        }
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

fn read_special_tokens(model_dir: &Path) -> Result<VibeVoiceSpecialTokens> {
    let path = model_dir.join("tokenizer_config.json");
    let raw = fs::read_to_string(path)?;
    let config: TokenizerConfig = serde_json::from_str(&raw)?;
    let by_token = config
        .added_tokens_decoder
        .iter()
        .filter_map(|(id, token)| {
            id.parse::<u32>()
                .ok()
                .map(|id| (token.content.as_str(), id))
        })
        .collect::<HashMap<_, _>>();
    let defaults = VibeVoiceSpecialTokens::default();
    Ok(VibeVoiceSpecialTokens {
        endoftext: by_token
            .get("<|endoftext|>")
            .copied()
            .unwrap_or(defaults.endoftext),
        im_start: by_token
            .get("<|im_start|>")
            .copied()
            .unwrap_or(defaults.im_start),
        im_end: by_token
            .get("<|im_end|>")
            .copied()
            .unwrap_or(defaults.im_end),
        object_ref_start: by_token
            .get("<|object_ref_start|>")
            .copied()
            .unwrap_or(defaults.object_ref_start),
        object_ref_end: by_token
            .get("<|object_ref_end|>")
            .copied()
            .unwrap_or(defaults.object_ref_end),
        box_start: by_token
            .get("<|box_start|>")
            .copied()
            .unwrap_or(defaults.box_start),
        speech_start: by_token
            .get("<|vision_start|>")
            .copied()
            .unwrap_or(defaults.speech_start),
        speech_end: by_token
            .get("<|vision_end|>")
            .copied()
            .unwrap_or(defaults.speech_end),
        speech_pad: by_token
            .get("<|vision_pad|>")
            .copied()
            .unwrap_or(defaults.speech_pad),
        image_pad: by_token
            .get("<|image_pad|>")
            .copied()
            .unwrap_or(defaults.image_pad),
    })
}
