//! Keeping a transcript in the alphabet of the language that was asked for.
//!
//! Parakeet v3 has no language input: it picks the language from the audio,
//! and on a short clip it sometimes picks wrong, writing English speech in
//! Cyrillic ("Ютюб" for "YouTube"). When a request names its language, the
//! tokens written in another alphabet are taken out of the joint's choice,
//! so the decoder falls back to its best spelling in the right one.
//!
//! A request with no language, or a model whose vocabulary has one alphabet
//! (v2 is English only), decodes exactly as before.

use candle_core::{DType, Device, Tensor};

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Script {
    Latin,
    Cyrillic,
    Greek,
}

const SCRIPTS: [Script; 3] = [Script::Latin, Script::Cyrillic, Script::Greek];

/// The alphabet `language` is written in: a name ("English") or a code
/// ("en", "en-US") of one of the languages Parakeet v3 transcribes. `None`
/// for `auto` and for anything else, which leaves the decoder unrestricted.
pub(super) fn language_script(language: &str) -> Option<Script> {
    let language = language.trim().to_ascii_lowercase();
    let code = language.split(['-', '_']).next().unwrap_or("");
    let script = match code {
        "bg" | "bulgarian" | "ru" | "russian" | "uk" | "ukrainian" => Script::Cyrillic,
        "el" | "greek" => Script::Greek,
        "hr" | "croatian" | "cs" | "czech" | "da" | "danish" | "nl" | "dutch" | "en"
        | "english" | "et" | "estonian" | "fi" | "finnish" | "fr" | "french" | "de" | "german"
        | "hu" | "hungarian" | "it" | "italian" | "lv" | "latvian" | "lt" | "lithuanian" | "mt"
        | "maltese" | "pl" | "polish" | "pt" | "portuguese" | "ro" | "romanian" | "sk"
        | "slovak" | "sl" | "slovenian" | "es" | "spanish" | "sv" | "swedish" => Script::Latin,
        _ => return None,
    };
    Some(script)
}

fn char_script(c: char) -> Option<Script> {
    if !c.is_alphabetic() {
        return None;
    }
    match u32::from(c) {
        0x0041..=0x024F | 0x1E00..=0x1EFF => Some(Script::Latin),
        0x0400..=0x052F => Some(Script::Cyrillic),
        0x0370..=0x03FF | 0x1F00..=0x1FFF => Some(Script::Greek),
        _ => None,
    }
}

/// Whether a request in `script` may emit `piece`. A Latin-script language
/// takes no Cyrillic or Greek letters. Cyrillic and Greek text still quotes
/// Latin names, so those only lose each other's letters.
fn allowed(piece: &str, script: Script) -> bool {
    piece
        .chars()
        .filter_map(char_script)
        .all(|found| found == script || (found == Script::Latin && script != Script::Latin))
}

/// Per alphabet, what is added to a row of joint logits: negative infinity
/// on the tokens that alphabet may not emit, zero elsewhere (the blank and
/// the durations included).
pub(super) struct ScriptPenalties {
    /// `[1, width]` each; `None` where nothing is banned.
    penalties: [Option<Tensor>; 3],
    /// `[1, width]` zeros, for the unrestricted rows of a mixed batch.
    zero: Option<Tensor>,
}

impl ScriptPenalties {
    /// No restriction, whatever the language.
    pub(super) fn none() -> Self {
        Self {
            penalties: [None, None, None],
            zero: None,
        }
    }

    /// From the vocabulary: `pieces[id]` is token `id`'s text. `width` is
    /// the joint's output width; ids past the pieces are never banned.
    pub(super) fn build(
        pieces: &[String],
        width: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut penalties = [None, None, None];
        for (slot, script) in penalties.iter_mut().zip(SCRIPTS) {
            let mut row = vec![0f32; width];
            let mut banned = 0usize;
            for (value, piece) in row.iter_mut().zip(pieces) {
                if !allowed(piece, script) {
                    *value = f32::NEG_INFINITY;
                    banned += 1;
                }
            }
            if banned > 0 {
                *slot = Some(Tensor::from_vec(row, (1, width), device)?.to_dtype(dtype)?);
            }
        }
        let zero = if penalties.iter().any(Option::is_some) {
            Some(Tensor::zeros((1, width), dtype, device)?)
        } else {
            None
        };
        Ok(Self { penalties, zero })
    }

    /// The penalty row `[1, width]` of a request in `language`, if it has one.
    pub(super) fn for_language(&self, language: Option<&str>) -> Option<&Tensor> {
        let script = language_script(language?)?;
        let index = SCRIPTS.iter().position(|known| *known == script)?;
        self.penalties[index].as_ref()
    }

    /// The penalties `[rows, width]` of a batch whose rows asked for
    /// `languages`, or `None` when no row is restricted.
    pub(super) fn for_rows<'a>(
        &self,
        languages: impl Iterator<Item = Option<&'a str>>,
    ) -> Result<Option<Tensor>> {
        let rows = languages
            .map(|language| self.for_language(language))
            .collect::<Vec<_>>();
        let (Some(zero), true) = (self.zero.as_ref(), rows.iter().any(Option::is_some)) else {
            return Ok(None);
        };
        let rows = rows
            .into_iter()
            .map(|row| row.unwrap_or(zero))
            .collect::<Vec<_>>();
        Ok(Some(Tensor::cat(&rows, 0)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_map_to_their_alphabet_by_name_or_code() {
        assert_eq!(language_script("English"), Some(Script::Latin));
        assert_eq!(language_script(" en-US "), Some(Script::Latin));
        assert_eq!(language_script("de"), Some(Script::Latin));
        assert_eq!(language_script("Russian"), Some(Script::Cyrillic));
        assert_eq!(language_script("uk"), Some(Script::Cyrillic));
        assert_eq!(language_script("Greek"), Some(Script::Greek));
        assert_eq!(language_script("auto"), None);
        assert_eq!(language_script("Chinese"), None);
        assert_eq!(language_script(""), None);
    }

    #[test]
    fn latin_languages_lose_other_alphabets_and_the_others_keep_latin() {
        assert!(allowed("\u{2581}you", Script::Latin));
        assert!(allowed("ça", Script::Latin));
        assert!(allowed("\u{2581}", Script::Latin));
        assert!(allowed("1,5%", Script::Latin));
        assert!(!allowed("\u{2581}ют", Script::Latin));
        assert!(!allowed("αι", Script::Latin));
        assert!(allowed("\u{2581}ют", Script::Cyrillic));
        assert!(allowed("\u{2581}you", Script::Cyrillic));
        assert!(!allowed("αι", Script::Cyrillic));
        assert!(allowed("αι", Script::Greek));
        assert!(!allowed("ют", Script::Greek));
    }

    #[test]
    fn penalties_ban_only_foreign_tokens_and_only_for_named_languages() {
        let pieces = ["\u{2581}a", "б", "γ", "."].map(String::from);
        // Width 7: four tokens, the blank and two durations.
        let scripts = ScriptPenalties::build(&pieces, 7, DType::F32, &Device::Cpu).unwrap();
        let row = |language: &str| {
            scripts
                .for_language(Some(language))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()
                .remove(0)
        };
        let ninf = f32::NEG_INFINITY;
        assert_eq!(row("en"), [0.0, ninf, ninf, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(row("Russian"), [0.0, 0.0, ninf, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(row("el"), [0.0, ninf, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(scripts.for_language(None).is_none());
        assert!(scripts.for_language(Some("auto")).is_none());

        // A batch is restricted row by row; one without a named language
        // has no penalty at all.
        let batch = scripts
            .for_rows([None, Some("en")].into_iter())
            .unwrap()
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(batch[0], [0.0; 7]);
        assert_eq!(batch[1], row("en"));
        assert!(scripts
            .for_rows([None, Some("auto")].into_iter())
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_one_alphabet_vocabulary_is_never_restricted() {
        let pieces = ["\u{2581}a", "b", "'"].map(String::from);
        let scripts = ScriptPenalties::build(&pieces, 6, DType::F32, &Device::Cpu).unwrap();
        assert!(scripts.for_language(Some("en")).is_none());
        assert!(scripts
            .for_rows([Some("en")].into_iter())
            .unwrap()
            .is_none());
        assert!(ScriptPenalties::none().for_language(Some("en")).is_none());
    }
}
