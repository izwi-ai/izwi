use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

pub fn load_tokenizer_vocab(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to read tokenizer vocab {}: {}",
            path.display(),
            e
        ))
    })?;

    let mut vocab = Vec::new();
    for line in raw.lines() {
        let token = line.split('\t').next().unwrap_or("").trim();
        if token.is_empty() {
            continue;
        }
        vocab.push(token.to_string());
    }

    if vocab.is_empty() {
        return Err(Error::ModelLoadError(format!(
            "Tokenizer vocab at {} is empty",
            path.display()
        )));
    }

    Ok(vocab)
}

pub fn decode_tokens(ids: &[usize], vocab: &[String]) -> String {
    let mut out = String::new();

    for &id in ids {
        let Some(tok) = vocab.get(id) else {
            continue;
        };

        if tok == "<unk>" || tok == "<pad>" {
            continue;
        }
        if tok.starts_with("<|") && tok.ends_with("|>") {
            continue;
        }

        if tok.starts_with('▁') {
            let piece = tok.trim_start_matches('▁');
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(piece);
            continue;
        }

        if let Some(piece) = tok.strip_prefix("##") {
            out.push_str(piece);
            continue;
        }

        // SentencePiece tokens without a leading ▁ continue the current word.
        out.push_str(tok);
    }

    normalize_text(out)
}

/// Collapse whitespace and drop the space a `▁.` or `▁,` piece adds. A space
/// before `! ? : ;` is kept: the model writes French with it ("Ça va ?").
fn normalize_text(mut s: String) -> String {
    s = s.split_whitespace().collect::<Vec<_>>().join(" ");

    for punct in [".", ","] {
        s = s.replace(&format!(" {punct}"), punct);
    }

    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::decode_tokens;
    use crate::models::shared::detokenize::stream_deltas;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn incremental_deltas_concatenate_to_the_full_decode() {
        let vocab: Vec<String> = [
            "▁hel",
            "lo",
            "▁world",
            "▁",
            "▁.",
            "▁,",
            "▁?",
            ".",
            "##ing",
            "<unk>",
            "<pad>",
            "<|nospeech|>",
            "▁你",
            "好",
            "▁世界",
            "😀",
            "▁🎉",
            "▁Ça",
            "é",
            "▁!",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(0x8_3_4);
        for _ in 0..500 {
            let ids = (0..rng.random_range(0..60))
                .map(|_| rng.random_range(0..vocab.len() + 2))
                .collect::<Vec<_>>();
            let (deltas, detokenizer) =
                stream_deltas(&ids, |ids: &[usize]| Ok(decode_tokens(ids, &vocab)));
            assert_eq!(deltas.concat(), decode_tokens(&ids, &vocab), "{ids:?}");
            assert_eq!(detokenizer.divergences(), 0, "{ids:?}");
        }
    }

    #[test]
    fn detokenizer_keeps_french_spacing_before_high_punctuation() {
        let vocab: Vec<String> = [
            "▁Bonjour",
            "▁!",
            "▁Ça",
            "▁va",
            "▁?",
            "▁Oui",
            "▁,",
            "▁merci",
            "▁.",
            "▁Note",
            "▁:",
            "▁fin",
            "▁;",
            "▁ok",
            "?",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let ids: Vec<usize> = (0..vocab.len()).collect();

        assert_eq!(
            decode_tokens(&ids, &vocab),
            "Bonjour ! Ça va ? Oui, merci. Note : fin ; ok?"
        );
    }
}
