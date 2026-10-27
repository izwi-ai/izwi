//! The multilingual checkpoint's Chinese frontend (`ChineseCangjieConverter`
//! in upstream's `models/tokenizers/tokenizer.py`): words segmented by
//! spacy-pkuseg and joined by spaces, then every Lo character the Cangjie
//! table knows spelled as `[cj_…]` tokens ending in `[cj_.]`.
//!
//! The segmenter ports spacy-pkuseg 1.0.2's `pkuseg.cut`
//! (<https://github.com/explosion/spacy-pkuseg>, MIT; Copyright (c) 2020
//! ExplosionAI GmbH, 2018-2019 pkuseg authors) with what `pkuseg()` loads by
//! default: the `spacy_ontonotes` CRF and the post-processing dictionary.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::Path;

use icu_properties::props::GeneralCategory;
use icu_properties::CodePointMapData;
use serde::Deserialize;

use super::text::{python_split, python_strip};
use crate::error::{Error, Result};

/// The Cangjie table upstream downloads from the checkpoint's repository.
pub const CANGJIE_FILE: &str = "Cangjie5_TC.json";
/// spacy-pkuseg's default model, as its release ships it.
pub const PKUSEG_MODEL_FILE: &str = "spacy_ontonotes.zip";
/// spacy-pkuseg's `dicts/default.pkl`, which merges segmented words back.
pub const PKUSEG_DICT_FILE: &str = "pkuseg_default.pkl";

pub struct ChineseFrontend {
    segmenter: Pkuseg,
    cangjie: CangjieTable,
}

impl ChineseFrontend {
    pub fn load(dir: &Path) -> Result<Self> {
        let cangjie = std::fs::read_to_string(dir.join(CANGJIE_FILE))
            .map_err(|error| missing(CANGJIE_FILE, error))?;
        Ok(Self {
            segmenter: Pkuseg::load(&dir.join(PKUSEG_MODEL_FILE), &dir.join(PKUSEG_DICT_FILE))?,
            cangjie: CangjieTable::parse(&cangjie)?,
        })
    }

    /// Upstream's `ChineseCangjieConverter.__call__` on already lowercased,
    /// NFKD text.
    pub fn convert(&self, text: &str) -> String {
        let words = self.segmenter.cut(text).join(" ");
        let mut out = String::with_capacity(words.len() * 4);
        for c in words.chars() {
            match self.cangjie.spell(c) {
                Some(tokens) if is_other_letter(c) => out.push_str(tokens),
                _ => out.push(c),
            }
        }
        out
    }
}

fn missing(file: &str, error: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!(
        "Chatterbox Multilingual's Chinese frontend cannot read {file} ({error}); download the \
         model again"
    ))
}

fn invalid(file: &str, detail: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("Chatterbox {file}: {detail}"))
}

/// `unicodedata.category(c) == "Lo"`.
fn is_other_letter(c: char) -> bool {
    CodePointMapData::<GeneralCategory>::new().get(c) == GeneralCategory::OtherLetter
}

/// Each glyph of the table as the tokens upstream's `_cangjie_encode` gives
/// it: its code, then its position among the entries sharing that code when
/// it is not the first.
struct CangjieTable {
    spelled: HashMap<char, String>,
}

impl CangjieTable {
    fn parse(json: &str) -> Result<Self> {
        let entries: Vec<String> =
            serde_json::from_str(json).map_err(|error| invalid(CANGJIE_FILE, error))?;
        // Later entries for a word replace its code; every entry joins its
        // code's list, as upstream's `word2cj` and `cj2word` do.
        let mut word_code = HashMap::<&str, &str>::new();
        let mut code_words = HashMap::<&str, Vec<&str>>::new();
        for entry in &entries {
            let mut fields = entry.split('\t');
            let (Some(word), Some(code)) = (fields.next(), fields.next()) else {
                return Err(invalid(
                    CANGJIE_FILE,
                    format!("entry {entry:?} has no code"),
                ));
            };
            word_code.insert(word, code);
            code_words.entry(code).or_default().push(word);
        }
        let mut spelled = HashMap::with_capacity(word_code.len());
        for (word, code) in word_code {
            let mut chars = word.chars();
            let (Some(glyph), None) = (chars.next(), chars.next()) else {
                continue;
            };
            let index = code_words[code]
                .iter()
                .position(|listed| *listed == word)
                .expect("every word is listed under its code");
            let index = if index > 0 {
                index.to_string()
            } else {
                String::new()
            };
            let mut tokens = String::new();
            for c in code.chars().chain(index.chars()).chain(['.']) {
                tokens.push_str("[cj_");
                tokens.push(c);
                tokens.push(']');
            }
            spelled.insert(glyph, tokens);
        }
        Ok(Self { spelled })
    }

    fn spell(&self, c: char) -> Option<&str> {
        self.spelled.get(&c).map(String::as_str)
    }
}

/// spacy-pkuseg's segmenter: a first-order linear-chain CRF over character
/// features, then the dictionary merge (`Postprocesser`).
struct Pkuseg {
    feature_ids: HashMap<Box<str>, u32>,
    words: HashSet<Box<str>>,
    /// `tags × features` node weights, then `tags × tags` edge weights.
    weights: Vec<f64>,
    tags: usize,
    features: usize,
    /// Whether each tag begins a word (its name contains "B").
    begins_word: Vec<bool>,
    merges: HashSet<Box<str>>,
}

/// `features.msgpack`; its bigram list is only used in training.
#[derive(Deserialize)]
struct PkusegFeatures {
    unigram: Vec<Box<str>>,
    feature_to_idx: HashMap<Box<str>, u32>,
    tag_to_idx: HashMap<String, u32>,
}

/// spacy-pkuseg's `config.wordMax` and `config.wordMin`.
const WORD_LENGTHS: [usize; 5] = [6, 5, 4, 3, 2];
const NO_WORD: &str = "**noWord";

impl Pkuseg {
    fn load(model: &Path, dict: &Path) -> Result<Self> {
        let file = File::open(model).map_err(|error| missing(PKUSEG_MODEL_FILE, error))?;
        let mut archive = zip::ZipArchive::new(BufReader::new(file))
            .map_err(|error| invalid(PKUSEG_MODEL_FILE, error))?;
        let features: PkusegFeatures = {
            let entry = archive
                .by_name("features.msgpack")
                .map_err(|error| invalid(PKUSEG_MODEL_FILE, error))?;
            rmp_serde::from_read(BufReader::new(entry))
                .map_err(|error| invalid(PKUSEG_MODEL_FILE, error))?
        };
        let mut npz = Vec::new();
        archive
            .by_name("weights.npz")
            .and_then(|mut entry| Ok(entry.read_to_end(&mut npz)?))
            .map_err(|error| invalid(PKUSEG_MODEL_FILE, error))?;
        let mut npz = zip::ZipArchive::new(Cursor::new(npz))
            .map_err(|error| invalid(PKUSEG_MODEL_FILE, error))?;
        let sizes = read_npy(&mut npz, "sizes.npy", "<i8")?;
        let weights = read_npy(&mut npz, "w.npy", "<f8")?;
        let [tags, feature_count] = sizes[..] else {
            return Err(invalid(
                PKUSEG_MODEL_FILE,
                "sizes.npy is not [tags, features]",
            ));
        };
        let (tags, feature_count) = (
            i64::from_le_bytes(tags) as usize,
            i64::from_le_bytes(feature_count) as usize,
        );
        if weights.len() != tags * (feature_count + tags)
            || features.tag_to_idx.len() != tags
            || features.feature_to_idx.len() != feature_count
            || features
                .feature_to_idx
                .values()
                .any(|&id| id as usize >= feature_count)
        {
            return Err(invalid(PKUSEG_MODEL_FILE, "features and weights disagree"));
        }
        let mut begins_word = vec![false; tags];
        for (tag, &id) in &features.tag_to_idx {
            *begins_word
                .get_mut(id as usize)
                .ok_or_else(|| invalid(PKUSEG_MODEL_FILE, format!("tag {tag} out of range")))? =
                tag.contains('B');
        }
        let merges = read_pickled_str(dict)?;
        Ok(Self {
            feature_ids: features.feature_to_idx,
            words: features.unigram.into_iter().collect(),
            weights: weights.into_iter().map(f64::from_le_bytes).collect(),
            tags,
            features: feature_count,
            begins_word,
            merges: python_strip(&merges).split('\n').map(Box::from).collect(),
        })
    }

    /// `pkuseg.cut`: whitespace-separated chunks, each segmented and merged.
    fn cut(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        for chunk in python_split(python_strip(text)) {
            let chunk: Vec<char> = chunk.chars().collect();
            words.extend(self.merge(self.segment(&chunk)));
        }
        words
    }

    /// `pkuseg._cut`: Viterbi tags over the chunk, a word starting at each
    /// tag that begins one.
    fn segment(&self, chars: &[char]) -> Vec<String> {
        let nodes: Vec<String> = chars.iter().map(|&c| node(c)).collect();
        let mut key = String::new();
        let features: Vec<Vec<u32>> = (0..nodes.len())
            .map(|index| self.node_features(&nodes, index, &mut key))
            .collect();
        let tags = self.viterbi(&features);
        let mut words: Vec<String> = Vec::new();
        for (position, (&c, &tag)) in chars.iter().zip(&tags).enumerate() {
            match words.last_mut() {
                Some(word) if position > 0 && !self.begins_word[tag] => word.push(c),
                _ => words.push(c.to_string()),
            }
        }
        words
    }

    fn feature(&self, key: &str, out: &mut Vec<u32>) {
        if let Some(&id) = self.feature_ids.get(key) {
            out.push(id);
        }
    }

    /// `__get_node_features_idx`, in its order: the bias, the character
    /// n-grams, then the dictionary-word features.
    fn node_features(&self, nodes: &[String], i: usize, key: &mut String) -> Vec<u32> {
        let mut out = vec![0];
        let mut lookup = |parts: &[&str], out: &mut Vec<u32>| {
            key.clear();
            parts.iter().for_each(|part| key.push_str(part));
            self.feature(key, out);
        };
        let len = nodes.len();
        let c = nodes[i].as_str();
        lookup(&["c.", c], &mut out);
        if i > 0 {
            let prev = nodes[i - 1].as_str();
            lookup(&["c-1.", prev], &mut out);
            lookup(&["c-1c.", prev, ".", c], &mut out);
        }
        if i + 1 < len {
            let next = nodes[i + 1].as_str();
            lookup(&["c1.", next], &mut out);
            lookup(&["cc1.", c, ".", next], &mut out);
        }
        if i > 1 {
            let before = nodes[i - 2].as_str();
            lookup(&["c-2.", before], &mut out);
            lookup(&["c-2c-1.", before, ".", &nodes[i - 1]], &mut out);
        }
        if i + 2 < len {
            lookup(&["c2.", &nodes[i + 2]], &mut out);
        }

        // Words ending at, starting at, ending before and starting after i.
        let word = |start: isize, length: usize| -> Option<String> {
            let start = usize::try_from(start).ok().filter(|&start| start < len)?;
            if start + length > len {
                return None;
            }
            let word = nodes[start..start + length].concat();
            self.words.contains(word.as_str()).then_some(word)
        };
        let i = i as isize;
        let ending_at: Vec<Option<String>> = WORD_LENGTHS
            .iter()
            .map(|&l| word(i - l as isize + 1, l))
            .collect();
        for found in ending_at.iter().flatten() {
            lookup(&["w-1.", found], &mut out);
        }
        let starting_at: Vec<Option<String>> = WORD_LENGTHS.iter().map(|&l| word(i, l)).collect();
        for found in starting_at.iter().flatten() {
            lookup(&["w1.", found], &mut out);
        }
        let ending_before: Vec<Option<String>> = WORD_LENGTHS
            .iter()
            .map(|&l| word(i - l as isize, l))
            .collect();
        let starting_after: Vec<Option<String>> =
            WORD_LENGTHS.iter().map(|&l| word(i + 1, l)).collect();
        let name = |word: &Option<String>| word.as_deref().unwrap_or(NO_WORD).to_string();
        for pre in &ending_before {
            for post in &starting_at {
                lookup(&["ww.l.", &name(pre), "*", &name(post)], &mut out);
            }
        }
        for pre in &ending_at {
            for post in &starting_after {
                lookup(&["ww.r.", &name(pre), "*", &name(post)], &mut out);
            }
        }
        out
    }

    /// `decodeViterbi_fast`: scores summed in upstream's order (edges start
    /// at 1), decoded right to left with its tie-breaking.
    fn viterbi(&self, features: &[Vec<u32>]) -> Vec<usize> {
        let tags = self.tags;
        let width = features.len();
        let mut node = vec![0f64; width * tags];
        for (position, ids) in features.iter().enumerate() {
            for &id in ids {
                for tag in 0..tags {
                    node[position * tags + tag] += self.weights[id as usize * tags + tag];
                }
            }
        }
        let edges = self.features * tags;
        // edge(y, next) = edge_score[y, next] = 1 + w[edges + next·tags + y]
        let edge = |tag: usize, next: usize| 1.0 + self.weights[edges + next * tags + tag];

        let mut best = vec![0f64; width * tags];
        let mut from = vec![0usize; width * tags];
        best[(width - 1) * tags..].copy_from_slice(&node[(width - 1) * tags..]);
        for position in (0..width - 1).rev() {
            for tag in 0..tags {
                for next in 0..tags {
                    let score = best[(position + 1) * tags + next]
                        + node[position * tags + tag]
                        + edge(tag, next);
                    let cell = position * tags + tag;
                    if next == 0 || score >= best[cell] {
                        best[cell] = score;
                        from[cell] = next;
                    }
                }
            }
        }
        let mut tag = 0;
        for candidate in 1..tags {
            if best[tag] < best[candidate] {
                tag = candidate;
            }
        }
        let mut states = vec![tag; width];
        for position in 1..width {
            tag = from[(position - 1) * tags + tag];
            states[position] = tag;
        }
        states
    }

    /// `Postprocesser.post_process` with only the default dictionary: runs
    /// of 7 down to 2 words whose concatenation it lists are merged, unless
    /// it lists every one of them too.
    fn merge(&self, mut words: Vec<String>) -> Vec<String> {
        for span in (2..8).rev() {
            let mut i = 0;
            while i + span <= words.len() {
                let merged = words[i..i + span].concat();
                if self.merges.contains(merged.as_str())
                    && !words[i..i + span]
                        .iter()
                        .all(|word| self.merges.contains(word.as_str()))
                {
                    words.splice(i..i + span, [merged]);
                }
                i += 1;
            }
        }
        words
    }
}

/// `FeatureExtractor.normalize_text` for one character: keywords become
/// `&`, then digits and full-width letters collapse to classes.
fn node(c: char) -> String {
    const NUMBERS: &str = "0123456789.几二三四五六七八九十千万亿兆零１２３４５６７８９０％";
    // Upstream's set, with its repeated ｇ and missing ｊ.
    const LETTERS: &str = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ\
                           ａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ／・－";
    let c = if "-._,|/*:".contains(c) { '&' } else { c };
    if NUMBERS.contains(c) {
        "**Num".to_string()
    } else if LETTERS.contains(c) {
        "**Letter".to_string()
    } else {
        c.to_string()
    }
}

/// The little-endian elements of a one-dimensional `.npy` member of
/// `archive` whose dtype is `descr`, as raw 8-byte words.
fn read_npy<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
    descr: &str,
) -> Result<Vec<[u8; 8]>> {
    let bad = |detail: &str| invalid(PKUSEG_MODEL_FILE, format!("{name}: {detail}"));
    let mut bytes = Vec::new();
    archive
        .by_name(name)
        .and_then(|mut entry| Ok(entry.read_to_end(&mut bytes)?))
        .map_err(|error| invalid(PKUSEG_MODEL_FILE, error))?;
    let body = bytes
        .strip_prefix(b"\x93NUMPY\x01\x00")
        .ok_or_else(|| bad("not a version 1.0 .npy"))?;
    let header_len = u16::from_le_bytes([body[0], body[1]]) as usize;
    let header = std::str::from_utf8(
        body.get(2..2 + header_len)
            .ok_or_else(|| bad("short header"))?,
    )
    .map_err(|_| bad("header is not text"))?;
    if !header.contains(&format!("'descr': '{descr}'"))
        || !header.contains("'fortran_order': False")
    {
        return Err(bad(&format!("expected a C-ordered {descr} array")));
    }
    let count: usize = header
        .split_once("'shape': (")
        .and_then(|(_, rest)| rest.split_once(",)"))
        .and_then(|(count, _)| count.parse().ok())
        .ok_or_else(|| bad("expected a one-dimensional shape"))?;
    let data = &body[2 + header_len..];
    if data.len() != count * 8 {
        return Err(bad("data does not match its shape"));
    }
    Ok(data.as_chunks::<8>().0.to_vec())
}

/// `dicts/default.pkl` is a protocol-3 pickle of one string; it is read as
/// exactly that (PROTO 3, BINUNICODE, BINPUT 0, STOP) and nothing is
/// unpickled.
fn read_pickled_str(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|error| missing(PKUSEG_DICT_FILE, error))?;
    let bad = || invalid(PKUSEG_DICT_FILE, "expected a pickled string");
    let body = bytes.strip_prefix(b"\x80\x03X").ok_or_else(bad)?;
    let len = u32::from_le_bytes(body.get(..4).ok_or_else(bad)?.try_into().expect("4 bytes"));
    let (text, tail) = body[4..].split_at_checked(len as usize).ok_or_else(bad)?;
    if tail != b"q\x00." {
        return Err(bad());
    }
    String::from_utf8(text.to_vec()).map_err(|_| bad())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cangjie_spells_codes_with_their_index() {
        let table = CangjieTable::parse(
            r#"["日\ta", "曰\ta", "昌\taa", "林\tdd\textra", "x\tq", "日月\tab"]"#,
        )
        .unwrap();
        assert_eq!(table.spell('日'), Some("[cj_a][cj_.]"));
        assert_eq!(table.spell('曰'), Some("[cj_a][cj_1][cj_.]"));
        assert_eq!(table.spell('林'), Some("[cj_d][cj_d][cj_.]"));
        assert_eq!(table.spell('月'), None);
    }

    /// A later entry moves a glyph to another code; its index counts every
    /// word listed under that code before it, phrases included.
    #[test]
    fn cangjie_follows_upstreams_dictionaries() {
        let table = CangjieTable::parse(r#"["甲\tx", "乙\ty", "丙丁\ty", "甲\ty"]"#).unwrap();
        assert_eq!(table.spell('甲'), Some("[cj_y][cj_2][cj_.]"));
        assert_eq!(table.spell('乙'), Some("[cj_y][cj_.]"));
    }

    #[test]
    fn nodes_normalize_like_pkuseg() {
        assert_eq!(node('.'), "&");
        assert_eq!(node(','), "&");
        assert_eq!(node('7'), "**Num");
        assert_eq!(node('三'), "**Num");
        assert_eq!(node('Ｇ'), "**Letter");
        assert_eq!(node('ｊ'), "ｊ");
        assert_eq!(node('a'), "a");
        assert_eq!(node('中'), "中");
    }

    #[test]
    fn only_other_letters_are_spelled() {
        assert!(is_other_letter('中'));
        assert!(is_other_letter('か'));
        assert!(!is_other_letter('〇'));
        assert!(!is_other_letter('「'));
        assert!(!is_other_letter('a'));
    }

    #[test]
    fn the_merge_dictionary_is_read_without_unpickling() {
        let dir = std::env::temp_dir().join(format!("izwi-pkuseg-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(PKUSEG_DICT_FILE);
        let text = "中国\n人民\n";
        let mut pickle = b"\x80\x03X".to_vec();
        pickle.extend((text.len() as u32).to_le_bytes());
        pickle.extend(text.as_bytes());
        pickle.extend(b"q\x00.");
        std::fs::write(&path, &pickle).unwrap();
        assert_eq!(read_pickled_str(&path).unwrap(), text);
        let stop = pickle.len() - 1;
        pickle[stop] = b'R';
        std::fs::write(&path, &pickle).unwrap();
        assert!(read_pickled_str(&path).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
