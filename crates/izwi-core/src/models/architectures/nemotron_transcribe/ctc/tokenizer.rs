//! The CTC head's SentencePiece tokenizer, read from the serialized
//! `ModelProto` the checkpoint embeds (`ctc_timestamp.tokenizer.model_proto`).
//!
//! The aligner re-tokenises the decoded transcript with this model, so its
//! ids must be exactly sentencepiece's `encode_as_ids` (NeMo's
//! `SentencePieceTokenizer.text_to_ids` with `legacy=False`). This is a port
//! of the three pieces of sentencepiece that path runs, not an approximation:
//!
//! * **Normalizer** (`normalizer.cc`): user-defined symbols pass through
//!   verbatim; elsewhere the longest rule of the precompiled charsmap (a
//!   darts-clone double-array trie plus a NUL-separated replacement blob,
//!   here `nmt_nfkc`) replaces the prefix, else one UTF-8 character is
//!   copied; then the dummy prefix, extra-whitespace removal and `▁`
//!   escaping as the `NormalizerSpec` sets them. `tokenizers`' `Precompiled`
//!   normalizer is not used: it matches grapheme by grapheme and takes the
//!   shortest trie hit, which is not what sentencepiece does.
//! * **BPE** (`bpe_model.cc`): one symbol per character (a user-defined
//!   symbol is one frozen symbol), then repeatedly the adjacent pair whose
//!   concatenation is a piece with the highest score, the leftmost on ties;
//!   pieces of type UNUSED are re-split along the merges that formed them.
//! * **Ids**: a symbol that is not a piece is `<unk>`; runs of unknown
//!   symbols collapse into one `<unk>` as sentencepiece emits them (bytes
//!   instead when the model enables byte fallback).
//!
//! Only BPE models are accepted; the CTC model is BPE (23,862 pieces,
//! `nmt_nfkc`, `remove_extra_whitespaces: false`, no byte fallback).
//! The protobuf reader is hand-rolled and reads only the fields used here.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::error::{Error, Result};

/// sentencepiece's whitespace marker, U+2581.
const SPACE_SYMBOL: &str = "\u{2581}";
/// sentencepiece records at most this many charsmap trie hits per prefix.
const MAX_TRIE_RESULTS: usize = 32;

/// `ModelProto.SentencePiece.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl PieceType {
    fn from_wire(value: u64) -> Result<Self> {
        Ok(match value {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            other => {
                return Err(proto_error(format!("unknown piece type {other}")));
            }
        })
    }
}

#[derive(Debug, Clone)]
struct Piece {
    text: String,
    score: f32,
    kind: PieceType,
}

/// The CTC SentencePiece BPE tokenizer.
#[derive(Debug, Clone)]
pub struct CtcTokenizer {
    pieces: Vec<Piece>,
    /// NORMAL, USER_DEFINED and UNUSED pieces: what BPE may merge into.
    normal: HashMap<String, u32>,
    /// UNKNOWN, CONTROL and BYTE pieces.
    reserved: HashMap<String, u32>,
    unk_id: u32,
    byte_fallback: bool,
    unk_surface: String,
    normalizer: Normalizer,
    user_symbols: PrefixMatcher,
}

impl CtcTokenizer {
    /// Parses a serialized SentencePiece `ModelProto` (the bytes of a
    /// `.model` file). Fails on malformed protobuf, a non-BPE model, a
    /// duplicate piece, a missing or repeated `<unk>`, or a broken charsmap.
    pub fn from_model_proto(bytes: &[u8]) -> Result<Self> {
        let proto = ModelProto::parse(bytes)?;
        if proto.model_type != MODEL_TYPE_BPE {
            return Err(proto_error(format!(
                "model_type {} is not BPE (2); only BPE CTC tokenizers are supported",
                proto.model_type
            )));
        }
        if proto.pieces.is_empty() || proto.pieces.len() > u32::MAX as usize {
            return Err(proto_error(format!(
                "{} pieces is not a usable vocabulary",
                proto.pieces.len()
            )));
        }
        let mut normal = HashMap::new();
        let mut reserved = HashMap::new();
        let mut unk_id = None;
        let mut user_defined = Vec::new();
        for (id, piece) in proto.pieces.iter().enumerate() {
            let id = id as u32;
            if piece.text.is_empty() {
                return Err(proto_error(format!("piece {id} is empty")));
            }
            let map = match piece.kind {
                PieceType::Normal | PieceType::UserDefined | PieceType::Unused => &mut normal,
                PieceType::Unknown | PieceType::Control | PieceType::Byte => &mut reserved,
            };
            if map.insert(piece.text.clone(), id).is_some() {
                return Err(proto_error(format!(
                    "piece {:?} is defined twice",
                    piece.text
                )));
            }
            match piece.kind {
                PieceType::UserDefined => user_defined.push(piece.text.clone()),
                PieceType::Unknown if unk_id.is_some() => {
                    return Err(proto_error("more than one <unk> piece".to_string()));
                }
                PieceType::Unknown => unk_id = Some(id),
                _ => {}
            }
        }
        let unk_id = unk_id.ok_or_else(|| proto_error("no <unk> piece".to_string()))?;
        let normalizer = Normalizer::new(&proto.normalizer, proto.treat_whitespace_as_suffix)?;
        Ok(Self {
            pieces: proto.pieces,
            normal,
            reserved,
            unk_id,
            byte_fallback: proto.byte_fallback,
            unk_surface: proto.unk_surface,
            normalizer,
            user_symbols: PrefixMatcher::new(user_defined),
        })
    }

    /// Token ids of `text`, exactly sentencepiece's `encode_as_ids`
    /// (no BOS/EOS, no sampling).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalize(text);
        let mut ids = Vec::new();
        for (piece, id) in self.bpe(&normalized) {
            if id != self.unk_id {
                ids.push(id);
            } else if self.byte_fallback {
                for byte in piece.bytes() {
                    ids.push(self.piece_to_id(&format!("<0x{byte:02X}>")));
                }
            } else if ids.last() != Some(&self.unk_id) {
                // A run of unknown symbols is one `<unk>`.
                ids.push(id);
            }
        }
        ids
    }

    /// Number of pieces (the CTC head has one more class, the blank).
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether the model has no pieces (never true for a parsed model).
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The id of `piece`, or `<unk>`'s when it is not a piece
    /// (sentencepiece's `piece_to_id`).
    pub fn piece_to_id(&self, piece: &str) -> u32 {
        self.reserved
            .get(piece)
            .or_else(|| self.normal.get(piece))
            .copied()
            .unwrap_or(self.unk_id)
    }

    /// The piece of `id`, or `None` past the vocabulary.
    pub fn id_to_piece(&self, id: u32) -> Option<&str> {
        self.pieces
            .get(id as usize)
            .map(|piece| piece.text.as_str())
    }

    /// The `<unk>` id.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// The text sentencepiece's `decode` gives for `ids`, as far as the
    /// aligner's character check reads it: control pieces vanish, `<unk>`
    /// is its surface (` ⁇ `), byte pieces are their bytes (invalid UTF-8
    /// replaced), others are the piece with `▁` as a space, the first
    /// piece's leading `▁` dropped. Ids past the vocabulary are skipped.
    pub(crate) fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        let mut bytes = Vec::new();
        let mut first = true;
        for &id in ids {
            let Some(piece) = self.pieces.get(id as usize) else {
                continue;
            };
            if piece.kind == PieceType::Byte {
                if let Some(byte) = byte_piece_value(&piece.text) {
                    bytes.push(byte);
                    continue;
                }
            }
            if !bytes.is_empty() {
                text.push_str(&String::from_utf8_lossy(&bytes));
                bytes.clear();
                first = false;
            }
            match piece.kind {
                PieceType::Control => {}
                PieceType::Unknown => {
                    text.push_str(&self.unk_surface);
                    first = false;
                }
                _ => {
                    let mut surface = piece.text.as_str();
                    if first && (self.normalizer.add_dummy_prefix || self.normalizer.remove_extra) {
                        surface = surface.strip_prefix(SPACE_SYMBOL).unwrap_or(surface);
                    }
                    text.push_str(&surface.replace(SPACE_SYMBOL, " "));
                    first = false;
                }
            }
        }
        if !bytes.is_empty() {
            text.push_str(&String::from_utf8_lossy(&bytes));
        }
        text
    }

    /// sentencepiece's normalized form of `text` (`▁` for spaces).
    pub(crate) fn normalize(&self, text: &str) -> String {
        self.normalizer.normalize(text, &self.user_symbols)
    }

    /// `bpe_model.cc`'s `Model::Encode`: the final pieces of `normalized`
    /// and their ids (`<unk>`'s for a symbol that is not a piece).
    fn bpe<'a>(&self, normalized: &'a str) -> Vec<(&'a str, u32)> {
        let mut symbols = Vec::new();
        let mut offset = 0;
        while offset < normalized.len() {
            let rest = &normalized[offset..];
            let (len, freeze) = match self.user_symbols.longest(rest) {
                Some(len) => (len, true),
                None => (rest.chars().next().map_or(1, char::len_utf8), false),
            };
            let index = symbols.len() as isize;
            symbols.push(Symbol {
                start: offset,
                len,
                prev: index - 1,
                next: if offset + len < normalized.len() {
                    index + 1
                } else {
                    -1
                },
                freeze,
            });
            offset += len;
        }
        if symbols.is_empty() {
            return Vec::new();
        }

        let mut agenda = BinaryHeap::new();
        // Merged UNUSED pieces and the two symbols that formed them.
        let mut rev_merge: HashMap<&'a str, (&'a str, &'a str)> = HashMap::new();
        let mut add_pair =
            |symbols: &[Symbol], agenda: &mut BinaryHeap<SymbolPair>, left: isize, right: isize| {
                if left < 0 || right < 0 {
                    return;
                }
                let (l, r) = (&symbols[left as usize], &symbols[right as usize]);
                if l.freeze || r.freeze {
                    return;
                }
                let piece = &normalized[l.start..l.start + l.len + r.len];
                let Some(&id) = self.normal.get(piece) else {
                    return;
                };
                let entry = &self.pieces[id as usize];
                agenda.push(SymbolPair {
                    score: entry.score,
                    left: left as usize,
                    right: right as usize,
                    size: piece.len(),
                });
                if entry.kind == PieceType::Unused {
                    rev_merge.insert(
                        piece,
                        (
                            &normalized[l.start..l.start + l.len],
                            &normalized[r.start..r.start + r.len],
                        ),
                    );
                }
            };
        for right in 1..symbols.len() {
            add_pair(&symbols, &mut agenda, right as isize - 1, right as isize);
        }
        while let Some(top) = agenda.pop() {
            let (left_len, right_len) = (symbols[top.left].len, symbols[top.right].len);
            if left_len == 0 || right_len == 0 || left_len + right_len != top.size {
                continue;
            }
            symbols[top.left].len += right_len;
            let next = symbols[top.right].next;
            symbols[top.left].next = next;
            if next >= 0 {
                symbols[next as usize].prev = top.left as isize;
            }
            symbols[top.right].len = 0;
            let prev = symbols[top.left].prev;
            add_pair(&symbols, &mut agenda, prev, top.left as isize);
            add_pair(&symbols, &mut agenda, top.left as isize, next);
        }

        let mut output = Vec::new();
        let mut index = 0isize;
        while index >= 0 {
            let symbol = &symbols[index as usize];
            let piece = &normalized[symbol.start..symbol.start + symbol.len];
            self.resegment(piece, &rev_merge, &mut output);
            index = symbol.next;
        }
        output
    }

    /// Splits a merged UNUSED piece back along the merge that formed it.
    fn resegment<'a>(
        &self,
        piece: &'a str,
        rev_merge: &HashMap<&'a str, (&'a str, &'a str)>,
        output: &mut Vec<(&'a str, u32)>,
    ) {
        let id = self.piece_to_id(piece);
        if self.pieces[id as usize].kind == PieceType::Unused {
            if let Some(&(left, right)) = rev_merge.get(piece) {
                self.resegment(left, rev_merge, output);
                self.resegment(right, rev_merge, output);
                return;
            }
        }
        output.push((piece, id));
    }
}

/// One BPE symbol: a byte span of the normalized text in a linked list.
#[derive(Debug, Clone)]
struct Symbol {
    start: usize,
    len: usize,
    prev: isize,
    next: isize,
    freeze: bool,
}

/// A candidate merge; the heap pops the highest score, the leftmost first.
#[derive(Debug, Clone)]
struct SymbolPair {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}

impl Ord for SymbolPair {
    fn cmp(&self, other: &Self) -> Ordering {
        // C++ float `<`: -0.0 equals 0.0, NaN compares equal to nothing.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for SymbolPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SymbolPair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SymbolPair {}

/// The byte a `<0xXX>` piece stands for.
fn byte_piece_value(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2)
        .then(|| u8::from_str_radix(hex, 16).ok())
        .flatten()
}

/// Longest user-defined symbol at the start of a string
/// (`normalizer::PrefixMatcher`).
#[derive(Debug, Clone, Default)]
struct PrefixMatcher {
    /// Symbols by first byte, longest first.
    by_first_byte: HashMap<u8, Vec<String>>,
}

impl PrefixMatcher {
    fn new(symbols: Vec<String>) -> Self {
        let mut by_first_byte: HashMap<u8, Vec<String>> = HashMap::new();
        for symbol in symbols {
            if let Some(&first) = symbol.as_bytes().first() {
                by_first_byte.entry(first).or_default().push(symbol);
            }
        }
        for bucket in by_first_byte.values_mut() {
            bucket.sort_by_key(|symbol| std::cmp::Reverse(symbol.len()));
        }
        Self { by_first_byte }
    }

    /// Byte length of the longest symbol `text` starts with.
    fn longest(&self, text: &str) -> Option<usize> {
        self.longest_bytes(text.as_bytes())
    }

    fn longest_bytes(&self, text: &[u8]) -> Option<usize> {
        let bucket = self.by_first_byte.get(text.first()?)?;
        bucket
            .iter()
            .find(|symbol| text.starts_with(symbol.as_bytes()))
            .map(String::len)
    }
}

/// sentencepiece's `Normalizer` for one `NormalizerSpec`.
#[derive(Debug, Clone)]
struct Normalizer {
    /// darts-clone double-array units; empty for the identity normalizer.
    trie: Vec<u32>,
    /// NUL-terminated replacements the trie values point into.
    replacements: Vec<u8>,
    add_dummy_prefix: bool,
    remove_extra: bool,
    escape_whitespaces: bool,
    whitespace_as_suffix: bool,
}

impl Normalizer {
    fn new(spec: &NormalizerSpec, whitespace_as_suffix: bool) -> Result<Self> {
        let (trie, replacements) = if spec.precompiled_charsmap.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let blob = &spec.precompiled_charsmap;
            if blob.len() <= 4 {
                return Err(proto_error("the precompiled charsmap is broken".into()));
            }
            let size = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
            if size > blob.len() - 4 || !size.is_multiple_of(4) {
                return Err(proto_error(
                    "the charsmap trie exceeds the charsmap blob".into(),
                ));
            }
            let trie = blob[4..4 + size]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|unit| u32::from_le_bytes(*unit))
                .collect();
            (trie, blob[4 + size..].to_vec())
        };
        Ok(Self {
            trie,
            replacements,
            add_dummy_prefix: spec.add_dummy_prefix,
            remove_extra: spec.remove_extra_whitespaces,
            escape_whitespaces: spec.escape_whitespaces,
            whitespace_as_suffix,
        })
    }

    /// darts-clone `commonPrefixSearch`: `(value, length)` of every key
    /// that prefixes `key`, shortest first, at most `MAX_TRIE_RESULTS`.
    fn longest_rule(&self, key: &[u8]) -> Option<(usize, usize)> {
        let unit = |pos: usize| self.trie.get(pos).copied();
        let offset = |unit: u32| ((unit >> 10) << ((unit & (1 << 9)) >> 6)) as usize;
        let mut node = offset(unit(0)?);
        let mut longest = None;
        let mut hits = 0;
        for (i, &byte) in key.iter().enumerate() {
            node ^= byte as usize;
            let current = unit(node)?;
            if current & ((1 << 31) | 0xFF) != byte as u32 {
                break;
            }
            node ^= offset(current);
            if (current >> 8) & 1 == 1 {
                if hits < MAX_TRIE_RESULTS {
                    let value = (unit(node)? & ((1 << 31) - 1)) as usize;
                    longest = Some((value, i + 1));
                }
                hits += 1;
            }
        }
        longest
    }

    /// `Normalizer::NormalizePrefix`: the replacement for the start of
    /// `input` and how many bytes it consumes.
    fn normalize_prefix<'a>(
        &'a self,
        input: &'a [u8],
        matcher: &PrefixMatcher,
    ) -> (&'a [u8], usize) {
        if let Some(len) = matcher.longest_bytes(input) {
            return (&input[..len], len);
        }
        if let Some((value, len)) = self.longest_rule(input) {
            let tail = self.replacements.get(value..).unwrap_or(&[]);
            let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
            return (&tail[..end], len);
        }
        // The input is a `str`: one whole character.
        let len = utf8_len(input[0]).min(input.len());
        (&input[..len], len)
    }

    /// `Normalizer::Normalize` without the alignment map.
    fn normalize(&self, text: &str, matcher: &PrefixMatcher) -> String {
        let mut input = text.as_bytes();
        if self.remove_extra {
            while !input.is_empty() {
                let (replacement, consumed) = self.normalize_prefix(input, matcher);
                if replacement != b" " {
                    break;
                }
                input = &input[consumed..];
            }
        }
        if input.is_empty() {
            return String::new();
        }
        let space: &[u8] = if self.escape_whitespaces {
            SPACE_SYMBOL.as_bytes()
        } else {
            b" "
        };
        let mut out: Vec<u8> = Vec::with_capacity(input.len() * 3);
        if !self.whitespace_as_suffix && self.add_dummy_prefix {
            out.extend_from_slice(space);
        }
        let mut is_prev_space = self.remove_extra;
        while !input.is_empty() {
            let (mut replacement, consumed) = self.normalize_prefix(input, matcher);
            if is_prev_space {
                while let Some(rest) = replacement.strip_prefix(b" ") {
                    replacement = rest;
                }
            }
            if !replacement.is_empty() {
                for &byte in replacement {
                    if self.escape_whitespaces && byte == b' ' {
                        out.extend_from_slice(SPACE_SYMBOL.as_bytes());
                    } else {
                        out.push(byte);
                    }
                }
                is_prev_space = replacement.ends_with(b" ");
            }
            input = &input[consumed..];
            if !self.remove_extra {
                is_prev_space = false;
            }
        }
        if self.remove_extra {
            while out.ends_with(space) {
                out.truncate(out.len() - space.len());
            }
        }
        if self.whitespace_as_suffix && self.add_dummy_prefix {
            out.extend_from_slice(space);
        }
        match String::from_utf8(out) {
            Ok(text) => text,
            Err(error) => String::from_utf8_lossy(error.as_bytes()).into_owned(),
        }
    }
}

/// Bytes in the UTF-8 sequence a lead byte starts (sentencepiece's
/// `OneCharLen`).
fn utf8_len(lead: u8) -> usize {
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4][(lead >> 4) as usize]
}

fn proto_error(message: String) -> Error {
    Error::TokenizationError(format!("CTC SentencePiece model: {message}"))
}

const MODEL_TYPE_BPE: u64 = 2;

/// `NormalizerSpec` fields the tokenizer reads, with protobuf defaults.
#[derive(Debug, Clone)]
struct NormalizerSpec {
    precompiled_charsmap: Vec<u8>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

impl Default for NormalizerSpec {
    fn default() -> Self {
        Self {
            precompiled_charsmap: Vec::new(),
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

/// `ModelProto` fields the tokenizer reads, with protobuf defaults.
#[derive(Debug, Clone)]
struct ModelProto {
    pieces: Vec<Piece>,
    model_type: u64,
    treat_whitespace_as_suffix: bool,
    byte_fallback: bool,
    unk_surface: String,
    normalizer: NormalizerSpec,
}

impl ModelProto {
    fn parse(bytes: &[u8]) -> Result<Self> {
        let mut proto = Self {
            pieces: Vec::new(),
            model_type: 1,
            treat_whitespace_as_suffix: false,
            byte_fallback: false,
            unk_surface: " \u{2047} ".to_string(),
            normalizer: NormalizerSpec::default(),
        };
        let mut reader = WireReader::new(bytes);
        while let Some((field, value)) = reader.next_field()? {
            match (field, value) {
                (1, WireValue::Bytes(piece)) => proto.pieces.push(parse_piece(piece)?),
                (2, WireValue::Bytes(trainer)) => proto.parse_trainer(trainer)?,
                (3, WireValue::Bytes(normalizer)) => proto.parse_normalizer(normalizer)?,
                _ => {}
            }
        }
        Ok(proto)
    }

    fn parse_trainer(&mut self, bytes: &[u8]) -> Result<()> {
        let mut reader = WireReader::new(bytes);
        while let Some((field, value)) = reader.next_field()? {
            match (field, value) {
                (3, WireValue::Varint(value)) => self.model_type = value,
                (24, WireValue::Varint(value)) => self.treat_whitespace_as_suffix = value != 0,
                (35, WireValue::Varint(value)) => self.byte_fallback = value != 0,
                (44, WireValue::Bytes(value)) => self.unk_surface = utf8(value, "unk_surface")?,
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_normalizer(&mut self, bytes: &[u8]) -> Result<()> {
        let spec = &mut self.normalizer;
        let mut reader = WireReader::new(bytes);
        while let Some((field, value)) = reader.next_field()? {
            match (field, value) {
                (2, WireValue::Bytes(value)) => spec.precompiled_charsmap = value.to_vec(),
                (3, WireValue::Varint(value)) => spec.add_dummy_prefix = value != 0,
                (4, WireValue::Varint(value)) => spec.remove_extra_whitespaces = value != 0,
                (5, WireValue::Varint(value)) => spec.escape_whitespaces = value != 0,
                _ => {}
            }
        }
        Ok(())
    }
}

fn parse_piece(bytes: &[u8]) -> Result<Piece> {
    let mut piece = Piece {
        text: String::new(),
        score: 0.0,
        kind: PieceType::Normal,
    };
    let mut reader = WireReader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match (field, value) {
            (1, WireValue::Bytes(value)) => piece.text = utf8(value, "piece")?,
            (2, WireValue::Fixed32(value)) => piece.score = f32::from_bits(value),
            (3, WireValue::Varint(value)) => piece.kind = PieceType::from_wire(value)?,
            _ => {}
        }
    }
    Ok(piece)
}

fn utf8(bytes: &[u8], what: &str) -> Result<String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| proto_error(format!("{what} is not UTF-8")))
}

/// One protobuf field value.
#[derive(Debug, Clone, Copy)]
enum WireValue<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

/// A minimal protobuf wire-format reader (no groups).
struct WireReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> WireReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| proto_error("truncated varint".into()))?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(proto_error("varint longer than 10 bytes".into()))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| proto_error("truncated field".into()))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// The next `(field number, value)`, or `None` at the end.
    fn next_field(&mut self) -> Result<Option<(u64, WireValue<'a>)>> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = key >> 3;
        let value = match key & 7 {
            0 => WireValue::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                WireValue::Fixed64
            }
            2 => {
                let len = usize::try_from(self.varint()?)
                    .map_err(|_| proto_error("field length overflows".into()))?;
                WireValue::Bytes(self.take(len)?)
            }
            5 => {
                let raw = self.take(4)?;
                WireValue::Fixed32(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
            }
            wire => {
                return Err(proto_error(format!(
                    "unsupported wire type {wire} (field {field})"
                )));
            }
        };
        Ok(Some((field, value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal protobuf writer for test models.
    #[derive(Default)]
    struct Proto(Vec<u8>);

    impl Proto {
        fn varint(&mut self, mut value: u64) -> &mut Self {
            loop {
                let byte = (value & 0x7f) as u8;
                value >>= 7;
                if value == 0 {
                    self.0.push(byte);
                    return self;
                }
                self.0.push(byte | 0x80);
            }
        }
        fn key(&mut self, field: u64, wire: u64) -> &mut Self {
            self.varint(field << 3 | wire)
        }
        fn int(&mut self, field: u64, value: i64) -> &mut Self {
            self.key(field, 0).varint(value as u64)
        }
        fn bytes(&mut self, field: u64, payload: &[u8]) -> &mut Self {
            self.key(field, 2).varint(payload.len() as u64);
            self.0.extend_from_slice(payload);
            self
        }
        fn float(&mut self, field: u64, value: f32) -> &mut Self {
            self.key(field, 5);
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }
    }

    const NORMAL: i64 = 1;
    const UNKNOWN: i64 = 2;
    const USER_DEFINED: i64 = 4;
    const UNUSED: i64 = 5;
    const BYTE: i64 = 6;

    struct Spec<'a> {
        pieces: &'a [(&'a str, f32, i64)],
        byte_fallback: bool,
        charsmap: Vec<u8>,
        /// add_dummy_prefix, remove_extra_whitespaces, escape_whitespaces.
        flags: Option<(bool, bool, bool)>,
    }

    fn model(spec: Spec<'_>) -> Vec<u8> {
        let mut proto = Proto::default();
        for &(piece, score, kind) in spec.pieces {
            let mut p = Proto::default();
            p.bytes(1, piece.as_bytes()).float(2, score).int(3, kind);
            proto.bytes(1, &p.0);
        }
        let mut trainer = Proto::default();
        trainer.int(3, 2).int(4, spec.pieces.len() as i64);
        // An unknown fixed64 field and an unknown string are skipped.
        trainer.key(99, 1);
        trainer.0.extend_from_slice(&7u64.to_le_bytes());
        trainer.bytes(98, b"ignored");
        if spec.byte_fallback {
            trainer.int(35, 1);
        }
        proto.bytes(2, &trainer.0);
        let mut normalizer = Proto::default();
        normalizer.bytes(1, b"test");
        if !spec.charsmap.is_empty() {
            normalizer.bytes(2, &spec.charsmap);
        }
        if let Some((prefix, extra, escape)) = spec.flags {
            normalizer
                .int(3, prefix as i64)
                .int(4, extra as i64)
                .int(5, escape as i64);
        }
        proto.bytes(3, &normalizer.0);
        proto.0
    }

    const PIECES: &[(&str, f32, i64)] = &[
        ("<unk>", 0.0, UNKNOWN),
        ("\u{2581}", -5.0, NORMAL),
        ("a", -5.0, NORMAL),
        ("b", -5.0, NORMAL),
        ("c", -5.0, NORMAL),
        ("ab", -2.0, NORMAL),
        ("bc", -2.0, NORMAL),
        ("abc", -1.0, UNUSED),
        ("<x>", 0.0, USER_DEFINED),
        ("<x>y", 0.0, USER_DEFINED),
    ];

    fn tiny(flags: Option<(bool, bool, bool)>) -> CtcTokenizer {
        CtcTokenizer::from_model_proto(&model(Spec {
            pieces: PIECES,
            byte_fallback: false,
            charsmap: Vec::new(),
            flags,
        }))
        .unwrap()
    }

    #[test]
    fn bpe_merges_by_score_leftmost_first_and_resegments_unused_pieces() {
        let tokenizer = tiny(None);
        assert_eq!(tokenizer.len(), PIECES.len());
        // "ab" and "bc" tie: the leftmost merges; "abc" is UNUSED, so it is
        // split back along the merge that formed it.
        assert_eq!(tokenizer.encode("abc"), vec![1, 5, 4]);
        assert_eq!(tokenizer.encode("bcab"), vec![1, 6, 5]);
        // User-defined symbols are matched longest-first and never merge.
        assert_eq!(tokenizer.encode("a<x>b"), vec![1, 2, 8, 3]);
        assert_eq!(tokenizer.encode("a<x>yb"), vec![1, 2, 9, 3]);
        // A run of unknown characters is one <unk>.
        assert_eq!(tokenizer.encode("zzq a"), vec![1, 0, 1, 2]);
        assert_eq!(tokenizer.encode(""), Vec::<u32>::new());
        assert_eq!(tokenizer.piece_to_id("ab"), 5);
        assert_eq!(tokenizer.piece_to_id("nope"), 0);
        assert_eq!(tokenizer.id_to_piece(8), Some("<x>"));
        assert_eq!(tokenizer.id_to_piece(99), None);
        assert_eq!(tokenizer.decode(&[1, 5, 4]), "abc");
        assert_eq!(tokenizer.decode(&[2, 1, 0, 3]), "a  \u{2047} b");
    }

    #[test]
    fn whitespace_follows_the_normalizer_spec() {
        // Defaults: dummy prefix, extra whitespace removed, `▁` escaping.
        assert_eq!(tiny(None).normalize("  a   b  "), "\u{2581}a\u{2581}b");
        assert_eq!(tiny(None).normalize("   "), "");
        // remove_extra_whitespaces false keeps every space.
        let keep = tiny(Some((true, false, true)));
        assert_eq!(
            keep.normalize(" a  b "),
            "\u{2581}\u{2581}a\u{2581}\u{2581}b\u{2581}"
        );
        assert_eq!(keep.encode(" "), vec![1, 1]);
        // No dummy prefix; no escaping.
        assert_eq!(
            tiny(Some((false, true, true))).normalize("a b"),
            "a\u{2581}b"
        );
        assert_eq!(tiny(Some((true, true, false))).normalize("a  b"), " a b");
    }

    #[test]
    fn byte_fallback_spells_unknown_characters_in_bytes() {
        let pieces = [
            ("<unk>", 0.0, UNKNOWN),
            ("\u{2581}", -1.0, NORMAL),
            ("<0x7A>", 0.0, BYTE),
            ("<0xC3>", 0.0, BYTE),
            ("<0xA9>", 0.0, BYTE),
        ];
        let tokenizer = CtcTokenizer::from_model_proto(&model(Spec {
            pieces: &pieces,
            byte_fallback: true,
            charsmap: Vec::new(),
            flags: None,
        }))
        .unwrap();
        assert_eq!(tokenizer.encode("zé"), vec![1, 2, 3, 4]);
        assert_eq!(tokenizer.decode(&[1, 2, 3, 4]), "zé");
    }

    /// A hand-built darts-clone trie: "a" -> "x", "ab" -> "yz".
    fn charsmap() -> Vec<u8> {
        let leaf = |value: u32| (1u32 << 31) | value;
        let mut units = vec![0u32; 98];
        units[0] = 1 << 10; // root, offset 1
        units[0x60] = 0x61 | (1 << 8) | (1 << 10); // 'a', has leaf, offset 1
        units[0x61] = leaf(0); // "a" -> blob[0..] = "x"
        units[0x03] = 0x62 | (1 << 8) | (1 << 10); // 'b' under 'a'
        units[0x02] = leaf(2); // "ab" -> blob[2..] = "yz"
        let mut blob = ((units.len() * 4) as u32).to_le_bytes().to_vec();
        for unit in units {
            blob.extend_from_slice(&unit.to_le_bytes());
        }
        blob.extend_from_slice(b"x\0yz\0");
        blob
    }

    #[test]
    fn the_charsmap_trie_applies_the_longest_rule() {
        let tokenizer = CtcTokenizer::from_model_proto(&model(Spec {
            pieces: PIECES,
            byte_fallback: false,
            charsmap: charsmap(),
            flags: None,
        }))
        .unwrap();
        assert_eq!(tokenizer.normalize("aab c"), "\u{2581}xyz\u{2581}c");
        // User-defined symbols are not normalized.
        assert_eq!(tokenizer.normalize("<x>a"), "\u{2581}<x>x");
        // A charsmap whose trie overruns the blob (also by less than its
        // 4-byte header) is rejected, not read past its end.
        let blob_len = charsmap().len() as u32;
        for size in [10_000, blob_len - 1, blob_len - 2] {
            let mut broken = charsmap();
            broken[0..4].copy_from_slice(&size.to_le_bytes());
            assert!(CtcTokenizer::from_model_proto(&model(Spec {
                pieces: PIECES,
                byte_fallback: false,
                charsmap: broken,
                flags: None,
            }))
            .is_err());
        }
    }

    #[test]
    fn malformed_or_unsupported_models_are_rejected() {
        let good = model(Spec {
            pieces: PIECES,
            byte_fallback: false,
            charsmap: Vec::new(),
            flags: None,
        });
        assert!(CtcTokenizer::from_model_proto(&good[..good.len() - 3]).is_err());
        // No <unk>.
        assert!(CtcTokenizer::from_model_proto(&model(Spec {
            pieces: &[("a", 0.0, NORMAL)],
            byte_fallback: false,
            charsmap: Vec::new(),
            flags: None,
        }))
        .is_err());
        // A duplicate piece.
        assert!(CtcTokenizer::from_model_proto(&model(Spec {
            pieces: &[
                ("<unk>", 0.0, UNKNOWN),
                ("a", 0.0, NORMAL),
                ("a", 0.0, NORMAL)
            ],
            byte_fallback: false,
            charsmap: Vec::new(),
            flags: None,
        }))
        .is_err());
        // A unigram model.
        let mut unigram = Proto::default();
        let mut piece = Proto::default();
        piece.bytes(1, b"<unk>").int(3, UNKNOWN);
        unigram.bytes(1, &piece.0);
        let mut trainer = Proto::default();
        trainer.int(3, 1);
        unigram.bytes(2, &trainer.0);
        assert!(CtcTokenizer::from_model_proto(&unigram.0).is_err());
    }

    #[test]
    fn the_synthetic_ctc_model_encodes_as_sentencepiece() {
        // Ids sentencepiece 0.2.1 gives for the family's synthetic model.
        let proto = crate::models::architectures::nemotron_transcribe::synthetic::ctc_model_proto();
        let tokenizer = CtcTokenizer::from_model_proto(&proto).unwrap();
        for (text, ids) in [
            ("w153 w200", vec![23, 14, 18, 16, 23, 15, 13, 13]),
            ("<spk:1> w177 w14", vec![11, 2, 23, 14, 20, 20, 23, 14, 17]),
            ("w9  w10", vec![23, 22, 23, 14, 13]),
            (
                " w142 <spk:0>w3 x",
                vec![23, 14, 17, 15, 11, 1, 12, 16, 11, 0],
            ),
            ("<ev:e3> w150", vec![11, 0, 16, 0, 23, 14, 18, 13]),
        ] {
            assert_eq!(tokenizer.encode(text), ids, "{text:?}");
        }
    }
}
