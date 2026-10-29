//! The 7-file checkpoint: index, memory maps, census and FP8 ingestion.
//!
//! One `model.safetensors.index.json` maps every tensor of the backbone
//! shards, `perception.safetensors`, the CTC timestamp head and the MTP head
//! to its file. The census derives the exact tensor set (name, dtype, shape)
//! from the config and fails the load on any missing, extra or reshaped
//! tensor, before a byte is uploaded. The CTC and MTP files are optional:
//! without the CTC file word timestamps are unavailable; the MTP head is
//! never loaded (speculative decoding is deferred), only checked by prefix.
//!
//! ModelOpt FP8 linears carry `weight` (E4M3), a scalar `weight_scale` and a
//! scalar static `input_scale` (and `k_scale` / `v_scale` on attention).
//! [`Fp8Weight`] keeps the bytes and both scales; the kernels' block-scaled
//! layout takes the scalar broadcast to every 128×128 block, which is exact.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use half::bf16;
use memmap2::Mmap;
use safetensors::tensor::{Dtype, TensorInfo};
use safetensors::SafeTensors;
use serde::Deserialize;

use super::config::{NemotronLayerKind, NemotronTranscribeConfig};
use crate::error::{Error, Result};

pub const INDEX_FILE: &str = "model.safetensors.index.json";
pub const PERCEPTION_FILE: &str = "perception.safetensors";
pub const CTC_FILE: &str = "model-ctc-bf16.safetensors";
pub const MTP_FILE: &str = "model-mtp-fp8.safetensors";
/// Every weight file of the published checkpoint, in index order.
pub const WEIGHT_FILES: [&str; 7] = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    PERCEPTION_FILE,
    CTC_FILE,
    MTP_FILE,
];
pub const MTP_PREFIX: &str = "llm.mtp.layers.";
pub const CTC_PREFIX: &str = "ctc_timestamp.";

/// The dtype a census entry expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CensusDtype {
    Bf16,
    F32,
    F8E4M3,
    U8,
    I64,
}

impl CensusDtype {
    fn matches(self, dtype: Dtype) -> bool {
        matches!(
            (self, dtype),
            (Self::Bf16, Dtype::BF16)
                | (Self::F32, Dtype::F32)
                | (Self::F8E4M3, Dtype::F8_E4M3)
                | (Self::U8, Dtype::U8)
                | (Self::I64, Dtype::I64)
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F8E4M3 => "F8_E4M3",
            Self::U8 => "U8",
            Self::I64 => "I64",
        }
    }
}

/// One tensor the census expects.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CensusEntry {
    pub name: String,
    pub dtype: CensusDtype,
    pub shape: Vec<usize>,
}

/// The CTC head's geometry, from the CTC artifact's metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcGeometry {
    pub feat_in: usize,
    /// Labels excluding the blank (the blank is the last class).
    pub num_classes: usize,
    pub heads: usize,
    pub layers: usize,
    pub ff_dim: usize,
    pub rotary_dims: usize,
    pub upsample: usize,
    pub residual_scale: f32,
    pub vocabulary: Vec<String>,
    /// SentencePiece's tokenizer settings, as published.
    pub tokenizer_config: serde_json::Value,
    pub model_proto_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct RawIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawCtcDecoder {
    feat_in: usize,
    num_classes: usize,
    vocabulary: Vec<String>,
    use_transformer: bool,
    n_heads: usize,
    n_layers: usize,
    #[serde(default)]
    qkv_bias: bool,
    qk_norm: bool,
    ff_expansion: f64,
    pre_block_norm: bool,
    self_attention_model: String,
    rotary_fraction: f64,
    #[serde(default)]
    xscaling: bool,
    residual: bool,
    residual_scale: f64,
    #[serde(default)]
    learnable_residual_scale: bool,
    upsampling_factor: usize,
    upsampling_mode: String,
}

struct MappedFile {
    mmap: Mmap,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
    metadata: Option<HashMap<String, String>>,
}

/// An FP8 linear's bytes and scales, borrowed from the mapping.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Weight<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub weight_scale: f32,
    pub input_scale: f32,
}

impl Fp8Weight<'_> {
    /// The scalar `weight_scale` broadcast over the kernels' 128×128 block
    /// grid, `[ceil(rows/128), ceil(cols/128)]`.
    pub fn block_scales(&self) -> Vec<f32> {
        vec![self.weight_scale; self.rows.div_ceil(128) * self.cols.div_ceil(128)]
    }

    /// Row `row` dequantized exactly (`decode(byte) * weight_scale`, F32).
    pub fn dequantized_row(&self, row: usize) -> Vec<f32> {
        self.bytes[row * self.cols..(row + 1) * self.cols]
            .iter()
            .map(|&b| crate::kernels::cuda::fp8::decode_e4m3fn(b) * self.weight_scale)
            .collect()
    }
}

/// The memory-mapped checkpoint.
pub struct NemotronCheckpoint {
    model_dir: PathBuf,
    files: Vec<(String, MappedFile)>,
    /// Tensor name → index into `files`.
    index: BTreeMap<String, usize>,
}

impl std::fmt::Debug for NemotronCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NemotronCheckpoint")
            .field("model_dir", &self.model_dir)
            .field(
                "files",
                &self.files.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            )
            .field("tensors", &self.index.len())
            .finish()
    }
}

fn load_error(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!(
        "Nemotron 3.5 Transcribe checkpoint: {}",
        message.into()
    ))
}

impl NemotronCheckpoint {
    /// Maps every file the index names that exists; the backbone shards and
    /// `perception.safetensors` must all exist.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join(INDEX_FILE);
        let raw = std::fs::read(&index_path)
            .map_err(|err| load_error(format!("cannot read {}: {err}", index_path.display())))?;
        let index: RawIndex = serde_json::from_slice(&raw)
            .map_err(|err| load_error(format!("invalid {INDEX_FILE}: {err}")))?;
        let names = index.weight_map.values().cloned().collect::<BTreeSet<_>>();
        for name in &names {
            if !WEIGHT_FILES.contains(&name.as_str()) {
                return Err(load_error(format!(
                    "the index names an unexpected file {name:?}"
                )));
            }
        }
        let mut files = Vec::new();
        for name in WEIGHT_FILES {
            if !names.contains(name) {
                if name == CTC_FILE || name == MTP_FILE {
                    continue;
                }
                return Err(load_error(format!("the index does not name {name}")));
            }
            let path = model_dir.join(name);
            if !path.is_file() {
                if name == CTC_FILE || name == MTP_FILE {
                    continue;
                }
                return Err(load_error(format!(
                    "missing weight file {}",
                    path.display()
                )));
            }
            let file = File::open(&path)
                .map_err(|err| load_error(format!("cannot open {}: {err}", path.display())))?;
            // SAFETY: the mapping is read-only and the file is not modified
            // through this process while the checkpoint lives.
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|err| load_error(format!("cannot map {}: {err}", path.display())))?;
            let (header, metadata) = SafeTensors::read_metadata(&mmap)
                .map_err(|err| load_error(format!("invalid header in {name}: {err}")))?;
            let tensors = metadata
                .tensors()
                .into_iter()
                .map(|(key, info)| (key, info.clone()))
                .collect::<HashMap<_, _>>();
            files.push((
                name.to_string(),
                MappedFile {
                    data_start: 8 + header,
                    metadata: metadata.metadata().clone(),
                    tensors,
                    mmap,
                },
            ));
        }
        let mut tensor_index = BTreeMap::new();
        for (tensor, file) in &index.weight_map {
            let Some(position) = files.iter().position(|(name, _)| name == file) else {
                continue;
            };
            if !files[position].1.tensors.contains_key(tensor) {
                return Err(load_error(format!(
                    "the index maps {tensor} to {file}, which does not hold it"
                )));
            }
            tensor_index.insert(tensor.clone(), position);
        }
        for (position, (name, file)) in files.iter().enumerate() {
            for tensor in file.tensors.keys() {
                if tensor_index.get(tensor) != Some(&position) {
                    return Err(load_error(format!(
                        "{name} holds {tensor}, which the index does not map to it"
                    )));
                }
            }
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            files,
            index: tensor_index,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn has_file(&self, name: &str) -> bool {
        self.files.iter().any(|(file, _)| file == name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    fn file_of(&self, name: &str) -> Result<&MappedFile> {
        let position = self
            .index
            .get(name)
            .ok_or_else(|| load_error(format!("missing tensor {name}")))?;
        Ok(&self.files[*position].1)
    }

    /// The tensor's dtype and shape.
    pub fn info(&self, name: &str) -> Result<(Dtype, Vec<usize>)> {
        let file = self.file_of(name)?;
        let info = &file.tensors[name];
        Ok((info.dtype, info.shape.clone()))
    }

    /// The tensor's raw bytes.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let file = self.file_of(name)?;
        let info = &file.tensors[name];
        let (start, end) = info.data_offsets;
        file.mmap
            .get(file.data_start + start..file.data_start + end)
            .ok_or_else(|| load_error(format!("{name} lies outside its file")))
    }

    /// A file's `__metadata__`.
    pub fn file_metadata(&self, file: &str) -> Option<&HashMap<String, String>> {
        self.files
            .iter()
            .find(|(name, _)| name == file)
            .and_then(|(_, mapped)| mapped.metadata.as_ref())
    }

    /// A BF16 or F32 tensor's values as F32, checked against `shape`.
    pub fn f32_values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (dtype, found) = self.info(name)?;
        if found != shape {
            return Err(load_error(format!(
                "{name} has shape {found:?}; expected {shape:?}"
            )));
        }
        let bytes = self.bytes(name)?;
        match dtype {
            Dtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()),
            Dtype::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            other => Err(load_error(format!("{name} is {other:?}, not BF16 or F32"))),
        }
    }

    /// A BF16 tensor's raw values.
    pub fn bf16_values(&self, name: &str, shape: &[usize]) -> Result<Vec<bf16>> {
        let (dtype, found) = self.info(name)?;
        if dtype != Dtype::BF16 || found != shape {
            return Err(load_error(format!(
                "{name} is {dtype:?} {found:?}; expected BF16 {shape:?}"
            )));
        }
        Ok(self
            .bytes(name)?
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    /// A scalar F32 tensor (`[]`), required finite and positive.
    pub fn positive_scalar(&self, name: &str) -> Result<f32> {
        let value = self.f32_values(name, &[])?[0];
        if !value.is_finite() || value <= 0.0 {
            return Err(load_error(format!(
                "{name} = {value} is not a positive scale"
            )));
        }
        Ok(value)
    }

    /// The FP8 linear `module` (`module.weight`, `.weight_scale`,
    /// `.input_scale`), checked against `[rows, cols]`.
    pub fn fp8(&self, module: &str, rows: usize, cols: usize) -> Result<Fp8Weight<'_>> {
        let weight = format!("{module}.weight");
        let (dtype, shape) = self.info(&weight)?;
        if dtype != Dtype::F8_E4M3 || shape != [rows, cols] {
            return Err(load_error(format!(
                "{weight} is {dtype:?} {shape:?}; expected F8_E4M3 [{rows}, {cols}]"
            )));
        }
        let bytes = self.bytes(&weight)?;
        // A NaN byte (0x7f / 0xff) would poison every product it meets.
        if bytes.iter().any(|&b| b & 0x7f == 0x7f) {
            return Err(load_error(format!("{weight} contains an E4M3 NaN")));
        }
        Ok(Fp8Weight {
            bytes,
            rows,
            cols,
            weight_scale: self.positive_scalar(&format!("{module}.weight_scale"))?,
            input_scale: self.positive_scalar(&format!("{module}.input_scale"))?,
        })
    }

    /// The CTC head's geometry, or `None` without the CTC file.
    pub fn ctc_geometry(&self) -> Result<Option<CtcGeometry>> {
        let Some(metadata) = self.file_metadata(CTC_FILE) else {
            return Ok(None);
        };
        if metadata.get("ctc_timestamp_format").map(String::as_str)
            != Some("nemo_ctc_timestamp_artifact_v1")
        {
            return Err(load_error("the CTC artifact has an unknown format"));
        }
        let raw: RawCtcDecoder = serde_json::from_str(
            metadata
                .get("ctc_timestamp_decoder_config")
                .ok_or_else(|| load_error("the CTC artifact has no decoder config"))?,
        )
        .map_err(|err| load_error(format!("invalid CTC decoder config: {err}")))?;
        let tokenizer_config = metadata
            .get("ctc_timestamp_tokenizer_config")
            .map(|text| serde_json::from_str(text))
            .transpose()
            .map_err(|err| load_error(format!("invalid CTC tokenizer config: {err}")))?
            .unwrap_or(serde_json::Value::Null);
        if !raw.use_transformer
            || raw.qkv_bias
            || !raw.qk_norm
            || !raw.pre_block_norm
            || raw.self_attention_model != "rope"
            || raw.xscaling
            || !raw.residual
            || raw.learnable_residual_scale
            || raw.upsampling_mode != "conv_transpose"
            || raw.n_heads == 0
            || !raw.feat_in.is_multiple_of(raw.n_heads)
            || raw.vocabulary.len() != raw.num_classes
        {
            return Err(load_error(
                "the CTC head uses settings this family does not implement",
            ));
        }
        let head_dim = raw.feat_in / raw.n_heads;
        let rotary_dims = (head_dim as f64 * raw.rotary_fraction).round() as usize;
        let ff_dim = (raw.feat_in as f64 * raw.ff_expansion).round() as usize;
        let proto = format!("{CTC_PREFIX}tokenizer.model_proto");
        let (_, proto_shape) = self.info(&proto)?;
        Ok(Some(CtcGeometry {
            feat_in: raw.feat_in,
            num_classes: raw.num_classes,
            heads: raw.n_heads,
            layers: raw.n_layers,
            ff_dim,
            rotary_dims,
            upsample: raw.upsampling_factor,
            // The head runs in BF16 upstream, so its 0.1 is bf16(0.1).
            residual_scale: bf16::from_f64(raw.residual_scale).to_f32(),
            vocabulary: raw.vocabulary,
            tokenizer_config,
            model_proto_bytes: proto_shape.first().copied().unwrap_or(0),
        }))
    }

    /// Checks the tensor set against the census: every expected tensor at
    /// its dtype and shape, and nothing else (MTP tensors by prefix only).
    pub fn validate_census(&self, config: &NemotronTranscribeConfig) -> Result<()> {
        let ctc = self.ctc_geometry()?;
        let expected = expected_census(config, ctc.as_ref());
        let mut seen = BTreeSet::new();
        for entry in &expected {
            let (dtype, shape) = self.info(&entry.name)?;
            if !entry.dtype.matches(dtype) || shape != entry.shape {
                return Err(load_error(format!(
                    "{} is {dtype:?} {shape:?}; expected {} {:?}",
                    entry.name,
                    entry.dtype.name(),
                    entry.shape
                )));
            }
            seen.insert(entry.name.as_str());
        }
        for name in self.index.keys() {
            if !seen.contains(name.as_str()) && !name.starts_with(MTP_PREFIX) {
                return Err(load_error(format!("unexpected tensor {name}")));
            }
        }
        Ok(())
    }
}

/// The FP8 linear `module` of `[rows, cols]`: three census entries, plus the
/// attention KV scale when `kv_scale` names one.
fn fp8_entries(
    out: &mut Vec<CensusEntry>,
    module: &str,
    rows: usize,
    cols: usize,
    kv_scale: Option<&str>,
) {
    out.push(CensusEntry {
        name: format!("{module}.weight"),
        dtype: CensusDtype::F8E4M3,
        shape: vec![rows, cols],
    });
    for scale in ["weight_scale", "input_scale"].into_iter().chain(kv_scale) {
        out.push(CensusEntry {
            name: format!("{module}.{scale}"),
            dtype: CensusDtype::F32,
            shape: vec![],
        });
    }
}

fn entry(out: &mut Vec<CensusEntry>, name: impl Into<String>, dtype: CensusDtype, shape: &[usize]) {
    out.push(CensusEntry {
        name: name.into(),
        dtype,
        shape: shape.to_vec(),
    });
}

/// A linear that is FP8 unless the exclude list names it: the census and
/// the loader agree on which through this one predicate.
pub fn is_fp8_linear(config: &NemotronTranscribeConfig, module: &str) -> bool {
    !config.fp8_excludes(module)
}

fn linear_entries(
    out: &mut Vec<CensusEntry>,
    config: &NemotronTranscribeConfig,
    module: &str,
    rows: usize,
    cols: usize,
    bias: bool,
) {
    if is_fp8_linear(config, module) {
        fp8_entries(out, module, rows, cols, None);
    } else {
        entry(
            out,
            format!("{module}.weight"),
            CensusDtype::Bf16,
            &[rows, cols],
        );
    }
    if bias {
        entry(out, format!("{module}.bias"), CensusDtype::Bf16, &[rows]);
    }
}

fn transformer_entries(
    out: &mut Vec<CensusEntry>,
    config: &NemotronTranscribeConfig,
    prefix: &str,
    encoder: &super::config::SpeechTransformerConfig,
    pre_encode: bool,
    dtype: CensusDtype,
) {
    let d = encoder.d_model;
    let norm = |out: &mut Vec<CensusEntry>, name: String, width: usize| {
        entry(out, format!("{name}.weight"), dtype, &[width]);
        entry(out, format!("{name}.bias"), dtype, &[width]);
    };
    if pre_encode {
        linear_entries(
            out,
            config,
            &format!("{prefix}.pre_encode.proj"),
            d,
            encoder.feat_in * encoder.stacking,
            false,
        );
    }
    norm(out, format!("{prefix}.embed_norm"), d);
    norm(out, format!("{prefix}.final_norm"), d);
    for layer in 0..encoder.layers {
        let l = format!("{prefix}.layers.{layer}");
        norm(out, format!("{l}.norm1"), d);
        norm(out, format!("{l}.norm2"), d);
        if dtype == CensusDtype::F32 {
            // The CTC head stores every tensor F32 and has no FP8 linears.
            entry(out, format!("{l}.attn.w_qkv.weight"), dtype, &[3 * d, d]);
            entry(out, format!("{l}.attn.out_proj.weight"), dtype, &[d, d]);
            entry(out, format!("{l}.attn.out_proj.bias"), dtype, &[d]);
            entry(
                out,
                format!("{l}.ffn.net.0.weight"),
                dtype,
                &[encoder.ff_dim, d],
            );
            entry(out, format!("{l}.ffn.net.0.bias"), dtype, &[encoder.ff_dim]);
            entry(
                out,
                format!("{l}.ffn.net.3.weight"),
                dtype,
                &[d, encoder.ff_dim],
            );
            entry(out, format!("{l}.ffn.net.3.bias"), dtype, &[d]);
        } else {
            linear_entries(out, config, &format!("{l}.attn.w_qkv"), 3 * d, d, false);
            linear_entries(out, config, &format!("{l}.attn.out_proj"), d, d, true);
            linear_entries(
                out,
                config,
                &format!("{l}.ffn.net.0"),
                encoder.ff_dim,
                d,
                true,
            );
            linear_entries(
                out,
                config,
                &format!("{l}.ffn.net.3"),
                d,
                encoder.ff_dim,
                true,
            );
        }
        if encoder.qk_norm {
            norm(out, format!("{l}.attn.q_norm"), encoder.head_dim());
            norm(out, format!("{l}.attn.k_norm"), encoder.head_dim());
        }
    }
}

/// Every tensor the family expects, except the MTP head's.
pub fn expected_census(
    config: &NemotronTranscribeConfig,
    ctc: Option<&CtcGeometry>,
) -> Vec<CensusEntry> {
    use CensusDtype::{Bf16, F32};
    let mut out = Vec::new();
    let fe = &config.frontend;
    for prefix in [
        "perception.preprocessor.featurizer",
        "perception.encoder.diarization_model.preprocessor.featurizer",
    ] {
        entry(
            &mut out,
            format!("{prefix}.fb"),
            Bf16,
            &[1, fe.n_mels, fe.n_fft / 2 + 1],
        );
        entry(&mut out, format!("{prefix}.window"), Bf16, &[fe.win_length]);
    }
    let asr = &config.asr_encoder;
    transformer_entries(
        &mut out,
        config,
        "perception.encoder.asr_encoder",
        asr,
        true,
        Bf16,
    );
    let diar = &config.diar_encoder;
    transformer_entries(
        &mut out,
        config,
        "perception.encoder.diarization_model.encoder",
        diar,
        true,
        Bf16,
    );
    let sf = &config.sortformer;
    let m = "perception.encoder.diarization_model.sortformer_modules";
    let pair = |out: &mut Vec<CensusEntry>, name: &str, w: &[usize], b: usize| {
        entry(out, format!("{m}.{name}.weight"), Bf16, w);
        entry(out, format!("{m}.{name}.bias"), Bf16, &[b]);
    };
    pair(
        &mut out,
        "encoder_proj",
        &[sf.tf_d_model, sf.fc_d_model],
        sf.tf_d_model,
    );
    pair(
        &mut out,
        "first_hidden_to_hidden",
        &[sf.tf_d_model, sf.tf_d_model],
        sf.tf_d_model,
    );
    pair(
        &mut out,
        "hidden_to_spks",
        &[sf.speakers, 2 * sf.tf_d_model],
        sf.speakers,
    );
    pair(
        &mut out,
        "single_hidden_to_spks",
        &[sf.speakers, sf.tf_d_model],
        sf.speakers,
    );
    pair(
        &mut out,
        "subpixel_upsample",
        &[sf.tf_d_model * sf.upsample, sf.tf_d_model, 3],
        sf.tf_d_model * sf.upsample,
    );
    entry(
        &mut out,
        format!("{m}.learnable_sil_emb"),
        Bf16,
        &[sf.fc_d_model],
    );
    for (name, width) in [("asr_norm", asr.d_model), ("diar_norm", sf.speakers)] {
        entry(
            &mut out,
            format!("perception.encoder.{name}.weight"),
            Bf16,
            &[width],
        );
        entry(
            &mut out,
            format!("perception.encoder.{name}.bias"),
            Bf16,
            &[width],
        );
    }
    let d = &config.decoder;
    entry(
        &mut out,
        "perception.proj.weight",
        Bf16,
        &[d.hidden, asr.d_model],
    );
    entry(&mut out, "perception.proj.bias", Bf16, &[d.hidden]);

    entry(
        &mut out,
        "backbone.embeddings.weight",
        Bf16,
        &[d.vocab, d.hidden],
    );
    entry(&mut out, "backbone.norm_f.weight", Bf16, &[d.hidden]);
    entry(&mut out, "lm_head.weight", Bf16, &[d.vocab, d.hidden]);
    let mb = &d.mamba;
    let moe = &d.moe;
    let at = &d.attention;
    for (index, kind) in d.layers.iter().enumerate() {
        let l = format!("backbone.layers.{index}");
        entry(&mut out, format!("{l}.norm.weight"), Bf16, &[d.hidden]);
        let x = format!("{l}.mixer");
        match kind {
            NemotronLayerKind::Mamba => {
                for name in ["A_log", "D", "dt_bias"] {
                    entry(&mut out, format!("{x}.{name}"), Bf16, &[mb.heads]);
                }
                entry(
                    &mut out,
                    format!("{x}.conv1d.weight"),
                    Bf16,
                    &[mb.conv_channels(), 1, mb.conv_kernel],
                );
                entry(
                    &mut out,
                    format!("{x}.conv1d.bias"),
                    Bf16,
                    &[mb.conv_channels()],
                );
                entry(&mut out, format!("{x}.norm.weight"), Bf16, &[mb.inner()]);
                linear_entries(
                    &mut out,
                    config,
                    &format!("{x}.in_proj"),
                    mb.in_proj_rows(),
                    d.hidden,
                    false,
                );
                linear_entries(
                    &mut out,
                    config,
                    &format!("{x}.out_proj"),
                    d.hidden,
                    mb.inner(),
                    false,
                );
            }
            NemotronLayerKind::Moe => {
                entry(
                    &mut out,
                    format!("{x}.gate.weight"),
                    Bf16,
                    &[moe.experts, d.hidden],
                );
                entry(
                    &mut out,
                    format!("{x}.gate.e_score_correction_bias"),
                    F32,
                    &[moe.experts],
                );
                for expert in 0..moe.experts {
                    let e = format!("{x}.experts.{expert}");
                    linear_entries(
                        &mut out,
                        config,
                        &format!("{e}.up_proj"),
                        moe.intermediate,
                        d.hidden,
                        false,
                    );
                    linear_entries(
                        &mut out,
                        config,
                        &format!("{e}.down_proj"),
                        d.hidden,
                        moe.intermediate,
                        false,
                    );
                }
                let s = format!("{x}.shared_experts");
                linear_entries(
                    &mut out,
                    config,
                    &format!("{s}.up_proj"),
                    moe.shared_intermediate,
                    d.hidden,
                    false,
                );
                linear_entries(
                    &mut out,
                    config,
                    &format!("{s}.down_proj"),
                    d.hidden,
                    moe.shared_intermediate,
                    false,
                );
            }
            NemotronLayerKind::Attention => {
                fp8_entries(
                    &mut out,
                    &format!("{x}.q_proj"),
                    at.heads * at.head_dim,
                    d.hidden,
                    None,
                );
                fp8_entries(
                    &mut out,
                    &format!("{x}.k_proj"),
                    at.kv_heads * at.head_dim,
                    d.hidden,
                    Some("k_scale"),
                );
                fp8_entries(
                    &mut out,
                    &format!("{x}.v_proj"),
                    at.kv_heads * at.head_dim,
                    d.hidden,
                    Some("v_scale"),
                );
                fp8_entries(
                    &mut out,
                    &format!("{x}.o_proj"),
                    d.hidden,
                    at.heads * at.head_dim,
                    None,
                );
            }
        }
    }

    if let Some(ctc) = ctc {
        let c = "ctc_timestamp.decoder";
        let width = ctc.feat_in;
        entry(
            &mut out,
            format!("{c}.upsample.weight"),
            F32,
            &[width, width, ctc.upsample],
        );
        entry(&mut out, format!("{c}.upsample.bias"), F32, &[width]);
        entry(
            &mut out,
            format!("{c}.upsample_factor"),
            CensusDtype::I64,
            &[],
        );
        entry(&mut out, format!("{c}.residual_scale"), F32, &[]);
        entry(
            &mut out,
            format!("{c}.decoder_layers.0.weight"),
            F32,
            &[ctc.num_classes + 1, width, 1],
        );
        entry(
            &mut out,
            format!("{c}.decoder_layers.0.bias"),
            F32,
            &[ctc.num_classes + 1],
        );
        let trunk = super::config::SpeechTransformerConfig {
            feat_in: width,
            d_model: width,
            heads: ctc.heads,
            layers: ctc.layers,
            ff_dim: ctc.ff_dim,
            qk_norm: true,
            rotary_dims: ctc.rotary_dims,
            rope_base: 10_000.0,
            stacking: 1,
            layer_norm_eps: 1e-5,
        };
        transformer_entries(
            &mut out,
            config,
            &format!("{c}.transformer"),
            &trunk,
            false,
            F32,
        );
        entry(
            &mut out,
            "ctc_timestamp.tokenizer.model_proto",
            CensusDtype::U8,
            &[ctc.model_proto_bytes],
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The census derived from the published config, collapsed to name
    /// patterns, equals the census read from the published headers (MTP
    /// tensors aside).
    #[test]
    fn the_derived_census_equals_the_published_headers() {
        let config = super::super::config::tests::published();
        let ctc = CtcGeometry {
            feat_in: 1280,
            num_classes: 23_862,
            heads: 16,
            layers: 2,
            ff_dim: 640,
            rotary_dims: 40,
            upsample: 2,
            residual_scale: bf16::from_f64(0.1).to_f32(),
            vocabulary: Vec::new(),
            tokenizer_config: serde_json::Value::Null,
            model_proto_bytes: 632_302,
        };
        let digits = regex::Regex::new(r"\.\d+(\.|$)").unwrap();
        let collapse = |name: &str| {
            let mut out = name.to_string();
            while digits.is_match(&out) {
                out = digits.replace_all(&out, ".N$1").into_owned();
            }
            out
        };
        let mut derived = BTreeMap::<String, usize>::new();
        for e in expected_census(&config, Some(&ctc)) {
            let key = format!(
                "{} {} [{}]",
                collapse(&e.name),
                e.dtype.name(),
                e.shape
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            *derived.entry(key).or_default() += 1;
        }
        let published =
            include_str!("../../../../tests/fixtures/nemotron_transcribe/tensor_census.txt");
        let mut expected = BTreeMap::<String, usize>::new();
        for line in published.lines().filter(|l| !l.trim().is_empty()) {
            let (count, rest) = line.split_once(' ').unwrap();
            if rest.starts_with(MTP_PREFIX) {
                continue;
            }
            *expected.entry(rest.to_string()).or_default() += count.parse::<usize>().unwrap();
        }
        assert_eq!(derived, expected);
        // 20,208 tensors in all, of which the MTP head has 796.
        let total = expected_census(&config, Some(&ctc)).len();
        assert_eq!(total, 20_208 - 796);
    }

    #[test]
    fn block_scales_broadcast_the_scalar_and_rows_dequantize_exactly() {
        let bytes = (0..=255u8)
            .filter(|b| b & 0x7f != 0x7f)
            .take(4 * 160)
            .collect::<Vec<_>>();
        let bytes = bytes
            .iter()
            .cycle()
            .take(4 * 160)
            .copied()
            .collect::<Vec<_>>();
        let w = Fp8Weight {
            bytes: &bytes,
            rows: 4,
            cols: 160,
            weight_scale: 0.0011,
            input_scale: 0.03,
        };
        assert_eq!(w.block_scales(), vec![0.0011; 2]);
        let row = w.dequantized_row(1);
        for (i, v) in row.iter().enumerate() {
            assert_eq!(
                *v,
                crate::kernels::cuda::fp8::decode_e4m3fn(bytes[160 + i]) * 0.0011
            );
        }
    }
}
