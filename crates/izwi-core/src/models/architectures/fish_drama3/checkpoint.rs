//! The sharded checkpoint: index, memory maps, census, FP8 experts and
//! row-slice reads.
//!
//! `model.safetensors.index.json` maps each of the 62,245 published tensors
//! to one of 15 shards. Every shard is memory-mapped once; the index and the
//! shard headers must agree both ways. The census derives the exact tensor
//! set (name, dtype, shape) from the config and fails the load on any
//! missing, extra, retyped or reshaped tensor, before a byte is uploaded.
//!
//! Only the routed experts are FP8 (`bsgemm-moe`): each
//! `mlp.experts.N.{gate,up,down}_proj.weight` is E4M3 with an F32
//! `weight_scale_inv` of one scale per 128×128 block. Despite its name the
//! scale is **multiplied**: `W[r, c] = e4m3(codes[r, c]) * scale[r / 128][c / 128]`.
//! Everything else is BF16.
//!
//! The main head is read by rows: only the 4,097 candidate rows of
//! `lm_head` (eos and the semantic band) are ever loaded, straight from the
//! mapping, so the 0.96 GiB tensor is never materialised.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use half::bf16;
use memmap2::Mmap;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;

use super::config::{FishDrama3Config, FishDrama3LayerKind};
use crate::error::{Error, Result};
use crate::kernels::cuda::fp8::decode_e4m3fn;

pub const INDEX_FILE: &str = "model.safetensors.index.json";
pub const CODEC_FILE: &str = "codec.pth";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
pub const SPECIAL_TOKENS_MAP_FILE: &str = "special_tokens_map.json";
/// Files beside the weights the family needs (parsed by later stages).
pub const COMPANION_FILES: [&str; 4] = [
    CODEC_FILE,
    TOKENIZER_FILE,
    TOKENIZER_CONFIG_FILE,
    SPECIAL_TOKENS_MAP_FILE,
];

pub const TRUNK_PREFIX: &str = "model.language_model";
pub const LM_HEAD: &str = "lm_head.weight";
pub const CODEBOOK_EMBEDDINGS: &str = "audio_decoder.codebook_embeddings.weight";
pub const TOWER_SHARED_PREFIX: &str = "audio_decoder_mtp_shared.layers";
pub const TOWER_RESIDUAL_PREFIX: &str = "audio_decoder_mtp.layers";
pub const TOWER_HEADS_PREFIX: &str = "audio_decoder_mtp.heads";
/// The block scales' suffix beside an FP8 `.weight`.
pub const FP8_SCALE_SUFFIX: &str = "weight_scale_inv";

/// `model.language_model.layers.{layer}`.
pub fn trunk_layer(layer: usize) -> String {
    format!("{TRUNK_PREFIX}.layers.{layer}")
}

/// The routed expert module `...layers.{layer}.mlp.experts.{expert}`.
pub fn expert(layer: usize, expert: usize) -> String {
    format!("{}.mlp.experts.{expert}", trunk_layer(layer))
}

/// Shared tower layer `index` (it embeds the semantic code).
pub fn tower_shared_layer(index: usize) -> String {
    format!("{TOWER_SHARED_PREFIX}.{index}")
}

/// Residual tower layer `index`: it injects code `index` and feeds head
/// `index`, which predicts code `index + 1`.
pub fn tower_residual_layer(index: usize) -> String {
    format!("{TOWER_RESIDUAL_PREFIX}.{index}")
}

/// Codebook head `index`'s weight.
pub fn tower_head(index: usize) -> String {
    format!("{TOWER_HEADS_PREFIX}.{index}.weight")
}

/// The dtypes this checkpoint holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CensusDtype {
    Bf16,
    F32,
    F8E4M3,
}

impl CensusDtype {
    pub fn safetensors(self) -> Dtype {
        match self {
            Self::Bf16 => Dtype::BF16,
            Self::F32 => Dtype::F32,
            Self::F8E4M3 => Dtype::F8_E4M3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F8E4M3 => "F8_E4M3",
        }
    }

    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::F8E4M3 => 1,
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

fn entry(out: &mut Vec<CensusEntry>, name: impl Into<String>, dtype: CensusDtype, shape: &[usize]) {
    out.push(CensusEntry {
        name: name.into(),
        dtype,
        shape: shape.to_vec(),
    });
}

/// An FP8 expert projection `[rows, cols]` and its block scales.
fn fp8_entries(
    out: &mut Vec<CensusEntry>,
    config: &FishDrama3Config,
    module: &str,
    rows: usize,
    cols: usize,
) {
    entry(
        out,
        format!("{module}.weight"),
        CensusDtype::F8E4M3,
        &[rows, cols],
    );
    entry(
        out,
        format!("{module}.{FP8_SCALE_SUFFIX}"),
        CensusDtype::F32,
        &config.fp8_scale_shape(rows, cols),
    );
}

/// One tower layer: its codebook embedding (`semantic_embedding` in the
/// shared layers, `codebook_embedding` in the residual ones), the
/// `eh_proj` merge of the normed embedding and hidden state, and the block.
fn tower_layer_entries(
    out: &mut Vec<CensusEntry>,
    config: &FishDrama3Config,
    prefix: &str,
    embedding: &str,
) {
    use CensusDtype::Bf16;
    let t = &config.tower;
    let d = t.dim;
    entry(
        out,
        format!("{prefix}.{embedding}.weight"),
        Bf16,
        &[t.codebook_vocab, d],
    );
    for norm in ["enorm", "hnorm", "final_layernorm"] {
        entry(out, format!("{prefix}.{norm}.weight"), Bf16, &[d]);
    }
    entry(out, format!("{prefix}.eh_proj.weight"), Bf16, &[d, 2 * d]);
    let b = format!("{prefix}.transformer_layer");
    let head_dim = t.attention.head_dim;
    entry(
        out,
        format!("{b}.attention.wqkv.weight"),
        Bf16,
        &[t.wqkv_rows(), d],
    );
    entry(
        out,
        format!("{b}.attention.wo.weight"),
        Bf16,
        &[d, t.attention.q_width()],
    );
    // Applied whatever `attention_qk_norm` says (config.rs).
    entry(
        out,
        format!("{b}.attention.q_norm.weight"),
        Bf16,
        &[head_dim],
    );
    entry(
        out,
        format!("{b}.attention.k_norm.weight"),
        Bf16,
        &[head_dim],
    );
    entry(out, format!("{b}.attention_norm.weight"), Bf16, &[d]);
    entry(out, format!("{b}.ffn_norm.weight"), Bf16, &[d]);
    let f = format!("{b}.feed_forward");
    entry(out, format!("{f}.w1.weight"), Bf16, &[t.intermediate, d]);
    entry(out, format!("{f}.w3.weight"), Bf16, &[t.intermediate, d]);
    entry(out, format!("{f}.w2.weight"), Bf16, &[d, t.intermediate]);
}

/// Every tensor the family expects, derived from the config.
pub fn expected_census(config: &FishDrama3Config) -> Vec<CensusEntry> {
    use CensusDtype::Bf16;
    let t = &config.trunk;
    let h = t.hidden;
    let mut out = Vec::new();
    entry(
        &mut out,
        format!("{TRUNK_PREFIX}.embed_tokens.weight"),
        Bf16,
        &[t.vocab, h],
    );
    entry(&mut out, format!("{TRUNK_PREFIX}.norm.weight"), Bf16, &[h]);
    // Untied; only the compact head's rows are ever read.
    entry(&mut out, LM_HEAD, Bf16, &[t.vocab, h]);
    for (layer, kind) in t.layers.iter().enumerate() {
        let l = trunk_layer(layer);
        entry(&mut out, format!("{l}.input_layernorm.weight"), Bf16, &[h]);
        entry(
            &mut out,
            format!("{l}.post_attention_layernorm.weight"),
            Bf16,
            &[h],
        );
        match kind {
            FishDrama3LayerKind::LinearAttention => {
                let la = &t.linear;
                let x = format!("{l}.linear_attn");
                let conv = la.conv_channels();
                entry(&mut out, format!("{x}.A_log"), Bf16, &[la.value_heads]);
                entry(&mut out, format!("{x}.dt_bias"), Bf16, &[la.value_heads]);
                entry(
                    &mut out,
                    format!("{x}.conv1d.weight"),
                    Bf16,
                    &[conv, 1, la.conv_kernel],
                );
                entry(
                    &mut out,
                    format!("{x}.in_proj_qkv.weight"),
                    Bf16,
                    &[conv, h],
                );
                entry(
                    &mut out,
                    format!("{x}.in_proj_z.weight"),
                    Bf16,
                    &[la.value_width(), h],
                );
                for name in ["in_proj_a", "in_proj_b"] {
                    entry(
                        &mut out,
                        format!("{x}.{name}.weight"),
                        Bf16,
                        &[la.value_heads, h],
                    );
                }
                entry(
                    &mut out,
                    format!("{x}.norm.weight"),
                    Bf16,
                    &[la.value_head_dim],
                );
                entry(
                    &mut out,
                    format!("{x}.out_proj.weight"),
                    Bf16,
                    &[h, la.value_width()],
                );
            }
            FishDrama3LayerKind::FullAttention => {
                let at = &t.attention;
                let x = format!("{l}.self_attn");
                // Query and gate, packed per head.
                entry(
                    &mut out,
                    format!("{x}.q_proj.weight"),
                    Bf16,
                    &[2 * at.q_width(), h],
                );
                entry(
                    &mut out,
                    format!("{x}.k_proj.weight"),
                    Bf16,
                    &[at.kv_width(), h],
                );
                entry(
                    &mut out,
                    format!("{x}.v_proj.weight"),
                    Bf16,
                    &[at.kv_width(), h],
                );
                entry(
                    &mut out,
                    format!("{x}.o_proj.weight"),
                    Bf16,
                    &[h, at.q_width()],
                );
                entry(&mut out, format!("{x}.q_norm.weight"), Bf16, &[at.head_dim]);
                entry(&mut out, format!("{x}.k_norm.weight"), Bf16, &[at.head_dim]);
            }
        }
        let moe = &t.moe;
        let m = format!("{l}.mlp");
        entry(
            &mut out,
            format!("{m}.gate.weight"),
            Bf16,
            &[moe.experts, h],
        );
        let s = format!("{m}.shared_expert");
        let si = moe.shared_intermediate;
        entry(&mut out, format!("{s}.gate_proj.weight"), Bf16, &[si, h]);
        entry(&mut out, format!("{s}.up_proj.weight"), Bf16, &[si, h]);
        entry(&mut out, format!("{s}.down_proj.weight"), Bf16, &[h, si]);
        entry(
            &mut out,
            format!("{m}.shared_expert_gate.weight"),
            Bf16,
            &[1, h],
        );
        for e in 0..moe.experts {
            let x = expert(layer, e);
            fp8_entries(
                &mut out,
                config,
                &format!("{x}.gate_proj"),
                moe.intermediate,
                h,
            );
            fp8_entries(
                &mut out,
                config,
                &format!("{x}.up_proj"),
                moe.intermediate,
                h,
            );
            fp8_entries(
                &mut out,
                config,
                &format!("{x}.down_proj"),
                h,
                moe.intermediate,
            );
        }
    }

    let tower = &config.tower;
    entry(
        &mut out,
        CODEBOOK_EMBEDDINGS,
        Bf16,
        &[tower.codebooks * tower.codebook_vocab, tower.dim],
    );
    for index in 0..tower.shared_layers {
        tower_layer_entries(
            &mut out,
            config,
            &tower_shared_layer(index),
            "semantic_embedding",
        );
    }
    for index in 0..tower.residual_layers() {
        tower_layer_entries(
            &mut out,
            config,
            &tower_residual_layer(index),
            "codebook_embedding",
        );
        entry(
            &mut out,
            tower_head(index),
            Bf16,
            &[tower.codebook_vocab, tower.dim],
        );
    }
    out
}

/// Every file beside the weights that later stages read must be present
/// and non-empty. `codec.pth` is parsed by the codec stage.
pub fn check_companion_files(model_dir: &Path) -> Result<()> {
    for name in COMPANION_FILES {
        let path = model_dir.join(name);
        match std::fs::metadata(&path) {
            Ok(meta) if meta.is_file() && meta.len() > 0 => {}
            Ok(_) => {
                return Err(load_error(format!(
                    "{} is empty or not a file",
                    path.display()
                )))
            }
            Err(err) => return Err(load_error(format!("missing {}: {err}", path.display()))),
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RawIndex {
    #[serde(default)]
    metadata: Option<RawIndexMetadata>,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawIndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
}

/// Where a tensor's bytes live.
#[derive(Debug, Clone)]
struct Located {
    file: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    /// Absolute byte range in the file's mapping.
    start: usize,
    end: usize,
}

/// A routed expert projection: E4M3 codes borrowed from the mapping and its
/// F32 block scales.
#[derive(Debug, Clone)]
pub struct Fp8BlockWeight<'a> {
    /// Row-major `[rows, cols]` E4M3FN bytes.
    pub codes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub block: usize,
    /// Row-major `[ceil(rows / block), ceil(cols / block)]`.
    pub scales: Vec<f32>,
}

impl Fp8BlockWeight<'_> {
    pub fn scale_shape(&self) -> [usize; 2] {
        [
            self.rows.div_ceil(self.block),
            self.cols.div_ceil(self.block),
        ]
    }

    /// The scale multiplying element `(row, col)`.
    pub fn scale(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let [_, scale_cols] = self.scale_shape();
        self.scales
            .get((row / self.block) * scale_cols + col / self.block)
            .copied()
    }

    fn scale_row(&self, row: usize) -> &[f32] {
        let [_, scale_cols] = self.scale_shape();
        let start = (row / self.block) * scale_cols;
        &self.scales[start..start + scale_cols]
    }

    fn dequantize_into(&self, row: usize, codes: &[u8], out: &mut Vec<f32>) {
        let scales = self.scale_row(row);
        out.extend(
            codes
                .iter()
                .enumerate()
                .map(|(col, &code)| decode_e4m3fn(code) * scales[col / self.block]),
        );
    }

    /// Row `row` dequantized exactly in F32.
    pub fn dequantized_row(&self, row: usize) -> Option<Vec<f32>> {
        let codes = self.codes.chunks_exact(self.cols).nth(row)?;
        let mut out = Vec::with_capacity(self.cols);
        self.dequantize_into(row, codes, &mut out);
        Some(out)
    }

    /// The whole matrix dequantized exactly in F32, row-major.
    pub fn dequantized(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for (row, codes) in self.codes.chunks_exact(self.cols).enumerate() {
            self.dequantize_into(row, codes, &mut out);
        }
        out
    }
}

struct MappedFile {
    name: String,
    mmap: Mmap,
}

/// The memory-mapped checkpoint.
pub struct FishDrama3Checkpoint {
    model_dir: PathBuf,
    files: Vec<MappedFile>,
    tensors: BTreeMap<String, Located>,
}

impl std::fmt::Debug for FishDrama3Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3Checkpoint")
            .field("model_dir", &self.model_dir)
            .field(
                "files",
                &self.files.iter().map(|file| &file.name).collect::<Vec<_>>(),
            )
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

fn load_error(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!("Fish Audio Drama 3 checkpoint: {}", message.into()))
}

/// A shard name the index may use: one plain `*.safetensors` file name.
fn safe_shard_name(name: &str) -> bool {
    let path = Path::new(name);
    name.ends_with(".safetensors")
        && path.components().count() == 1
        && matches!(
            path.components().next(),
            Some(std::path::Component::Normal(_))
        )
}

/// Maps the safetensors file `path` (`name` in messages) as file `position`
/// of a checkpoint: the mapping and where each of its tensors lies.
fn map_file(
    path: &Path,
    name: &str,
    position: usize,
) -> Result<(MappedFile, Vec<(String, Located)>)> {
    let file = File::open(path)
        .map_err(|err| load_error(format!("cannot open {}: {err}", path.display())))?;
    // SAFETY: the mapping is read-only and the file is not modified through
    // this process while the checkpoint lives.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|err| load_error(format!("cannot map {}: {err}", path.display())))?;
    let (header, metadata) = SafeTensors::read_metadata(&mmap)
        .map_err(|err| load_error(format!("invalid header in {name}: {err}")))?;
    let data_start = 8 + header;
    let held = metadata
        .tensors()
        .into_iter()
        .map(|(tensor, info)| {
            let (start, end) = info.data_offsets;
            let located = Located {
                file: position,
                dtype: info.dtype,
                shape: info.shape.clone(),
                start: data_start + start,
                end: data_start + end,
            };
            (tensor, located)
        })
        .collect();
    Ok((
        MappedFile {
            name: name.to_string(),
            mmap,
        },
        held,
    ))
}

fn shape_text(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

impl FishDrama3Checkpoint {
    /// Maps every shard the index names; the index and the shard headers
    /// must agree tensor for tensor.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join(INDEX_FILE);
        let raw = std::fs::read(&index_path)
            .map_err(|err| load_error(format!("cannot read {}: {err}", index_path.display())))?;
        let index: RawIndex = serde_json::from_slice(&raw)
            .map_err(|err| load_error(format!("invalid {INDEX_FILE}: {err}")))?;
        let shard_names = index.weight_map.values().collect::<BTreeSet<_>>();
        if shard_names.is_empty() {
            return Err(load_error(format!("{INDEX_FILE} names no shard")));
        }
        let mut files = Vec::with_capacity(shard_names.len());
        let mut tensors = BTreeMap::new();
        for name in shard_names {
            if !safe_shard_name(name) {
                return Err(load_error(format!(
                    "{INDEX_FILE} names an unsafe shard {name:?}"
                )));
            }
            let (file, held) = map_file(&model_dir.join(name), name, files.len())?;
            for (tensor, located) in held {
                if tensors.insert(tensor.clone(), located).is_some() {
                    return Err(load_error(format!("{tensor} is held by two shards")));
                }
            }
            files.push(file);
        }
        for (tensor, shard) in &index.weight_map {
            let held_by = tensors.get(tensor).map(|located| &files[located.file].name);
            if held_by != Some(shard) {
                return Err(load_error(format!(
                    "{INDEX_FILE} maps {tensor} to {shard}, which does not hold it"
                )));
            }
        }
        if tensors.len() != index.weight_map.len() {
            let stray = tensors
                .keys()
                .find(|name| !index.weight_map.contains_key(*name))
                .cloned()
                .unwrap_or_default();
            return Err(load_error(format!(
                "a shard holds {stray}, which {INDEX_FILE} does not map"
            )));
        }
        if let Some(total) = index.metadata.and_then(|metadata| metadata.total_size) {
            let bytes = tensors
                .values()
                .map(|located| (located.end - located.start) as u64)
                .sum::<u64>();
            if bytes != total {
                return Err(load_error(format!(
                    "the shards hold {bytes} tensor bytes; {INDEX_FILE} declares {total}"
                )));
            }
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            files,
            tensors,
        })
    }

    /// A view over whole safetensors `paths` without an index (the parity
    /// slices and their RED variants): every tensor `rename` names (`None`
    /// leaves it out), a later file's tensor replacing an earlier one of the
    /// same name. No census: the caller loads only what its config needs.
    #[cfg(test)]
    pub(crate) fn open_view(
        paths: &[PathBuf],
        rename: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = BTreeMap::new();
        for path in paths {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| load_error(format!("{} names no file", path.display())))?;
            let (file, held) = map_file(path, name, files.len())?;
            for (tensor, located) in held {
                if let Some(renamed) = rename(&tensor) {
                    tensors.insert(renamed, located);
                }
            }
            files.push(file);
        }
        let model_dir = paths
            .first()
            .and_then(|path| path.parent())
            .ok_or_else(|| load_error("a checkpoint view needs a file"))?
            .to_path_buf();
        Ok(Self {
            model_dir,
            files,
            tensors,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// The shard file names, sorted.
    pub fn shard_files(&self) -> impl Iterator<Item = &str> {
        self.files.iter().map(|file| file.name.as_str())
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Tensor names, sorted.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    fn located(&self, name: &str) -> Result<&Located> {
        self.tensors
            .get(name)
            .ok_or_else(|| load_error(format!("missing tensor {name}")))
    }

    /// The tensor's stored dtype and shape.
    pub fn info(&self, name: &str) -> Result<(Dtype, &[usize])> {
        let located = self.located(name)?;
        Ok((located.dtype, &located.shape))
    }

    /// The tensor's raw bytes, borrowed from the mapping.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let located = self.located(name)?;
        self.files[located.file]
            .mmap
            .get(located.start..located.end)
            .ok_or_else(|| load_error(format!("{name} lies outside its shard")))
    }

    /// Checks the tensor set against [`expected_census`]: every expected
    /// tensor at its dtype and shape, and nothing else.
    pub fn validate_census(&self, config: &FishDrama3Config) -> Result<()> {
        const SHOWN: usize = 8;
        let expected = expected_census(config);
        let mut problems = Vec::new();
        let mut names = HashSet::with_capacity(expected.len());
        for e in &expected {
            names.insert(e.name.as_str());
            let Some(found) = self.tensors.get(&e.name) else {
                problems.push(format!("missing {}", e.name));
                continue;
            };
            if found.dtype != e.dtype.safetensors() {
                problems.push(format!(
                    "{} is {:?}; expected {}",
                    e.name,
                    found.dtype,
                    e.dtype.name()
                ));
            } else if found.shape != e.shape {
                problems.push(format!(
                    "{} has shape {}; expected {}",
                    e.name,
                    shape_text(&found.shape),
                    shape_text(&e.shape)
                ));
            }
        }
        for name in self.tensors.keys() {
            if !names.contains(name.as_str()) {
                problems.push(format!("unexpected tensor {name}"));
            }
        }
        if problems.is_empty() {
            return Ok(());
        }
        let more = problems.len().saturating_sub(SHOWN);
        let mut message = format!(
            "the tensors do not match the census ({} problems): {}",
            problems.len(),
            problems[..problems.len().min(SHOWN)].join("; ")
        );
        if more > 0 {
            message.push_str(&format!("; and {more} more"));
        }
        Err(load_error(message))
    }

    /// A BF16 tensor's bytes, checked against `shape`.
    fn bf16_bytes(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let (dtype, found) = self.info(name)?;
        if dtype != Dtype::BF16 || found != shape {
            return Err(load_error(format!(
                "{name} is {dtype:?} {}; expected BF16 {}",
                shape_text(found),
                shape_text(shape)
            )));
        }
        self.bytes(name)
    }

    /// A BF16 tensor's values in F32 (exact), checked against `shape`.
    pub fn bf16_to_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(bf16_bytes_to_f32(self.bf16_bytes(name, shape)?))
    }

    /// A BF16 tensor as a Candle tensor of `dtype` on `device`, checked
    /// against `shape`. The cast runs on the host before the upload.
    pub fn bf16_tensor(
        &self,
        name: &str,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let bytes = self.bf16_bytes(name, shape)?;
        bf16_bytes_to_tensor(bytes, shape, dtype, device)
    }

    /// The BF16 bytes of rows `rows` of the 2-D tensor `name`, in order:
    /// rows must be strictly ascending and in range. Only those rows are
    /// touched (runs of consecutive rows are copied at once), so a slice of
    /// a large table never materialises the rest.
    fn bf16_row_bytes(&self, name: &str, rows: &[usize]) -> Result<(Vec<u8>, usize)> {
        let (dtype, shape) = self.info(name)?;
        let &[count, cols] = shape else {
            return Err(load_error(format!(
                "{name} has shape {}; a row slice needs a matrix",
                shape_text(shape)
            )));
        };
        if dtype != Dtype::BF16 {
            return Err(load_error(format!(
                "{name} is {dtype:?}; a row slice reads BF16"
            )));
        }
        if rows.is_empty() {
            return Err(load_error(format!("an empty row slice of {name}")));
        }
        if let Some(pair) = rows.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(load_error(format!(
                "the row slice of {name} is not strictly ascending at {} then {}",
                pair[0], pair[1]
            )));
        }
        let last = rows[rows.len() - 1];
        if last >= count {
            return Err(load_error(format!(
                "row {last} is past {name}'s {count} rows"
            )));
        }
        let data = self.bytes(name)?;
        let row_bytes = cols * 2;
        let mut out = Vec::with_capacity(rows.len() * row_bytes);
        let mut next = 0;
        while next < rows.len() {
            let first = rows[next];
            let mut end = first + 1;
            next += 1;
            while next < rows.len() && rows[next] == end {
                end += 1;
                next += 1;
            }
            let run = data
                .get(first * row_bytes..end * row_bytes)
                .ok_or_else(|| load_error(format!("rows of {name} lie outside its shard")))?;
            out.extend_from_slice(run);
        }
        Ok((out, cols))
    }

    /// Rows `rows` (strictly ascending) of the BF16 matrix `name`, in F32.
    pub fn bf16_rows_to_f32(&self, name: &str, rows: &[usize]) -> Result<Vec<f32>> {
        let (bytes, _) = self.bf16_row_bytes(name, rows)?;
        Ok(bf16_bytes_to_f32(&bytes))
    }

    /// Rows `rows` (strictly ascending) of the BF16 matrix `name` as a
    /// `[rows.len(), cols]` tensor of `dtype` on `device`.
    pub fn bf16_rows_tensor(
        &self,
        name: &str,
        rows: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let (bytes, cols) = self.bf16_row_bytes(name, rows)?;
        bf16_bytes_to_tensor(&bytes, &[rows.len(), cols], dtype, device)
    }

    /// The routed expert projection `module` (`module.weight` E4M3 and
    /// `module.weight_scale_inv` F32), checked against `[rows, cols]` and
    /// the `block` grid. Codes holding the E4M3 NaN and non-finite scales
    /// are refused: either would poison every product it meets.
    pub fn fp8_block_weight(
        &self,
        module: &str,
        rows: usize,
        cols: usize,
        block: usize,
    ) -> Result<Fp8BlockWeight<'_>> {
        let weight = format!("{module}.weight");
        let (dtype, shape) = self.info(&weight)?;
        if dtype != Dtype::F8_E4M3 || shape != [rows, cols] {
            return Err(load_error(format!(
                "{weight} is {dtype:?} {}; expected F8_E4M3 [{rows}, {cols}]",
                shape_text(shape)
            )));
        }
        if block == 0 || rows == 0 || cols == 0 {
            return Err(load_error(format!(
                "{weight}: an empty FP8 weight or block grid"
            )));
        }
        let codes = self.bytes(&weight)?;
        if let Some(position) = codes.iter().position(|&code| code & 0x7f == 0x7f) {
            return Err(load_error(format!(
                "{weight} holds the E4M3 NaN at element {position}"
            )));
        }
        let scale_name = format!("{module}.{FP8_SCALE_SUFFIX}");
        let scale_shape = [rows.div_ceil(block), cols.div_ceil(block)];
        let (scale_dtype, found) = self.info(&scale_name)?;
        if scale_dtype != Dtype::F32 || found != scale_shape {
            return Err(load_error(format!(
                "{scale_name} is {scale_dtype:?} {}; expected F32 {}",
                shape_text(found),
                shape_text(&scale_shape)
            )));
        }
        let scales = self
            .bytes(&scale_name)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        if let Some(bad) = scales.iter().find(|scale| !scale.is_finite()) {
            return Err(load_error(format!("{scale_name} holds the scale {bad}")));
        }
        Ok(Fp8BlockWeight {
            codes,
            rows,
            cols,
            block,
            scales,
        })
    }
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b).to_f32())
        .collect()
}

fn bf16_bytes_to_tensor(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let host = Tensor::from_raw_buffer(bytes, DType::BF16, shape, &Device::Cpu)?;
    Ok(host.to_dtype(dtype)?.to_device(device)?)
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::{published, synthetic, synthetic_json};
    use super::super::synthetic::{pattern_tensors, temp_dir, write_checkpoint, TempDir, Written};
    use super::*;
    use sha2::{Digest, Sha256};

    const CENSUS: &str = include_str!("../../../../tests/fixtures/fish_drama3/tensor_census.txt");
    const PUBLISHED_NAMES: &str =
        include_str!("../../../../tests/fixtures/fish_drama3/published_names.txt");

    /// `layers.12.mlp.experts.3.x` → `layers.N.mlp.experts.N.x`.
    fn collapse(name: &str) -> String {
        name.split('.')
            .map(|part| {
                if !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()) {
                    "N"
                } else {
                    part
                }
            })
            .collect::<Vec<_>>()
            .join(".")
    }

    /// The census derived from the published config, collapsed to name
    /// patterns, equals the census read from the 15 published headers.
    #[test]
    fn the_derived_census_equals_the_published_headers() {
        let census = expected_census(&published());
        let mut derived = BTreeMap::<String, usize>::new();
        for e in &census {
            let key = format!(
                "{} {} {}",
                collapse(&e.name),
                e.dtype.name(),
                shape_text(&e.shape)
            );
            *derived.entry(key).or_default() += 1;
        }
        let mut expected = BTreeMap::<String, usize>::new();
        for line in CENSUS.lines().filter(|line| !line.trim().is_empty()) {
            let (count, rest) = line.split_once(' ').unwrap();
            *expected.entry(rest.to_string()).or_default() += count.parse::<usize>().unwrap();
        }
        assert_eq!(derived, expected);
        assert_eq!(census.len(), 62_245);
        // The index's total_size.
        let bytes = census
            .iter()
            .map(|e| e.shape.iter().product::<usize>() * e.dtype.size_in_bytes())
            .sum::<usize>();
        assert_eq!(bytes, 39_594_895_616);
    }

    /// The derived names are exactly the published index's names.
    #[test]
    fn the_derived_names_hash_to_the_published_index() {
        let mut fields = BTreeMap::new();
        for line in PUBLISHED_NAMES.lines().filter(|l| !l.starts_with('#')) {
            let (key, value) = line.split_once(' ').unwrap();
            fields.insert(key, value);
        }
        let names = expected_census(&published())
            .into_iter()
            .map(|e| e.name)
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len().to_string(), fields["count"]);
        let joined = names.into_iter().collect::<Vec<_>>().join("\n");
        assert_eq!(
            hex::encode(Sha256::digest(joined.as_bytes())),
            fields["sha256"]
        );
    }

    #[test]
    fn the_nan_screen_matches_the_e4m3_decoder_on_every_code() {
        for code in 0..=255u8 {
            assert_eq!(
                code & 0x7f == 0x7f,
                decode_e4m3fn(code).is_nan(),
                "{code:#04x}"
            );
        }
        // Spot values the multiply test below relies on.
        assert_eq!(decode_e4m3fn(0x38), 1.0);
        assert_eq!(decode_e4m3fn(0x40), 2.0);
        assert_eq!(decode_e4m3fn(0x3c), 1.5);
        assert_eq!(decode_e4m3fn(0xb8), -1.0);
        assert_eq!(decode_e4m3fn(0x7e), 448.0);
        assert_eq!(decode_e4m3fn(0x01), 2f32.powi(-9));
    }

    /// The synthetic checkpoint with one edit to its tensor list.
    fn written(edit: impl FnOnce(&mut Vec<Written>)) -> TempDir {
        let mut tensors = pattern_tensors(&synthetic());
        edit(&mut tensors);
        let dir = temp_dir();
        write_checkpoint(&dir.0, &tensors, &synthetic_json());
        dir
    }

    fn census_error(edit: impl FnOnce(&mut Vec<Written>)) -> String {
        let dir = written(edit);
        let checkpoint = FishDrama3Checkpoint::open(&dir.0).unwrap();
        checkpoint
            .validate_census(&synthetic())
            .unwrap_err()
            .to_string()
    }

    fn position(tensors: &[Written], name: &str) -> usize {
        tensors.iter().position(|t| t.0 == name).unwrap()
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn a_synthetic_two_shard_checkpoint_round_trips() {
        let config = synthetic();
        let tensors = pattern_tensors(&config);
        let dir = temp_dir();
        write_checkpoint(&dir.0, &tensors, &synthetic_json());
        let checkpoint = FishDrama3Checkpoint::open(&dir.0).unwrap();
        checkpoint.validate_census(&config).unwrap();
        check_companion_files(&dir.0).unwrap();
        assert_eq!(checkpoint.tensor_count(), tensors.len());
        assert_eq!(
            checkpoint.shard_files().collect::<Vec<_>>(),
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );

        // BF16 values, exactly.
        let norm = format!("{TRUNK_PREFIX}.norm.weight");
        let (_, _, _, bytes) = &tensors[position(&tensors, &norm)];
        let values = checkpoint.bf16_to_f32(&norm, &[256]).unwrap();
        assert!(same_bits(&values, &bf16_bytes_to_f32(bytes)));
        assert!(checkpoint.bf16_to_f32(&norm, &[255]).is_err());
        let tensor = checkpoint
            .bf16_tensor(&norm, &[256], DType::F32, &Device::Cpu)
            .unwrap();
        assert_eq!(tensor.dims(), [256]);
        assert!(same_bits(&tensor.to_vec1::<f32>().unwrap(), &values));
        let half = checkpoint
            .bf16_tensor(&norm, &[256], DType::BF16, &Device::Cpu)
            .unwrap();
        assert_eq!(half.dtype(), DType::BF16);

        // Row slices equal the rows of the whole table.
        let full = checkpoint.bf16_to_f32(LM_HEAD, &[64, 256]).unwrap();
        for rows in [
            config.ids.compact_head_rows(),
            vec![0, 5, 6, 7, 63],
            vec![63],
        ] {
            let slice = checkpoint.bf16_rows_to_f32(LM_HEAD, &rows).unwrap();
            let expected = rows
                .iter()
                .flat_map(|&r| full[r * 256..(r + 1) * 256].iter().copied())
                .collect::<Vec<_>>();
            assert!(same_bits(&slice, &expected), "{rows:?}");
            let tensor = checkpoint
                .bf16_rows_tensor(LM_HEAD, &rows, DType::F32, &Device::Cpu)
                .unwrap();
            assert_eq!(tensor.dims(), [rows.len(), 256]);
            assert!(same_bits(
                &tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                &expected
            ));
        }
        for bad in [vec![], vec![5, 3], vec![5, 5], vec![64]] {
            assert!(
                checkpoint.bf16_rows_to_f32(LM_HEAD, &bad).is_err(),
                "{bad:?}"
            );
        }
        assert!(checkpoint.bf16_rows_to_f32(&norm, &[0]).is_err());
        let gate = format!("{}.gate_proj", expert(0, 1));
        assert!(checkpoint
            .bf16_rows_to_f32(&format!("{gate}.weight"), &[0])
            .is_err());

        // FP8 codes and scales as written.
        let w = checkpoint.fp8_block_weight(&gate, 64, 256, 128).unwrap();
        let (_, _, _, codes) = &tensors[position(&tensors, &format!("{gate}.weight"))];
        assert_eq!(w.codes, codes.as_slice());
        assert_eq!(w.scale_shape(), [1, 2]);
        assert!(checkpoint.fp8_block_weight(&gate, 256, 64, 128).is_err());
        assert!(checkpoint.fp8_block_weight(&gate, 64, 256, 64).is_err());
    }

    /// The device path is the host path plus an upload: the values read
    /// back from Metal are the host's, bit for bit.
    #[cfg(feature = "metal")]
    #[test]
    fn bf16_reads_upload_to_metal_unchanged() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("bf16_reads_upload_to_metal_unchanged")
        else {
            return;
        };
        let dir = written(|_| {});
        let checkpoint = FishDrama3Checkpoint::open(&dir.0).unwrap();
        let rows = synthetic().ids.compact_head_rows();
        let host_f32 = |tensor: &Tensor| {
            tensor
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let host = checkpoint
                .bf16_rows_tensor(LM_HEAD, &rows, dtype, &Device::Cpu)
                .unwrap();
            let metal = checkpoint
                .bf16_rows_tensor(LM_HEAD, &rows, dtype, &device)
                .unwrap();
            assert!(metal.device().is_metal());
            assert_eq!((metal.dtype(), metal.dims()), (dtype, host.dims()));
            assert!(same_bits(&host_f32(&metal), &host_f32(&host)), "{dtype:?}");
        }
    }

    /// One expert with hand-set codes and scales: the dequantized values
    /// are `e4m3(code) * scale`, never `/ scale`.
    #[test]
    fn fp8_blocks_dequantize_by_multiplying_the_scale() {
        let gate = format!("{}.gate_proj", expert(0, 1));
        let dir = written(|tensors| {
            let at = position(tensors, &format!("{gate}.weight"));
            let codes = &mut tensors[at].3;
            codes.fill(0);
            // [64, 256]: two 128-column blocks.
            codes[0] = 0x38; // (0, 0): 1.0, block (0, 0)
            codes[3 * 256 + 130] = 0x40; // (3, 130): 2.0, block (0, 1)
            codes[10 * 256 + 127] = 0x3c; // (10, 127): 1.5, last column of block (0, 0)
            codes[10 * 256 + 128] = 0x3c; // (10, 128): 1.5, first column of block (0, 1)
            codes[63 * 256 + 255] = 0xb8; // (63, 255): -1.0, block (0, 1)
            let at = position(tensors, &format!("{gate}.{FP8_SCALE_SUFFIX}"));
            tensors[at].3 = [0.5f32, 0.25]
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
        });
        let checkpoint = FishDrama3Checkpoint::open(&dir.0).unwrap();
        let w = checkpoint.fp8_block_weight(&gate, 64, 256, 128).unwrap();
        assert_eq!(w.scales, [0.5, 0.25]);
        assert_eq!(w.scale(10, 127), Some(0.5));
        assert_eq!(w.scale(10, 128), Some(0.25));
        assert_eq!(w.scale(64, 0), None);
        let all = w.dequantized();
        let at = |r: usize, c: usize| all[r * 256 + c];
        assert_eq!(at(0, 0), 0.5);
        assert_eq!(at(3, 130), 0.5);
        assert_eq!(at(10, 127), 0.75);
        assert_eq!(at(10, 128), 0.375);
        assert_eq!(at(63, 255), -0.25);
        assert_eq!(all.iter().filter(|v| **v != 0.0).count(), 5);
        let row = w.dequantized_row(10).unwrap();
        assert!(same_bits(&row, &all[10 * 256..11 * 256]));
        assert!(w.dequantized_row(64).is_none());
    }

    #[test]
    fn fp8_nan_codes_and_non_finite_scales_are_refused() {
        let gate = format!("{}.gate_proj", expert(1, 0));
        let dir = written(|tensors| {
            let at = position(tensors, &format!("{gate}.weight"));
            tensors[at].3[77] = 0xff;
        });
        let checkpoint = FishDrama3Checkpoint::open(&dir.0).unwrap();
        let err = checkpoint
            .fp8_block_weight(&gate, 64, 256, 128)
            .unwrap_err();
        assert!(err.to_string().contains("NaN at element 77"), "{err}");

        let dir = written(|tensors| {
            let at = position(tensors, &format!("{gate}.{FP8_SCALE_SUFFIX}"));
            tensors[at].3 = [f32::INFINITY, 1.0]
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
        });
        let checkpoint = FishDrama3Checkpoint::open(&dir.0).unwrap();
        assert!(checkpoint.fp8_block_weight(&gate, 64, 256, 128).is_err());
    }

    #[test]
    fn the_census_rejects_missing_extra_retyped_and_reshaped_tensors() {
        let head = tower_head(1);
        let err = census_error(|tensors| {
            let at = position(tensors, &head);
            tensors.remove(at);
        });
        assert!(err.contains(&format!("missing {head}")), "{err}");

        let extra = tower_head(2);
        let err = census_error(|tensors| {
            tensors.push((
                extra.clone(),
                Dtype::BF16,
                vec![8, 256],
                vec![0; 8 * 256 * 2],
            ));
        });
        assert!(err.contains(&format!("unexpected tensor {extra}")), "{err}");

        let err = census_error(|tensors| {
            let at = position(tensors, LM_HEAD);
            tensors[at] = (
                LM_HEAD.into(),
                Dtype::F32,
                vec![64, 256],
                vec![0; 64 * 256 * 4],
            );
        });
        assert!(
            err.contains("lm_head.weight is F32; expected BF16"),
            "{err}"
        );

        let norm = format!("{TRUNK_PREFIX}.norm.weight");
        let err = census_error(|tensors| {
            let at = position(tensors, &norm);
            tensors[at] = (norm.clone(), Dtype::BF16, vec![255], vec![0; 255 * 2]);
        });
        assert!(
            err.contains(&format!("{norm} has shape [255]; expected [256]")),
            "{err}"
        );

        // Many problems are counted and the first few shown.
        let err = census_error(|tensors| tensors.retain(|t| !t.0.contains(".experts.")));
        assert!(
            err.contains("(96 problems)") && err.contains("and 88 more"),
            "{err}"
        );
    }

    #[test]
    fn the_index_and_the_shards_must_agree() {
        let dir = written(|_| {});
        let index_path = dir.0.join(INDEX_FILE);
        let original = std::fs::read_to_string(&index_path).unwrap();
        let mut index: serde_json::Value = serde_json::from_str(&original).unwrap();

        // The index maps a tensor to the shard that does not hold it.
        index["weight_map"][LM_HEAD] = "model-00001-of-00002.safetensors".into();
        std::fs::write(&index_path, index.to_string()).unwrap();
        let err = FishDrama3Checkpoint::open(&dir.0).unwrap_err();
        assert!(err.to_string().contains("does not hold it"), "{err}");

        // A shard holds a tensor the index does not map.
        let mut index: serde_json::Value = serde_json::from_str(&original).unwrap();
        index["weight_map"].as_object_mut().unwrap().remove(LM_HEAD);
        std::fs::write(&index_path, index.to_string()).unwrap();
        let err = FishDrama3Checkpoint::open(&dir.0).unwrap_err();
        assert!(err.to_string().contains("does not map"), "{err}");

        // A wrong total, and an unsafe shard name.
        let mut index: serde_json::Value = serde_json::from_str(&original).unwrap();
        index["metadata"]["total_size"] = 1.into();
        std::fs::write(&index_path, index.to_string()).unwrap();
        assert!(FishDrama3Checkpoint::open(&dir.0).is_err());
        let mut index: serde_json::Value = serde_json::from_str(&original).unwrap();
        index["weight_map"][LM_HEAD] = "../model-00002-of-00002.safetensors".into();
        std::fs::write(&index_path, index.to_string()).unwrap();
        let err = FishDrama3Checkpoint::open(&dir.0).unwrap_err();
        assert!(err.to_string().contains("unsafe shard"), "{err}");

        std::fs::write(&index_path, original).unwrap();
        assert!(FishDrama3Checkpoint::open(&dir.0).is_ok());
    }

    #[test]
    fn the_codec_and_tokenizer_files_must_be_present() {
        let dir = written(|_| {});
        check_companion_files(&dir.0).unwrap();
        std::fs::remove_file(dir.0.join(CODEC_FILE)).unwrap();
        let err = check_companion_files(&dir.0).unwrap_err();
        assert!(err.to_string().contains(CODEC_FILE), "{err}");
        std::fs::write(dir.0.join(CODEC_FILE), b"").unwrap();
        assert!(check_companion_files(&dir.0).is_err());
        std::fs::write(dir.0.join(CODEC_FILE), b"PK").unwrap();
        std::fs::remove_file(dir.0.join(TOKENIZER_FILE)).unwrap();
        assert!(check_companion_files(&dir.0).is_err());
    }
}
