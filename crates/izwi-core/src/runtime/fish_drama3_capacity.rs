//! Fish Audio Drama 3's load-time rows (§7.3), sealed into its adapter
//! through Fish S2's interface ([`ResolvedFishServingCapacity`]). Live
//! resource admission stays authoritative: the managed state fit and the
//! per-request reservations still price every allocation.
//!
//! One row costs, at the backend's KV dtype:
//!
//! ```text
//! row_bytes = recurrent + conv state (62.8 MiB published)
//!           + KV per position (trunk and tower) × (prompt rows + frames + 1)
//!           + the span workspace + the decode workspace
//!           + the codec stream history, twice (current and rollback)
//!           + the codec quantum's workspace + its output credit
//! rows      = (headroom − safety reserve − 2 × reference preparation) / row_bytes
//! ```
//!
//! The rows are planned at the worst admitted request, 8,192 prompt rows and
//! 4,096 frames (or fewer when the operator caps the context), so every
//! request the budgets admit fits a sealed row. They are capped by the
//! family's width: one on the CPU, one on Metal (two with
//! `IZWI_FISH_DRAMA3_NATIVE_BATCHING`), eight on CUDA (D12), and by the
//! operator's row limits.

use candle_core::DType;

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::models::architectures::fish_drama3::codec::FishDrama3DacConfig;
use crate::models::architectures::fish_drama3::config::FishDrama3Config;
use crate::models::architectures::fish_drama3::model::{
    FishDrama3StateGeometry, DEFAULT_PREFILL_SPAN_ROWS, FISH_DRAMA3_AUDIO_CHUNK_FRAMES,
};
use crate::models::architectures::fish_drama3::physical::{
    kv_dtype, FishDrama3Workspace, STATE_CONTEXT_TOKENS,
};
use crate::models::architectures::fish_drama3::prompt::MAX_PROMPT_ROWS;
use crate::models::architectures::fish_drama3::reference_cache::MAX_REFERENCE_SECONDS;
use crate::runtime::fish_capacity::ResolvedFishServingCapacity;

/// Metal's opt-in second decode row.
pub(crate) const NATIVE_BATCHING_ENV: &str = "IZWI_FISH_DRAMA3_NATIVE_BATCHING";
/// CUDA's automatic decode width (D12).
pub(crate) const CUDA_WIDTH: usize = 8;
/// CUDA decode rows an operator may ask for (`IZWI_FISH_DRAMA3_CUDA_WIDTH`).
pub(crate) const CUDA_WIDTH_ENV: &str = "IZWI_FISH_DRAMA3_CUDA_WIDTH";

/// CUDA decode rows: `value` (1 to the CUDA ceiling), else [`CUDA_WIDTH`].
/// On the H100 Drama 3 stayed at ~4.9 audio-s/s from 8 to 64 streams, the
/// width of eight; its decode graphs cover 1 to 4 rows and 8, so a wider
/// step runs eagerly, and whether that pays is measured per GPU. An invalid
/// value keeps the default (with a warning).
pub(crate) fn cuda_width(value: Option<&str>) -> usize {
    let Some(value) = value else {
        return CUDA_WIDTH;
    };
    match value.trim().parse::<usize>() {
        Ok(rows) if (1..=crate::catalog::EXPLICIT_WIDTH_CEILING).contains(&rows) => rows,
        _ => {
            tracing::warn!(
                value,
                default = CUDA_WIDTH,
                "Ignoring {CUDA_WIDTH_ENV} outside 1..=32"
            );
            CUDA_WIDTH
        }
    }
}
/// Prefill rows per span on CUDA (the CPU and Metal take
/// [`DEFAULT_PREFILL_SPAN_ROWS`]).
pub(crate) const CUDA_PREFILL_SPAN_ROWS: usize = 1_024;

fn overflow() -> Error {
    Error::ModelLoadError("Fish Audio Drama 3 serving capacity overflows".into())
}

fn sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |total, value| total.checked_add(*value))
        .ok_or_else(overflow)
}

fn to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| overflow())
}

/// Whether the operator opted Metal into its second row.
pub(crate) fn native_batching_from_env() -> Result<bool> {
    match crate::env::value(NATIVE_BATCHING_ENV) {
        Some(value) => izwi_env::parse_bool(NATIVE_BATCHING_ENV, &value)
            .map_err(|error| Error::ConfigError(error.to_string())),
        None => Ok(false),
    }
}

/// The widest decode batch the family serves on `backend`.
pub(crate) fn decode_width(backend: BackendKind, native_batching: bool) -> usize {
    if backend == BackendKind::Cuda {
        return cuda_width(crate::env::value(CUDA_WIDTH_ENV).as_deref());
    }
    let certified =
        crate::catalog::certified_decode_width(crate::catalog::ModelFamily::FishDrama3Tts, backend);
    if backend == BackendKind::Metal && native_batching {
        certified.max(2)
    } else {
        certified
    }
}

/// The backend's own prefill grid cell.
pub(crate) fn default_prefill_span_rows(backend: BackendKind) -> usize {
    match backend {
        BackendKind::Cpu | BackendKind::Metal => DEFAULT_PREFILL_SPAN_ROWS,
        BackendKind::Cuda => CUDA_PREFILL_SPAN_ROWS,
    }
}

/// The load's prefill grid cell on `backend`: `IZWI_FISH_DRAMA3_PREFILL_SPAN`
/// (which the load reads too, so admission and the model agree), else the
/// backend's.
pub(crate) fn prefill_span_rows(backend: BackendKind) -> Result<usize> {
    crate::models::architectures::fish_drama3::model::prefill_span_rows(
        std::env::var(crate::models::architectures::fish_drama3::model::PREFILL_SPAN_ENV)
            .ok()
            .as_deref(),
        default_prefill_span_rows(backend),
    )
}

/// The paged KV dtype a load on `backend` plans before its weights exist:
/// the backend's (§6), or F32 under `IZWI_FISH_DRAMA3_DTYPE=f32`. A loaded
/// model reports its own ([`ResolvedFishDrama3Capacity::resolve_loaded`]),
/// which its self-check may have moved to F32.
pub(crate) fn planned_kv_dtype(backend: BackendKind) -> Result<DType> {
    use crate::models::architectures::fish_drama3::precision::{activation_dtype, DTYPE_ENV};
    use crate::models::architectures::fish_drama3::FishDrama3Residency;
    let residency = match backend {
        BackendKind::Cpu => FishDrama3Residency::Cpu,
        BackendKind::Metal => FishDrama3Residency::Metal,
        BackendKind::Cuda => FishDrama3Residency::Cuda,
    };
    let activations = activation_dtype(residency, std::env::var(DTYPE_ENV).ok().as_deref())?;
    Ok(
        if residency == FishDrama3Residency::Metal && activations == DType::F32 {
            DType::F32
        } else {
            kv_dtype(backend)
        },
    )
}

/// What one retained row holds and uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FishDrama3RowCost {
    /// Recurrent and conv state, the same for every row.
    pub(crate) tensor_state_bytes: u64,
    /// Both paged domains over the planned positions, plus one.
    pub(crate) kv_bytes: u64,
    pub(crate) span_workspace_bytes: u64,
    pub(crate) decode_workspace_bytes: u64,
    /// The codec stream's current and rollback histories.
    pub(crate) codec_history_bytes: u64,
    /// One codec quantum's decode workspace.
    pub(crate) codec_workspace_bytes: u64,
    /// One codec quantum's PCM, reserved before the codec runs.
    pub(crate) output_credit_bytes: u64,
}

impl FishDrama3RowCost {
    /// A row planned for `prompt_rows` and `frames`, its KV in `kv`, its
    /// prefill in spans of `workspace.span_rows`.
    pub(crate) fn of(
        config: &FishDrama3Config,
        kv: DType,
        prompt_rows: usize,
        frames: usize,
        workspace: &FishDrama3Workspace,
    ) -> Result<Self> {
        let geometry = FishDrama3StateGeometry::of(config);
        let per_position =
            geometry.trunk.bytes_per_position(kv) + geometry.tower.bytes_per_position(kv);
        let positions = prompt_rows
            .checked_add(frames)
            .and_then(|positions| positions.checked_add(1))
            .ok_or_else(overflow)?;
        let codec = FishDrama3DacConfig::current();
        let quantum_samples = codec
            .samples_per_frame()?
            .checked_mul(FISH_DRAMA3_AUDIO_CHUNK_FRAMES)
            .ok_or_else(overflow)?;
        Ok(Self {
            tensor_state_bytes: to_u64(geometry.tensor_bytes())?,
            kv_bytes: to_u64(per_position.checked_mul(positions).ok_or_else(overflow)?)?,
            span_workspace_bytes: workspace.span_bytes,
            decode_workspace_bytes: workspace.decode_row_bytes,
            codec_history_bytes: codec
                .streaming_history_bound_bytes()?
                .checked_mul(2)
                .ok_or_else(overflow)?,
            codec_workspace_bytes:
                crate::models::architectures::fish_drama3::codec::streaming_decode_workspace_bytes(
                    FISH_DRAMA3_AUDIO_CHUNK_FRAMES,
                )?,
            output_credit_bytes: to_u64(
                quantum_samples
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(overflow)?,
            )?,
        })
    }

    pub(crate) fn total(&self) -> Result<u64> {
        sum(&[
            self.tensor_state_bytes,
            self.kv_bytes,
            self.span_workspace_bytes,
            self.decode_workspace_bytes,
            self.codec_history_bytes,
            self.codec_workspace_bytes,
            self.output_credit_bytes,
        ])
    }
}

/// The operator's row ceilings: the engine's retained and staged rows, and
/// an explicit batch width (`None` when automatic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FishDrama3RowCeilings {
    pub(crate) retained: usize,
    pub(crate) staged: usize,
    pub(crate) batch: Option<usize>,
}

/// Bytes two references' preparations may hold at once: the longest
/// reference (60 s) at the highest accepted sample rate, bounded by the
/// codec's own maximum.
pub(crate) fn reference_preparation_bytes() -> Result<u64> {
    use crate::models::architectures::fish_drama3::codec::{
        maximum_preparation_workspace_bytes, preparation_workspace_bytes,
    };
    let rate = FishDrama3DacConfig::MAX_REFERENCE_SAMPLE_RATE;
    let samples = usize::try_from(u64::from(MAX_REFERENCE_SECONDS) * u64::from(rate))
        .map_err(|_| overflow())?;
    preparation_workspace_bytes(samples, rate)?
        .min(maximum_preparation_workspace_bytes()?)
        .checked_mul(2)
        .ok_or_else(overflow)
}

/// The rows a Drama 3 load seals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedFishDrama3Capacity {
    /// Retained, staged and decode rows (one number: every retained row may
    /// decode in the same quantum).
    pub(crate) rows: usize,
    /// The family's width on the backend.
    pub(crate) width: usize,
    pub(crate) kv_dtype: DType,
    /// Prompt rows and frames a row is planned for.
    pub(crate) prompt_rows: usize,
    pub(crate) frames: usize,
    pub(crate) workspace: FishDrama3Workspace,
    pub(crate) row: FishDrama3RowCost,
    pub(crate) row_bytes: u64,
    /// Two references' preparation workspace, held outside the rows.
    pub(crate) preparation_bytes: u64,
    pub(crate) safety_reserve_bytes: u64,
    pub(crate) planning_headroom_bytes: Option<u64>,
}

impl ResolvedFishDrama3Capacity {
    /// The rows `headroom` holds on `backend` for a context of at most
    /// `context_tokens` positions per row (the operator's explicit context,
    /// or [`STATE_CONTEXT_TOKENS`]). Unknown headroom seals one row; known
    /// headroom too small for a row seals one as well, and the managed
    /// state fit then refuses or shrinks it.
    pub(crate) fn resolve(
        config: &FishDrama3Config,
        backend: BackendKind,
        context_tokens: Option<usize>,
        ceilings: FishDrama3RowCeilings,
        headroom: Option<u64>,
        safety_reserve_bytes: u64,
        native_batching: bool,
    ) -> Result<Self> {
        Self::resolve_with(
            config,
            backend,
            planned_kv_dtype(backend)?,
            prefill_span_rows(backend)?,
            context_tokens,
            ceilings,
            headroom,
            safety_reserve_bytes,
            native_batching,
        )
    }

    /// [`Self::resolve`] for a loaded `model`: its own paged KV dtype (F32
    /// after a breached Metal self-check) and prefill cell.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resolve_loaded(
        model: &crate::models::architectures::fish_drama3::FishDrama3Model,
        backend: BackendKind,
        context_tokens: Option<usize>,
        ceilings: FishDrama3RowCeilings,
        headroom: Option<u64>,
        safety_reserve_bytes: u64,
        native_batching: bool,
    ) -> Result<Self> {
        Self::resolve_with(
            model.config(),
            backend,
            model.kv_dtype(),
            model.prefill_span_rows(),
            context_tokens,
            ceilings,
            headroom,
            safety_reserve_bytes,
            native_batching,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_with(
        config: &FishDrama3Config,
        backend: BackendKind,
        kv: DType,
        span_rows: usize,
        context_tokens: Option<usize>,
        ceilings: FishDrama3RowCeilings,
        headroom: Option<u64>,
        safety_reserve_bytes: u64,
        native_batching: bool,
    ) -> Result<Self> {
        let positions = context_tokens
            .unwrap_or(STATE_CONTEXT_TOKENS)
            .clamp(1, STATE_CONTEXT_TOKENS);
        let prompt_rows = positions.min(MAX_PROMPT_ROWS);
        let frames = positions - prompt_rows;
        let workspace = FishDrama3Workspace::of(config, span_rows)?;
        let row = FishDrama3RowCost::of(config, kv, prompt_rows, frames, &workspace)?;
        let row_bytes = row.total()?;
        let preparation_bytes = reference_preparation_bytes()?;
        let width = decode_width(backend, native_batching);
        let fitting = match headroom {
            None => 1,
            Some(headroom) => {
                let usable = headroom
                    .saturating_sub(safety_reserve_bytes)
                    .saturating_sub(preparation_bytes);
                usize::try_from(usable / row_bytes.max(1)).unwrap_or(usize::MAX)
            }
        };
        let ceiling = |limit: usize| if limit == 0 { usize::MAX } else { limit };
        let rows = fitting
            .min(width)
            .min(ceiling(ceilings.retained))
            .min(ceiling(ceilings.staged))
            .min(ceilings.batch.map_or(usize::MAX, ceiling))
            .max(1);
        Ok(Self {
            rows,
            width,
            kv_dtype: kv,
            prompt_rows,
            frames,
            workspace,
            row,
            row_bytes,
            preparation_bytes,
            safety_reserve_bytes,
            planning_headroom_bytes: headroom,
        })
    }

    /// The positions a retained row reaches: its prompt and frames.
    pub(crate) fn positions(&self) -> usize {
        self.prompt_rows + self.frames
    }

    /// The same rows in Fish S2's sealing interface, which the adapter's
    /// stage descriptors read: one prefill span per prefill work quantum,
    /// the span's workspace per prompt row, decode and codec rows, and each
    /// row's model and codec workspace.
    pub(crate) fn serving(&self) -> ResolvedFishServingCapacity {
        ResolvedFishServingCapacity {
            ar_rows: self.rows,
            prefill_tokens: self.workspace.span_rows as u64,
            prefill_workspace_per_token: self.workspace.span_bytes_per_row(),
            codec_rows: self.rows,
            active_rows: self.rows,
            staged_rows: self.rows,
            ar_workspace_per_row: self
                .workspace
                .span_bytes
                .max(self.workspace.decode_row_bytes),
            codec_workspace_per_row: self.row.codec_workspace_bytes,
            planning_headroom_bytes: self.planning_headroom_bytes,
            native_batching: self.rows > 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_width_takes_one_to_thirty_two_rows_else_eight() {
        assert_eq!(cuda_width(None), 8);
        assert_eq!(cuda_width(Some("16")), 16);
        assert_eq!(cuda_width(Some(" 32 ")), 32);
        for invalid in ["0", "33", "wide"] {
            assert_eq!(cuda_width(Some(invalid)), 8, "{invalid}");
        }
    }
    use crate::models::architectures::fish_drama3::config::FishDrama3ConfigPolicy;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    fn published() -> FishDrama3Config {
        FishDrama3Config::from_json(
            include_str!("../../tests/fixtures/fish_drama3/config.json"),
            FishDrama3ConfigPolicy::Published,
        )
        .unwrap()
    }

    const NO_CEILINGS: FishDrama3RowCeilings = FishDrama3RowCeilings {
        retained: 0,
        staged: 0,
        batch: None,
    };

    fn resolve(
        backend: BackendKind,
        headroom: Option<u64>,
        native: bool,
    ) -> ResolvedFishDrama3Capacity {
        ResolvedFishDrama3Capacity::resolve(
            &published(),
            backend,
            None,
            NO_CEILINGS,
            headroom,
            GIB,
            native,
        )
        .unwrap()
    }

    /// Each term from its own arithmetic, for the published model.
    #[test]
    fn a_published_row_adds_up_its_terms() {
        let config = published();
        let workspace = FishDrama3Workspace::of(&config, 256).unwrap();
        let row = FishDrama3RowCost::of(&config, DType::F32, 8_192, 4_096, &workspace).unwrap();
        // 30 × ([32, 128, 128] + [3, 8192]) F32.
        assert_eq!(row.tensor_state_bytes, 65_863_680);
        // (10 + 13) layers × K/V × 2 heads × 256 × 4 B = 92 KiB a position,
        // over 8,192 + 4,096 + 1 positions.
        assert_eq!(row.kv_bytes, 94_208 * 12_289);
        assert_eq!(row.span_workspace_bytes, workspace.span_bytes);
        assert_eq!(row.decode_workspace_bytes, workspace.decode_row_bytes);
        let codec = FishDrama3DacConfig::current();
        assert_eq!(
            row.codec_history_bytes,
            2 * codec.streaming_history_bound_bytes().unwrap()
        );
        // One 16-frame quantum of 2,048-sample F32 frames.
        assert_eq!(row.output_credit_bytes, 16 * 2_048 * 4);
        assert_eq!(
            row.total().unwrap(),
            65_863_680
                + 94_208 * 12_289
                + workspace.span_bytes
                + workspace.decode_row_bytes
                + row.codec_history_bytes
                + row.codec_workspace_bytes
                + 131_072
        );
        // The worst admitted row in F32, about 3.1 GiB: §4.2's 1.17 GiB of
        // KV and state, 128 MiB of span workspace, and the codec quantum's
        // conservative workspace (1.8 GiB for 16 frames over a 127-frame
        // window, the envelope Fish S2 charges per codec row too).
        assert_eq!(row.span_workspace_bytes, 134_021_120);
        assert!(row.codec_workspace_bytes > GIB + 800 * MIB);
        let total = row.total().unwrap();
        assert!(
            total > 3 * GIB && total < 3 * GIB + 200 * MIB,
            "{total} {row:?}"
        );

        // 16-bit KV halves the KV term and nothing else.
        let half = FishDrama3RowCost::of(&config, DType::F16, 8_192, 4_096, &workspace).unwrap();
        assert_eq!(half.kv_bytes * 2, row.kv_bytes);
        assert_eq!(half.kv_bytes, 47_104 * 12_289);
        assert_eq!(
            row.total().unwrap() - half.total().unwrap(),
            row.kv_bytes - half.kv_bytes
        );
    }

    #[test]
    fn rows_divide_the_headroom_left_after_the_reserve_and_two_preparations() {
        let config = published();
        let ceilings = FishDrama3RowCeilings {
            retained: 0,
            staged: 0,
            batch: None,
        };
        let one = ResolvedFishDrama3Capacity::resolve(
            &config,
            BackendKind::Cuda,
            None,
            ceilings,
            Some(0),
            0,
            false,
        )
        .unwrap();
        let (row, preparation, reserve) = (one.row_bytes, one.preparation_bytes, 3 * GIB);
        assert_eq!(preparation, reference_preparation_bytes().unwrap());
        assert!(preparation > 0);
        for fitting in 1..=8u64 {
            for slack in [0, row - 1] {
                let headroom = reserve + preparation + fitting * row + slack;
                let capacity = ResolvedFishDrama3Capacity::resolve(
                    &config,
                    BackendKind::Cuda,
                    None,
                    ceilings,
                    Some(headroom),
                    reserve,
                    false,
                )
                .unwrap();
                assert_eq!(capacity.rows as u64, fitting, "headroom {headroom}");
                assert_eq!(capacity.row_bytes, row);
            }
        }
        // Too little for one row still seals one; the state fit decides.
        assert_eq!(one.rows, 1);
        // Unknown memory seals one.
        assert_eq!(resolve(BackendKind::Cuda, None, false).rows, 1);
    }

    #[test]
    fn rows_are_capped_by_the_family_width_and_the_operator() {
        let plenty = Some(1024 * GIB);
        assert_eq!(resolve(BackendKind::Cpu, plenty, false).rows, 1);
        assert_eq!(resolve(BackendKind::Cpu, plenty, true).rows, 1);
        assert_eq!(resolve(BackendKind::Metal, plenty, false).rows, 1);
        assert_eq!(resolve(BackendKind::Metal, plenty, true).rows, 2);
        let cuda = resolve(BackendKind::Cuda, plenty, false);
        assert_eq!((cuda.rows, cuda.width), (8, 8));
        assert_eq!(resolve(BackendKind::Cuda, plenty, true).rows, 8);
        for ceilings in [
            FishDrama3RowCeilings {
                retained: 3,
                staged: 0,
                batch: None,
            },
            FishDrama3RowCeilings {
                retained: 0,
                staged: 3,
                batch: None,
            },
            FishDrama3RowCeilings {
                retained: 0,
                staged: 0,
                batch: Some(3),
            },
        ] {
            let capped = ResolvedFishDrama3Capacity::resolve(
                &published(),
                BackendKind::Cuda,
                None,
                ceilings,
                plenty,
                GIB,
                false,
            )
            .unwrap();
            assert_eq!(capped.rows, 3, "{ceilings:?}");
        }
    }

    #[test]
    fn sixteen_bit_kv_fits_more_rows_than_f32() {
        // CUDA prices BF16 KV; the same row in F32 doubles only its KV.
        let cuda = resolve(BackendKind::Cuda, Some(0), false);
        assert_eq!(cuda.kv_dtype, DType::BF16);
        let f32_row = FishDrama3RowCost::of(
            &published(),
            DType::F32,
            cuda.prompt_rows,
            cuda.frames,
            &cuda.workspace,
        )
        .unwrap();
        assert_eq!(f32_row.kv_bytes, 2 * cuda.row.kv_bytes);
        // Headroom for eight 16-bit rows holds fewer F32 rows.
        let headroom = GIB + cuda.preparation_bytes + 8 * cuda.row_bytes;
        let rows_at = |row_bytes: u64| (headroom - GIB - cuda.preparation_bytes) / row_bytes;
        assert_eq!(rows_at(cuda.row_bytes), 8);
        assert!(rows_at(f32_row.total().unwrap()) < 8);
        assert_eq!(
            resolve(BackendKind::Cpu, Some(0), false).kv_dtype,
            DType::F32
        );
        assert_eq!(
            resolve(BackendKind::Metal, Some(0), false).kv_dtype,
            DType::F16
        );
    }

    #[test]
    fn an_explicit_context_plans_shorter_rows() {
        let full = resolve(BackendKind::Cuda, None, false);
        assert_eq!((full.prompt_rows, full.frames), (8_192, 4_096));
        assert_eq!(full.positions(), STATE_CONTEXT_TOKENS);
        let short = ResolvedFishDrama3Capacity::resolve(
            &published(),
            BackendKind::Cuda,
            Some(2_000),
            NO_CEILINGS,
            None,
            GIB,
            false,
        )
        .unwrap();
        assert_eq!((short.prompt_rows, short.frames), (2_000, 0));
        assert!(short.row_bytes < full.row_bytes);
        // A larger explicit context is still the worst admitted row.
        let long = ResolvedFishDrama3Capacity::resolve(
            &published(),
            BackendKind::Cuda,
            Some(1 << 20),
            NO_CEILINGS,
            None,
            GIB,
            false,
        )
        .unwrap();
        assert_eq!(long.row_bytes, full.row_bytes);
    }

    #[test]
    fn the_serving_capacity_carries_the_rows_and_workspaces() {
        let capacity = resolve(BackendKind::Cuda, Some(1024 * GIB), false);
        let serving = capacity.serving();
        assert_eq!(
            (
                serving.ar_rows,
                serving.active_rows,
                serving.staged_rows,
                serving.codec_rows
            ),
            (8, 8, 8, 8)
        );
        assert!(serving.native_batching);
        assert_eq!(serving.prefill_tokens, 1_024);
        assert!(
            serving.prefill_tokens * serving.prefill_workspace_per_token
                >= capacity.workspace.span_bytes
        );
        assert_eq!(serving.ar_workspace_per_row, capacity.workspace.span_bytes);
        assert_eq!(
            serving.codec_workspace_per_row,
            capacity.row.codec_workspace_bytes
        );
        let cpu = resolve(BackendKind::Cpu, Some(1024 * GIB), false).serving();
        assert_eq!((cpu.ar_rows, cpu.prefill_tokens), (1, 256));
        assert!(!cpu.native_batching);
    }

    #[test]
    fn the_metal_opt_in_reads_the_shared_boolean_vocabulary() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        for (value, expected) in [("1", true), ("on", true), ("false", false)] {
            std::env::set_var(NATIVE_BATCHING_ENV, value);
            assert_eq!(native_batching_from_env().unwrap(), expected, "{value}");
        }
        std::env::set_var(NATIVE_BATCHING_ENV, "sometimes");
        assert!(native_batching_from_env().is_err());
        std::env::remove_var(NATIVE_BATCHING_ENV);
        assert!(!native_batching_from_env().unwrap());
    }
}
