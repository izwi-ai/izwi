//! The state a served Inkling 2 sequence keeps in engine-owned storage.
//!
//! Two domains advance together, one step per token:
//!
//! * **attention**: paged keys and values, one physical layer per layer
//!   application (the loop's passes each have their own). Keys are stored
//!   as the model normalised them; the model composes its own attention over
//!   the gathered context, because its scores add a per-query position bias
//!   and a sink logit that the paged kernels do not implement. Windowed
//!   applications share the page table and narrow their own view.
//! * **convolution**: a ring of the last `kernel - 1` inputs of each of the
//!   four short convolutions of every layer application.

use candle_core::{DType, Tensor};

use super::config::{Inkling2Config, CONV_STATES_PER_APPLICATION};
use super::trunk::{Inkling2Context, Inkling2ConvHistory, Inkling2ConvState, Inkling2RowsContext};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kv::v2::{
    AttentionMask, AttentionPattern, BoundedShape, CheckpointPolicy, InferenceStateContract,
    KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec,
    PlacementPolicy, PositionSemantics, PrefixPolicy, RingStateDomainSpec, ShapeAxis,
    ShapeDimension, ShapeExtent, StateClock, StateComponentId, StateDType, StateDomainHeader,
    StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope, TensorComponentSpec,
    TensorRole, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

pub(crate) const INKLING2_ATTENTION_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const INKLING2_CONV_STATE_DOMAIN: StateDomainId = StateDomainId::new(2);
/// Keys and values of the sliding-window layer applications, when they have
/// a domain of their own.
pub(crate) const INKLING2_WINDOWED_STATE_DOMAIN: StateDomainId = StateDomainId::new(3);
const INKLING2_MAIN_STATE_GROUP: StateGroupId = StateGroupId::new(1);

fn state_error(message: impl std::fmt::Display) -> Error {
    Error::InferenceError(format!("Inkling 2 state: {message}"))
}

/// Paged KV storage per backend. Activations stay F32; keys and values are
/// cast on write and on gather. Metal and CUDA halve the bytes.
pub(crate) fn kv_dtype(backend: BackendKind) -> StateDType {
    match backend {
        BackendKind::Cpu => StateDType::F32,
        BackendKind::Metal => StateDType::F16,
        BackendKind::Cuda => StateDType::Bf16,
    }
}

/// The ring component of convolution `slot` (key, value, attention, mlp) of
/// layer application `application`.
pub(crate) fn conv_component(application: usize, slot: usize) -> Result<StateComponentId> {
    u32::try_from(application * CONV_STATES_PER_APPLICATION + slot + 1)
        .map(StateComponentId::new)
        .map_err(|_| state_error("convolution component index exceeds u32"))
}

/// Channels of each convolution of an application, in component order.
fn conv_widths(config: &Inkling2Config, layer: usize) -> [usize; CONV_STATES_PER_APPLICATION] {
    let kv = config.heads(layer).kv_width();
    [kv, kv, config.hidden, config.hidden]
}

fn retained_header(id: StateDomainId) -> StateDomainHeader {
    StateDomainHeader {
        id,
        scope: StateScope::Retained,
        clock: StateClock::DecoderTokens,
        placement: PlacementPolicy::BackendLocalWithHostOffload,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::Transactional,
    }
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::ModelLoadError(format!("Inkling 2 {what} exceeds u32")))
}

/// The drafter's layers as the caches hold them: each keeps one key and one
/// value per committed position (made from the main model's states there),
/// as a full-attention paged layer of the main domain after the layer
/// applications, so the engine accounts, shares and releases them with the
/// rest of the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DrafterLayers {
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
}

/// The scheduler-owned state contract of a chat sequence.
///
/// With `share_prefixes` the keys and values are one domain that publishes
/// its committed pages, and the convolution ring a snapshot at each page, so
/// a prompt that begins as an earlier one did attaches that work instead of
/// repeating it. Every layer application then keeps every page.
///
/// Without it, the sliding-window applications (most of them) have a domain
/// of their own. The engine releases the pages of a domain whose layers all
/// look back the same number of tokens once no query can reach them, so a
/// long conversation holds whole-history keys and values only for the
/// full-attention applications.
///
/// The two do not combine: the engine shares snapshot prefixes only for a
/// contract with one paged domain, and frees window pages only for a domain
/// with one window.
pub(crate) fn managed_contract(
    config: &Inkling2Config,
    kv_dtype: StateDType,
    share_prefixes: bool,
    drafter: Option<DrafterLayers>,
) -> Result<InferenceStateContract> {
    let applications = config.applications();
    let split = !share_prefixes && has_both_kinds(config);
    let all_layers = applications
        .iter()
        .enumerate()
        .map(|(index, application)| {
            let heads = config.heads(application.layer);
            Ok(PagedAttentionLayerSpec {
                // One physical layer per application: the loop's passes do
                // not share keys.
                model_layer: to_u32(index, "layer application")?,
                query_heads: to_u32(heads.heads, "query-head count")?,
                kv_heads: to_u32(heads.kv_heads, "KV-head count")?,
                key_head_dim: to_u32(heads.head_dim, "head dimension")?,
                value_head_dim: to_u32(heads.head_dim, "head dimension")?,
                pattern: match heads.window {
                    Some(window) => AttentionPattern::SlidingWindow {
                        window_tokens: to_u32(window, "sliding window")?,
                    },
                    None => AttentionPattern::Full,
                },
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Raw,
                attention_logit_softcap: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let drafter_layers = drafter
        .map(|drafter| {
            (0..drafter.layers)
                .map(|layer| {
                    Ok(PagedAttentionLayerSpec {
                        model_layer: to_u32(applications.len() + layer, "drafter layer")?,
                        query_heads: to_u32(drafter.heads, "drafter head count")?,
                        kv_heads: to_u32(drafter.kv_heads, "drafter KV-head count")?,
                        key_head_dim: to_u32(drafter.head_dim, "drafter head dimension")?,
                        value_head_dim: to_u32(drafter.head_dim, "drafter head dimension")?,
                        pattern: AttentionPattern::Full,
                        mask: AttentionMask::Causal,
                        key_encoding: KeyEncoding::Raw,
                        attention_logit_softcap: None,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let preferred_tokens = to_u32(default_kv_page_size(), "page size")?;
    let mut attention_header = retained_header(INKLING2_ATTENTION_STATE_DOMAIN);
    let mut convolution_header = retained_header(INKLING2_CONV_STATE_DOMAIN);
    if share_prefixes {
        attention_header.prefix = PrefixPolicy::CommittedPages {
            positions: PositionSemantics::Absolute,
        };
        convolution_header.prefix = PrefixPolicy::CommittedSnapshots {
            interval_steps: u64::from(preferred_tokens),
        };
    }
    let paged = |header, layers| {
        StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
            header,
            layers,
            page_size: PageSizeConstraint {
                min_tokens: 1,
                preferred_tokens,
                max_tokens: preferred_tokens.max(256),
                multiple_of: 1,
            },
            accepted_dtypes: vec![kv_dtype],
        })
    };
    let (mut full_layers, windowed_layers): (Vec<_>, Vec<_>) = if split {
        all_layers
            .into_iter()
            .partition(|layer| layer.pattern == AttentionPattern::Full)
    } else {
        (all_layers, Vec::new())
    };
    full_layers.extend(drafter_layers);
    let attention = paged(attention_header, full_layers);
    let windowed = split.then(|| {
        paged(
            retained_header(INKLING2_WINDOWED_STATE_DOMAIN),
            windowed_layers,
        )
    });
    let mut components = Vec::with_capacity(applications.len() * CONV_STATES_PER_APPLICATION);
    for (index, application) in applications.iter().enumerate() {
        for (slot, width) in conv_widths(config, application.layer)
            .into_iter()
            .enumerate()
        {
            components.push(TensorComponentSpec {
                id: conv_component(index, slot)?,
                role: TensorRole::ConvolutionState,
                shape: BoundedShape {
                    dimensions: vec![
                        ShapeDimension {
                            axis: ShapeAxis::Batch,
                            extent: ShapeExtent::Fixed { value: 1 },
                        },
                        ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed {
                                value: width as u64,
                            },
                        },
                    ],
                },
                accepted_dtypes: vec![StateDType::F32],
            });
        }
    }
    let convolution = StateDomainSpec::Ring(RingStateDomainSpec {
        header: convolution_header,
        components_per_step: components,
        capacity_steps: (config.conv_kernel - 1) as u64,
    });
    let mut domains = vec![attention, convolution];
    let mut group = vec![INKLING2_ATTENTION_STATE_DOMAIN, INKLING2_CONV_STATE_DOMAIN];
    if let Some(windowed) = windowed {
        domains.push(windowed);
        group.push(INKLING2_WINDOWED_STATE_DOMAIN);
    }
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains,
        groups: vec![StateGroupSpec {
            id: INKLING2_MAIN_STATE_GROUP,
            domains: group,
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

/// `(model_layer, kv_heads, key_head_dim, value_head_dim)` of every paged
/// layer, for the cache's geometry check.
pub(crate) fn paged_geometry(
    config: &Inkling2Config,
    drafter: Option<DrafterLayers>,
) -> Vec<(u32, usize, usize, usize)> {
    let applications = config.applications();
    let mut geometry = applications
        .iter()
        .enumerate()
        .map(|(index, application)| {
            let heads = config.heads(application.layer);
            (index as u32, heads.kv_heads, heads.head_dim, heads.head_dim)
        })
        .collect::<Vec<_>>();
    if let Some(drafter) = drafter {
        geometry.extend((0..drafter.layers).map(|layer| {
            (
                (applications.len() + layer) as u32,
                drafter.kv_heads,
                drafter.head_dim,
                drafter.head_dim,
            )
        }));
    }
    geometry
}

/// The widest span the fused attention kernel serves; a wider prefill span
/// is composed from gathered keys and values. On an M1 Pro at the published
/// geometry (24 heads over 8, F16 keys, 48 calls queued) the kernel is 5 to
/// 25 times faster for one query and 2.5 to 6 times for 8 or 16, but at 64
/// and 256 queries it is up to a third slower than the composed form, which
/// does a span's queries in one matrix product (`physical.rs`,
/// `paged_bias_sink_timing_fused_versus_eager_on_metal`).
pub(crate) const FUSED_ATTENTION_MAX_SPAN: usize = 16;

/// Whether the model has sliding-window and full-attention applications.
fn has_both_kinds(config: &Inkling2Config) -> bool {
    let windowed = config
        .applications()
        .iter()
        .filter(|application| config.heads(application.layer).window.is_some())
        .count();
    windowed > 0 && windowed < config.application_count()
}

/// Where a layer application's keys and values live: in the windowed cache
/// or the main one, and at which of that cache's layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Route {
    windowed: bool,
    ordinal: usize,
}

/// A sequence's keys and values: one cache for every layer application, or
/// (see [`managed_contract`]) the full-attention applications in `main` and
/// the sliding-window ones in `windowed`. Both always hold the same number
/// of tokens' positions; the windowed one may have let old pages go.
pub(crate) struct Inkling2Caches {
    main: PhysicalPagedKvCache,
    windowed: Option<PhysicalPagedKvCache>,
}

impl From<PhysicalPagedKvCache> for Inkling2Caches {
    fn from(main: PhysicalPagedKvCache) -> Self {
        Self {
            main,
            windowed: None,
        }
    }
}

/// One prepared append over [`Inkling2Caches`].
pub(crate) struct Inkling2Prepared {
    main: PreparedPhysicalPagedStep,
    windowed: Option<PreparedPhysicalPagedStep>,
}

impl Inkling2Caches {
    pub(crate) fn new(
        main: PhysicalPagedKvCache,
        windowed: Option<PhysicalPagedKvCache>,
    ) -> Result<Self> {
        if let Some(windowed) = &windowed {
            if windowed.context_len() != main.context_len() {
                return Err(state_error(format!(
                    "the windowed cache holds {} tokens, the main one {}",
                    windowed.context_len(),
                    main.context_len()
                )));
            }
        }
        Ok(Self { main, windowed })
    }

    pub(crate) fn main(&self) -> &PhysicalPagedKvCache {
        &self.main
    }

    pub(crate) fn windowed(&self) -> Option<&PhysicalPagedKvCache> {
        self.windowed.as_ref()
    }

    pub(crate) fn context_len(&self) -> usize {
        self.main.context_len()
    }

    /// Whether `next` is the same sequence's reservation for its next
    /// quantum: the same arenas, layout and length.
    pub(crate) fn continued_by(&self, next: &Self) -> bool {
        let same = |a: &PhysicalPagedKvCache, b: &PhysicalPagedKvCache| {
            a.arena().id() == b.arena().id() && a.context_len() == b.context_len()
        };
        same(&self.main, &next.main)
            && match (&self.windowed, &next.windowed) {
                (None, None) => true,
                (Some(a), Some(b)) => same(a, b),
                _ => false,
            }
    }

    /// Checks that the caches cover exactly the model's layer applications
    /// in this layout.
    pub(crate) fn validate(
        &self,
        config: &Inkling2Config,
        drafter: Option<DrafterLayers>,
    ) -> Result<()> {
        let geometry = paged_geometry(config, drafter);
        let applications = config.applications();
        match &self.windowed {
            None => self.main.validate_sparse_model_layers(&geometry),
            Some(windowed) => {
                // The drafter's layers come after the applications and are
                // full-attention ones.
                let (sliding, full): (Vec<_>, Vec<_>) =
                    geometry.into_iter().partition(|(layer, ..)| {
                        applications
                            .get(*layer as usize)
                            .is_some_and(|application| {
                                config.heads(application.layer).window.is_some()
                            })
                    });
                self.main.validate_sparse_model_layers(&full)?;
                windowed.validate_sparse_model_layers(&sliding)
            }
        }
    }

    /// Per layer application: whether it lives in the windowed cache, and at
    /// which of its cache's layers.
    pub(crate) fn application_routes(&self, config: &Inkling2Config) -> Vec<(bool, usize)> {
        self.routes(config)
            .into_iter()
            .map(|route| (route.windowed, route.ordinal))
            .collect()
    }

    /// Commits one token at `position` whose writes a graphed step made and
    /// the sealed completions vouch for.
    pub(crate) fn commit_shared(
        &mut self,
        position: usize,
        main: std::sync::Arc<crate::backends::kv::KvWriteBatchCompletion>,
        windowed: Option<std::sync::Arc<crate::backends::kv::KvWriteBatchCompletion>>,
    ) -> Result<()> {
        self.main.commit_shared_completion(position, 1, main)?;
        match (&mut self.windowed, windowed) {
            (Some(cache), Some(completion)) => {
                cache.commit_shared_completion(position, 1, completion)
            }
            (None, None) => Ok(()),
            _ => Err(state_error(
                "a graphed step's completions do not match the caches",
            )),
        }
    }

    /// The main cache's layer that holds drafter layer `layer`: after every
    /// layer application the main cache holds.
    pub(crate) fn drafter_ordinal(&self, config: &Inkling2Config, layer: usize) -> usize {
        self.routes(config)
            .iter()
            .filter(|route| !route.windowed)
            .count()
            + layer
    }

    /// Commits the first `accepted` tokens of a prepared append and drops
    /// the rest (a verified draft that was only partly kept).
    pub(crate) fn commit_prefix(
        &mut self,
        prepared: Inkling2Prepared,
        accepted: usize,
    ) -> Result<()> {
        self.main.commit_prepared_prefix(prepared.main, accepted)?;
        match (&mut self.windowed, prepared.windowed) {
            (Some(cache), Some(prepared)) => cache.commit_prepared_prefix(prepared, accepted),
            (None, None) => Ok(()),
            _ => Err(state_error("a prepared append does not match its caches")),
        }
    }

    /// Writes the drafter's keys and values of a span (`[tokens, kv_heads,
    /// head_dim]`, F32) for drafter layer `layer`.
    pub(crate) fn write_drafter_span(
        &self,
        config: &Inkling2Config,
        prepared: &mut Inkling2Prepared,
        layer: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<()> {
        let storage = self.main.arena().config().dtype;
        self.main.write_span(
            self.drafter_ordinal(config, layer),
            &mut prepared.main,
            &keys.to_dtype(storage)?,
            &values.to_dtype(storage)?,
        )
    }

    fn routes(&self, config: &Inkling2Config) -> Vec<Route> {
        let (mut main, mut sliding) = (0, 0);
        config
            .applications()
            .iter()
            .map(|application| {
                let windowed =
                    self.windowed.is_some() && config.heads(application.layer).window.is_some();
                let counter = if windowed { &mut sliding } else { &mut main };
                let ordinal = *counter;
                *counter += 1;
                Route { windowed, ordinal }
            })
            .collect()
    }

    pub(crate) fn prepare_append(&self, start: usize, tokens: usize) -> Result<Inkling2Prepared> {
        Ok(Inkling2Prepared {
            main: self.main.prepare_append(start, tokens)?,
            windowed: self
                .windowed
                .as_ref()
                .map(|cache| cache.prepare_append(start, tokens))
                .transpose()?,
        })
    }

    pub(crate) fn commit(&mut self, prepared: Inkling2Prepared) -> Result<()> {
        self.main.commit_prepared(prepared.main)?;
        match (&mut self.windowed, prepared.windowed) {
            (Some(cache), Some(prepared)) => cache.commit_prepared(prepared),
            (None, None) => Ok(()),
            _ => Err(state_error("a prepared append does not match its caches")),
        }
    }

    /// Aborts both appends; the first failure is returned.
    pub(crate) fn abort(&mut self, prepared: Inkling2Prepared) -> Result<()> {
        let main = self.main.abort_prepared(prepared.main);
        let windowed = match (&mut self.windowed, prepared.windowed) {
            (Some(cache), Some(prepared)) => cache.abort_prepared(prepared),
            _ => Ok(()),
        };
        main.and(windowed)
    }

    pub(crate) fn take_completed_writes(
        &mut self,
    ) -> Vec<std::sync::Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        let mut writes = self.main.take_completed_writes();
        if let Some(windowed) = &mut self.windowed {
            writes.extend(windowed.take_completed_writes());
        }
        writes
    }

    /// The context a span's layer applications write to and read from.
    /// With `fused`, attention runs in the arena's fused kernel; otherwise
    /// the trunk composes it from gathered keys and values.
    pub(crate) fn context<'a>(
        &'a self,
        config: &Inkling2Config,
        prepared: &'a mut Inkling2Prepared,
        fused: bool,
    ) -> Inkling2PagedContext<'a> {
        Inkling2PagedContext {
            routes: self.routes(config),
            caches: self,
            prepared,
            fused,
        }
    }
}

/// The keys and values of a batched decode step: one token of each row's
/// sequence, every row's caches in one arena and one layout.
pub(crate) struct Inkling2RowsPagedContext<'a> {
    rows: Vec<(&'a Inkling2Caches, &'a mut Inkling2Prepared)>,
    routes: Vec<Route>,
}

impl<'a> Inkling2RowsPagedContext<'a> {
    pub(crate) fn new(
        config: &Inkling2Config,
        rows: Vec<(&'a Inkling2Caches, &'a mut Inkling2Prepared)>,
    ) -> Result<Self> {
        let Some((first, _)) = rows.first() else {
            return Err(state_error("a batch of rows is empty"));
        };
        let split = first.windowed.is_some();
        // The cache checks each step is one token.
        if rows
            .iter()
            .any(|(caches, _)| caches.windowed.is_some() != split)
        {
            return Err(state_error("the rows of a batch differ in cache layout"));
        }
        let routes = first.routes(config);
        Ok(Self { rows, routes })
    }
}

impl Inkling2RowsContext for Inkling2RowsPagedContext<'_> {
    fn attend_rows(
        &mut self,
        application: usize,
        keys: &Tensor,
        values: &Tensor,
        queries: &Tensor,
        profiles: &Tensor,
        sinks: &Tensor,
        scale: f32,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let route = *self
            .routes
            .get(application)
            .ok_or_else(|| state_error(format!("no layer application {application}")))?;
        let mut targets = Vec::with_capacity(self.rows.len());
        for (caches, prepared) in self.rows.iter_mut() {
            let (cache, step) = match (route.windowed, &caches.windowed) {
                (true, Some(cache)) => (
                    cache,
                    prepared
                        .windowed
                        .as_mut()
                        .ok_or_else(|| state_error("the windowed append was not prepared"))?,
                ),
                _ => (&caches.main, &mut prepared.main),
            };
            targets.push((cache, step));
        }
        let storage = targets[0].0.arena().config().dtype;
        let (keys, values) = (keys.to_dtype(storage)?, values.to_dtype(storage)?);
        let by_row =
            |tensor: &Tensor| -> Result<Tensor> { Ok(tensor.transpose(0, 1)?.contiguous()?) };
        PhysicalPagedKvCache::write_and_attend_biased_rows(
            &mut targets,
            route.ordinal,
            &keys,
            &values,
            &by_row(queries)?,
            &by_row(profiles)?,
            &by_row(&sinks.squeeze(2)?)?,
            scale,
            window,
        )
    }
}

/// Keys and values in the engine's paged caches, for one prepared append.
pub(crate) struct Inkling2PagedContext<'a> {
    caches: &'a Inkling2Caches,
    prepared: &'a mut Inkling2Prepared,
    routes: Vec<Route>,
    fused: bool,
}

impl Inkling2PagedContext<'_> {
    /// The cache, its layer and its prepared append for `application`,
    /// checked to continue at `first`.
    fn target(
        &mut self,
        application: usize,
        first: usize,
    ) -> Result<(&PhysicalPagedKvCache, usize, &mut PreparedPhysicalPagedStep)> {
        let route = *self
            .routes
            .get(application)
            .ok_or_else(|| state_error(format!("no layer application {application}")))?;
        let (cache, prepared) = match (route.windowed, &self.caches.windowed) {
            (true, Some(cache)) => (
                cache,
                self.prepared
                    .windowed
                    .as_mut()
                    .ok_or_else(|| state_error("the windowed append was not prepared"))?,
            ),
            _ => (&self.caches.main, &mut self.prepared.main),
        };
        if first != cache.context_len() {
            return Err(state_error(format!(
                "a span at {first} does not continue a cache of {} tokens",
                cache.context_len()
            )));
        }
        Ok((cache, route.ordinal, prepared))
    }
}

impl Inkling2Context for Inkling2PagedContext<'_> {
    fn append(
        &mut self,
        application: usize,
        keys: &Tensor,
        values: &Tensor,
        window: Option<usize>,
        first: usize,
    ) -> Result<(Tensor, Tensor, usize)> {
        let (cache, ordinal, prepared) = self.target(application, first)?;
        // The arena stores keys and values in its own dtype (F16 on Metal,
        // BF16 on CUDA); activations are F32.
        let storage = cache.arena().config().dtype;
        let (keys, values) = (keys.to_dtype(storage)?, values.to_dtype(storage)?);
        let (keys, values, oldest) =
            cache.write_and_gather(ordinal, prepared, &keys, &values, window)?;
        Ok((
            keys.to_dtype(DType::F32)?,
            values.to_dtype(DType::F32)?,
            oldest,
        ))
    }

    fn attend(
        &mut self,
        application: usize,
        keys: &Tensor,
        values: &Tensor,
        queries: &Tensor,
        profiles: &Tensor,
        sinks: &Tensor,
        scale: f32,
        window: Option<usize>,
        first: usize,
    ) -> Result<Option<Tensor>> {
        if !self.fused || queries.dim(1)? > FUSED_ATTENTION_MAX_SPAN {
            return Ok(None);
        }
        let (cache, ordinal, prepared) = self.target(application, first)?;
        let storage = cache.arena().config().dtype;
        let (keys, values) = (keys.to_dtype(storage)?, values.to_dtype(storage)?);
        // The kernel takes one row per token; the trunk holds heads first.
        let by_token =
            |tensor: &Tensor| -> Result<Tensor> { Ok(tensor.transpose(0, 1)?.contiguous()?) };
        Ok(Some(cache.write_and_attend_biased(
            ordinal,
            prepared,
            &keys,
            &values,
            &by_token(queries)?,
            &by_token(profiles)?,
            &by_token(&sinks.squeeze(2)?)?,
            scale,
            window,
        )?))
    }
}

/// The convolution histories of a served sequence, mirrored into the
/// engine's tensor-state arena at each committed step.
#[derive(Debug, Clone)]
pub(crate) struct Inkling2ConvRuntimeState {
    cursor: u64,
    /// `None` between a stage and the next restore.
    conv: Option<Inkling2ConvState>,
    /// The state before the first token.
    initial: Inkling2ConvState,
}

impl Inkling2ConvRuntimeState {
    pub(crate) fn new(initial: Inkling2ConvState) -> Self {
        Self {
            cursor: 0,
            conv: Some(initial.clone()),
            initial,
        }
    }

    pub(crate) fn cursor(&self) -> u64 {
        self.cursor
    }

    /// The live histories.
    pub(crate) fn conv(&self) -> Result<&Inkling2ConvState> {
        self.conv
            .as_ref()
            .ok_or_else(|| state_error("the convolution state was staged and not restored"))
    }

    /// Replaces the histories after a span of `steps` tokens.
    pub(crate) fn advance(&mut self, conv: Inkling2ConvState, steps: usize) {
        self.conv = Some(conv);
        self.cursor += steps as u64;
    }

    /// Reads the committed histories back from the arena.
    pub(crate) fn restore(
        &mut self,
        arena: &TensorStateArena,
        sequence: PhysicalStateSequenceId,
    ) -> Result<()> {
        let Some(snapshot) = arena.read(sequence, INKLING2_CONV_STATE_DOMAIN)? else {
            if self.cursor == 0 {
                self.conv = Some(self.initial.clone());
                return Ok(());
            }
            return Err(state_error("the retained convolution state disappeared"));
        };
        let expected = self.initial.histories.len() * CONV_STATES_PER_APPLICATION;
        if snapshot.components.len() != expected {
            return Err(state_error(
                "the retained convolution component count changed",
            ));
        }
        if snapshot.cursor == 0 {
            self.conv = Some(self.initial.clone());
            self.cursor = 0;
            return Ok(());
        }
        let mut histories = Vec::with_capacity(self.initial.histories.len());
        for (application, initial) in self.initial.histories.iter().enumerate() {
            let mut tensors = Vec::with_capacity(CONV_STATES_PER_APPLICATION);
            for (slot, shape) in initial.tensors().into_iter().enumerate() {
                let value = &snapshot.components[application * CONV_STATES_PER_APPLICATION + slot];
                if value.component != conv_component(application, slot)? {
                    return Err(state_error(
                        "a retained convolution component changed identity",
                    ));
                }
                let (steps, width) = shape.dims2()?;
                match value.tensor.as_ref() {
                    Some(tensor)
                        if tensor.dims() == [steps, 1, width] && tensor.dtype() == DType::F32 =>
                    {
                        tensors.push(tensor.reshape((steps, width))?);
                    }
                    _ => {
                        return Err(state_error(
                            "a retained convolution tensor changed geometry",
                        ))
                    }
                }
            }
            let tensors: [Tensor; CONV_STATES_PER_APPLICATION] = tensors
                .try_into()
                .map_err(|_| state_error("a convolution history set is incomplete"))?;
            histories.push(Inkling2ConvHistory::from_tensors(tensors));
        }
        self.conv = Some(Inkling2ConvState { histories });
        self.cursor = snapshot.cursor;
        Ok(())
    }

    /// Stages the histories for commit at `target_cursor` (the KV cursor).
    pub(crate) fn stage(
        &mut self,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<()> {
        if self.cursor != target_cursor {
            return Err(state_error(
                "the convolution cursor does not match the target KV cursor",
            ));
        }
        let expected_cursor = arena
            .read_transaction_base(transaction, INKLING2_CONV_STATE_DOMAIN)?
            .map(|snapshot| snapshot.cursor)
            .unwrap_or(0);
        let conv = self.conv()?;
        let mut values = Vec::with_capacity(conv.histories.len() * CONV_STATES_PER_APPLICATION);
        for (application, history) in conv.histories.iter().enumerate() {
            for (slot, tensor) in history.tensors().into_iter().enumerate() {
                let (steps, width) = tensor.dims2()?;
                values.push(StateComponentValue {
                    component: conv_component(application, slot)?,
                    tensor: Some(tensor.reshape((steps, 1, width))?),
                });
            }
        }
        arena.stage_replace(
            transaction,
            INKLING2_CONV_STATE_DOMAIN,
            expected_cursor,
            target_cursor,
            values,
        )?;
        self.conv = None;
        Ok(())
    }
}

/// The drafter's view of a sequence's committed keys and values, read from
/// the main cache's drafter layers.
pub(crate) struct Inkling2DrafterCacheContext<'a> {
    pub(crate) caches: &'a Inkling2Caches,
    pub(crate) config: &'a Inkling2Config,
}

impl super::drafter::DrafterContext for Inkling2DrafterCacheContext<'_> {
    fn committed(&self) -> usize {
        self.caches.context_len()
    }

    fn span(&self, layer: usize, first: usize) -> Result<Option<(Tensor, Tensor)>> {
        if first >= self.caches.context_len() {
            return Ok(None);
        }
        let (keys, values) = self
            .caches
            .main
            .gather_committed(self.caches.drafter_ordinal(self.config, layer), first)?;
        Ok(Some((
            keys.to_dtype(DType::F32)?,
            values.to_dtype(DType::F32)?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::{published, tiny};
    use super::*;

    fn paged(domain: &StateDomainSpec) -> &PagedAttentionDomainSpec {
        match domain {
            StateDomainSpec::PagedAttention(attention) => attention,
            _ => panic!("not a paged attention domain"),
        }
    }

    fn windowed_layers(attention: &PagedAttentionDomainSpec) -> usize {
        attention
            .layers
            .iter()
            .filter(|layer| matches!(layer.pattern, AttentionPattern::SlidingWindow { .. }))
            .count()
    }

    /// Sharing prefixes: one paged layer per application in one domain that
    /// publishes its pages, and a ring of four histories per application
    /// that publishes snapshots.
    #[test]
    fn the_shared_contract_has_one_paged_domain_and_four_rings_per_application() {
        for config in [tiny(), published()] {
            let contract = managed_contract(&config, StateDType::F32, true, None).unwrap();
            let applications = config.application_count();
            assert_eq!(contract.domains.len(), 2);
            let attention = paged(&contract.domains[0]);
            assert_eq!(attention.layers.len(), applications);
            assert!(matches!(
                attention.header.prefix,
                PrefixPolicy::CommittedPages { .. }
            ));
            let StateDomainSpec::Ring(ring) = &contract.domains[1] else {
                panic!("second domain is not a ring");
            };
            assert!(matches!(
                ring.header.prefix,
                PrefixPolicy::CommittedSnapshots { .. }
            ));
            assert_eq!(ring.components_per_step.len(), 4 * applications);
            assert_eq!(ring.capacity_steps, 3);
            if config.hidden == 2304 {
                // 48 applications: 40 windowed, 8 full.
                assert_eq!((applications, windowed_layers(attention)), (48, 40));
            }
            assert_eq!(paged_geometry(&config, None).len(), applications);
        }
    }

    /// Not sharing: the sliding-window applications have a domain of their
    /// own, all with one window, so the engine can release their pages.
    #[test]
    fn the_unshared_contract_gives_the_windowed_applications_their_own_domain() {
        for config in [tiny(), published()] {
            let contract = managed_contract(&config, StateDType::F32, false, None).unwrap();
            let applications = config.application_count();
            assert_eq!(contract.domains.len(), 3);
            let (full, windowed) = (paged(&contract.domains[0]), paged(&contract.domains[2]));
            assert_eq!(full.header.id, INKLING2_ATTENTION_STATE_DOMAIN);
            assert_eq!(windowed.header.id, INKLING2_WINDOWED_STATE_DOMAIN);
            assert_eq!(windowed_layers(full), 0);
            assert_eq!(windowed_layers(windowed), windowed.layers.len());
            assert_eq!(full.layers.len() + windowed.layers.len(), applications);
            assert!(windowed
                .layers
                .windows(2)
                .all(|pair| pair[0].pattern == pair[1].pattern));
            for domain in [full, windowed] {
                assert_eq!(domain.header.prefix, PrefixPolicy::Disabled);
                // Layer ids rise within a domain.
                assert!(domain
                    .layers
                    .windows(2)
                    .all(|pair| pair[0].model_layer < pair[1].model_layer));
            }
            assert_eq!(contract.groups.len(), 1);
            assert_eq!(contract.groups[0].domains.len(), 3);
            if config.hidden == 2304 {
                assert_eq!((full.layers.len(), windowed.layers.len()), (8, 40));
            }
        }
    }

    #[test]
    fn conv_components_are_distinct_and_one_based() {
        let mut seen = std::collections::BTreeSet::new();
        for application in 0..48 {
            for slot in 0..CONV_STATES_PER_APPLICATION {
                assert!(seen.insert(conv_component(application, slot).unwrap()));
            }
        }
        assert_eq!(seen.len(), 192);
        assert_eq!(conv_component(0, 0).unwrap(), StateComponentId::new(1));
    }
}
