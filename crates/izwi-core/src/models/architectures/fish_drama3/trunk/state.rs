//! Per-request retained state of the trunk's DeltaNet layers (§7.2, R1).
//!
//! Each linear-attention layer keeps:
//! * its recurrent state, F32 `[value_heads, key_dim, value_dim]`
//!   (published: 30 × [32, 128, 128]);
//! * its conv history, the last `kernel - 1` pre-conv columns of
//!   `[channels]` F32 (published: 30 × 3 × 8192), as a ring: a step
//!   overwrites the oldest column and advances, a span replaces all three.
//!   Serialized, the history is `[kernel - 1, channels]`, oldest first, the
//!   order the shared GDN conv kernels read; it holds the same values as
//!   upstream's `[channels, kernel - 1]` conv state.
//!
//! Tensors are never mutated in place (a forward builds new ones), so a
//! clone is a snapshot. That gives the two checkpoint mechanisms later
//! slices need:
//! * a quantum checkpoint ([`FishDrama3TrunkState::begin_quantum`],
//!   `commit_quantum`, `abort_quantum`) rewinds the trunk's paged KV view
//!   and this state together inside one process;
//! * the engine's transactional [`TensorStateArena`]: staging hands the
//!   tensors to the arena and leaves this state drained until a restore
//!   reads the committed ones back, as qwen36moe's decode state does.
//!
//! The paged domains (the trunk's and the tower's attention, ids 1 and 2)
//! join these two tensor domains in one consistency group in the family's
//! state contract (Phase 2).

use candle_core::{DType, Device, Tensor};

use super::super::config::FishDrama3TrunkConfig;
use super::deltanet::DeltaGeometry;
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    BoundedShape, CheckpointPolicy, PlacementPolicy, PrefixPolicy, ShapeAxis, ShapeDimension,
    ShapeExtent, StateClock, StateComponentId, StateDType, StateDomainHeader, StateDomainId,
    StateDomainSpec, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};

/// The DeltaNet recurrent states, one component per linear layer.
pub(crate) const RECURRENT_STATE_DOMAIN: StateDomainId = StateDomainId::new(3);
/// The DeltaNet conv histories, one component per linear layer.
pub(crate) const CONVOLUTION_STATE_DOMAIN: StateDomainId = StateDomainId::new(4);

fn state_error(message: impl Into<String>) -> Error {
    Error::InferenceError(format!(
        "Fish Audio Drama 3 trunk state: {}",
        message.into()
    ))
}

/// The conv history of one layer.
#[derive(Clone)]
pub(crate) struct FishDrama3ConvRing {
    /// `kernel - 1` columns of `[channels]` F32; `slots[next]` is the oldest.
    slots: Vec<Tensor>,
    next: usize,
}

impl FishDrama3ConvRing {
    fn zeros(columns: usize, channels: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            slots: (0..columns)
                .map(|_| Tensor::zeros(channels, DType::F32, device))
                .collect::<candle_core::Result<_>>()?,
            next: 0,
        })
    }

    /// The ring over `history` `[columns, channels]`, oldest first.
    fn from_history(history: &Tensor, columns: usize, channels: usize) -> Result<Self> {
        if history.elem_count() != columns * channels || history.dtype() != DType::F32 {
            return Err(state_error("a conv history has the wrong size or dtype"));
        }
        let history = history.reshape((columns, channels))?;
        Ok(Self {
            slots: (0..columns)
                .map(|column| history.get(column))
                .collect::<candle_core::Result<_>>()?,
            next: 0,
        })
    }

    fn ordered(&self) -> impl Iterator<Item = &Tensor> {
        let len = self.slots.len();
        (0..len).map(move |offset| &self.slots[(self.next + offset) % len])
    }

    /// `[columns, channels]`, oldest first.
    pub(crate) fn history(&self) -> Result<Tensor> {
        Ok(Tensor::stack(&self.ordered().collect::<Vec<_>>(), 0)?)
    }

    /// The columns oldest first, flattened, on the host.
    pub(crate) fn ordered_host(&self) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for slot in self.ordered() {
            out.extend(slot.to_device(&Device::Cpu)?.to_vec1::<f32>()?);
        }
        Ok(out)
    }

    /// Appends `column` `[channels]` in place of the oldest column.
    pub(crate) fn push(&mut self, column: Tensor, channels: usize) -> Result<()> {
        if self.slots.is_empty() || column.dims() != [channels] || column.dtype() != DType::F32 {
            return Err(state_error("a conv column does not fit its ring"));
        }
        self.slots[self.next] = column;
        self.next = (self.next + 1) % self.slots.len();
        Ok(())
    }

    /// The three columns oldest first, as the fused conv kernel reads them.
    pub(crate) fn columns(&self) -> Result<[&Tensor; 3]> {
        let columns = self.ordered().collect::<Vec<_>>();
        columns
            .try_into()
            .map_err(|_| state_error("the fused conv reads a three-column history"))
    }

    /// Replaces the history with `history` `[columns, channels]` F32 on the
    /// device, oldest first (the fused conv kernel's next history).
    pub(crate) fn set_history(&mut self, history: &Tensor) -> Result<()> {
        let channels = self.slots.first().map_or(0, Tensor::elem_count);
        *self = Self::from_history(history, self.slots.len(), channels)?;
        Ok(())
    }

    /// Replaces the history with `columns` (`[columns, channels]` host values,
    /// oldest first).
    pub(crate) fn replace(
        &mut self,
        columns: &[f32],
        channels: usize,
        device: &Device,
    ) -> Result<()> {
        if columns.len() != self.slots.len() * channels {
            return Err(state_error("a conv history does not fit its ring"));
        }
        self.slots = columns
            .chunks_exact(channels)
            .map(|column| Tensor::from_slice(column, channels, device))
            .collect::<candle_core::Result<_>>()?;
        self.next = 0;
        Ok(())
    }
}

/// One linear-attention layer's state.
#[derive(Clone)]
pub(crate) struct FishDrama3LinearState {
    /// `[value_heads, key_dim, value_dim]` F32.
    recurrent: Tensor,
    pub(crate) conv: FishDrama3ConvRing,
}

impl FishDrama3LinearState {
    pub(crate) fn recurrent_host(&self) -> Result<Vec<f32>> {
        Ok(self
            .recurrent
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?)
    }

    /// `[value_heads, key_dim, value_dim]` F32.
    pub(crate) fn recurrent(&self) -> &Tensor {
        &self.recurrent
    }

    /// Installs the next recurrent state, a device tensor of the same shape.
    pub(crate) fn set_recurrent_tensor(&mut self, next: Tensor) -> Result<()> {
        if next.dims() != self.recurrent.dims() || next.dtype() != DType::F32 {
            return Err(state_error(format!(
                "a recurrent state of {:?} {:?} replaces one of {:?}",
                next.dims(),
                next.dtype(),
                self.recurrent.dims()
            )));
        }
        self.recurrent = next;
        Ok(())
    }

    pub(crate) fn set_recurrent(
        &mut self,
        values: Vec<f32>,
        geo: &DeltaGeometry,
        device: &Device,
    ) -> Result<()> {
        self.recurrent = Tensor::from_vec(
            values,
            (geo.value_heads, geo.key_dim, geo.value_dim),
            device,
        )?;
        Ok(())
    }
}

/// The shapes every linear layer's state takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StateGeometry {
    layers: usize,
    delta: DeltaGeometry,
    channels: usize,
    columns: usize,
}

impl StateGeometry {
    fn of(config: &FishDrama3TrunkConfig) -> Self {
        Self {
            layers: config.linear_attention_layers().len(),
            delta: DeltaGeometry::of(&config.linear),
            channels: config.linear.conv_channels(),
            columns: config.linear.conv_kernel - 1,
        }
    }

    fn recurrent_elements(&self) -> usize {
        self.delta.state_len()
    }

    fn conv_elements(&self) -> usize {
        self.channels * self.columns
    }
}

/// One request's trunk state: every linear layer's recurrent state and conv
/// history, and the position the next row takes (the rows consumed, equal
/// to the trunk's paged KV length).
#[derive(Clone)]
pub struct FishDrama3TrunkState {
    /// `None` while the tensors live in the arena (staged, not restored).
    layers: Option<Vec<FishDrama3LinearState>>,
    position: usize,
    geometry: StateGeometry,
    device: Device,
    tensor_sequence: Option<PhysicalStateSequenceId>,
}

impl std::fmt::Debug for FishDrama3TrunkState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3TrunkState")
            .field("position", &self.position)
            .field("drained", &self.layers.is_none())
            .field("layers", &self.geometry.layers)
            .finish()
    }
}

/// A rollback point of one request's trunk: its paged KV view and its
/// tensors, as they were when the quantum began.
pub struct FishDrama3TrunkCheckpoint {
    cache: PhysicalPagedKvCheckpoint,
    layers: Option<Vec<FishDrama3LinearState>>,
    position: usize,
}

impl FishDrama3TrunkState {
    /// The zero state at position 0 (upstream starts from zeros).
    pub fn new(config: &FishDrama3TrunkConfig, device: &Device) -> Result<Self> {
        let geometry = StateGeometry::of(config);
        let delta = geometry.delta;
        let layers = (0..geometry.layers)
            .map(|_| {
                Ok(FishDrama3LinearState {
                    recurrent: Tensor::zeros(
                        (delta.value_heads, delta.key_dim, delta.value_dim),
                        DType::F32,
                        device,
                    )?,
                    conv: FishDrama3ConvRing::zeros(geometry.columns, geometry.channels, device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers: Some(layers),
            position: 0,
            geometry,
            device: device.clone(),
            tensor_sequence: None,
        })
    }

    /// The next row's position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether the tensors were staged to the arena and not restored.
    pub fn is_drained(&self) -> bool {
        self.layers.is_none()
    }

    /// Bytes of one request's state (both domains).
    pub fn bytes(&self) -> usize {
        self.geometry.layers
            * (self.geometry.recurrent_elements() + self.geometry.conv_elements())
            * 4
    }

    pub(crate) fn layers(&self) -> Result<&[FishDrama3LinearState]> {
        self.layers
            .as_deref()
            .ok_or_else(|| state_error("the state was staged and not restored"))
    }

    /// Installs a forward's result: the layers' next state and the next
    /// position.
    pub(crate) fn install(&mut self, layers: Vec<FishDrama3LinearState>, position: usize) {
        self.layers = Some(layers);
        self.position = position;
    }

    fn check_cache(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        if cache.context_len() != self.position {
            return Err(state_error(format!(
                "the paged KV holds {} rows; the state is at {}",
                cache.context_len(),
                self.position
            )));
        }
        Ok(())
    }

    /// Begins a quantum: a rollback point for `cache` and this state.
    pub fn begin_quantum(&self, cache: &PhysicalPagedKvCache) -> Result<FishDrama3TrunkCheckpoint> {
        self.check_cache(cache)?;
        Ok(FishDrama3TrunkCheckpoint {
            cache: cache.logical_checkpoint(),
            layers: self.layers.clone(),
            position: self.position,
        })
    }

    /// Accepts a quantum: the clocks still agree and never moved backwards.
    pub fn commit_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
        checkpoint: &FishDrama3TrunkCheckpoint,
    ) -> Result<()> {
        self.check_cache(cache)?;
        if self.position < checkpoint.position {
            return Err(state_error("a quantum moved the state backwards"));
        }
        Ok(())
    }

    /// Rewinds `cache` and this state to `checkpoint`.
    pub fn abort_quantum(
        &mut self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: FishDrama3TrunkCheckpoint,
    ) -> Result<()> {
        cache.restore_logical_checkpoint(checkpoint.cache)?;
        self.layers = checkpoint.layers;
        self.position = checkpoint.position;
        Ok(())
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self
            .tensor_sequence
            .is_some_and(|current| current != sequence)
        {
            return Err(state_error("the tensor-state sequence changed"));
        }
        self.tensor_sequence = Some(sequence);
        Ok(())
    }

    /// Reads the committed tensors back from `arena`. A sequence with
    /// nothing committed keeps its current (fresh) state.
    pub(crate) fn restore_tensor_state(&mut self, arena: &TensorStateArena) -> Result<()> {
        let sequence = self
            .tensor_sequence
            .ok_or_else(|| state_error("no tensor-state sequence is bound"))?;
        let recurrent = arena.read(sequence, RECURRENT_STATE_DOMAIN)?;
        let convolution = arena.read(sequence, CONVOLUTION_STATE_DOMAIN)?;
        let (recurrent, convolution) = match (recurrent, convolution) {
            (None, None) if self.layers.is_some() => return Ok(()),
            (Some(recurrent), Some(convolution)) => (recurrent, convolution),
            _ => {
                return Err(state_error(
                    "the recurrent and conv domains are not both committed",
                ))
            }
        };
        if recurrent.cursor != convolution.cursor
            || recurrent.components.len() != self.geometry.layers
            || convolution.components.len() != self.geometry.layers
        {
            return Err(state_error(
                "the committed domains do not cover every layer at one cursor",
            ));
        }
        let g = self.geometry;
        let layers = recurrent
            .components
            .iter()
            .zip(convolution.components.iter())
            .map(|(recurrent, convolution)| {
                let (Some(state), Some(history)) = (&recurrent.tensor, &convolution.tensor) else {
                    return Err(state_error("a committed component is absent"));
                };
                if state.elem_count() != g.recurrent_elements() || state.dtype() != DType::F32 {
                    return Err(state_error(
                        "a committed recurrent state has the wrong size",
                    ));
                }
                Ok(FishDrama3LinearState {
                    recurrent: state.reshape((
                        g.delta.value_heads,
                        g.delta.key_dim,
                        g.delta.value_dim,
                    ))?,
                    conv: FishDrama3ConvRing::from_history(history, g.columns, g.channels)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.layers = Some(layers);
        self.position = usize::try_from(recurrent.cursor)
            .map_err(|_| state_error("the committed cursor exceeds usize"))?;
        Ok(())
    }

    /// Stages this state into `transaction` at the current position and
    /// leaves it drained: the arena then holds the only handles, so an
    /// engine abort cannot expose a half-published state.
    pub(crate) fn stage_tensor_state(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        let transaction = PhysicalStateTransactionId::new(transaction)?;
        let layers = self.layers()?;
        let target = self.position as u64;
        let base = |domain| -> Result<u64> {
            Ok(arena
                .read_transaction_base(transaction, domain)?
                .map(|snapshot| snapshot.cursor)
                .unwrap_or(0))
        };
        let mut recurrent = Vec::with_capacity(layers.len());
        let mut convolution = Vec::with_capacity(layers.len());
        for (index, layer) in layers.iter().enumerate() {
            let component = StateComponentId::new(
                u32::try_from(index + 1).map_err(|_| state_error("too many layers"))?,
            );
            recurrent.push(StateComponentValue {
                component,
                tensor: Some(layer.recurrent.clone()),
            });
            convolution.push(StateComponentValue {
                component,
                tensor: Some(layer.conv.history()?),
            });
        }
        arena.stage_replace(
            transaction,
            RECURRENT_STATE_DOMAIN,
            base(RECURRENT_STATE_DOMAIN)?,
            target,
            recurrent,
        )?;
        arena.stage_replace(
            transaction,
            CONVOLUTION_STATE_DOMAIN,
            base(CONVOLUTION_STATE_DOMAIN)?,
            target,
            convolution,
        )?;
        self.layers = None;
        Ok(())
    }

    /// Every layer's recurrent state, then every conv history (oldest
    /// first), flattened: for exact comparisons in tests.
    #[cfg(test)]
    pub(crate) fn values(&self) -> Vec<f32> {
        let layers = self.layers().unwrap();
        let mut out = Vec::new();
        for layer in layers {
            out.extend(layer.recurrent_host().unwrap());
        }
        for layer in layers {
            out.extend(layer.conv.ordered_host().unwrap());
        }
        out
    }
}

/// The recurrent and conv tensor domains of the family's state contract:
/// retained, transactional, clocked by decoder rows, never shared as
/// prefixes in v1.
pub(crate) fn tensor_state_domains(config: &FishDrama3TrunkConfig) -> Result<[StateDomainSpec; 2]> {
    let g = StateGeometry::of(config);
    let header = |id| StateDomainHeader {
        id,
        scope: crate::kv::v2::StateScope::Retained,
        clock: StateClock::DecoderTokens,
        placement: PlacementPolicy::BackendLocalWithHostOffload,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::Transactional,
    };
    let components = |elements: usize, role: TensorRole| {
        (0..g.layers)
            .map(|index| {
                Ok(TensorComponentSpec {
                    id: StateComponentId::new(
                        u32::try_from(index + 1).map_err(|_| state_error("too many layers"))?,
                    ),
                    role: role.clone(),
                    shape: BoundedShape {
                        dimensions: vec![ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed {
                                value: elements as u64,
                            },
                        }],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                })
            })
            .collect::<Result<Vec<_>>>()
    };
    Ok([
        StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(RECURRENT_STATE_DOMAIN),
            components: components(g.recurrent_elements(), TensorRole::RecurrentHidden)?,
        }),
        StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(CONVOLUTION_STATE_DOMAIN),
            components: components(g.conv_elements(), TensorRole::ConvolutionState)?,
        }),
    ])
}

#[cfg(test)]
mod tests {
    use super::super::super::config::tests::published;
    use super::*;

    #[test]
    fn the_published_state_is_62_8_mib_per_request() {
        let config = published();
        let g = StateGeometry::of(&config.trunk);
        assert_eq!(g.layers, 30);
        assert_eq!(g.recurrent_elements(), 32 * 128 * 128);
        assert_eq!(g.conv_elements(), 3 * 8192);
        // 30 x ([32, 128, 128] + [3, 8192]) F32.
        assert_eq!(
            30 * (g.recurrent_elements() + g.conv_elements()) * 4,
            65_863_680
        );
        let domains = tensor_state_domains(&config.trunk).unwrap();
        for domain in &domains {
            let StateDomainSpec::Tensor(spec) = domain else {
                panic!("tensor domain expected");
            };
            assert_eq!(spec.components.len(), 30);
            assert_eq!(spec.header.checkpoint, CheckpointPolicy::Transactional);
        }
    }

    #[test]
    fn the_ring_overwrites_its_oldest_column_and_serializes_oldest_first() {
        let device = Device::Cpu;
        let mut ring = FishDrama3ConvRing::zeros(3, 2, &device).unwrap();
        for value in 1..=5 {
            let column = Tensor::from_vec(vec![value as f32, -(value as f32)], 2, &device).unwrap();
            ring.push(column, 2).unwrap();
        }
        assert_eq!(
            ring.ordered_host().unwrap(),
            vec![3.0, -3.0, 4.0, -4.0, 5.0, -5.0]
        );
        let history = ring.history().unwrap();
        assert_eq!(history.dims(), [3, 2]);
        let back = FishDrama3ConvRing::from_history(&history, 3, 2).unwrap();
        assert_eq!(back.ordered_host().unwrap(), ring.ordered_host().unwrap());
        let mut replaced = ring.clone();
        replaced
            .replace(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, &device)
            .unwrap();
        assert_eq!(
            replaced.ordered_host().unwrap(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        assert!(ring
            .push(Tensor::zeros(3, DType::F32, &device).unwrap(), 2)
            .is_err());
    }
}
