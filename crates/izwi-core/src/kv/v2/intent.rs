//! Per-step state update descriptions that adapters hand to invocation arenas.

use serde::{Deserialize, Serialize};

use super::contract::{ShapeAxis, StateComponentId, StateDomainId};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DomainStepIntent {
    pub(crate) domain: StateDomainId,
    pub(crate) expected_cursor: u64,
    pub(crate) target_cursor: u64,
    pub(crate) update: StateUpdateKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum StateUpdateKind {
    PagedAppend {
        input_tokens: u64,
        read_visible_start: u64,
        commit_visible_start: u64,
    },
    StaticInitialize {
        source_identity: [u8; 32],
        components: Vec<ComponentShapeInstantiation>,
    },
    TensorReplace {
        components: Vec<ComponentShapeInstantiation>,
    },
    Append {
        steps: u64,
        components_per_step: Vec<ComponentShapeInstantiation>,
    },
    RingAdvance {
        steps: u64,
        components_per_step: Vec<ComponentShapeInstantiation>,
    },
    Reset,
    NoOp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ComponentShapeInstantiation {
    pub(crate) component: StateComponentId,
    pub(crate) dimensions: Vec<ShapeDimensionValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ShapeDimensionValue {
    pub(crate) axis: ShapeAxis,
    pub(crate) units: u64,
}
