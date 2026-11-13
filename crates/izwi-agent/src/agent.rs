use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One agent: who it is, what it may call and whom it may hand the
/// conversation to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Letters, digits, `_` or `-`: the id names the agent's transfer tool.
    pub id: String,
    pub name: String,
    pub system_prompt: String,
    pub default_model: String,
    /// The registry tools the agent may call; `None` is all of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// The agents this one may hand the conversation to. Each is offered
    /// to the model as the tool `transfer_to_<id>`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub handoffs: Vec<String>,
    /// When another agent should hand over to this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_description: Option<String>,
    /// What the agent must collect before the conversation moves on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<TaskSpec>,
}

/// A structured task: the agent gathers the fields of `parameters` from the
/// user and records them by calling `complete_<name>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSpec {
    /// Letters, digits or `_`.
    pub name: String,
    pub description: String,
    /// The JSON Schema of the task's result (an object).
    pub parameters: Value,
    /// The agent that takes over once the task is recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub then: Option<String>,
}

impl AgentDefinition {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            system_prompt: system_prompt.into(),
            default_model: default_model.into(),
            tools: None,
            handoffs: Vec::new(),
            handoff_description: None,
            task: None,
        }
    }

    pub(crate) fn transfer_tool_name(id: &str) -> String {
        format!("transfer_to_{id}")
    }
}

impl TaskSpec {
    pub(crate) fn tool_name(&self) -> String {
        format!("complete_{}", self.name)
    }

    /// The required fields `result` lacks.
    pub(crate) fn missing_fields(&self, result: &Value) -> Vec<String> {
        self.parameters["required"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|field| result.get(field).is_none_or(Value::is_null))
            .map(str::to_string)
            .collect()
    }
}

/// The agents of a session. The first is where a new session starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSet {
    agents: Vec<AgentDefinition>,
}

impl AgentSet {
    /// At most this many agents in a session.
    pub const MAX_AGENTS: usize = 16;

    pub fn single(agent: AgentDefinition) -> Self {
        Self {
            agents: vec![agent],
        }
    }

    /// The set of `agents`, when their ids are distinct and well formed and
    /// every handoff names one of them.
    pub fn new(agents: Vec<AgentDefinition>) -> Result<Self> {
        let invalid = |message: String| Err(AgentError::InvalidAgent(message));
        if agents.is_empty() {
            return invalid("a session needs at least one agent".to_string());
        }
        if agents.len() > Self::MAX_AGENTS {
            return invalid(format!(
                "a session has at most {} agents, not {}",
                Self::MAX_AGENTS,
                agents.len()
            ));
        }
        let is_id = |id: &str, dash: bool| {
            !id.is_empty()
                && id.len() <= 48
                && id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || (dash && c == '-'))
        };
        for (index, agent) in agents.iter().enumerate() {
            if !is_id(&agent.id, true) {
                return invalid(format!(
                    "agent id {:?} must be 1 to 48 letters, digits, `_` or `-`",
                    agent.id
                ));
            }
            if agents[..index].iter().any(|other| other.id == agent.id) {
                return invalid(format!("agent `{}` is defined twice", agent.id));
            }
            let targets = agent
                .handoffs
                .iter()
                .chain(agent.task.as_ref().and_then(|task| task.then.as_ref()));
            for target in targets {
                if !agents.iter().any(|other| &other.id == target) {
                    return invalid(format!(
                        "agent `{}` hands off to `{target}`, which is not defined",
                        agent.id
                    ));
                }
            }
            if agent.handoffs.contains(&agent.id) {
                return invalid(format!("agent `{}` hands off to itself", agent.id));
            }
            if let Some(task) = &agent.task {
                if !is_id(&task.name, false) {
                    return invalid(format!(
                        "task name {:?} must be 1 to 48 letters, digits or `_`",
                        task.name
                    ));
                }
                if !task.parameters.is_object() {
                    return invalid(format!("task `{}` needs an object schema", task.name));
                }
            }
        }
        Ok(Self { agents })
    }

    pub fn first(&self) -> &AgentDefinition {
        &self.agents[0]
    }

    pub fn get(&self, id: &str) -> Option<&AgentDefinition> {
        self.agents.iter().find(|agent| agent.id == id)
    }

    pub fn len(&self) -> usize {
        self.agents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn agent(id: &str) -> AgentDefinition {
        AgentDefinition::new(id, id, "", "model")
    }

    #[test]
    fn a_set_checks_ids_and_handoff_targets() {
        let mut front = agent("front-desk");
        front.handoffs = vec!["billing".to_string()];
        let set = AgentSet::new(vec![front.clone(), agent("billing")]).unwrap();
        assert_eq!(set.first().id, "front-desk");
        assert!(set.get("billing").is_some());

        assert!(AgentSet::new(Vec::new()).is_err());
        assert!(
            AgentSet::new(vec![front.clone()]).is_err(),
            "unknown target"
        );
        assert!(AgentSet::new(vec![agent("a b")]).is_err());
        assert!(AgentSet::new(vec![agent("a"), agent("a")]).is_err());
        let mut own = agent("a");
        own.handoffs = vec!["a".to_string()];
        assert!(AgentSet::new(vec![own]).is_err());
    }

    #[test]
    fn a_task_reports_the_required_fields_it_lacks() {
        let task = TaskSpec {
            name: "booking".to_string(),
            description: String::new(),
            parameters: json!({"type": "object", "required": ["name", "date"]}),
            then: None,
        };
        assert_eq!(task.tool_name(), "complete_booking");
        assert_eq!(
            task.missing_fields(&json!({"name": "Ada", "date": null})),
            ["date"]
        );
        assert!(task
            .missing_fields(&json!({"name": "Ada", "date": "Friday"}))
            .is_empty());
    }
}
