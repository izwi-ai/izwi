use crate::agent::AgentDefinition;
use crate::errors::{AgentError, Result};
use crate::memory::{MemoryMessage, MemoryMessageMeta, MemoryMessageRole, MemoryStore};
use crate::model::{ModelBackend, ModelRequest};
use crate::planner::{PlanSummary, Planner, PlannerContext, PlannerDecision};
use crate::session::{AgentEvent, AgentSession, AgentTurnResult, ToolCallRecord, TurnInput};
use crate::tools::{ToolContext, ToolRegistry};

#[derive(Debug, Clone, Copy)]
pub struct AgentTurnOptions {
    pub max_output_tokens: usize,
    pub max_tool_calls: usize,
}

impl Default for AgentTurnOptions {
    fn default() -> Self {
        Self {
            max_output_tokens: 1536,
            max_tool_calls: 1,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgentEngine;

impl AgentEngine {
    #[allow(clippy::too_many_arguments)]
    pub async fn run_turn<M, B, P>(
        &self,
        agent: &AgentDefinition,
        session: &AgentSession,
        input: TurnInput,
        model_id: Option<String>,
        memory: &M,
        model: &B,
        planner: &P,
        tools: &ToolRegistry,
        options: AgentTurnOptions,
    ) -> Result<AgentTurnResult>
    where
        M: MemoryStore,
        B: ModelBackend,
        P: Planner,
    {
        let user_text = input.text.trim().to_string();
        if user_text.is_empty() {
            return Err(AgentError::InvalidInput(
                "Turn input text cannot be empty".to_string(),
            ));
        }

        let mut events = vec![AgentEvent::TurnStarted {
            session_id: session.id.clone(),
            thread_id: session.thread_id.clone(),
        }];

        let history = if agent.capabilities.memory {
            memory.load_messages(&session.thread_id).await?
        } else {
            Vec::new()
        };

        memory
            .append_message(
                &session.thread_id,
                MemoryMessageRole::User,
                user_text.clone(),
                MemoryMessageMeta::default(),
            )
            .await?;

        let planning_decision = if agent.capabilities.planning {
            planner.decide(&PlannerContext {
                user_text: user_text.clone(),
                planning_mode: agent.planning_mode,
                tools_enabled: agent.capabilities.tools,
                available_tool_names: tools.tool_names(),
            })?
        } else {
            PlannerDecision::DirectRespond
        };

        let mut plan: Option<PlanSummary> = None;
        if let PlannerDecision::PlanThenAct(plan_summary) = planning_decision {
            events.push(AgentEvent::PlanCreated {
                steps: plan_summary.steps.clone(),
            });
            plan = Some(plan_summary);
        }

        let mut tool_calls = Vec::new();
        let mut tool_context_note: Option<String> = None;
        if agent.capabilities.tools && options.max_tool_calls > 0 {
            if let Some(tool) = tools.find_auto_tool(&user_text) {
                events.push(AgentEvent::ToolCallStarted {
                    name: tool.name().to_string(),
                });
                let output = tool
                    .invoke(ToolContext {
                        session_id: session.id.clone(),
                        thread_id: session.thread_id.clone(),
                        user_text: user_text.clone(),
                    })
                    .await?;
                events.push(AgentEvent::ToolCallCompleted {
                    name: tool.name().to_string(),
                    output: output.text.clone(),
                });
                tool_context_note = Some(format!("Tool `{}` output: {}", tool.name(), output.text));
                tool_calls.push(ToolCallRecord {
                    name: tool.name().to_string(),
                    input_summary: "auto".to_string(),
                    output: output.text,
                });
            }
        }

        let mut model_messages = Vec::with_capacity(history.len() + 4);
        model_messages.push(MemoryMessage {
            role: MemoryMessageRole::System,
            content: agent.system_prompt.clone(),
        });
        model_messages.extend(history.into_iter());

        if let Some(plan_summary) = &plan {
            model_messages.push(MemoryMessage {
                role: MemoryMessageRole::System,
                content: format!(
                    "Planning summary for this request (do not expose as hidden reasoning): {}",
                    plan_summary.steps.join(" | ")
                ),
            });
        }

        if let Some(tool_note) = tool_context_note {
            model_messages.push(MemoryMessage {
                role: MemoryMessageRole::System,
                content: format!("Use this verified tool result if relevant: {tool_note}"),
            });
        }

        model_messages.push(MemoryMessage {
            role: MemoryMessageRole::User,
            content: user_text,
        });

        let resolved_model_id = model_id.unwrap_or_else(|| agent.default_model.clone());
        let output = model
            .generate(ModelRequest {
                model_id: resolved_model_id.clone(),
                messages: model_messages,
                max_output_tokens: options.max_output_tokens.max(1),
            })
            .await?;

        memory
            .append_message(
                &session.thread_id,
                MemoryMessageRole::Assistant,
                output.text.clone(),
                MemoryMessageMeta {
                    model_id: Some(resolved_model_id.clone()),
                    tokens_generated: Some(output.tokens_generated),
                    generation_time_ms: Some(output.generation_time_ms),
                },
            )
            .await?;

        events.push(AgentEvent::AssistantMessage {
            content: output.text.clone(),
        });
        events.push(AgentEvent::TurnCompleted {
            session_id: session.id.clone(),
            thread_id: session.thread_id.clone(),
        });

        Ok(AgentTurnResult {
            assistant_text: output.text,
            model_id: resolved_model_id,
            plan,
            tool_calls,
            events,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentCapabilities;
    use crate::errors::Result;
    use crate::model::{ModelBackend, ModelOutput, ModelRequest};
    use crate::planner::{PlanningMode, SimplePlanner};
    use crate::tools::{TimeTool, ToolRegistry};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct TestMemory {
        messages: Arc<Mutex<Vec<MemoryMessage>>>,
    }

    #[async_trait]
    impl MemoryStore for TestMemory {
        async fn load_messages(&self, _thread_id: &str) -> Result<Vec<MemoryMessage>> {
            Ok(self.messages.lock().expect("lock").clone())
        }

        async fn append_message(
            &self,
            _thread_id: &str,
            role: MemoryMessageRole,
            content: String,
            _meta: MemoryMessageMeta,
        ) -> Result<()> {
            self.messages
                .lock()
                .expect("lock")
                .push(MemoryMessage { role, content });
            Ok(())
        }
    }

    #[derive(Default)]
    struct TestModel;

    #[async_trait]
    impl ModelBackend for TestModel {
        async fn generate(&self, request: ModelRequest) -> Result<ModelOutput> {
            let prompt = request
                .messages
                .iter()
                .find(|m| m.role == MemoryMessageRole::User)
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(ModelOutput {
                text: format!("Echo: {prompt}"),
                tokens_generated: 5,
                generation_time_ms: 1.0,
            })
        }
    }

    #[tokio::test]
    async fn engine_persists_user_and_assistant_messages() {
        let agent = AgentDefinition {
            id: "voice".to_string(),
            name: "Voice Agent".to_string(),
            system_prompt: "Be helpful".to_string(),
            default_model: "Qwen3.5-0.8B".to_string(),
            capabilities: AgentCapabilities::default(),
            planning_mode: PlanningMode::Auto,
        };
        let session = AgentSession {
            id: "sess_1".to_string(),
            agent_id: "voice".to_string(),
            thread_id: "thread_1".to_string(),
            created_at: 0,
            updated_at: 0,
        };
        let memory = TestMemory::default();
        let model = TestModel;
        let planner = SimplePlanner;
        let mut tools = ToolRegistry::new();
        tools.register(TimeTool);

        let result = AgentEngine
            .run_turn(
                &agent,
                &session,
                TurnInput {
                    text: "What time is it?".to_string(),
                },
                None,
                &memory,
                &model,
                &planner,
                &tools,
                AgentTurnOptions::default(),
            )
            .await
            .expect("engine should succeed");

        assert!(result.assistant_text.starts_with("Echo:"));
        let stored = memory.messages.lock().expect("lock");
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, MemoryMessageRole::User);
        assert_eq!(stored[1].role, MemoryMessageRole::Assistant);
    }
}
