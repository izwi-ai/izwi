use crate::agent::AgentDefinition;
use crate::errors::{AgentError, Result};
use crate::memory::{MemoryMessage, MemoryMessageMeta, MemoryMessageRole, MemoryStore};
use crate::model::{ModelBackend, ModelRequest};
use crate::planner::{PlanSummary, Planner, PlannerContext, PlannerDecision};
use crate::reasoning::strip_reasoning;
use crate::session::{AgentEvent, AgentSession, AgentTurnResult, ToolCallRecord, TurnInput};
use crate::tools::{ToolContext, ToolRegistry};

#[derive(Debug, Clone, Copy)]
pub struct AgentTurnOptions {
    pub max_output_tokens: usize,
    pub max_tool_calls: usize,
    /// How much of the thread's history the prompt carries.
    pub history: HistoryBudget,
    /// Whether the turn stores the assistant reply itself. A caller that
    /// learns later how much of the reply reached the user (voice: an
    /// interrupted reply was only partly heard) stores it with
    /// [`AgentEngine::commit_assistant_reply`] instead.
    pub persist_assistant_reply: bool,
}

impl Default for AgentTurnOptions {
    fn default() -> Self {
        Self {
            max_output_tokens: 1536,
            max_tool_calls: 1,
            history: HistoryBudget::default(),
            persist_assistant_reply: true,
        }
    }
}

/// The most conversation history a prompt carries: the newest messages that
/// fit both limits, starting at a user message. System messages at the head
/// of the history (memory context) are always kept. The token count is an
/// estimate (four characters a token); the model's own context check stays
/// authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryBudget {
    pub max_messages: usize,
    pub max_tokens: usize,
}

impl Default for HistoryBudget {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

impl HistoryBudget {
    pub const UNBOUNDED: Self = Self {
        max_messages: usize::MAX,
        max_tokens: usize::MAX,
    };

    /// The part of `history` this budget keeps.
    pub fn apply(self, mut history: Vec<MemoryMessage>) -> Vec<MemoryMessage> {
        let pinned = history
            .iter()
            .take_while(|message| message.role == MemoryMessageRole::System)
            .count();
        let conversation = history.split_off(pinned);
        let mut tokens = 0usize;
        let mut start = conversation.len();
        for (index, message) in conversation.iter().enumerate().rev() {
            tokens = tokens.saturating_add(estimate_tokens(&message.content));
            if conversation.len() - index > self.max_messages || tokens > self.max_tokens {
                break;
            }
            start = index;
        }
        // A window never opens on the reply to a user message it dropped.
        while conversation
            .get(start)
            .is_some_and(|message| message.role != MemoryMessageRole::User)
        {
            start += 1;
        }
        history.extend(conversation.into_iter().skip(start));
        history
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4).max(1)
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
            options
                .history
                .apply(memory.load_messages(&session.thread_id).await?)
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

        // One leading system message: chat templates such as Qwen3.5's reject
        // a system message anywhere else, so the memory context, the plan and
        // a tool result join the agent's prompt instead of following it.
        let mut system_parts = vec![agent.system_prompt.clone()];
        let mut conversation = Vec::with_capacity(history.len() + 1);
        for message in history {
            if message.role == MemoryMessageRole::System {
                system_parts.push(message.content);
            } else {
                conversation.push(message);
            }
        }
        if let Some(plan_summary) = &plan {
            system_parts.push(format!(
                "Planning summary for this request (do not expose as hidden reasoning): {}",
                plan_summary.steps.join(" | ")
            ));
        }
        if let Some(tool_note) = tool_context_note {
            system_parts.push(format!(
                "Use this verified tool result if relevant: {tool_note}"
            ));
        }
        system_parts.retain(|part| !part.trim().is_empty());

        let mut model_messages = Vec::with_capacity(conversation.len() + 2);
        model_messages.push(MemoryMessage {
            role: MemoryMessageRole::System,
            content: system_parts.join("\n\n"),
        });
        model_messages.extend(conversation);
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

        if options.persist_assistant_reply {
            self.commit_assistant_reply(
                memory,
                session,
                &output.text,
                MemoryMessageMeta {
                    model_id: Some(resolved_model_id.clone()),
                    tokens_generated: Some(output.tokens_generated),
                    generation_time_ms: Some(output.generation_time_ms),
                },
            )
            .await?;
        }

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

    /// Store an assistant reply in the session's thread, without its
    /// reasoning. An empty reply is not stored.
    pub async fn commit_assistant_reply<M: MemoryStore>(
        &self,
        memory: &M,
        session: &AgentSession,
        reply: &str,
        meta: MemoryMessageMeta,
    ) -> Result<()> {
        let reply = strip_reasoning(reply);
        if reply.is_empty() {
            return Ok(());
        }
        memory
            .append_message(
                &session.thread_id,
                MemoryMessageRole::Assistant,
                reply,
                meta,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentCapabilities;
    use crate::errors::Result;
    use crate::model::{ModelBackend, ModelOutput, ModelRequest};
    use crate::planner::{PlanningMode, SimplePlanner};
    use crate::session::AgentTurnResult;
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
    struct TestModel {
        /// Prefix every reply with a reasoning block.
        thinks: bool,
        prompts: Mutex<Vec<Vec<MemoryMessage>>>,
    }

    #[async_trait]
    impl ModelBackend for TestModel {
        async fn generate(&self, request: ModelRequest) -> Result<ModelOutput> {
            let prompt = request
                .messages
                .iter()
                .rev()
                .find(|m| m.role == MemoryMessageRole::User)
                .map(|m| m.content.clone())
                .unwrap_or_default();
            self.prompts.lock().expect("lock").push(request.messages);
            let thinking = if self.thinks {
                "<think>hmm</think>"
            } else {
                ""
            };
            Ok(ModelOutput {
                text: format!("{thinking}Echo: {prompt}"),
                tokens_generated: 5,
                generation_time_ms: 1.0,
            })
        }
    }

    fn message(role: MemoryMessageRole, content: &str) -> MemoryMessage {
        MemoryMessage {
            role,
            content: content.to_string(),
        }
    }

    fn test_agent() -> (AgentDefinition, AgentSession) {
        (
            AgentDefinition {
                id: "voice".to_string(),
                name: "Voice Agent".to_string(),
                system_prompt: "Be helpful".to_string(),
                default_model: "Qwen3.5-4B".to_string(),
                capabilities: AgentCapabilities {
                    tools: false,
                    ..AgentCapabilities::default()
                },
                planning_mode: PlanningMode::Auto,
            },
            AgentSession {
                id: "sess_1".to_string(),
                agent_id: "voice".to_string(),
                thread_id: "thread_1".to_string(),
                created_at: 0,
                updated_at: 0,
            },
        )
    }

    async fn turn(
        memory: &TestMemory,
        model: &TestModel,
        text: &str,
        options: AgentTurnOptions,
    ) -> AgentTurnResult {
        let (agent, session) = test_agent();
        AgentEngine
            .run_turn(
                &agent,
                &session,
                TurnInput {
                    text: text.to_string(),
                },
                None,
                memory,
                model,
                &SimplePlanner,
                &ToolRegistry::new(),
                options,
            )
            .await
            .expect("turn")
    }

    /// SPK-F13: the budget keeps the newest whole turns that fit, never
    /// opens on an orphaned reply, and always keeps the leading system
    /// context.
    #[test]
    fn history_budget_keeps_the_newest_turns_and_the_system_context() {
        use MemoryMessageRole::{Assistant, System, User};
        let history = vec![
            message(System, "memory context"),
            message(User, "u1"),
            message(Assistant, "a1"),
            message(User, "u2"),
            message(Assistant, "a2"),
            message(User, "u3"),
            message(Assistant, "a3"),
        ];
        let kept = |budget: HistoryBudget| -> Vec<String> {
            budget
                .apply(history.clone())
                .into_iter()
                .map(|message| message.content)
                .collect()
        };
        let by_messages = |max_messages| HistoryBudget {
            max_messages,
            max_tokens: usize::MAX,
        };
        assert_eq!(
            kept(by_messages(4)),
            ["memory context", "u2", "a2", "u3", "a3"]
        );
        // Three messages would open on a2, a reply without its question.
        assert_eq!(kept(by_messages(3)), ["memory context", "u3", "a3"]);
        assert_eq!(kept(by_messages(0)), ["memory context"]);
        // Each two-character message estimates at one token.
        let by_tokens = HistoryBudget {
            max_messages: usize::MAX,
            max_tokens: 2,
        };
        assert_eq!(kept(by_tokens), ["memory context", "u3", "a3"]);
        assert_eq!(kept(HistoryBudget::UNBOUNDED).len(), history.len());
    }

    /// SPK-F13: the prompt carries only the budgeted history, and the stored
    /// reply has no reasoning.
    #[tokio::test]
    async fn turns_bound_the_prompt_history_and_store_replies_without_reasoning() {
        let memory = TestMemory::default();
        let model = TestModel {
            thinks: true,
            ..TestModel::default()
        };
        let options = AgentTurnOptions {
            history: HistoryBudget {
                max_messages: 2,
                max_tokens: usize::MAX,
            },
            ..AgentTurnOptions::default()
        };
        for text in ["one", "two", "three"] {
            let result = turn(&memory, &model, text, options).await;
            assert!(result.assistant_text.starts_with("<think>"));
        }
        let stored = memory.messages.lock().expect("lock").clone();
        assert_eq!(stored.len(), 6);
        assert_eq!(stored[5].content, "Echo: three");
        // System prompt, the previous turn and the new user message.
        let prompts = model.prompts.lock().expect("lock");
        let last: Vec<&str> = prompts[2].iter().map(|m| m.content.as_str()).collect();
        assert_eq!(last, ["Be helpful", "two", "Echo: two", "three"]);
    }

    /// SPK-F13: a caller that defers the reply stores it later, once it
    /// knows how much of it the user heard.
    /// Chat templates such as Qwen3.5's accept a system message only first:
    /// memory, a plan and a tool result all join the one leading message.
    #[tokio::test]
    async fn the_prompt_has_one_leading_system_message() {
        let memory = TestMemory::default();
        memory.messages.lock().expect("lock").extend([
            message(MemoryMessageRole::System, "Remembered: likes tea"),
            message(MemoryMessageRole::User, "Hi"),
            message(MemoryMessageRole::Assistant, "Hello"),
        ]);
        let model = TestModel::default();
        let (mut agent, session) = test_agent();
        agent.capabilities.tools = true;
        let mut tools = ToolRegistry::new();
        tools.register(TimeTool);
        AgentEngine
            .run_turn(
                &agent,
                &session,
                TurnInput {
                    text: "Help me plan my time today".to_string(),
                },
                None,
                &memory,
                &model,
                &SimplePlanner,
                &tools,
                AgentTurnOptions::default(),
            )
            .await
            .expect("turn");

        let prompt = model.prompts.lock().expect("lock").pop().expect("prompt");
        let roles: Vec<_> = prompt.iter().map(|message| message.role).collect();
        assert_eq!(
            roles,
            [
                MemoryMessageRole::System,
                MemoryMessageRole::User,
                MemoryMessageRole::Assistant,
                MemoryMessageRole::User,
            ]
        );
        let system = &prompt[0].content;
        for part in [
            "Be helpful",
            "Remembered: likes tea",
            "Planning summary",
            "Tool `time`",
        ] {
            assert!(system.contains(part), "{part} missing from {system}");
        }
    }

    #[tokio::test]
    async fn a_deferred_reply_is_stored_only_when_committed() {
        let memory = TestMemory::default();
        let model = TestModel::default();
        let options = AgentTurnOptions {
            persist_assistant_reply: false,
            ..AgentTurnOptions::default()
        };
        let result = turn(&memory, &model, "hello", options).await;
        assert_eq!(memory.messages.lock().expect("lock").len(), 1);

        let (_, session) = test_agent();
        AgentEngine
            .commit_assistant_reply(
                &memory,
                &session,
                &format!("<think>x</think>{}", &result.assistant_text[..4]),
                MemoryMessageMeta::default(),
            )
            .await
            .expect("commit");
        let stored = memory.messages.lock().expect("lock").clone();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[1].role, MemoryMessageRole::Assistant);
        assert_eq!(stored[1].content, "Echo");
    }

    #[tokio::test]
    async fn engine_persists_user_and_assistant_messages() {
        let agent = AgentDefinition {
            id: "voice".to_string(),
            name: "Voice Agent".to_string(),
            system_prompt: "Be helpful".to_string(),
            default_model: "Qwen3.5-4B".to_string(),
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
        let model = TestModel::default();
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
