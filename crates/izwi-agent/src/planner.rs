use crate::errors::Result;
use crate::tools::has_any_word;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlanningMode {
    Off,
    #[default]
    Auto,
    On,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    pub mode: PlanningMode,
    pub steps: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum PlannerDecision {
    DirectRespond,
    PlanThenAct(PlanSummary),
}

#[derive(Debug, Clone)]
pub struct PlannerContext {
    pub user_text: String,
    pub planning_mode: PlanningMode,
    pub tools_enabled: bool,
    pub available_tool_names: Vec<String>,
}

pub trait Planner: Send + Sync {
    fn decide(&self, ctx: &PlannerContext) -> Result<PlannerDecision>;
}

#[derive(Debug, Clone, Default)]
pub struct SimplePlanner;

impl Planner for SimplePlanner {
    fn decide(&self, ctx: &PlannerContext) -> Result<PlannerDecision> {
        if ctx.planning_mode == PlanningMode::Off {
            return Ok(PlannerDecision::DirectRespond);
        }

        let lower = ctx.user_text.to_lowercase();
        let should_plan = match ctx.planning_mode {
            PlanningMode::On => true,
            PlanningMode::Off => false,
            // Whole words: "planet", "explanation" or "relearned" do not plan.
            PlanningMode::Auto => {
                has_any_word(
                    &lower,
                    &[
                        "research", "compare", "teach", "learn", "language", "lesson", "plan",
                        "planning",
                    ],
                ) || lower.starts_with("how do i")
                    || lower.starts_with("how can i")
            }
        };

        if !should_plan {
            return Ok(PlannerDecision::DirectRespond);
        }

        let mut steps = if has_any_word(&lower, &["teach", "learn", "language", "lesson"]) {
            vec![
                "Identify the user's learning goal and current level.".to_string(),
                "Teach in short steps with simple examples and checks.".to_string(),
                "End with a small practice task or recap.".to_string(),
            ]
        } else if has_any_word(&lower, &["research", "compare"]) {
            vec![
                "Clarify the topic and what the user wants compared or researched.".to_string(),
                "Gather facts from available tools or prior context.".to_string(),
                "Return a concise summary with clear next steps.".to_string(),
            ]
        } else {
            vec![
                "Break the request into a few concrete steps.".to_string(),
                "Work through the steps and keep the user updated.".to_string(),
                "Summarize the result and next actions.".to_string(),
            ]
        };

        if ctx.tools_enabled && !ctx.available_tool_names.is_empty() {
            steps.insert(
                1,
                "Use tools only when needed, then integrate results.".to_string(),
            );
        }

        Ok(PlannerDecision::PlanThenAct(PlanSummary {
            mode: ctx.planning_mode,
            steps,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_plans_on_whole_words_only() {
        let decide = |text: &str| {
            SimplePlanner
                .decide(&PlannerContext {
                    user_text: text.to_string(),
                    planning_mode: PlanningMode::Auto,
                    tools_enabled: false,
                    available_tool_names: Vec::new(),
                })
                .expect("decision")
        };
        for chat in [
            "Tell me about the planet Mars",
            "Thanks for the explanation",
        ] {
            assert!(
                matches!(decide(chat), PlannerDecision::DirectRespond),
                "{chat}"
            );
        }
        assert!(matches!(
            decide("Help me plan a trip"),
            PlannerDecision::PlanThenAct(_)
        ));
    }

    #[test]
    fn auto_mode_plans_for_teaching_requests() {
        let planner = SimplePlanner;
        let decision = planner
            .decide(&PlannerContext {
                user_text: "Teach me basic Spanish greetings".to_string(),
                planning_mode: PlanningMode::Auto,
                tools_enabled: true,
                available_tool_names: vec!["time".to_string()],
            })
            .expect("planner should succeed");

        match decision {
            PlannerDecision::PlanThenAct(plan) => {
                assert!(!plan.steps.is_empty());
            }
            PlannerDecision::DirectRespond => panic!("expected planning"),
        }
    }

    #[test]
    fn off_mode_disables_planning() {
        let planner = SimplePlanner;
        let decision = planner
            .decide(&PlannerContext {
                user_text: "research rust lifetimes".to_string(),
                planning_mode: PlanningMode::Off,
                tools_enabled: true,
                available_tool_names: vec!["time".to_string()],
            })
            .expect("planner should succeed");

        assert!(matches!(decision, PlannerDecision::DirectRespond));
    }
}
