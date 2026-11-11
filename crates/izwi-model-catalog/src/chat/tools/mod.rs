//! Tool calling for every chat family, in one place: which format a family
//! was trained on ([`ToolDialect`]), how tool definitions, past calls and
//! tool results are written into its prompt ([`render`]), how a request's
//! `tool_choice` applies ([`choice`]), and the incremental parser that turns
//! the model's markup back into calls ([`parse`]).

mod choice;
mod dialect;
mod json;
mod lfm;
mod parse;
mod render;
mod schema;

pub use choice::{ToolChoice, ToolPlan};
pub use dialect::ToolDialect;
pub use json::{is_deferred_tool, python_json, tool_function, tool_to_json, write_python_json};
pub use lfm::{TOOL_CALL_END as LFM_TOOL_CALL_END, TOOL_CALL_START as LFM_TOOL_CALL_START};
pub use parse::{
    parse_answer, ParsedOutput, ToolCall, ToolCallDelta, ToolCallDeltaFunction, ToolCallFunction,
    ToolCallParser,
};
pub use render::{
    append_assistant_tool_calls, assistant_content_with_reasoning, hermes_tools_prompt,
    push_chatml_tool_response, qwen_xml_tools_prompt, render_tool_calls,
};
pub use schema::ToolSchemas;
