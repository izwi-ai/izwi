//! The tool-calling format each chat family was trained on.

use crate::catalog::{ModelFamily, ModelVariant};

use super::lfm;

/// How a model family reads tool definitions and writes tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolDialect {
    /// Qwen3: `<tool_call>\n{"name": f, "arguments": {...}}\n</tool_call>`
    /// (the Hermes format). Results are `<tool_response>` blocks in a user
    /// turn.
    HermesJson,
    /// Qwen3.5/3.6/3.8: `<tool_call><function=f><parameter=k>v</parameter>
    /// </function></tool_call>`. A Hermes JSON body is accepted too. Results
    /// as for Qwen3.
    QwenXml,
    /// LFM3 and LFM3-Audio: the Qwen XML call shape, but calls sit right
    /// after the text one blank line apart, and results are native `tool`
    /// turns.
    Lfm3Xml,
    /// LFM2.5: `<|tool_call_start|>[f(k='v')]<|tool_call_end|>`, with native
    /// `tool` turns.
    LfmPythonic,
}

impl ToolDialect {
    /// The dialect of a chat model, `None` when its template has no tools
    /// (Gemma 3).
    pub fn of(variant: ModelVariant) -> Option<Self> {
        match variant.family() {
            ModelFamily::Qwen3Chat | ModelFamily::Qwen3MoeChat => Some(Self::HermesJson),
            ModelFamily::Qwen35Chat | ModelFamily::Qwen35MoeChat | ModelFamily::Qwen38Chat => {
                Some(Self::QwenXml)
            }
            ModelFamily::Lfm3Chat | ModelFamily::Lfm3Audio => Some(Self::Lfm3Xml),
            ModelFamily::Lfm2Chat => Some(Self::LfmPythonic),
            _ => None,
        }
    }

    /// The marker that opens a tool call in the model's output.
    pub fn call_opener(self) -> &'static str {
        match self {
            Self::HermesJson | Self::QwenXml | Self::Lfm3Xml => "<tool_call>",
            Self::LfmPythonic => lfm::TOOL_CALL_START,
        }
    }

    /// The marker that closes a tool call (LFM2.5: the whole call list).
    pub fn call_closer(self) -> &'static str {
        match self {
            Self::HermesJson | Self::QwenXml | Self::Lfm3Xml => "</tool_call>",
            Self::LfmPythonic => lfm::TOOL_CALL_END,
        }
    }

    /// Whether the family's template has a native `tool` turn. The Qwen
    /// templates read results as `<tool_response>` blocks in a user turn.
    pub fn has_tool_role(self) -> bool {
        matches!(self, Self::Lfm3Xml | Self::LfmPythonic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_chat_family_with_a_tool_template_has_a_dialect() {
        assert_eq!(
            ToolDialect::of(ModelVariant::Qwen314BGguf),
            Some(ToolDialect::HermesJson)
        );
        assert_eq!(
            ToolDialect::of(ModelVariant::Qwen359BGguf),
            Some(ToolDialect::QwenXml)
        );
        assert_eq!(
            ToolDialect::of(ModelVariant::Qwen3827BFp8),
            Some(ToolDialect::QwenXml)
        );
        assert_eq!(
            ToolDialect::of(ModelVariant::Lfm317BGguf),
            Some(ToolDialect::Lfm3Xml)
        );
        assert_eq!(
            ToolDialect::of(ModelVariant::Lfm2512BInstructGguf),
            Some(ToolDialect::LfmPythonic)
        );
        assert_eq!(ToolDialect::of(ModelVariant::Gemma34BIt), None);
    }
}
