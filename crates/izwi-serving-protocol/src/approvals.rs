//! The shared worker-approval record and its on-disk text format.
//!
//! Gateway worker approvals are the control-plane contract between the
//! operator (and the supervisor's rollout coordinator) and the gateway: one
//! line per approved worker endpoint, pinned to an exact deployment
//! contract. The format is frozen at two forms — the standalone
//! compatibility form `URL|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION`
//! and the versioned fleet form
//! `v1|URL|NODE_ID|WORKER_ID|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION`
//! — so the supervisor can render views the gateway parses without either
//! side owning a private dialect of the other's format.
//!
//! This module is the single definition of that format: parsing, bounded
//! rendering, and the file-level text rules (one approval per line, `#`
//! comments, blank lines skipped, hard entry ceiling). Callers own the
//! filesystem: reading, atomic replacement, and locking stay with the
//! gateway loader and the supervisor's rollout coordinator respectively.

use std::fmt;
use std::str::FromStr;

use crate::identity::{DeploymentId, ModelAlias, ModelGeneration, NodeId, WorkerId};
use crate::types::TaskKind;

/// Hard ceiling on one rendered approval line.
pub const MAX_APPROVAL_LINE_BYTES: usize = 4 * 1024;
/// Hard ceiling on one approval endpoint field.
pub const MAX_APPROVAL_ENDPOINT_BYTES: usize = 2 * 1024;
/// Hard ceiling on approvals parsed from one shared file view.
pub const MAX_APPROVALS_FILE_ENTRIES: usize = 256;
/// Hard ceiling on the shared approvals file itself.
pub const MAX_APPROVALS_FILE_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayWorkerApproval {
    pub endpoint: String,
    pub identity: GatewayWorkerApprovalIdentity,
    pub task: TaskKind,
    pub public_model: ModelAlias,
    pub deployment_id: DeploymentId,
    pub model_generation: ModelGeneration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayWorkerApprovalIdentity {
    /// Compatibility form used only by the standalone/single-node profile.
    DiscoverFromAuthenticatedEndpoint,
    /// Versioned fleet form whose logical identities are operator-approved.
    V1 {
        node_id: NodeId,
        worker_id: WorkerId,
    },
}

impl GatewayWorkerApproval {
    pub fn pinned_identity(&self) -> Option<(&NodeId, &WorkerId)> {
        match &self.identity {
            GatewayWorkerApprovalIdentity::DiscoverFromAuthenticatedEndpoint => None,
            GatewayWorkerApprovalIdentity::V1 { node_id, worker_id } => Some((node_id, worker_id)),
        }
    }

    /// Renders the canonical line for this approval: the versioned fleet
    /// form when the identity is pinned, the standalone compatibility form
    /// otherwise. Parsing the rendered line yields an equal approval.
    pub fn render_line(&self) -> String {
        let task = match self.task {
            TaskKind::Chat => "chat",
            TaskKind::TextToSpeech => "text_to_speech",
            TaskKind::SpeechToText => "speech_to_text",
        };
        let tail = format!(
            "{}|{}|{}|{}",
            task,
            self.public_model,
            self.deployment_id,
            self.model_generation.get(),
        );
        match &self.identity {
            GatewayWorkerApprovalIdentity::DiscoverFromAuthenticatedEndpoint => {
                format!("{}|{}", self.endpoint, tail)
            }
            GatewayWorkerApprovalIdentity::V1 { node_id, worker_id } => {
                format!("v1|{}|{}|{}|{}", self.endpoint, node_id, worker_id, tail)
            }
        }
    }
}

impl fmt::Display for GatewayWorkerApproval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.render_line())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GatewayWorkerApprovalError {
    #[error("gateway worker approval exceeds its encoded size limit")]
    ApprovalTooLong,
    #[error(
        "gateway worker approval must use the standalone 5-field form or v1 8-field fleet form"
    )]
    InvalidApprovalSyntax,
    #[error("gateway worker approval uses an unsupported version")]
    UnsupportedApprovalVersion,
    #[error("gateway worker approval endpoint exceeds its size limit")]
    EndpointTooLong,
    #[error("gateway worker approval task must be chat, text_to_speech, or speech_to_text")]
    UnknownTask,
    #[error("gateway worker approval has an invalid public model alias")]
    InvalidPublicModel,
    #[error("gateway worker approval has an invalid node identifier")]
    InvalidNodeId,
    #[error("gateway worker approval has an invalid worker identifier")]
    InvalidWorkerId,
    #[error("gateway worker approval has an invalid deployment identifier")]
    InvalidDeploymentId,
    #[error("gateway worker approval model generation must be a non-zero integer")]
    InvalidGeneration,
}

impl FromStr for GatewayWorkerApproval {
    type Err = GatewayWorkerApprovalError;

    /// Parses the standalone compatibility form
    /// `URL|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION`, or the fleet form
    /// `v1|URL|NODE_ID|WORKER_ID|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION`.
    ///
    /// The endpoint is subsequently validated by `WorkerClient`; this parser
    /// only applies retention bounds and parses the statically pinned route.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() > MAX_APPROVAL_LINE_BYTES {
            return Err(GatewayWorkerApprovalError::ApprovalTooLong);
        }
        let fields = value.split('|').map(str::trim).collect::<Vec<_>>();
        if fields.iter().any(|field| field.is_empty()) {
            return Err(GatewayWorkerApprovalError::InvalidApprovalSyntax);
        }
        let (endpoint, identity, task, public_model, deployment_id, generation) = match fields
            .as_slice()
        {
            ["v1", endpoint, node_id, worker_id, task, public_model, deployment_id, generation] => {
                (
                    *endpoint,
                    GatewayWorkerApprovalIdentity::V1 {
                        node_id: NodeId::new(*node_id)
                            .map_err(|_| GatewayWorkerApprovalError::InvalidNodeId)?,
                        worker_id: WorkerId::new(*worker_id)
                            .map_err(|_| GatewayWorkerApprovalError::InvalidWorkerId)?,
                    },
                    *task,
                    *public_model,
                    *deployment_id,
                    *generation,
                )
            }
            ["v1", ..] => return Err(GatewayWorkerApprovalError::InvalidApprovalSyntax),
            [version, ..] if version.starts_with('v') => {
                return Err(GatewayWorkerApprovalError::UnsupportedApprovalVersion);
            }
            [endpoint, task, public_model, deployment_id, generation] => (
                *endpoint,
                GatewayWorkerApprovalIdentity::DiscoverFromAuthenticatedEndpoint,
                *task,
                *public_model,
                *deployment_id,
                *generation,
            ),
            _ => return Err(GatewayWorkerApprovalError::InvalidApprovalSyntax),
        };
        if endpoint.len() > MAX_APPROVAL_ENDPOINT_BYTES {
            return Err(GatewayWorkerApprovalError::EndpointTooLong);
        }
        let task = match task {
            "chat" => TaskKind::Chat,
            "text_to_speech" => TaskKind::TextToSpeech,
            "speech_to_text" => TaskKind::SpeechToText,
            _ => return Err(GatewayWorkerApprovalError::UnknownTask),
        };
        let generation = generation
            .parse::<u64>()
            .map_err(|_| GatewayWorkerApprovalError::InvalidGeneration)
            .and_then(|generation| {
                ModelGeneration::new(generation)
                    .map_err(|_| GatewayWorkerApprovalError::InvalidGeneration)
            })?;
        Ok(Self {
            endpoint: endpoint.to_string(),
            identity,
            task,
            public_model: ModelAlias::new(public_model)
                .map_err(|_| GatewayWorkerApprovalError::InvalidPublicModel)?,
            deployment_id: DeploymentId::new(deployment_id)
                .map_err(|_| GatewayWorkerApprovalError::InvalidDeploymentId)?,
            model_generation: generation,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ApprovalsFileError {
    #[error("approvals text exceeds its encoded size limit")]
    TextTooLarge,
    #[error("approvals text contains more than {limit} entries")]
    TooManyEntries { limit: usize },
    #[error("approvals text line {line} is invalid: {detail}")]
    InvalidEntry { line: usize, detail: String },
}

/// Parses the text of a shared approvals file view: one approval per line,
/// `#` starts a comment, blank lines are skipped, hard entry ceiling. Any
/// invalid entry fails the whole view so a partially-parsed view can never
/// drive admission.
pub fn parse_approvals_text(text: &str) -> Result<Vec<GatewayWorkerApproval>, ApprovalsFileError> {
    if text.len() as u64 > MAX_APPROVALS_FILE_BYTES {
        return Err(ApprovalsFileError::TextTooLarge);
    }
    let mut approvals = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let entry = line.split('#').next().unwrap_or("").trim();
        if entry.is_empty() {
            continue;
        }
        if approvals.len() >= MAX_APPROVALS_FILE_ENTRIES {
            return Err(ApprovalsFileError::TooManyEntries {
                limit: MAX_APPROVALS_FILE_ENTRIES,
            });
        }
        match GatewayWorkerApproval::from_str(entry) {
            Ok(approval) => approvals.push(approval),
            Err(error) => {
                return Err(ApprovalsFileError::InvalidEntry {
                    line: index + 1,
                    detail: error.to_string(),
                })
            }
        }
    }
    Ok(approvals)
}

/// Renders approvals back to file text: one canonical line per approval, in
/// order. Rendering a parsed view yields a file that parses to an equal
/// view (comments are not preserved).
pub fn render_approvals_text(approvals: &[GatewayWorkerApproval]) -> String {
    let mut text = String::new();
    for approval in approvals {
        text.push_str(&approval.render_line());
        text.push('\n');
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_both_approval_forms() {
        let standalone: GatewayWorkerApproval =
            "http://127.0.0.1:19091|chat|chat-model|chat-prod|7"
                .parse()
                .unwrap();
        assert_eq!(standalone.endpoint, "http://127.0.0.1:19091");
        assert_eq!(
            standalone.identity,
            GatewayWorkerApprovalIdentity::DiscoverFromAuthenticatedEndpoint
        );
        assert_eq!(standalone.public_model.as_str(), "chat-model");
        assert_eq!(standalone.deployment_id.as_str(), "chat-prod");
        assert_eq!(
            standalone.model_generation,
            ModelGeneration::new(7).unwrap()
        );

        let fleet: GatewayWorkerApproval =
            "v1|https://worker-b.internal:9470|node-1|worker-b|chat|chat-model|chat-prod|7"
                .parse()
                .unwrap();
        assert_eq!(
            fleet.pinned_identity(),
            Some((
                &NodeId::new("node-1").unwrap(),
                &WorkerId::new("worker-b").unwrap()
            ))
        );
    }

    #[test]
    fn rejects_malformed_approvals() {
        let cases = [
            ("", GatewayWorkerApprovalError::InvalidApprovalSyntax),
            ("a|b|c|d", GatewayWorkerApprovalError::InvalidApprovalSyntax),
            (
                "v1|http://e|n|w|chat|m|d",
                GatewayWorkerApprovalError::InvalidApprovalSyntax,
            ),
            (
                "v2|http://e|n|w|chat|m|d|1",
                GatewayWorkerApprovalError::UnsupportedApprovalVersion,
            ),
            (
                "http://e|voice|m|d|1",
                GatewayWorkerApprovalError::UnknownTask,
            ),
            (
                "http://e|chat|m|d|0",
                GatewayWorkerApprovalError::InvalidGeneration,
            ),
            (
                "http://e|chat|m|d|not-a-number",
                GatewayWorkerApprovalError::InvalidGeneration,
            ),
            (
                "http://e|chat|m with | pipes|d|1",
                GatewayWorkerApprovalError::InvalidApprovalSyntax,
            ),
        ];
        for (line, expected) in cases {
            assert_eq!(
                line.parse::<GatewayWorkerApproval>().unwrap_err(),
                expected,
                "line {line:?}"
            );
        }
    }

    #[test]
    fn oversized_line_is_rejected() {
        let endpoint = format!("http://127.0.0.1:{}/", 9000);
        let padding = "a".repeat(MAX_APPROVAL_LINE_BYTES);
        let error = format!("{endpoint}|chat|{padding}|d|1")
            .parse::<GatewayWorkerApproval>()
            .unwrap_err();
        assert_eq!(error, GatewayWorkerApprovalError::ApprovalTooLong);
        let error = format!("{}|chat|m|d|1", "h".repeat(MAX_APPROVAL_ENDPOINT_BYTES + 1))
            .parse::<GatewayWorkerApproval>()
            .unwrap_err();
        assert_eq!(error, GatewayWorkerApprovalError::EndpointTooLong);
    }

    #[test]
    fn render_line_round_trips_both_forms() {
        let standalone: GatewayWorkerApproval =
            "http://127.0.0.1:19091|chat|chat-model|chat-prod|7"
                .parse()
                .unwrap();
        assert_eq!(
            standalone.render_line(),
            "http://127.0.0.1:19091|chat|chat-model|chat-prod|7"
        );
        assert_eq!(
            standalone
                .render_line()
                .parse::<GatewayWorkerApproval>()
                .unwrap(),
            standalone
        );

        let fleet: GatewayWorkerApproval =
            "v1|https://worker-b.internal:9470|node-1|worker-b|speech_to_text|asr-model|asr-prod|3"
                .parse()
                .unwrap();
        assert_eq!(
            fleet.render_line(),
            "v1|https://worker-b.internal:9470|node-1|worker-b|speech_to_text|asr-model|asr-prod|3"
        );
        assert_eq!(
            fleet
                .render_line()
                .parse::<GatewayWorkerApproval>()
                .unwrap(),
            fleet
        );
    }

    #[test]
    fn file_text_parses_comments_blank_lines_and_reports_line_numbers() {
        let text = "# loopback replica\nhttp://127.0.0.1:9470|chat|tiny-model|deploy-a|1\n\n# v1 fleet\nv1|https://worker-b.internal:9470|node-1|worker-b|chat|tiny-model|deploy-a|2\n";
        let entries = parse_approvals_text(text).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].deployment_id.as_str(), "deploy-a");
        assert_eq!(
            entries[1].model_generation,
            ModelGeneration::new(2).unwrap()
        );

        let error = parse_approvals_text(
            "http://127.0.0.1:9470|chat|tiny-model|deploy-a|1\nnot-an-approval\n",
        )
        .unwrap_err();
        assert_eq!(
            error,
            ApprovalsFileError::InvalidEntry { line: 2, detail: "gateway worker approval must use the standalone 5-field form or v1 8-field fleet form".to_string() }
        );
    }

    #[test]
    fn file_text_rejects_oversized_and_overfull_views() {
        let oversized = format!(
            "# {}\nhttp://127.0.0.1:9470|chat|tiny-model|deploy-a|1\n",
            "x".repeat(MAX_APPROVALS_FILE_BYTES as usize)
        );
        assert_eq!(
            parse_approvals_text(&oversized).unwrap_err(),
            ApprovalsFileError::TextTooLarge
        );

        let mut overfull = String::new();
        for index in 0..=MAX_APPROVALS_FILE_ENTRIES {
            overfull.push_str(&format!(
                "http://127.0.0.1:{}/|chat|tiny-model|deploy-a|1\n",
                9000 + (index % 1000)
            ));
        }
        assert_eq!(
            parse_approvals_text(&overfull).unwrap_err(),
            ApprovalsFileError::TooManyEntries {
                limit: MAX_APPROVALS_FILE_ENTRIES
            }
        );
    }

    #[test]
    fn rendered_file_text_round_trips_through_the_parser() {
        let approvals = vec![
            "http://127.0.0.1:9470|chat|tiny-model|deploy-a|1"
                .parse()
                .unwrap(),
            "v1|https://worker-b.internal:9470|node-1|worker-b|speech_to_text|asr|asr-deploy|4"
                .parse()
                .unwrap(),
        ];
        let text = render_approvals_text(&approvals);
        assert_eq!(parse_approvals_text(&text).unwrap(), approvals);
    }
}
