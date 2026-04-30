//! OSS enterprise hook contracts for Izwi.
//!
//! The open-source build uses the no-op implementations in this crate. A private
//! enterprise repository can depend on these traits, implement them, and pass an
//! [`EnterpriseHooks`] bundle into the reusable server entry point.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

pub type HookResult<T> = Result<T, HookError>;
pub type HookMetadata = BTreeMap<String, String>;

#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("enterprise hook denied request: {0}")]
    Denied(String),
    #[error("enterprise hook failed: {0}")]
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Principal {
    pub id: String,
    pub display_name: Option<String>,
    pub tenant_id: Option<String>,
    pub roles: Vec<String>,
    pub attributes: HookMetadata,
}

impl Principal {
    pub fn local_anonymous() -> Self {
        Self {
            id: "local-anonymous".to_string(),
            display_name: Some("Local user".to_string()),
            tenant_id: None,
            roles: vec!["local".to_string()],
            attributes: HookMetadata::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeaderPair {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestEnvelope {
    pub correlation_id: String,
    pub method: String,
    pub path: String,
    pub headers: Vec<HeaderPair>,
    pub remote_addr: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnterpriseAction {
    HttpRequest,
    Inference,
    ModelDownload,
    ModelLoad,
    ModelDelete,
    DataRead,
    DataWrite,
    DataDelete,
    DataExport,
    VoiceClone,
    AgentToolExecute,
    AdminOperation,
    DesktopStartup,
    SecretRead,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    HttpRoute,
    InferenceRequest,
    ModelArtifact,
    DataRecord,
    MediaObject,
    VoiceProfile,
    AgentTool,
    DesktopRuntime,
    AdminWorkflow,
    Secret,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceDescriptor {
    pub kind: ResourceKind,
    pub id: Option<String>,
    pub model_id: Option<String>,
    pub tenant_id: Option<String>,
    pub attributes: HookMetadata,
}

impl ResourceDescriptor {
    pub fn http_route(path: impl Into<String>) -> Self {
        Self {
            kind: ResourceKind::HttpRoute,
            id: Some(path.into()),
            model_id: None,
            tenant_id: None,
            attributes: HookMetadata::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationRequest {
    pub principal: Principal,
    pub action: EnterpriseAction,
    pub resource: ResourceDescriptor,
    pub request: Option<RequestEnvelope>,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationDecision {
    pub allowed: bool,
    pub reason: Option<String>,
}

impl AuthorizationDecision {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
        }
    }

    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditCategory {
    Request,
    Inference,
    Data,
    Model,
    Admin,
    Agent,
    Desktop,
    Security,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    Success,
    Failure,
    Denied,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub category: AuditCategory,
    pub action: EnterpriseAction,
    pub outcome: AuditOutcome,
    pub principal: Option<Principal>,
    pub resource: Option<ResourceDescriptor>,
    pub correlation_id: Option<String>,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataGovernanceAction {
    Read,
    Write,
    Delete,
    Export,
    Retain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataGovernanceRequest {
    pub principal: Option<Principal>,
    pub action: DataGovernanceAction,
    pub resource: ResourceDescriptor,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataGovernanceDecision {
    pub allowed: bool,
    pub retain_for_days: Option<u32>,
    pub reason: Option<String>,
}

impl DataGovernanceDecision {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            retain_for_days: None,
            reason: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArtifactOperation {
    ResolveCatalog,
    Download,
    Verify,
    Load,
    Unload,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactRequest {
    pub principal: Option<Principal>,
    pub operation: ModelArtifactOperation,
    pub model_id: String,
    pub source: Option<String>,
    pub expected_sha256: Option<String>,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactDecision {
    pub allowed: bool,
    pub source_override: Option<String>,
    pub required_sha256: Option<String>,
    pub reason: Option<String>,
}

impl ModelArtifactDecision {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            source_override: None,
            required_sha256: None,
            reason: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservabilityEventKind {
    Request,
    Inference,
    Runtime,
    Model,
    Data,
    Admin,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservabilityEvent {
    pub kind: ObservabilityEventKind,
    pub name: String,
    pub principal: Option<Principal>,
    pub correlation_id: Option<String>,
    pub attributes: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaRequest {
    pub principal: Option<Principal>,
    pub action: EnterpriseAction,
    pub resource: ResourceDescriptor,
    pub estimated_units: Option<u64>,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaDecision {
    pub allowed: bool,
    pub reason: Option<String>,
}

impl QuotaDecision {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecretRequest {
    pub key: String,
    pub purpose: String,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecretValue {
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdminWorkflowRequest {
    pub principal: Option<Principal>,
    pub action: EnterpriseAction,
    pub resource: ResourceDescriptor,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdminWorkflowDecision {
    pub allowed: bool,
    pub requires_approval: bool,
    pub reason: Option<String>,
}

impl AdminWorkflowDecision {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            requires_approval: false,
            reason: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentToolRequest {
    pub principal: Option<Principal>,
    pub tool_name: String,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentToolDecision {
    pub allowed: bool,
    pub reason: Option<String>,
}

impl AgentToolDecision {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesktopPolicyRequest {
    pub host: String,
    pub metadata: HookMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesktopPolicy {
    pub allow_local_server: bool,
    pub allow_telemetry: bool,
    pub managed_server_url: Option<String>,
    pub metadata: HookMetadata,
}

impl DesktopPolicy {
    pub fn community_default() -> Self {
        Self {
            allow_local_server: true,
            allow_telemetry: true,
            managed_server_url: None,
            metadata: HookMetadata::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Edition {
    Community,
    Enterprise,
    Other(String),
}

#[async_trait]
pub trait AuthProvider: Send + Sync {
    async fn authenticate(&self, request: &RequestEnvelope) -> HookResult<Principal>;
}

#[async_trait]
pub trait PolicyEngine: Send + Sync {
    async fn authorize(&self, request: &AuthorizationRequest) -> HookResult<AuthorizationDecision>;
}

#[async_trait]
pub trait AuditSink: Send + Sync {
    async fn record(&self, event: AuditEvent) -> HookResult<()>;
}

#[async_trait]
pub trait DataGovernanceProvider: Send + Sync {
    async fn evaluate(&self, request: &DataGovernanceRequest)
        -> HookResult<DataGovernanceDecision>;
}

#[async_trait]
pub trait ModelArtifactPolicy: Send + Sync {
    async fn evaluate(&self, request: &ModelArtifactRequest) -> HookResult<ModelArtifactDecision>;
}

#[async_trait]
pub trait ObservabilitySink: Send + Sync {
    async fn record(&self, event: ObservabilityEvent) -> HookResult<()>;
}

#[async_trait]
pub trait QuotaLimiter: Send + Sync {
    async fn evaluate(&self, request: &QuotaRequest) -> HookResult<QuotaDecision>;
}

#[async_trait]
pub trait SecretProvider: Send + Sync {
    async fn get_secret(&self, request: &SecretRequest) -> HookResult<Option<SecretValue>>;
}

#[async_trait]
pub trait AdminWorkflowHooks: Send + Sync {
    async fn review(&self, request: &AdminWorkflowRequest) -> HookResult<AdminWorkflowDecision>;
}

#[async_trait]
pub trait AgentToolPolicy: Send + Sync {
    async fn authorize_tool(&self, request: &AgentToolRequest) -> HookResult<AgentToolDecision>;
}

#[async_trait]
pub trait DesktopPolicyProvider: Send + Sync {
    async fn desktop_policy(&self, request: &DesktopPolicyRequest) -> HookResult<DesktopPolicy>;
}

pub trait EditionCapabilities: Send + Sync {
    fn edition(&self) -> Edition;
    fn capability_enabled(&self, capability: &str) -> bool;
}

#[derive(Clone)]
pub struct EnterpriseHooks {
    pub auth: Arc<dyn AuthProvider>,
    pub policy: Arc<dyn PolicyEngine>,
    pub audit: Arc<dyn AuditSink>,
    pub data_governance: Arc<dyn DataGovernanceProvider>,
    pub model_artifacts: Arc<dyn ModelArtifactPolicy>,
    pub observability: Arc<dyn ObservabilitySink>,
    pub quotas: Arc<dyn QuotaLimiter>,
    pub secrets: Arc<dyn SecretProvider>,
    pub admin_workflows: Arc<dyn AdminWorkflowHooks>,
    pub agent_tools: Arc<dyn AgentToolPolicy>,
    pub desktop_policy: Arc<dyn DesktopPolicyProvider>,
    pub edition: Arc<dyn EditionCapabilities>,
}

impl EnterpriseHooks {
    pub fn noop() -> Self {
        Self {
            auth: Arc::new(NoopAuthProvider),
            policy: Arc::new(NoopPolicyEngine),
            audit: Arc::new(NoopAuditSink),
            data_governance: Arc::new(NoopDataGovernanceProvider),
            model_artifacts: Arc::new(NoopModelArtifactPolicy),
            observability: Arc::new(NoopObservabilitySink),
            quotas: Arc::new(NoopQuotaLimiter),
            secrets: Arc::new(NoopSecretProvider),
            admin_workflows: Arc::new(NoopAdminWorkflowHooks),
            agent_tools: Arc::new(NoopAgentToolPolicy),
            desktop_policy: Arc::new(NoopDesktopPolicyProvider),
            edition: Arc::new(NoopEditionCapabilities),
        }
    }
}

impl Default for EnterpriseHooks {
    fn default() -> Self {
        Self::noop()
    }
}

impl fmt::Debug for EnterpriseHooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EnterpriseHooks")
            .field("edition", &self.edition.edition())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
pub struct NoopAuthProvider;

#[async_trait]
impl AuthProvider for NoopAuthProvider {
    async fn authenticate(&self, _request: &RequestEnvelope) -> HookResult<Principal> {
        Ok(Principal::local_anonymous())
    }
}

#[derive(Debug, Default)]
pub struct NoopPolicyEngine;

#[async_trait]
impl PolicyEngine for NoopPolicyEngine {
    async fn authorize(
        &self,
        _request: &AuthorizationRequest,
    ) -> HookResult<AuthorizationDecision> {
        Ok(AuthorizationDecision::allow())
    }
}

#[derive(Debug, Default)]
pub struct NoopAuditSink;

#[async_trait]
impl AuditSink for NoopAuditSink {
    async fn record(&self, _event: AuditEvent) -> HookResult<()> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct NoopDataGovernanceProvider;

#[async_trait]
impl DataGovernanceProvider for NoopDataGovernanceProvider {
    async fn evaluate(
        &self,
        _request: &DataGovernanceRequest,
    ) -> HookResult<DataGovernanceDecision> {
        Ok(DataGovernanceDecision::allow())
    }
}

#[derive(Debug, Default)]
pub struct NoopModelArtifactPolicy;

#[async_trait]
impl ModelArtifactPolicy for NoopModelArtifactPolicy {
    async fn evaluate(&self, _request: &ModelArtifactRequest) -> HookResult<ModelArtifactDecision> {
        Ok(ModelArtifactDecision::allow())
    }
}

#[derive(Debug, Default)]
pub struct NoopObservabilitySink;

#[async_trait]
impl ObservabilitySink for NoopObservabilitySink {
    async fn record(&self, _event: ObservabilityEvent) -> HookResult<()> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct NoopQuotaLimiter;

#[async_trait]
impl QuotaLimiter for NoopQuotaLimiter {
    async fn evaluate(&self, _request: &QuotaRequest) -> HookResult<QuotaDecision> {
        Ok(QuotaDecision::allow())
    }
}

#[derive(Debug, Default)]
pub struct NoopSecretProvider;

#[async_trait]
impl SecretProvider for NoopSecretProvider {
    async fn get_secret(&self, _request: &SecretRequest) -> HookResult<Option<SecretValue>> {
        Ok(None)
    }
}

#[derive(Debug, Default)]
pub struct NoopAdminWorkflowHooks;

#[async_trait]
impl AdminWorkflowHooks for NoopAdminWorkflowHooks {
    async fn review(&self, _request: &AdminWorkflowRequest) -> HookResult<AdminWorkflowDecision> {
        Ok(AdminWorkflowDecision::allow())
    }
}

#[derive(Debug, Default)]
pub struct NoopAgentToolPolicy;

#[async_trait]
impl AgentToolPolicy for NoopAgentToolPolicy {
    async fn authorize_tool(&self, _request: &AgentToolRequest) -> HookResult<AgentToolDecision> {
        Ok(AgentToolDecision::allow())
    }
}

#[derive(Debug, Default)]
pub struct NoopDesktopPolicyProvider;

#[async_trait]
impl DesktopPolicyProvider for NoopDesktopPolicyProvider {
    async fn desktop_policy(&self, _request: &DesktopPolicyRequest) -> HookResult<DesktopPolicy> {
        Ok(DesktopPolicy::community_default())
    }
}

#[derive(Debug, Default)]
pub struct NoopEditionCapabilities;

impl EditionCapabilities for NoopEditionCapabilities {
    fn edition(&self) -> Edition {
        Edition::Community
    }

    fn capability_enabled(&self, _capability: &str) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> RequestEnvelope {
        RequestEnvelope {
            correlation_id: "request-1".to_string(),
            method: "GET".to_string(),
            path: "/v1/models".to_string(),
            headers: Vec::new(),
            remote_addr: None,
        }
    }

    #[tokio::test]
    async fn noop_hooks_authenticate_local_principal() {
        let hooks = EnterpriseHooks::noop();

        let principal = hooks
            .auth
            .authenticate(&request())
            .await
            .expect("noop auth should succeed");

        assert_eq!(principal.id, "local-anonymous");
        assert!(principal.roles.iter().any(|role| role == "local"));
    }

    #[tokio::test]
    async fn noop_policy_allows_requests() {
        let hooks = EnterpriseHooks::noop();
        let principal = Principal::local_anonymous();
        let decision = hooks
            .policy
            .authorize(&AuthorizationRequest {
                principal,
                action: EnterpriseAction::HttpRequest,
                resource: ResourceDescriptor::http_route("/v1/models"),
                request: Some(request()),
                metadata: HookMetadata::new(),
            })
            .await
            .expect("noop policy should succeed");

        assert!(decision.allowed);
    }

    #[test]
    fn default_hooks_report_community_edition() {
        let hooks = EnterpriseHooks::default();

        assert_eq!(hooks.edition.edition(), Edition::Community);
        assert!(!hooks.edition.capability_enabled("enterprise.sso"));
    }
}
