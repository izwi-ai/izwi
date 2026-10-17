use std::sync::{Mutex, MutexGuard, OnceLock};

/// Serializes tests that read or write process environment variables.
pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("environment lock poisoned")
}

/// A one-worker registry dispatcher for the worker at `endpoint`, approved as
/// the worker describes itself. Its status is fresh for the registry's
/// default TTL; nothing polls it afterwards.
pub(crate) async fn single_worker_dispatcher(
    model: izwi_model_catalog::ModelVariant,
    endpoint: &str,
    credentials: izwi_serving_protocol::ServiceCredentials,
) -> crate::app::remote_chat_dispatch::RemoteChatDispatcher {
    let client = izwi_serving_client::WorkerClient::new(
        endpoint,
        credentials,
        izwi_serving_client::WorkerClientConfig::default(),
    )
    .expect("worker client should initialize");
    single_worker_dispatcher_with_client(model, client).await
}

/// [`single_worker_dispatcher`] over a caller-configured client.
pub(crate) async fn single_worker_dispatcher_with_client(
    model: izwi_model_catalog::ModelVariant,
    client: izwi_serving_client::WorkerClient,
) -> crate::app::remote_chat_dispatch::RemoteChatDispatcher {
    use crate::app::remote_chat_dispatch::{RemoteChatDispatchConfig, RemoteChatDispatcher};
    use crate::worker_registry::{
        ApprovedDeployment, ApprovedWorker, BackendPolicy, WorkerRegistry, WorkerRegistryConfig,
    };

    let descriptor = client
        .descriptor()
        .await
        .expect("descriptor should be available");
    let status = client.status().await.expect("status should be available");
    let deployment = status
        .deployments
        .first()
        .expect("worker should advertise a deployment")
        .clone();
    let registry =
        WorkerRegistry::new(WorkerRegistryConfig::default()).expect("registry should initialize");
    registry
        .approve(ApprovedWorker {
            descriptor,
            client,
            approved_deployments: std::collections::BTreeMap::from([(
                deployment.deployment_id.clone(),
                ApprovedDeployment::from_loaded(&deployment),
            )]),
            validated_capacity: status.capacity.max_active_invocations
                + status.capacity.max_queued_invocations,
        })
        .expect("worker should be approved");
    registry
        .observe_status(status)
        .expect("initial status should be valid");
    RemoteChatDispatcher::new(
        registry,
        RemoteChatDispatchConfig {
            public_model_variant: model,
            deployment_id: deployment.deployment_id,
            policy_revision: izwi_serving_protocol::PolicyRevision::new("test-policy-v1")
                .expect("static policy revision"),
            backend_policy: BackendPolicy::ANY,
            // A worker answers "admission unknown" once runtime admission
            // outlasts this wait, so it must cover a test runtime's admission.
            max_queue_wait: std::time::Duration::from_secs(1),
            max_output_tokens: 128,
            max_output_bytes: 4096,
            slow_consumer_timeout: std::time::Duration::from_millis(100),
            prefix_affinity: None,
        },
    )
    .expect("dispatcher should initialize")
}
