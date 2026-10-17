//! Adopting approved workers into the gateway registry and polling their status.

use anyhow::Context;
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::{
    ModelReadiness, NodeId, ServiceCredentials, WorkerDescriptor, WorkerId, WorkerStatus,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::cli::GatewayTopology;
use crate::{app, gateway_deployments, worker_registry};

pub struct GatewayWorkerStatusPoller {
    pub tasks: Vec<tokio::task::JoinHandle<()>>,
    /// Boot-time poller count. Runtime adoption mutates the live poller pool;
    /// this snapshot pins the boot-sized pool for diagnostics and tests.
    pub poller_count: usize,
}

#[derive(Clone)]
pub struct GatewayWorkerExpectation {
    pub client: WorkerClient,
    pub worker_id: WorkerId,
    pub node_id: NodeId,
    pub deployment: worker_registry::ApprovedDeployment,
    pub validated_capacity: u32,
}

impl Drop for GatewayWorkerStatusPoller {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic per-worker poll cadence: ±10% of the configured interval,
/// drawn from the worker identity so co-configured pollers de-synchronize
/// (serving plan §5.2 "status every 2 seconds with jitter") while the exact
/// cadence stays reproducible under test.
pub fn jittered_poll_interval(base: Duration, seed: u64) -> Duration {
    let mut state = seed | 1;
    let draw = splitmix64(&mut state);
    let per_mille = 90 + (draw % 21);
    let millis = base.as_millis().saturating_mul(u128::from(per_mille)) / 100;
    Duration::from_millis(millis as u64).max(Duration::from_millis(1))
}

pub fn poll_jitter_seed(worker_id: &WorkerId, node_id: &NodeId) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    worker_id.hash(&mut hasher);
    node_id.hash(&mut hasher);
    hasher.finish()
}

/// The widest jittered cadence (110% of the poll interval) must stay below the
/// freshness TTL, or a healthy worker would flap expired between observations.
pub fn validate_status_cadence(ttl: Duration, poll: Duration) -> anyhow::Result<()> {
    if poll.is_zero() {
        anyhow::bail!("--gateway-worker-status-poll-ms must be non-zero");
    }
    let widest = poll.as_millis().saturating_mul(11) / 10;
    if widest >= ttl.as_millis() {
        anyhow::bail!(
            "--gateway-worker-status-poll-ms must be at least 10% below the status TTL to leave room for poll jitter"
        );
    }
    Ok(())
}

pub fn initial_gateway_worker_expectation(
    client: WorkerClient,
    descriptor: &WorkerDescriptor,
    status: &WorkerStatus,
    approval: &gateway_deployments::GatewayWorkerApproval,
) -> anyhow::Result<GatewayWorkerExpectation> {
    if let Some((node_id, worker_id)) = approval.pinned_identity() {
        if descriptor.node_id != *node_id || descriptor.worker_id != *worker_id {
            anyhow::bail!(
                "worker descriptor does not match its operator-approved node and worker identity"
            );
        }
    }
    if status.worker_id != descriptor.worker_id
        || status.node_id != descriptor.node_id
        || status.incarnation_id != descriptor.incarnation_id
    {
        anyhow::bail!("initial worker descriptor and status identity do not match");
    }
    let selected_deployment = status
        .deployments
        .iter()
        .find(|deployment| deployment.deployment_id == approval.deployment_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "worker {} does not advertise configured deployment {}",
                descriptor.worker_id,
                approval.deployment_id
            )
        })?;
    if selected_deployment.public_model != approval.public_model
        || selected_deployment.model_generation != approval.model_generation
        || selected_deployment.task != approval.task
    {
        anyhow::bail!(
            "worker {} deployment {} does not match its configured task/model/generation",
            descriptor.worker_id,
            approval.deployment_id
        );
    }
    let validated_capacity = configured_worker_capacity(status)?;
    Ok(GatewayWorkerExpectation {
        client,
        worker_id: descriptor.worker_id.clone(),
        node_id: descriptor.node_id.clone(),
        deployment: worker_registry::ApprovedDeployment::from_loaded(selected_deployment),
        validated_capacity,
    })
}

pub fn validate_gateway_worker_endpoint_policy(
    topology: GatewayTopology,
    uses_https: bool,
    uses_numeric_loopback_http: bool,
) -> anyhow::Result<()> {
    match topology {
        GatewayTopology::Standalone if !uses_numeric_loopback_http => {
            anyhow::bail!("standalone worker endpoints must use numeric-loopback HTTP")
        }
        GatewayTopology::FleetOneGateway if !uses_https => {
            anyhow::bail!("fleet-one-gateway worker endpoints must use HTTPS")
        }
        _ => {}
    }
    Ok(())
}

pub fn configured_worker_capacity(status: &WorkerStatus) -> anyhow::Result<u32> {
    status
        .capacity
        .max_active_invocations
        .checked_add(status.capacity.max_queued_invocations)
        .filter(|capacity| *capacity > 0)
        .ok_or_else(|| anyhow::anyhow!("worker has invalid configured capacity"))
}

fn validate_gateway_worker_observation(
    expected: &GatewayWorkerExpectation,
    descriptor: &WorkerDescriptor,
    status: WorkerStatus,
) -> anyhow::Result<WorkerStatus> {
    if descriptor.worker_id != expected.worker_id || descriptor.node_id != expected.node_id {
        anyhow::bail!("restarted worker changed its approved logical worker or node identity");
    }
    if status.incarnation_id != descriptor.incarnation_id {
        anyhow::bail!("restarted worker descriptor and status identity do not match");
    }
    validate_gateway_worker_status(expected, status)
}

fn validate_gateway_worker_status(
    expected: &GatewayWorkerExpectation,
    mut status: WorkerStatus,
) -> anyhow::Result<WorkerStatus> {
    if status.worker_id != expected.worker_id || status.node_id != expected.node_id {
        anyhow::bail!("worker status changed its approved logical worker or node identity");
    }
    if configured_worker_capacity(&status)? != expected.validated_capacity {
        anyhow::bail!("restarted worker changed its validated capacity");
    }
    let deployment = status
        .deployments
        .iter()
        .find(|deployment| deployment.deployment_id == expected.deployment.deployment_id)
        .ok_or_else(|| anyhow::anyhow!("restarted worker omitted its approved deployment"))?;
    if worker_registry::ApprovedDeployment::from_loaded(deployment) != expected.deployment {
        anyhow::bail!("restarted worker changed its approved deployment contract");
    }

    // Retain only the configured deployment in the local routing view. Other
    // worker-local deployments are neither approved nor selectable merely
    // because an authenticated endpoint advertised them.
    status
        .deployments
        .retain(|deployment| deployment.deployment_id == expected.deployment.deployment_id);
    Ok(status)
}

pub fn approve_gateway_worker(
    registry: &worker_registry::WorkerRegistry,
    expected: &GatewayWorkerExpectation,
    descriptor: WorkerDescriptor,
    status: WorkerStatus,
) -> anyhow::Result<()> {
    let status = validate_gateway_worker_observation(expected, &descriptor, status)?;
    registry
        .approve(worker_registry::ApprovedWorker {
            descriptor,
            client: expected.client.clone(),
            approved_deployments: BTreeMap::from([(
                expected.deployment.deployment_id.clone(),
                expected.deployment.clone(),
            )]),
            validated_capacity: expected.validated_capacity,
        })
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    registry
        .observe_status(status)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

/// Adopts one approved worker into the running gateway: builds its
/// authenticated client, validates the live worker against the approval's
/// exact deployment contract, approves the replica into the deployment
/// table (filling the generation's learned contract), registers and
/// observes it in the registry, and completes a pending rollout cutover
/// when the adoption observes the successor Ready. Used at boot and at
/// runtime by the shared-approvals adoption path.
/// Builds the private client for one approval and applies the topology's
/// endpoint policy. Its errors are configuration errors no retry can fix.
pub fn gateway_worker_client(
    approval: &gateway_deployments::GatewayWorkerApproval,
    credentials: &ServiceCredentials,
    client_config: &WorkerClientConfig,
    topology: GatewayTopology,
) -> anyhow::Result<WorkerClient> {
    let client = WorkerClient::new(
        approval.endpoint.trim(),
        credentials.clone(),
        client_config.clone(),
    )?;
    validate_gateway_worker_endpoint_policy(
        topology,
        client.uses_https(),
        client.uses_numeric_loopback_http(),
    )?;
    Ok(client)
}

/// Two approved endpoints answered as the same logical worker.
#[derive(Debug, thiserror::Error)]
#[error("endpoint {endpoint} reports worker {worker_id}, already adopted at another endpoint")]
pub struct DuplicateGatewayWorker {
    pub endpoint: String,
    pub worker_id: WorkerId,
}

/// One adopted worker: its logical identity and its status poller.
pub struct AdoptedGatewayWorker {
    pub worker_id: WorkerId,
    pub poller: tokio::task::JoinHandle<()>,
}

/// Adopts approved workers into the registry and starts their pollers. Boot
/// and the runtime reconcile loop share it, so a worker that is down at boot
/// is adopted exactly as one a shared approvals view adds later.
#[derive(Clone)]
pub struct GatewayWorkerAdopter {
    pub registry: worker_registry::WorkerRegistry,
    pub deployment_table: Arc<std::sync::Mutex<gateway_deployments::GatewayDeploymentTable>>,
    pub credentials: ServiceCredentials,
    pub client_config: WorkerClientConfig,
    pub topology: GatewayTopology,
    pub fleet: Option<Arc<app::fleet_coordinator::FleetCoordinator>>,
    pub polling_interval: Duration,
}

impl GatewayWorkerAdopter {
    /// Adopts every approval whose endpoint has no adopted worker yet and
    /// returns the ones that could not be adopted, with why.
    pub async fn adopt_missing(
        &self,
        approvals: &[gateway_deployments::GatewayWorkerApproval],
        adopted: &mut BTreeMap<String, AdoptedGatewayWorker>,
    ) -> Vec<(String, anyhow::Error)> {
        let mut pending = Vec::new();
        for approval in approvals {
            let endpoint = approval.endpoint.trim().to_string();
            if endpoint.is_empty() || adopted.contains_key(&endpoint) {
                continue;
            }
            let adopted_ids: BTreeSet<WorkerId> = adopted
                .values()
                .map(|worker| worker.worker_id.clone())
                .collect();
            match adopt_gateway_worker(
                &self.registry,
                &self.deployment_table,
                approval,
                &self.credentials,
                &self.client_config,
                self.topology,
                &adopted_ids,
            )
            .await
            {
                Ok(expected) => {
                    info!(
                        worker_id = %expected.worker_id,
                        endpoint = %endpoint,
                        generation = expected.deployment.model_generation.get(),
                        "Approved worker adopted"
                    );
                    let worker_id = expected.worker_id.clone();
                    let poller = spawn_gateway_worker_poller(
                        expected,
                        self.registry.clone(),
                        Arc::clone(&self.deployment_table),
                        self.fleet.clone(),
                        self.polling_interval,
                    );
                    adopted.insert(endpoint, AdoptedGatewayWorker { worker_id, poller });
                }
                Err(error) => pending.push((endpoint, error)),
            }
        }
        pending
    }
}

/// Validates one approved worker against its live descriptor and status and
/// registers it. `adopted_ids` are the logical workers already adopted at
/// other endpoints; a duplicate fails before the registry changes.
async fn adopt_gateway_worker(
    registry: &worker_registry::WorkerRegistry,
    deployment_table: &Arc<std::sync::Mutex<gateway_deployments::GatewayDeploymentTable>>,
    approval: &gateway_deployments::GatewayWorkerApproval,
    credentials: &ServiceCredentials,
    client_config: &WorkerClientConfig,
    topology: GatewayTopology,
    adopted_ids: &BTreeSet<WorkerId>,
) -> anyhow::Result<GatewayWorkerExpectation> {
    let endpoint = approval.endpoint.trim();
    let client = gateway_worker_client(approval, credentials, client_config, topology)?;
    let descriptor = client
        .descriptor()
        .await
        .with_context(|| format!("failed to read approved worker descriptor from {endpoint}"))?;
    if adopted_ids.contains(&descriptor.worker_id) {
        return Err(DuplicateGatewayWorker {
            endpoint: endpoint.to_string(),
            worker_id: descriptor.worker_id,
        }
        .into());
    }
    let status = client
        .status()
        .await
        .with_context(|| format!("failed to read initial worker status from {endpoint}"))?;
    let expectation = initial_gateway_worker_expectation(client, &descriptor, &status, approval)?;
    // Register first: a failed registry approval must never mutate the
    // deployment table (a premature cutover could strand admission with no
    // eligible generation).
    approve_gateway_worker(registry, &expectation, descriptor, status.clone())?;
    let cutover_completed = {
        let mut table = deployment_table
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        table
            .approve_replica(expectation.deployment.clone())
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        status
            .deployments
            .iter()
            .find(|deployment| {
                deployment.deployment_id == expectation.deployment.deployment_id
                    && deployment.readiness == ModelReadiness::Ready
            })
            .is_some_and(|deployment| {
                table.observe_generation_ready(
                    expectation.deployment.task,
                    &expectation.deployment.public_model,
                    &expectation.deployment.deployment_id,
                    deployment.model_generation,
                )
            })
    };
    if cutover_completed {
        info!(
            worker_id = %expectation.worker_id,
            deployment = %expectation.deployment.deployment_id,
            generation = expectation.deployment.model_generation.get(),
            "Rollout cutover completed: successor generation is admission-eligible"
        );
    }
    Ok(expectation)
}

/// Spawns the bounded status poller for one adopted worker expectation.
fn spawn_gateway_worker_poller(
    expected: GatewayWorkerExpectation,
    registry: worker_registry::WorkerRegistry,
    deployment_table: Arc<std::sync::Mutex<gateway_deployments::GatewayDeploymentTable>>,
    fleet: Option<std::sync::Arc<app::fleet_coordinator::FleetCoordinator>>,
    polling_interval: Duration,
) -> tokio::task::JoinHandle<()> {
    let cadence = jittered_poll_interval(
        polling_interval,
        poll_jitter_seed(&expected.worker_id, &expected.node_id),
    );
    debug!(
        worker_id = %expected.worker_id,
        configured_poll_ms = polling_interval.as_millis() as u64,
        jittered_poll_ms = cadence.as_millis() as u64,
        "Worker status poller cadence"
    );
    tokio::spawn(async move {
        // Owned by this task, so stopping the poller stops its publisher.
        let publisher = fleet
            .as_ref()
            .map(app::fleet_coordinator::FleetCoordinator::spawn_status_publisher);
        let mut ticker = tokio::time::interval(cadence);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // The adopting observation was recorded synchronously.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match refresh_gateway_worker_status(&registry, &expected, publisher.as_ref()).await {
                Ok(status) => {
                    // DS6: a Ready observation of a pending rollout
                    // successor completes the cutover atomically in
                    // the deployment table (DINV-07).
                    let cutover = status.deployments.iter().find(|deployment| {
                        deployment.deployment_id == expected.deployment.deployment_id
                            && deployment.readiness == ModelReadiness::Ready
                    });
                    let completed = cutover.is_some_and(|deployment| {
                        let mut table = deployment_table
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        table.observe_generation_ready(
                            deployment.task,
                            &deployment.public_model,
                            &deployment.deployment_id,
                            deployment.model_generation,
                        )
                    });
                    if completed {
                        info!(
                            worker_id = %expected.worker_id,
                            deployment = %expected.deployment.deployment_id,
                            generation = expected.deployment.model_generation.get(),
                            "Rollout cutover completed: successor generation is admission-eligible"
                        );
                    }
                }
                Err(error) => {
                    warn!(
                        worker_id = %expected.worker_id,
                        error = %error,
                        "Worker status refresh failed"
                    );
                }
            }
        }
    })
}

/// Records one polled status in the registry, then hands it to the fleet
/// publisher. The registry observation is what keeps the worker fresh, so it
/// never waits on the coordination store; the publisher shares it with peer
/// gateways in the background.
pub async fn refresh_gateway_worker_status(
    registry: &worker_registry::WorkerRegistry,
    expected: &GatewayWorkerExpectation,
    fleet: Option<&app::fleet_coordinator::FleetStatusPublisher>,
) -> anyhow::Result<WorkerStatus> {
    let status = validate_gateway_worker_status(expected, expected.client.status().await?)?;
    match registry.observe_status(status.clone()) {
        Ok(()) => {}
        Err(worker_registry::WorkerRegistryError::UnknownOrStaleIncarnation) => {
            let descriptor = expected.client.descriptor().await?;
            approve_gateway_worker(registry, expected, descriptor, status.clone())?;
        }
        Err(error) => return Err(anyhow::anyhow!(error.to_string())),
    }
    if let Some(publisher) = fleet {
        publisher.publish(status.clone());
    }
    Ok(status)
}
