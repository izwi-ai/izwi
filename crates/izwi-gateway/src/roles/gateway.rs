//! The gateway role: validates its configuration, builds the gateway state over the approved
//! workers and serves it.

use anyhow::Context;
use izwi_hooks::EnterpriseHooks;
use izwi_model_catalog::parse_model_variant;
use izwi_serving_client::WorkerClientConfig;
use izwi_serving_protocol::{
    render_approvals_text, CredentialId, DeploymentId, ModelAlias, ModelGeneration, NdjsonLimits,
    PolicyRevision, ServiceBearerToken, ServiceCredentials, TaskKind, MAX_REMAINING_TIME_MS,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::cli::{BackendArg, GatewayArgs, GatewayTopology};
use crate::config::GatewayConfig;
use crate::logging::{service_name, SERVICE_VERSION};
use crate::roles::gateway_workers::{
    gateway_worker_client, validate_status_cadence, AdoptedGatewayWorker, DuplicateGatewayWorker,
    GatewayWorkerAdopter, GatewayWorkerStatusPoller,
};
use crate::shutdown::{
    gateway_shutdown_requested, serve_with_shutdown_plan, ShutdownPhases, ShutdownPlan,
};
use crate::{
    app, gateway, gateway_deployments, gateway_shared_approvals, gateway_worker_tls,
    worker_registry, GatewayPerimeterConfig, GatewayRateQuotaConfig,
    GatewayTenantConcurrencyConfig,
};

pub const MAX_CONFIGURED_GATEWAY_WORKERS: usize = 256;
const MAX_GATEWAY_STATUS_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_GATEWAY_ADMISSION_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_GATEWAY_STREAM_PHASE_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_GATEWAY_SHUTDOWN_BUDGET_SECS: u64 = 600;
const GATEWAY_NDJSON_MAX_LINE_BYTES: usize = 1024 * 1024;
const GATEWAY_NDJSON_MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;
const GATEWAY_NDJSON_MAX_EVENTS: usize = 8192;

/// Apply a fleet partition to the per-gateway quota budgets.
///
/// When `partition` is `Some`, each per-tenant rate and concurrency limit is
/// divided by the fleet size (floored to 1), so N gateways each own a strict
/// non-overlapping slice of the total configured budget. No shared atomic
/// counter or coordination service is required. A crashed gateway releases
/// its partition immediately; the other gateways' partitions are unaffected.
fn apply_fleet_partition(
    rate_quota: GatewayRateQuotaConfig,
    tenant_concurrency: GatewayTenantConcurrencyConfig,
    partition: Option<crate::gateway_fleet::FleetPartition>,
) -> (GatewayRateQuotaConfig, GatewayTenantConcurrencyConfig) {
    let Some(partition) = partition else {
        return (rate_quota, tenant_concurrency);
    };
    let requests = partition.partition_limit(rate_quota.requests_per_minute());
    let burst = partition.partition_limit(rate_quota.burst_requests());
    let tracked = rate_quota.max_tracked_tenants();
    let tokens = rate_quota
        .tokens_per_minute()
        .map(|tokens| partition.partition_limit(tokens));
    let rate_quota = GatewayRateQuotaConfig::new(requests, burst, tracked)
        .and_then(|config| config.with_tokens_per_minute(tokens))
        .expect("partitioned rate-quota must remain valid");
    let per_tenant = u32::try_from(tenant_concurrency.max_active_per_tenant())
        .ok()
        .map(|value| partition.partition_concurrency(value))
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(1);
    let owned = u32::try_from(tenant_concurrency.max_owned_work())
        .ok()
        .map(|value| partition.partition_concurrency(value))
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(1);
    let tenant_concurrency =
        GatewayTenantConcurrencyConfig::new(per_tenant.max(1).min(owned.max(1)), owned.max(1))
            .expect("partitioned tenant concurrency must remain valid");
    (rate_quota, tenant_concurrency)
}

/// Serves the gateway role until shutdown. The caller has set up logging.
pub async fn run_gateway(
    config: GatewayConfig,
    enterprise_hooks: EnterpriseHooks,
) -> anyhow::Result<()> {
    let args = &config.args;
    let perimeter = GatewayPerimeterConfig::from_env()?;
    let fleet_partition = crate::gateway_fleet::FleetPartition::from_env()?;
    if let Some(partition) = fleet_partition {
        info!(
            service = service_name(),
            version = SERVICE_VERSION,
            partition = partition.index(),
            fleet_size = partition.size(),
            "Gateway fleet partition active: per-tenant budgets are divided across gateways (explicit quota fallback; the worker stays the atomic admission arbiter)"
        );
    }
    if let Ok(Some(shared)) = crate::gateway_shared_approvals::SharedApprovalsConfig::from_env() {
        match crate::gateway_shared_approvals::SharedApprovalsView::load(shared) {
            Ok(view) => info!(
                service = service_name(),
                version = SERVICE_VERSION,
                approvals = view.approvals().len(),
                "Shared fleet approvals loaded: all gateways reading this file approve the same worker set"
            ),
            Err(error) => warn!(error = %error, "Shared fleet approvals file could not be loaded at startup"),
        }
    }
    let rate_quota = GatewayRateQuotaConfig::from_env()?;
    let tenant_concurrency = GatewayTenantConcurrencyConfig::from_env(args.gateway_max_in_flight)?;
    let (rate_quota, tenant_concurrency) =
        apply_fleet_partition(rate_quota, tenant_concurrency, fleet_partition);
    perimeter.validate_public_ingress(&config.serve)?;
    // DS0.5 scoped credentials: the manifest is the only activation switch.
    // Unset keeps the durable store untouched and authentication byte-identical.
    let principal_keys = match crate::gateway_principal_keys::manifest_path_from_env()? {
        Some(manifest_path) => {
            let store = crate::db::sqlite::StoreDatabase::from_default_path()?;
            let directory = crate::gateway_principal_keys::GatewayPrincipalDirectory::bootstrap(
                &store,
                &manifest_path,
                &perimeter,
            )
            .await?;
            info!(
                service = service_name(),
                version = SERVICE_VERSION,
                principals = directory.len(),
                "Gateway scoped principal keys loaded from the durable store"
            );
            directory
        }
        None => crate::gateway_principal_keys::GatewayPrincipalDirectory::empty(),
    };
    let (state, _status_poller) = gateway_state(&config, enterprise_hooks, perimeter).await?;
    let state = state
        .with_rate_quota_config(rate_quota)
        .with_tenant_concurrency_config(tenant_concurrency)
        .with_principal_keys(principal_keys)
        .with_body_read_timeout(Duration::from_millis(args.gateway_body_read_timeout_ms));
    state.lifecycle.mark_ready();

    info!(
        service = service_name(),
        version = SERVICE_VERSION,
        "Starting Izwi public API gateway"
    );
    let app = gateway::create_gateway_router(state.clone(), &config.serve);
    let addr = format!("{}:{}", config.serve.host, config.serve.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Gateway listening on http://{}", listener.local_addr()?);

    let plan = ShutdownPlan {
        prestop: Duration::from_secs(args.gateway_prestop_secs),
        budget: Duration::from_secs(args.gateway_shutdown_budget_secs),
    };
    serve_gateway(listener, app, state, gateway_shutdown_requested(), plan).await
}

pub async fn serve_gateway<S>(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    state: gateway::GatewayState,
    shutdown_requested: S,
    plan: ShutdownPlan,
) -> anyhow::Result<()>
where
    S: std::future::Future<Output = ()> + Send + 'static,
{
    let finished = serve_with_shutdown_plan(
        |signal| async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(signal)
                .await
        },
        state,
        shutdown_requested,
        plan,
    )
    .await;
    match finished {
        Some(result) => result?,
        None => warn!(
            budget_secs = plan.budget.as_secs(),
            "Gateway shutdown budget spent; dropping remaining connections"
        ),
    }
    Ok(())
}

impl ShutdownPhases for gateway::GatewayState {
    fn begin_prestop(&self) {
        gateway::GatewayState::begin_prestop(self);
    }

    fn begin_drain(&self) {
        gateway::GatewayState::begin_drain(self);
    }

    fn end_streams(&self) {
        gateway::GatewayState::end_streams(self);
    }
}

pub async fn gateway_state(
    config: &GatewayConfig,
    enterprise_hooks: EnterpriseHooks,
    perimeter: GatewayPerimeterConfig,
) -> anyhow::Result<(gateway::GatewayState, GatewayWorkerStatusPoller)> {
    let args = &config.args;
    validate_gateway_topology_source(args)?;
    if args.worker_endpoint.is_some()
        && !(args.gateway_worker_endpoints.is_empty() && args.gateway_worker_approvals.is_empty())
    {
        anyhow::bail!(
            "--worker-endpoint cannot be combined with registry worker endpoints or approvals; it is the one-worker form of --gateway-worker-endpoint"
        );
    }
    if args.worker_incarnation.is_some() {
        if args.worker_endpoint.is_none() {
            anyhow::bail!(
                "--worker-incarnation applies only to the legacy --worker-endpoint form; registry routing validates each discovered incarnation"
            );
        }
        warn!(
            service = service_name(),
            version = SERVICE_VERSION,
            "--worker-incarnation is ignored: the gateway fences every attempt to the worker incarnation it observed and adopts a restarted worker"
        );
    }

    validate_gateway_limits(args)?;
    let public_model_variant = parse_model_variant(required_gateway_value(
        &args.public_model,
        "--public-model",
    )?)?;
    let public_model = ModelAlias::new(public_model_variant.dir_name())?;
    let (cli_approvals, worker_approvals, shared_approvals_view) =
        configured_gateway_worker_approvals(args, &public_model)?;
    if worker_approvals.is_empty() && shared_approvals_view.is_none() {
        anyhow::bail!(
            "gateway mode requires a worker: --gateway-worker-approval, --gateway-worker-endpoint, --worker-endpoint, or a shared approvals file"
        );
    }
    let credentials = gateway_worker_credentials(args)?;
    // With a shared approvals file, runtime view adoption (DS6) can add
    // workers beyond the boot view; the registry capacity follows the
    // approvals ceiling. Without one, the boot view is the exact capacity.
    let shared_capacity = shared_approvals_view.is_some();
    let registry_config = worker_registry::WorkerRegistryConfig {
        max_workers: if shared_capacity {
            worker_approvals
                .len()
                .max(izwi_serving_protocol::MAX_APPROVALS_FILE_ENTRIES)
        } else {
            worker_approvals.len()
        },
        max_deployments_per_worker: 32,
        max_local_dispatches: args.gateway_max_in_flight,
        status_ttl: Duration::from_millis(args.gateway_worker_status_ttl_ms),
        routing: worker_registry::RoutingConfig {
            prefix_balance_threshold: args.gateway_router_balance_threshold,
            kv_pressure_pct: args.gateway_router_kv_pressure_pct,
        },
        ..worker_registry::WorkerRegistryConfig::default()
    };
    let registry = worker_registry::WorkerRegistry::new(registry_config)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let worker_tls = gateway_worker_tls::worker_client_tls_from_env()?;
    validate_gateway_topology_policy(args.gateway_topology, &worker_approvals, &worker_tls)?;
    let client_config = gateway_worker_client_config(args, worker_tls);

    // DS6: the approval view defines the pool/generation structure before any
    // worker is validated, so a rollout window in the approvals reaches
    // selection and the replica path accepts both window generations. The
    // table is shared: the approvals refresh adopts changed views and the
    // poller completes pending cutovers at runtime.
    let rendered_view = render_approvals_text(&worker_approvals);
    let deployment_table = {
        let mut table = gateway_deployments::GatewayDeploymentTable::default();
        table
            .apply_view(&worker_approvals)
            .map_err(|error| anyhow::anyhow!("configured approvals view: {error}"))?;
        Arc::new(std::sync::Mutex::new(table))
    };
    let approvals_reporter =
        gateway_shared_approvals::GatewayApprovalsReporter::new(Arc::clone(&deployment_table));
    approvals_reporter.record(
        shared_approvals_view
            .as_ref()
            .map(gateway_shared_approvals::SharedApprovalsView::file_digest),
        &worker_approvals,
    );
    registry.set_generation_gate({
        let deployment_table = Arc::clone(&deployment_table);
        Arc::new(move |task, public_model| {
            let table = deployment_table
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            table.eligible_generation(task, public_model)
        })
    });

    let mut endpoints = BTreeSet::new();
    for approval in &worker_approvals {
        let endpoint = approval.endpoint.trim();
        if endpoint.is_empty() {
            anyhow::bail!("configured gateway worker endpoints must not be empty");
        }
        if !endpoints.insert(endpoint.to_string()) {
            anyhow::bail!("duplicate configured gateway worker endpoint: {endpoint}");
        }
    }

    // Approval errors a restart cannot fix fail boot. A worker that is merely
    // unreachable is adopted with retry once the pollers start below, so one
    // down replica never keeps the gateway from booting.
    for approval in &worker_approvals {
        gateway_worker_client(
            approval,
            &credentials,
            &client_config,
            args.gateway_topology,
        )
        .with_context(|| format!("invalid approved worker {}", approval.endpoint.trim()))?;
    }

    // Every chat model the approvals view names is served, the primary
    // `--public-model` first; requests route by their `model`.
    let chat_models = {
        let table = deployment_table
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut aliases = vec![public_model.clone()];
        for approval in &worker_approvals {
            if approval.task == TaskKind::Chat && !aliases.contains(&approval.public_model) {
                aliases.push(approval.public_model.clone());
            }
        }
        aliases
            .into_iter()
            .map(|alias| {
                let deployment = table.select(TaskKind::Chat, &alias).ok_or_else(|| {
                    anyhow::anyhow!(
                        "configured gateway worker approvals do not include chat model {alias}"
                    )
                })?;
                let variant = parse_model_variant(alias.as_str()).with_context(|| {
                    format!("approved chat model {alias} is not a model this gateway knows")
                })?;
                Ok((variant, deployment.deployment_id().clone()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    let policy_revision = PolicyRevision::new(args.gateway_policy_revision.trim())?;
    let dispatchers = chat_models
        .into_iter()
        .map(|(public_model_variant, deployment_id)| {
            app::remote_chat_dispatch::RemoteChatDispatcher::new(
                registry.clone(),
                app::remote_chat_dispatch::RemoteChatDispatchConfig {
                    public_model_variant,
                    deployment_id,
                    policy_revision: policy_revision.clone(),
                    backend_policy: gateway_backend_policy(config.backend.as_ref()),
                    max_queue_wait: Duration::from_millis(args.gateway_worker_queue_wait_ms),
                    max_output_tokens: 4096,
                    max_output_bytes: 512 * 1024,
                    slow_consumer_timeout: Duration::from_millis(
                        args.gateway_slow_consumer_timeout_ms,
                    ),
                    prefix_affinity: prefix_affinity_config(args),
                },
            )
            .map_err(|error| anyhow::anyhow!(error.message))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Multi-gateway fleets share in-transit capacity claims
    // through one coordination database — a SQLite file by default, or a
    // server-backed database URL for shared fleets (DS5). Unset means
    // single-gateway operation with purely process-local state. When the
    // coordination database is set, the shared-atomic claim path is the
    // default selection posture: the worker remains the atomic admission
    // arbiter, and 1/N partitioning stays the explicitly-chosen quota
    // fallback (`IZWI_GATEWAY_FLEET_PARTITION`/`_SIZE`).
    let fleet_claim_ttl = crate::gateway_fleet::fleet_claim_ttl_from_env()?;
    let fleet_claim_timeout = crate::gateway_fleet::fleet_claim_timeout_from_env()?;
    let mut fleet_maintenance: Option<tokio::task::JoinHandle<()>> = None;
    let fleet = match crate::gateway_fleet::fleet_database_from_env()? {
        Some(fleet_database) => {
            let store_database = match fleet_database {
                crate::gateway_fleet::FleetDatabase::Path(path) => {
                    crate::db::StoreDatabase::new(path)
                }
                crate::gateway_fleet::FleetDatabase::Url(url) => {
                    crate::db::StoreDatabase::from_coordination_url(url)
                }
            };
            let store = crate::gateway_fleet::FleetStore::new(store_database);
            // Two fleet gateways booting simultaneously race the SQLite
            // journal-mode setup before the busy timeout is in play, so the
            // first open retries briefly before failing closed.
            let mut opened = Err(anyhow::anyhow!("fleet coordination database never opened"));
            for _ in 0..10 {
                match store.connection().await {
                    Ok(_) => {
                        opened = Ok(());
                        break;
                    }
                    Err(error) => opened = Err(error),
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            opened.context("Failed to open fleet coordination database")?;
            let coordinator = Arc::new(
                app::fleet_coordinator::FleetCoordinator::new(
                    store,
                    crate::gateway_fleet::gateway_identity()?,
                )
                .with_claim_ttl(fleet_claim_ttl)
                .with_claim_timeout(fleet_claim_timeout),
            );
            let released = coordinator.release_own_claims().await;
            info!(
                service = service_name(),
                version = SERVICE_VERSION,
                gateway_id = coordinator.gateway_id(),
                released_own_claims = released,
                claim_ttl_ms = fleet_claim_ttl.as_millis() as u64,
                claim_timeout_ms = fleet_claim_timeout.as_millis() as u64,
                selection_mode = "shared_atomic_claims",
                // Parsed once already in run_gateway; re-read here so the
                // posture log names the effective quota mode.
                quota_mode = if crate::gateway_fleet::FleetPartition::from_env()?.is_some() {
                    "partitioned_1_of_n"
                } else {
                    "worker_authoritative"
                },
                "Fleet coordination enabled: capacity claims are shared"
            );
            let maintenance_coordinator = coordinator.clone();
            fleet_maintenance = Some(tokio::spawn(async move {
                let mut ticker =
                    tokio::time::interval(app::fleet_coordinator::FLEET_MAINTENANCE_INTERVAL);
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                // The first tick fires immediately; the boot path just
                // released this gateway's own claims, so sweep on the cadence.
                ticker.tick().await;
                loop {
                    ticker.tick().await;
                    let reaped = maintenance_coordinator.maintenance_sweep().await;
                    if reaped > 0 {
                        debug!(
                            service = service_name(),
                            reaped_claims = reaped,
                            "Fleet coordination maintenance sweep"
                        );
                    }
                }
            }));
            Some(coordinator)
        }
        None => None,
    };
    let mut dispatchers = dispatchers
        .into_iter()
        .map(|dispatcher| match fleet.clone() {
            Some(coordinator) => dispatcher.with_fleet_coordinator(coordinator),
            None => dispatcher,
        });
    let polling_interval = Duration::from_millis(args.gateway_worker_status_poll_ms);
    // `tasks` collects the maintenance and approvals reconcile tasks that
    // GatewayWorkerStatusPoller aborts on drop. Status pollers live in the
    // adoption map and terminate with the gateway runtime.
    let mut tasks: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    if let Some(maintenance) = fleet_maintenance {
        tasks.push(maintenance);
    }
    let adopter = GatewayWorkerAdopter {
        registry: registry.clone(),
        deployment_table: Arc::clone(&deployment_table),
        credentials: credentials.clone(),
        client_config: client_config.clone(),
        topology: args.gateway_topology,
        fleet: fleet.clone(),
        polling_interval,
    };
    let mut adopted = BTreeMap::new();
    let mut boot_pending = 0;
    for (endpoint, error) in adopter.adopt_missing(&worker_approvals, &mut adopted).await {
        if let Some(duplicate) = error.downcast_ref::<DuplicateGatewayWorker>() {
            anyhow::bail!(
                "configured gateway endpoints must identify distinct logical workers; {duplicate}"
            );
        }
        boot_pending += 1;
        warn!(
            endpoint = %endpoint,
            error = %format!("{error:#}"),
            "Approved worker is not adoptable yet; the gateway boots without it and keeps retrying"
        );
    }
    let boot_poller_count = adopted.len();

    // Reconcile worker adoption at runtime: retry approved workers that were
    // unavailable at boot, and, with a shared approvals file, keep a locally
    // cached fresh view and adopt changed views (DS6). The rollout
    // coordinator drives generation transitions through atomic file writes;
    // adoption applies the DINV-07 eligibility rule in one table mutation,
    // and workers approved by the view are adopted (or retried) into the
    // registry and poller set. A rejected view retains the previous approval
    // state fail-closed.
    if shared_approvals_view.is_some() || boot_pending > 0 {
        let mut view = shared_approvals_view;
        let interval = view.as_ref().map_or(
            polling_interval,
            gateway_shared_approvals::SharedApprovalsView::ttl,
        );
        let refresh_table = Arc::clone(&deployment_table);
        let refresh_reporter = approvals_reporter.clone();
        let refresh_cli_approvals = cli_approvals.clone();
        let boot_approvals = worker_approvals.clone();
        tasks.push(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await;
            let mut last_applied = Some(rendered_view);
            loop {
                ticker.tick().await;
                let effective = match view.as_mut() {
                    Some(view) => {
                        if let Err(error) = view.refresh_if_due() {
                            warn!(error = %error, "Shared fleet approvals refresh failed; retaining previous view");
                            continue;
                        }
                        let mut effective = refresh_cli_approvals.clone();
                        effective.extend(view.approvals().iter().cloned());
                        let rendered = render_approvals_text(&effective);
                        if last_applied.as_deref() == Some(rendered.as_str()) {
                            refresh_reporter.record(Some(view.file_digest()), &effective);
                        } else {
                            let applied = refresh_table
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .apply_view(&effective);
                            match applied {
                                Ok(outcome) => {
                                    refresh_reporter.record(Some(view.file_digest()), &effective);
                                    last_applied = Some(rendered);
                                    info!(
                                        service = service_name(),
                                        version = SERVICE_VERSION,
                                        approvals = effective.len(),
                                        added_generations = outcome.added.len(),
                                        removed_generations = outcome.removed.len(),
                                        "Shared fleet approvals adopted: deployment pool generations updated"
                                    );
                                }
                                Err(error) => {
                                    warn!(error = %error, "Shared fleet approvals view rejected; retaining previous approval state");
                                }
                            }
                        }
                        effective
                    }
                    None => boot_approvals.clone(),
                };
                // Stop pollers for workers the view removed before adopting,
                // so a worker that moved endpoints is not a duplicate of its
                // own stale entry. Registry records of removed workers lapse
                // naturally; the eligibility gate already excludes them.
                let view_endpoints: BTreeSet<String> = effective
                    .iter()
                    .map(|approval| approval.endpoint.trim().to_string())
                    .collect();
                adopted.retain(|endpoint, worker: &mut AdoptedGatewayWorker| {
                    let keep = view_endpoints.contains(endpoint);
                    if !keep {
                        worker.poller.abort();
                        info!(endpoint = %endpoint, "Removed approval poller stopped");
                    }
                    keep
                });
                let pending = adopter.adopt_missing(&effective, &mut adopted).await;
                for (endpoint, error) in &pending {
                    debug!(
                        endpoint = %endpoint,
                        error = %format!("{error:#}"),
                        "Approved worker adoption not complete yet; will retry"
                    );
                }
                if view.is_none() && pending.is_empty() {
                    break;
                }
            }
        }));
    }

    let primary = dispatchers
        .next()
        .expect("the primary chat model always has a dispatcher");
    let state = dispatchers.fold(
        gateway::GatewayState::new(
            primary,
            enterprise_hooks,
            perimeter,
            config.serve.request_timeout_secs,
            args.gateway_max_in_flight,
        ),
        gateway::GatewayState::with_chat_model,
    );
    let state = state
        .with_approvals_reporter(approvals_reporter)
        .with_compat_profile(config.compat_profile);
    let state = if args.gateway_realtime == "on" {
        // Stage pools resolve per approved task for the public model; boot
        // fails closed only when no realtime stage can be served at all. A
        // missing individual stage refuses its admits with a policy close at
        // session time.
        let asr_deployment_id = {
            let table = deployment_table
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            table
                .select(TaskKind::SpeechToText, &public_model)
                .map(|deployment| deployment.deployment_id().clone())
        };
        let tts_deployment_id = {
            let table = deployment_table
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            table
                .select(TaskKind::TextToSpeech, &public_model)
                .map(|deployment| deployment.deployment_id().clone())
        };
        if asr_deployment_id.is_none() && tts_deployment_id.is_none() {
            return Err(anyhow::anyhow!(
                "--gateway-realtime is on but no approved worker advertises a realtime speech_to_text or text_to_speech model {public_model}"
            ));
        }
        let relay = app::realtime_relay::GatewayRealtimeRelay::new(
            registry,
            app::realtime_relay::RealtimeRelayConfig {
                deployment_id: asr_deployment_id,
                tts_deployment_id,
                public_model: public_model.clone(),
                policy_revision: policy_revision.clone(),
                backend_policy: gateway_backend_policy(config.backend.as_ref()),
                max_sessions: args.gateway_realtime_max_sessions,
                session_budget: Duration::from_millis(args.gateway_realtime_session_budget_ms),
            },
            credentials,
            client_config.tls.clone(),
        )
        .map_err(|error| anyhow::anyhow!(error))?;
        info!(
            service = service_name(),
            version = SERVICE_VERSION,
            asr_stage = relay.config().deployment_id.is_some(),
            tts_stage = relay.config().tts_deployment_id.is_some(),
            max_sessions = args.gateway_realtime_max_sessions,
            "Gateway realtime relay enabled for izwi-realtime-v1 sessions"
        );
        state.with_realtime_relay(relay)
    } else {
        state
    };
    let poller = GatewayWorkerStatusPoller {
        poller_count: boot_poller_count,
        tasks,
    };
    info!(
        service = service_name(),
        version = SERVICE_VERSION,
        pollers = poller.poller_count,
        interval_ms = polling_interval.as_millis() as u64,
        "Gateway worker status polling started"
    );
    Ok((state, poller))
}

pub fn configured_gateway_worker_approvals(
    args: &GatewayArgs,
    legacy_public_model: &ModelAlias,
) -> anyhow::Result<(
    Vec<gateway_deployments::GatewayWorkerApproval>,
    Vec<gateway_deployments::GatewayWorkerApproval>,
    Option<gateway_shared_approvals::SharedApprovalsView>,
)> {
    // Returns (CLI-only approvals, effective merged view, shared file view).
    // `--worker-endpoint` is the one-worker form of the legacy endpoint list.
    let legacy_endpoints: Vec<String> = args
        .gateway_worker_endpoints
        .iter()
        .chain(args.worker_endpoint.as_ref())
        .cloned()
        .collect();
    let mut merged = if !args.gateway_worker_approvals.is_empty() {
        if !legacy_endpoints.is_empty() {
            anyhow::bail!(
                "--gateway-worker-approval cannot be combined with --gateway-worker-endpoint"
            );
        }
        if args.worker_deployment.is_some() || args.worker_model_generation.is_some() {
            anyhow::bail!(
                "--worker-deployment and --worker-model-generation apply only to legacy --gateway-worker-endpoint configuration"
            );
        }
        args.gateway_worker_approvals.clone()
    } else if !legacy_endpoints.is_empty() {
        let deployment_id = DeploymentId::new(required_gateway_value(
            &args.worker_deployment,
            "--worker-deployment",
        )?)?;
        let model_generation =
            ModelGeneration::new(args.worker_model_generation.ok_or_else(|| {
                anyhow::anyhow!("gateway mode requires --worker-model-generation")
            })?)?;
        legacy_endpoints
            .into_iter()
            .map(|endpoint| gateway_deployments::GatewayWorkerApproval {
                endpoint,
                identity: gateway_deployments::GatewayWorkerApprovalIdentity::DiscoverFromAuthenticatedEndpoint,
                task: TaskKind::Chat,
                public_model: legacy_public_model.clone(),
                deployment_id: deployment_id.clone(),
                model_generation,
            })
            .collect()
    } else {
        // No CLI approvals and no legacy endpoints: the shared approvals
        // file (the DS6 rollout channel) may be the sole approval source.
        Vec::new()
    };

    // Multi-gateway fleets keep one authoritative approvals file that every
    // gateway reads, so all gateways approve the same worker set. Shared
    // entries augment (never replace) explicit CLI approvals; the existing
    // duplicate-endpoint check below rejects any conflict fail-closed.
    let cli_only = merged.clone();
    let shared_view = match gateway_shared_approvals::SharedApprovalsConfig::from_env() {
        Ok(Some(config)) => {
            let view = gateway_shared_approvals::SharedApprovalsView::load(config)
                .map_err(|error| anyhow::anyhow!("shared fleet approvals: {error}"))?;
            merged.extend(view.approvals().iter().cloned());
            Some(view)
        }
        Ok(None) => None,
        Err(error) => anyhow::bail!("shared fleet approvals: {error}"),
    };
    Ok((cli_only, merged, shared_view))
}

pub fn validate_gateway_topology_policy(
    topology: GatewayTopology,
    approvals: &[gateway_deployments::GatewayWorkerApproval],
    tls: &izwi_serving_client::WorkerClientTlsConfig,
) -> anyhow::Result<()> {
    let mut pinned_worker_ids = BTreeSet::new();
    for approval in approvals {
        if let Some((_node_id, worker_id)) = approval.pinned_identity() {
            if !pinned_worker_ids.insert(worker_id.clone()) {
                anyhow::bail!("duplicate operator-approved logical worker identity: {worker_id}");
            }
        }
    }
    if topology == GatewayTopology::Standalone {
        return Ok(());
    }
    if approvals.is_empty() {
        anyhow::bail!("fleet-one-gateway topology requires at least one worker approval");
    }
    if !tls.has_client_identity() {
        anyhow::bail!(
            "fleet-one-gateway topology requires a client certificate and private key for mutual TLS"
        );
    }
    if approvals
        .iter()
        .any(|approval| approval.pinned_identity().is_none())
    {
        anyhow::bail!(
            "fleet-one-gateway topology requires versioned v1 approvals with pinned node and worker identities"
        );
    }
    Ok(())
}

pub fn validate_gateway_topology_source(args: &GatewayArgs) -> anyhow::Result<()> {
    if args.gateway_topology == GatewayTopology::FleetOneGateway
        && args.gateway_worker_approvals.is_empty()
    {
        anyhow::bail!(
            "fleet-one-gateway topology requires versioned --gateway-worker-approval entries; pinned and legacy endpoint modes are standalone-only"
        );
    }
    Ok(())
}

fn required_gateway_value<'a>(value: &'a Option<String>, name: &str) -> anyhow::Result<&'a str> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("gateway mode requires {name}"))
}

fn gateway_worker_credentials(args: &GatewayArgs) -> anyhow::Result<ServiceCredentials> {
    Ok(ServiceCredentials {
        credential_id: CredentialId::new(required_gateway_value(
            &args.worker_credential_id,
            "--worker-credential-id",
        )?)?,
        bearer_token: ServiceBearerToken::new(required_gateway_value(
            &args.worker_bearer_token,
            "--worker-bearer-token or IZWI_GATEWAY_WORKER_BEARER_TOKEN",
        )?)?,
    })
}

fn gateway_backend_policy(backend: Option<&BackendArg>) -> worker_registry::BackendPolicy {
    match backend {
        Some(BackendArg::Cpu) => worker_registry::BackendPolicy::CPU_ONLY,
        Some(BackendArg::Metal) => worker_registry::BackendPolicy::METAL_ONLY,
        Some(BackendArg::Cuda) => worker_registry::BackendPolicy::CUDA_ONLY,
        Some(BackendArg::Auto) | None => worker_registry::BackendPolicy::ANY,
    }
}

pub fn validate_gateway_limits(args: &GatewayArgs) -> anyhow::Result<()> {
    if args.gateway_max_in_flight == 0
        || args.gateway_max_in_flight > tokio::sync::Semaphore::MAX_PERMITS
    {
        anyhow::bail!(
            "--gateway-max-in-flight must be between 1 and {}",
            tokio::sync::Semaphore::MAX_PERMITS
        );
    }
    if args.gateway_worker_queue_wait_ms == 0 {
        anyhow::bail!("--gateway-worker-queue-wait-ms must be non-zero");
    }
    let admission_timeout = Duration::from_millis(args.gateway_worker_admission_timeout_ms);
    if admission_timeout.is_zero() || admission_timeout > MAX_GATEWAY_ADMISSION_TIMEOUT {
        anyhow::bail!(
            "--gateway-worker-admission-timeout-ms must be between 1 and {}",
            MAX_GATEWAY_ADMISSION_TIMEOUT.as_millis()
        );
    }
    if Duration::from_millis(args.gateway_worker_queue_wait_ms) > admission_timeout {
        anyhow::bail!(
            "--gateway-worker-queue-wait-ms must not exceed the worker admission timeout"
        );
    }
    for (name, value) in [
        (
            "--gateway-worker-first-output-timeout-ms",
            args.gateway_worker_first_output_timeout_ms,
        ),
        (
            "--gateway-worker-progress-idle-timeout-ms",
            args.gateway_worker_progress_idle_timeout_ms,
        ),
    ] {
        let timeout = Duration::from_millis(value);
        if timeout.is_zero() || timeout > MAX_GATEWAY_STREAM_PHASE_TIMEOUT {
            anyhow::bail!(
                "{name} must be between 1 and {}",
                MAX_GATEWAY_STREAM_PHASE_TIMEOUT.as_millis()
            );
        }
    }
    if args.gateway_shutdown_budget_secs == 0
        || args.gateway_shutdown_budget_secs > MAX_GATEWAY_SHUTDOWN_BUDGET_SECS
        || args.gateway_prestop_secs >= args.gateway_shutdown_budget_secs
    {
        anyhow::bail!(
            "--gateway-shutdown-budget-secs must be between 1 and {MAX_GATEWAY_SHUTDOWN_BUDGET_SECS} and longer than --gateway-prestop-secs"
        );
    }
    let body_read_timeout = Duration::from_millis(args.gateway_body_read_timeout_ms);
    if body_read_timeout.is_zero() || body_read_timeout > MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT {
        anyhow::bail!(
            "--gateway-body-read-timeout-ms must be between 1 and {}",
            MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT.as_millis()
        );
    }
    if Duration::from_millis(args.gateway_worker_first_output_timeout_per_kib_ms)
        > MAX_GATEWAY_STREAM_PHASE_TIMEOUT
    {
        anyhow::bail!(
            "--gateway-worker-first-output-timeout-per-kib-ms must be at most {}",
            MAX_GATEWAY_STREAM_PHASE_TIMEOUT.as_millis()
        );
    }
    let slow_consumer_timeout = Duration::from_millis(args.gateway_slow_consumer_timeout_ms);
    if slow_consumer_timeout.is_zero() || slow_consumer_timeout > MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT
    {
        anyhow::bail!(
            "--gateway-slow-consumer-timeout-ms must be between 1 and {}",
            MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT.as_millis()
        );
    }
    let configured_workers = args
        .gateway_worker_endpoints
        .len()
        .checked_add(args.gateway_worker_approvals.len())
        .ok_or_else(|| anyhow::anyhow!("configured gateway worker count overflowed"))?;
    if configured_workers > MAX_CONFIGURED_GATEWAY_WORKERS {
        anyhow::bail!(
            "at most {MAX_CONFIGURED_GATEWAY_WORKERS} gateway worker endpoints or approvals are supported"
        );
    }
    let ttl = Duration::from_millis(args.gateway_worker_status_ttl_ms);
    let poll = Duration::from_millis(args.gateway_worker_status_poll_ms);
    if ttl.is_zero() || ttl > MAX_GATEWAY_STATUS_TTL {
        anyhow::bail!("--gateway-worker-status-ttl-ms is outside the supported range");
    }
    validate_status_cadence(ttl, poll)?;
    validate_router_args(args)?;
    validate_realtime_relay_args(args)?;
    Ok(())
}

/// Parses and bounds-checks the routing knobs. Fail-closed on any
/// out-of-range or non-`on|off` value so typos never silently change routing.
fn validate_router_args(args: &GatewayArgs) -> anyhow::Result<()> {
    match args.gateway_router_prefix_affinity.as_str() {
        "on" | "off" => {}
        other => {
            anyhow::bail!("--gateway-router-prefix-affinity must be `on` or `off`, got `{other}`")
        }
    }
    let usage = args.gateway_router_kv_pressure_pct;
    if usage.is_nan() || usage <= 0.0 || usage > 100.0 {
        anyhow::bail!("--gateway-router-kv-pressure-pct must be greater than 0 and at most 100");
    }
    worker_registry::RoutingConfig {
        prefix_balance_threshold: args.gateway_router_balance_threshold,
        kv_pressure_pct: usage,
    }
    .validate()
    .map_err(|error| anyhow::anyhow!("--gateway-router-balance-threshold: {error}"))?;
    app::prefix_router::PrefixAffinityConfig {
        block_bytes: args.gateway_router_prefix_block_bytes,
        max_blocks: args.gateway_router_prefix_max_blocks,
        max_entries_per_worker: args.gateway_router_prefix_entries,
    }
    .validate()
    .map_err(|error| anyhow::anyhow!(error.message))?;
    Ok(())
}

/// Parses and bounds-checks the realtime relay knobs (DS3.6).
fn validate_realtime_relay_args(args: &GatewayArgs) -> anyhow::Result<()> {
    match args.gateway_realtime.as_str() {
        "on" | "off" => {}
        other => anyhow::bail!("--gateway-realtime must be `on` or `off`, got `{other}`"),
    }
    if args.gateway_realtime == "off" {
        return Ok(());
    }
    if args.gateway_realtime_max_sessions == 0
        || args.gateway_realtime_max_sessions > app::realtime_relay::MAX_RELAY_SESSIONS
    {
        anyhow::bail!(
            "--gateway-realtime-max-sessions must be between 1 and {}",
            app::realtime_relay::MAX_RELAY_SESSIONS
        );
    }
    let budget = Duration::from_millis(args.gateway_realtime_session_budget_ms);
    if budget.is_zero() || budget > Duration::from_millis(MAX_REMAINING_TIME_MS) {
        anyhow::bail!(
            "--gateway-realtime-session-budget-ms must be between 1 and {MAX_REMAINING_TIME_MS}"
        );
    }
    Ok(())
}

fn prefix_affinity_config(args: &GatewayArgs) -> Option<app::prefix_router::PrefixAffinityConfig> {
    (args.gateway_router_prefix_affinity == "on").then_some(
        app::prefix_router::PrefixAffinityConfig {
            block_bytes: args.gateway_router_prefix_block_bytes,
            max_blocks: args.gateway_router_prefix_max_blocks,
            max_entries_per_worker: args.gateway_router_prefix_entries,
        },
    )
}

pub fn gateway_worker_client_config(
    args: &GatewayArgs,
    tls: izwi_serving_client::WorkerClientTlsConfig,
) -> WorkerClientConfig {
    WorkerClientConfig {
        max_in_flight: args.gateway_max_in_flight,
        request_timeout: Duration::from_millis(args.gateway_worker_admission_timeout_ms),
        first_output_timeout: Duration::from_millis(args.gateway_worker_first_output_timeout_ms),
        first_output_timeout_per_kib: Duration::from_millis(
            args.gateway_worker_first_output_timeout_per_kib_ms,
        ),
        progress_timeout: Duration::from_millis(args.gateway_worker_progress_idle_timeout_ms),
        ndjson_limits: NdjsonLimits {
            max_line_bytes: GATEWAY_NDJSON_MAX_LINE_BYTES,
            max_total_bytes: GATEWAY_NDJSON_MAX_TOTAL_BYTES,
            max_events: GATEWAY_NDJSON_MAX_EVENTS,
            ..NdjsonLimits::default()
        },
        tls,
        ..WorkerClientConfig::default()
    }
}

#[cfg(test)]
#[path = "gateway_tests.rs"]
mod tests;
