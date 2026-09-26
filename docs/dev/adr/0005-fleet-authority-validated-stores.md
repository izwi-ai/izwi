# ADR 0005: The multi-gateway fleet profile is worker-arbitrated, store-validated, and explicitly MySQL-free

Status: Accepted
Date: 2026-09-26

## Context

DS5 of the distributed-serving plan turns the multi-gateway profile from
"controls exist, profile unsupported" into a validated configuration. The
building blocks already existed: a shared coordination database
(`IZWI_GATEWAY_FLEET_DB_PATH`) holding worker observations and short-lived
capacity claims, advisory claim steering in worker selection, worker-local
admission semaphores as the true capacity authority, and 1/N tenant-quota
partitioning (`IZWI_GATEWAY_FLEET_SIZE`/`_PARTITION`). The gaps were that the
coordination store could only be opened as a SQLite file, the PostgreSQL/MySQL
dialect SQL had never executed anywhere, the quota posture was implicit, and
no process-level evidence existed for cross-gateway admission, quota, crash
recovery, or the supervisor generation fence.

## Decision

1. **The worker stays the only atomic admission authority.** Fleet capacity
   claims are advisory load signals that steer selection; a gateway that loses
   the claim race proceeds with one bounded alternate dispatch and the worker's
   semaphore arbitrates. No admission decision ever depends on the coordination
   store being reachable (DINV-06's degradation half).
2. **Posture is explicit, never a mixture (DINV-06's quota half).** When the
   coordination database is set, the shared-atomic claim path is the default
   selection posture and the gateway logs it (`selection_mode =
   shared_atomic_claims`). Tenant quota is either partitioned 1/N
   (explicitly chosen via the partition env pair, logged as the fallback) or
   worker-authoritative per gateway — the two quota modes are named at
   startup, not inferred.
3. **PostgreSQL is the validated server-backed store.** The store layer opens
   bounded database URLs (`IZWI_DATABASE_URL` for the durable store, database
   URLs accepted by the fleet reference), the migrator translates its shared
   DDL per backend (epoch-millis INTEGER → BIGINT, REAL → DOUBLE PRECISION,
   SQLite-only `COLLATE NOCASE` becomes a `LOWER(name)` functional unique
   index), and the fleet claim is guarded by a worker-keyed transaction-scoped
   advisory lock on PostgreSQL — SQLite's single-writer lock does not exist
   there, and without the lock two concurrent claimers can both observe the
   same live-claim count and overspend a credit. Execution suites run against
   real PostgreSQL locally (Homebrew) and in CI (service container).
4. **MySQL stays written but unvalidated.** The dialect SQL exists, but a
   live MySQL 8 probe fails on the first table: `TEXT PRIMARY KEY` requires a
   key length (MySQL error 1170), and the same structural burden applies to
   every keyed TEXT column, TEXT defaults, prefix indexes, and partial
   indexes. Validating MySQL means maintaining a full MySQL schema variant;
   that is a separate decision with its own evidence gate, not a side effect
   of this phase.
5. **A validated fleet does not leak.** A bounded background sweep reaps
   expired claims and prunes observation rows stale beyond 24h; the claim TTL
   is operator-tunable (`IZWI_GATEWAY_FLEET_CLAIM_TTL_MS`, bounded
   100ms–300s, historical 30s default unchanged).
6. **Cross-process evidence is the acceptance bar.** The fleet rig proves
   T07/T20/P8.1/steering/crash+TTL/outage with two real gateway processes
   against a shared SQLite or PostgreSQL coordination database, and the
   supervisor generation fence is proven with real supervisor processes
   (T25). The rig's worker endpoints are in-process mock workers on real
   loopback TCP — no worker binary exists, and every property under test
   lives in the gateway processes and the shared store.

## Consequences

- The support matrix moves "multiple gateways" from *not supported* to
  *validated on SQLite (single host) and PostgreSQL (shared fleet)*, with
  circuit state and accepted-work ownership still explicitly per-gateway.
- Simultaneous fleet boot on one SQLite file races the journal-mode setup
  before the busy timeout applies; gateways retry the coordination-database
  open briefly before failing closed.
- The fleet coordination database remains a runtime dependency of steering
  and quota partitioning only — never of request admission. Deleting,
  closing, or corrupting it degrades selection precision, never correctness.
- A future MySQL lane must budget for a dedicated schema variant and its own
  conformance suite before any fleet claim is recorded.
