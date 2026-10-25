//! A pool of execution units that admits waiters in weighted-fair order
//! across flows instead of first come, first served.
//!
//! Physical admission sits in front of the device thread with one unit per
//! execution group, so the device's arbiter only ever holds one model job
//! and never chooses between lanes. This pool applies the arbiter's
//! start-time fair queueing at admission instead: each flow (a loaded model)
//! is charged for how long it holds its units divided by its weight, and the
//! waiting flow with the least charge goes next. A chat model stepping in
//! short slices then gets many turns while a long transcription window holds
//! the device once, rather than strictly alternating with it. Waiters of one
//! flow keep their order, and a waiter that cannot fit blocks later ones, so
//! a wide request is not starved by narrow ones.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use tokio::sync::oneshot;

/// Weighted hold time is kept in nanoseconds scaled by this factor, as the
/// device arbiter does, so integer division by a weight keeps resolution.
const VIRTUAL_TIME_SCALE: u128 = 840;

#[derive(Debug)]
pub(crate) struct FairPermits {
    state: Mutex<FairState>,
}

#[derive(Debug)]
struct FairState {
    available: usize,
    closed: bool,
    flows: HashMap<u64, FairFlow>,
    /// Charge of the last flow granted units. A flow that starts waiting
    /// starts here, so idle time is not banked as credit.
    virtual_time: u128,
    next_order: u64,
}

#[derive(Debug)]
struct FairFlow {
    served: u128,
    waiting: VecDeque<Waiter>,
}

#[derive(Debug)]
struct Waiter {
    order: u64,
    units: usize,
    grant: oneshot::Sender<()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FairAcquireError {
    Closed,
    Timeout,
}

/// Units held from a [`FairPermits`] pool. Dropping it returns them and
/// charges the flow for the time they were held.
#[derive(Debug)]
pub(crate) struct FairPermit {
    permits: Arc<FairPermits>,
    units: usize,
    flow: u64,
    weight: u64,
    granted: Instant,
}

impl Drop for FairPermit {
    fn drop(&mut self) {
        let charge =
            self.granted.elapsed().as_nanos() * VIRTUAL_TIME_SCALE / u128::from(self.weight);
        let mut state = self.permits.lock();
        state.available += self.units;
        let virtual_time = state.virtual_time;
        let flow = state.flows.entry(self.flow).or_insert_with(|| FairFlow {
            served: virtual_time,
            waiting: VecDeque::new(),
        });
        flow.served = flow.served.saturating_add(charge);
        state.dispatch();
    }
}

impl FairState {
    /// Grants units to waiters, least-charged flow first, while the next
    /// one fits.
    fn dispatch(&mut self) {
        let virtual_time = self.virtual_time;
        // A drained flow with no debt left carries nothing a new flow at
        // `virtual_time` would not.
        self.flows
            .retain(|_, flow| !flow.waiting.is_empty() || flow.served > virtual_time);
        loop {
            let Some((&id, flow)) = self
                .flows
                .iter()
                .filter(|(_, flow)| !flow.waiting.is_empty())
                .min_by_key(|(_, flow)| (flow.served, flow.waiting[0].order))
            else {
                return;
            };
            let (units, served) = (flow.waiting[0].units, flow.served);
            if units > self.available {
                return;
            }
            let waiter = self
                .flows
                .get_mut(&id)
                .and_then(|flow| flow.waiting.pop_front())
                .expect("the chosen flow has a waiter");
            self.available -= units;
            self.virtual_time = self.virtual_time.max(served);
            if waiter.grant.send(()).is_err() {
                self.available += units;
            }
        }
    }

    fn remove_waiter(&mut self, flow: u64, order: u64) -> bool {
        let Some(waiting) = self.flows.get_mut(&flow).map(|flow| &mut flow.waiting) else {
            return false;
        };
        let Some(position) = waiting.iter().position(|waiter| waiter.order == order) else {
            return false;
        };
        waiting.remove(position);
        true
    }
}

impl FairPermits {
    pub(crate) fn new(units: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(FairState {
                available: units,
                closed: false,
                flows: HashMap::new(),
                virtual_time: 0,
                next_order: 0,
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, FairState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Refuses every waiter and every later acquire.
    pub(crate) fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        for flow in state.flows.values_mut() {
            flow.waiting.clear();
        }
    }

    /// Waits until `flow` may take `units`, or until `deadline`. A caller
    /// that stops waiting leaves the queue, and units granted to it as it
    /// left are returned.
    pub(crate) async fn acquire(
        self: &Arc<Self>,
        units: usize,
        flow: u64,
        weight: u64,
        deadline: Option<Instant>,
    ) -> Result<FairPermit, FairAcquireError> {
        let weight = weight.max(1);
        let permit = || FairPermit {
            permits: Arc::clone(self),
            units,
            flow,
            weight,
            granted: Instant::now(),
        };
        let (order, receiver) = {
            let mut state = self.lock();
            if state.closed {
                return Err(FairAcquireError::Closed);
            }
            let virtual_time = state.virtual_time;
            let anyone_waiting = state.flows.values().any(|flow| !flow.waiting.is_empty());
            let entry = state.flows.entry(flow).or_insert_with(|| FairFlow {
                served: virtual_time,
                waiting: VecDeque::new(),
            });
            if entry.waiting.is_empty() {
                entry.served = entry.served.max(virtual_time);
            }
            let served = entry.served;
            if !anyone_waiting && state.available >= units {
                state.available -= units;
                state.virtual_time = state.virtual_time.max(served);
                return Ok(permit());
            }
            let order = state.next_order;
            state.next_order += 1;
            let (grant, receiver) = oneshot::channel();
            state
                .flows
                .get_mut(&flow)
                .expect("inserted above")
                .waiting
                .push_back(Waiter {
                    order,
                    units,
                    grant,
                });
            (order, receiver)
        };
        let mut waiting = Waiting {
            permits: self,
            flow,
            order,
            units,
            receiver,
            settled: false,
        };
        let granted = match deadline {
            Some(deadline) => {
                match tokio::time::timeout_at(deadline.into(), &mut waiting.receiver).await {
                    Ok(granted) => granted,
                    Err(_) => return Err(FairAcquireError::Timeout),
                }
            }
            None => (&mut waiting.receiver).await,
        };
        waiting.settled = true;
        match granted {
            Ok(()) => Ok(permit()),
            Err(_) => Err(FairAcquireError::Closed),
        }
    }
}

/// A queued acquire. If it is dropped before it settles, the waiter leaves
/// the queue, or, when its grant raced the drop, the units go back.
struct Waiting<'a> {
    permits: &'a Arc<FairPermits>,
    flow: u64,
    order: u64,
    units: usize,
    receiver: oneshot::Receiver<()>,
    settled: bool,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        let mut state = self.permits.lock();
        if state.remove_waiter(self.flow, self.order) {
            state.dispatch();
            return;
        }
        if self.receiver.try_recv().is_ok() {
            state.available += self.units;
            state.dispatch();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FairAcquireError, FairPermits};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Spawns an acquire that records `label` when granted and then holds
    /// its units for `hold`.
    fn spawn_acquire(
        permits: &Arc<FairPermits>,
        flow: u64,
        weight: u64,
        hold: Duration,
        label: &'static str,
        grants: &Arc<Mutex<Vec<&'static str>>>,
    ) -> tokio::task::JoinHandle<()> {
        let permits = Arc::clone(permits);
        let grants = Arc::clone(grants);
        tokio::spawn(async move {
            let permit = permits
                .acquire(1, flow, weight, None)
                .await
                .expect("granted");
            grants.lock().unwrap().push(label);
            tokio::time::sleep(hold).await;
            drop(permit);
        })
    }

    async fn settle() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    /// A flow that just held the units long goes behind a flow that has not
    /// had a turn, even though its second waiter arrived first.
    #[tokio::test]
    async fn a_flow_charged_for_a_long_hold_yields_to_a_waiting_flow() {
        let permits = FairPermits::new(1);
        let grants = Arc::new(Mutex::new(Vec::new()));
        let long = spawn_acquire(&permits, 1, 1, Duration::from_millis(40), "a1", &grants);
        settle().await;
        let a2 = spawn_acquire(&permits, 1, 1, Duration::ZERO, "a2", &grants);
        settle().await;
        let b1 = spawn_acquire(&permits, 2, 1, Duration::ZERO, "b1", &grants);
        for task in [long, a2, b1] {
            task.await.unwrap();
        }
        assert_eq!(*grants.lock().unwrap(), ["a1", "b1", "a2"]);
    }

    /// With equal holds, a flow of weight 4 is served about four times as
    /// often as a flow of weight 1 while both keep waiting.
    #[tokio::test]
    async fn weights_set_each_flows_share_of_turns() {
        let permits = FairPermits::new(1);
        let grants = Arc::new(Mutex::new(Vec::new()));
        let gate = permits.acquire(1, 99, 1, None).await.unwrap();
        let mut tasks = Vec::new();
        for _ in 0..12 {
            tasks.push(spawn_acquire(
                &permits,
                1,
                4,
                Duration::from_millis(10),
                "heavy",
                &grants,
            ));
            tasks.push(spawn_acquire(
                &permits,
                2,
                1,
                Duration::from_millis(10),
                "light",
                &grants,
            ));
        }
        settle().await;
        drop(gate);
        for task in tasks {
            task.await.unwrap();
        }
        let grants = grants.lock().unwrap();
        let heavy_in_first_ten = grants[..10]
            .iter()
            .filter(|label| **label == "heavy")
            .count();
        assert!(heavy_in_first_ten >= 7, "{grants:?}");
    }

    /// Waiters of one flow keep their arrival order.
    #[tokio::test]
    async fn one_flows_waiters_are_served_in_order() {
        let permits = FairPermits::new(1);
        let grants = Arc::new(Mutex::new(Vec::new()));
        let gate = permits.acquire(1, 1, 1, None).await.unwrap();
        let mut tasks = Vec::new();
        for label in ["first", "second", "third"] {
            tasks.push(spawn_acquire(
                &permits,
                1,
                1,
                Duration::ZERO,
                label,
                &grants,
            ));
            settle().await;
        }
        drop(gate);
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(*grants.lock().unwrap(), ["first", "second", "third"]);
    }

    /// A waiter that times out or is dropped leaves the queue, and the units
    /// reach the next waiter.
    #[tokio::test]
    async fn abandoned_waiters_leave_and_return_nothing_twice() {
        let permits = FairPermits::new(1);
        let held = permits.acquire(1, 1, 1, None).await.unwrap();
        let expired = permits
            .acquire(1, 2, 1, Some(Instant::now() + Duration::from_millis(5)))
            .await;
        assert_eq!(expired.err(), Some(FairAcquireError::Timeout));
        let dropped = {
            let permits = Arc::clone(&permits);
            tokio::spawn(async move { permits.acquire(1, 3, 1, None).await })
        };
        settle().await;
        dropped.abort();
        let _ = dropped.await;
        let waiting = {
            let permits = Arc::clone(&permits);
            tokio::spawn(async move { permits.acquire(1, 4, 1, None).await.map(|_| ()) })
        };
        settle().await;
        drop(held);
        waiting
            .await
            .unwrap()
            .expect("the live waiter gets the unit");
        let again = permits
            .acquire(1, 5, 1, Some(Instant::now() + Duration::from_millis(50)))
            .await;
        assert!(again.is_ok(), "exactly one unit is back in the pool");
    }

    #[tokio::test]
    async fn closing_refuses_waiters_and_later_acquires() {
        let permits = FairPermits::new(1);
        let held = permits.acquire(1, 1, 1, None).await.unwrap();
        let waiting = {
            let permits = Arc::clone(&permits);
            tokio::spawn(async move { permits.acquire(1, 2, 1, None).await.map(|_| ()) })
        };
        settle().await;
        permits.close();
        assert_eq!(waiting.await.unwrap(), Err(FairAcquireError::Closed));
        assert_eq!(
            permits.acquire(1, 3, 1, None).await.err(),
            Some(FairAcquireError::Closed)
        );
        drop(held);
    }
}
