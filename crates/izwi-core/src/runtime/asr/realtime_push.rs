//! Realtime ASR pushes on the direct path, batched across streams. A push
//! that arrives while a batch runs waits, and the next batch carries every
//! waiting push of the same model through one device job and one model call
//! (Nemotron's streaming cohort). An idle stream's push starts a batch of
//! one at once, so batching adds no latency when there is nothing to batch.

use super::*;

use tokio::sync::{oneshot, OwnedMutexGuard};

/// What a push keeps alive until its device work ends, even if the caller
/// stops waiting: the stream's ordering, the model's residency, the session
/// quota and the stream's operation count.
pub(super) struct RealtimePushHolds {
    pub(super) _ordering: OwnedMutexGuard<()>,
    pub(super) _residency: ModelResidencyLease,
    pub(super) _session: Arc<RealtimeAsrSessionLease>,
    pub(super) _operation: RealtimeAsrOperationGuard,
}

/// One stream's push, waiting for a batch.
pub(super) struct RealtimePush {
    pub(super) model: Arc<NativeAsrModel>,
    pub(super) contract: LoadedExecutionContract,
    pub(super) job: JobLease,
    pub(super) state: Arc<StdMutex<RuntimeAsrRealtimeState>>,
    pub(super) samples: Vec<f32>,
    pub(super) sample_rate: u32,
    pub(super) _holds: RealtimePushHolds,
}

type PushReply = oneshot::Sender<Result<Vec<NativeAsrRealtimeEvent>>>;

/// Pushes waiting for the next batch, and whether a batch task is running.
pub(super) struct RealtimePushBatcher<R> {
    queue: StdMutex<PushQueue<R>>,
}

struct PushQueue<R> {
    waiting: Vec<R>,
    draining: bool,
}

impl<R> Default for RealtimePushBatcher<R> {
    fn default() -> Self {
        Self {
            queue: StdMutex::new(PushQueue {
                waiting: Vec::new(),
                draining: false,
            }),
        }
    }
}

impl<R: Send + 'static> RealtimePushBatcher<R> {
    /// Queues `row`. When no batch task is running, starts one that runs
    /// `run_batch` on everything waiting, again and again, until nothing
    /// waits. `run_batch` must answer every row it is given.
    pub(super) fn submit<F, Fut>(self: &Arc<Self>, row: R, run_batch: F)
    where
        F: Fn(Vec<R>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        {
            let mut queue = self
                .queue
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            queue.waiting.push(row);
            if queue.draining {
                return;
            }
            queue.draining = true;
        }
        let batcher = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                let batch = {
                    let mut queue = batcher
                        .queue
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    if queue.waiting.is_empty() {
                        queue.draining = false;
                        return;
                    }
                    std::mem::take(&mut queue.waiting)
                };
                run_batch(batch).await;
            }
        });
    }
}

impl RuntimeService {
    /// Pushes `push`'s audio with any other streams' pushes waiting for the
    /// same model, and returns this stream's events.
    pub(super) async fn push_realtime_batched(
        &self,
        push: RealtimePush,
    ) -> Result<Vec<NativeAsrRealtimeEvent>> {
        let (reply, events) = oneshot::channel();
        let coordinator = Arc::clone(&self.coordinator);
        self.asr_realtime_sessions
            .pushes
            .submit((push, reply), move |rows| {
                run_realtime_push_batch(Arc::clone(&coordinator), rows)
            });
        events.await.map_err(|_| {
            Error::InferenceError("realtime ASR push batch stopped before replying".into())
        })?
    }
}

pub(super) type RealtimePushRow = (RealtimePush, PushReply);

/// Runs the waiting pushes as one device job per loaded model and answers
/// each of them.
async fn run_realtime_push_batch(
    coordinator: Arc<InferenceCoordinator>,
    rows: Vec<RealtimePushRow>,
) {
    let mut groups: Vec<Vec<RealtimePushRow>> = Vec::new();
    for row in rows {
        match groups
            .iter_mut()
            .find(|group| Arc::ptr_eq(&group[0].0.model, &row.0.model))
        {
            Some(group) => group.push(row),
            None => groups.push(vec![row]),
        }
    }
    for group in groups {
        let (pushes, replies): (Vec<_>, Vec<_>) = group.into_iter().unzip();
        let job = pushes[0].job.clone();
        let contract = pushes[0].contract.clone();
        tracing::debug!(rows = pushes.len(), "realtime ASR push batch");
        let result = coordinator
            .run_loaded_blocking_stage(
                &job,
                contract,
                WorkUnit::AtomicJob {
                    kind: "asr.realtime.push".to_string(),
                },
                move || Ok(push_rows(pushes)),
            )
            .await;
        match result {
            Ok(outcomes) => {
                for (reply, outcome) in replies.into_iter().zip(outcomes) {
                    let _ = reply.send(outcome);
                }
            }
            Err(error) => {
                for reply in replies {
                    let _ = reply.send(Err(error.replicate()));
                }
            }
        }
    }
}

/// The device work for one group: every row in one model call, or each row
/// alone when the batch failed without changing any row, so one stream's
/// failure does not end the others.
fn push_rows(pushes: Vec<RealtimePush>) -> Vec<Result<Vec<NativeAsrRealtimeEvent>>> {
    let model = Arc::clone(&pushes[0].model);
    let mut guards = Vec::with_capacity(pushes.len());
    for push in &pushes {
        match push.state.lock() {
            Ok(guard) => guards.push(guard),
            Err(_) => {
                return pushes
                    .iter()
                    .map(|_| {
                        Err(Error::InferenceError(
                            "realtime ASR state mutex poisoned".into(),
                        ))
                    })
                    .collect();
            }
        }
    }
    // Streaming internals replace and release model-owned tensors while
    // decoding. Restore each row's full pending claim first so physical frees
    // cannot be observed as unclaimed headroom; a row that cannot is not run.
    let mut outcomes = pushes
        .iter()
        .map(|push| {
            push.job
                .prepare_materialized_release(JobResourceObservation::default())
                .err()
                .map(Err)
        })
        .collect::<Vec<_>>();
    let results = {
        let mut live = pushes
            .iter()
            .zip(guards.iter_mut())
            .zip(&outcomes)
            .filter(|(_, outcome)| outcome.is_none())
            .map(|((push, guard), _)| (push, &mut **guard))
            .collect::<Vec<_>>();
        if live.len() > 1 {
            push_together(&model, &mut live)
        } else {
            push_each_alone(&model, &mut live)
        }
    };
    let mut results = results.into_iter();
    for outcome in outcomes.iter_mut().filter(|outcome| outcome.is_none()) {
        *outcome = results.next();
    }
    pushes
        .iter()
        .zip(&guards)
        .zip(outcomes)
        .map(|((push, state), outcome)| {
            let events = outcome.unwrap_or_else(|| {
                Err(Error::InferenceError(
                    "realtime ASR push batch lost a row".into(),
                ))
            })?;
            let observation =
                realtime_state_with_input_observation(&state.native, push.samples.capacity())?;
            push.job.record_materialized_usage(observation)?;
            push.job
                .prepare_materialized_release(realtime_state_observation(&state.native)?)?;
            Ok(events)
        })
        .collect()
}

/// Every row in one model call. A failure that changed no row retries each
/// row alone; one that committed some rows fails them all.
fn push_together(
    model: &NativeAsrModel,
    rows: &mut [(&RealtimePush, &mut RuntimeAsrRealtimeState)],
) -> Vec<Result<Vec<NativeAsrRealtimeEvent>>> {
    let revisions = rows
        .iter()
        .map(|(_, state)| state.physical.revision)
        .collect::<Vec<_>>();
    let batched = {
        let inputs = rows
            .iter()
            .map(|(push, _)| (push.samples.as_slice(), push.sample_rate))
            .collect::<Vec<_>>();
        let mut states = rows
            .iter_mut()
            .map(|(_, state)| &mut **state)
            .collect::<Vec<_>>();
        RuntimeAsrRealtimeState::transact_batch(&mut states, model, |natives| {
            let mut batch = natives
                .iter_mut()
                .zip(&inputs)
                .map(|(native, &(samples, sample_rate))| (&mut **native, samples, sample_rate))
                .collect::<Vec<_>>();
            model.push_realtime_stream_samples_batch(&mut batch)
        })
    };
    match batched {
        Ok(events) => events.into_iter().map(Ok).collect(),
        Err(error) => {
            let untouched = rows
                .iter()
                .zip(&revisions)
                .all(|((_, state), &revision)| state.physical.revision == revision);
            if !untouched {
                return rows.iter().map(|_| Err(error.replicate())).collect();
            }
            tracing::warn!(
                error = %error,
                rows = rows.len(),
                "realtime ASR push batch failed; retrying each stream alone"
            );
            push_each_alone(model, rows)
        }
    }
}

fn push_each_alone(
    model: &NativeAsrModel,
    rows: &mut [(&RealtimePush, &mut RuntimeAsrRealtimeState)],
) -> Vec<Result<Vec<NativeAsrRealtimeEvent>>> {
    rows.iter_mut()
        .map(|(push, state)| {
            state.transact(model, |native| {
                model.push_realtime_stream_samples(native, &push.samples, push.sample_rate)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::RealtimePushBatcher;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::sync::{oneshot, Notify};

    /// Rows that arrive while a batch runs wait and go out together in the
    /// next batch; an idle batcher runs a lone row at once.
    #[tokio::test]
    async fn rows_waiting_behind_a_running_batch_form_the_next_batch() {
        let batcher = Arc::new(RealtimePushBatcher::<(u32, oneshot::Sender<u32>)>::default());
        let batches = Arc::new(Mutex::new(Vec::<Vec<u32>>::new()));
        let release = Arc::new(Notify::new());
        let run = {
            let batches = Arc::clone(&batches);
            let release = Arc::clone(&release);
            move |rows: Vec<(u32, oneshot::Sender<u32>)>| {
                let batches = Arc::clone(&batches);
                let release = Arc::clone(&release);
                async move {
                    let ids = rows.iter().map(|(id, _)| *id).collect::<Vec<_>>();
                    let first = batches.lock().unwrap().is_empty();
                    batches.lock().unwrap().push(ids);
                    if first {
                        release.notified().await;
                    }
                    for (id, reply) in rows {
                        let _ = reply.send(id * 10);
                    }
                }
            }
        };

        let mut replies = Vec::new();
        for id in 1..=4 {
            let (reply, answer) = oneshot::channel();
            batcher.submit((id, reply), run.clone());
            replies.push(answer);
            if id == 1 {
                // Let the first batch start and block before the rest queue.
                while batches.lock().unwrap().is_empty() {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
        }
        release.notify_one();

        let mut answers = Vec::new();
        for answer in replies {
            answers.push(answer.await.expect("every row is answered"));
        }
        assert_eq!(answers, [10, 20, 30, 40]);
        assert_eq!(*batches.lock().unwrap(), [vec![1], vec![2, 3, 4]]);

        // The drain task stopped; the next row starts a new one.
        let (reply, answer) = oneshot::channel();
        batcher.submit((5, reply), run);
        assert_eq!(answer.await.unwrap(), 50);
        assert_eq!(batches.lock().unwrap().last().unwrap(), &vec![5]);
    }

    /// A caller that stops waiting does not stop its batch: the other rows
    /// in it are still answered.
    #[tokio::test]
    async fn a_dropped_caller_does_not_strand_the_rest_of_its_batch() {
        let batcher = Arc::new(RealtimePushBatcher::<(u32, oneshot::Sender<u32>)>::default());
        let run = |rows: Vec<(u32, oneshot::Sender<u32>)>| async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            for (id, reply) in rows {
                let _ = reply.send(id);
            }
        };
        let (first_reply, first) = oneshot::channel();
        let (second_reply, second) = oneshot::channel();
        batcher.submit((1, first_reply), run);
        batcher.submit((2, second_reply), run);
        drop(first);
        assert_eq!(second.await.unwrap(), 2);
    }
}
