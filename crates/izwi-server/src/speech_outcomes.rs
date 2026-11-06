//! Outcomes of `/v1/audio/speech` requests, for `GET /v1/audio/speech/{id}`.
//!
//! Every speech request (a buffered file, a chunked `audio` stream, an SSE
//! stream, a batch item, a text-in socket utterance) records its status here
//! when it is admitted and its outcome when it ends: finish reason, usage,
//! audio duration and timings. A client that streamed audio can learn
//! afterwards whether the stream ended normally, as SGLang-Omni's lookup
//! allows for its raw PCM streams.
//!
//! The store is process-local and bounded (`IZWI_SPEECH_OUTCOME_CAPACITY`
//! entries, `IZWI_SPEECH_OUTCOME_TTL_SECS`): records are lost on restart and
//! the oldest are evicted first, so a missing id proves nothing. No audio is
//! kept.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// The id prefix the store hands out.
pub(crate) const SPEECH_ID_PREFIX: &str = "speech_";

/// A new speech request id: `speech_<uuid>`.
pub(crate) fn new_speech_id() -> String {
    format!("{SPEECH_ID_PREFIX}{}", uuid::Uuid::new_v4())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechOutcomeStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechMode {
    Buffered,
    AudioStream,
    Sse,
    BatchItem,
    SocketUtterance,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub(crate) struct SpeechUsage {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
    pub(crate) total_tokens: usize,
    /// SGLang-Omni's names for the same counts.
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
}

impl SpeechUsage {
    pub(crate) fn new(input_tokens: usize, output_tokens: usize) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            prompt_tokens: input_tokens,
            completion_tokens: output_tokens,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct SpeechOutcomeError {
    pub(crate) message: String,
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) code: Option<&'static str>,
}

/// One request's record, as the lookup returns it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct SpeechOutcome {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) status: SpeechOutcomeStatus,
    pub(crate) model: String,
    pub(crate) mode: SpeechMode,
    pub(crate) response_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) audio_duration_secs: Option<f32>,
    /// `stop` (the model ended), `length` (the output budget cut it),
    /// `cancelled` or `error`; SGLang-Omni's field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<SpeechUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) timing: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<SpeechOutcomeError>,
    pub(crate) created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) completed_at: Option<u64>,
    /// SGLang-Omni's name for `id`.
    pub(crate) request_id: String,
    #[serde(skip)]
    tenant_key: Option<[u8; 32]>,
    #[serde(skip)]
    touched: Instant,
}

/// How a request ended.
#[derive(Debug, Clone, Default)]
pub(crate) struct SpeechCompletion {
    pub(crate) sample_rate: Option<u32>,
    pub(crate) audio_duration_secs: Option<f32>,
    pub(crate) finish_reason: Option<&'static str>,
    pub(crate) usage: Option<SpeechUsage>,
    pub(crate) timing: Option<serde_json::Value>,
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[derive(Default)]
struct Records {
    order: VecDeque<String>,
    by_id: HashMap<String, SpeechOutcome>,
}

pub(crate) struct SpeechOutcomeStore {
    records: Mutex<Records>,
    capacity: usize,
    ttl: Duration,
}

impl std::fmt::Debug for SpeechOutcomeStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechOutcomeStore")
            .field("capacity", &self.capacity)
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

impl SpeechOutcomeStore {
    pub(crate) fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            records: Mutex::new(Records::default()),
            capacity: capacity.max(1),
            ttl,
        }
    }

    pub(crate) fn from_tuning() -> Self {
        let tuning = &crate::env_config::tuning().speech_api;
        Self::new(tuning.outcome_capacity, tuning.outcome_ttl)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Records> {
        self.records
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn evict(&self, records: &mut Records) {
        let now = Instant::now();
        while let Some(oldest) = records.order.front() {
            let expired = records
                .by_id
                .get(oldest)
                .is_none_or(|record| now.duration_since(record.touched) > self.ttl);
            if records.order.len() <= self.capacity && !expired {
                break;
            }
            let oldest = records.order.pop_front().expect("front");
            records.by_id.remove(&oldest);
        }
    }

    /// Records an admitted request.
    pub(crate) fn begin(
        &self,
        id: &str,
        tenant_key: Option<[u8; 32]>,
        model: &str,
        mode: SpeechMode,
        response_format: &str,
    ) {
        let mut records = self.lock();
        if records.by_id.contains_key(id) {
            return;
        }
        records.order.push_back(id.to_string());
        records.by_id.insert(
            id.to_string(),
            SpeechOutcome {
                id: id.to_string(),
                object: "audio.speech",
                status: SpeechOutcomeStatus::InProgress,
                model: model.to_string(),
                mode,
                response_format: response_format.to_string(),
                sample_rate: None,
                audio_duration_secs: None,
                finish_reason: None,
                usage: None,
                timing: None,
                error: None,
                created_at: unix_now(),
                completed_at: None,
                request_id: id.to_string(),
                tenant_key,
                touched: Instant::now(),
            },
        );
        self.evict(&mut records);
    }

    fn finish(&self, id: &str, update: impl FnOnce(&mut SpeechOutcome)) {
        let mut records = self.lock();
        if let Some(record) = records.by_id.get_mut(id) {
            if record.status != SpeechOutcomeStatus::InProgress {
                return;
            }
            update(record);
            record.completed_at = Some(unix_now());
            record.touched = Instant::now();
        }
    }

    pub(crate) fn complete(&self, id: &str, completion: SpeechCompletion) {
        self.finish(id, |record| {
            record.status = SpeechOutcomeStatus::Completed;
            record.sample_rate = completion.sample_rate.or(record.sample_rate);
            record.audio_duration_secs = completion.audio_duration_secs;
            record.finish_reason = completion.finish_reason.or(Some("stop"));
            record.usage = completion.usage;
            record.timing = completion.timing;
        });
    }

    pub(crate) fn fail(
        &self,
        id: &str,
        message: &str,
        kind: &'static str,
        code: Option<&'static str>,
    ) {
        self.finish(id, |record| {
            record.status = SpeechOutcomeStatus::Failed;
            record.finish_reason = Some("error");
            record.error = Some(SpeechOutcomeError {
                message: message.to_string(),
                kind,
                code,
            });
        });
    }

    pub(crate) fn cancel(&self, id: &str) {
        self.finish(id, |record| {
            record.status = SpeechOutcomeStatus::Cancelled;
            record.finish_reason = Some("cancelled");
        });
    }

    /// The record for `id` (with or without the `speech_` prefix), if the
    /// caller's tenant made it.
    pub(crate) fn get(&self, id: &str, tenant_key: Option<[u8; 32]>) -> Option<SpeechOutcome> {
        let mut records = self.lock();
        self.evict(&mut records);
        let id = id.trim();
        let record = records
            .by_id
            .get(id)
            .or_else(|| records.by_id.get(&format!("{SPEECH_ID_PREFIX}{id}")))?;
        (record.tenant_key == tenant_key).then(|| record.clone())
    }
}

/// Marks a request cancelled if it is dropped while still in progress: a
/// client that disconnects mid-stream drops the response future, and with it
/// this guard.
pub(crate) struct SpeechOutcomeGuard {
    store: std::sync::Arc<SpeechOutcomeStore>,
    id: String,
}

impl SpeechOutcomeGuard {
    pub(crate) fn new(store: std::sync::Arc<SpeechOutcomeStore>, id: String) -> Self {
        Self { store, id }
    }
}

impl Drop for SpeechOutcomeGuard {
    fn drop(&mut self) {
        self.store.cancel(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcomes_end_once_and_stay_with_their_tenant() {
        let store = SpeechOutcomeStore::new(8, Duration::from_secs(60));
        let tenant = Some([7u8; 32]);
        store.begin("speech_a", tenant, "Kokoro-82M", SpeechMode::Sse, "mp3");
        assert_eq!(
            store.get("speech_a", tenant).unwrap().status,
            SpeechOutcomeStatus::InProgress
        );
        store.complete(
            "speech_a",
            SpeechCompletion {
                sample_rate: Some(24_000),
                audio_duration_secs: Some(1.5),
                finish_reason: Some("length"),
                usage: Some(SpeechUsage::new(0, 40)),
                timing: None,
            },
        );
        store.cancel("speech_a");
        let outcome = store.get("a", tenant).expect("bare uuid finds it");
        assert_eq!(outcome.status, SpeechOutcomeStatus::Completed);
        assert_eq!(outcome.finish_reason, Some("length"));
        assert!(store.get("speech_a", None).is_none(), "other tenant");
        let json = serde_json::to_value(&outcome).unwrap();
        assert_eq!(json["object"], "audio.speech");
        assert_eq!(json["usage"]["completion_tokens"], 40);
        assert_eq!(json["request_id"], "speech_a");
    }

    #[test]
    fn the_oldest_outcomes_are_evicted() {
        let store = SpeechOutcomeStore::new(2, Duration::from_secs(60));
        for id in ["speech_1", "speech_2", "speech_3"] {
            store.begin(id, None, "m", SpeechMode::Buffered, "wav");
        }
        assert!(store.get("speech_1", None).is_none());
        assert!(store.get("speech_3", None).is_some());
        let expired = SpeechOutcomeStore::new(8, Duration::ZERO);
        expired.begin("speech_x", None, "m", SpeechMode::Buffered, "wav");
        std::thread::sleep(Duration::from_millis(2));
        assert!(expired.get("speech_x", None).is_none());
    }
}
