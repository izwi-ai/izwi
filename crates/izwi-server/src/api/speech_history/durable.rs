use super::*;
use crate::api::tts_long_form::{generate_speech_plan_stream_with_progress, SpeechTextPlan};
use crate::batch_runtime::speech_progress::{SpeechCheckpoint, SpeechPcmBatch};

pub(crate) struct DurableSpeechProgress {
    state: AppState,
    attempt: StageExecutionContext,
    artifact_tenant: crate::artifact_store::ArtifactTenant,
    checkpoint: SpeechCheckpoint,
}
impl DurableSpeechProgress {
    async fn new(
        state: &AppState,
        attempt: &StageExecutionContext,
        identity: serde_json::Value,
        total: usize,
        tenant_key: Option<[u8; 32]>,
    ) -> anyhow::Result<Self> {
        let checkpoint = match attempt.claimed().stage.progress_json.as_ref() {
            Some(progress) => SpeechCheckpoint::recover(progress, &identity)?,
            None => SpeechCheckpoint::new(
                identity,
                u32::try_from(total)?,
                std::env::var("IZWI_TTS_MAX_JOURNAL_BYTES")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .filter(|v| *v > 0)
                    .unwrap_or(8 * 1024 * 1024 * 1024),
            )?,
        };
        checkpoint.save(attempt).await?;
        Ok(Self {
            state: state.clone(),
            attempt: attempt.clone(),
            artifact_tenant: crate::artifact_store::ArtifactTenant::from_scheduling_key(tenant_key),
            checkpoint,
        })
    }
    pub(crate) fn artifact_fingerprint(&self) -> Option<&str> {
        self.checkpoint
            .identity
            .get("artifact_fingerprint")
            .and_then(serde_json::Value::as_str)
    }
    pub(crate) fn completed_text_bytes(&self) -> usize {
        self.checkpoint.completed_text_bytes as usize
    }
    pub(crate) fn next_sequence(&self) -> usize {
        self.checkpoint.next_sequence as usize
    }
    pub(crate) fn completed_segments(&self) -> usize {
        self.checkpoint.completed_segments as usize
    }
    pub(crate) fn completed_tokens(&self) -> usize {
        self.checkpoint.completed_tokens
    }
    pub(crate) fn completed_execution_ms(&self) -> f32 {
        self.checkpoint.completed_execution_ms
    }
    pub(crate) fn completed_duration_secs(&self) -> f32 {
        self.checkpoint.completed_duration_secs
    }
    pub(crate) async fn begin_segment(
        &mut self,
        index: usize,
        _range: std::ops::Range<usize>,
        total: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            index == self.completed_segments(),
            "Speech segment recovery order changed"
        );
        // Exact-context splitting may add a segment before any PCM is published.
        self.checkpoint.active_segment = None;
        self.checkpoint.total_segments = self.checkpoint.total_segments.max(u32::try_from(total)?);
        self.checkpoint.begin_segment(&self.attempt).await?;
        Ok(())
    }
    pub(crate) async fn publish_chunk(&mut self, chunk: &AudioChunk) -> anyhow::Result<()> {
        self.attempt.ensure_active().await?;
        let rate = chunk.sample_rate_or(self.state.runtime.sample_rate().await);
        let bytes: Vec<u8> = chunk
            .samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        anyhow::ensure!(
            bytes.len() <= 1024 * 1024,
            "Speech PCM replay batch exceeds one MiB"
        );
        self.checkpoint
            .publish_pcm(
                &self.attempt,
                &self.state.artifact_store,
                &self.artifact_tenant,
                bytes,
                chunk.samples.len() as u64,
                rate,
            )
            .await?;
        Ok(())
    }
    pub(crate) async fn complete_segment(
        &mut self,
        text_end: usize,
        tokens: usize,
        execution_ms: f32,
        duration_secs: f32,
    ) -> anyhow::Result<()> {
        self.checkpoint.completed_tokens += tokens;
        self.checkpoint.completed_execution_ms += execution_ms;
        self.checkpoint.completed_duration_secs += duration_secs;
        self.checkpoint
            .complete_segment(&self.attempt, text_end as u64)
            .await?;
        if self.checkpoint.completed_segments < self.checkpoint.total_segments {
            return Err(crate::batch_runtime::worker::SpeechStageYield.into());
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn synthesize_fish_record(
    state: &AppState,
    ctx: &RequestContext,
    tenant_key: Option<[u8; 32]>,
    req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
    variant: ModelVariant,
    model_id: String,
    input_text: String,
    target_record_id: Option<String>,
    workload: WorkloadClass,
    attempt: Option<&StageExecutionContext>,
    projection_attempt: Option<&RuntimeProjectionAttempt>,
    claimed: Option<&ClaimedStage>,
) -> Result<
    (
        SpeechHistoryRecord,
        Option<crate::batch_runtime::types::RuntimeArtifact>,
    ),
    ApiError,
> {
    let synthesis = async {
        let record_id = match target_record_id {
            Some(id) => id,
            None => {
                create_pending_record(state, route_kind, &model_id, &input_text, &req)
                    .await
                    .map_err(|e| anyhow::anyhow!(e.message))?
                    .id
            }
        };
        let mut base = build_generation_request(
            req.clone(),
            ctx.correlation_id.clone(),
            input_text.clone(),
            true,
            variant,
        );
        base.runtime_context.tenant_key = tenant_key;
        let plan = SpeechTextPlan::fish(&input_text, base.config.options.max_tokens)?;
        state.runtime.load_model(variant).await?;
        let artifact_fingerprint = state
            .runtime
            .fish_s2_artifact_fingerprint()
            .await
            .context("Fish model did not publish an artifact fingerprint")?;
        let mut progress = if let Some(attempt) = attempt {
            Some(
                DurableSpeechProgress::new(
                    state,
                    attempt,
                    serde_json::json!({
                    "plan": plan, "model_snapshot": claimed.map(|c| &c.job.model_snapshot_json),
                    "model": model_id, "artifact_fingerprint": artifact_fingerprint,
                    }),
                    plan.segments.len(),
                    tenant_key,
                )
                .await?,
            )
        } else {
            None
        };
        let mut spool: Option<SpeechWavSpool> = None;
        let mut stats = StreamRequestStatistics::default();
        {
            let (tx, mut rx) = mpsc::channel::<AudioChunk>(2);
            let runner = generate_speech_plan_stream_with_progress(
                state,
                variant,
                base,
                tx,
                workload,
                progress.as_mut(),
            );
            tokio::pin!(runner);
            let consume = async {
                while let Some(chunk) = rx.recv().await {
                    stats.observe(&chunk);
                    if chunk.samples.is_empty() {
                        continue;
                    }
                    let rate = chunk.sample_rate_or(state.runtime.sample_rate().await);
                    if attempt.is_none() {
                        append_samples(&mut spool, &chunk.samples, rate).await?;
                    }
                }
                Ok::<_, anyhow::Error>(())
            };
            tokio::try_join!(async { runner.await.map_err(anyhow::Error::from) }, consume)?;
        }
        if let Some(progress) = progress.as_mut() {
            progress.checkpoint.total_segments = progress.checkpoint.completed_segments;
            progress.checkpoint.save(&progress.attempt).await?;
        }
        if let Some(attempt) = attempt {
            attempt.ensure_active().await?;
        }
        // Rebuild the bounded local finalization spool from committed PCM objects.
        if let (Some(progress), Some(claimed)) = (progress.as_ref(), claimed) {
            let exact_pcm_bytes = usize::try_from(
                progress
                    .checkpoint
                    .committed_samples
                    .checked_mul(2)
                    .context("Speech PCM size overflow")?,
            )?;
            let sample_rate = progress
                .checkpoint
                .sample_rate
                .context("Speech completed without a sample rate")?;
            let limit = std::env::var("IZWI_TTS_STREAM_MAX_PCM_BYTES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(u32::MAX as usize - 44)
                .min(u32::MAX as usize - 44);
            spool = Some(SpeechWavSpool::new_reserved(sample_rate, limit, exact_pcm_bytes).await?);
            let mut cursor = None;
            let mut restored_samples = 0_u64;
            let mut restored_sequence = 0_u64;
            loop {
                let page = state
                    .batch_runtime_store
                    .speech_pcm_after(&claimed.job.id, cursor, 16)
                    .await?;
                if page.is_empty() {
                    break;
                }
                for artifact in page {
                    let batch: SpeechPcmBatch =
                        serde_json::from_value(artifact.metadata_json.clone())?;
                    anyhow::ensure!(
                        batch.sequence == restored_sequence
                            && batch.sample_offset == restored_samples,
                        "Speech replay checkpoint has missing or reordered PCM"
                    );
                    anyhow::ensure!(
                        batch.sequence < progress.checkpoint.next_sequence,
                        "Uncheckpointed speech replay publication"
                    );
                    let artifact_tenant =
                        crate::artifact_store::ArtifactTenant::from_scheduling_key(tenant_key);
                    let pcm = read_pcm(state, &artifact_tenant, &artifact, &batch).await?;
                    append_samples(&mut spool, &pcm, batch.sample_rate).await?;
                    restored_samples += batch.sample_count;
                    restored_sequence += 1;
                    cursor = Some(batch.sequence);
                }
            }
            anyhow::ensure!(
                restored_samples == progress.checkpoint.committed_samples
                    && restored_sequence == progress.checkpoint.next_sequence,
                "speech_replay_expired: committed PCM is no longer available for recovery"
            );
        }
        let file = spool
            .context("Speech completed without audio")?
            .finish_file()
            .await?;
        let filename = default_audio_filename(route_kind, "wav");
        let checksum = sha256_file(file.path()).await?;
        let mut storage_key = state
            .media_ingest
            .persist_generated_audio_file(
                format!("{record_id}-output-{}", uuid::Uuid::new_v4()),
                Some(&filename),
                "audio/wav",
                file.path().to_path_buf(),
                file.len(),
                route_kind.as_db_value(),
            )
            .await?;
        let output_artifact = if let Some(attempt) = attempt {
            let published = attempt.publish_output_artifact(NewStageOutputArtifact {
                publication_key: "primary-audio".to_string(), artifact_kind: RuntimeArtifactKind::Audio,
                artifact_role: RuntimeArtifactRole::OutputPrimary, media_asset_id: None, text_asset_id: None,
                storage_key: Some(storage_key.clone()), content_type: Some("audio/wav".into()), filename: Some(filename.clone()),
                size_bytes: Some(file.len()), sha256: Some(checksum.clone()),
                metadata_json: serde_json::json!({"sample_rate": file.sample_rate(), "sample_count": file.sample_count()}), retention_policy: "default".into(),
            }).await;
            let artifact = match published {
                Ok(artifact) => artifact,
                Err(error) => {
                    if let Ok(None) = state
                        .batch_runtime_store
                        .stage_output_for_key(attempt.lease(), "primary-audio")
                        .await
                    {
                        let _ = state.media_ingest.delete_object(&storage_key).await;
                    }
                    return Err(error);
                }
            };
            anyhow::ensure!(
                artifact.sha256.as_deref() == Some(checksum.as_str())
                    && artifact.size_bytes == Some(file.len()),
                "Primary speech publication changed within an attempt"
            );
            if let Some(committed_key) = artifact.storage_key.as_ref() {
                if committed_key != &storage_key {
                    let _ = state.media_ingest.delete_object(&storage_key).await;
                    storage_key = committed_key.clone();
                }
            }
            Some(artifact)
        } else {
            None
        };
        let duration = file.sample_count() as f64 / file.sample_rate() as f64;
        let completion = CompleteSpeechHistoryRecord {
            model_id: Some(model_id),
            speaker: req.speaker,
            language: req.language,
            saved_voice_id: req.saved_voice_id,
            speed: req.speed.map(f64::from),
            input_text,
            voice_description: req.voice_description,
            reference_text: req.reference_text,
            generation_time_ms: stats.execution_ms as f64,
            audio_duration_secs: Some(duration),
            rtf: Some(if duration > 0.0 {
                stats.execution_ms as f64 / 1000.0 / duration
            } else {
                0.0
            }),
            tokens_generated: Some(stats.tokens),
            audio_mime_type: "audio/wav".into(),
            audio_filename: Some(filename),
            audio_bytes: Vec::new(),
            preexisting_audio_storage_path: Some(storage_key),
        };
        let record = match projection_attempt {
            Some(projection) => {
                state
                    .speech_history_store
                    .complete_record_for_attempt(route_kind, record_id, projection, completion)
                    .await?
            }
            None => {
                state
                    .speech_history_store
                    .complete_record(route_kind, record_id, completion)
                    .await?
            }
        }
        .context("Speech attempt lost ownership before completion")?;
        Ok::<_, anyhow::Error>((record, output_artifact))
    };
    let max_secs = std::env::var("IZWI_TTS_MAX_JOB_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(24 * 3600);
    let elapsed_secs = claimed
        .map(|claimed| {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            now_ms.saturating_sub(claimed.job.created_at) / 1000
        })
        .unwrap_or(0);
    if elapsed_secs >= max_secs {
        return Err(ApiError::internal(
            "speech_job_timeout: overall job deadline exceeded",
        ));
    }
    let result = tokio::time::timeout(Duration::from_secs(max_secs - elapsed_secs), async {
        if let Some(attempt) = attempt {
            tokio::select! {
                result = synthesis => result,
                result = async {
                    loop {
                        attempt.ensure_active().await?;
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                    #[allow(unreachable_code)]
                    Ok::<(), anyhow::Error>(())
                } => { result?; anyhow::bail!("Speech job cancelled") }
            }
        } else {
            synthesis.await
        }
    })
    .await
    .map_err(|_| ApiError::internal("speech_job_timeout: overall job deadline exceeded"))?;
    result.map_err(|error| ApiError::internal(error.to_string()))
}

async fn append_samples(
    spool: &mut Option<SpeechWavSpool>,
    samples: &[f32],
    rate: u32,
) -> anyhow::Result<()> {
    if spool.is_none() {
        let limit = std::env::var("IZWI_TTS_STREAM_MAX_PCM_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(u32::MAX as usize - 44)
            .min(u32::MAX as usize - 44);
        *spool = Some(SpeechWavSpool::new(rate, limit)?);
    }
    let pcm = AudioEncoder::new(rate, 1).encode(samples, AudioFormat::RawI16)?;
    spool.as_mut().unwrap().append_pcm(&pcm).await
}

async fn read_pcm(
    state: &AppState,
    tenant: &crate::artifact_store::ArtifactTenant,
    artifact: &crate::batch_runtime::types::RuntimeArtifact,
    batch: &SpeechPcmBatch,
) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        batch.version == 1 && batch.sample_count <= 262144,
        "Invalid speech replay metadata"
    );
    anyhow::ensure!(
        artifact.publication_key.as_deref()
            == Some(
                crate::batch_runtime::speech_progress::pcm_publication_key(batch.sequence).as_str()
            ),
        "Speech replay sequence does not match its publication key"
    );
    let expected = batch.sample_count * 4;
    anyhow::ensure!(
        expected <= crate::batch_runtime::speech_progress::MAX_PCM_REPLAY_BATCH_BYTES,
        "Speech replay PCM exceeds the per-object limit"
    );
    anyhow::ensure!(
        artifact.text_asset_id.is_none(),
        "Speech replay artifact exposes an invalid text reference"
    );
    let bytes = match (
        artifact.media_asset_id.as_deref(),
        artifact.storage_key.as_deref(),
    ) {
        (Some(id), None) => {
            let id = crate::artifact_store::ArtifactId::parse(id.to_string())?;
            let descriptor = state.artifact_store.stat(tenant, &id).await?;
            anyhow::ensure!(
                descriptor.content_type == "audio/pcm-f32le"
                    && descriptor.size_bytes == expected
                    && artifact.content_type.as_deref() == Some(descriptor.content_type.as_str())
                    && artifact.size_bytes == Some(expected)
                    && artifact.sha256.as_deref() == Some(descriptor.sha256.as_str()),
                "Speech replay PCM descriptor mismatch"
            );
            state.artifact_store.read(tenant, &id).await?.bytes
        }
        (None, Some(key)) => {
            // Compatibility for journals written before opaque replay adoption.
            use tokio::io::AsyncReadExt;
            let object = state.media_ingest.read_object_stream(key).await?;
            anyhow::ensure!(
                object
                    .metadata
                    .content_length
                    .is_none_or(|length| length == expected),
                "Speech replay PCM size mismatch"
            );
            let mut bytes = Vec::with_capacity(expected as usize + 1);
            object
                .reader
                .take(expected + 1)
                .read_to_end(&mut bytes)
                .await?;
            bytes
        }
        _ => anyhow::bail!("Speech replay artifact has an ambiguous storage reference"),
    };
    anyhow::ensure!(
        bytes.len() as u64 == expected,
        "Speech replay PCM size mismatch"
    );
    let actual_sha256 = sha256_hex(&bytes);
    anyhow::ensure!(
        artifact.sha256.as_deref() == Some(actual_sha256.as_str()),
        "Speech replay checksum mismatch"
    );
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

#[derive(Deserialize, Default)]
pub(crate) struct ReplayQuery {
    after_sequence: Option<u64>,
}
pub(crate) async fn replay_text_to_speech(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(record_id): Path<String>,
    Query(query): Query<ReplayQuery>,
) -> Result<Response, ApiError> {
    replay_response(state, ctx.tenant_key(), record_id, query.after_sequence).await
}

pub(super) async fn replay_response(
    state: AppState,
    tenant: Option<[u8; 32]>,
    record_id: String,
    after: Option<u64>,
) -> Result<Response, ApiError> {
    let job = state
        .batch_runtime_store
        .get_latest_job_for_route_record(
            RuntimeJobKind::TtsSpeech,
            SpeechRouteKind::TextToSpeech.as_db_value(),
            &record_id,
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Speech job not found"))?;
    let request: BatchSpeechRequest =
        serde_json::from_value(job.request_json).map_err(|e| ApiError::internal(e.to_string()))?;
    if request.tenant_key != tenant {
        return Err(ApiError::forbidden("Speech job belongs to another tenant"));
    }
    let artifact_tenant =
        crate::artifact_store::ArtifactTenant::from_scheduling_key(request.tenant_key);
    let record = state
        .speech_history_store
        .get_record(SpeechRouteKind::TextToSpeech, record_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Speech record not found"))?;
    // A cursor names a committed chunk, never an arbitrary future position.
    if let Some(sequence) = after {
        let page = state
            .batch_runtime_store
            .speech_pcm_after(&job.id, sequence.checked_sub(1), 1)
            .await
            .map_err(map_store_error)?;
        if page
            .first()
            .and_then(|a| a.metadata_json["sequence"].as_u64())
            != Some(sequence)
        {
            return Err(ApiError::bad_request("Invalid speech replay cursor"));
        }
    }
    let stream = async_stream::stream! {
        yield Ok::<_, Infallible>(sse_json(serde_json::json!({"event":"created", "record": record, "durable":true})));
        let mut cursor = after;
        let mut started = false;
        let mut heartbeat = tokio::time::Instant::now();
        let mut last_progress = None;
        loop {
            let page = match state.batch_runtime_store.speech_pcm_after(&job.id, cursor, 16).await {
                Ok(page) => page,
                Err(error) => { yield Ok(sse_json(serde_json::json!({"event":"error","error":error.to_string()}))); break; }
            };
            let had_data = !page.is_empty();
            let mut failure = None;
            for artifact in page {
                let result: anyhow::Result<_> = async {
                    let batch: SpeechPcmBatch =
                        serde_json::from_value(artifact.metadata_json.clone())?;
                    let expected_sequence = cursor
                        .map(|sequence| sequence.checked_add(1).context("Speech replay cursor overflow"))
                        .transpose()?
                        .unwrap_or(0);
                    anyhow::ensure!(
                        batch.sequence == expected_sequence,
                        "Speech replay journal has a missing or reordered sequence"
                    );
                    let samples = read_pcm(&state, &artifact_tenant, &artifact, &batch).await?;
                    let bytes = AudioEncoder::new(batch.sample_rate, 1).encode(&samples, AudioFormat::RawI16)?;
                    Ok((batch, bytes))
                }.await;
                match result {
                    Ok((batch, bytes)) => {
                        if !started {
                            yield Ok(sse_json(serde_json::json!({"event":"start","request_id":record_id,"sample_rate":batch.sample_rate,"audio_format":"pcm_i16"})));
                            started = true;
                        }
                        yield Ok(sse_json(serde_json::json!({"event":"chunk","request_id":record_id,"sequence":batch.sequence,"sample_count":batch.sample_count,"sample_rate":batch.sample_rate,"audio_format":"pcm_i16","audio_base64":base64::engine::general_purpose::STANDARD.encode(bytes)})));
                        cursor = Some(batch.sequence);
                    }
                    Err(error) => { failure = Some(error.to_string()); break; }
                }
            }
            if let Some(error) = failure { yield Ok(sse_json(serde_json::json!({"event":"error","error":error}))); break; }
            if had_data { continue; }
            let current = state.speech_history_store.get_record(SpeechRouteKind::TextToSpeech, record_id.clone()).await;
            match current {
                Ok(Some(record)) if record.processing_status == SpeechHistoryProcessingStatus::Ready => {
                    // The final PCM may commit between the page read and ready.
                    match state.batch_runtime_store.speech_pcm_after(&job.id, cursor, 1).await {
                        Ok(page) if !page.is_empty() => continue,
                        Ok(_) => {},
                        Err(error) => { yield Ok(sse_json(serde_json::json!({"event":"error","error":error.to_string()}))); break; }
                    }
                    yield Ok(sse_json(serde_json::json!({"event":"final","request_id":record_id,"generation_time_ms":record.generation_time_ms,"audio_duration_secs":record.audio_duration_secs,"rtf":record.rtf,"tokens_generated":record.tokens_generated,"record":record})));
                    yield Ok(sse_json(serde_json::json!({"event":"done"}))); break;
                }
                Ok(Some(record)) if record.processing_status == SpeechHistoryProcessingStatus::Failed => {
                    match state.batch_runtime_store.get_job(&job.id).await {
                        Ok(Some(job)) if matches!(job.status, RuntimeJobStatus::Created | RuntimeJobStatus::Queued | RuntimeJobStatus::Running | RuntimeJobStatus::Retrying | RuntimeJobStatus::Postprocessing) => {
                            tokio::time::sleep(Duration::from_millis(250)).await; continue;
                        }
                        _ => {}
                    }
                    yield Ok(sse_json(serde_json::json!({"event":"error","error":record.processing_error,"record":record}))); break;
                }
                Ok(Some(_)) => {},
                _ => { yield Ok(sse_json(serde_json::json!({"event":"error","error":"Speech record unavailable"}))); break; }
            }
            if let Ok(stages) = state.batch_runtime_store.list_stages_for_job(&job.id).await {
                if let Some(progress) = stages.first().and_then(|stage| stage.progress_json.as_ref()) {
                    let value = serde_json::json!({"event":"progress", "completed_segments":progress["completed_segments"], "total_segments":progress["total_segments"], "processed_text_bytes":progress["completed_text_bytes"]});
                    if last_progress.as_ref() != Some(&value) { yield Ok(sse_json(value.clone())); last_progress = Some(value); }
                }
            }
            if heartbeat.elapsed() >= Duration::from_secs(10) {
                yield Ok(bytes::Bytes::from_static(b": keepalive\n\n"));
                heartbeat = tokio::time::Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    };
    use futures::StreamExt;
    let stream_budget = crate::speech_resource_budget::ByteBudget::new(4 * 1024 * 1024);
    let stream = stream.map(move |item| {
        let payload = String::from_utf8(item.expect("infallible speech stream").to_vec())?;
        let global = crate::speech_resource_budget::event_budget().reserve(payload.len())?;
        let local = stream_budget.reserve(payload.len())?;
        Ok::<_, anyhow::Error>(bytes::Bytes::from_owner(BudgetedStreamEvent {
            payload,
            _global_reservation: global,
            _stream_reservation: local,
        }))
    });
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache, no-transform")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(stream))
        .expect("static SSE response"))
}
fn sse_json(event: serde_json::Value) -> bytes::Bytes {
    bytes::Bytes::from(format!("data: {event}\n\n"))
}

async fn sha256_file(path: &std::path::Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncReadExt;
    let mut source = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 64 * 1024];
    loop {
        let length = source.read(&mut buffer).await?;
        if length == 0 {
            break;
        }
        hasher.update(&buffer[..length]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(super) async fn cleanup_expired_replay(state: &AppState) -> anyhow::Result<()> {
    let retention = std::env::var("IZWI_TTS_REPLAY_RETENTION_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(86400);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as u64;
    for artifact in state
        .batch_runtime_store
        .expired_speech_pcm(now.saturating_sub(retention.saturating_mul(1000)))
        .await?
    {
        let job = state
            .batch_runtime_store
            .get_job(&artifact.job_id)
            .await?
            .context("Speech replay artifact has no parent job")?;
        let tenant = speech_artifact_tenant(&job)?;
        if delete_pcm_artifact(state, &tenant, &artifact).await? {
            state
                .batch_runtime_store
                .remove_speech_pcm_artifact(&artifact.id)
                .await?;
        }
    }
    Ok(())
}

pub(super) async fn cleanup_record_replay(
    state: &AppState,
    route: SpeechRouteKind,
    record_id: &str,
) -> anyhow::Result<()> {
    let Some(job) = state
        .batch_runtime_store
        .get_latest_job_for_route_record(RuntimeJobKind::TtsSpeech, route.as_db_value(), record_id)
        .await?
    else {
        return Ok(());
    };
    let tenant = speech_artifact_tenant(&job)?;
    anyhow::ensure!(
        state
            .batch_runtime_store
            .fence_speech_replay_deletion(&job.id)
            .await?,
        "Speech job resumed before recording deletion; retry deletion to cancel it"
    );
    loop {
        let page = state
            .batch_runtime_store
            .speech_pcm_after(&job.id, None, 64)
            .await?;
        if page.is_empty() {
            return Ok(());
        }
        for artifact in page {
            anyhow::ensure!(
                delete_pcm_artifact(state, &tenant, &artifact).await?,
                "Speech replay cleanup is durably pending; retry deletion"
            );
            state
                .batch_runtime_store
                .remove_speech_pcm_artifact(&artifact.id)
                .await?;
        }
    }
}

fn speech_artifact_tenant(
    job: &crate::batch_runtime::types::RuntimeJob,
) -> anyhow::Result<crate::artifact_store::ArtifactTenant> {
    let request: BatchSpeechRequest = serde_json::from_value(job.request_json.clone())
        .context("Invalid server-authored speech job request")?;
    Ok(crate::artifact_store::ArtifactTenant::from_scheduling_key(
        request.tenant_key,
    ))
}

async fn delete_pcm_artifact(
    state: &AppState,
    tenant: &crate::artifact_store::ArtifactTenant,
    artifact: &crate::batch_runtime::types::RuntimeArtifact,
) -> anyhow::Result<bool> {
    match (
        artifact.media_asset_id.as_deref(),
        artifact.storage_key.as_deref(),
    ) {
        (Some(id), None) => {
            let id = crate::artifact_store::ArtifactId::parse(id.to_string())?;
            match state.artifact_store.delete(tenant, &id).await {
                Ok(_) => Ok(true),
                Err(crate::artifact_store::ArtifactStoreError::DeleteIncomplete) => Ok(false),
                Err(error) => Err(error.into()),
            }
        }
        (None, Some(key)) => {
            state.media_ingest.delete_object(key).await?;
            Ok(true)
        }
        _ => anyhow::bail!("Speech replay artifact has an ambiguous storage reference"),
    }
}
