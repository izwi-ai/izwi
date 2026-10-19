-- Izwi application schema for PostgreSQL (provider-managed migrations).
-- Generated from crates/izwi-db/src/migrator.rs: do not edit. Regenerate with
--   IZWI_WRITE_POSTGRES_SCHEMA=1 cargo test -p izwi-db --lib postgres_schema
-- Apply it to an empty schema. It records the migrations it contains in
-- schema_migrations, so Izwi applies nothing further when it starts.

BEGIN;

CREATE TABLE IF NOT EXISTS runtime_admission_locks (id TEXT PRIMARY KEY, lock_value BIGINT NOT NULL DEFAULT 1);

CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model_id TEXT NULL,
    system_prompt TEXT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    content_parts TEXT NULL,
    created_at BIGINT NOT NULL,
    tokens_generated BIGINT NULL,
    generation_time_ms DOUBLE PRECISION NULL,
    FOREIGN KEY(thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_threads_updated_at ON chat_threads(updated_at DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created_at ON chat_messages(thread_id, created_at, id);

CREATE TABLE IF NOT EXISTS voice_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    observational_memory_enabled BIGINT NOT NULL DEFAULT 1,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_sessions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('modular', 'unified')),
    system_prompt TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    ended_at BIGINT NULL,
    FOREIGN KEY(profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS voice_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    utterance_id TEXT NOT NULL,
    utterance_seq BIGINT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('modular', 'unified')),
    status TEXT NOT NULL CHECK(status IN ('processing', 'ok', 'error', 'timeout', 'interrupted', 'no_input')),
    status_reason TEXT NULL,
    vad_end_reason TEXT NULL,
    user_text TEXT NULL,
    assistant_text TEXT NULL,
    assistant_raw_text TEXT NULL,
    language TEXT NULL,
    audio_duration_secs DOUBLE PRECISION NULL,
    asr_model_id TEXT NULL,
    text_model_id TEXT NULL,
    tts_model_id TEXT NULL,
    s2s_model_id TEXT NULL,
    speaker TEXT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES voice_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_voice_sessions_updated_at ON voice_sessions(updated_at DESC, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_turns_session_utterance ON voice_turns(session_id, utterance_seq);

CREATE INDEX IF NOT EXISTS idx_voice_turns_session_created_at ON voice_turns(session_id, created_at ASC, id ASC);

CREATE TABLE IF NOT EXISTS voice_observations (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    canonical_summary TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    source_turn_id TEXT NULL,
    source_user_text TEXT NULL,
    source_assistant_text TEXT NULL,
    times_seen BIGINT NOT NULL DEFAULT 1,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    forgotten_at BIGINT NULL,
    FOREIGN KEY(profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_voice_observations_profile_updated_at ON voice_observations(profile_id, updated_at DESC, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_observations_profile_canonical_active ON voice_observations(profile_id, canonical_summary) WHERE forgotten_at IS NULL;

CREATE TABLE IF NOT EXISTS onboarding_state (
    id TEXT PRIMARY KEY,
    completed_at BIGINT NULL,
    analytics_opt_in BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transcription_records (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    transcription_mode TEXT NOT NULL DEFAULT 'transcription',
    model_id TEXT NULL,
    aligner_model_id TEXT NULL,
    language TEXT NULL,
    processing_status TEXT NOT NULL DEFAULT 'ready',
    processing_error TEXT NULL,
    processing_progress_json TEXT NULL,
    runtime_stage_id TEXT NULL,
    runtime_attempt_token TEXT NULL,
    duration_secs DOUBLE PRECISION NULL,
    processing_time_ms DOUBLE PRECISION NOT NULL,
    rtf DOUBLE PRECISION NULL,
    audio_mime_type TEXT NOT NULL,
    audio_filename TEXT NULL,
    audio_storage_path TEXT NOT NULL,
    transcription TEXT NOT NULL,
    segments_json TEXT NOT NULL DEFAULT '[]',
    words_json TEXT NOT NULL DEFAULT '[]',
    speaker_attributed_text TEXT NULL,
    speaker_turns_json TEXT NOT NULL DEFAULT '[]',
    saa_status TEXT NOT NULL DEFAULT 'not_requested',
    saa_warnings_json TEXT NOT NULL DEFAULT '[]',
    summary_status TEXT NOT NULL DEFAULT 'not_requested',
    summary_model_id TEXT NULL,
    summary_text TEXT NULL,
    summary_error TEXT NULL,
    summary_updated_at BIGINT NULL
);

CREATE INDEX IF NOT EXISTS idx_transcription_records_created_at ON transcription_records(created_at DESC);

CREATE TABLE IF NOT EXISTS diarization_records (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    model_id TEXT NULL,
    asr_model_id TEXT NULL,
    aligner_model_id TEXT NULL,
    llm_model_id TEXT NULL,
    processing_status TEXT NOT NULL DEFAULT 'ready',
    processing_error TEXT NULL,
    min_speakers BIGINT NULL,
    max_speakers BIGINT NULL,
    min_speech_duration_ms DOUBLE PRECISION NULL,
    min_silence_duration_ms DOUBLE PRECISION NULL,
    enable_llm_refinement BIGINT NOT NULL,
    processing_time_ms DOUBLE PRECISION NOT NULL,
    duration_secs DOUBLE PRECISION NULL,
    rtf DOUBLE PRECISION NULL,
    speaker_count BIGINT NOT NULL,
    alignment_coverage DOUBLE PRECISION NULL,
    unattributed_words BIGINT NOT NULL,
    llm_refined BIGINT NOT NULL,
    asr_text TEXT NOT NULL,
    raw_transcript TEXT NOT NULL,
    transcript TEXT NOT NULL,
    summary_status TEXT NOT NULL DEFAULT 'not_requested',
    summary_model_id TEXT NULL,
    summary_text TEXT NULL,
    summary_error TEXT NULL,
    summary_updated_at BIGINT NULL,
    segments_json TEXT NOT NULL,
    words_json TEXT NOT NULL,
    utterances_json TEXT NOT NULL,
    speaker_name_overrides_json TEXT NOT NULL DEFAULT '{}',
    audio_mime_type TEXT NOT NULL,
    audio_filename TEXT NULL,
    audio_storage_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_diarization_records_created_at ON diarization_records(created_at DESC);

CREATE TABLE IF NOT EXISTS speech_history_records (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    route_kind TEXT NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'ready',
    processing_error TEXT NULL,
    runtime_stage_id TEXT NULL,
    runtime_attempt_token TEXT NULL,
    model_id TEXT NULL,
    speaker TEXT NULL,
    language TEXT NULL,
    saved_voice_id TEXT NULL,
    speed DOUBLE PRECISION NULL,
    input_text TEXT NOT NULL,
    voice_description TEXT NULL,
    reference_text TEXT NULL,
    generation_time_ms DOUBLE PRECISION NOT NULL,
    audio_duration_secs DOUBLE PRECISION NULL,
    rtf DOUBLE PRECISION NULL,
    tokens_generated BIGINT NULL,
    audio_mime_type TEXT NOT NULL,
    audio_filename TEXT NULL,
    audio_storage_path TEXT NOT NULL,
    audio_media_asset_id TEXT NULL,
    audio_artifact_tenant TEXT NULL,
    stop_reason TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_speech_history_route_created_at ON speech_history_records(route_kind, created_at DESC);

CREATE TABLE IF NOT EXISTS saved_voices (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    name TEXT NOT NULL ,
    reference_text TEXT NOT NULL,
    audio_mime_type TEXT NOT NULL,
    audio_filename TEXT NULL,
    audio_storage_path TEXT NOT NULL,
    source_route_kind TEXT NULL,
    source_record_id TEXT NULL,
    permission_scope TEXT NOT NULL DEFAULT 'legacy_local',
    consent_status TEXT NOT NULL DEFAULT 'granted',
    allowed_uses_json TEXT NOT NULL DEFAULT '["local_tts","voice_clone_reference"]',
    permission_provenance TEXT NULL,
    permission_revoked_at BIGINT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_voices_name_nocase ON saved_voices(LOWER(name));

CREATE INDEX IF NOT EXISTS idx_saved_voices_updated_at ON saved_voices(updated_at DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS studio_projects (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    name TEXT NOT NULL,
    source_filename TEXT NULL,
    source_text TEXT NOT NULL,
    model_id TEXT NULL,
    voice_mode TEXT NOT NULL,
    speaker TEXT NULL,
    saved_voice_id TEXT NULL,
    speed DOUBLE PRECISION NULL
);

CREATE TABLE IF NOT EXISTS studio_project_segments (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    position BIGINT NOT NULL,
    text TEXT NOT NULL,
    model_id TEXT NULL,
    voice_mode TEXT NULL,
    speaker TEXT NULL,
    saved_voice_id TEXT NULL,
    speech_record_id TEXT NULL,
    updated_at BIGINT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS studio_project_folders (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    name TEXT NOT NULL,
    parent_id TEXT NULL,
    sort_order BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS studio_project_meta (
    project_id TEXT PRIMARY KEY,
    folder_id TEXT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    default_export_format TEXT NOT NULL DEFAULT 'wav',
    last_render_job_id TEXT NULL,
    last_rendered_at BIGINT NULL,
    FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE,
    FOREIGN KEY(folder_id) REFERENCES studio_project_folders(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS studio_project_pronunciations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_text TEXT NOT NULL,
    replacement_text TEXT NOT NULL,
    locale TEXT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS studio_project_snapshots (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    label TEXT NULL,
    project_json TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS studio_project_render_jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT NULL,
    queued_segment_ids_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_studio_projects_updated_at ON studio_projects(updated_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_studio_project_segments_project_position ON studio_project_segments(project_id, position ASC);

CREATE INDEX IF NOT EXISTS idx_studio_project_folders_parent ON studio_project_folders(parent_id, sort_order ASC, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_studio_project_pronunciations_project ON studio_project_pronunciations(project_id, updated_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_studio_project_snapshots_project ON studio_project_snapshots(project_id, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_studio_project_render_jobs_project ON studio_project_render_jobs(project_id, created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS media_assets (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    asset_kind TEXT NOT NULL,
    storage_namespace TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    content_type TEXT NOT NULL,
    filename TEXT NULL,
    size_bytes BIGINT NOT NULL,
    sha256 TEXT NULL,
    duration_secs DOUBLE PRECISION NULL,
    sample_rate_hz BIGINT NULL,
    channel_count BIGINT NULL,
    peak_amplitude DOUBLE PRECISION NULL,
    rms_amplitude DOUBLE PRECISION NULL,
    source_asset_id TEXT NULL,
    canonical_profile_version TEXT NULL,
    scan_status TEXT NOT NULL DEFAULT 'not_scanned',
    retention_policy TEXT NOT NULL DEFAULT 'default',
    deleted_at BIGINT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(source_asset_id) REFERENCES media_assets(id) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_media_assets_storage_key ON media_assets(storage_key);

CREATE INDEX IF NOT EXISTS idx_media_assets_created_at ON media_assets(created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS artifact_cleanup_intents (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    available_at BIGINT NOT NULL,
    storage_key TEXT NOT NULL UNIQUE,
    tenant_scope TEXT NOT NULL,
    reason TEXT NOT NULL CHECK(reason = 'artifact_deleted'),
    attempt_count BIGINT NOT NULL DEFAULT 0,
    last_error TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifact_cleanup_due ON artifact_cleanup_intents(available_at ASC, created_at ASC, id ASC);

CREATE TABLE IF NOT EXISTS provider_write_reservations (
    write_id TEXT PRIMARY KEY,
    reservation_token TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    expires_at BIGINT NOT NULL,
    available_at BIGINT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('reserved', 'stored', 'cleanup_pending', 'cleanup_claimed')),
    tenant_scope TEXT NOT NULL,
    storage_namespace TEXT NOT NULL,
    content_type TEXT NOT NULL,
    filename TEXT NULL,
    expected_size_bytes BIGINT NOT NULL,
    expected_sha256 TEXT NOT NULL,
    provider_request_json TEXT NULL,
    storage_key TEXT NULL,
    cleanup_claim_token TEXT NULL,
    cleanup_claim_expires_at BIGINT NULL,
    cleanup_attempt_count BIGINT NOT NULL DEFAULT 0,
    last_error TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_provider_write_cleanup_due ON provider_write_reservations(state, available_at ASC, expires_at ASC, created_at ASC, write_id ASC);

CREATE TABLE IF NOT EXISTS text_assets (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    language_hint TEXT NULL,
    character_count BIGINT NOT NULL,
    sha256 TEXT NULL,
    safety_status TEXT NOT NULL DEFAULT 'unchecked',
    retention_policy TEXT NOT NULL DEFAULT 'default',
    structure_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_text_assets_created_at ON text_assets(created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS runtime_jobs (
    id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    queued_at BIGINT NULL,
    started_at BIGINT NULL,
    finished_at BIGINT NULL,
    job_kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing', 'completed', 'failed', 'cancelled', 'expired')),
    priority BIGINT NOT NULL DEFAULT 0,
    model_id TEXT NULL,
    capability TEXT NULL,
    route_record_kind TEXT NULL,
    route_record_id TEXT NULL,
    input_media_asset_id TEXT NULL,
    input_text_asset_id TEXT NULL,
    request_json TEXT NOT NULL DEFAULT '{}',
    model_snapshot_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NULL,
    error_code TEXT NULL,
    error_message TEXT NULL,
    attempt_count BIGINT NOT NULL DEFAULT 0,
    max_attempts BIGINT NOT NULL DEFAULT 1,
    retry_policy_json TEXT NOT NULL DEFAULT '{}',
    idempotency_key TEXT NULL,
    correlation_id TEXT NULL,
    cancellation_reason TEXT NULL,
    cancellation_state TEXT NULL,
    FOREIGN KEY(input_media_asset_id) REFERENCES media_assets(id) ON DELETE SET NULL,
    FOREIGN KEY(input_text_asset_id) REFERENCES text_assets(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_runtime_jobs_status_priority ON runtime_jobs(status, priority DESC, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_runtime_jobs_route_record ON runtime_jobs(route_record_kind, route_record_id);

CREATE INDEX IF NOT EXISTS idx_runtime_jobs_idempotency_key ON runtime_jobs(idempotency_key);

CREATE TABLE IF NOT EXISTS job_stages (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    sequence BIGINT NOT NULL,
    stage_kind TEXT NOT NULL,
    queue_class TEXT NOT NULL DEFAULT 'batch',
    resource_hints_json TEXT NOT NULL DEFAULT '{"version":1}',
    resource_target TEXT NOT NULL DEFAULT 'any',
    required_backend TEXT NULL,
    required_device_class TEXT NULL,
    min_resource_memory_bytes BIGINT NULL,
    resource_concurrency_weight BIGINT NOT NULL DEFAULT 1,
    status TEXT NOT NULL CHECK(status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing', 'completed', 'failed', 'cancelled', 'expired', 'skipped')),
    capability TEXT NULL,
    model_id TEXT NULL,
    worker_id TEXT NULL,
    lease_expires_at BIGINT NULL,
    available_at BIGINT NULL,
    attempt_token TEXT NULL,
    attempt_count BIGINT NOT NULL DEFAULT 0,
    max_attempts BIGINT NOT NULL DEFAULT 1,
    input_artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    output_artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    progress_json TEXT NULL,
    started_at BIGINT NULL,
    finished_at BIGINT NULL,
    error_code TEXT NULL,
    error_message TEXT NULL,
    cancellation_state TEXT NULL,
    FOREIGN KEY(job_id) REFERENCES runtime_jobs(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_job_stages_job_sequence ON job_stages(job_id, sequence);

CREATE INDEX IF NOT EXISTS idx_job_stages_status_lease ON job_stages(status, lease_expires_at, sequence);

CREATE TABLE IF NOT EXISTS runtime_artifacts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    stage_id TEXT NULL,
    producer_attempt_count BIGINT NULL,
    producer_attempt_token TEXT NULL,
    publication_key TEXT NULL,
    created_at BIGINT NOT NULL,
    artifact_kind TEXT NOT NULL,
    artifact_role TEXT NOT NULL,
    media_asset_id TEXT NULL,
    text_asset_id TEXT NULL,
    storage_key TEXT NULL,
    content_type TEXT NULL,
    filename TEXT NULL,
    size_bytes BIGINT NULL,
    sha256 TEXT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    retention_policy TEXT NOT NULL DEFAULT 'default',
    FOREIGN KEY(job_id) REFERENCES runtime_jobs(id) ON DELETE CASCADE,
    FOREIGN KEY(stage_id) REFERENCES job_stages(id) ON DELETE SET NULL,
    FOREIGN KEY(media_asset_id) REFERENCES media_assets(id) ON DELETE SET NULL,
    FOREIGN KEY(text_asset_id) REFERENCES text_assets(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_job_role ON runtime_artifacts(job_id, artifact_role, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_stage ON runtime_artifacts(stage_id);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    expires_at BIGINT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NULL,
    runtime_job_id TEXT NULL,
    conflict_message TEXT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(operation, idempotency_key),
    FOREIGN KEY(runtime_job_id) REFERENCES runtime_jobs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_idempotency_keys_runtime_job ON idempotency_keys(runtime_job_id);

CREATE TABLE IF NOT EXISTS durable_idempotency_keys_v2 (
    tenant_scope TEXT NOT NULL,
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    expires_at BIGINT NOT NULL,
    digest_version BIGINT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('reserved', 'committed')),
    reservation_token TEXT NOT NULL,
    runtime_job_id TEXT NULL,
    response_json TEXT NULL,
    PRIMARY KEY(tenant_scope, operation, idempotency_key),
    FOREIGN KEY(runtime_job_id) REFERENCES runtime_jobs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_durable_idempotency_v2_expiry ON durable_idempotency_keys_v2(expires_at ASC, created_at ASC, tenant_scope ASC, operation ASC, idempotency_key ASC);

CREATE INDEX IF NOT EXISTS idx_durable_idempotency_v2_runtime_job ON durable_idempotency_keys_v2(runtime_job_id);

CREATE TABLE IF NOT EXISTS runtime_worker_heartbeats (
    worker_id TEXT PRIMARY KEY,
    started_at BIGINT NOT NULL,
    last_heartbeat_at BIGINT NOT NULL,
    status TEXT NOT NULL,
    queue_names_json TEXT NOT NULL DEFAULT '[]',
    instance_id TEXT NOT NULL DEFAULT '',
    registration_version BIGINT NOT NULL DEFAULT 1,
    registration_json TEXT NOT NULL DEFAULT '{}',
    heartbeat_version BIGINT NOT NULL DEFAULT 1,
    available_slots BIGINT NOT NULL DEFAULT 0,
    heartbeat_details_json TEXT NOT NULL DEFAULT '{}',
    current_job_id TEXT NULL,
    current_stage_id TEXT NULL,
    diagnostic_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(current_job_id) REFERENCES runtime_jobs(id) ON DELETE SET NULL,
    FOREIGN KEY(current_stage_id) REFERENCES job_stages(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_runtime_worker_heartbeats_last ON runtime_worker_heartbeats(last_heartbeat_at DESC);

CREATE TABLE IF NOT EXISTS gateway_principal_keys (
    principal_id TEXT PRIMARY KEY,
    roles_json TEXT NOT NULL,
    tenant_id TEXT NULL,
    key_salt TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

ALTER TABLE provider_write_reservations ADD COLUMN IF NOT EXISTS provider_request_json TEXT NULL;

ALTER TABLE runtime_jobs ADD COLUMN IF NOT EXISTS cancellation_state TEXT NULL;

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS cancellation_state TEXT NULL;

ALTER TABLE runtime_jobs ADD COLUMN IF NOT EXISTS admission_tenant TEXT NULL;

ALTER TABLE runtime_jobs ADD COLUMN IF NOT EXISTS replay_swept_at BIGINT NULL;

ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS system_prompt TEXT NULL;

ALTER TABLE media_assets ADD COLUMN IF NOT EXISTS source_asset_id TEXT NULL;

ALTER TABLE media_assets ADD COLUMN IF NOT EXISTS canonical_profile_version TEXT NULL;

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS queue_class TEXT NOT NULL DEFAULT 'batch';

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS resource_hints_json TEXT NOT NULL DEFAULT '{"version":1}';

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS resource_target TEXT NOT NULL DEFAULT 'any';

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS required_backend TEXT NULL;

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS required_device_class TEXT NULL;

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS min_resource_memory_bytes BIGINT NULL;

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS resource_concurrency_weight BIGINT NOT NULL DEFAULT 1;

ALTER TABLE runtime_worker_heartbeats ADD COLUMN IF NOT EXISTS instance_id TEXT NOT NULL DEFAULT '';

ALTER TABLE runtime_worker_heartbeats ADD COLUMN IF NOT EXISTS registration_version BIGINT NOT NULL DEFAULT 1;

ALTER TABLE runtime_worker_heartbeats ADD COLUMN IF NOT EXISTS registration_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE runtime_worker_heartbeats ADD COLUMN IF NOT EXISTS heartbeat_version BIGINT NOT NULL DEFAULT 1;

ALTER TABLE runtime_worker_heartbeats ADD COLUMN IF NOT EXISTS available_slots BIGINT NOT NULL DEFAULT 0;

ALTER TABLE runtime_worker_heartbeats ADD COLUMN IF NOT EXISTS heartbeat_details_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS available_at BIGINT NULL;

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS attempt_token TEXT NULL;

ALTER TABLE runtime_artifacts ADD COLUMN IF NOT EXISTS producer_attempt_count BIGINT NULL;

ALTER TABLE runtime_artifacts ADD COLUMN IF NOT EXISTS producer_attempt_token TEXT NULL;

ALTER TABLE runtime_artifacts ADD COLUMN IF NOT EXISTS publication_key TEXT NULL;

ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS content_parts TEXT NULL;

ALTER TABLE onboarding_state ADD COLUMN IF NOT EXISTS analytics_opt_in BIGINT NOT NULL DEFAULT 0;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS transcription_mode TEXT NOT NULL DEFAULT 'transcription';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS aligner_model_id TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS processing_status TEXT NOT NULL DEFAULT 'ready';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS processing_error TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS processing_progress_json TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS runtime_stage_id TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS runtime_attempt_token TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS segments_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS words_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS speaker_attributed_text TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS speaker_turns_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS saa_status TEXT NOT NULL DEFAULT 'not_requested';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS saa_warnings_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS summary_status TEXT NOT NULL DEFAULT 'not_requested';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS summary_model_id TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS summary_text TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS summary_error TEXT NULL;

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS summary_updated_at BIGINT NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS saved_voice_id TEXT NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS speed DOUBLE PRECISION NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS processing_status TEXT NOT NULL DEFAULT 'ready';

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS processing_error TEXT NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS runtime_stage_id TEXT NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS runtime_attempt_token TEXT NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS audio_media_asset_id TEXT NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS audio_artifact_tenant TEXT NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS stop_reason TEXT NULL;

ALTER TABLE saved_voices ADD COLUMN IF NOT EXISTS permission_scope TEXT NOT NULL DEFAULT 'legacy_local';

ALTER TABLE saved_voices ADD COLUMN IF NOT EXISTS consent_status TEXT NOT NULL DEFAULT 'granted';

ALTER TABLE saved_voices ADD COLUMN IF NOT EXISTS allowed_uses_json TEXT NOT NULL DEFAULT '["local_tts","voice_clone_reference"]';

ALTER TABLE saved_voices ADD COLUMN IF NOT EXISTS permission_provenance TEXT NULL;

ALTER TABLE saved_voices ADD COLUMN IF NOT EXISTS permission_revoked_at BIGINT NULL;

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS speaker_name_overrides_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS processing_status TEXT NOT NULL DEFAULT 'ready';

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS processing_error TEXT NULL;

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS summary_status TEXT NOT NULL DEFAULT 'not_requested';

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS summary_model_id TEXT NULL;

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS summary_text TEXT NULL;

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS summary_error TEXT NULL;

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS summary_updated_at BIGINT NULL;

ALTER TABLE studio_project_segments ADD COLUMN IF NOT EXISTS model_id TEXT NULL;

ALTER TABLE studio_project_segments ADD COLUMN IF NOT EXISTS voice_mode TEXT NULL;

ALTER TABLE studio_project_segments ADD COLUMN IF NOT EXISTS speaker TEXT NULL;

ALTER TABLE studio_project_segments ADD COLUMN IF NOT EXISTS saved_voice_id TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_job_stages_claim_ready ON job_stages(status, available_at, lease_expires_at, sequence);

UPDATE job_stages SET queue_class = CASE WHEN stage_kind IN ('asr_transcribe', 'asr_infer') THEN 'batch_asr' WHEN stage_kind IN ('tts_synthesize', 'tts_generate') THEN 'batch_tts' WHEN stage_kind IN ('diarization', 'diarization_segment') THEN 'diarization' WHEN stage_kind IN ('export', 'encode', 'notify') THEN 'export' WHEN stage_kind IN ('evaluation', 'evaluate') THEN 'evaluation' ELSE queue_class END WHERE queue_class = 'batch';

CREATE INDEX IF NOT EXISTS idx_job_stages_queue_claim_ready ON job_stages(queue_class, status, available_at, lease_expires_at, sequence);

CREATE INDEX IF NOT EXISTS idx_job_stages_queue_resources ON job_stages(queue_class, resource_target, required_backend, required_device_class, min_resource_memory_bytes, resource_concurrency_weight, status);

CREATE UNIQUE INDEX IF NOT EXISTS idx_runtime_artifacts_attempt_publication ON runtime_artifacts(stage_id, producer_attempt_token, publication_key);

CREATE UNIQUE INDEX IF NOT EXISTS idx_media_assets_source_profile ON media_assets(source_asset_id, canonical_profile_version);

CREATE UNIQUE INDEX IF NOT EXISTS idx_speech_history_audio_media_asset ON speech_history_records(audio_media_asset_id);

CREATE TABLE IF NOT EXISTS runtime_quota_usage (
    quota_kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    used BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY(quota_kind, scope)
);

CREATE INDEX IF NOT EXISTS idx_durable_idempotency_v2_tenant_expiry ON durable_idempotency_keys_v2(tenant_scope, expires_at ASC);

CREATE INDEX IF NOT EXISTS idx_runtime_jobs_status_updated ON runtime_jobs(status, updated_at ASC);

CREATE INDEX IF NOT EXISTS idx_runtime_jobs_replay_sweep ON runtime_jobs(replay_swept_at, updated_at ASC);

CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_job_publication ON runtime_artifacts(job_id, publication_key);

CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_media_asset ON runtime_artifacts(media_asset_id);

CREATE INDEX IF NOT EXISTS idx_media_assets_retention ON media_assets(retention_policy, deleted_at, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_media_assets_deleted ON media_assets(deleted_at ASC);

CREATE INDEX IF NOT EXISTS idx_idempotency_keys_created ON idempotency_keys(created_at ASC);

INSERT INTO runtime_quota_usage (quota_kind, scope, used) SELECT 'durable_idempotency', tenant_scope, COUNT(*) FROM durable_idempotency_keys_v2 WHERE 1 = 1 GROUP BY tenant_scope ON CONFLICT (quota_kind, scope) DO NOTHING;

INSERT INTO runtime_quota_usage (quota_kind, scope, used) SELECT 'durable_idempotency', '', COUNT(*) FROM durable_idempotency_keys_v2 WHERE 1 = 1 ON CONFLICT (quota_kind, scope) DO NOTHING;

INSERT INTO runtime_quota_usage (quota_kind, scope, used) SELECT 'provider_write', tenant_scope, COUNT(*) FROM provider_write_reservations WHERE 1 = 1 GROUP BY tenant_scope ON CONFLICT (quota_kind, scope) DO NOTHING;

INSERT INTO runtime_quota_usage (quota_kind, scope, used) SELECT 'provider_write', '', COUNT(*) FROM provider_write_reservations WHERE 1 = 1 ON CONFLICT (quota_kind, scope) DO NOTHING;

ALTER TABLE job_stages ADD COLUMN IF NOT EXISTS resource_hints_version BIGINT NOT NULL DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_job_stages_claimable ON job_stages(queue_class, available_at) WHERE status IN ('queued', 'retrying');

INSERT INTO runtime_quota_usage (quota_kind, scope, used) SELECT 'artifact_cleanup', tenant_scope, COUNT(*) FROM artifact_cleanup_intents WHERE 1 = 1 GROUP BY tenant_scope ON CONFLICT (quota_kind, scope) DO NOTHING;

INSERT INTO runtime_quota_usage (quota_kind, scope, used) SELECT 'artifact_cleanup', '', COUNT(*) FROM artifact_cleanup_intents WHERE 1 = 1 ON CONFLICT (quota_kind, scope) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_runtime_jobs_input_media ON runtime_jobs(input_media_asset_id);

CREATE INDEX IF NOT EXISTS idx_transcription_records_audio_path ON transcription_records(audio_storage_path);

CREATE INDEX IF NOT EXISTS idx_diarization_records_audio_path ON diarization_records(audio_storage_path);

CREATE INDEX IF NOT EXISTS idx_speech_history_audio_path ON speech_history_records(audio_storage_path);

CREATE INDEX IF NOT EXISTS idx_saved_voices_audio_path ON saved_voices(audio_storage_path);

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS alignment_warnings_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE transcription_records ADD COLUMN IF NOT EXISTS owner_node_id TEXT NULL;

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS warnings_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS language TEXT NULL;

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS vad_gate BIGINT NULL;

ALTER TABLE diarization_records ADD COLUMN IF NOT EXISTS owner_node_id TEXT NULL;

ALTER TABLE speech_history_records ADD COLUMN IF NOT EXISTS owner_node_id TEXT NULL;

ALTER TABLE runtime_artifacts ALTER COLUMN publication_key TYPE TEXT COLLATE "C";

ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS model_id TEXT NULL;

ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS finish_reason TEXT NULL;

ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS reasoning_ms BIGINT NULL;

ALTER TABLE studio_project_segments ADD COLUMN IF NOT EXISTS render_fingerprint TEXT NULL;

ALTER TABLE studio_project_segments ADD COLUMN IF NOT EXISTS render_status TEXT NULL;

ALTER TABLE studio_project_segments ADD COLUMN IF NOT EXISTS render_error TEXT NULL;

CREATE TABLE IF NOT EXISTS schema_migrations (
    component TEXT NOT NULL,
    version BIGINT NOT NULL,
    name TEXT NOT NULL,
    applied_at BIGINT NOT NULL,
    PRIMARY KEY(component, version)
);

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 1, 'baseline', 0) ON CONFLICT DO NOTHING;

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 2, 'stage claim columns', 0) ON CONFLICT DO NOTHING;

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 3, 'artifact cleanup quota', 0) ON CONFLICT DO NOTHING;

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 4, 'media reference indexes', 0) ON CONFLICT DO NOTHING;

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 5, 'diarization alignment and node ownership', 0) ON CONFLICT DO NOTHING;

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 6, 'byte-ordered artifact publication keys', 0) ON CONFLICT DO NOTHING;

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 7, 'chat message model and finish reason', 0) ON CONFLICT DO NOTHING;

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 8, 'chat message reasoning time', 0) ON CONFLICT DO NOTHING;

INSERT INTO schema_migrations (component, version, name, applied_at) VALUES ('app', 9, 'studio segment render state', 0) ON CONFLICT DO NOTHING;

INSERT INTO voice_profiles (id, name, system_prompt, observational_memory_enabled, created_at, updated_at) VALUES ('voice_profile_default', 'Primary Voice Profile', 'You are a helpful voice assistant. Reply with concise spoken-friendly language. Avoid markdown. Do not output <think> tags or internal reasoning. Return only the final spoken answer. Keep responses brief unless asked for details.', 1, 0, 0) ON CONFLICT (id) DO NOTHING;

COMMIT;
