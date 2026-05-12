//! OpenAPI document served by the local API server.

use axum::Json;
use serde::Serialize;
use serde_json::{Map, Value, json};
use utoipa::{OpenApi, ToSchema};

use crate::api::admin::models::{
    AdminModelActionResponse, AdminModelDownloadProgressEvent, AdminModelInfo,
    AdminModelRouteCapabilities, AdminModelsResponse, AdminSpeechModelCapabilities,
};
use crate::api::openai::audio::align::{
    AlignmentJsonRequest, AlignmentMultipartRequest, AlignmentResponse, AlignmentWord,
    VerboseAlignmentResponse,
};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Izwi API",
        version = env!("CARGO_PKG_VERSION"),
        description = "Local HTTP API for the Izwi runtime, OpenAI-compatible endpoints, and first-party workflow surfaces."
    ),
    servers(
        (url = "/", description = "Local Izwi server")
    ),
    paths(
        livez,
        readyz,
        list_models,
        get_model,
        create_chat_completion,
        create_speech,
        create_transcription,
        create_alignment,
        create_response,
        get_response,
        delete_response,
        cancel_response,
        list_response_input_items,
    ),
    components(schemas(
        ApiErrorBody,
        ApiErrorEnvelope,
        AdminModelActionResponse,
        AdminModelDownloadProgressEvent,
        AdminModelInfo,
        AdminModelRouteCapabilities,
        AdminModelsResponse,
        AdminSpeechModelCapabilities,
        AlignmentJsonRequest,
        AlignmentMultipartRequest,
        AlignmentResponse,
        AlignmentWord,
        ChatCompletionChoice,
        ChatCompletionChunk,
        ChatCompletionDelta,
        ChatCompletionMessage,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionStreamOptions,
        CursorPagination,
        CursorPaginationQuery,
        LiveResponse,
        OpenAiModel,
        OpenAiModelsResponse,
        ProbeCheck,
        ReadyResponse,
        ResponseDeletedObject,
        ResponseInputItemsList,
        ResponseObject,
        ResponsesCreateRequest,
        ServerSentEvent,
        SpeechRequest,
        SpeechStreamEvent,
        TranscriptionJsonRequest,
        TranscriptionMultipartRequest,
        TranscriptionResponse,
        TranscriptionTimestampSegment,
        TranscriptionTimestampWord,
        Usage,
        VerboseAlignmentResponse,
        VerboseTranscriptionResponse,
    )),
    tags(
        (name = "Runtime", description = "Runtime health, readiness, and operational probes"),
        (name = "OpenAI Compatible", description = "OpenAI-compatible API surface, including preview compatibility routes"),
        (name = "Admin", description = "Local administrative API surface")
    )
)]
pub struct IzwiOpenApi;

pub async fn openapi_json() -> Json<Value> {
    Json(document())
}

pub fn document() -> Value {
    let mut doc = serde_json::to_value(IzwiOpenApi::openapi())
        .expect("generated OpenAPI document should serialize");
    add_scalar_navigation_paths(&mut doc);
    doc
}

fn add_scalar_navigation_paths(doc: &mut Value) {
    // Utoipa owns rich schemas for the stable contract. These lightweight path
    // items keep Scalar navigation in sync with preview routes documented in
    // docs/user/api.md while those contracts continue to mature.
    add_tag(
        doc,
        "Speech to Text",
        "Preview persisted transcription workflows",
    );
    add_tag(
        doc,
        "Diarization",
        "Preview speaker diarization and speaker-labeling workflows",
    );
    add_tag(
        doc,
        "Text to Speech",
        "Preview persisted speech generation and saved voice workflows",
    );
    add_tag(doc, "Studio", "Preview Studio project workflow APIs");
    add_tag(doc, "Chat", "Preview chat thread and agent session APIs");
    add_tag(
        doc,
        "Voice",
        "Preview voice profile, memory, and session APIs",
    );
    add_tag(doc, "Media", "Preview local media serving APIs");
    add_tag(
        doc,
        "Preferences",
        "Preview onboarding and user preference APIs",
    );
    add_tag(doc, "Realtime", "Preview WebSocket realtime APIs");
    add_tag(
        doc,
        "Reference",
        "Local API reference and OpenAPI document routes",
    );

    let paths = doc
        .get_mut("paths")
        .and_then(Value::as_object_mut)
        .expect("OpenAPI document should contain a paths object");

    add_operation(
        paths,
        "/docs",
        "get",
        "Reference",
        "Get API reference",
        "Open the local Scalar API reference UI.",
        ok_response(),
    );
    add_operation(
        paths,
        "/docs/scalar.js",
        "get",
        "Reference",
        "Get Scalar JavaScript",
        "Fetch the bundled Scalar API reference JavaScript asset.",
        ok_response(),
    );
    add_operation(
        paths,
        "/openapi.json",
        "get",
        "Reference",
        "Get OpenAPI document",
        "Fetch the generated OpenAPI document for the local server.",
        ok_response(),
    );

    add_operation(
        paths,
        "/v1/live",
        "get",
        "Runtime",
        "Check liveness",
        "Server process is alive.",
        ok_response(),
    );
    add_operation(
        paths,
        "/v1/ready",
        "get",
        "Runtime",
        "Check readiness",
        "Server readiness probe under the versioned namespace.",
        response_with_statuses(&[("200", "Ready"), ("503", "Alive but not ready")]),
    );
    add_operation(
        paths,
        "/v1/health",
        "get",
        "Runtime",
        "Get health details",
        "Rich backend, device, dtype, CUDA, and fused-attention status used by izwi status.",
        ok_response(),
    );
    add_operation(
        paths,
        "/v1/metrics",
        "get",
        "Runtime",
        "Get metrics",
        "JSON runtime telemetry snapshot.",
        ok_response(),
    );
    add_operation(
        paths,
        "/v1/metrics/prometheus",
        "get",
        "Runtime",
        "Get Prometheus metrics",
        "Runtime telemetry in Prometheus text format.",
        ok_response(),
    );
    add_operation(
        paths,
        "/internal/health",
        "get",
        "Runtime",
        "Get internal health details",
        "Compatibility alias for rich runtime health.",
        ok_response(),
    );
    add_operation(
        paths,
        "/internal/live",
        "get",
        "Runtime",
        "Check internal liveness",
        "Compatibility alias for liveness.",
        ok_response(),
    );
    add_operation(
        paths,
        "/internal/ready",
        "get",
        "Runtime",
        "Check internal readiness",
        "Compatibility alias for readiness.",
        response_with_statuses(&[("200", "Ready"), ("503", "Alive but not ready")]),
    );
    add_operation(
        paths,
        "/internal/metrics",
        "get",
        "Runtime",
        "Get internal metrics",
        "Compatibility alias for JSON runtime telemetry.",
        ok_response(),
    );
    add_operation(
        paths,
        "/internal/metrics/prometheus",
        "get",
        "Runtime",
        "Get internal Prometheus metrics",
        "Compatibility alias for Prometheus runtime telemetry.",
        ok_response(),
    );

    add_operation(
        paths,
        "/v1/admin/models",
        "get",
        "Admin",
        "List model variants",
        "List all known enabled model variants, local status, modalities, and route capabilities.",
        json_schema_response(
            "AdminModelsResponse",
            "Model variants with local lifecycle state and voice-app capabilities",
        ),
    );
    add_operation_with_params(
        paths,
        "/v1/admin/models/{variant}",
        "get",
        "Admin",
        "Get model variant",
        "Fetch one model variant's local lifecycle state and voice-app capabilities.",
        &[("variant", "Model variant identifier")],
        json_schema_response(
            "AdminModelInfo",
            "Model lifecycle state and voice-app capabilities",
        ),
    );
    add_operation_with_params(
        paths,
        "/v1/admin/models/{variant}",
        "delete",
        "Admin",
        "Delete model variant",
        "Unload and delete local model files for a variant.",
        &[("variant", "Model variant identifier")],
        json_schema_response("AdminModelActionResponse", "Model delete result"),
    );
    add_operation_with_params(
        paths,
        "/v1/admin/models/{variant}/download",
        "post",
        "Admin",
        "Start model download",
        "Start a model download in the background.",
        &[("variant", "Model variant identifier")],
        json_schema_response("AdminModelActionResponse", "Model download start result"),
    );
    add_operation_with_params(
        paths,
        "/v1/admin/models/{variant}/download/progress",
        "get",
        "Admin",
        "Stream download progress",
        "Server-sent model download progress events.",
        &[("variant", "Model variant identifier")],
        event_stream_schema_response(
            "AdminModelDownloadProgressEvent",
            "Server-sent model download progress events",
        ),
    );
    add_operation_with_params(
        paths,
        "/v1/admin/models/{variant}/download/cancel",
        "post",
        "Admin",
        "Cancel model download",
        "Cancel an active model download.",
        &[("variant", "Model variant identifier")],
        json_schema_response("AdminModelActionResponse", "Model download cancel result"),
    );
    add_operation_with_params(
        paths,
        "/v1/admin/models/{variant}/load",
        "post",
        "Admin",
        "Load model",
        "Load model weights into runtime memory.",
        &[("variant", "Model variant identifier")],
        json_schema_response("AdminModelActionResponse", "Model load result"),
    );
    add_operation_with_params(
        paths,
        "/v1/admin/models/{variant}/unload",
        "post",
        "Admin",
        "Unload model",
        "Unload model weights from runtime memory.",
        &[("variant", "Model variant identifier")],
        json_schema_response("AdminModelActionResponse", "Model unload result"),
    );

    add_collection(
        paths,
        "/v1/speech-to-text/jobs",
        "Speech to Text",
        "speech-to-text jobs",
        "speech-to-text job",
        "List or create canonical saved transcription and diarization jobs.",
    );
    add_get_patch_put_delete_member(
        paths,
        "/v1/speech-to-text/jobs/{record_id}",
        "Speech to Text",
        "speech-to-text job",
        "Fetch, update, or delete a canonical saved speech-text job.",
        &[("record_id", "Speech-text job identifier")],
    );
    add_operation_with_params(
        paths,
        "/v1/speech-to-text/jobs/{record_id}/audio",
        "get",
        "Speech to Text",
        "Download speech-to-text job audio",
        "Fetch stored source audio for a speech-text job.",
        &[("record_id", "Speech-text job identifier")],
        binary_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/speech-to-text/jobs/{record_id}/reruns",
        "post",
        "Speech to Text",
        "Rerun speech-to-text job",
        "Re-run diarization from stored source audio.",
        &[("record_id", "Speech-text job identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/speech-to-text/jobs/{record_id}/cancel",
        "post",
        "Speech to Text",
        "Cancel speech-to-text job",
        "Cancel an in-flight diarization job.",
        &[("record_id", "Speech-text job identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/speech-to-text/jobs/{record_id}/summary/regenerate",
        "post",
        "Speech to Text",
        "Regenerate speech-to-text summary",
        "Regenerate a transcription or diarization summary.",
        &[("record_id", "Speech-text job identifier")],
        preview_response(),
    );

    add_collection(
        paths,
        "/v1/diarizations",
        "Diarization",
        "diarization records",
        "diarization record",
        "List or create saved diarization records.",
    );
    add_get_patch_put_delete_member(
        paths,
        "/v1/diarizations/{record_id}",
        "Diarization",
        "diarization record",
        "Fetch, update, or delete a saved diarization record.",
        &[("record_id", "Diarization record identifier")],
    );
    add_operation_with_params(
        paths,
        "/v1/diarizations/{record_id}/audio",
        "get",
        "Diarization",
        "Download diarization audio",
        "Fetch stored diarization source audio.",
        &[("record_id", "Diarization record identifier")],
        binary_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/diarizations/{record_id}/reruns",
        "post",
        "Diarization",
        "Rerun diarization record",
        "Re-run diarization from a saved record.",
        &[("record_id", "Diarization record identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/diarizations/{record_id}/cancel",
        "post",
        "Diarization",
        "Cancel diarization record",
        "Cancel an in-flight diarization record.",
        &[("record_id", "Diarization record identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/diarizations/{record_id}/summary/regenerate",
        "post",
        "Diarization",
        "Regenerate diarization summary",
        "Regenerate a diarization summary.",
        &[("record_id", "Diarization record identifier")],
        preview_response(),
    );

    add_speech_history_family(paths, "/v1/text-to-speech", "text-to-speech records");
    add_speech_history_family(paths, "/v1/voice-designs", "voice design records");
    add_speech_history_family(paths, "/v1/voice-clones", "voice clone records");
    add_collection(
        paths,
        "/v1/voices",
        "Text to Speech",
        "saved voices",
        "saved voice",
        "List or create reusable saved voice references.",
    );
    add_get_delete_member(
        paths,
        "/v1/voices/{voice_id}",
        "Text to Speech",
        "saved voice",
        "Fetch or delete saved voice metadata.",
        &[("voice_id", "Saved voice identifier")],
    );
    add_operation_with_params(
        paths,
        "/v1/voices/{voice_id}/audio",
        "get",
        "Text to Speech",
        "Download saved voice audio",
        "Fetch saved voice reference audio.",
        &[("voice_id", "Saved voice identifier")],
        binary_response(),
    );

    add_collection(
        paths,
        "/v1/studio/folders",
        "Studio",
        "Studio folders",
        "Studio folder",
        "List or create Studio project folders.",
    );
    add_collection(
        paths,
        "/v1/studio/projects",
        "Studio",
        "Studio projects",
        "Studio project",
        "List or create Studio projects.",
    );
    add_get_patch_delete_member(
        paths,
        "/v1/studio/projects/{project_id}",
        "Studio",
        "Studio project",
        "Fetch, update, or delete a Studio project.",
        &[("project_id", "Studio project identifier")],
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/audio",
        "get",
        "Studio",
        "Download Studio project audio",
        "Fetch combined or selected Studio project audio.",
        &[("project_id", "Studio project identifier")],
        binary_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/meta",
        "get",
        "Studio",
        "Get Studio project metadata",
        "Fetch Studio project metadata.",
        &[("project_id", "Studio project identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/meta",
        "patch",
        "Studio",
        "Update Studio project metadata",
        "Create or update Studio project metadata.",
        &[("project_id", "Studio project identifier")],
        preview_response(),
    );
    add_collection_with_params(
        paths,
        "/v1/studio/projects/{project_id}/pronunciations",
        "Studio",
        "Studio pronunciations",
        "Studio pronunciation",
        "List or create pronunciation overrides.",
        &[("project_id", "Studio project identifier")],
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/pronunciations/{pronunciation_id}",
        "delete",
        "Studio",
        "Delete Studio pronunciation",
        "Delete a pronunciation override.",
        &[
            ("project_id", "Studio project identifier"),
            ("pronunciation_id", "Pronunciation identifier"),
        ],
        preview_response(),
    );
    add_collection_with_params(
        paths,
        "/v1/studio/projects/{project_id}/snapshots",
        "Studio",
        "Studio snapshots",
        "Studio snapshot",
        "List or create project snapshots.",
        &[("project_id", "Studio project identifier")],
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/snapshots/{snapshot_id}/restore",
        "post",
        "Studio",
        "Restore Studio snapshot",
        "Restore a project from a snapshot.",
        &[
            ("project_id", "Studio project identifier"),
            ("snapshot_id", "Snapshot identifier"),
        ],
        preview_response(),
    );
    add_collection_with_params(
        paths,
        "/v1/studio/projects/{project_id}/render-jobs",
        "Studio",
        "Studio render jobs",
        "Studio render job",
        "List or create render jobs.",
        &[("project_id", "Studio project identifier")],
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/render-jobs/{job_id}",
        "patch",
        "Studio",
        "Update Studio render job",
        "Update render job state.",
        &[
            ("project_id", "Studio project identifier"),
            ("job_id", "Render job identifier"),
        ],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/segments",
        "post",
        "Studio",
        "Create Studio segment",
        "Create a Studio project segment.",
        &[("project_id", "Studio project identifier")],
        preview_response(),
    );
    add_get_patch_delete_member(
        paths,
        "/v1/studio/projects/{project_id}/segments/{segment_id}",
        "Studio",
        "Studio segment",
        "Fetch, update, or delete a Studio project segment.",
        &[
            ("project_id", "Studio project identifier"),
            ("segment_id", "Segment identifier"),
        ],
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/segments/{segment_id}/split",
        "post",
        "Studio",
        "Split Studio segment",
        "Split a Studio project segment.",
        &[
            ("project_id", "Studio project identifier"),
            ("segment_id", "Segment identifier"),
        ],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/segments/{segment_id}/merge-next",
        "post",
        "Studio",
        "Merge Studio segment",
        "Merge a Studio project segment with the next segment.",
        &[
            ("project_id", "Studio project identifier"),
            ("segment_id", "Segment identifier"),
        ],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/segments/reorder",
        "patch",
        "Studio",
        "Reorder Studio segments",
        "Reorder Studio project segments.",
        &[("project_id", "Studio project identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/segments/bulk-delete",
        "post",
        "Studio",
        "Delete Studio segments in bulk",
        "Bulk delete Studio project segments.",
        &[("project_id", "Studio project identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/studio/projects/{project_id}/segments/{segment_id}/render",
        "post",
        "Studio",
        "Render Studio segment",
        "Render a Studio project segment.",
        &[
            ("project_id", "Studio project identifier"),
            ("segment_id", "Segment identifier"),
        ],
        preview_response(),
    );

    add_collection(
        paths,
        "/v1/chat/threads",
        "Chat",
        "chat threads",
        "chat thread",
        "List or create durable local chat threads.",
    );
    add_get_patch_delete_member(
        paths,
        "/v1/chat/threads/{thread_id}",
        "Chat",
        "chat thread",
        "Fetch, update, or delete a chat thread.",
        &[("thread_id", "Chat thread identifier")],
    );
    add_collection_with_params(
        paths,
        "/v1/chat/threads/{thread_id}/messages",
        "Chat",
        "chat messages",
        "chat message",
        "List messages or send a new user message.",
        &[("thread_id", "Chat thread identifier")],
    );
    add_operation(
        paths,
        "/v1/agent/sessions",
        "post",
        "Chat",
        "Create agent session",
        "Create preview process-local agent session metadata and a linked chat thread.",
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/agent/sessions/{session_id}",
        "get",
        "Chat",
        "Get agent session",
        "Fetch retained process-local agent session metadata.",
        &[("session_id", "Agent session identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/agent/sessions/{session_id}/turns",
        "post",
        "Chat",
        "Create agent turn",
        "Run one agent turn.",
        &[("session_id", "Agent session identifier")],
        preview_response(),
    );

    add_operation(
        paths,
        "/v1/voice/profile",
        "get",
        "Voice",
        "Get voice profile",
        "Fetch voice profile settings.",
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/voice/profile",
        "patch",
        "Voice",
        "Update voice profile",
        "Update voice profile settings.",
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/voice/observations",
        "get",
        "Voice",
        "List voice observations",
        "List voice memory observations.",
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/voice/observations",
        "delete",
        "Voice",
        "Clear voice observations",
        "Clear voice memory observations.",
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/voice/observations/{observation_id}",
        "delete",
        "Voice",
        "Delete voice observation",
        "Delete one voice memory observation.",
        &[("observation_id", "Voice observation identifier")],
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/voice/sessions",
        "get",
        "Voice",
        "List voice sessions",
        "List persisted voice sessions.",
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/voice/sessions",
        "post",
        "Voice",
        "Create voice session",
        "Create a persisted voice session shell for external applications.",
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/voice/sessions/{session_id}",
        "get",
        "Voice",
        "Get voice session",
        "Fetch one persisted voice session.",
        &[("session_id", "Voice session identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/voice/sessions/{session_id}",
        "patch",
        "Voice",
        "Update voice session",
        "Update voice session metadata such as the prompt or ended state.",
        &[("session_id", "Voice session identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/voice/sessions/{session_id}",
        "delete",
        "Voice",
        "Delete voice session",
        "Delete a persisted voice session and its turns.",
        &[("session_id", "Voice session identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/voice/sessions/{session_id}/turns",
        "get",
        "Voice",
        "List voice session turns",
        "List turns for one persisted voice session.",
        &[("session_id", "Voice session identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/voice/sessions/{session_id}/end",
        "post",
        "Voice",
        "Update voice session end state",
        "Mark a persisted voice session ended.",
        &[("session_id", "Voice session identifier")],
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/voice/sessions/{session_id}/export",
        "get",
        "Voice",
        "Download voice session export",
        "Export a persisted voice session as JSON or text transcript.",
        &[("session_id", "Voice session identifier")],
        preview_response(),
    );

    add_operation(
        paths,
        "/v1/media",
        "get",
        "Media",
        "List media",
        "List local media objects served by the OSS media route.",
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/media",
        "post",
        "Media",
        "Create media",
        "Upload a base64 media object for local app workflows.",
        preview_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/media/{path}",
        "get",
        "Media",
        "Download media",
        "Fetch persisted local media by catch-all relative path. The path may contain slashes.",
        &[(
            "path",
            "Catch-all relative media path, including nested segments",
        )],
        binary_response(),
    );
    add_operation_with_params(
        paths,
        "/v1/media/{path}",
        "delete",
        "Media",
        "Delete media",
        "Delete a persisted local media object by catch-all relative path.",
        &[(
            "path",
            "Catch-all relative media path, including nested segments",
        )],
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/onboarding",
        "get",
        "Preferences",
        "Get onboarding state",
        "Fetch first-run onboarding state.",
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/onboarding/complete",
        "post",
        "Preferences",
        "Complete onboarding",
        "Mark first-run onboarding complete.",
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/preferences",
        "get",
        "Preferences",
        "Get preferences",
        "Fetch user preferences.",
        preview_response(),
    );
    add_operation(
        paths,
        "/v1/preferences/analytics",
        "put",
        "Preferences",
        "Update analytics preference",
        "Update analytics opt-in preference.",
        preview_response(),
    );

    add_operation(
        paths,
        "/v1/speech-to-text/realtime/ws",
        "get",
        "Realtime",
        "Open transcription realtime WebSocket",
        "Upgrade to the preview transcription realtime WebSocket protocol.",
        websocket_response(),
    );
    add_operation(
        paths,
        "/v1/voice/realtime/ws",
        "get",
        "Realtime",
        "Open voice realtime WebSocket",
        "Upgrade to the preview voice realtime WebSocket protocol.",
        websocket_response(),
    );
}

fn add_speech_history_family(
    paths: &mut Map<String, Value>,
    family_path: &str,
    plural_label: &str,
) {
    let singular_label = plural_label.strip_suffix('s').unwrap_or(plural_label);
    add_collection(
        paths,
        family_path,
        "Text to Speech",
        plural_label,
        singular_label,
        "List or create persisted speech records.",
    );
    let member = format!("{family_path}/{{record_id}}");
    let audio = format!("{family_path}/{{record_id}}/audio");
    add_get_delete_member(
        paths,
        &member,
        "Text to Speech",
        singular_label,
        "Fetch or delete a persisted speech record.",
        &[("record_id", "Speech record identifier")],
    );
    add_operation_with_params(
        paths,
        &audio,
        "get",
        "Text to Speech",
        &format!("Download {singular_label} audio"),
        "Fetch generated speech audio.",
        &[("record_id", "Speech record identifier")],
        binary_response(),
    );
}

fn add_collection(
    paths: &mut Map<String, Value>,
    path: &str,
    tag: &str,
    plural_label: &str,
    singular_label: &str,
    description: &str,
) {
    add_operation(
        paths,
        path,
        "get",
        tag,
        &format!("List {plural_label}"),
        description,
        preview_response(),
    );
    add_operation(
        paths,
        path,
        "post",
        tag,
        &format!("Create {singular_label}"),
        description,
        preview_response(),
    );
}

fn add_collection_with_params(
    paths: &mut Map<String, Value>,
    path: &str,
    tag: &str,
    plural_label: &str,
    singular_label: &str,
    description: &str,
    params: &[(&str, &str)],
) {
    add_operation_with_params(
        paths,
        path,
        "get",
        tag,
        &format!("List {plural_label}"),
        description,
        params,
        preview_response(),
    );
    add_operation_with_params(
        paths,
        path,
        "post",
        tag,
        &format!("Create {singular_label}"),
        description,
        params,
        preview_response(),
    );
}

fn add_get_delete_member(
    paths: &mut Map<String, Value>,
    path: &str,
    tag: &str,
    label: &str,
    description: &str,
    params: &[(&str, &str)],
) {
    add_operation_with_params(
        paths,
        path,
        "get",
        tag,
        &format!("Get {label}"),
        description,
        params,
        preview_response(),
    );
    add_operation_with_params(
        paths,
        path,
        "delete",
        tag,
        &format!("Delete {label}"),
        description,
        params,
        preview_response(),
    );
}

fn add_get_patch_delete_member(
    paths: &mut Map<String, Value>,
    path: &str,
    tag: &str,
    label: &str,
    description: &str,
    params: &[(&str, &str)],
) {
    add_operation_with_params(
        paths,
        path,
        "get",
        tag,
        &format!("Get {label}"),
        description,
        params,
        preview_response(),
    );
    add_operation_with_params(
        paths,
        path,
        "patch",
        tag,
        &format!("Update {label}"),
        description,
        params,
        preview_response(),
    );
    add_operation_with_params(
        paths,
        path,
        "delete",
        tag,
        &format!("Delete {label}"),
        description,
        params,
        preview_response(),
    );
}

fn add_get_patch_put_delete_member(
    paths: &mut Map<String, Value>,
    path: &str,
    tag: &str,
    label: &str,
    description: &str,
    params: &[(&str, &str)],
) {
    add_operation_with_params(
        paths,
        path,
        "get",
        tag,
        &format!("Get {label}"),
        description,
        params,
        preview_response(),
    );
    add_operation_with_params(
        paths,
        path,
        "patch",
        tag,
        &format!("Update {label}"),
        description,
        params,
        preview_response(),
    );
    add_operation_with_params(
        paths,
        path,
        "put",
        tag,
        &format!("Replace {label}"),
        description,
        params,
        preview_response(),
    );
    add_operation_with_params(
        paths,
        path,
        "delete",
        tag,
        &format!("Delete {label}"),
        description,
        params,
        preview_response(),
    );
}

fn add_operation(
    paths: &mut Map<String, Value>,
    path: &str,
    method: &str,
    tag: &str,
    summary: &str,
    description: &str,
    responses: Value,
) {
    add_operation_with_params(
        paths,
        path,
        method,
        tag,
        summary,
        description,
        &[],
        responses,
    );
}

fn add_operation_with_params(
    paths: &mut Map<String, Value>,
    path: &str,
    method: &str,
    tag: &str,
    summary: &str,
    description: &str,
    params: &[(&str, &str)],
    responses: Value,
) {
    let path_item = paths.entry(path.to_string()).or_insert_with(|| json!({}));
    let path_obj = path_item
        .as_object_mut()
        .expect("OpenAPI path item should be an object");
    let mut operation = json!({
        "tags": [tag],
        "summary": summary,
        "description": description,
        "responses": responses,
    });
    if !params.is_empty() {
        operation["parameters"] = Value::Array(
            params
                .iter()
                .map(|(name, description)| {
                    json!({
                        "name": name,
                        "in": "path",
                        "required": true,
                        "description": description,
                        "schema": { "type": "string" }
                    })
                })
                .collect(),
        );
    }
    path_obj.insert(method.to_string(), operation);
}

fn add_tag(doc: &mut Value, name: &str, description: &str) {
    let tags = doc
        .get_mut("tags")
        .and_then(Value::as_array_mut)
        .expect("OpenAPI document should contain tags array");
    if tags
        .iter()
        .any(|tag| tag.get("name").and_then(Value::as_str) == Some(name))
    {
        return;
    }
    tags.push(json!({
        "name": name,
        "description": description,
    }));
}

fn ok_response() -> Value {
    response_with_statuses(&[("200", "OK")])
}

fn preview_response() -> Value {
    response_with_statuses(&[
        ("200", "OK"),
        ("400", "Invalid request"),
        ("404", "Not found"),
        ("500", "Server error"),
    ])
}

fn binary_response() -> Value {
    response_with_statuses(&[
        ("200", "Binary media response"),
        ("404", "Not found"),
        ("500", "Server error"),
    ])
}

fn json_schema_response(schema: &str, description: &str) -> Value {
    let mut responses = response_with_statuses(&[
        ("400", "Invalid request"),
        ("404", "Not found"),
        ("500", "Server error"),
    ]);
    responses
        .as_object_mut()
        .expect("OpenAPI responses should be an object")
        .insert(
            "200".to_string(),
            json!({
                "description": description,
                "content": {
                    "application/json": {
                        "schema": schema_ref(schema)
                    }
                }
            }),
        );
    responses
}

fn event_stream_schema_response(schema: &str, description: &str) -> Value {
    let mut responses = response_with_statuses(&[("404", "Not found"), ("500", "Server error")]);
    responses
        .as_object_mut()
        .expect("OpenAPI responses should be an object")
        .insert(
            "200".to_string(),
            json!({
                "description": description,
                "content": {
                    "text/event-stream": {
                        "schema": schema_ref(schema)
                    }
                }
            }),
        );
    responses
}

fn schema_ref(schema: &str) -> Value {
    json!({ "$ref": format!("#/components/schemas/{schema}") })
}

fn websocket_response() -> Value {
    response_with_statuses(&[
        ("101", "WebSocket upgrade"),
        ("400", "Invalid WebSocket request"),
    ])
}

fn response_with_statuses(statuses: &[(&str, &str)]) -> Value {
    let mut responses = Map::new();
    for (status, description) in statuses {
        responses.insert(
            (*status).to_string(),
            json!({
                "description": description,
            }),
        );
    }
    Value::Object(responses)
}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/livez",
    tag = "Runtime",
    summary = "Check liveness",
    responses(
        (status = 200, description = "Server process is alive", body = LiveResponse)
    )
)]
fn livez() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/readyz",
    tag = "Runtime",
    summary = "Check readiness",
    responses(
        (status = 200, description = "Server is ready to serve requests", body = ReadyResponse),
        (status = 503, description = "Server is alive but not ready", body = ReadyResponse)
    )
)]
fn readyz() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "OpenAI Compatible",
    summary = "List models",
    responses(
        (status = 200, description = "List locally available OpenAI-compatible models", body = OpenAiModelsResponse),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn list_models() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/v1/models/{model}",
    tag = "OpenAI Compatible",
    summary = "Get model",
    params(
        ("model" = String, Path, description = "Model identifier")
    ),
    responses(
        (status = 200, description = "Retrieve a locally available OpenAI-compatible model", body = OpenAiModel),
        (status = 400, description = "Invalid model identifier", body = ApiErrorEnvelope),
        (status = 404, description = "Model not found", body = ApiErrorEnvelope)
    )
)]
fn get_model() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "OpenAI Compatible",
    summary = "Create chat completion",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Chat completion JSON when stream is false; server-sent events when stream is true", body = ChatCompletionResponse),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_chat_completion() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    tag = "OpenAI Compatible",
    summary = "Create speech",
    request_body = SpeechRequest,
    responses(
        (status = 200, description = "Generated audio bytes, or server-sent audio events when stream_format is sse"),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_speech() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    tag = "OpenAI Compatible",
    summary = "Create transcription",
    request_body(content = TranscriptionMultipartRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Transcription result as JSON, verbose JSON, text, SRT, VTT, or server-sent events", body = TranscriptionResponse),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_transcription() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/audio/align",
    tag = "OpenAI Compatible",
    summary = "Create forced alignment",
    request_body = AlignmentJsonRequest,
    responses(
        (status = 200, description = "Word-level alignment as JSON, verbose JSON, or text", body = AlignmentResponse),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 415, description = "Unsupported media type", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_alignment() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/responses",
    tag = "OpenAI Compatible",
    summary = "Create response",
    request_body = ResponsesCreateRequest,
    responses(
        (status = 200, description = "Preview process-local response object, or server-sent events when stream is true", body = ResponseObject),
        (status = 400, description = "Invalid request", body = ApiErrorEnvelope),
        (status = 500, description = "Server error", body = ApiErrorEnvelope)
    )
)]
fn create_response() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/v1/responses/{response_id}",
    tag = "OpenAI Compatible",
    summary = "Get response",
    params(
        ("response_id" = String, Path, description = "Process-local response identifier")
    ),
    responses(
        (status = 200, description = "Preview process-local stored response record", body = ResponseObject),
        (status = 404, description = "Response not found or evicted", body = ApiErrorEnvelope)
    )
)]
fn get_response() {}

#[allow(dead_code)]
#[utoipa::path(
    delete,
    path = "/v1/responses/{response_id}",
    tag = "OpenAI Compatible",
    summary = "Delete response",
    params(
        ("response_id" = String, Path, description = "Process-local response identifier")
    ),
    responses(
        (status = 200, description = "Preview deletion result for a process-local response record", body = ResponseDeletedObject),
        (status = 404, description = "Response not found or evicted", body = ApiErrorEnvelope)
    )
)]
fn delete_response() {}

#[allow(dead_code)]
#[utoipa::path(
    post,
    path = "/v1/responses/{response_id}/cancel",
    tag = "OpenAI Compatible",
    summary = "Cancel response",
    params(
        ("response_id" = String, Path, description = "Process-local response identifier")
    ),
    responses(
        (status = 200, description = "Preview cancellation result for a process-local response record", body = ResponseObject),
        (status = 404, description = "Response not found or evicted", body = ApiErrorEnvelope)
    )
)]
fn cancel_response() {}

#[allow(dead_code)]
#[utoipa::path(
    get,
    path = "/v1/responses/{response_id}/input_items",
    tag = "OpenAI Compatible",
    summary = "List response input items",
    params(
        ("response_id" = String, Path, description = "Process-local response identifier")
    ),
    responses(
        (status = 200, description = "Preview input items captured for a process-local response record", body = ResponseInputItemsList),
        (status = 404, description = "Response not found or evicted", body = ApiErrorEnvelope)
    )
)]
fn list_response_input_items() {}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ApiErrorEnvelope {
    pub error: ApiErrorBody,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ProbeCheck {
    pub name: String,
    pub ok: bool,
    pub message: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct LiveResponse {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ReadyResponse {
    pub status: String,
    pub version: String,
    pub ready: bool,
    pub phase: String,
    pub draining: bool,
    pub uptime_secs: u64,
    pub checks: Vec<ProbeCheck>,
    pub startup_warnings: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAiModelsResponse {
    pub object: String,
    pub data: Vec<OpenAiModel>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAiModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub root: Option<String>,
    pub parent: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct CursorPaginationQuery {
    pub limit: Option<usize>,
    pub cursor: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct CursorPagination {
    pub next_cursor: Option<String>,
    pub has_more: bool,
    pub limit: usize,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub stream_options: Option<ChatCompletionStreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<serde_json::Value>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: Option<serde_json::Value>,
    pub enable_thinking: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionStreamOptions {
    pub include_usage: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
    pub izwi_generation_time_ms: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionDelta>,
    pub usage: Option<Usage>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionDelta {
    pub index: usize,
    pub delta: ChatCompletionMessage,
    pub finish_reason: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    pub allow_format_fallback: Option<bool>,
    pub speed: Option<f32>,
    pub language: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub top_k: Option<usize>,
    pub stream: Option<bool>,
    pub stream_format: Option<String>,
    pub instructions: Option<String>,
    pub reference_audio: Option<String>,
    pub reference_text: Option<String>,
    pub saved_voice_id: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct SpeechStreamEvent {
    pub event: String,
    pub request_id: Option<String>,
    pub sequence: Option<usize>,
    pub audio_base64: Option<String>,
    pub sample_count: Option<usize>,
    pub is_final: Option<bool>,
    pub sample_rate: Option<u32>,
    pub audio_format: Option<String>,
    pub tokens_generated: Option<usize>,
    pub generation_time_ms: Option<f32>,
    pub audio_duration_secs: Option<f32>,
    pub rtf: Option<f32>,
    pub error: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionMultipartRequest {
    #[schema(value_type = String, format = Binary)]
    pub file: Option<String>,
    pub audio_base64: Option<String>,
    pub model: Option<String>,
    pub aligner_model: Option<String>,
    pub language: Option<String>,
    pub response_format: Option<String>,
    pub stream: Option<bool>,
    pub prompt: Option<String>,
    pub temperature: Option<f32>,
    pub timestamp_granularities: Option<Vec<String>>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionJsonRequest {
    pub audio_base64: String,
    pub model: Option<String>,
    pub aligner_model: Option<String>,
    pub language: Option<String>,
    pub response_format: Option<String>,
    pub stream: Option<bool>,
    pub timestamp_granularities: Option<Vec<String>>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionResponse {
    pub text: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct VerboseTranscriptionResponse {
    pub text: String,
    pub language: Option<String>,
    pub duration: f32,
    pub words: Option<Vec<TranscriptionTimestampWord>>,
    pub segments: Option<Vec<TranscriptionTimestampSegment>>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub izwi_asr_diagnostics: Option<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionTimestampWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionTimestampSegment {
    pub id: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponsesCreateRequest {
    pub model: String,
    pub input: Option<serde_json::Value>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub metadata: Option<serde_json::Value>,
    pub user: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub store: Option<bool>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: Option<serde_json::Value>,
    pub enable_thinking: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: String,
    pub model: String,
    pub output: Vec<serde_json::Value>,
    pub usage: ResponseUsage,
    pub error: Option<ApiErrorBody>,
    pub metadata: Option<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseDeletedObject {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseInputItemsList {
    pub object: String,
    pub data: Vec<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, ToSchema)]
pub struct ServerSentEvent {
    pub event: Option<String>,
    pub data: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::document;
    use std::collections::HashSet;

    #[test]
    fn openapi_documents_compatibility_contract_endpoints() {
        let contract: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../docs/openai-compatibility-contract.json"
        ))
        .expect("contract should parse");
        let supported = contract["scope"]["supported_endpoints"]
            .as_array()
            .expect("supported endpoints should be an array");
        let openapi = serde_json::to_value(document()).expect("openapi should serialize");
        let paths = openapi["paths"].as_object().expect("paths should exist");

        for endpoint in supported {
            let endpoint = endpoint
                .as_str()
                .expect("supported endpoint should be a string");
            let documented = endpoint.replace(":response_id", "{response_id}");
            assert!(
                paths.contains_key(&documented),
                "{documented} should be documented"
            );
        }
    }

    #[test]
    fn openapi_marks_stable_and_preview_methods() {
        let openapi = serde_json::to_value(document()).expect("openapi should serialize");
        let paths = openapi["paths"].as_object().expect("paths should exist");

        let expected = [
            ("/v1/models", "get"),
            ("/v1/models/{model}", "get"),
            ("/v1/chat/completions", "post"),
            ("/v1/audio/speech", "post"),
            ("/v1/audio/transcriptions", "post"),
            ("/v1/audio/align", "post"),
            ("/v1/responses", "post"),
            ("/v1/responses/{response_id}", "get"),
            ("/v1/responses/{response_id}", "delete"),
            ("/v1/responses/{response_id}/cancel", "post"),
            ("/v1/responses/{response_id}/input_items", "get"),
        ];

        for (path, method) in expected {
            assert!(
                paths
                    .get(path)
                    .and_then(|operations| operations.get(method))
                    .is_some(),
                "{method} {path} should be documented"
            );
        }

        let preview_paths: HashSet<&str> = [
            "/v1/responses",
            "/v1/responses/{response_id}",
            "/v1/responses/{response_id}/cancel",
            "/v1/responses/{response_id}/input_items",
        ]
        .into_iter()
        .collect();

        for path in preview_paths {
            let operations = paths.get(path).expect("preview path should exist");
            let has_compatible_tag = operations
                .as_object()
                .expect("operations should be an object")
                .values()
                .any(|operation| {
                    operation["tags"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|tag| tag.as_str() == Some("OpenAI Compatible"))
                });
            assert!(
                has_compatible_tag,
                "{path} should be tagged OpenAI Compatible"
            );
        }
    }

    #[test]
    fn openapi_wires_preview_route_groups_for_scalar_sidebar() {
        let openapi = document();
        let tags = openapi["tags"].as_array().expect("tags should exist");
        let tag_names: HashSet<&str> = tags.iter().filter_map(|tag| tag["name"].as_str()).collect();

        for tag in [
            "Runtime",
            "OpenAI Compatible",
            "Admin",
            "Speech to Text",
            "Diarization",
            "Text to Speech",
            "Studio",
            "Chat",
            "Voice",
            "Media",
            "Preferences",
            "Realtime",
            "Reference",
        ] {
            assert!(tag_names.contains(tag), "{tag} tag should exist");
        }
        for legacy_tag in [
            "openai-compatible",
            "openai-preview",
            "openai-style-preview",
            "speech-text-workflows",
            "speech-generation-workflows",
            "studio-workflows",
            "chat-agent-workflows",
            "voice-workflows",
        ] {
            assert!(
                !tag_names.contains(legacy_tag),
                "{legacy_tag} tag should not remain"
            );
        }

        let paths = openapi["paths"].as_object().expect("paths should exist");
        let expected = [
            (
                "/v1/admin/models/{variant}/download/progress",
                "get",
                "Admin",
            ),
            (
                "/v1/speech-to-text/jobs/{record_id}/reruns",
                "post",
                "Speech to Text",
            ),
            ("/v1/diarizations/{record_id}/reruns", "post", "Diarization"),
            (
                "/v1/text-to-speech/{record_id}/audio",
                "get",
                "Text to Speech",
            ),
            (
                "/v1/studio/projects/{project_id}/segments",
                "post",
                "Studio",
            ),
            ("/v1/chat/threads/{thread_id}/messages", "get", "Chat"),
            ("/v1/voice/profile", "patch", "Voice"),
            ("/v1/voice/sessions", "post", "Voice"),
            ("/v1/voice/sessions/{session_id}/turns", "get", "Voice"),
            ("/v1/media/{path}", "get", "Media"),
            ("/v1/media", "post", "Media"),
            ("/v1/preferences/analytics", "put", "Preferences"),
            ("/v1/voice/realtime/ws", "get", "Realtime"),
            ("/openapi.json", "get", "Reference"),
            ("/docs", "get", "Reference"),
            ("/docs/scalar.js", "get", "Reference"),
        ];

        for (path, method, tag) in expected {
            let operation = paths
                .get(path)
                .and_then(|operations| operations.get(method))
                .unwrap_or_else(|| panic!("{method} {path} should be documented"));
            let has_tag = operation["tags"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|actual| actual.as_str() == Some(tag));
            assert!(has_tag, "{method} {path} should be tagged {tag}");
        }

        for removed_path in [
            "/v1/text-to-speech-generations",
            "/v1/voice-design-generations",
            "/v1/voice-clone-generations",
            "/v1/transcriptions/jobs",
            "/v1/transcription/realtime/ws",
            "/v1/audio/diarize",
            "/v1/audio/diarizations",
        ] {
            assert!(
                !paths.contains_key(removed_path),
                "{removed_path} should not be documented"
            );
        }

        for operation in paths.values().flat_map(|path| {
            path.as_object()
                .into_iter()
                .flat_map(|operations| operations.values())
        }) {
            for tag in operation["tags"].as_array().into_iter().flatten() {
                let tag = tag.as_str().expect("operation tags should be strings");
                assert!(!tag.contains('-'), "{tag} should not contain hyphens");
            }
        }
    }

    #[test]
    fn openapi_types_admin_model_discovery_contract() {
        let openapi = document();
        let schemas = openapi["components"]["schemas"]
            .as_object()
            .expect("schemas should exist");

        for schema in [
            "AdminModelsResponse",
            "AdminModelInfo",
            "AdminModelRouteCapabilities",
            "AdminSpeechModelCapabilities",
            "AdminModelActionResponse",
            "AdminModelDownloadProgressEvent",
        ] {
            assert!(schemas.contains_key(schema), "{schema} schema should exist");
        }

        let paths = openapi["paths"].as_object().expect("paths should exist");
        assert_eq!(
            paths["/v1/admin/models"]["get"]["responses"]["200"]["content"]["application/json"]
                ["schema"]["$ref"]
                .as_str(),
            Some("#/components/schemas/AdminModelsResponse")
        );
        assert_eq!(
            paths["/v1/admin/models/{variant}/download/progress"]["get"]["responses"]["200"]
                ["content"]["text/event-stream"]["schema"]["$ref"]
                .as_str(),
            Some("#/components/schemas/AdminModelDownloadProgressEvent")
        );
    }

    #[test]
    fn openapi_uses_action_first_sidebar_summaries() {
        let openapi = document();
        let paths = openapi["paths"].as_object().expect("paths should exist");

        let expected = [
            ("/v1/audio/speech", "post", "Create speech"),
            ("/v1/responses/{response_id}", "get", "Get response"),
            ("/v1/speech-to-text/jobs", "get", "List speech-to-text jobs"),
            (
                "/v1/speech-to-text/jobs",
                "post",
                "Create speech-to-text job",
            ),
            ("/v1/diarizations", "get", "List diarization records"),
            ("/v1/diarizations", "post", "Create diarization record"),
            (
                "/v1/diarizations/{record_id}",
                "put",
                "Replace diarization record",
            ),
            ("/v1/text-to-speech", "get", "List text-to-speech records"),
            (
                "/v1/text-to-speech/{record_id}",
                "delete",
                "Delete text-to-speech record",
            ),
            ("/v1/voices/{voice_id}", "get", "Get saved voice"),
            (
                "/v1/studio/projects/{project_id}/audio",
                "get",
                "Download Studio project audio",
            ),
            (
                "/v1/chat/threads/{thread_id}/messages",
                "post",
                "Create chat message",
            ),
            (
                "/v1/voice/realtime/ws",
                "get",
                "Open voice realtime WebSocket",
            ),
        ];

        for (path, method, summary) in expected {
            let actual = paths
                .get(path)
                .and_then(|operations| operations.get(method))
                .and_then(|operation| operation["summary"].as_str())
                .unwrap_or_else(|| panic!("{method} {path} should have a summary"));
            assert_eq!(actual, summary, "{method} {path} summary should match");
        }

        let allowed_prefixes: HashSet<&str> = [
            "Cancel",
            "Check",
            "Clear",
            "Complete",
            "Create",
            "Delete",
            "Download",
            "Get",
            "List",
            "Load",
            "Merge",
            "Open",
            "Regenerate",
            "Render",
            "Reorder",
            "Replace",
            "Restore",
            "Rerun",
            "Split",
            "Start",
            "Stream",
            "Unload",
            "Update",
        ]
        .into_iter()
        .collect();

        for (path, operations) in paths {
            for (method, operation) in operations
                .as_object()
                .expect("path operations should be an object")
            {
                let summary = operation["summary"]
                    .as_str()
                    .unwrap_or_else(|| panic!("{method} {path} should have a summary"));
                assert!(
                    !summary.starts_with('/'),
                    "{method} {path} should not use a raw path as the sidebar label"
                );
                assert!(
                    !summary.contains('{'),
                    "{method} {path} should not expose path parameters in the sidebar label"
                );
                let first_word = summary
                    .split_whitespace()
                    .next()
                    .expect("summary should not be empty");
                assert!(
                    allowed_prefixes.contains(first_word),
                    "{method} {path} summary should start with an approved action verb: {summary}"
                );
            }
        }
    }

    #[test]
    fn openapi_omits_legacy_sidebar_routes() {
        let openapi = document();
        let paths = openapi["paths"].as_object().expect("paths should exist");

        for legacy_path in [
            "/v1/audio/diarize",
            "/v1/audio/diarizations",
            "/v1/transcriptions",
            "/v1/transcriptions/{record_id}",
            "/v1/transcriptions/{record_id}/audio",
            "/v1/transcriptions/{record_id}/summary/regenerate",
        ] {
            assert!(
                !paths.contains_key(legacy_path),
                "{legacy_path} should not be documented in OpenAPI"
            );
        }
    }
}
