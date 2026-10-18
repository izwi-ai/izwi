//! DP-F11: wire compatibility between protocol minors, from JSON literals of
//! what each side sends. Minor 3 is the last minor before negotiation; its
//! peers have strict enums and send no version header.
//!
//! | sender            | receiver          | covered by                                   |
//! |-------------------|-------------------|----------------------------------------------|
//! | minor-3 worker    | minor-4 gateway   | `minor3_worker_messages_decode_...`          |
//! | minor-4 worker    | minor-3 gateway   | `descriptor_for_a_minor3_gateway_...`        |
//! | minor-4 gateway   | minor-3 worker    | `requests_without_minor4_knobs_encode_...`   |
//! | minor-3 gateway   | minor-4 worker    | `minor3_gateway_requests_decode_...`         |
//! | minor-5 worker    | minor-4 gateway   | `newer_worker_variants_decode_as_unknown`    |

use izwi_serving_protocol::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn minor3_descriptor() -> Value {
    json!({
        "schema_version": {"major": 1, "minor": 3},
        "supported_protocol_versions": [{"major": 1, "minor": 3}],
        "worker_id": "worker-1",
        "node_id": "node-1",
        "incarnation_id": "inc-1",
        "build_version": "0.1.0",
        "assignment": {
            "backend": "cpu",
            "thread_budget": 4,
            "host_memory_limit_bytes": 1_073_741_824u64
        },
        "features": ["streaming", "cancellation", "attempt_query"]
    })
}

fn minor4_chat_descriptor() -> WorkerDescriptor {
    let mut descriptor: WorkerDescriptor = serde_json::from_value(minor3_descriptor()).unwrap();
    descriptor.schema_version = PROTOCOL_V1;
    descriptor.supported_protocol_versions = vec![PROTOCOL_V1];
    descriptor.features.extend([
        WorkerFeature::ChatSeed,
        WorkerFeature::ChatStopSequences,
        WorkerFeature::ChatTopK,
        WorkerFeature::ChatPenalties,
        WorkerFeature::ChatLogprobs,
        WorkerFeature::ChatJsonObject,
        WorkerFeature::ChatContextPolicy,
        WorkerFeature::ChatTools,
        WorkerFeature::ChatSkipSpecialTokens,
    ]);
    descriptor
}

/// The request a minor-3 gateway sends: resolved temperature and top-p,
/// no seed, no minor-4 fields. Values are exact in f32.
fn minor3_request() -> Value {
    json!({
        "schema_version": {"major": 1, "minor": 3},
        "request_id": "req-1",
        "attempt_id": "attempt-1",
        "expected_worker_incarnation": "inc-1",
        "deployment_id": "chat-v1",
        "expected_model_generation": 3,
        "caller": {
            "tenant_id": "tenant-1",
            "caller_id": "caller-1",
            "policy_revision": "policy-1",
            "permitted_actions": ["invoke", "cancel_own_invocation"]
        },
        "task": "chat",
        "service_class": "interactive",
        "remaining_time_ms": 10_000,
        "max_queue_wait_ms": 250,
        "output_limits": {"max_tokens": 32, "max_bytes": 4096},
        "requested_output_format": "text",
        "session_id": null,
        "request_digest": "sha256:abc",
        "input": {
            "kind": "chat",
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "parameters": {"temperature": 0.5, "top_p": 0.75, "seed": null}
        }
    })
}

fn chat_parameters(request: &InvocationRequest) -> &ChatParameters {
    let InvocationInput::Chat { parameters, .. } = &request.input;
    parameters
}

#[test]
fn minor3_worker_messages_decode_and_negotiate_minor3() {
    let descriptor: WorkerDescriptor = serde_json::from_value(minor3_descriptor()).unwrap();
    assert_eq!(
        descriptor.negotiated_version(),
        Some(SchemaVersion::new(1, 3))
    );
    assert!(descriptor
        .features
        .iter()
        .all(|feature| feature.introduced_in_minor() <= 3));

    let rejection: InvocationRejection = serde_json::from_value(json!({
        "schema_version": {"major": 1, "minor": 3},
        "accepted": false,
        "request_id": "req-1",
        "attempt_id": "attempt-1",
        "code": "invalid_request",
        "message": "too long",
        "retry_after_ms": null
    }))
    .unwrap();
    assert_eq!(rejection.code, RejectionCode::InvalidRequest);
}

/// A minor-3 gateway's strict decoder: the minor-3 feature set exactly.
#[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Minor3WorkerFeature {
    Streaming,
    Cancellation,
    AttemptQuery,
    RealtimeSocket,
}

#[derive(Debug, Deserialize)]
struct Minor3DescriptorFeatures {
    features: BTreeSet<Minor3WorkerFeature>,
}

#[test]
fn descriptor_for_a_minor3_gateway_omits_what_it_cannot_decode() {
    let descriptor = minor4_chat_descriptor();
    let full = serde_json::to_value(&descriptor).unwrap();
    assert!(
        serde_json::from_value::<Minor3DescriptorFeatures>(full).is_err(),
        "a minor-3 gateway rejects minor-4 features"
    );

    let served = descriptor.for_peer(UNDECLARED_PEER_VERSION);
    let decoded: Minor3DescriptorFeatures =
        serde_json::from_value(serde_json::to_value(&served).unwrap()).unwrap();
    assert_eq!(
        decoded.features,
        BTreeSet::from([
            Minor3WorkerFeature::Streaming,
            Minor3WorkerFeature::Cancellation,
            Minor3WorkerFeature::AttemptQuery,
        ])
    );
    // A minor-4 gateway sees everything.
    assert_eq!(descriptor.for_peer(PROTOCOL_V1), descriptor);
}

#[test]
fn requests_without_minor4_knobs_encode_exactly_as_minor3() {
    let mut request: InvocationRequest = serde_json::from_value(minor3_request()).unwrap();
    request.schema_version = SchemaVersion::new(1, 3);
    assert_eq!(serde_json::to_value(&request).unwrap(), minor3_request());
    assert!(chat_parameters(&request).required_features().is_empty());

    // Every minor-4 knob is a new key, and each needs its feature.
    let InvocationInput::Chat { parameters, .. } = &mut request.input;
    *parameters = ChatParameters {
        temperature: Some(0.5),
        top_p: Some(0.75),
        seed: Some(7),
        stop: vec!["END".into()],
        top_k: Some(50),
        repetition_penalty: Some(1.25),
        presence_penalty: Some(0.5),
        logprobs: Some(2),
        response_format: Some(ChatResponseFormat::JsonObject),
        context: Some(ChatContextPolicy {
            max_tokens: Some(16),
            truncation: ChatTruncation::Disabled,
        }),
        tools: vec![json!({"type": "function", "function": {"name": "now"}})],
        skip_special_tokens: Some(false),
    };
    request.validate().unwrap();
    let encoded = serde_json::to_value(&request).unwrap();
    assert_eq!(
        encoded["input"]["parameters"],
        json!({
            "temperature": 0.5,
            "top_p": 0.75,
            "seed": 7,
            "stop": ["END"],
            "top_k": 50,
            "repetition_penalty": 1.25,
            "presence_penalty": 0.5,
            "logprobs": 2,
            "response_format": "json_object",
            "context": {"max_tokens": 16, "truncation": "disabled"},
            "tools": [{"type": "function", "function": {"name": "now"}}],
            "skip_special_tokens": false
        })
    );
    assert_eq!(
        chat_parameters(&request).required_features(),
        minor4_chat_descriptor()
            .features
            .into_iter()
            .filter(|feature| feature.introduced_in_minor() == 4)
            .collect()
    );
    let decoded: InvocationRequest = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, request);
}

#[test]
fn minor3_gateway_requests_decode_with_every_minor4_knob_absent() {
    let request: InvocationRequest = serde_json::from_value(minor3_request()).unwrap();
    request.validate().unwrap();
    assert_eq!(request.schema_version, SchemaVersion::new(1, 3));
    assert_eq!(
        *chat_parameters(&request),
        ChatParameters {
            temperature: Some(0.5),
            top_p: Some(0.75),
            ..ChatParameters::default()
        }
    );
}

#[test]
fn newer_worker_variants_decode_as_unknown() {
    let mut descriptor = serde_json::to_value(minor4_chat_descriptor()).unwrap();
    descriptor["supported_protocol_versions"] = json!([{"major": 1, "minor": 5}]);
    descriptor["features"]
        .as_array_mut()
        .unwrap()
        .push(json!("speculative_decoding"));
    let decoded: WorkerDescriptor = serde_json::from_value(descriptor).unwrap();
    assert!(decoded.features.contains(&WorkerFeature::Unknown));
    assert_eq!(decoded.negotiated_version(), Some(PROTOCOL_V1));

    let event: InvocationEvent = serde_json::from_value(json!({
        "schema_version": {"major": 1, "minor": 5},
        "request_id": "req-1",
        "attempt_id": "attempt-1",
        "sequence": 2,
        "type": "tool_call_delta",
        "arguments": "{}"
    }))
    .unwrap();
    assert_eq!(event.event, InvocationEventKind::Unknown);
    assert!(!event.is_terminal());

    let completed: InvocationEvent = serde_json::from_value(json!({
        "schema_version": {"major": 1, "minor": 5},
        "request_id": "req-1",
        "attempt_id": "attempt-1",
        "sequence": 3,
        "type": "completed",
        "finish_reason": "tool_calls",
        "usage": null
    }))
    .unwrap();
    assert!(matches!(
        completed.event,
        InvocationEventKind::Completed {
            finish_reason: FinishReason::Unknown,
            ..
        }
    ));

    let status_enums = json!({
        "process": "hibernating",
        "readiness": "compiling",
        "cancellation": "preemptive",
        "format": "video",
        "error": "quota_exceeded"
    });
    assert_eq!(
        serde_json::from_value::<WorkerProcessState>(status_enums["process"].clone()).unwrap(),
        WorkerProcessState::Unknown
    );
    assert_eq!(
        serde_json::from_value::<ModelReadiness>(status_enums["readiness"].clone()).unwrap(),
        ModelReadiness::Unknown
    );
    assert_eq!(
        serde_json::from_value::<CancellationBehavior>(status_enums["cancellation"].clone())
            .unwrap(),
        CancellationBehavior::Unknown
    );
    assert_eq!(
        serde_json::from_value::<OutputFormat>(status_enums["format"].clone()).unwrap(),
        OutputFormat::Unknown
    );
    assert_eq!(
        serde_json::from_value::<InputFormat>(status_enums["format"].clone()).unwrap(),
        InputFormat::Unknown
    );
    assert_eq!(
        serde_json::from_value::<InvocationErrorCode>(status_enums["error"].clone()).unwrap(),
        InvocationErrorCode::Unknown
    );

    let frame: RealtimeServerFrame =
        serde_json::from_value(json!({"type": "transcript_revision", "text": "x"})).unwrap();
    assert_eq!(frame, RealtimeServerFrame::Unknown);
    let codec: RealtimeAudioCodec = serde_json::from_value(json!("opus")).unwrap();
    assert_eq!(codec, RealtimeAudioCodec::Unknown);
    assert!(RealtimeAudioSpec {
        codec,
        sample_rate: 16_000,
        channels: 1
    }
    .validate()
    .is_err());
}

#[test]
fn negotiation_picks_the_highest_common_minor() {
    let v = SchemaVersion::new;
    assert_eq!(PROTOCOL_V1, v(1, 4));
    assert_eq!(PROTOCOL_V1.negotiate(&[v(1, 3)]), Some(v(1, 3)));
    assert_eq!(PROTOCOL_V1.negotiate(&[v(1, 2), v(1, 3)]), Some(v(1, 3)));
    assert_eq!(PROTOCOL_V1.negotiate(&[v(1, 9)]), Some(v(1, 4)));
    assert_eq!(PROTOCOL_V1.negotiate(&[v(2, 0)]), None);
    assert_eq!(PROTOCOL_V1.negotiate(&[]), None);
    assert_eq!(
        SchemaVersion::parse(&PROTOCOL_V1.to_string()),
        Some(PROTOCOL_V1)
    );
    assert_eq!(SchemaVersion::parse(" 1.3 "), Some(v(1, 3)));
    assert_eq!(SchemaVersion::parse("1"), None);
    assert_eq!(SchemaVersion::parse("1.x"), None);
}

#[test]
fn minor4_parameters_are_range_checked() {
    let mut request: InvocationRequest = serde_json::from_value(minor3_request()).unwrap();
    let InvocationInput::Chat { parameters, .. } = &mut request.input;
    parameters.repetition_penalty = Some(0.5);
    assert_eq!(
        request.validate(),
        Err(ContractValidationError::InvalidSamplingParameter)
    );
    let InvocationInput::Chat { parameters, .. } = &mut request.input;
    parameters.repetition_penalty = None;
    parameters.presence_penalty = Some(f32::NAN);
    assert_eq!(
        request.validate(),
        Err(ContractValidationError::InvalidSamplingParameter)
    );
    let InvocationInput::Chat { parameters, .. } = &mut request.input;
    parameters.presence_penalty = None;
    parameters.logprobs = Some(MAX_TOP_LOGPROBS + 1);
    assert_eq!(
        request.validate(),
        Err(ContractValidationError::InvalidLogprobs)
    );
    let InvocationInput::Chat { parameters, .. } = &mut request.input;
    parameters.logprobs = Some(MAX_TOP_LOGPROBS);
    parameters.context = Some(ChatContextPolicy {
        max_tokens: Some(0),
        truncation: ChatTruncation::Auto,
    });
    assert_eq!(
        request.validate(),
        Err(ContractValidationError::InvalidContextPolicy)
    );
    let InvocationInput::Chat { parameters, .. } = &mut request.input;
    parameters.context = None;
    parameters.tools = vec![json!("not a tool")];
    assert_eq!(
        request.validate(),
        Err(ContractValidationError::InvalidTools)
    );
    let InvocationInput::Chat { parameters, .. } = &mut request.input;
    parameters.tools = vec![json!({}); MAX_CHAT_TOOLS + 1];
    assert_eq!(
        request.validate(),
        Err(ContractValidationError::InvalidTools)
    );
}

#[test]
fn rejection_codes_downgrade_for_a_gateway_that_cannot_decode_them() {
    let minor3 = SchemaVersion::new(1, 3);
    assert_eq!(
        RejectionCode::ContextLengthExceeded.for_peer(minor3),
        RejectionCode::InvalidRequest
    );
    assert_eq!(
        RejectionCode::ContextLengthExceeded.for_peer(PROTOCOL_V1),
        RejectionCode::ContextLengthExceeded
    );
    assert_eq!(
        RejectionCode::CapacityExhausted.for_peer(minor3),
        RejectionCode::CapacityExhausted
    );
}

#[test]
fn unsupported_parameters_are_named_by_their_openai_name() {
    let parameters = ChatParameters {
        presence_penalty: Some(0.5),
        logprobs: Some(0),
        ..ChatParameters::default()
    };
    assert_eq!(
        parameters.first_unsupported(&BTreeSet::new()),
        Some("presence_penalty")
    );
    assert_eq!(
        parameters.first_unsupported(&BTreeSet::from([WorkerFeature::ChatPenalties])),
        Some("logprobs")
    );
    assert_eq!(
        parameters.first_unsupported(&BTreeSet::from([
            WorkerFeature::ChatPenalties,
            WorkerFeature::ChatLogprobs
        ])),
        None
    );
}

#[test]
fn only_keeping_special_tokens_needs_the_feature() {
    let keep = ChatParameters {
        skip_special_tokens: Some(false),
        ..ChatParameters::default()
    };
    assert_eq!(
        keep.first_unsupported(&BTreeSet::new()),
        Some("skip_special_tokens")
    );
    assert_eq!(
        keep.required_features(),
        BTreeSet::from([WorkerFeature::ChatSkipSpecialTokens])
    );
    // Skipping is every worker's default, so asking for it needs nothing.
    let skip = ChatParameters {
        skip_special_tokens: Some(true),
        ..ChatParameters::default()
    };
    assert!(skip.required_features().is_empty());
}
