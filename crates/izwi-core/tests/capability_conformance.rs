use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use izwi_core::{
    capability_conformance_cases, required_conformance_capabilities, AudioChunk,
    ConformanceCapability, EngineConfig, EngineCoreRequest, GenerationConfig, GenerationRequest,
    ModelVariant, RuntimeService, RuntimeTelemetrySnapshot, VoiceSessionPhase,
};

#[test]
fn builtin_conformance_cases_cover_required_capabilities() {
    let covered = capability_conformance_cases()
        .iter()
        .map(|case| case.capability)
        .collect::<BTreeSet<_>>();

    for capability in required_conformance_capabilities() {
        assert!(
            covered.contains(capability),
            "missing conformance case for {}",
            capability.as_str()
        );
    }
}

#[test]
fn builtin_conformance_case_ids_are_unique_and_descriptive() {
    let mut ids = BTreeSet::new();

    for case in capability_conformance_cases() {
        assert!(ids.insert(case.id), "duplicate conformance case {}", case.id);
        assert!(
            case.id.contains(case.capability.as_str())
                || matches!(
                    case.capability,
                    ConformanceCapability::Vad | ConformanceCapability::Endpointing
                ),
            "case id `{}` should name its capability `{}`",
            case.id,
            case.capability.as_str()
        );
        assert!(!case.fixture.trim().is_empty());
    }
}

#[test]
fn public_runtime_reexports_remain_compile_visible() {
    let _engine_config = EngineConfig::default();
    let _generation_config = GenerationConfig::default();
    let _audio_chunk = AudioChunk::new("request".to_string(), 0, Vec::new());
    let _request = GenerationRequest::new("hello");
    let _engine_request =
        EngineCoreRequest::tts("hello").with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase);
    let _phase = VoiceSessionPhase::Idle;

    fn accepts_runtime_service(_: Option<&RuntimeService>) {}
    fn accepts_runtime_snapshot(_: Option<&RuntimeTelemetrySnapshot>) {}

    accepts_runtime_service(None);
    accepts_runtime_snapshot(None);
}

#[test]
fn product_crates_do_not_import_internal_model_architectures() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let product_roots = [
        workspace_root.join("crates/izwi-server/src"),
        workspace_root.join("crates/izwi-cli/src"),
    ];

    let mut violations = Vec::new();
    for root in product_roots {
        collect_rs_files(&root, &mut |path| {
            let Ok(source) = fs::read_to_string(path) else {
                return;
            };
            for forbidden in [
                "izwi_core::models::architectures",
                "izwi_core::models::registry",
                "izwi_core::models::shared",
            ] {
                if source.contains(forbidden) {
                    violations.push(format!("{} imports {forbidden}", path.display()));
                }
            }
        });
    }

    assert!(
        violations.is_empty(),
        "product crates should use public runtime/runtime_models/catalog APIs:\n{}",
        violations.join("\n")
    );
}

fn collect_rs_files(root: &Path, visit: &mut impl FnMut(&Path)) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, visit);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            visit(&path);
        }
    }
}
