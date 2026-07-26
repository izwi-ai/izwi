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
        assert!(
            ids.insert(case.id),
            "duplicate conformance case {}",
            case.id
        );
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

#[test]
fn legacy_inference_state_surface_does_not_expand_during_migration() {
    // This is a migration ratchet, not the final acceptance test. Every limit
    // must trend down to zero before the v1/opaque implementation is deleted.
    // Keeping the inventory in a compiled integration test prevents a new
    // model from quietly adding another model-owned cache while v2 is landing.
    const LEGACY_SYMBOL_LIMITS: &[(&str, usize)] = &[
        ("OpaqueModelOwned", 0),
        ("upgrade_kv_contract_v1", 0),
        ("CURRENT_KV_CONTRACT_ABI", 0),
        ("KvCacheContract", 0),
        ("KvDomainSpec::", 0),
        ("Qwen3ManagedCache", 33),
        ("Qwen3Cache", 63),
        ("DenseKvCache", 45),
        ("Qwen35DenseKvCache", 27),
        ("KvPage", 76),
        ("append_to_pages", 32),
        ("materialize_pages", 20),
        ("paged_decode_attention", 28),
        ("repeat_kv(", 56),
    ];

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_root = manifest_dir.join("src");
    let mut sources = Vec::new();
    collect_rs_files(&source_root, &mut |path| {
        if let Ok(source) = fs::read_to_string(path) {
            sources.push((path.to_path_buf(), source));
        }
    });

    let mut violations = Vec::new();
    for (symbol, limit) in LEGACY_SYMBOL_LIMITS {
        let count = sources
            .iter()
            .map(|(_, source)| source.matches(symbol).count())
            .sum::<usize>();
        if count > *limit {
            let files = sources
                .iter()
                .filter_map(|(path, source)| source.contains(symbol).then(|| path.display()))
                .map(|path| path.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            violations.push(format!(
                "legacy symbol `{symbol}` grew from at most {limit} to {count}: {files}"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "the v1/model-owned inference-state surface must only shrink:\n{}",
        violations.join("\n")
    );
}

#[test]
fn legacy_model_owned_state_symbols_are_confined_to_test_references() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_root = manifest_dir.join("src");
    let qwen_reference = source_root.join("models/architectures/qwen3/core.rs");
    let paged_reference = source_root.join("models/shared/attention/paged.rs");
    let mut violations = Vec::new();

    collect_rs_files(&source_root, &mut |path| {
        let Ok(source) = fs::read_to_string(path) else {
            return;
        };
        for symbol in [
            "OpaqueModelOwned",
            "upgrade_kv_contract_v1",
            "CURRENT_KV_CONTRACT_ABI",
            "KvCacheContract",
            "DenseKvCache",
            "Qwen35DenseKvCache",
        ] {
            if source.contains(symbol) {
                violations.push(format!("{} contains `{symbol}`", path.display()));
            }
        }
        if path != qwen_reference && source.contains("Qwen3Cache") {
            violations.push(format!(
                "{} contains model-owned `Qwen3Cache` outside its test reference",
                path.display()
            ));
        }
        if path != qwen_reference
            && path != paged_reference
            && ["append_to_pages", "materialize_pages"]
                .iter()
                .any(|symbol| source.contains(symbol))
        {
            violations.push(format!(
                "{} contains a materializing legacy page helper",
                path.display()
            ));
        }
    });

    assert!(
        violations.is_empty(),
        "production inference state must use the sole physical ABI-v2 architecture:\n{}",
        violations.join("\n")
    );
}

#[test]
fn qwen_model_owned_cache_and_materializing_pages_are_test_only() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let qwen_core = fs::read_to_string(manifest_dir.join("src/models/architectures/qwen3/core.rs"))
        .expect("read Qwen3 core");
    let paged = fs::read_to_string(manifest_dir.join("src/models/shared/attention/paged.rs"))
        .expect("read legacy paged helpers");
    let qwen_asr =
        fs::read_to_string(manifest_dir.join("src/models/architectures/qwen3/asr/mod.rs"))
            .expect("read Qwen3 ASR");

    assert!(
        qwen_core.contains("#[cfg(test)]\npub struct Qwen3Cache"),
        "the dependency-owned Qwen cache must never return to production"
    );
    assert!(
        paged.contains(
            "#[cfg(test)]\npub use legacy::{append_to_pages, materialize_pages, \
             paged_decode_attention, KvPage};"
        ),
        "materializing paged-attention helpers must remain test-only"
    );
    assert!(
        qwen_asr.contains(".forward_stateless_with_embeds(&embeds, 0, position_ids.as_ref())"),
        "cacheless forced-alignment work must not acquire a model-owned cache"
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
