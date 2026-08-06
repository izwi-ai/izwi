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
fn legacy_inference_state_surface_is_confined_to_test_references() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_root = manifest_dir.join("src");
    let mut sources = Vec::new();
    collect_rs_files(&source_root, &mut |path| {
        if let Ok(source) = fs::read_to_string(path) {
            sources.push((path.to_path_buf(), source));
        }
    });

    sources.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    let mut violations = Vec::new();
    for (path, source) in sources {
        let relative = path.strip_prefix(&source_root).unwrap_or(&path);
        violations.extend(legacy_state_violations(relative, &source));
    }

    assert!(
        violations.is_empty(),
        "production inference state must use the sole physical ABI-v2 architecture:\n{}",
        violations.join("\n")
    );
}

#[test]
fn legacy_ratchet_allows_physical_provider_names_but_rejects_legacy_items() {
    let provider = r#"
        pub struct PhysicalQwen3CacheProvider;
        pub struct PhysicalKvPageProvider;
        pub trait PhysicalKvCacheContractProvider {}
        pub fn physical_paged_decode_attention_provider() {}
    "#;
    assert!(legacy_state_violations(Path::new("backends/kv/provider.rs"), provider).is_empty());

    let physical_operation = "pub fn paged_decode_attention() {}";
    assert!(legacy_state_violations(Path::new("kernels/metal.rs"), physical_operation).is_empty());

    let documentation = r#"
        // DenseKvCache and KvPage are legacy names, not live definitions here.
        const MIGRATION_NOTE: &str = "do not call append_to_pages";
    "#;
    assert!(legacy_state_violations(
        Path::new("models/architectures/new_model.rs"),
        documentation
    )
    .is_empty());

    let legacy = r#"
        pub struct DenseKvCache;
        pub enum KvPage {}
        pub fn append_to_pages() {}
        pub fn materialize_pages() {}
        pub fn paged_decode_attention() {}
    "#;
    assert_eq!(
        legacy_state_violations(Path::new("models/architectures/new_model.rs"), legacy),
        vec![
            "models/architectures/new_model.rs contains model-owned `DenseKvCache`".to_string(),
            "models/architectures/new_model.rs contains legacy page type `KvPage`".to_string(),
            "models/architectures/new_model.rs contains materializing helper `append_to_pages`"
                .to_string(),
            "models/architectures/new_model.rs contains materializing helper `materialize_pages`"
                .to_string(),
            "models/architectures/new_model.rs bypasses the physical arena with `paged_decode_attention`"
                .to_string(),
        ]
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
        paged.contains("#[cfg(test)]\nmod legacy"),
        "the materializing paged-attention implementation must remain test-only"
    );
    assert!(
        qwen_asr.contains(".forward_stateless_with_embeds(&embeds, 0, position_ids.as_ref())"),
        "cacheless forced-alignment work must not acquire a model-owned cache"
    );
}

fn legacy_state_violations(path: &Path, source: &str) -> Vec<String> {
    const REMOVED_ABI_IDENTIFIERS: &[&str] = &[
        "OpaqueModelOwned",
        "upgrade_kv_contract_v1",
        "CURRENT_KV_CONTRACT_ABI",
        "KvCacheContract",
        "KvDomainSpec",
    ];
    const MODEL_OWNED_CACHE_IDENTIFIERS: &[&str] = &[
        "Qwen3ManagedCache",
        "Qwen3Cache",
        "DenseKvCache",
        "Qwen35DenseKvCache",
    ];
    const MATERIALIZING_HELPERS: &[&str] = &["append_to_pages", "materialize_pages"];

    // `repeat_kv` alone is not evidence of retained model-owned state: several
    // cacheless full-sequence attention paths legitimately expand GQA heads.
    // This ratchet instead rejects the cache/page definitions and helpers that
    // can reintroduce independent retained storage.

    let path = path.to_string_lossy().replace('\\', "/");
    let identifiers = rust_identifiers(source);
    let qwen_reference = path == "models/architectures/qwen3/core.rs";
    let paged_reference = path == "models/shared/attention/paged.rs";
    let physical_provider = path.starts_with("backends/kv/")
        || path.starts_with("backends/state/")
        || path.starts_with("kernels/");
    let mut violations = Vec::new();

    for identifier in REMOVED_ABI_IDENTIFIERS {
        if identifiers.contains(*identifier) {
            violations.push(format!(
                "{path} contains removed ABI identifier `{identifier}`"
            ));
        }
    }
    for identifier in MODEL_OWNED_CACHE_IDENTIFIERS {
        if identifiers.contains(*identifier) && !(qwen_reference && *identifier == "Qwen3Cache") {
            violations.push(format!("{path} contains model-owned `{identifier}`"));
        }
    }
    if identifiers.contains("KvPage") && !qwen_reference && !paged_reference {
        violations.push(format!("{path} contains legacy page type `KvPage`"));
    }
    for helper in MATERIALIZING_HELPERS {
        if identifiers.contains(*helper) && !qwen_reference && !paged_reference {
            violations.push(format!("{path} contains materializing helper `{helper}`"));
        }
    }
    if identifiers.contains("paged_decode_attention")
        && !qwen_reference
        && !paged_reference
        && !physical_provider
    {
        violations.push(format!(
            "{path} bypasses the physical arena with `paged_decode_attention`"
        ));
    }
    violations
}

fn rust_identifiers(source: &str) -> BTreeSet<String> {
    let bytes = source.as_bytes();
    let mut identifiers = BTreeSet::new();
    let mut cursor = 0;

    while cursor < bytes.len() {
        if bytes[cursor] == b'/' && bytes.get(cursor + 1) == Some(&b'/') {
            cursor += 2;
            while bytes.get(cursor).is_some_and(|byte| *byte != b'\n') {
                cursor += 1;
            }
            continue;
        }
        if bytes[cursor] == b'/' && bytes.get(cursor + 1) == Some(&b'*') {
            cursor += 2;
            let mut depth = 1_usize;
            while cursor < bytes.len() && depth != 0 {
                match (bytes[cursor], bytes.get(cursor + 1)) {
                    (b'/', Some(b'*')) => {
                        depth += 1;
                        cursor += 2;
                    }
                    (b'*', Some(b'/')) => {
                        depth -= 1;
                        cursor += 2;
                    }
                    _ => cursor += 1,
                }
            }
            continue;
        }
        if bytes[cursor] == b'r' {
            let mut quote = cursor + 1;
            while bytes.get(quote) == Some(&b'#') {
                quote += 1;
            }
            if bytes.get(quote) == Some(&b'"') {
                let hashes = quote - cursor - 1;
                cursor = quote + 1;
                while cursor < bytes.len() {
                    if bytes[cursor] == b'"'
                        && (0..hashes).all(|index| bytes.get(cursor + 1 + index) == Some(&b'#'))
                    {
                        cursor += hashes + 1;
                        break;
                    }
                    cursor += 1;
                }
                continue;
            }
        }
        if bytes[cursor] == b'"' {
            cursor += 1;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'\\' => cursor = (cursor + 2).min(bytes.len()),
                    b'"' => {
                        cursor += 1;
                        break;
                    }
                    _ => cursor += 1,
                }
            }
            continue;
        }
        if bytes[cursor] == b'_' || bytes[cursor].is_ascii_alphabetic() {
            let start = cursor;
            cursor += 1;
            while bytes
                .get(cursor)
                .is_some_and(|byte| *byte == b'_' || byte.is_ascii_alphanumeric())
            {
                cursor += 1;
            }
            identifiers.insert(source[start..cursor].to_string());
            continue;
        }
        cursor += 1;
    }

    identifiers
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
