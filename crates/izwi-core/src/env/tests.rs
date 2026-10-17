use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const REFERENCE: &str = "docs/dev/env/core.md";

fn var(name: &str) -> &'static CoreEnvVar {
    CORE_ENV
        .iter()
        .find(|var| var.name == name)
        .unwrap_or_else(|| panic!("{name} is not registered"))
}

/// Every `IZWI_*` literal in the crate's production source is registered,
/// and every registered name still appears there.
#[test]
fn the_registry_matches_the_source() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    // The registry itself names every variable; scan everything else.
    let source = izwi_env::production_env_names_excluding(&src, &[&src.join("env")]);
    let registered: BTreeSet<String> = CORE_ENV.iter().map(|var| var.name.to_string()).collect();
    let missing: Vec<&String> = source.difference(&registered).collect();
    // A test-harness variable may be read only inside test modules.
    let stale: Vec<&str> = CORE_ENV
        .iter()
        .filter(|var| var.read != EnvRead::TestHarness && !source.contains(var.name))
        .map(|var| var.name)
        .collect();
    assert!(
        missing.is_empty(),
        "register these in src/env/registry.rs: {missing:?}"
    );
    assert!(
        stale.is_empty(),
        "these are registered but no longer in the source: {stale:?}"
    );
}

#[test]
fn the_registry_is_sorted_and_unique() {
    for pair in CORE_ENV.windows(2) {
        assert!(
            pair[0].name < pair[1].name,
            "{} must come after {}",
            pair[1].name,
            pair[0].name
        );
    }
}

#[test]
fn only_booleans_take_aliases_and_aliases_do_not_shadow_the_vocabulary() {
    for var in CORE_ENV {
        if var.aliases.is_empty() {
            continue;
        }
        assert!(
            matches!(var.kind, EnvKind::Flag | EnvKind::Mode),
            "{}",
            var.name
        );
        for (alias, _) in var.aliases {
            assert!(
                izwi_env::parse_bool(var.name, alias).is_err(),
                "{}: {alias} is already a boolean",
                var.name
            );
        }
    }
}

/// `docs/dev/env/core.md` is rendered from the registry. Run with
/// `UPDATE_ENV_REFERENCE=1` to rewrite it.
#[test]
fn reference_doc_is_current() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(REFERENCE);
    let expected = render_reference();
    if std::env::var_os("UPDATE_ENV_REFERENCE").is_some() {
        std::fs::write(&path, &expected).unwrap();
    }
    let actual = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        actual == expected,
        "{REFERENCE} is stale; rerun with UPDATE_ENV_REFERENCE=1"
    );
}

#[test]
fn flags_take_the_shared_vocabulary_in_any_case() {
    let flag = var("IZWI_FUSED_KERNELS");
    for raw in ["1", "true", "TRUE", "yes", "On", " on "] {
        assert_eq!(flag.parse_flag(raw).ok(), Some(true), "{raw}");
    }
    for raw in ["0", "false", "No", "OFF", " off "] {
        assert_eq!(flag.parse_flag(raw).ok(), Some(false), "{raw}");
    }
    for raw in ["2", "enabled", "fasle", "legacy"] {
        assert!(flag.parse_flag(raw).is_err(), "{raw}");
    }
}

#[test]
fn a_flag_alias_means_its_registered_value() {
    let switch = CORE_ENV
        .iter()
        .find(|var| !var.aliases.is_empty())
        .expect("at least one switch registers an alias");
    for (alias, meaning) in switch.aliases {
        assert_eq!(switch.parse_flag(alias).ok(), Some(*meaning));
        assert_eq!(
            switch.parse_flag(&alias.to_ascii_uppercase()).ok(),
            Some(*meaning)
        );
    }
}

fn validate(pairs: &[(&str, &str)]) -> crate::Result<()> {
    let values: BTreeMap<&str, &str> = pairs.iter().copied().collect();
    validate_with(|name| {
        values
            .get(name)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

#[test]
fn startup_validation_rejects_a_boolean_outside_the_vocabulary() {
    validate(&[]).unwrap();
    validate(&[("IZWI_FUSED_KERNELS", "off"), ("IZWI_CUDA_MODE", "auto")]).unwrap();
    // An empty value is unset.
    validate(&[("IZWI_FUSED_KERNELS", "  ")]).unwrap();
    let error = validate(&[("IZWI_FUSED_KERNELS", "fasle")])
        .unwrap_err()
        .to_string();
    assert!(error.contains("IZWI_FUSED_KERNELS"), "{error}");
    let error = validate(&[("IZWI_CUDA_MODE", "sometimes")])
        .unwrap_err()
        .to_string();
    assert!(error.contains("IZWI_CUDA_MODE"), "{error}");
    // Every bad variable is named, not just the first.
    let error = validate(&[("IZWI_FUSED_KERNELS", "x"), ("IZWI_CUDA_MODE", "y")])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("IZWI_FUSED_KERNELS") && error.contains("IZWI_CUDA_MODE"),
        "{error}"
    );
}

#[test]
fn startup_validation_accepts_a_registered_alias() {
    let switch = CORE_ENV
        .iter()
        .find(|var| !var.aliases.is_empty())
        .expect("at least one switch registers an alias");
    validate(&[(switch.name, switch.aliases[0].0)]).unwrap();
}

#[test]
fn read_once_rereads_in_unit_tests() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static READS: AtomicUsize = AtomicUsize::new(0);
    static VALUE: ReadOnce<usize> = ReadOnce::new(|| READS.fetch_add(1, Ordering::SeqCst));
    assert_eq!(VALUE.get(), 0);
    assert_eq!(VALUE.get(), 1);
}
