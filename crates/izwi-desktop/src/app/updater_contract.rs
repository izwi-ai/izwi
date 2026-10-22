use serde::Serialize;

/// Which releases a build follows. A stable build follows stable releases;
/// a beta build follows betas and stable releases alike, so beta testers
/// move on to the release they tested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateChannel {
    Stable,
    Beta,
}

impl UpdateChannel {
    /// The channel of a build at `version`.
    pub fn for_version(version: &semver::Version) -> Self {
        if version.pre.is_empty() {
            UpdateChannel::Stable
        } else {
            UpdateChannel::Beta
        }
    }

    /// Whether a build on this channel takes a release at `version`.
    pub fn accepts(self, version: &semver::Version) -> bool {
        match self {
            UpdateChannel::Stable => version.pre.is_empty(),
            UpdateChannel::Beta => true,
        }
    }
}

/// The updater manifest a release at `version` publishes: `latest.json`
/// for a stable release, `latest-beta.json` for a prerelease.
pub fn manifest_asset_name(version: &semver::Version) -> &'static str {
    if version.pre.is_empty() {
        "latest.json"
    } else {
        "latest-beta.json"
    }
}

/// The version a release tag names: `v0.1.0`, `v0.1.0-beta-18`.
pub fn parse_release_tag(tag_name: &str) -> Option<semver::Version> {
    semver::Version::parse(tag_name.strip_prefix('v')?).ok()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdaterContract {
    pub channel: UpdateChannel,
    pub release_owner: String,
    pub release_repo: String,
    pub releases_per_page: usize,
}

impl UpdaterContract {
    /// The contract of a build at `version`.
    pub fn for_version(version: &semver::Version) -> Self {
        Self {
            channel: UpdateChannel::for_version(version),
            release_owner: "izwi-ai".to_string(),
            release_repo: "izwi".to_string(),
            releases_per_page: 50,
        }
    }
}

impl Default for UpdaterContract {
    /// This build's contract.
    fn default() -> Self {
        let version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
            .expect("the package version is semver");
        Self::for_version(&version)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlatformInstallBehavior {
    pub app_exits_during_install: bool,
    pub supports_restart_later: bool,
}

/// Numeric beta sequence of a semver prerelease label. Accepts the shipped
/// `beta-N` form as well as the dotted `beta.N` form.
fn beta_prerelease_sequence(prerelease: &str) -> Option<u64> {
    let digits = prerelease
        .strip_prefix("beta-")
        .or_else(|| prerelease.strip_prefix("beta."))?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// Whether `update` should replace `current`.
///
/// Semver compares `beta-18` as one alphanumeric identifier, so `beta-2`
/// would sort above `beta-18`. When both versions carry a beta sequence on
/// the same release core, compare the sequence numerically instead.
pub fn is_newer_release(current: &semver::Version, update: &semver::Version) -> bool {
    let core = |version: &semver::Version| (version.major, version.minor, version.patch);
    if core(update) == core(current) {
        if let (Some(current_beta), Some(update_beta)) = (
            beta_prerelease_sequence(current.pre.as_str()),
            beta_prerelease_sequence(update.pre.as_str()),
        ) {
            return update_beta > current_beta;
        }
    }
    update > current
}

pub fn github_releases_api_url(contract: &UpdaterContract) -> String {
    format!(
        "https://api.github.com/repos/{}/{}/releases?per_page={}",
        contract.release_owner, contract.release_repo, contract.releases_per_page
    )
}

pub fn github_manifest_url(
    contract: &UpdaterContract,
    tag_name: &str,
    manifest_asset_name: &str,
) -> String {
    format!(
        "https://github.com/{}/{}/releases/download/{}/{}",
        contract.release_owner, contract.release_repo, tag_name, manifest_asset_name
    )
}

pub fn install_behavior_for_platform(target_os: &str) -> PlatformInstallBehavior {
    match target_os {
        "windows" => PlatformInstallBehavior {
            app_exits_during_install: true,
            supports_restart_later: false,
        },
        "macos" | "linux" => PlatformInstallBehavior {
            app_exits_during_install: false,
            supports_restart_later: true,
        },
        _ => PlatformInstallBehavior {
            app_exits_during_install: false,
            supports_restart_later: false,
        },
    }
}

/// The manifest entry a build installs from. A Linux build installed from
/// the `.deb` updates with the `.deb` (`linux-deb`), not the AppImage.
pub fn updater_target_for_platform(target_os: &str, deb_install: bool) -> &'static str {
    match target_os {
        "windows" => "windows",
        "macos" => "macos",
        "linux" if deb_install => "linux-deb",
        "linux" => "linux",
        _ => "unknown",
    }
}

/// The system a release's `target` build needs, as "macOS 13.3", when the
/// `running` system is older: installing that build would leave an app the
/// system refuses to open. The release manifest sets the floor per target
/// (`"minimum_system_version": {"macos": "13.3"}`); a manifest without one,
/// or a version that can't be read, sets no floor.
pub fn unmet_system_floor(
    manifest: &serde_json::Value,
    target: &str,
    running: Option<&str>,
) -> Option<String> {
    let required = manifest
        .get("minimum_system_version")?
        .get(target)?
        .as_str()?;
    let too_old = compare_system_versions(running?, required)?.is_lt();
    too_old.then(|| format!("{} {}", system_name(target), required.trim()))
}

/// Dotted numeric versions compared part by part, a missing part as 0:
/// "13" equals "13.0", and "12.7.6" is older than "13.3".
fn compare_system_versions(left: &str, right: &str) -> Option<std::cmp::Ordering> {
    let parts = |text: &str| -> Option<Vec<u64>> {
        text.trim()
            .split('.')
            .map(|part| part.parse().ok())
            .collect()
    };
    let (left, right) = (parts(left)?, parts(right)?);
    let part = |parts: &[u64], index: usize| parts.get(index).copied().unwrap_or(0);
    Some(
        (0..left.len().max(right.len()))
            .map(|index| part(&left, index).cmp(&part(&right, index)))
            .find(|ordering| ordering.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal),
    )
}

fn system_name(target: &str) -> &'static str {
    match target {
        "macos" => "macOS",
        "windows" => "Windows",
        "linux" | "linux-deb" => "Linux",
        _ => "this system",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        github_manifest_url, github_releases_api_url, install_behavior_for_platform,
        is_newer_release, manifest_asset_name, parse_release_tag, unmet_system_floor,
        updater_target_for_platform, UpdateChannel, UpdaterContract,
    };

    fn version(text: &str) -> semver::Version {
        semver::Version::parse(text).unwrap()
    }

    #[test]
    fn parses_release_tags() {
        assert_eq!(
            parse_release_tag("v0.1.0-beta-12"),
            Some(version("0.1.0-beta-12"))
        );
        assert_eq!(parse_release_tag("v0.1.0"), Some(version("0.1.0")));
        assert_eq!(parse_release_tag("0.1.0"), None);
        assert_eq!(parse_release_tag("v0.1"), None);
        assert_eq!(parse_release_tag("nightly"), None);
    }

    #[test]
    fn stable_builds_follow_stable_releases_and_betas_follow_both() {
        let stable = UpdateChannel::for_version(&version("0.1.0"));
        let beta = UpdateChannel::for_version(&version("0.1.0-beta-18"));
        assert_eq!(stable, UpdateChannel::Stable);
        assert_eq!(beta, UpdateChannel::Beta);
        assert!(stable.accepts(&version("0.1.1")));
        assert!(!stable.accepts(&version("0.2.0-beta-1")));
        assert!(beta.accepts(&version("0.1.0")));
        assert!(beta.accepts(&version("0.1.0-beta-19")));
        assert_eq!(manifest_asset_name(&version("0.1.0")), "latest.json");
        assert_eq!(
            manifest_asset_name(&version("0.1.0-beta-19")),
            "latest-beta.json"
        );
    }

    #[test]
    fn builds_github_endpoints_from_contract() {
        let contract = UpdaterContract::for_version(&version("0.1.0"));
        assert_eq!(
            github_releases_api_url(&contract),
            "https://api.github.com/repos/izwi-ai/izwi/releases?per_page=50"
        );
        assert_eq!(
            github_manifest_url(&contract, "v0.1.1", "latest.json"),
            "https://github.com/izwi-ai/izwi/releases/download/v0.1.1/latest.json"
        );
    }

    #[test]
    fn exposes_platform_install_contract() {
        let windows = install_behavior_for_platform("windows");
        assert!(windows.app_exits_during_install);
        assert!(!windows.supports_restart_later);

        let macos = install_behavior_for_platform("macos");
        assert!(!macos.app_exits_during_install);
        assert!(macos.supports_restart_later);

        let linux = install_behavior_for_platform("linux");
        assert!(!linux.app_exits_during_install);
        assert!(linux.supports_restart_later);
    }

    #[test]
    fn maps_platform_to_custom_updater_target() {
        assert_eq!(updater_target_for_platform("windows", false), "windows");
        assert_eq!(updater_target_for_platform("macos", false), "macos");
        assert_eq!(updater_target_for_platform("linux", false), "linux");
        assert_eq!(updater_target_for_platform("linux", true), "linux-deb");
        assert_eq!(updater_target_for_platform("dragonfly", false), "unknown");
    }

    #[test]
    fn reports_a_system_floor_the_running_system_misses() {
        let manifest = serde_json::json!({
            "version": "v0.2.0",
            "minimum_system_version": { "macos": "13.3" },
        });
        let floor = |running| unmet_system_floor(&manifest, "macos", running);

        assert_eq!(floor(Some("12.7.6")), Some("macOS 13.3".into()));
        assert_eq!(floor(Some("13.2.1")), Some("macOS 13.3".into()));
        assert_eq!(floor(Some("13.3")), None);
        assert_eq!(floor(Some("13.3.0")), None);
        assert_eq!(floor(Some("15.1")), None);
        // An unreadable system version keeps the update on offer.
        assert_eq!(floor(None), None);
        assert_eq!(floor(Some("")), None);
        // Other targets, and manifests from before floors, set none.
        assert_eq!(unmet_system_floor(&manifest, "windows", Some("10.0")), None);
        assert_eq!(
            unmet_system_floor(
                &serde_json::json!({"version": "v0.1.0"}),
                "macos",
                Some("12.0")
            ),
            None
        );
        assert_eq!(
            unmet_system_floor(
                &serde_json::json!({"minimum_system_version": {"macos": "thirteen"}}),
                "macos",
                Some("12.0")
            ),
            None
        );
    }

    fn newer(current: &str, update: &str) -> bool {
        is_newer_release(
            &semver::Version::parse(current).unwrap(),
            &semver::Version::parse(update).unwrap(),
        )
    }

    #[test]
    fn orders_beta_sequences_numerically() {
        // Plain semver ranks "beta-2" above "beta-18"; the updater must not.
        assert!(
            semver::Version::parse("0.1.0-beta-2").unwrap()
                > semver::Version::parse("0.1.0-beta-18").unwrap()
        );
        assert!(newer("0.1.0-beta-2", "0.1.0-beta-18"));
        assert!(newer("0.1.0-beta-9", "0.1.0-beta-10"));
        assert!(!newer("0.1.0-beta-18", "0.1.0-beta-2"));
        assert!(!newer("0.1.0-beta-18", "0.1.0-beta-18"));
    }

    #[test]
    fn mixes_dashed_and_dotted_beta_labels() {
        assert!(newer("0.1.0-beta-18", "0.1.0-beta.19"));
        assert!(!newer("0.1.0-beta.19", "0.1.0-beta-18"));
    }

    #[test]
    fn falls_back_to_semver_outside_beta_sequences() {
        assert!(newer("0.1.0-beta-18", "0.1.0"));
        assert!(newer("0.1.0-beta-18", "0.2.0-beta-1"));
        assert!(!newer("0.2.0-beta-1", "0.1.0-beta-99"));
        assert!(newer("0.1.0", "0.1.1"));
    }
}
