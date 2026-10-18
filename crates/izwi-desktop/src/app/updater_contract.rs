use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateChannel {
    Beta010,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdaterContract {
    pub channel: UpdateChannel,
    pub release_owner: String,
    pub release_repo: String,
    pub manifest_asset_name: String,
    pub releases_per_page: usize,
}

impl Default for UpdaterContract {
    fn default() -> Self {
        Self {
            channel: UpdateChannel::Beta010,
            release_owner: "izwi-ai".to_string(),
            release_repo: "izwi".to_string(),
            manifest_asset_name: "latest-beta.json".to_string(),
            releases_per_page: 50,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlatformInstallBehavior {
    pub app_exits_during_install: bool,
    pub supports_restart_later: bool,
}

impl UpdateChannel {
    pub fn tag_prefix(self) -> &'static str {
        match self {
            UpdateChannel::Beta010 => "v0.1.0-beta-",
        }
    }
}

pub fn parse_beta_sequence(channel: UpdateChannel, tag_name: &str) -> Option<u64> {
    let suffix = tag_name.strip_prefix(channel.tag_prefix())?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    suffix.parse::<u64>().ok()
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

pub fn github_manifest_url(contract: &UpdaterContract, tag_name: &str) -> String {
    format!(
        "https://github.com/{}/{}/releases/download/{}/{}",
        contract.release_owner, contract.release_repo, tag_name, contract.manifest_asset_name
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

pub fn updater_target_for_platform(target_os: &str) -> &'static str {
    match target_os {
        "windows" => "windows",
        "macos" => "macos",
        "linux" => "linux",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        github_manifest_url, github_releases_api_url, install_behavior_for_platform,
        is_newer_release, parse_beta_sequence, updater_target_for_platform, UpdateChannel,
        UpdaterContract,
    };

    #[test]
    fn parses_beta_sequence_with_valid_tag() {
        let channel = UpdateChannel::Beta010;
        assert_eq!(parse_beta_sequence(channel, "v0.1.0-beta-12"), Some(12));
    }

    #[test]
    fn rejects_invalid_beta_tags() {
        let channel = UpdateChannel::Beta010;
        assert_eq!(parse_beta_sequence(channel, "v0.1.0-beta"), None);
        assert_eq!(parse_beta_sequence(channel, "v0.1.1-beta-1"), None);
        assert_eq!(parse_beta_sequence(channel, "v0.1.0-beta-1a"), None);
        assert_eq!(parse_beta_sequence(channel, "0.1.0-beta-1"), None);
    }

    #[test]
    fn accepts_supported_beta_tags() {
        let channel = UpdateChannel::Beta010;
        assert_eq!(parse_beta_sequence(channel, "v0.1.0-beta-1"), Some(1));
        assert_eq!(parse_beta_sequence(channel, "v0.1.0-alpha-1"), None);
    }

    #[test]
    fn builds_github_endpoints_from_contract() {
        let contract = UpdaterContract::default();
        assert_eq!(
            github_releases_api_url(&contract),
            "https://api.github.com/repos/izwi-ai/izwi/releases?per_page=50"
        );
        assert_eq!(
            github_manifest_url(&contract, "v0.1.0-beta-12"),
            "https://github.com/izwi-ai/izwi/releases/download/v0.1.0-beta-12/latest-beta.json"
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
        assert_eq!(updater_target_for_platform("windows"), "windows");
        assert_eq!(updater_target_for_platform("macos"), "macos");
        assert_eq!(updater_target_for_platform("linux"), "linux");
        assert_eq!(updater_target_for_platform("dragonfly"), "unknown");
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
