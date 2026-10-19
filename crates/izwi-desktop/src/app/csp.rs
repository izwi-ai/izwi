use std::collections::HashMap;

use tauri::utils::config::{Config, Csp, CspDirectiveSources};
use url::Url;

/// Hosts the static policy in `tauri.conf.json` already allows on any port.
/// CSP host-sources cannot express IPv6 literals, so a `[::1]` server URL
/// cannot be allowed; use `localhost` instead.
const POLICY_LOOPBACK_HOSTS: &[&str] = &["localhost", "127.0.0.1"];

/// Directives the webview needs to reach the Izwi server: API calls and
/// event streams, realtime sockets, and audio/image responses.
const SERVER_DIRECTIVES: &[&str] = &["connect-src", "img-src", "media-src"];

/// Extend the configured Content Security Policy so the webview can reach a
/// server outside the loopback hosts the static policy covers (for example a
/// LAN or remote `--server-url`). No-op for loopback servers or when no
/// policy is configured.
pub fn allow_server_origin(config: &mut Config, server_url: &Url) {
    let Some(host) = server_url.host_str() else {
        return;
    };
    if POLICY_LOOPBACK_HOSTS.contains(&host) {
        return;
    }
    let Some(csp) = config.app.security.csp.take() else {
        return;
    };

    let origin = server_url.origin().ascii_serialization();
    let socket_origin = match server_url.scheme() {
        "https" => origin.replacen("https://", "wss://", 1),
        _ => origin.replacen("http://", "ws://", 1),
    };

    let mut directives: HashMap<String, CspDirectiveSources> = csp.into();
    for directive in SERVER_DIRECTIVES {
        let sources = directives.entry((*directive).to_string()).or_default();
        if !sources.contains(&origin) {
            sources.push(&origin);
        }
        if *directive == "connect-src" && !sources.contains(&socket_origin) {
            sources.push(&socket_origin);
        }
    }
    config.app.security.csp = Some(Csp::DirectiveMap(directives));
}

#[cfg(test)]
mod tests {
    use super::allow_server_origin;
    use tauri::utils::config::{Config, Csp};
    use url::Url;

    fn config_with_policy() -> Config {
        let config: Config = serde_json::from_value(serde_json::json!({
            "identifier": "com.example.test",
            "app": { "security": { "csp": {
                "default-src": "'self'",
                "connect-src": "'self' http://localhost:*",
                "media-src": "'self' blob:"
            } } }
        }))
        .unwrap();
        assert!(matches!(
            config.app.security.csp,
            Some(Csp::DirectiveMap(_))
        ));
        config
    }

    fn sources(config: &Config, directive: &str) -> Vec<String> {
        let map: std::collections::HashMap<_, _> = config.app.security.csp.clone().unwrap().into();
        map.get(directive)
            .cloned()
            .map(Into::into)
            .unwrap_or_default()
    }

    #[test]
    fn leaves_policy_unchanged_for_loopback_servers() {
        let mut config = config_with_policy();
        let before = config.app.security.csp.clone();
        for url in ["http://localhost:8080", "http://127.0.0.1:9000"] {
            allow_server_origin(&mut config, &Url::parse(url).unwrap());
        }
        assert_eq!(config.app.security.csp, before);
    }

    #[test]
    fn adds_remote_server_origin_and_socket_origin() {
        let mut config = config_with_policy();
        allow_server_origin(&mut config, &Url::parse("http://10.0.0.5:8080").unwrap());

        let connect = sources(&config, "connect-src");
        assert!(connect.contains(&"http://10.0.0.5:8080".to_string()));
        assert!(connect.contains(&"ws://10.0.0.5:8080".to_string()));
        assert!(connect.contains(&"http://localhost:*".to_string()));
        assert!(sources(&config, "media-src").contains(&"http://10.0.0.5:8080".to_string()));
        assert!(sources(&config, "img-src").contains(&"http://10.0.0.5:8080".to_string()));
        assert_eq!(sources(&config, "default-src"), vec!["'self'".to_string()]);
    }

    #[test]
    fn uses_secure_socket_scheme_for_https_servers() {
        let mut config = config_with_policy();
        allow_server_origin(
            &mut config,
            &Url::parse("https://izwi.example.com").unwrap(),
        );
        let connect = sources(&config, "connect-src");
        assert!(connect.contains(&"https://izwi.example.com".to_string()));
        assert!(connect.contains(&"wss://izwi.example.com".to_string()));
    }

    #[test]
    fn does_nothing_without_a_policy() {
        let mut config = config_with_policy();
        config.app.security.csp = None;
        allow_server_origin(&mut config, &Url::parse("http://10.0.0.5:8080").unwrap());
        assert!(config.app.security.csp.is_none());
    }
}
