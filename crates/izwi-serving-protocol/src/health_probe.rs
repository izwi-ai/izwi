//! The `healthcheck` subcommands' HTTP probe. Container health checks used
//! to shell out to curl, which put curl in every image and bearer tokens on
//! the worker's command line. Each binary now probes its own listener with
//! this plain HTTP/1.1 GET: loopback only, so no TLS, proxy or client
//! library, and credentials stay in the environment.

use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

/// How long a probe may take end to end per I/O step.
pub const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// The address a health check dials for a listener bound to `host:port`:
/// a wildcard bind is reached on the matching loopback address.
pub fn probe_address(host: &str, port: u16) -> io::Result<SocketAddr> {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) if ip.is_unspecified() => {
            Ok(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port))
        }
        Ok(IpAddr::V6(ip)) if ip.is_unspecified() => {
            Ok(SocketAddr::new(Ipv6Addr::LOCALHOST.into(), port))
        }
        Ok(ip) => Ok(SocketAddr::new(ip, port)),
        Err(_) => (host, port).to_socket_addrs()?.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("{host} has no address"))
        }),
    }
}

/// GET `path` from `addr` and return the response status code.
pub fn probe_http(
    addr: SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
    timeout: Duration,
) -> io::Result<u16> {
    let invalid = |what: &str| io::Error::new(io::ErrorKind::InvalidInput, what.to_string());
    if !path.starts_with('/') || path.contains(char::is_whitespace) {
        return Err(invalid(
            "probe path must be an absolute path without spaces",
        ));
    }
    let mut request = format!(
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nUser-Agent: izwi-healthcheck\r\n\
         Connection: close\r\n"
    );
    for (name, value) in headers {
        if name.contains([':', '\r', '\n']) || value.contains(['\r', '\n']) {
            return Err(invalid("probe header names and values must be single-line"));
        }
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");

    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.write_all(request.as_bytes())?;

    // Only the status line matters: "HTTP/1.1 200 OK".
    let mut head = Vec::with_capacity(64);
    let mut byte = [0u8; 1];
    while head.len() < 1024 && !head.ends_with(b"\r\n") {
        if stream.read(&mut byte)? == 0 {
            break;
        }
        head.push(byte[0]);
    }
    let line = String::from_utf8_lossy(&head);
    let mut parts = line.split_whitespace();
    match (parts.next(), parts.next().map(str::parse::<u16>)) {
        (Some(version), Some(Ok(status))) if version.starts_with("HTTP/") => Ok(status),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("not an HTTP response: {:?}", line.trim_end()),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    fn serve_once(response: &'static str) -> (SocketAddr, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 512];
            while !request.ends_with(b"\r\n\r\n") {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            stream.write_all(response.as_bytes()).unwrap();
            String::from_utf8(request).unwrap()
        });
        (addr, handle)
    }

    #[test]
    fn wildcard_binds_are_probed_on_loopback() {
        assert_eq!(
            probe_address("0.0.0.0", 8080).unwrap(),
            "127.0.0.1:8080".parse().unwrap()
        );
        assert_eq!(
            probe_address("[::]", 9).unwrap(),
            "[::1]:9".parse().unwrap()
        );
        assert_eq!(
            probe_address("127.0.0.1", 9470).unwrap(),
            "127.0.0.1:9470".parse().unwrap()
        );
    }

    #[test]
    fn returns_the_status_and_sends_headers() {
        let (addr, server) = serve_once("HTTP/1.1 503 Service Unavailable\r\n\r\n");
        let status = probe_http(
            addr,
            "/readyz",
            &[("authorization", "Bearer token")],
            HEALTH_PROBE_TIMEOUT,
        )
        .unwrap();
        assert_eq!(status, 503);
        let request = server.join().unwrap();
        assert!(request.starts_with("GET /readyz HTTP/1.1\r\n"), "{request}");
        assert!(request.contains(&format!("Host: {addr}\r\n")), "{request}");
        assert!(
            request.contains("authorization: Bearer token\r\n"),
            "{request}"
        );
    }

    #[test]
    fn rejects_injection_and_non_http_replies() {
        let addr: SocketAddr = "127.0.0.1:9".parse().unwrap();
        for (path, headers) in [
            ("readyz", &[][..]),
            ("/a b", &[][..]),
            ("/readyz", &[("x", "a\r\nInjected: 1")][..]),
        ] {
            assert_eq!(
                probe_http(addr, path, headers, HEALTH_PROBE_TIMEOUT)
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
        let (addr, server) = serve_once("SSH-2.0-OpenSSH\r\n");
        assert_eq!(
            probe_http(addr, "/", &[], HEALTH_PROBE_TIMEOUT)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        server.join().unwrap();
    }
}
