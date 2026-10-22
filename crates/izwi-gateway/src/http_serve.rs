//! The HTTP listener of the local server and the gateway: an axum router on
//! hyper, with the protections `axum::serve` cannot configure because it
//! gives hyper no timer.
//!
//! - Request headers must arrive within [`HttpLimits::header_read_timeout`],
//!   counted from the moment the connection opens, so neither a client that
//!   trickles headers (slowloris) nor one that sends nothing holds a
//!   connection.
//! - Each connection serves one request: every response carries
//!   `Connection: close`, because hyper never times out a keep-alive
//!   connection waiting for its next request, so idle ones could otherwise
//!   fill the connection cap. A new connection per request costs little on
//!   loopback or behind a proxy. A WebSocket's 101 keeps its own
//!   `Connection: upgrade` (hyper's `keep_alive(false)` would overwrite it and
//!   break the upgrade), and streamed responses are unaffected.
//! - At most [`HttpLimits::max_connections`] HTTP connections are open at
//!   once; past that, new ones wait in the kernel's accept backlog instead of
//!   each getting a task. A WebSocket leaves the count once it upgrades.
//! - The shutdown signal stops accepting, asks open connections to close once
//!   their in-flight requests finish, and resolves when they have, as
//!   `axum::serve(..).with_graceful_shutdown(..)` did.
//!
//! The listener speaks HTTP/1.1 only. Browsers use HTTP/2 only over TLS, which
//! terminates in front of this listener, and the HTTP/2 protocol sniff would
//! wait for a client's first bytes with no timeout, before the header timer
//! starts: idle connections could then fill the connection cap.

use std::future::Future;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use axum::http::{header, HeaderValue};
use axum::Router;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder;
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::net::TcpListener;
use tokio::sync::{watch, Semaphore};
use tower::Layer;
use tower_http::set_header::SetResponseHeaderLayer;
use tracing::{debug, warn};

/// Connection limits for [`serve_router_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpLimits {
    pub header_read_timeout: Duration,
    pub max_connections: usize,
}

impl Default for HttpLimits {
    fn default() -> Self {
        Self {
            header_read_timeout: Duration::from_secs(30),
            max_connections: 1024,
        }
    }
}

/// Serve `router` on `listener` with the default [`HttpLimits`] until
/// `signal` resolves and the open connections have finished.
pub async fn serve_router(
    listener: TcpListener,
    router: Router,
    signal: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    serve_router_with(listener, router, signal, HttpLimits::default()).await
}

pub async fn serve_router_with(
    listener: TcpListener,
    router: Router,
    signal: impl Future<Output = ()> + Send + 'static,
    limits: HttpLimits,
) -> io::Result<()> {
    let mut builder = Builder::new(TokioExecutor::new()).http1_only();
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(limits.header_read_timeout);
    // hyper closes the connection after a response that says so; a
    // WebSocket's 101 already carries `Connection: upgrade` and keeps it.
    let service = SetResponseHeaderLayer::if_not_present(
        header::CONNECTION,
        HeaderValue::from_static("close"),
    )
    .layer(router);
    let graceful = GracefulShutdown::new();
    // Ends the first-byte waits below when shutdown begins; each holds a
    // graceful-shutdown watcher, which shutdown would otherwise wait out.
    let (stopping_tx, stopping) = watch::channel(false);
    let connections = Arc::new(Semaphore::new(limits.max_connections));
    tokio::pin!(signal);

    loop {
        let permit = tokio::select! {
            permit = connections.clone().acquire_owned() => {
                permit.expect("the connection semaphore is never closed")
            }
            () = &mut signal => break,
        };
        let stream = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(error) => {
                    accept_backoff(error).await;
                    continue;
                }
            },
            () = &mut signal => break,
        };
        let watcher = graceful.watcher();
        let builder = builder.clone();
        let service = service.clone();
        let mut stopping = stopping.clone();
        tokio::spawn(async move {
            let _permit = permit;
            // hyper starts its header timer at the first byte, so a client
            // that sends nothing is cut off here instead.
            let mut probe = [0u8; 1];
            let first_byte = tokio::select! {
                peeked = tokio::time::timeout(limits.header_read_timeout, stream.peek(&mut probe)) => {
                    matches!(peeked, Ok(Ok(read)) if read > 0)
                }
                _ = stopping.wait_for(|stopping| *stopping) => false,
            };
            if !first_byte {
                return;
            }
            let connection = builder
                .serve_connection_with_upgrades(
                    TokioIo::new(stream),
                    TowerToHyperService::new(service),
                )
                .into_owned();
            if let Err(error) = watcher.watch(connection).await {
                debug!(%error, "HTTP connection ended with an error");
            }
        });
    }

    drop(listener);
    let _ = stopping_tx.send(true);
    graceful.shutdown().await;
    Ok(())
}

/// A failed accept that belongs to one client is skipped; anything else
/// (out of file descriptors) would spin, so wait before accepting again.
async fn accept_backoff(error: io::Error) {
    if matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) {
        return;
    }
    warn!(%error, "accepting an HTTP connection failed; retrying in 1 s");
    tokio::time::sleep(Duration::from_secs(1)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::ws::WebSocketUpgrade;
    use axum::routing::get;
    use futures::{SinkExt, StreamExt};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use tokio::sync::oneshot;

    async fn start(
        limits: HttpLimits,
    ) -> (
        std::net::SocketAddr,
        oneshot::Sender<()>,
        tokio::task::JoinHandle<io::Result<()>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = Router::new().route("/", get(|| async { "ok" })).route(
            "/ws",
            get(|upgrade: WebSocketUpgrade| async move {
                upgrade.on_upgrade(|mut socket| async move {
                    while let Some(Ok(message)) = socket.recv().await {
                        if socket.send(message).await.is_err() {
                            break;
                        }
                    }
                })
            }),
        );
        let (stop, stopped) = oneshot::channel::<()>();
        let server = tokio::spawn(serve_router_with(
            listener,
            router,
            async move {
                let _ = stopped.await;
            },
            limits,
        ));
        (addr, stop, server)
    }

    async fn get_root(addr: std::net::SocketAddr) -> String {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET / HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn serves_requests_and_drains_on_shutdown() {
        let (addr, stop, server) = start(HttpLimits::default()).await;
        assert!(get_root(addr).await.ends_with("ok"));
        // A connection still waiting for its first byte must not hold up
        // shutdown for the 30 s header window.
        let _silent = TcpStream::connect(addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .expect("shutdown finishes")
            .unwrap()
            .unwrap();
        assert!(TcpStream::connect(addr).await.is_err());
    }

    #[tokio::test]
    async fn websockets_upgrade_and_outlive_the_one_request_rule() {
        let (addr, stop, _server) = start(HttpLimits::default()).await;
        let (mut socket, response) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
            .await
            .expect("the WebSocket upgrades");
        assert_eq!(response.status(), 101);
        for text in ["one", "two"] {
            socket
                .send(tokio_tungstenite::tungstenite::Message::text(text))
                .await
                .unwrap();
            let echoed = socket.next().await.unwrap().unwrap();
            assert_eq!(echoed.into_text().unwrap().as_str(), text);
        }
        drop(stop);
    }

    #[tokio::test]
    async fn trickled_headers_are_cut_off() {
        let (addr, stop, _server) = start(HttpLimits {
            header_read_timeout: Duration::from_millis(200),
            ..HttpLimits::default()
        })
        .await;
        let mut slow = TcpStream::connect(addr).await.unwrap();
        slow.write_all(b"GET / HTTP/1.1\r\nHost: test\r\n")
            .await
            .unwrap();
        // A reset or an EOF (possibly after a 408) both mean the server let
        // go of the connection.
        let mut rest = Vec::new();
        let _closed = tokio::time::timeout(Duration::from_secs(5), slow.read_to_end(&mut rest))
            .await
            .expect("the server closes a connection whose headers stall");
        drop(stop);
    }

    #[tokio::test]
    async fn silent_and_finished_connections_are_closed() {
        let limits = HttpLimits {
            header_read_timeout: Duration::from_millis(200),
            ..HttpLimits::default()
        };
        let (addr, stop, _server) = start(limits).await;

        // A connection that never sends a byte.
        let mut silent = TcpStream::connect(addr).await.unwrap();
        let mut rest = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), silent.read_to_end(&mut rest))
            .await
            .expect("the server closes a connection that sends nothing")
            .ok();

        // A client that asks for keep-alive still gets one request.
        let mut idle = TcpStream::connect(addr).await.unwrap();
        idle.write_all(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), idle.read_to_end(&mut response))
            .await
            .expect("the server closes the connection after its response")
            .ok();
        let response = String::from_utf8_lossy(&response);
        assert!(response.contains("connection: close"), "{response}");
        assert!(response.ends_with("ok"), "{response}");
        drop(stop);
    }

    #[tokio::test]
    async fn connections_past_the_cap_wait_for_a_free_slot() {
        let (addr, stop, _server) = start(HttpLimits {
            max_connections: 1,
            ..HttpLimits::default()
        })
        .await;
        let held = TcpStream::connect(addr).await.unwrap();
        // Let the server accept the first connection.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let waiting = tokio::spawn(get_root(addr));
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!waiting.is_finished(), "the second connection was served");
        drop(held);
        let response = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("the second connection is served once the first closes")
            .unwrap();
        assert!(response.ends_with("ok"));
        drop(stop);
    }
}
