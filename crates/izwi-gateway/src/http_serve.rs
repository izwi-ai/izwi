//! The HTTP listener of the local server and the gateway: an axum router on
//! hyper, with the protections `axum::serve` cannot configure because it
//! gives hyper no timer.
//!
//! - Request headers must arrive within [`HttpLimits::header_read_timeout`],
//!   so a client that trickles them (slowloris) cannot hold a connection.
//! - At most [`HttpLimits::max_connections`] connections are open at once;
//!   past that, new ones wait in the kernel's accept backlog instead of each
//!   getting a task.
//! - The shutdown signal stops accepting, asks open connections to close once
//!   their in-flight requests finish, and resolves when they have, as
//!   `axum::serve(..).with_graceful_shutdown(..)` did.

use std::future::Future;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder;
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
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
    let mut builder = Builder::new(TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(limits.header_read_timeout);
    // CONNECT carries WebSockets over HTTP/2.
    builder
        .http2()
        .timer(TokioTimer::new())
        .enable_connect_protocol();
    let graceful = GracefulShutdown::new();
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
        let connection = graceful.watch(
            builder
                .serve_connection_with_upgrades(
                    TokioIo::new(stream),
                    TowerToHyperService::new(router.clone()),
                )
                .into_owned(),
        );
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                debug!(%error, "HTTP connection ended with an error");
            }
            drop(permit);
        });
    }

    drop(listener);
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
    use axum::routing::get;
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
        let router = Router::new().route("/", get(|| async { "ok" }));
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
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .expect("shutdown finishes")
            .unwrap()
            .unwrap();
        assert!(TcpStream::connect(addr).await.is_err());
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
