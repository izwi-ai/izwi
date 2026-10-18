//! Optional OTLP/HTTP (JSON) export of request spans.
//!
//! Off unless `IZWI_OTLP_TRACES_ENDPOINT` names a collector's traces URL
//! (for an OpenTelemetry Collector, `http://host:4318/v1/traces`). Trace
//! context propagation and the ids in logs work without it.
//!
//! Only request spans are exported: the edge's `http_request` and the
//! worker's `worker_request`, which carry the W3C `trace_id`, `span_id` and
//! `parent_span_id` fields. Spans queue in a bounded channel and a background
//! thread posts them in batches; when the queue is full a span is dropped
//! rather than slowing a request.

use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// The collector traces URL; unset or empty disables export.
pub const OTLP_TRACES_ENDPOINT_ENV: &str = "IZWI_OTLP_TRACES_ENDPOINT";

const EXPORTED_SPANS: &[&str] = &["http_request", "worker_request"];
const QUEUE_CAPACITY: usize = 4096;
const MAX_BATCH: usize = 512;
const EXPORT_TIMEOUT: Duration = Duration::from_secs(5);
/// OTLP span kind SERVER.
const SPAN_KIND_SERVER: u8 = 2;

/// The exporter layer when `IZWI_OTLP_TRACES_ENDPOINT` is set, else `None`.
pub fn otlp_layer_from_env(service_name: &'static str) -> Option<OtlpLayer> {
    let endpoint = std::env::var(OTLP_TRACES_ENDPOINT_ENV).ok()?;
    let endpoint = endpoint.trim();
    (!endpoint.is_empty())
        .then(|| OtlpLayer::new(endpoint.to_string(), service_name, Duration::from_secs(2)))
}

/// A tracing layer that exports request spans to an OTLP/HTTP collector.
pub struct OtlpLayer {
    spans: SyncSender<Value>,
}

impl OtlpLayer {
    /// Export to `endpoint`, posting at least every `interval`.
    pub fn new(endpoint: String, service_name: &'static str, interval: Duration) -> Self {
        let (spans, queued) = sync_channel(QUEUE_CAPACITY);
        std::thread::Builder::new()
            .name("izwi-otlp-export".into())
            .spawn(move || export_loop(endpoint, service_name, interval, queued))
            .expect("spawn the OTLP export thread");
        Self { spans }
    }
}

/// What an exported span has recorded so far.
#[derive(Default)]
struct SpanRecord {
    started: Option<SystemTime>,
    fields: Vec<(String, String)>,
}

impl SpanRecord {
    fn field(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .rev()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

impl Visit for SpanRecord {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields
            .push((field.name().to_string(), format!("{value:?}")));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }
}

impl<S> Layer<S> for OtlpLayer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if !EXPORTED_SPANS.contains(&attrs.metadata().name()) {
            return;
        }
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut record = SpanRecord {
            started: Some(SystemTime::now()),
            ..SpanRecord::default()
        };
        attrs.record(&mut record);
        span.extensions_mut().insert(record);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(record) = span.extensions_mut().get_mut::<SpanRecord>() {
                values.record(record);
            }
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let Some(record) = span.extensions_mut().remove::<SpanRecord>() else {
            return;
        };
        if let Some(exported) = otlp_span(span.name(), &record, SystemTime::now()) {
            // A full queue drops the span: export never slows a request.
            let _ = self.spans.try_send(exported);
        }
    }
}

fn unix_nanos(at: SystemTime) -> String {
    at.duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0)
        .to_string()
}

fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// One OTLP JSON span, or `None` when the span carries no valid trace ids.
fn otlp_span(name: &str, record: &SpanRecord, ended: SystemTime) -> Option<Value> {
    let trace_id = record.field("trace_id").filter(|id| valid_hex(id, 32))?;
    let span_id = record.field("span_id").filter(|id| valid_hex(id, 16))?;
    let mut span = json!({
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": SPAN_KIND_SERVER,
        "startTimeUnixNano": unix_nanos(record.started.unwrap_or(ended)),
        "endTimeUnixNano": unix_nanos(ended),
    });
    if let Some(parent) = record
        .field("parent_span_id")
        .filter(|id| valid_hex(id, 16))
    {
        span["parentSpanId"] = json!(parent);
    }
    let mut attributes = Vec::new();
    for (key, attribute) in [
        ("method", "http.request.method"),
        ("path", "url.path"),
        ("status", "http.response.status_code"),
        ("correlation_id", "izwi.request.id"),
    ] {
        if let Some(value) = record.field(key) {
            attributes.push(json!({"key": attribute, "value": {"stringValue": value}}));
        }
    }
    span["attributes"] = Value::Array(attributes);
    // OTLP status: 2 = ERROR for a server error, else unset.
    let failed = record
        .field("status")
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| status >= 500);
    span["status"] = json!({ "code": if failed { 2 } else { 0 } });
    Some(span)
}

fn export_request(service_name: &str, spans: Vec<Value>) -> Value {
    json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": service_name}}
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "izwi"},
                "spans": spans,
            }]
        }]
    })
}

fn export_loop(
    endpoint: String,
    service_name: &'static str,
    interval: Duration,
    queued: Receiver<Value>,
) {
    let client = match reqwest::blocking::Client::builder()
        .timeout(EXPORT_TIMEOUT)
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            eprintln!("OTLP export disabled: {error}");
            return;
        }
    };
    let mut batch = Vec::new();
    let mut flush_at = Instant::now() + interval;
    loop {
        let closed = match queued.recv_timeout(flush_at.saturating_duration_since(Instant::now())) {
            Ok(span) => {
                batch.push(span);
                false
            }
            Err(RecvTimeoutError::Timeout) => false,
            Err(RecvTimeoutError::Disconnected) => true,
        };
        if !batch.is_empty() && (closed || batch.len() >= MAX_BATCH || Instant::now() >= flush_at) {
            let body = export_request(service_name, std::mem::take(&mut batch));
            // Export is best effort; a collector outage only loses spans.
            let _ = client.post(&endpoint).json(&body).send();
        }
        if Instant::now() >= flush_at {
            flush_at = Instant::now() + interval;
        }
        if closed {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use tracing_subscriber::layer::SubscriberExt;

    /// Accept one HTTP request and return its body.
    fn collector() -> (String, std::thread::JoinHandle<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let handle = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(_) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    Err(_) => return String::new(),
                }
            };
            stream.set_nonblocking(false).unwrap();
            let mut reader = BufReader::new(stream);
            let mut length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            reader
                .get_mut()
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                .unwrap();
            String::from_utf8(body).unwrap()
        });
        (endpoint, handle)
    }

    #[test]
    fn request_spans_are_exported_as_otlp_json() {
        let (endpoint, received) = collector();
        let layer = OtlpLayer::new(endpoint, "izwi-test", Duration::from_millis(50));
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(
                "http_request",
                method = "POST",
                path = "/v1/chat/completions",
                trace_id = "4bf92f3577b34da6a3ce929d0e0e4736",
                span_id = "00f067aa0ba902b7",
                parent_span_id = "a3ce929d0e0e4736",
                status = tracing::field::Empty,
            );
            span.record("status", 503u64);
            // Unexported spans and spans without ids are ignored.
            drop(tracing::info_span!("other_span", trace_id = "x"));
            drop(span);
        });

        let body = received.join().unwrap();
        assert!(!body.is_empty(), "nothing was exported");
        let body: Value = serde_json::from_str(&body).unwrap();
        let resource = &body["resourceSpans"][0];
        assert_eq!(
            resource["resource"]["attributes"][0]["value"]["stringValue"],
            "izwi-test"
        );
        let spans = resource["scopeSpans"][0]["spans"].as_array().unwrap();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(span["spanId"], "00f067aa0ba902b7");
        assert_eq!(span["parentSpanId"], "a3ce929d0e0e4736");
        assert_eq!(span["name"], "http_request");
        assert_eq!(span["status"]["code"], 2);
        let start: u128 = span["startTimeUnixNano"].as_str().unwrap().parse().unwrap();
        let end: u128 = span["endTimeUnixNano"].as_str().unwrap().parse().unwrap();
        assert!(start > 0 && start <= end);
        assert!(span["attributes"].as_array().unwrap().contains(
            &json!({"key": "url.path", "value": {"stringValue": "/v1/chat/completions"}})
        ));
    }

    #[test]
    fn export_is_off_without_an_endpoint() {
        std::env::remove_var(OTLP_TRACES_ENDPOINT_ENV);
        assert!(otlp_layer_from_env("izwi-test").is_none());
    }
}
