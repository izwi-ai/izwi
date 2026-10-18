//! W3C Trace Context for requests that cross process boundaries.
//!
//! The edge accepts a caller's `traceparent` (or starts a trace), each
//! process gives its own work a fresh span id, and the gateway sends its span
//! as the parent on worker calls and realtime dials. Logs carry the ids, so
//! one trace id finds a request in every process.

use std::fmt;

/// The W3C `traceparent` header.
pub const TRACEPARENT_HEADER: &str = "traceparent";

const SAMPLED: u8 = 0x01;

/// One span's position in a trace: the trace id, this span's id and its
/// sampling decision.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceContext {
    trace_id: [u8; 16],
    span_id: [u8; 8],
    flags: u8,
}

impl TraceContext {
    /// A new trace, sampled, with a random trace and span id.
    pub fn new_root() -> Self {
        Self {
            trace_id: random_nonzero::<16>(),
            span_id: random_nonzero::<8>(),
            flags: SAMPLED,
        }
    }

    /// A new span in the same trace, keeping the sampling decision.
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id,
            span_id: random_nonzero::<8>(),
            flags: self.flags & SAMPLED,
        }
    }

    /// Continue the trace of an incoming `traceparent`, or start one when it
    /// is absent or malformed. Returns this process's span and the caller's
    /// span id when there was one.
    pub fn continue_from(header: Option<&str>) -> (Self, Option<[u8; 8]>) {
        match header.and_then(Self::parse) {
            Some(parent) => (parent.child(), Some(parent.span_id)),
            None => (Self::new_root(), None),
        }
    }

    /// Parse a `traceparent` value. Version `00` must be exactly four
    /// fields; a later version may append fields, which are ignored. All-zero
    /// ids, version `ff` and uppercase hex are rejected.
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        let mut fields = value.split('-');
        let version = parse_hex::<1>(fields.next()?)?[0];
        let trace_id = parse_hex::<16>(fields.next()?)?;
        let span_id = parse_hex::<8>(fields.next()?)?;
        let flags = parse_hex::<1>(fields.next()?)?[0];
        if version == 0xff
            || (version == 0 && fields.next().is_some())
            || trace_id == [0; 16]
            || span_id == [0; 8]
        {
            return None;
        }
        Some(Self {
            trace_id,
            span_id,
            flags,
        })
    }

    pub fn trace_id(&self) -> String {
        hex(&self.trace_id)
    }

    pub fn span_id(&self) -> String {
        hex(&self.span_id)
    }

    pub fn trace_id_bytes(&self) -> [u8; 16] {
        self.trace_id
    }

    pub fn span_id_bytes(&self) -> [u8; 8] {
        self.span_id
    }

    pub fn sampled(&self) -> bool {
        self.flags & SAMPLED != 0
    }

    /// The version `00` `traceparent` value naming this span as the parent.
    pub fn header_value(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id(),
            self.span_id(),
            self.flags & SAMPLED
        )
    }
}

impl fmt::Debug for TraceContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.header_value())
    }
}

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_hex<const N: usize>(field: &str) -> Option<[u8; N]> {
    if field.len() != N * 2
        || !field
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut bytes = [0u8; N];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&field[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

fn random_nonzero<const N: usize>() -> [u8; N] {
    loop {
        let random = uuid::Uuid::new_v4().into_bytes();
        let mut bytes = [0u8; N];
        bytes.copy_from_slice(&random[..N]);
        if bytes != [0; N] {
            return bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn parses_and_renders_the_w3c_example() {
        let context = TraceContext::parse(EXAMPLE).expect("valid traceparent");
        assert_eq!(context.trace_id(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(context.span_id(), "00f067aa0ba902b7");
        assert!(context.sampled());
        assert_eq!(context.header_value(), EXAMPLE);
    }

    #[test]
    fn rejects_malformed_values() {
        for invalid in [
            "",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
            "0-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        ] {
            assert!(TraceContext::parse(invalid).is_none(), "{invalid:?}");
        }
        // A later version may append fields.
        let future = "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-later";
        assert!(TraceContext::parse(future).is_some());
    }

    #[test]
    fn continuing_keeps_the_trace_and_starts_a_new_span() {
        let (context, parent) = TraceContext::continue_from(Some(EXAMPLE));
        assert_eq!(context.trace_id(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_ne!(context.span_id(), "00f067aa0ba902b7");
        assert_eq!(
            parent.map(|id| hex(&id)).as_deref(),
            Some("00f067aa0ba902b7")
        );

        let (root, parent) = TraceContext::continue_from(Some("garbage"));
        assert!(parent.is_none());
        assert!(root.sampled());
        assert_ne!(root.trace_id(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_ne!(TraceContext::new_root(), TraceContext::new_root());
    }

    #[test]
    fn unsampled_parents_stay_unsampled() {
        let parent =
            TraceContext::parse("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00").unwrap();
        assert!(!parent.child().sampled());
        assert!(parent.child().header_value().ends_with("-00"));
    }
}
