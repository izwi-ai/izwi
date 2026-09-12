use serde::de::DeserializeOwned;
use std::marker::PhantomData;

pub const DEFAULT_MAX_NDJSON_LINE_BYTES: usize = 256 * 1024;
pub const DEFAULT_MAX_NDJSON_TOTAL_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_MAX_NDJSON_EVENTS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalFragmentPolicy {
    /// A non-empty final fragment without a newline is a truncated stream.
    RequireNewline,
    /// Parse one final non-empty fragment when EOF is explicitly reported with `finish`.
    AllowAtEof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NdjsonLimits {
    pub max_line_bytes: usize,
    pub max_total_bytes: usize,
    pub max_events: usize,
    pub final_fragment: FinalFragmentPolicy,
}

impl Default for NdjsonLimits {
    fn default() -> Self {
        Self {
            max_line_bytes: DEFAULT_MAX_NDJSON_LINE_BYTES,
            max_total_bytes: DEFAULT_MAX_NDJSON_TOTAL_BYTES,
            max_events: DEFAULT_MAX_NDJSON_EVENTS,
            final_fragment: FinalFragmentPolicy::RequireNewline,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NdjsonDecodeError {
    #[error("invalid NDJSON limits: line, total, and event limits must be non-zero and line must not exceed total")]
    InvalidLimits,
    #[error("NDJSON stream exceeded total byte limit of {limit}")]
    TotalBytesExceeded { limit: usize },
    #[error("NDJSON line {line} exceeded byte limit of {limit}")]
    LineBytesExceeded { line: usize, limit: usize },
    #[error("NDJSON stream exceeded event limit of {limit}")]
    EventCountExceeded { limit: usize },
    #[error("NDJSON line {line} is empty")]
    EmptyLine { line: usize },
    #[error("malformed JSON on NDJSON line {line}: {message}")]
    MalformedJson { line: usize, message: String },
    #[error("NDJSON stream ended with an unterminated final fragment of {bytes} bytes")]
    UnterminatedFinalFragment { bytes: usize },
    #[error("NDJSON decoder was already finished")]
    AlreadyFinished,
}

/// Incrementally parses a bounded NDJSON response without accumulating the full stream.
///
/// Once an error occurs the decoder is poisoned, preventing callers from accidentally treating a
/// suffix of a malformed stream as a new valid stream.
pub struct NdjsonDecoder<T> {
    limits: NdjsonLimits,
    buffer: Vec<u8>,
    total_bytes: usize,
    event_count: usize,
    next_line: usize,
    finished: bool,
    poisoned: bool,
    marker: PhantomData<fn() -> T>,
}

impl<T> NdjsonDecoder<T>
where
    T: DeserializeOwned,
{
    pub fn new(limits: NdjsonLimits) -> Result<Self, NdjsonDecodeError> {
        if limits.max_line_bytes == 0
            || limits.max_total_bytes == 0
            || limits.max_events == 0
            || limits.max_line_bytes > limits.max_total_bytes
        {
            return Err(NdjsonDecodeError::InvalidLimits);
        }
        Ok(Self {
            limits,
            buffer: Vec::with_capacity(limits.max_line_bytes.min(8192)),
            total_bytes: 0,
            event_count: 0,
            next_line: 1,
            finished: false,
            poisoned: false,
            marker: PhantomData,
        })
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<T>, NdjsonDecodeError> {
        if self.finished || self.poisoned {
            return Err(NdjsonDecodeError::AlreadyFinished);
        }
        let new_total = self
            .total_bytes
            .checked_add(chunk.len())
            .filter(|total| *total <= self.limits.max_total_bytes)
            .ok_or_else(|| {
                self.fail(NdjsonDecodeError::TotalBytesExceeded {
                    limit: self.limits.max_total_bytes,
                })
            })?;
        self.total_bytes = new_total;

        let mut events = Vec::new();
        let mut fragment_start = 0;
        for (offset, byte) in chunk.iter().enumerate() {
            if *byte != b'\n' {
                continue;
            }
            self.extend_fragment(&chunk[fragment_start..offset])?;
            let mut line = std::mem::take(&mut self.buffer);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            match self.parse_line(&line) {
                Ok(event) => events.push(event),
                Err(error) => {
                    self.poisoned = true;
                    return Err(error);
                }
            }
            fragment_start = offset + 1;
        }
        self.extend_fragment(&chunk[fragment_start..])?;
        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<T>, NdjsonDecodeError> {
        if self.finished || self.poisoned {
            return Err(NdjsonDecodeError::AlreadyFinished);
        }
        self.finished = true;
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        if self.buffer.len() > self.limits.max_line_bytes {
            return Err(NdjsonDecodeError::LineBytesExceeded {
                line: self.next_line,
                limit: self.limits.max_line_bytes,
            });
        }
        match self.limits.final_fragment {
            FinalFragmentPolicy::RequireNewline => {
                Err(NdjsonDecodeError::UnterminatedFinalFragment {
                    bytes: self.buffer.len(),
                })
            }
            FinalFragmentPolicy::AllowAtEof => {
                let line = std::mem::take(&mut self.buffer);
                self.parse_line(&line).map(|event| vec![event])
            }
        }
    }

    pub const fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub const fn event_count(&self) -> usize {
        self.event_count
    }

    fn extend_fragment(&mut self, fragment: &[u8]) -> Result<(), NdjsonDecodeError> {
        if self.buffer.len().saturating_add(fragment.len()) > self.limits.max_line_bytes {
            return Err(self.fail(NdjsonDecodeError::LineBytesExceeded {
                line: self.next_line,
                limit: self.limits.max_line_bytes,
            }));
        }
        self.buffer.extend_from_slice(fragment);
        Ok(())
    }

    fn parse_line(&mut self, line: &[u8]) -> Result<T, NdjsonDecodeError> {
        if line.is_empty() {
            return Err(NdjsonDecodeError::EmptyLine {
                line: self.next_line,
            });
        }
        if self.event_count >= self.limits.max_events {
            return Err(NdjsonDecodeError::EventCountExceeded {
                limit: self.limits.max_events,
            });
        }
        let line_number = self.next_line;
        let event =
            serde_json::from_slice(line).map_err(|error| NdjsonDecodeError::MalformedJson {
                line: line_number,
                message: error.to_string(),
            })?;
        self.event_count += 1;
        self.next_line += 1;
        Ok(event)
    }

    fn fail(&mut self, error: NdjsonDecodeError) -> NdjsonDecodeError {
        self.poisoned = true;
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, PartialEq, Eq, Deserialize)]
    struct Event {
        value: u32,
    }

    fn limits() -> NdjsonLimits {
        NdjsonLimits {
            max_line_bytes: 32,
            max_total_bytes: 96,
            max_events: 2,
            final_fragment: FinalFragmentPolicy::RequireNewline,
        }
    }

    #[test]
    fn parses_fragmented_and_coalesced_lines_incrementally() {
        let mut decoder = NdjsonDecoder::<Event>::new(limits()).unwrap();
        assert!(decoder.push(b"{\"val").unwrap().is_empty());
        assert_eq!(
            decoder.push(b"ue\":1}\n{\"value\":2}\r\n").unwrap(),
            vec![Event { value: 1 }, Event { value: 2 }]
        );
        assert!(decoder.finish().unwrap().is_empty());
        assert_eq!(decoder.event_count(), 2);
    }

    #[test]
    fn rejects_oversized_total_before_buffering_chunk() {
        let mut decoder = NdjsonDecoder::<Event>::new(limits()).unwrap();
        let error = decoder.push(&[b'x'; 97]).unwrap_err();
        assert_eq!(error, NdjsonDecodeError::TotalBytesExceeded { limit: 96 });
        assert_eq!(decoder.total_bytes(), 0);
        assert_eq!(
            decoder.push(b"{\"value\":1}\n").unwrap_err(),
            NdjsonDecodeError::AlreadyFinished
        );
    }

    #[test]
    fn rejects_oversized_line_with_and_without_newline() {
        let mut decoder = NdjsonDecoder::<Event>::new(limits()).unwrap();
        assert!(matches!(
            decoder.push(&[b'x'; 33]),
            Err(NdjsonDecodeError::LineBytesExceeded { line: 1, limit: 32 })
        ));
        assert!(decoder.buffer.len() <= 32);

        let mut decoder = NdjsonDecoder::<Event>::new(limits()).unwrap();
        let mut line = vec![b'x'; 33];
        line.push(b'\n');
        assert!(matches!(
            decoder.push(&line),
            Err(NdjsonDecodeError::LineBytesExceeded { line: 1, limit: 32 })
        ));
    }

    #[test]
    fn rejects_excess_events_and_malformed_or_empty_lines() {
        let mut decoder = NdjsonDecoder::<Event>::new(limits()).unwrap();
        decoder.push(b"{\"value\":1}\n{\"value\":2}\n").unwrap();
        assert_eq!(
            decoder.push(b"{\"value\":3}\n").unwrap_err(),
            NdjsonDecodeError::EventCountExceeded { limit: 2 }
        );

        let mut decoder = NdjsonDecoder::<Event>::new(limits()).unwrap();
        assert!(matches!(
            decoder.push(b"not-json\n"),
            Err(NdjsonDecodeError::MalformedJson { line: 1, .. })
        ));

        let mut decoder = NdjsonDecoder::<Event>::new(limits()).unwrap();
        assert_eq!(
            decoder.push(b"\n").unwrap_err(),
            NdjsonDecodeError::EmptyLine { line: 1 }
        );
    }

    #[test]
    fn final_fragment_policy_is_explicit() {
        let mut strict = NdjsonDecoder::<Event>::new(limits()).unwrap();
        strict.push(b"{\"value\":1}").unwrap();
        assert_eq!(
            strict.finish().unwrap_err(),
            NdjsonDecodeError::UnterminatedFinalFragment { bytes: 11 }
        );

        let mut permissive_limits = limits();
        permissive_limits.final_fragment = FinalFragmentPolicy::AllowAtEof;
        let mut permissive = NdjsonDecoder::<Event>::new(permissive_limits).unwrap();
        permissive.push(b"{\"value\":1}").unwrap();
        assert_eq!(permissive.finish().unwrap(), vec![Event { value: 1 }]);
    }

    #[test]
    fn invalid_limits_are_rejected() {
        let mut invalid = limits();
        invalid.max_events = 0;
        assert!(matches!(
            NdjsonDecoder::<Event>::new(invalid),
            Err(NdjsonDecodeError::InvalidLimits)
        ));
    }
}
