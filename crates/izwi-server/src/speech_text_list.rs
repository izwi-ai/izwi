//! Filters shared by the transcription and diarization history lists.

/// Optional list filters; the default lists every record.
#[derive(Debug, Clone, Default)]
pub struct SpeechTextListFilter {
    /// Case-insensitive match on the transcript text or the audio filename.
    pub search: Option<String>,
    /// Processing statuses to keep (pending, processing, ready, failed);
    /// empty keeps all.
    pub statuses: Vec<String>,
    /// Only records that identify speakers (speaker-attributed transcripts
    /// and diarizations).
    pub with_speakers: bool,
}

pub const PROCESSING_STATUSES: [&str; 4] = ["pending", "processing", "ready", "failed"];

impl SpeechTextListFilter {
    /// LIKE pattern for `search`, or `None` when there is no search.
    pub fn search_pattern(&self) -> Option<String> {
        self.search
            .as_deref()
            .map(str::trim)
            .filter(|search| !search.is_empty())
            .map(crate::db::raw::like_contains_pattern)
    }

    /// `,status,…,` for matching with `instr`, or `None` for all statuses.
    pub fn status_list(&self) -> Option<String> {
        (!self.statuses.is_empty()).then(|| format!(",{},", self.statuses.join(",")))
    }
}
