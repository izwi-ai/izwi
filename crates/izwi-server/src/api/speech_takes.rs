//! Take editing (Fish Audio Drama 3, §8.4): what a speech record keeps so a
//! later take can continue it, and the `previous_take_ids` a request names.
//!
//! The trained format holds earlier takes as completed turns before the new
//! text (`FS-omni/generate.py:158-169`): each turn is a chunk's text and the
//! codes generated for it. A take's chunks become history the way a long
//! text's earlier chunks do (the chunk session's `FishDrama3History`): the
//! first take's first chunk stays pinned and the latest chunk follows, and
//! a prompt holds as many of the two as fit its rows. So a take keeps only
//! its first and last chunks ([`SpeechTake`]): the chunks between never
//! reach a later prompt.
//!
//! - Only a model whose prompt holds earlier turns
//!   (`continues_speech_across_chunks`) keeps takes; every other model's
//!   records are unchanged.
//! - A request names at most [`MAX_PREVIOUS_TAKES`] takes, oldest first:
//!   one is pinned and one is the latest, so a third could never reach the
//!   prompt.
//! - A previous take is a record of the request's own route that `GET`
//!   returns (records belong to no caller), of the same model, finished
//!   with its codes kept.

use izwi_core::{parse_tts_model_variant, ModelVariant, SpeechChunkTurn};

use crate::api::tts_long_form::{decode_chunk_turn, encode_chunk_turn};
use crate::batch_runtime::speech_progress::{
    SpeechTakeCodes, SpeechTakeTurnCodes, SPEECH_TAKE_CODES_VERSION,
};
use crate::batch_runtime::store::sha256_hex;
use crate::error::ApiError;
use crate::speech_history_store::{
    SpeechHistoryProcessingStatus, SpeechRouteKind, SpeechTakeInputs, SpeechTakeReference,
};
use crate::state::AppState;

/// The most takes one request continues.
pub(crate) const MAX_PREVIOUS_TAKES: usize = 2;

/// Whether `variant`'s records keep their takes for a later take to
/// continue: a model whose prompt holds earlier turns.
pub(crate) fn keeps_takes(variant: ModelVariant) -> bool {
    variant.continues_speech_across_chunks()
}

/// The chunks a take keeps: its first and its latest, with the count.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SpeechTake {
    chunks: usize,
    first: Option<(usize, SpeechChunkTurn)>,
    last: Option<(usize, SpeechChunkTurn)>,
}

impl SpeechTake {
    /// Records the take's next chunk; a chunk without frames is not history.
    pub(crate) fn push(&mut self, turn: SpeechChunkTurn) {
        let index = self.chunks;
        self.chunks += 1;
        if turn.codes.frames() == 0 {
            return;
        }
        if self.first.is_none() {
            self.first = Some((index, turn));
        } else {
            self.last = Some((index, turn));
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.first.is_none()
    }

    /// The kept chunks, oldest first.
    pub(crate) fn into_turns(self) -> Vec<SpeechChunkTurn> {
        self.first
            .into_iter()
            .chain(self.last)
            .map(|(_, turn)| turn)
            .collect()
    }

    /// The take as its record and a speech checkpoint store it.
    pub(crate) fn to_stored(&self) -> Result<SpeechTakeCodes, izwi_core::Error> {
        let turns = self
            .first
            .iter()
            .chain(&self.last)
            .map(|(chunk, turn)| {
                Ok(SpeechTakeTurnCodes {
                    chunk: *chunk,
                    turn: encode_chunk_turn(turn)?,
                })
            })
            .collect::<Result<Vec<_>, izwi_core::Error>>()?;
        Ok(SpeechTakeCodes {
            version: SPEECH_TAKE_CODES_VERSION,
            chunks: self.chunks,
            turns,
        })
    }

    /// A stored take, checked: its version, at most two kept chunks in
    /// order within the count, each of whole frames.
    pub(crate) fn from_stored(stored: &SpeechTakeCodes) -> Result<Self, izwi_core::Error> {
        let invalid =
            |reason: &str| izwi_core::Error::InvalidInput(format!("Invalid stored take: {reason}"));
        if stored.version != SPEECH_TAKE_CODES_VERSION {
            return Err(invalid("unknown version"));
        }
        if stored.turns.len() > 2
            || stored
                .turns
                .windows(2)
                .any(|pair| pair[0].chunk >= pair[1].chunk)
            || stored
                .turns
                .last()
                .is_some_and(|turn| turn.chunk >= stored.chunks)
        {
            return Err(invalid("its chunks are out of order"));
        }
        let mut take = Self {
            chunks: stored.chunks,
            ..Self::default()
        };
        for (index, kept) in stored.turns.iter().enumerate() {
            let turn = (kept.chunk, decode_chunk_turn(&kept.turn)?);
            if index == 0 {
                take.first = Some(turn);
            } else {
                take.last = Some(turn);
            }
        }
        Ok(take)
    }

    /// The take as its record's `generated_codes_json` holds it.
    pub(crate) fn to_json(&self) -> Result<String, izwi_core::Error> {
        serde_json::to_string(&self.to_stored()?)
            .map_err(|error| izwi_core::Error::InferenceError(error.to_string()))
    }

    fn from_json(json: &str) -> Result<Self, izwi_core::Error> {
        let stored = serde_json::from_str(json).map_err(|_| {
            izwi_core::Error::InvalidInput("Invalid stored take: not its JSON".into())
        })?;
        Self::from_stored(&stored)
    }
}

/// A take a request continues: its record, the hash of its stored codes
/// (for durable plan identities and idempotency digests) and its kept
/// chunks.
#[derive(Debug, Clone)]
pub(crate) struct PreviousTake {
    pub id: String,
    pub codes_sha256: String,
    pub turns: Vec<SpeechChunkTurn>,
}

/// The chunks of `takes`, oldest first: the history a request starts from.
pub(crate) fn previous_turns(takes: Vec<PreviousTake>) -> Vec<SpeechChunkTurn> {
    takes.into_iter().flat_map(|take| take.turns).collect()
}

/// Refuse `previous_take_ids` a request cannot carry, before any lookup: a
/// model that keeps no takes, more than [`MAX_PREVIOUS_TAKES`], an empty id
/// and a duplicate.
pub(crate) fn check_previous_take_ids(
    variant: ModelVariant,
    ids: &[String],
) -> Result<(), ApiError> {
    if ids.is_empty() {
        return Ok(());
    }
    if !keeps_takes(variant) {
        return Err(ApiError::bad_request(format!(
            "{variant} does not continue earlier takes: `previous_take_ids` is for models \
             whose prompt holds them (Fish Audio Drama 3)."
        )));
    }
    if ids.len() > MAX_PREVIOUS_TAKES {
        return Err(ApiError::bad_request(format!(
            "`previous_take_ids` names {} takes; a take continues at most {MAX_PREVIOUS_TAKES}.",
            ids.len()
        )));
    }
    for (index, id) in ids.iter().enumerate() {
        if id.trim().is_empty() {
            return Err(ApiError::bad_request(format!(
                "`previous_take_ids[{index}]` is empty."
            )));
        }
        if ids[..index].contains(id) {
            return Err(ApiError::bad_request(format!(
                "`previous_take_ids` names `{id}` twice."
            )));
        }
    }
    Ok(())
}

/// The takes `ids` names, read from `route_kind`'s records, each checked:
/// 404 for one that does not exist, 400 for another model's take and for a
/// take without kept codes (unfinished, failed, or made before takes kept
/// them). Every request and every stage attempt reads them afresh.
pub(crate) async fn load_previous_takes(
    state: &AppState,
    route_kind: SpeechRouteKind,
    variant: ModelVariant,
    ids: &[String],
) -> Result<Vec<PreviousTake>, ApiError> {
    check_previous_take_ids(variant, ids)?;
    let mut takes = Vec::with_capacity(ids.len());
    for id in ids {
        let stored = state
            .speech_history_store
            .get_take(route_kind, id)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?
            .ok_or_else(|| ApiError::not_found(format!("Previous take `{id}` not found.")))?;
        let model = stored
            .model_id
            .as_deref()
            .map(izwi_core::normalize_stored_model_id)
            .and_then(|model| parse_tts_model_variant(&model).ok());
        if model != Some(variant) {
            return Err(ApiError::bad_request(format!(
                "Previous take `{id}` was made by {}, not {variant}; a take continues only \
                 takes of its own model.",
                stored.model_id.as_deref().unwrap_or("an unknown model")
            )));
        }
        let codes = stored
            .generated_codes_json
            .filter(|_| stored.processing_status == SpeechHistoryProcessingStatus::Ready)
            .ok_or_else(|| {
                ApiError::bad_request(format!(
                    "Previous take `{id}` kept no codes to continue from: it is unfinished, \
                     failed, or was made before takes kept them."
                ))
            })?;
        let take = SpeechTake::from_json(&codes)
            .map_err(|error| ApiError::internal(format!("Previous take `{id}`: {error}")))?;
        takes.push(PreviousTake {
            id: id.clone(),
            codes_sha256: sha256_hex(codes.as_bytes()),
            turns: take.into_turns(),
        });
    }
    Ok(takes)
}

/// What a new take of `variant` keeps of its request: each reference's
/// speaker and saved voice (an inline clip is speaker-only), the seed and
/// the take it continues (the last of `previous_take_ids`). Nothing for a
/// model that keeps no takes.
pub(crate) fn take_inputs(
    variant: ModelVariant,
    req: &crate::api::speech_history::CreateSpeechHistoryRecordRequest,
) -> SpeechTakeInputs {
    if !keeps_takes(variant) {
        return SpeechTakeInputs::default();
    }
    let references = if !req.references.is_empty() {
        req.references
            .iter()
            .map(|reference| SpeechTakeReference {
                speaker: crate::api::speech_dialogue::reference_speaker(reference),
                saved_voice_id: reference.saved_voice_id.clone(),
            })
            .collect()
    } else if req.saved_voice_id.is_some() || req.reference_audio.is_some() {
        vec![SpeechTakeReference {
            speaker: 0,
            saved_voice_id: req.saved_voice_id.clone(),
        }]
    } else {
        Vec::new()
    };
    SpeechTakeInputs {
        references: Some(references),
        seed: req.sampling.seed,
        parent_take_id: req.previous_take_ids.last().cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::SpeechFrameCodes;

    fn turn(text: &str, frames: usize, base: u32) -> SpeechChunkTurn {
        SpeechChunkTurn {
            text: text.into(),
            codes: SpeechFrameCodes::new(
                (0..10)
                    .map(|k| (0..frames).map(|t| base + (k * 100 + t) as u32).collect())
                    .collect(),
            ),
        }
    }

    /// A take keeps its first and last chunks with their indices, and its
    /// stored form round-trips; one chunk is kept once.
    #[test]
    fn a_take_keeps_its_first_and_last_chunks() {
        let mut take = SpeechTake::default();
        assert!(take.is_empty());
        take.push(turn("one", 3, 1));
        let single = take.to_stored().unwrap();
        assert_eq!(single.chunks, 1);
        assert_eq!(single.turns.len(), 1);
        assert_eq!(single.turns[0].chunk, 0);
        take.push(turn("two", 2, 2));
        take.push(turn("three", 4, 3));
        let stored = take.to_stored().unwrap();
        assert_eq!(stored.version, 1);
        assert_eq!(stored.chunks, 3);
        assert_eq!(
            stored.turns.iter().map(|t| t.chunk).collect::<Vec<_>>(),
            [0, 2]
        );
        let json = take.to_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["turns"][1]["text"], "three");
        assert_eq!(value["turns"][1]["codebooks"], 10);
        let restored = SpeechTake::from_json(&json).unwrap();
        assert_eq!(restored, take);
        assert_eq!(
            restored.into_turns(),
            vec![turn("one", 3, 1), turn("three", 4, 3)]
        );
    }

    /// Stored takes are checked: an unknown version, chunks out of order or
    /// past the count, codes that do not fill whole frames.
    #[test]
    fn invalid_stored_takes_are_refused() {
        let mut take = SpeechTake::default();
        take.push(turn("one", 3, 1));
        take.push(turn("two", 2, 2));
        let stored = take.to_stored().unwrap();
        let refused = |change: &dyn Fn(&mut SpeechTakeCodes)| {
            let mut bad = stored.clone();
            change(&mut bad);
            SpeechTake::from_stored(&bad).unwrap_err().to_string()
        };
        assert!(refused(&|bad| bad.version = 2).contains("unknown version"));
        assert!(refused(&|bad| bad.turns.swap(0, 1)).contains("out of order"));
        assert!(refused(&|bad| bad.chunks = 1).contains("out of order"));
        assert!(refused(&|bad| bad.turns[1].turn.codes = "AAA=".into()).contains("whole frames"));
        assert!(SpeechTake::from_json("{").is_err());
    }

    /// Only models that keep takes take `previous_take_ids`; at most two,
    /// none empty or repeated.
    #[test]
    fn previous_take_ids_are_checked_before_any_lookup() {
        let drama3 = ModelVariant::FishAudioDrama3;
        let ids = |ids: &[&str]| ids.iter().map(|id| id.to_string()).collect::<Vec<_>>();
        assert!(check_previous_take_ids(drama3, &ids(&["a", "b"])).is_ok());
        assert!(check_previous_take_ids(ModelVariant::FishAudioS2Pro, &[]).is_ok());
        let message = |variant, list: &[&str]| {
            check_previous_take_ids(variant, &ids(list))
                .unwrap_err()
                .message
        };
        assert!(message(ModelVariant::FishAudioS2Pro, &["a"]).contains("does not continue"));
        assert!(message(drama3, &["a", "b", "c"]).contains("at most 2"));
        assert!(message(drama3, &["a", " "]).contains("`previous_take_ids[1]` is empty"));
        assert!(message(drama3, &["a", "a"]).contains("`a` twice"));
    }
}
