//! Long text as a chunk session (§3.7, §7.5, `inference.py:912-1043`).
//!
//! The text is cut by [`plan_chunks`]; each chunk is its own generation
//! with its own frame budget, over a prompt holding every reference, the
//! pinned first chunk and the latest chunk with the codes generated for
//! them. When that prompt is too long, history goes oldest first, then the
//! pinned chunk; references always stay. A chunk joins the history only
//! when it produced frames, as upstream's loop does. The engine (Phase 2)
//! runs one request per chunk through [`FishDrama3ChunkSession::plan_next`]
//! and [`FishDrama3ChunkSession::complete`]; [`FishDrama3Model::run_chunk_session`]
//! does the same in process.

use std::collections::VecDeque;

use super::super::chunking::{plan_chunks, FishDrama3History, HISTORY_RECENT_TURNS};
use super::super::prompt::{
    build_chunk_prompt, FishDrama3Codes, FishDrama3HistoryTurn, FishDrama3Prompt,
    FishDrama3Reference, MAX_PROMPT_ROWS,
};
use super::super::sampling::FishDrama3SamplingParams;
use super::super::tokenizer::FishDrama3Tokenizer;
use super::budget::frame_budget;
use super::{FishDrama3CancelCheck, FishDrama3GenerationParams, FishDrama3Model};
use crate::error::{Error, Result};

/// How a session runs its chunks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3ChunkSessionOptions {
    /// Chunk `i` samples with `seed.wrapping_add(i)`, so chunks draw
    /// different noise while each stays reproducible on its own.
    pub sampling: FishDrama3SamplingParams,
    /// Each chunk's explicit budget; `None` is the automatic one per chunk.
    pub max_tokens: Option<usize>,
    pub mask_first_eos: bool,
    /// Generate the whole text as one chunk.
    pub single_chunk: bool,
    /// The prompt row limit history selection works to.
    pub max_prompt_rows: usize,
}

impl FishDrama3ChunkSessionOptions {
    pub fn new(sampling: FishDrama3SamplingParams) -> Self {
        Self {
            sampling,
            max_tokens: None,
            mask_first_eos: true,
            single_chunk: false,
            max_prompt_rows: MAX_PROMPT_ROWS,
        }
    }
}

/// The next chunk's generation: its prompt with the history that fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FishDrama3ChunkPlan {
    pub index: usize,
    pub text: String,
    pub prompt: FishDrama3Prompt,
    /// History turns the prompt holds, oldest first (chunk indices).
    pub history: Vec<usize>,
    pub params: FishDrama3GenerationParams,
}

/// A generated chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3ChunkResult {
    pub index: usize,
    pub codes: FishDrama3Codes,
    pub prompt_rows: usize,
    pub history: Vec<usize>,
}

/// One long-text request's chunks and history.
#[derive(Debug, Clone)]
pub struct FishDrama3ChunkSession {
    references: Vec<FishDrama3Reference>,
    chunks: Vec<String>,
    next: usize,
    history: FishDrama3History,
    /// The chunks behind `history`'s turns, in its order (pinned first),
    /// kept by the same policy.
    pinned_chunk: Option<usize>,
    recent_chunks: VecDeque<usize>,
    options: FishDrama3ChunkSessionOptions,
}

impl FishDrama3ChunkSession {
    pub fn new(
        text: &str,
        references: Vec<FishDrama3Reference>,
        options: FishDrama3ChunkSessionOptions,
    ) -> Result<Self> {
        options.sampling.validate()?;
        frame_budget(text, options.max_tokens)?;
        if options.max_prompt_rows == 0 || options.max_prompt_rows > MAX_PROMPT_ROWS {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 prompts hold 1 to {MAX_PROMPT_ROWS} rows; got {}",
                options.max_prompt_rows
            )));
        }
        Ok(Self {
            references,
            chunks: plan_chunks(text, options.single_chunk),
            next: 0,
            history: FishDrama3History::new(),
            pinned_chunk: None,
            recent_chunks: VecDeque::new(),
            options,
        })
    }

    pub fn chunks(&self) -> &[String] {
        &self.chunks
    }

    pub fn history(&self) -> &FishDrama3History {
        &self.history
    }

    pub fn is_done(&self) -> bool {
        self.next == self.chunks.len()
    }

    /// The next chunk's plan, or `None` when every chunk is done. The
    /// prompt keeps as much history as fits both the row limit and
    /// `positions` beside the chunk's budget; when nothing fits it is
    /// [`Error::ContextLengthExceeded`].
    pub fn plan_next(
        &self,
        tokenizer: &FishDrama3Tokenizer,
        positions: usize,
    ) -> Result<Option<FishDrama3ChunkPlan>> {
        let Some(text) = self.chunks.get(self.next) else {
            return Ok(None);
        };
        let max_frames = frame_budget(text, self.options.max_tokens)?;
        let max_rows = self
            .options
            .max_prompt_rows
            .min(positions.saturating_sub(max_frames));
        let chunk = build_chunk_prompt(tokenizer, &self.references, &self.history, text, max_rows)?;
        let history = self.chunk_indices(&chunk.history)?;
        let mut sampling = self.options.sampling;
        // Chunk `i` draws from `seed + i`, wrapping at `u64::MAX`.
        sampling.seed = sampling.seed.wrapping_add(self.next as u64);
        Ok(Some(FishDrama3ChunkPlan {
            index: self.next,
            text: text.clone(),
            prompt: chunk.prompt,
            history,
            params: FishDrama3GenerationParams {
                sampling,
                max_frames,
                mask_first_eos: self.options.mask_first_eos,
            },
        }))
    }

    /// The chunk index of each selected turn (by identity in `history`).
    fn chunk_indices(&self, selected: &[&FishDrama3HistoryTurn]) -> Result<Vec<usize>> {
        let all = self
            .history
            .candidates()
            .into_iter()
            .next()
            .unwrap_or_default();
        let chunks = self
            .pinned_chunk
            .iter()
            .chain(&self.recent_chunks)
            .copied()
            .collect::<Vec<_>>();
        if all.len() != chunks.len() {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 history and its chunk list disagree".into(),
            ));
        }
        selected
            .iter()
            .map(|turn| {
                all.iter()
                    .position(|held| std::ptr::eq(*held, *turn))
                    .map(|position| chunks[position])
                    .ok_or_else(|| {
                        Error::InferenceError("Fish Audio Drama 3 history lost a turn".into())
                    })
            })
            .collect()
    }

    /// Records chunk `plan.index`'s codes; it joins the history only when
    /// it made frames.
    pub fn complete(
        &mut self,
        plan: &FishDrama3ChunkPlan,
        codes: &FishDrama3Codes,
    ) -> Result<FishDrama3ChunkResult> {
        if plan.index != self.next {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 chunk {} completed while chunk {} is next",
                plan.index, self.next
            )));
        }
        if codes.frames() > 0 {
            self.history.push(FishDrama3HistoryTurn {
                text: plan.text.clone(),
                codes: codes.clone(),
            });
            if self.pinned_chunk.is_none() {
                self.pinned_chunk = Some(plan.index);
            } else {
                if self.recent_chunks.len() == HISTORY_RECENT_TURNS {
                    self.recent_chunks.pop_front();
                }
                self.recent_chunks.push_back(plan.index);
            }
        }
        self.next += 1;
        Ok(FishDrama3ChunkResult {
            index: plan.index,
            codes: codes.clone(),
            prompt_rows: plan.prompt.rows,
            history: plan.history.clone(),
        })
    }

    /// Runs every remaining chunk through `generate`, polling `check`
    /// between chunks.
    pub fn run_with(
        &mut self,
        tokenizer: &FishDrama3Tokenizer,
        positions: usize,
        check: FishDrama3CancelCheck<'_>,
        mut generate: impl FnMut(&FishDrama3ChunkPlan) -> Result<FishDrama3Codes>,
    ) -> Result<Vec<FishDrama3ChunkResult>> {
        let mut results = Vec::with_capacity(self.chunks.len() - self.next);
        while let Some(plan) = self.plan_next(tokenizer, positions)? {
            check()?;
            let codes = generate(&plan)?;
            results.push(self.complete(&plan, &codes)?);
        }
        Ok(results)
    }
}

impl FishDrama3Model {
    /// Runs `session`'s remaining chunks, each as its own generation over
    /// private caches.
    pub fn run_chunk_session(
        &self,
        session: &mut FishDrama3ChunkSession,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<FishDrama3ChunkResult>> {
        session.run_with(&self.tokenizer, self.max_positions(), check, |plan| {
            self.generate_frames(plan.prompt.clone(), plan.params, check)
        })
    }
}
