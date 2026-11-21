//! Exact incremental decoding. Histories are immutable, independently allocated
//! tensors: cloning a checkpoint shares storage until the next successful push.
//!
//! On CUDA ([`Audio8CodecCalls`]) every push runs in a call of a fixed number
//! of slots with uniform shapes: histories start as zeros at their full
//! length (zero history is exactly causal padding), the attention cache is
//! always `window - 1` keys with each row's own valid count masked, and each
//! row rotates at its own positions. A push of `incoming` frames then has
//! the same shapes at every stream age and in every call, so cuBLAS and
//! cuDNN pick the same kernels and a row's PCM never depends on its peers.
use super::*;
use std::sync::Arc;

type AttentionState = (Tensor, Tensor);

/// Rows every CUDA codec call runs unless [`AUDIO8_CODEC_CALL_SLOTS_ENV`]
/// says otherwise. A quantum gathers about twenty rows under load: eight
/// slots run them in three calls a chunk with little padding, while one
/// stream alone pays for seven spare slots.
pub(crate) const AUDIO8_CODEC_CALL_SLOTS: usize = 8;
pub(crate) const AUDIO8_CODEC_CALL_SLOTS_ENV: &str = "IZWI_AUDIO8_CODEC_CALL_SLOTS";
const MAX_CODEC_CALL_SLOTS: usize = 64;

/// How streaming codec pushes run on CUDA: every call has `slots` rows
/// (spare rows decode code-0 frames from zero history and are dropped);
/// `shared`, rows of different streams fill one call, otherwise each row
/// runs in a call of its own (still `slots` wide, so still row-invariant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Audio8CodecCalls {
    pub(crate) slots: usize,
    pub(crate) shared: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Audio8DacStreamState {
    histories: Vec<Tensor>,
    attention: Vec<AttentionState>,
    /// After a batched push: this row's histories and attention are lane
    /// `.1` of the batch's tensors (and the two vectors above are empty), so
    /// the rows that decoded together keep one `[rows, ...]` copy instead of
    /// splitting it per row, and hand it back whole if they push together
    /// again in the same order.
    shared: Option<(Arc<Audio8DacBatchedHistory>, usize)>,
    frames: usize,
    samples: usize,
    finished: bool,
}

/// The retained state a codec batch produced: every history and attention
/// tensor with one row per lane.
#[derive(Debug)]
struct Audio8DacBatchedHistory {
    histories: Vec<Tensor>,
    attention: Vec<AttentionState>,
    rows: usize,
}

impl Audio8DacStreamState {
    /// This row's histories and attention as its own `[1, ...]` tensors
    /// (views of a shared batch's lane; no copy).
    fn row_tensors(&self) -> Result<(Vec<Tensor>, Vec<AttentionState>)> {
        match &self.shared {
            None => Ok((self.histories.clone(), self.attention.clone())),
            Some((batch, lane)) => Ok((
                batch
                    .histories
                    .iter()
                    .map(|history| history.narrow(0, *lane, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?,
                batch
                    .attention
                    .iter()
                    .map(|(k, v)| Ok((k.narrow(0, *lane, 1)?, v.narrow(0, *lane, 1)?)))
                    .collect::<Result<Vec<_>>>()?,
            )),
        }
    }

    /// This row as its own tensors (lane views of a shared batch).
    fn unshared(&self) -> Result<Self> {
        let (histories, attention) = self.row_tensors()?;
        Ok(Self {
            histories,
            attention,
            shared: None,
            frames: self.frames,
            samples: self.samples,
            finished: self.finished,
        })
    }

    fn history_counts(&self) -> (usize, usize) {
        match &self.shared {
            None => (self.histories.len(), self.attention.len()),
            Some((batch, _)) => (batch.histories.len(), batch.attention.len()),
        }
    }

    pub(crate) fn decoded_frames(&self) -> usize {
        self.frames
    }
    pub(crate) fn emitted_samples(&self) -> usize {
        self.samples
    }
    /// This row's retained bytes; a shared batch's are split evenly between
    /// its lanes (each lane holds the same shapes).
    pub(crate) fn retained_tensor_bytes(&self) -> u64 {
        let (histories, attention, rows) = match &self.shared {
            None => (&self.histories, &self.attention, 1),
            Some((batch, _)) => (&batch.histories, &batch.attention, batch.rows.max(1)),
        };
        histories
            .iter()
            .chain(attention.iter().flat_map(|(k, v)| [k, v]))
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum::<u64>()
            / rows as u64
    }
    /// The full oracle discards the causal transpose tail. All stable samples
    /// have already been emitted; repeated terminal flushes therefore are empty.
    pub(crate) fn flush(&mut self) -> Vec<f32> {
        self.finished = true;
        self.histories.clear();
        self.attention.clear();
        self.shared = None;
        Vec::new()
    }
}

impl Audio8DacConfig {
    /// F32 retained history at batch one, independent of utterance length.
    /// Account two copies when retaining a transactional rollback checkpoint.
    pub(crate) fn streaming_history_bound_bytes(&self) -> Result<u64> {
        let mut elements = (self.post_transformer_layers as u128)
            * 2
            * self.post_transformer_kv_heads as u128
            * self.transformer_head_dim as u128
            * self.transformer_window_size.saturating_sub(1) as u128;
        elements += self.downsample_factors.len() as u128 * self.latent_dim as u128 * 6;
        elements += self.latent_dim as u128 * 6;
        for idx in 0..self.decoder_rates.len() {
            let input = self.decoder_dim / (1usize << idx);
            let output = self.decoder_dim / (1usize << (idx + 1));
            elements += input as u128; // kernel=2*stride: one preceding input
            elements += output as u128 * (6 + 18 + 54);
        }
        elements += (self.decoder_dim / (1usize << self.decoder_rates.len())) as u128 * 6;
        u64::try_from(elements * 4)
            .map_err(|_| Error::ConfigError("Audio8 codec stream history overflow".into()))
    }
}

impl Audio8DacDecoder {
    /// Atomically consume new codebook frames. An error leaves `state` unchanged.
    pub(crate) fn push_frames(
        &self,
        state: &mut Audio8DacStreamState,
        codebooks: &[Vec<u32>],
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Vec<f32>> {
        if state.finished {
            return Err(Error::InvalidInput(
                "Audio8 codec stream is already flushed".into(),
            ));
        }
        if codebooks.len() == self.config.num_codebooks() && codebooks.iter().all(Vec::is_empty) {
            check_cancelled()?;
            return Ok(Vec::new());
        }
        check_cancelled()?;
        if self.calls.is_some() {
            // The same fixed-slot call a batched push runs, so a stream's
            // bits do not depend on which entry point decoded it.
            let mut states = [state.clone()];
            let samples = self
                .push_frames_batch(&mut states, &[codebooks.to_vec()], check_cancelled)?
                .pop()
                .unwrap_or_default();
            let [next] = states;
            *state = next;
            return Ok(samples);
        }
        let incoming = codebooks.first().map(Vec::len).unwrap_or(0);
        if state
            .frames
            .checked_add(incoming)
            .is_none_or(|frames| frames > Audio8DacConfig::MAX_QUANTIZER_FRAMES)
        {
            return Err(Error::InvalidInput(
                "Audio8 codec stream exceeds its absolute position capacity".into(),
            ));
        }
        let codes = codebooks_to_tensor(codebooks, &self.config, self.decoder_device()?)?;
        let mut candidate = state.unshared()?;
        let z = self.push_tensor(&mut candidate, &codes, check_cancelled, None)?;
        let samples = z.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        check_cancelled()?;
        *state = candidate;
        Ok(samples)
    }

    /// How this decoder's streaming calls run (`None`: age-bucketed batches).
    pub(crate) fn codec_calls(&self) -> Option<Audio8CodecCalls> {
        self.calls
    }

    /// Sets the streaming call policy a load runs: on CUDA, fixed-slot calls
    /// ([`AUDIO8_CODEC_CALL_SLOTS`], or [`AUDIO8_CODEC_CALL_SLOTS_ENV`]),
    /// shared across streams once [`Self::check_shared_calls`] shows rows of
    /// a call come out as each does alone; elsewhere the age-bucketed path.
    pub(crate) fn resolve_codec_calls(&mut self) -> Result<Option<Audio8CodecCalls>> {
        if !self.decoder_device()?.is_cuda() {
            self.calls = None;
            return Ok(None);
        }
        let slots = match crate::env::value(AUDIO8_CODEC_CALL_SLOTS_ENV) {
            None => AUDIO8_CODEC_CALL_SLOTS,
            Some(raw) => raw
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|slots| (1..=MAX_CODEC_CALL_SLOTS).contains(slots))
                .ok_or_else(|| {
                    Error::ConfigError(format!(
                        "{AUDIO8_CODEC_CALL_SLOTS_ENV} must be 1..={MAX_CODEC_CALL_SLOTS}, got {raw:?}"
                    ))
                })?,
        };
        let shared = match self.check_shared_calls(slots) {
            Ok(true) => {
                tracing::info!(slots, "Audio8 codec calls are shared across requests");
                true
            }
            Ok(false) => {
                tracing::warn!(
                    slots,
                    "Audio8 codec rows decode together differently from alone: each request's \
                     chunks keep calls of their own"
                );
                false
            }
            Err(error) => {
                tracing::warn!(
                    slots,
                    %error,
                    "Audio8 codec shared-call check failed: each request's chunks keep calls of \
                     their own"
                );
                false
            }
        };
        self.calls = Some(Audio8CodecCalls { slots, shared });
        Ok(self.calls)
    }

    /// Sets the fixed-slot path on any device, so CPU tests exercise what
    /// CUDA runs.
    #[cfg(test)]
    pub(crate) fn set_codec_calls(&mut self, calls: Option<Audio8CodecCalls>) {
        self.calls = calls;
    }

    /// Whether streams pushed together in one call of `slots` come out bit
    /// for bit as each pushed alone in such a call (spare rows zero), on
    /// this decoder's device: `slots` streams of xorshift codes, a first
    /// chunk and then a full chunk, decoded together and one at a time.
    pub(crate) fn check_shared_calls(&self, slots: usize) -> Result<bool> {
        let slots = slots.max(1);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut chunk = |frames: usize| {
            (0..slots)
                .map(|_| {
                    (0..self.config.num_codebooks())
                        .map(|book| {
                            let size = if book == 0 {
                                self.config.semantic_codebook_size
                            } else {
                                self.config.residual_codebook_size
                            }
                            .max(1) as u64;
                            (0..frames)
                                .map(|_| {
                                    seed ^= seed << 13;
                                    seed ^= seed >> 7;
                                    seed ^= seed << 17;
                                    (seed % size) as u32
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let bits = |rows: &[Vec<f32>]| {
            rows.iter()
                .map(|row| row.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let mut together = vec![Audio8DacStreamState::default(); slots];
        let mut alone = together.clone();
        for frames in [
            crate::models::architectures::audio8::AUDIO8_FIRST_AUDIO_FRAMES,
            crate::models::architectures::audio8::AUDIO8_AUDIO_CHUNK_FRAMES,
        ] {
            let codes = chunk(frames);
            let shared = self.push_frames_uniform(
                &mut together,
                &codes,
                Audio8CodecCalls {
                    slots,
                    shared: true,
                },
                &|| Ok(()),
            )?;
            let apart = self.push_frames_uniform(
                &mut alone,
                &codes,
                Audio8CodecCalls {
                    slots,
                    shared: false,
                },
                &|| Ok(()),
            )?;
            if bits(&shared) != bits(&apart) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Exact shape buckets share all codec projections/convolutions/attention.
    /// Different stream ages remain separate buckets, never concatenated in time.
    /// No candidate history is published unless every group and readback succeeds.
    /// A group keeps its retained state as one batch the rows share by lane:
    /// when the same rows push together again in the same order, the batch is
    /// used as it is, instead of concatenating every row's history and
    /// attention tensors and splitting them again after the push (one copy
    /// of each per row per push).
    ///
    /// With [`Audio8CodecCalls`] (CUDA) rows group by incoming frames alone
    /// and run in fixed-slot calls ([`Self::push_frames_uniform`]).
    pub(crate) fn push_frames_batch(
        &self,
        states: &mut [Audio8DacStreamState],
        codebooks: &[Vec<Vec<u32>>],
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        if let Some(calls) = self.calls {
            return self.push_frames_uniform(states, codebooks, calls, check_cancelled);
        }
        let (groups, tensors) =
            self.batch_rows(states, codebooks, check_cancelled, |state| state.frames)?;
        let mut candidates = states.to_vec();
        let mut outputs = vec![Vec::new(); states.len()];
        for rows in groups.values() {
            let first = &states[rows[0]];
            for &row in rows {
                if states[row].history_counts() != first.history_counts()
                    || states[row].samples != first.samples
                {
                    return Err(Error::InvalidInput(
                        "Audio8 codec batch history metadata differs".into(),
                    ));
                }
            }
            let mut combined = Audio8DacStreamState {
                shared: None,
                ..first.clone()
            };
            match shared_batch(states, rows) {
                // The rows pushed together last time, in this order: their
                // batch is this push's input as it is.
                Some(batch) => {
                    combined.histories = batch.histories.clone();
                    combined.attention = batch.attention.clone();
                }
                None => {
                    let row_tensors = rows
                        .iter()
                        .map(|&row| states[row].row_tensors())
                        .collect::<Result<Vec<_>>>()?;
                    combined.histories = (0..row_tensors[0].0.len())
                        .map(|history| {
                            let tensors = row_tensors
                                .iter()
                                .map(|(histories, _)| &histories[history])
                                .collect::<Vec<_>>();
                            Tensor::cat(&tensors, 0).map_err(Error::from)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    combined.attention = (0..row_tensors[0].1.len())
                        .map(|layer| {
                            let k = row_tensors
                                .iter()
                                .map(|(_, attention)| &attention[layer].0)
                                .collect::<Vec<_>>();
                            let v = row_tensors
                                .iter()
                                .map(|(_, attention)| &attention[layer].1)
                                .collect::<Vec<_>>();
                            Ok((Tensor::cat(&k, 0)?, Tensor::cat(&v, 0)?))
                        })
                        .collect::<Result<Vec<_>>>()?;
                }
            }
            let codes = Tensor::cat(
                &rows.iter().map(|&row| &tensors[row]).collect::<Vec<_>>(),
                0,
            )?;
            tracing::debug!(
                stage = "audio8_codec",
                native_rows = rows.len(),
                history_frames = first.frames,
                incoming_frames = codes.dim(2)?,
                "Executing compatible codec tensor batch"
            );
            let audio = self
                .push_tensor(&mut combined, &codes, check_cancelled, None)?
                .to_dtype(DType::F32)?;
            let batch = Arc::new(Audio8DacBatchedHistory {
                histories: std::mem::take(&mut combined.histories),
                attention: std::mem::take(&mut combined.attention),
                rows: rows.len(),
            });
            for (lane, &row) in rows.iter().enumerate() {
                outputs[row] = audio.narrow(0, lane, 1)?.flatten_all()?.to_vec1::<f32>()?;
                candidates[row] = Audio8DacStreamState {
                    histories: Vec::new(),
                    attention: Vec::new(),
                    shared: Some((batch.clone(), lane)),
                    frames: combined.frames,
                    samples: combined.samples,
                    finished: false,
                };
            }
        }
        check_cancelled()?;
        states.clone_from_slice(&candidates);
        Ok(outputs)
    }

    /// Validates a batch's rows and their code tensors, grouped by
    /// `(age(state), incoming frames)` in row order.
    #[allow(clippy::type_complexity)]
    fn batch_rows(
        &self,
        states: &[Audio8DacStreamState],
        codebooks: &[Vec<Vec<u32>>],
        check_cancelled: CancelCheck<'_>,
        age: impl Fn(&Audio8DacStreamState) -> usize,
    ) -> Result<(
        std::collections::BTreeMap<(usize, usize), Vec<usize>>,
        Vec<Tensor>,
    )> {
        if states.is_empty() || states.len() != codebooks.len() {
            return Err(Error::InvalidInput(
                "Audio8 codec batch rows do not match".into(),
            ));
        }
        let mut groups = std::collections::BTreeMap::<(usize, usize), Vec<usize>>::new();
        let mut tensors = Vec::with_capacity(states.len());
        for (row, (state, codes)) in states.iter().zip(codebooks).enumerate() {
            check_cancelled()?;
            let incoming = codes.first().map(Vec::len).unwrap_or(0);
            if state.finished
                || incoming == 0
                || state
                    .frames
                    .checked_add(incoming)
                    .is_none_or(|n| n > Audio8DacConfig::MAX_QUANTIZER_FRAMES)
            {
                return Err(Error::InvalidInput(
                    "Audio8 codec batch contains flushed or invalid-length row".into(),
                ));
            }
            tensors.push(codebooks_to_tensor(
                codes,
                &self.config,
                self.decoder_device()?,
            )?);
            groups.entry((age(state), incoming)).or_default().push(row);
        }
        Ok((groups, tensors))
    }

    /// Fixed-slot pushes: rows group by incoming frames only, whatever their
    /// age, and run `calls.slots` at a time (one at a time unless
    /// `calls.shared`), each call padded to `calls.slots` rows. No candidate
    /// state is published unless every call and readback succeeds.
    fn push_frames_uniform(
        &self,
        states: &mut [Audio8DacStreamState],
        codebooks: &[Vec<Vec<u32>>],
        calls: Audio8CodecCalls,
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        let (groups, tensors) = self.batch_rows(states, codebooks, check_cancelled, |_| 0)?;
        let slots = calls.slots.max(1);
        let per_call = if calls.shared { slots } else { 1 };
        let mut candidates = states.to_vec();
        let mut outputs = vec![Vec::new(); states.len()];
        for (&(_, incoming), rows) in &groups {
            for call in rows.chunks(per_call) {
                self.push_call(
                    states,
                    call,
                    &tensors,
                    incoming,
                    slots,
                    &mut candidates,
                    &mut outputs,
                    check_cancelled,
                )?;
            }
        }
        check_cancelled()?;
        states.clone_from_slice(&candidates);
        Ok(outputs)
    }

    /// One call of `slots` rows: `rows` (all pushing `incoming` frames) and
    /// spare rows of code-0 frames from zero history. Writes each row's PCM
    /// and candidate state; spare rows' outputs and histories are dropped.
    #[allow(clippy::too_many_arguments)]
    fn push_call(
        &self,
        states: &[Audio8DacStreamState],
        rows: &[usize],
        codes: &[Tensor],
        incoming: usize,
        slots: usize,
        candidates: &mut [Audio8DacStreamState],
        outputs: &mut [Vec<f32>],
        check_cancelled: CancelCheck<'_>,
    ) -> Result<()> {
        let spare = slots.checked_sub(rows.len()).ok_or_else(|| {
            Error::InvalidInput("Audio8 codec call has more rows than slots".into())
        })?;
        let fresh = |state: &Audio8DacStreamState| {
            state.shared.is_none() && state.histories.is_empty() && state.attention.is_empty()
        };
        let mut combined = Audio8DacStreamState::default();
        // A call of fresh streams starts from zero history inside the push.
        if let Some(template) = rows.iter().find(|&&row| !fresh(&states[row])) {
            let counts = states[*template].history_counts();
            if rows
                .iter()
                .any(|&row| !fresh(&states[row]) && states[row].history_counts() != counts)
            {
                return Err(Error::InvalidInput(
                    "Audio8 codec call history metadata differs".into(),
                ));
            }
            let (template_histories, template_attention) = states[*template].row_tensors()?;
            // Each history and attention tensor as its parts along the batch.
            let (mut histories, mut attention) = match shared_batch(states, rows) {
                Some(batch) => (
                    batch
                        .histories
                        .iter()
                        .map(|history| vec![history.clone()])
                        .collect::<Vec<_>>(),
                    batch
                        .attention
                        .iter()
                        .map(|(k, v)| (vec![k.clone()], vec![v.clone()]))
                        .collect::<Vec<_>>(),
                ),
                None => {
                    let mut histories = vec![Vec::new(); template_histories.len()];
                    let mut attention = vec![(Vec::new(), Vec::new()); template_attention.len()];
                    for &row in rows {
                        let (row_histories, row_attention) = if fresh(&states[row]) {
                            (
                                template_histories
                                    .iter()
                                    .map(|history| zero_rows(history, 1))
                                    .collect::<Result<Vec<_>>>()?,
                                template_attention
                                    .iter()
                                    .map(|(k, v)| Ok((zero_rows(k, 1)?, zero_rows(v, 1)?)))
                                    .collect::<Result<Vec<_>>>()?,
                            )
                        } else {
                            states[row].row_tensors()?
                        };
                        for (parts, history) in histories.iter_mut().zip(row_histories) {
                            parts.push(history);
                        }
                        for ((keys, values), (k, v)) in attention.iter_mut().zip(row_attention) {
                            keys.push(k);
                            values.push(v);
                        }
                    }
                    (histories, attention)
                }
            };
            if spare > 0 {
                for (parts, template) in histories.iter_mut().zip(&template_histories) {
                    parts.push(zero_rows(template, spare)?);
                }
                for ((keys, values), (k, v)) in attention.iter_mut().zip(&template_attention) {
                    keys.push(zero_rows(k, spare)?);
                    values.push(zero_rows(v, spare)?);
                }
            }
            let join = |parts: &[Tensor]| -> Result<Tensor> {
                match parts {
                    [one] => Ok(one.clone()),
                    parts => Tensor::cat(parts, 0).map_err(Error::from),
                }
            };
            combined.histories = histories
                .iter()
                .map(|parts| join(parts))
                .collect::<Result<Vec<_>>>()?;
            combined.attention = attention
                .iter()
                .map(|(keys, values)| Ok((join(keys)?, join(values)?)))
                .collect::<Result<Vec<_>>>()?;
        }
        let mut code_parts = rows
            .iter()
            .map(|&row| codes[row].clone())
            .collect::<Vec<_>>();
        if spare > 0 {
            let first = &code_parts[0];
            code_parts.push(Tensor::zeros(
                (spare, first.dim(1)?, incoming),
                first.dtype(),
                first.device(),
            )?);
        }
        let code_tensor = Tensor::cat(&code_parts, 0)?;
        let starts = rows
            .iter()
            .map(|&row| states[row].frames)
            .chain(std::iter::repeat_n(0, spare))
            .collect::<Vec<_>>();
        tracing::debug!(
            stage = "audio8_codec",
            native_rows = rows.len(),
            slots,
            incoming_frames = incoming,
            "Executing fixed-slot codec call"
        );
        let audio = self
            .push_tensor(&mut combined, &code_tensor, check_cancelled, Some(&starts))?
            .to_dtype(DType::F32)?;
        let samples = audio.dim(2)?;
        let pcm = audio
            .narrow(0, 0, rows.len())?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Retain only the real rows (a copy: a narrow keeps the whole call).
        let keep = |tensor: Tensor| -> Result<Tensor> {
            if spare == 0 {
                Ok(tensor)
            } else {
                Ok(tensor.narrow(0, 0, rows.len())?.copy()?)
            }
        };
        let batch = Arc::new(Audio8DacBatchedHistory {
            histories: std::mem::take(&mut combined.histories)
                .into_iter()
                .map(keep)
                .collect::<Result<Vec<_>>>()?,
            attention: std::mem::take(&mut combined.attention)
                .into_iter()
                .map(|(k, v)| Ok((keep(k)?, keep(v)?)))
                .collect::<Result<Vec<_>>>()?,
            rows: rows.len(),
        });
        for (lane, &row) in rows.iter().enumerate() {
            outputs[row] = pcm[lane * samples..(lane + 1) * samples].to_vec();
            candidates[row] = Audio8DacStreamState {
                histories: Vec::new(),
                attention: Vec::new(),
                shared: Some((batch.clone(), lane)),
                frames: states[row].frames + incoming,
                samples: states[row].samples + samples,
                finished: false,
            };
        }
        Ok(())
    }

    /// `starts`: each row's absolute first frame, for fixed-slot calls
    /// (zero-filled histories, per-row positions and masks); `None` runs the
    /// age-bucketed path at `state.frames`.
    fn push_tensor(
        &self,
        state: &mut Audio8DacStreamState,
        codes: &Tensor,
        check_cancelled: CancelCheck<'_>,
        starts: Option<&[usize]>,
    ) -> Result<Tensor> {
        let frames = codes.dim(2)?;
        let zero_fill = starts.is_some();
        let mut next = state.clone();
        let q = &self.quantizer;
        let semantic = q
            .semantic_quantizer
            .decode_codes(&codes.narrow(1, 0, 1)?, check_cancelled)?;
        let residual = q.residual_quantizer.decode_codes(
            &codes.narrow(1, 1, self.config.residual_codebooks)?,
            check_cancelled,
        )?;
        let mut z = transformer(
            &q.post_module,
            &semantic.broadcast_add(&residual)?,
            &mut next,
            check_cancelled,
            starts,
        )?;
        let mut cursor = 0;
        for block in &q.upsample {
            check_cancelled()?;
            z = transpose(
                &block.transposed,
                &z,
                &mut next.histories,
                &mut cursor,
                zero_fill,
            )?;
            let residual = z.clone();
            let b = &block.convnext;
            let mut h = conv(&b.dwconv, &z, &mut next.histories, &mut cursor, zero_fill)?
                .transpose(1, 2)?;
            h = b
                .pwconv2
                .forward(&b.pwconv1.forward(&b.norm.forward(&h)?)?.gelu_erf()?)?;
            if let Some(gamma) = &b.gamma {
                h = h.broadcast_mul(&gamma.reshape((1, 1, gamma.dim(0)?))?)?;
            }
            z = residual.broadcast_add(&h.transpose(1, 2)?)?;
        }
        let d = &self.decoder;
        z = conv(&d.first, &z, &mut next.histories, &mut cursor, zero_fill)?;
        for block in &d.blocks {
            check_cancelled()?;
            z = transpose(
                &block.transposed,
                &block.snake.forward(&z)?,
                &mut next.histories,
                &mut cursor,
                zero_fill,
            )?;
            for unit in &block.residuals {
                check_cancelled()?;
                let h = conv(
                    &unit.conv1,
                    &unit.snake1.forward(&z)?,
                    &mut next.histories,
                    &mut cursor,
                    zero_fill,
                )?;
                let h = conv(
                    &unit.conv2,
                    &unit.snake2.forward(&h)?,
                    &mut next.histories,
                    &mut cursor,
                    zero_fill,
                )?;
                z = z.broadcast_add(&h)?;
            }
        }
        z = conv(
            &d.final_conv,
            &d.final_snake.forward(&z)?,
            &mut next.histories,
            &mut cursor,
            zero_fill,
        )?
        .tanh()?;
        let samples = z.dim(2)?;
        check_cancelled()?;
        next.frames += frames;
        next.samples += samples;
        *state = next;
        Ok(z)
    }
}

/// The batch `rows` (in this order) share as lanes `0..rows.len()`, if they
/// are exactly its rows.
fn shared_batch<'a>(
    states: &'a [Audio8DacStreamState],
    rows: &[usize],
) -> Option<&'a Arc<Audio8DacBatchedHistory>> {
    let (batch, _) = states[rows[0]].shared.as_ref()?;
    (batch.rows == rows.len()
        && rows.iter().enumerate().all(|(lane, &row)| {
            states[row]
                .shared
                .as_ref()
                .is_some_and(|(shared, shared_lane)| {
                    Arc::ptr_eq(shared, batch) && *shared_lane == lane
                })
        }))
    .then_some(batch)
}

/// `rows` zero rows shaped like `tensor`'s rows (`[rows, ...]`).
fn zero_rows(tensor: &Tensor, rows: usize) -> Result<Tensor> {
    let mut dims = tensor.dims().to_vec();
    dims[0] = rows;
    Tensor::zeros(dims, tensor.dtype(), tensor.device()).map_err(Error::from)
}

/// `zero_fill`: a missing history is `keep` zero frames (exactly causal
/// padding), so the joined input always has the same length.
fn history(
    x: &Tensor,
    keep: usize,
    states: &mut Vec<Tensor>,
    cursor: &mut usize,
    zero_fill: bool,
) -> Result<(Tensor, usize)> {
    if keep == 0 {
        return Ok((x.clone(), 0));
    }
    let zeros = match states.get(*cursor) {
        None if zero_fill => {
            let (batch, channels, _) = x.dims3()?;
            Some(Tensor::zeros(
                (batch, channels, keep),
                x.dtype(),
                x.device(),
            )?)
        }
        _ => None,
    };
    let prior = states.get(*cursor).or(zeros.as_ref());
    let prior_len = prior.map(|p| p.dim(2)).transpose()?.unwrap_or(0);
    let joined = match prior {
        Some(p) => Tensor::cat(&[p, x], 2)?,
        None => x.clone(),
    };
    let len = joined.dim(2)?;
    // copy is mandatory: narrow/contiguous can retain the full chunk allocation.
    let tail = joined
        .narrow(2, len.saturating_sub(keep), len.min(keep))?
        .copy()?;
    if *cursor == states.len() {
        states.push(tail);
    } else {
        states[*cursor] = tail;
    }
    *cursor += 1;
    Ok((joined, prior_len))
}

fn conv(
    layer: &Audio8CausalConv1d,
    x: &Tensor,
    states: &mut Vec<Tensor>,
    cursor: &mut usize,
    zero_fill: bool,
) -> Result<Tensor> {
    if layer.stride != 1 {
        return Err(Error::ConfigError(
            "Audio8 streaming decoder requires unit-stride causal convolutions".into(),
        ));
    }
    let (joined, prior) = history(x, layer.padding_total, states, cursor, zero_fill)?;
    layer
        .forward(&joined)?
        .narrow(2, prior, x.dim(2)?)
        .map_err(Error::from)
}

fn transpose(
    layer: &Audio8CausalConvTranspose1d,
    x: &Tensor,
    states: &mut Vec<Tensor>,
    cursor: &mut usize,
    zero_fill: bool,
) -> Result<Tensor> {
    if layer.left_trim != 0 {
        return Err(Error::ConfigError(
            "Audio8 streaming transpose requires causal left alignment".into(),
        ));
    }
    let stride = layer.conv.config().stride;
    let (joined, prior) = history(
        x,
        layer.right_trim.div_ceil(stride),
        states,
        cursor,
        zero_fill,
    )?;
    // Replaying just the overlapping input support avoids accumulating bias twice.
    layer
        .forward(&joined)?
        .narrow(2, prior * stride, x.dim(2)? * stride)
        .map_err(Error::from)
}

/// `starts`: each row's absolute first frame (fixed-slot calls), whose
/// rotary tables and mask every layer shares; `None` puts the whole batch
/// at `state.frames`.
fn transformer(
    module: &Audio8WindowLimitedTransformer,
    x: &Tensor,
    state: &mut Audio8DacStreamState,
    cancel: CancelCheck<'_>,
    starts: Option<&[usize]>,
) -> Result<Tensor> {
    let mut h = if module.channels_first {
        x.transpose(1, 2)?
    } else {
        x.clone()
    };
    if let Some(p) = &module.input_proj {
        h = p.forward(&h)?;
    }
    let rows = match (starts, module.layers.first()) {
        (Some(starts), Some(first)) => Some(RowAttention::new(
            &first.attention,
            starts,
            h.dim(1)?,
            h.dtype(),
        )?),
        _ => None,
    };
    for (idx, layer) in module.layers.iter().enumerate() {
        cancel()?;
        let normed = layer.attention_norm.forward(&h)?;
        let (attn, kv) = match &rows {
            Some(rows) => {
                rows.attention(&layer.attention, &normed, state.attention.get(idx), cancel)?
            }
            None => attention(
                &layer.attention,
                &normed,
                state.attention.get(idx),
                state.frames,
                cancel,
            )?,
        };
        if idx == state.attention.len() {
            state.attention.push(kv);
        } else {
            state.attention[idx] = kv;
        }
        h = h.broadcast_add(&attn.broadcast_mul(&layer.attention_scale.reshape((
            1,
            1,
            layer.attention_scale.dim(0)?,
        ))?)?)?;
        cancel()?;
        let ff = layer
            .feed_forward
            .forward(&layer.ffn_norm.forward(&h)?)?
            .broadcast_mul(&layer.ffn_scale.reshape((1, 1, layer.ffn_scale.dim(0)?))?)?;
        h = h.broadcast_add(&ff)?;
    }
    h = module.norm.forward(&h)?;
    if let Some(p) = &module.output_proj {
        h = p.forward(&h)?;
    }
    if module.channels_first {
        h.transpose(1, 2).map_err(Error::from)
    } else {
        Ok(h)
    }
}

fn attention(
    layer: &Audio8DacAttention,
    x: &Tensor,
    prior: Option<&AttentionState>,
    position: usize,
    cancel: CancelCheck<'_>,
) -> Result<(Tensor, AttentionState)> {
    let (batch, len, _) = x.dims3()?;
    let qdim = layer.num_heads * layer.head_dim;
    let kvdim = layer.num_kv_heads * layer.head_dim;
    let qkv = layer.wqkv.forward(x)?;
    let q = layer
        .rotary
        .apply(
            &qkv.narrow(2, 0, qdim)?
                .reshape((batch, len, layer.num_heads, layer.head_dim))?,
            position,
        )?
        .transpose(1, 2)?;
    let k = layer
        .rotary
        .apply(
            &qkv.narrow(2, qdim, kvdim)?.reshape((
                batch,
                len,
                layer.num_kv_heads,
                layer.head_dim,
            ))?,
            position,
        )?
        .transpose(1, 2)?;
    let v = qkv
        .narrow(2, qdim + kvdim, kvdim)?
        .reshape((batch, len, layer.num_kv_heads, layer.head_dim))?
        .transpose(1, 2)?;
    let (k, v) = match prior {
        Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
        None => (k, v),
    };
    let keys = k.dim(2)?;
    let keep = keys.min(layer.window_size - 1);
    let next = (
        k.narrow(2, keys - keep, keep)?.copy()?,
        v.narrow(2, keys - keep, keep)?.copy()?,
    );
    let k = repeat_kv(&k.transpose(1, 2)?, layer.num_heads, layer.num_kv_heads)?
        .transpose(1, 2)?
        .reshape((batch * layer.num_heads, keys, layer.head_dim))?;
    let v = repeat_kv(&v.transpose(1, 2)?, layer.num_heads, layer.num_kv_heads)?
        .transpose(1, 2)?
        .reshape((batch * layer.num_heads, keys, layer.head_dim))?;
    let q = q.reshape((batch * layer.num_heads, len, layer.head_dim))?;
    let mut outputs = Vec::new();
    for start in (0..len).step_by(ATTENTION_QUERY_BLOCK) {
        cancel()?;
        let count = ATTENTION_QUERY_BLOCK.min(len - start);
        let first_query = keys - len + start;
        let first_key = (first_query + 1).saturating_sub(layer.window_size);
        let key_count = first_query + count - first_key;
        let scores = (q.narrow(1, start, count)?.contiguous()?.matmul(
            &k.narrow(1, first_key, key_count)?
                .transpose(1, 2)?
                .contiguous()?,
        )? * (layer.head_dim as f64).sqrt().recip())?;
        let mask = (first_query..first_query + count)
            .flat_map(|row| {
                (first_key..first_key + key_count).map(move |col| {
                    if col <= row && col >= (row + 1).saturating_sub(layer.window_size) {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask =
            Tensor::from_vec(mask, (1, count, key_count), x.device())?.to_dtype(x.dtype())?;
        outputs.push(
            ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?
                .matmul(&v.narrow(1, first_key, key_count)?.contiguous()?)?,
        );
    }
    let output = Tensor::cat(&outputs, 1)?
        .reshape((batch, layer.num_heads, len, layer.head_dim))?
        .transpose(1, 2)?
        .reshape((batch, len, qdim))?;
    Ok((layer.wo.forward(&output)?, next))
}

/// What every layer of one fixed-slot push shares: each row's rotary
/// tables and its mask over `window - 1` cached keys (zero until the stream
/// has that many) followed by the push's own keys.
struct RowAttention {
    cos: Tensor,
    sin: Tensor,
    /// `[rows, 1, len, window - 1 + len]`: 0 where a key is real and in the
    /// query's window, -inf elsewhere.
    mask: Tensor,
    cached: usize,
}

impl RowAttention {
    fn new(layer: &Audio8DacAttention, starts: &[usize], len: usize, dtype: DType) -> Result<Self> {
        let cached = layer.window_size.checked_sub(1).ok_or_else(|| {
            Error::ConfigError("Audio8 streaming attention requires a window".into())
        })?;
        let (cos, sin) = layer.rotary.row_tables(starts, len)?;
        let keys = cached + len;
        let mut mask = Vec::with_capacity(starts.len() * len * keys);
        for &start in starts {
            // Key j is frame `start - cached + j`; the first `cached - start`
            // are padding. Query i sees keys i..=cached + i (its window).
            let first_real = cached.saturating_sub(start);
            for query in 0..len {
                mask.extend((0..keys).map(|key| {
                    if key >= query.max(first_real) && key <= cached + query {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                }));
            }
        }
        let mask =
            Tensor::from_vec(mask, (starts.len(), 1, len, keys), cos.device())?.to_dtype(dtype)?;
        Ok(Self {
            cos,
            sin,
            mask,
            cached,
        })
    }

    /// One layer's windowed attention over every row at its own positions,
    /// with the same shapes at every stream age.
    fn attention(
        &self,
        layer: &Audio8DacAttention,
        x: &Tensor,
        prior: Option<&AttentionState>,
        cancel: CancelCheck<'_>,
    ) -> Result<(Tensor, AttentionState)> {
        let (batch, len, _) = x.dims3()?;
        let qdim = layer.num_heads * layer.head_dim;
        let kvdim = layer.num_kv_heads * layer.head_dim;
        let qkv = layer.wqkv.forward(x)?;
        let q = layer
            .rotary
            .apply_rows(
                &qkv.narrow(2, 0, qdim)?
                    .reshape((batch, len, layer.num_heads, layer.head_dim))?,
                &self.cos,
                &self.sin,
            )?
            .transpose(1, 2)?;
        let k = layer
            .rotary
            .apply_rows(
                &qkv.narrow(2, qdim, kvdim)?.reshape((
                    batch,
                    len,
                    layer.num_kv_heads,
                    layer.head_dim,
                ))?,
                &self.cos,
                &self.sin,
            )?
            .transpose(1, 2)?;
        let v = qkv
            .narrow(2, qdim + kvdim, kvdim)?
            .reshape((batch, len, layer.num_kv_heads, layer.head_dim))?
            .transpose(1, 2)?;
        let (pk, pv) = match prior {
            Some((pk, pv)) if pk.dim(2)? == self.cached && pv.dim(2)? == self.cached => {
                (pk.clone(), pv.clone())
            }
            Some(_) => {
                return Err(Error::InferenceError(
                    "Audio8 fixed-slot attention cache has the wrong length".into(),
                ))
            }
            None => {
                let zeros = Tensor::zeros(
                    (batch, layer.num_kv_heads, self.cached, layer.head_dim),
                    k.dtype(),
                    k.device(),
                )?;
                (zeros.clone(), zeros)
            }
        };
        let k = Tensor::cat(&[&pk, &k], 2)?;
        let v = Tensor::cat(&[&pv, &v], 2)?;
        let keys = self.cached + len;
        let next = (
            k.narrow(2, len, self.cached)?.copy()?,
            v.narrow(2, len, self.cached)?.copy()?,
        );
        let k = repeat_kv(&k.transpose(1, 2)?, layer.num_heads, layer.num_kv_heads)?
            .transpose(1, 2)?
            .reshape((batch * layer.num_heads, keys, layer.head_dim))?;
        let v = repeat_kv(&v.transpose(1, 2)?, layer.num_heads, layer.num_kv_heads)?
            .transpose(1, 2)?
            .reshape((batch * layer.num_heads, keys, layer.head_dim))?;
        let q = q.reshape((batch * layer.num_heads, len, layer.head_dim))?;
        let mut outputs = Vec::new();
        for start in (0..len).step_by(ATTENTION_QUERY_BLOCK) {
            cancel()?;
            let count = ATTENTION_QUERY_BLOCK.min(len - start);
            // Queries start..start + count see keys start..start + count + cached.
            let key_count = count + self.cached;
            let scores = (q.narrow(1, start, count)?.contiguous()?.matmul(
                &k.narrow(1, start, key_count)?
                    .transpose(1, 2)?
                    .contiguous()?,
            )? * (layer.head_dim as f64).sqrt().recip())?;
            let mask = self
                .mask
                .narrow(2, start, count)?
                .narrow(3, start, key_count)?;
            let scores = scores
                .reshape((batch, layer.num_heads, count, key_count))?
                .broadcast_add(&mask)?
                .reshape((batch * layer.num_heads, count, key_count))?;
            outputs.push(
                ops::softmax_last_dim(&scores)?
                    .matmul(&v.narrow(1, start, key_count)?.contiguous()?)?,
            );
        }
        let output = Tensor::cat(&outputs, 1)?
            .reshape((batch, layer.num_heads, len, layer.head_dim))?
            .transpose(1, 2)?
            .reshape((batch, len, qdim))?;
        Ok((layer.wo.forward(&output)?, next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn close(left: &Tensor, right: &Tensor) {
        assert_eq!(left.dims(), right.dims());
        let a = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert!((a - b).abs() < 1e-5, "sample {i}: {a} versus {b}");
        }
    }

    #[test]
    fn convolution_and_transpose_preserve_dilated_history_and_bias() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            (0..43).map(|i| (i as f32 / 7.0).sin()).collect::<Vec<_>>(),
            (1, 1, 43),
            &device,
        )
        .unwrap();
        // The plain convolution and the depthwise shift-and-accumulate
        // keep the same history (one channel is a depthwise layer too).
        let forms = [1, 3, 9].into_iter().flat_map(|dilation| {
            [
                (dilation, None),
                (
                    dilation,
                    Some(
                        crate::kernels::depthwise_conv::DepthwiseConv1dSpec::padded(0, 0)
                            .with_dilation(dilation),
                    ),
                ),
            ]
        });
        for (dilation, depthwise) in forms {
            let layer = Audio8CausalConv1d {
                conv: Conv1d::new(
                    Tensor::from_vec(vec![0.2f32, -0.3, 0.4], (1, 1, 3), &device).unwrap(),
                    Some(Tensor::new(&[0.37f32], &device).unwrap()),
                    Conv1dConfig {
                        dilation,
                        ..Default::default()
                    },
                ),
                effective_kernel: 2 * dilation + 1,
                stride: 1,
                padding_total: 2 * dilation,
                depthwise,
            };
            let mut states = Vec::new();
            let chunks = (0..43)
                .map(|i| {
                    conv(
                        &layer,
                        &x.narrow(2, i, 1).unwrap(),
                        &mut states,
                        &mut 0,
                        false,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            close(
                &layer.forward(&x).unwrap(),
                &Tensor::cat(&chunks, 2).unwrap(),
            );
            assert_eq!(states[0].elem_count(), 2 * dilation);
        }
        for stride in [1, 2, 4, 8] {
            let layer = Audio8CausalConvTranspose1d {
                conv: ConvTranspose1d::new(
                    Tensor::from_vec(
                        (0..2 * stride)
                            .map(|i| i as f32 / 11.0 - 0.2)
                            .collect::<Vec<_>>(),
                        (1, 1, 2 * stride),
                        &device,
                    )
                    .unwrap(),
                    Some(Tensor::new(&[0.37f32], &device).unwrap()),
                    ConvTranspose1dConfig {
                        stride,
                        ..Default::default()
                    },
                ),
                left_trim: 0,
                right_trim: stride,
            };
            let mut states = Vec::new();
            let chunks = (0..43)
                .map(|i| {
                    transpose(
                        &layer,
                        &x.narrow(2, i, 1).unwrap(),
                        &mut states,
                        &mut 0,
                        false,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            close(
                &layer.forward(&x).unwrap(),
                &Tensor::cat(&chunks, 2).unwrap(),
            );
            assert_eq!(states[0].elem_count(), 1);
        }
    }

    #[test]
    fn bounded_attention_matches_absolute_position_full_oracle() {
        let device = Device::Cpu;
        let weight = |out, input| {
            Tensor::from_vec(
                (0..out * input)
                    .map(|i| ((i * 7 % 23) as f32 - 11.0) / 31.0)
                    .collect::<Vec<_>>(),
                (out, input),
                &device,
            )
            .unwrap()
        };
        let layer = Audio8DacAttention {
            wqkv: Linear::new(weight(8, 4), None),
            wo: Linear::new(weight(4, 4), None),
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rotary: Audio8RotaryCache::new(160, 2, 10000.0, DType::BF16, &device).unwrap(),
            window_size: 7,
        };
        let x = Tensor::from_vec(
            (0..149 * 4)
                .map(|i| ((i * 13 % 29) as f32 - 11.0) / 17.0)
                .collect::<Vec<_>>(),
            (1, 149, 4),
            &device,
        )
        .unwrap();
        for size in [1, 3, 11, 65] {
            let mut kv = None;
            let mut parts = Vec::new();
            for start in (0..149).step_by(size) {
                let (out, next) = attention(
                    &layer,
                    &x.narrow(1, start, size.min(149 - start)).unwrap(),
                    kv.as_ref(),
                    start,
                    &|| Ok(()),
                )
                .unwrap();
                assert!(next.0.dim(2).unwrap() <= 6);
                parts.push(out);
                kv = Some(next);
            }
            close(
                &layer.forward(&x, &|| Ok(())).unwrap(),
                &Tensor::cat(&parts, 1).unwrap(),
            );
        }
    }
    /// Run with the released codec.pth and a frozen real generated codebook
    /// matrix (JSON array of codebook rows). Never substitutes CPU or fake codes.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA, IZWI_AUDIO8_CODEC_PATH and IZWI_AUDIO8_CODEBOOK_FIXTURE"]
    fn cuda_real_codec_streaming_matches_frozen_codebooks() {
        use crate::models::architectures::audio8::codec::Audio8CodecWeights;
        let device = Device::new_cuda(0).expect("CUDA device");
        let path = std::env::var("IZWI_AUDIO8_CODEC_PATH").expect("codec.pth path");
        let fixture = std::env::var("IZWI_AUDIO8_CODEBOOK_FIXTURE").expect("real codebook JSON");
        let codebooks: Vec<Vec<u32>> =
            serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
        assert!(
            codebooks[0].len() > 128,
            "fixture must cross the decoder attention window"
        );
        let weights =
            Audio8CodecWeights::load(std::path::Path::new(&path), &device, DType::F32).unwrap();
        let codec =
            Audio8DacDecoder::load(Audio8DacConfig::current(), weights.var_builder()).unwrap();
        let full = codec
            .decode_codebooks(&codebooks)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for chunk in [1, 4, 16, 32] {
            let mut state = Audio8DacStreamState::default();
            let mut samples = Vec::new();
            for start in (0..codebooks[0].len()).step_by(chunk) {
                let end = (start + chunk).min(codebooks[0].len());
                let rows = codebooks
                    .iter()
                    .map(|row| row[start..end].to_vec())
                    .collect::<Vec<_>>();
                samples.extend(codec.push_frames(&mut state, &rows, &|| Ok(())).unwrap());
                assert!(
                    state.retained_tensor_bytes()
                        <= codec.config.streaming_history_bound_bytes().unwrap()
                );
            }
            samples.extend(state.flush());
            assert_eq!(samples.len(), full.len());
            let max_error = samples
                .iter()
                .zip(&full)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // F32 reduction order may change with launch shapes; 1e-4 is a
            // qualification threshold, not a claim of measured CUDA parity.
            assert!(
                max_error < 1e-4,
                "chunk {chunk}: maximum sample error {max_error}"
            );
        }
    }

    /// Rows that push together keep one shared batch, lane by lane, push
    /// after push (their histories are never split per row or concatenated
    /// again) and decode bit for bit as from their own concatenated tensors,
    /// reordered rows included; each lane reports its own retained bytes and
    /// decodes as its row does alone; a row pushed alone leaves the batch for
    /// its own tensors.
    #[test]
    fn rows_pushed_together_keep_one_shared_batch() {
        let device = Device::Cpu;
        let codec = super::super::tests::tiny_decoder_fixture(&device, true);
        let codes = |step: usize| {
            (0..3)
                .map(|row| {
                    (0..2)
                        .map(|book| {
                            (0..2)
                                .map(|frame| ((step + 2 * row + book + frame) % 8) as u32)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let mut states = vec![Audio8DacStreamState::default(); 3];
        let mut scalar = states.clone();
        let mut previous: Option<Arc<Audio8DacBatchedHistory>> = None;
        for step in 0..4 {
            if step == 2 {
                // The same rows in another order cannot reuse their lanes.
                states.rotate_left(1);
                scalar.rotate_left(1);
            }
            let codes = codes(step);
            // The same push from each row's own tensors (concatenated):
            // reusing the shared batch must give the same bits.
            let mut unshared = states
                .iter()
                .map(|state| state.unshared().unwrap())
                .collect::<Vec<_>>();
            let concatenated = codec
                .push_frames_batch(&mut unshared, &codes, &|| Ok(()))
                .unwrap();
            let actual = codec
                .push_frames_batch(&mut states, &codes, &|| Ok(()))
                .unwrap();
            assert_eq!(actual, concatenated, "step {step}");
            for (row, (state, codes)) in scalar.iter_mut().zip(&codes).enumerate() {
                let expected = codec.push_frames(state, codes, &|| Ok(())).unwrap();
                let error = crate::kernels::numeric_check::max_abs_diff(&actual[row], &expected);
                assert!(error < 1e-5, "step {step} row {row}: {error}");
            }
            let (batch, _) = states[0].shared.as_ref().expect("a shared batch");
            for (lane, state) in states.iter().enumerate() {
                let (shared, shared_lane) = state.shared.as_ref().unwrap();
                assert!(Arc::ptr_eq(shared, batch));
                assert_eq!(*shared_lane, lane);
                assert!(state.histories.is_empty() && state.attention.is_empty());
                assert_eq!(
                    state.retained_tensor_bytes(),
                    scalar[lane].retained_tensor_bytes()
                );
            }
            // Each push publishes a new batch; the last one is released.
            if let Some(previous) = previous.take() {
                assert!(!Arc::ptr_eq(&previous, batch));
                assert_eq!(Arc::strong_count(&previous), 1);
            }
            previous = Some(batch.clone());
        }
        let alone = codes(9).swap_remove(1);
        codec
            .push_frames(&mut states[1], &alone, &|| Ok(()))
            .unwrap();
        assert!(states[1].shared.is_none() && !states[1].histories.is_empty());
    }

    fn fixed_slots(device: &Device, slots: usize, shared: bool) -> Audio8DacDecoder {
        let mut codec = super::super::tests::tiny_decoder_fixture(device, true);
        codec.set_codec_calls(Some(Audio8CodecCalls { slots, shared }));
        codec
    }

    /// `frames` frames of a stream's codebooks (two books of eight codes).
    fn stream_codes(seed: usize, frames: usize) -> Vec<Vec<u32>> {
        (0..2)
            .map(|book| {
                (0..frames)
                    .map(|frame| ((seed * 5 + frame * 3 + book * 7 + frame / 5) % 8) as u32)
                    .collect()
            })
            .collect()
    }

    fn slice(codes: &[Vec<u32>], from: usize, to: usize) -> Vec<Vec<u32>> {
        codes.iter().map(|book| book[from..to].to_vec()).collect()
    }

    fn shapes(state: &Audio8DacStreamState) -> Vec<Vec<usize>> {
        let (histories, attention) = state.row_tensors().unwrap();
        histories
            .iter()
            .chain(attention.iter().flat_map(|(k, v)| [k, v]))
            .map(|tensor| tensor.dims().to_vec())
            .collect()
    }

    fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
        rows.iter()
            .map(|row| row.iter().map(|v| v.to_bits()).collect())
            .collect()
    }

    /// A fixed-slot push (zero-filled histories, a full attention cache,
    /// per-row positions and masks) decodes what the age-bucketed push
    /// does, and keeps the same retained shapes at every stream age, across
    /// the attention window.
    #[test]
    fn fixed_slot_push_matches_bucketed_push_at_one_shape() {
        let device = Device::Cpu;
        let bucketed = super::super::tests::tiny_decoder_fixture(&device, true);
        let fixed = fixed_slots(&device, 3, true);
        let codes = stream_codes(1, 53);
        let mut legacy = Audio8DacStreamState::default();
        let mut uniform = Audio8DacStreamState::default();
        let mut retained = None;
        let mut start = 0;
        for size in [4, 16, 1, 3, 16, 13] {
            let chunk = slice(&codes, start, start + size);
            start += size;
            let expected = bucketed
                .push_frames(&mut legacy, &chunk, &|| Ok(()))
                .unwrap();
            let actual = fixed.push_frames(&mut uniform, &chunk, &|| Ok(())).unwrap();
            assert_eq!(actual.len(), expected.len());
            let error = crate::kernels::numeric_check::max_abs_diff(&actual, &expected);
            assert!(error < 1e-5, "frame {start}: {error}");
            assert_eq!(uniform.decoded_frames(), legacy.decoded_frames());
            assert_eq!(uniform.emitted_samples(), legacy.emitted_samples());
            // Full-length histories from the first push on.
            let shape = shapes(&uniform);
            assert_eq!(*retained.get_or_insert_with(|| shape.clone()), shape);
            assert!(
                uniform.retained_tensor_bytes()
                    <= fixed.config.streaming_history_bound_bytes().unwrap()
            );
        }
        assert_eq!(start, 53);
        // A row in a call of three keeps only its own lane.
        let (batch, _) = uniform.shared.as_ref().unwrap();
        assert_eq!(batch.rows, 1);
        assert!(batch.histories.iter().all(|h| h.dim(0).unwrap() == 1));
    }

    /// Streams of different ages pushed together in fixed-slot calls (more
    /// streams than slots, fresh and aged rows in one call, rows of
    /// different chunk sizes in one batch) decode bit for bit as each
    /// pushed alone in a call of its own, push after push; the shared-call
    /// check agrees.
    fn check_rows_pushed_together_match_each_alone(device: &Device) {
        let together = fixed_slots(device, 4, true);
        let apart = fixed_slots(device, 4, false);
        assert!(together.check_shared_calls(4).unwrap());
        let streams = (0..6).map(|row| stream_codes(row, 80)).collect::<Vec<_>>();
        let mut shared = vec![Audio8DacStreamState::default(); 6];
        let mut alone = shared.clone();
        let mut cursor = [0usize; 6];
        // Different ages: row r first pushes r % 3 chunks of four alone.
        for row in 0..6 {
            for _ in 0..row % 3 {
                let chunk = slice(&streams[row], cursor[row], cursor[row] + 4);
                cursor[row] += 4;
                let a = together
                    .push_frames(&mut shared[row], &chunk, &|| Ok(()))
                    .unwrap();
                let b = apart
                    .push_frames(&mut alone[row], &chunk, &|| Ok(()))
                    .unwrap();
                assert_eq!(bits(&[a]), bits(&[b]));
            }
        }
        for sizes in [[4; 6], [16; 6], [16, 4, 16, 1, 16, 3], [16; 6]] {
            let codes = (0..6)
                .map(|row| {
                    let chunk = slice(&streams[row], cursor[row], cursor[row] + sizes[row]);
                    cursor[row] += sizes[row];
                    chunk
                })
                .collect::<Vec<_>>();
            let a = together
                .push_frames_batch(&mut shared, &codes, &|| Ok(()))
                .unwrap();
            let b = apart
                .push_frames_batch(&mut alone, &codes, &|| Ok(()))
                .unwrap();
            assert_eq!(bits(&a), bits(&b), "sizes {sizes:?}");
            for (row, (a, b)) in shared.iter().zip(&alone).enumerate() {
                assert_eq!(a.decoded_frames(), cursor[row]);
                assert_eq!(b.decoded_frames(), cursor[row]);
                assert_eq!(a.emitted_samples(), b.emitted_samples());
            }
        }
    }

    #[test]
    fn rows_pushed_together_in_fixed_slots_match_each_alone() {
        check_rows_pushed_together_match_each_alone(&Device::Cpu);
    }

    /// The same on a CUDA device, where cuBLAS and cuDNN pick kernels by
    /// shape: rows of a fixed-slot call must not see their peers.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rows_pushed_together_in_fixed_slots_match_each_alone() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        check_rows_pushed_together_match_each_alone(&device);
    }

    /// A request's chunks end at the same absolute frames however its
    /// quanta fall (frames arriving one at a time or in bursts, decoded
    /// alone or beside another stream), so its PCM is the same bits.
    #[test]
    fn chunk_boundaries_do_not_depend_on_scheduling() {
        use crate::models::architectures::audio8::{audio8_codec_chunks, audio8_codec_quantum_end};
        let device = Device::Cpu;
        let codec = fixed_slots(&device, 4, true);
        let total = 75;
        let codes = stream_codes(3, total);
        let peer_codes = stream_codes(4, 200);
        // `arrivals`: frames generated before each quantum is considered.
        let decode = |arrivals: &[usize], with_peer: bool| {
            let mut state = Audio8DacStreamState::default();
            let mut peer = Audio8DacStreamState::default();
            let mut peer_cursor = 0;
            let mut pcm = Vec::new();
            let mut pushes = Vec::new();
            for &generated in arrivals {
                let finished = generated == total;
                // Quanta until no whole chunk (or tail) is due.
                loop {
                    let decoded = state.decoded_frames();
                    let end = audio8_codec_quantum_end(decoded, generated, finished, 64);
                    if end == decoded {
                        break;
                    }
                    for (from, to) in audio8_codec_chunks(decoded, end) {
                        pushes.push((from, to));
                        let chunk = slice(&codes, from, to);
                        if with_peer {
                            let peer_chunk = slice(&peer_codes, peer_cursor, peer_cursor + 16);
                            peer_cursor += 16;
                            let mut states = [state.clone(), peer.clone()];
                            let mut out = codec
                                .push_frames_batch(&mut states, &[chunk, peer_chunk], &|| Ok(()))
                                .unwrap();
                            [state, peer] = states;
                            pcm.extend(out.swap_remove(0));
                        } else {
                            pcm.extend(codec.push_frames(&mut state, &chunk, &|| Ok(())).unwrap());
                        }
                    }
                }
            }
            assert_eq!(state.decoded_frames(), total);
            (pushes, pcm)
        };
        let steady = (1..=total).collect::<Vec<_>>();
        let (pushes, pcm) = decode(&steady, false);
        assert_eq!(
            pushes,
            [(0, 4), (4, 20), (20, 36), (36, 52), (52, 68), (68, 75)]
        );
        for (arrivals, with_peer) in [
            (vec![3, 30, 31, 74, 75], false),
            (vec![9, 70, 75], true),
            (vec![75], true),
        ] {
            let (other_pushes, other_pcm) = decode(&arrivals, with_peer);
            assert_eq!(other_pushes, pushes, "{arrivals:?}");
            assert_eq!(
                bits(&[other_pcm]),
                bits(std::slice::from_ref(&pcm)),
                "{arrivals:?}"
            );
        }
    }
}
