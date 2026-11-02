//! Exact incremental decoding. Histories are immutable, independently allocated
//! tensors: cloning a checkpoint shares storage until the next successful push.
use super::*;

type AttentionState = (Tensor, Tensor);

#[derive(Debug, Clone, Default)]
pub struct FishDrama3DacStreamState {
    histories: Vec<Tensor>,
    attention: Vec<AttentionState>,
    frames: usize,
    samples: usize,
    finished: bool,
}

impl FishDrama3DacStreamState {
    pub(crate) fn decoded_frames(&self) -> usize {
        self.frames
    }
    pub(crate) fn emitted_samples(&self) -> usize {
        self.samples
    }
    pub(crate) fn retained_tensor_bytes(&self) -> u64 {
        self.histories
            .iter()
            .chain(self.attention.iter().flat_map(|(k, v)| [k, v]))
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum()
    }
    /// The full oracle discards the causal transpose tail. All stable samples
    /// have already been emitted; repeated terminal flushes therefore are empty.
    pub(crate) fn flush(&mut self) -> Vec<f32> {
        self.finished = true;
        self.histories.clear();
        self.attention.clear();
        Vec::new()
    }
}

impl FishDrama3DacConfig {
    /// F32 retained history at batch one, independent of utterance length.
    /// Account two copies when retaining a transactional rollback checkpoint.
    pub(crate) fn streaming_history_bound_bytes(&self) -> Result<u64> {
        let mut elements = (self.transformer_layers as u128)
            * 2
            * self.transformer_kv_heads as u128
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
        u64::try_from(elements * 4).map_err(|_| {
            Error::ConfigError("Fish Audio Drama 3 codec stream history overflow".into())
        })
    }
}

impl FishDrama3DacDecoder {
    /// Atomically consume new codebook frames. An error leaves `state` unchanged.
    pub(crate) fn push_frames(
        &self,
        state: &mut FishDrama3DacStreamState,
        codebooks: &[Vec<u32>],
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Vec<f32>> {
        if state.finished {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 codec stream is already flushed".into(),
            ));
        }
        if codebooks.len() == self.config.num_codebooks() && codebooks.iter().all(Vec::is_empty) {
            check_cancelled()?;
            return Ok(Vec::new());
        }
        check_cancelled()?;
        let incoming = codebooks.first().map(Vec::len).unwrap_or(0);
        if state
            .frames
            .checked_add(incoming)
            .is_none_or(|frames| frames > FishDrama3DacConfig::MAX_QUANTIZER_FRAMES)
        {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 codec stream exceeds its absolute position capacity".into(),
            ));
        }
        let codes = codebooks_to_tensor(codebooks, &self.config, self.decoder_device()?)?;
        let mut candidate = state.clone();
        let z = self.push_tensor(&mut candidate, &codes, check_cancelled)?;
        let samples = z.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        check_cancelled()?;
        *state = candidate;
        Ok(samples)
    }

    /// Exact shape buckets share all codec projections/convolutions/attention.
    /// Different stream ages remain separate buckets, never concatenated in time.
    /// No candidate history is published unless every group and readback succeeds.
    pub(crate) fn push_frames_batch(
        &self,
        states: &mut [FishDrama3DacStreamState],
        codebooks: &[Vec<Vec<u32>>],
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        if states.is_empty() || states.len() != codebooks.len() {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 codec batch rows do not match".into(),
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
                    .is_none_or(|n| n > FishDrama3DacConfig::MAX_QUANTIZER_FRAMES)
            {
                return Err(Error::InvalidInput(
                    "Fish Audio Drama 3 codec batch contains flushed or invalid-length row".into(),
                ));
            }
            tensors.push(codebooks_to_tensor(
                codes,
                &self.config,
                self.decoder_device()?,
            )?);
            groups
                .entry((state.frames, incoming))
                .or_default()
                .push(row);
        }
        let mut candidates = states.to_vec();
        let mut outputs = vec![Vec::new(); states.len()];
        for rows in groups.values() {
            let first = &states[rows[0]];
            let mut combined = first.clone();
            combined.histories = (0..first.histories.len())
                .map(|history| {
                    let tensors = rows
                        .iter()
                        .map(|&row| {
                            states[row].histories.get(history).ok_or_else(|| {
                                Error::InvalidInput(
                                    "Fish Audio Drama 3 codec history layouts differ".into(),
                                )
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Tensor::cat(&tensors, 0).map_err(Error::from)
                })
                .collect::<Result<Vec<_>>>()?;
            combined.attention = (0..first.attention.len())
                .map(|layer| {
                    let kv = rows
                        .iter()
                        .map(|&row| {
                            states[row].attention.get(layer).ok_or_else(|| {
                                Error::InvalidInput(
                                    "Fish Audio Drama 3 codec attention layouts differ".into(),
                                )
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let k = kv.iter().map(|(k, _)| k).collect::<Vec<_>>();
                    let v = kv.iter().map(|(_, v)| v).collect::<Vec<_>>();
                    Ok((Tensor::cat(&k, 0)?, Tensor::cat(&v, 0)?))
                })
                .collect::<Result<Vec<_>>>()?;
            for &row in rows {
                if states[row].histories.len() != first.histories.len()
                    || states[row].attention.len() != first.attention.len()
                    || states[row].samples != first.samples
                {
                    return Err(Error::InvalidInput(
                        "Fish Audio Drama 3 codec batch history metadata differs".into(),
                    ));
                }
            }
            let codes = Tensor::cat(
                &rows.iter().map(|&row| &tensors[row]).collect::<Vec<_>>(),
                0,
            )?;
            tracing::debug!(
                stage = "fish_drama3_codec",
                native_rows = rows.len(),
                history_frames = first.frames,
                incoming_frames = codes.dim(2)?,
                "Executing compatible codec tensor batch"
            );
            let audio = self
                .push_tensor(&mut combined, &codes, check_cancelled)?
                .to_dtype(DType::F32)?;
            for (lane, &row) in rows.iter().enumerate() {
                outputs[row] = audio.narrow(0, lane, 1)?.flatten_all()?.to_vec1::<f32>()?;
                candidates[row] = FishDrama3DacStreamState {
                    histories: combined
                        .histories
                        .iter()
                        .map(|h| h.narrow(0, lane, 1)?.copy())
                        .collect::<candle_core::Result<Vec<_>>>()?,
                    attention: combined
                        .attention
                        .iter()
                        .map(|(k, v)| {
                            Ok((k.narrow(0, lane, 1)?.copy()?, v.narrow(0, lane, 1)?.copy()?))
                        })
                        .collect::<Result<Vec<_>>>()?,
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

    fn push_tensor(
        &self,
        state: &mut FishDrama3DacStreamState,
        codes: &Tensor,
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Tensor> {
        let frames = codes.dim(2)?;
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
        )?;
        let mut cursor = 0;
        for block in &q.upsample {
            check_cancelled()?;
            z = transpose(&block.transposed, &z, &mut next.histories, &mut cursor)?;
            let residual = z.clone();
            let b = &block.convnext;
            let mut h = conv(&b.dwconv, &z, &mut next.histories, &mut cursor)?.transpose(1, 2)?;
            h = b
                .pwconv2
                .forward(&b.pwconv1.forward(&b.norm.forward(&h)?)?.gelu_erf()?)?;
            if let Some(gamma) = &b.gamma {
                h = h.broadcast_mul(&gamma.reshape((1, 1, gamma.dim(0)?))?)?;
            }
            z = residual.broadcast_add(&h.transpose(1, 2)?)?;
        }
        let d = &self.decoder;
        z = conv(&d.first, &z, &mut next.histories, &mut cursor)?;
        for block in &d.blocks {
            check_cancelled()?;
            z = transpose(
                &block.transposed,
                &block.snake.forward(&z)?,
                &mut next.histories,
                &mut cursor,
            )?;
            for unit in &block.residuals {
                check_cancelled()?;
                let h = conv(
                    &unit.conv1,
                    &unit.snake1.forward(&z)?,
                    &mut next.histories,
                    &mut cursor,
                )?;
                let h = conv(
                    &unit.conv2,
                    &unit.snake2.forward(&h)?,
                    &mut next.histories,
                    &mut cursor,
                )?;
                z = z.broadcast_add(&h)?;
            }
        }
        z = conv(
            &d.final_conv,
            &d.final_snake.forward(&z)?,
            &mut next.histories,
            &mut cursor,
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

fn history(
    x: &Tensor,
    keep: usize,
    states: &mut Vec<Tensor>,
    cursor: &mut usize,
) -> Result<(Tensor, usize)> {
    if keep == 0 {
        return Ok((x.clone(), 0));
    }
    let prior = states.get(*cursor);
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
    layer: &FishDrama3CausalConv1d,
    x: &Tensor,
    states: &mut Vec<Tensor>,
    cursor: &mut usize,
) -> Result<Tensor> {
    if layer.stride != 1 {
        return Err(Error::ConfigError(
            "Fish Audio Drama 3 codec streaming decoder requires unit-stride causal convolutions"
                .into(),
        ));
    }
    let (joined, prior) = history(x, layer.padding_total, states, cursor)?;
    layer
        .forward(&joined)?
        .narrow(2, prior, x.dim(2)?)
        .map_err(Error::from)
}

fn transpose(
    layer: &FishDrama3CausalConvTranspose1d,
    x: &Tensor,
    states: &mut Vec<Tensor>,
    cursor: &mut usize,
) -> Result<Tensor> {
    if layer.left_trim != 0 {
        return Err(Error::ConfigError(
            "Fish Audio Drama 3 codec streaming transpose requires causal left alignment".into(),
        ));
    }
    let stride = layer.conv.config().stride;
    let (joined, prior) = history(x, layer.right_trim.div_ceil(stride), states, cursor)?;
    // Replaying just the overlapping input support avoids accumulating bias twice.
    layer
        .forward(&joined)?
        .narrow(2, prior * stride, x.dim(2)? * stride)
        .map_err(Error::from)
}

fn transformer(
    module: &FishDrama3WindowLimitedTransformer,
    x: &Tensor,
    state: &mut FishDrama3DacStreamState,
    cancel: CancelCheck<'_>,
) -> Result<Tensor> {
    let mut h = if module.channels_first {
        x.transpose(1, 2)?
    } else {
        x.clone()
    };
    if let Some(p) = &module.input_proj {
        h = p.forward(&h)?;
    }
    for (idx, layer) in module.layers.iter().enumerate() {
        cancel()?;
        let (attn, kv) = attention(
            &layer.attention,
            &layer.attention_norm.forward(&h)?,
            state.attention.get(idx),
            state.frames,
            cancel,
        )?;
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
    layer: &FishDrama3DacAttention,
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
            let layer = FishDrama3CausalConv1d {
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
                .map(|i| conv(&layer, &x.narrow(2, i, 1).unwrap(), &mut states, &mut 0).unwrap())
                .collect::<Vec<_>>();
            close(
                &layer.forward(&x).unwrap(),
                &Tensor::cat(&chunks, 2).unwrap(),
            );
            assert_eq!(states[0].elem_count(), 2 * dilation);
        }
        for stride in [1, 2, 4, 8] {
            let layer = FishDrama3CausalConvTranspose1d {
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
                    transpose(&layer, &x.narrow(2, i, 1).unwrap(), &mut states, &mut 0).unwrap()
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
        let layer = FishDrama3DacAttention {
            wqkv: Linear::new(weight(8, 4), None),
            wo: Linear::new(weight(4, 4), None),
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rotary: FishDrama3CodecRotaryCache::new(160, 2, 10000.0, DType::BF16, &device).unwrap(),
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
    #[ignore = "requires CUDA, IZWI_FISH_DRAMA3_CODEC_PATH and IZWI_FISH_DRAMA3_CODEBOOK_FIXTURE"]
    fn cuda_real_codec_streaming_matches_frozen_codebooks() {
        use crate::models::architectures::fish_drama3::codec::FishDrama3CodecWeights;
        let device = Device::new_cuda(0).expect("CUDA device");
        let path = std::env::var("IZWI_FISH_DRAMA3_CODEC_PATH").expect("codec.pth path");
        let fixture =
            std::env::var("IZWI_FISH_DRAMA3_CODEBOOK_FIXTURE").expect("real codebook JSON");
        let codebooks: Vec<Vec<u32>> =
            serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
        assert!(
            codebooks[0].len() > 128,
            "fixture must cross the decoder attention window"
        );
        let weights =
            FishDrama3CodecWeights::load(std::path::Path::new(&path), &device, DType::F32).unwrap();
        let codec =
            FishDrama3DacDecoder::load(FishDrama3DacConfig::current(), weights.var_builder())
                .unwrap();
        let full = codec
            .decode_codebooks(&codebooks)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for chunk in [1, 4, 16, 32] {
            let mut state = FishDrama3DacStreamState::default();
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
}
