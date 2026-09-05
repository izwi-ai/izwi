//! Device sampling primitives with explicit, caller-owned RNG draws and history.
//! Status is 1 for a valid result and 0 for an invalid/non-finite distribution.
//! No function mutates the caller's history or RNG. CPU implementations are
//! portable numerical references; CUDA never transfers a vocabulary to the host.
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor,
};

#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
}
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.,
            repetition_penalty: 1.,
            presence_penalty: 0.,
            frequency_penalty: 0.,
            top_k: 0,
            top_p: 1.,
            min_p: 0.,
        }
    }
}
fn rows(t: &Tensor) -> Result<(usize, usize)> {
    let (r, v) = t.dims2()?;
    if r == 0 || v == 0 || v >= 1 << 24 || r.checked_mul(v).is_none_or(|n| n > i32::MAX as usize) {
        candle_core::bail!("sampling requires nonempty bounded [rows,vocab]")
    }
    Ok((r, v))
}
fn same(a: &Tensor, b: &Tensor) -> Result<()> {
    if !a.device().same_device(b.device()) {
        candle_core::bail!("sampling device mismatch")
    }
    Ok(())
}
/// F32/F16/BF16 [rows,vocab] -> U32 [rows,2]: token, finite status.
pub fn greedy_rows(logits: &Tensor) -> Result<Tensor> {
    rows(logits)?;
    logits
        .to_dtype(DType::F32)?
        .contiguous()?
        .apply_op1_no_bwd(&Greedy)
}
/// Penalties, temperature, top-k, then top-p and min-p relative to maximum.
/// Counts include only the history valid at each row's position. Invalid rows
/// become zero mass and yield status=0 when sampled.
pub fn distributions(logits: &Tensor, counts: &Tensor, params: &SamplingParams) -> Result<Tensor> {
    rows(logits)?;
    same(logits, counts)?;
    if logits.dims() != counts.dims()
        || counts.dtype() != DType::U32
        || !params.temperature.is_finite()
        || !params.repetition_penalty.is_finite()
        || params.repetition_penalty <= 0.
        || !params.presence_penalty.is_finite()
        || !params.frequency_penalty.is_finite()
        || !params.top_p.is_finite()
        || !params.min_p.is_finite()
        || !(0.0..=1.0).contains(&params.min_p)
    {
        candle_core::bail!("invalid sampling parameters or history counts")
    }
    logits
        .to_dtype(DType::F32)?
        .contiguous()?
        .apply_op2_no_bwd(&counts.contiguous()?, &Distribution(*params))
}
/// Probabilities [rows,vocab], uniforms [rows] -> U32 token/status [rows,2].
pub fn sample_rows(probabilities: &Tensor, uniforms: &Tensor) -> Result<Tensor> {
    let (r, _) = rows(probabilities)?;
    same(probabilities, uniforms)?;
    if probabilities.dtype() != DType::F32
        || uniforms.dtype() != DType::F32
        || uniforms.dims() != [r]
    {
        candle_core::bail!("invalid sampling probability/draw shape")
    };
    probabilities
        .contiguous()?
        .apply_op2_no_bwd(&uniforms.contiguous()?, &Sample)
}
/// p/q [rows,vocab], proposal IDs [rows], draws [rows,2] ->
/// U32 [rows,3]: accepted, proposal-or-residual-token, status.
/// p and q must be the normalized distributions actually used for sampling.
/// The caller commits only the accepted prefix and first correction.
pub fn verify_rows(
    p: &Tensor,
    q: &Tensor,
    proposals: &Tensor,
    uniforms: &Tensor,
) -> Result<Tensor> {
    let (r, _) = rows(p)?;
    same(p, q)?;
    same(p, proposals)?;
    same(p, uniforms)?;
    if p.dtype() != DType::F32
        || q.dtype() != DType::F32
        || p.dims() != q.dims()
        || proposals.dtype() != DType::U32
        || proposals.dims() != [r]
        || uniforms.dtype() != DType::F32
        || uniforms.dims() != [r, 2]
    {
        candle_core::bail!("invalid speculative p/q contract")
    }
    let meta = Tensor::cat(
        &[&proposals.to_dtype(DType::F32)?.reshape((r, 1))?, uniforms],
        1,
    )?
    .contiguous()?;
    p.contiguous()?
        .apply_op3_no_bwd(&q.contiguous()?, &meta, &Verify)
}
fn f32s<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let CpuStorage::F32(s) = s else {
        candle_core::bail!("F32 sampling storage required")
    };
    let (a, b) = l
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("contiguous sampling storage required".into()))?;
    Ok(&s[a..b])
}
fn u32s<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [u32]> {
    let CpuStorage::U32(s) = s else {
        candle_core::bail!("U32 sampling storage required")
    };
    let (a, b) = l
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("contiguous counts required".into()))?;
    Ok(&s[a..b])
}
fn argmax(x: &[f32]) -> Option<usize> {
    x.iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|(ai, a), (bi, b)| {
            a.partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| bi.cmp(ai))
        })
        .map(|(i, _)| i)
}
fn reference_probs(x: &[f32], counts: &[u32], p: SamplingParams) -> Vec<f32> {
    let mut values = x.to_vec();
    for (v, &count) in values.iter_mut().zip(counts) {
        if v.is_finite() {
            if count > 0 {
                if p.repetition_penalty > 1. {
                    *v = if *v > 0. {
                        *v / p.repetition_penalty
                    } else {
                        *v * p.repetition_penalty
                    };
                }
                *v -= p.presence_penalty + p.frequency_penalty * count as f32;
            }
            if p.temperature > 1e-5 {
                *v /= p.temperature;
            }
        }
    }
    let mut out = vec![0.; x.len()];
    let Some(best) = argmax(&values) else {
        return out;
    };
    if p.temperature <= 1e-5 {
        out[best] = 1.;
        return out;
    }
    let mut ids = (0..x.len())
        .filter(|&i| values[i].is_finite())
        .collect::<Vec<_>>();
    ids.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if p.top_k > 0 {
        ids.truncate(p.top_k);
    }
    let weights = ids
        .iter()
        .map(|&i| (values[i] - values[best]).exp())
        .collect::<Vec<_>>();
    let sum: f32 = weights.iter().sum();
    let mut cumulative = 0.;
    let mut keep = ids.len();
    for (i, w) in weights.iter().enumerate() {
        cumulative += w / sum;
        if cumulative >= p.top_p.clamp(1e-6, 1.) {
            keep = i + 1;
            break;
        }
    }
    let mut total = 0.;
    for (&id, &w) in ids.iter().zip(&weights).take(keep) {
        if w >= p.min_p {
            out[id] = w;
            total += w;
        }
    }
    if total > 0. {
        for v in &mut out {
            *v /= total;
        }
    }
    out
}
fn draw_ranked(p: &[f32], u: f32, rank: &[usize]) -> Option<u32> {
    if !(0.0..1.0).contains(&u) || p.iter().any(|v| !v.is_finite() || *v < 0.) {
        return None;
    }
    let sum: f32 = p.iter().sum();
    if !sum.is_finite() || sum <= 0. {
        return None;
    }
    let mut c = 0.;
    for &i in rank {
        let v = p[i];
        c += v;
        if v > 0. && u * sum < c {
            return Some(i as u32);
        }
    }
    rank.iter()
        .rev()
        .copied()
        .find(|&i| p[i] > 0.)
        .map(|i| i as u32)
}
fn rank(p: &[f32]) -> Vec<usize> {
    let mut ids = (0..p.len()).collect::<Vec<_>>();
    ids.sort_by(|&a, &b| {
        p[b].partial_cmp(&p[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    ids
}
fn draw(p: &[f32], u: f32) -> Option<u32> {
    draw_ranked(p, u, &rank(p))
}
struct Greedy;
struct Distribution(SamplingParams);
struct Sample;
struct Verify;
impl CustomOp1 for Greedy {
    fn name(&self) -> &'static str {
        "qwen38-greedy-status"
    }
    fn cpu_fwd(&self, x: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (r, v) = l.shape().dims2()?;
        let mut out = Vec::with_capacity(r * 2);
        for row in f32s(x, l)?.chunks(v) {
            let i = argmax(row);
            out.extend([i.unwrap_or(0) as u32, u32::from(i.is_some())]);
        }
        Ok((CpuStorage::U32(out), Shape::from((r, 2))))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        device::greedy(x, l)
    }
}
impl CustomOp2 for Distribution {
    fn name(&self) -> &'static str {
        "qwen38-sampling-distribution"
    }
    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        l: &Layout,
        c: &CpuStorage,
        cl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (_, v) = l.shape().dims2()?;
        let mut out = Vec::new();
        for (row, counts) in f32s(x, l)?.chunks(v).zip(u32s(c, cl)?.chunks(v)) {
            out.extend(reference_probs(row, counts, self.0));
        }
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        l: &Layout,
        c: &candle_core::CudaStorage,
        cl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        device::distribution(x, l, c, cl, self.0)
    }
}
impl CustomOp2 for Sample {
    fn name(&self) -> &'static str {
        "qwen38-sample-status"
    }
    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        l: &Layout,
        u: &CpuStorage,
        ul: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (r, v) = l.shape().dims2()?;
        let mut out = Vec::new();
        for (row, &u) in f32s(x, l)?.chunks(v).zip(f32s(u, ul)?) {
            let i = draw(row, u);
            out.extend([i.unwrap_or(0), u32::from(i.is_some())]);
        }
        Ok((CpuStorage::U32(out), Shape::from((r, 2))))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        l: &Layout,
        u: &candle_core::CudaStorage,
        ul: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        device::select(x, l, None, (u, ul))
    }
}
impl CustomOp3 for Verify {
    fn name(&self) -> &'static str {
        "qwen38-verify-p-q"
    }
    fn cpu_fwd(
        &self,
        p: &CpuStorage,
        l: &Layout,
        q: &CpuStorage,
        ql: &Layout,
        m: &CpuStorage,
        ml: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (r, v) = l.shape().dims2()?;
        let mut out = Vec::new();
        for ((p, q), meta) in f32s(p, l)?
            .chunks(v)
            .zip(f32s(q, ql)?.chunks(v))
            .zip(f32s(m, ml)?.chunks(3))
        {
            let token = meta[0] as usize;
            let valid = meta[0] >= 0.
                && meta[0] < v as f32
                && meta[0].fract() == 0.
                && (0.0..1.0).contains(&meta[1])
                && (0.0..1.0).contains(&meta[2])
                && p.iter().chain(q).all(|x| x.is_finite() && *x >= 0.)
                && token < v
                && q[token] > 0.;
            if !valid {
                out.extend([0, 0, 0]);
            } else if meta[1] < (p[token] / q[token]).min(1.) {
                out.extend([1, token as u32, 1]);
            } else {
                let residual = p
                    .iter()
                    .zip(q)
                    .map(|(p, q)| (p - q).max(0.))
                    .collect::<Vec<_>>();
                let i = draw_ranked(&residual, meta[2], &rank(p));
                out.extend([0, i.unwrap_or(0), u32::from(i.is_some())]);
            }
        }
        Ok((CpuStorage::U32(out), Shape::from((r, 3))))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        p: &candle_core::CudaStorage,
        l: &Layout,
        q: &candle_core::CudaStorage,
        ql: &Layout,
        m: &candle_core::CudaStorage,
        ml: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        device::select(p, l, Some((q, ql)), (m, ml))
    }
}
#[cfg(feature = "cuda")]
#[path = "sampling_device.rs"]
mod device;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn greedy_reduces_rows_with_ties_offsets_and_nonfinite_status() {
        let x = Tensor::from_vec(
            vec![
                9f32,
                9.,
                -1.,
                0.,
                -0.,
                f32::NAN,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                2.,
                3.,
                3.,
            ],
            (4, 3),
            &Device::Cpu,
        )
        .unwrap()
        .narrow(0, 1, 3)
        .unwrap();
        assert_eq!(
            greedy_rows(&x).unwrap().to_vec2::<u32>().unwrap(),
            vec![vec![0, 1], vec![0, 0], vec![1, 1]]
        );
    }
    #[test]
    fn penalties_temperature_and_nucleus_order_have_known_probabilities() {
        let p = SamplingParams {
            temperature: 1.,
            repetition_penalty: 2.,
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            top_k: 3,
            top_p: 0.8,
            min_p: 0.2,
        };
        let result = reference_probs(&[4., 3., 2., 1., f32::INFINITY], &[2, 0, 0, 0, 0], p);
        // Penalized first logit becomes 1; top-k [3,2,1], nucleus retains [3,2].
        assert_eq!(result[0], 0.);
        assert_eq!(result[3], 0.);
        assert_eq!(result[4], 0.);
        assert!((result[1] - 1. / (1. + (-1f32).exp())).abs() < 1e-6);
        assert!((result.iter().sum::<f32>() - 1.).abs() < 1e-6);
        let p = SamplingParams {
            min_p: 0.5,
            ..SamplingParams::default()
        };
        let result = reference_probs(&[0., -1., -2.], &[0; 3], p);
        assert_eq!(result, vec![1., 0., 0.]);
    }
    #[test]
    fn stochastic_p_over_q_acceptance_and_positive_residual_preserve_target_mass() {
        let p = [0.4f32, 0.35, 0.25];
        let q = [0.8f32, 0.1, 0.1];
        let mut emitted = [0f64; 3];
        let residual = p
            .iter()
            .zip(q)
            .map(|(p, q)| (p - q).max(0.))
            .collect::<Vec<_>>();
        let norm = residual.iter().sum::<f32>();
        for d in 0..3 {
            let accept = (p[d] / q[d]).min(1.);
            emitted[d] += (q[d] * accept) as f64;
            for t in 0..3 {
                emitted[t] += (q[d] * (1. - accept) * residual[t] / norm) as f64;
            }
        }
        for (t, expected) in emitted.iter().zip(p) {
            assert!((t - expected as f64).abs() < 1e-6);
        }
        let p = Tensor::from_vec(p.to_vec(), (1, 3), &Device::Cpu).unwrap();
        let q = Tensor::from_vec(q.to_vec(), (1, 3), &Device::Cpu).unwrap();
        let proposal = Tensor::new(&[0u32], &Device::Cpu).unwrap();
        let accept = Tensor::new(&[[0.1f32, 0.8]], &Device::Cpu).unwrap();
        let reject = Tensor::new(&[[0.75f32, 0.2]], &Device::Cpu).unwrap();
        assert_eq!(
            verify_rows(&p, &q, &proposal, &accept)
                .unwrap()
                .to_vec2::<u32>()
                .unwrap(),
            vec![vec![1, 0, 1]]
        );
        assert_eq!(
            verify_rows(&p, &q, &proposal, &reject)
                .unwrap()
                .to_vec2::<u32>()
                .unwrap(),
            vec![vec![0, 1, 1]]
        );
    }
    #[test]
    fn seeded_cdf_uses_probability_rank_and_rejects_bad_draws() {
        let p = [0.1f32, 0.7, 0.2];
        assert_eq!(draw(&p, 0.1), Some(1));
        assert_eq!(draw(&p, 0.75), Some(2));
        assert_eq!(draw(&p, 0.95), Some(0));
        assert_eq!(draw(&p, 1.), None);
        assert_eq!(draw(&[0., 0.], 0.5), None);
        assert_eq!(draw(&[f32::NAN, 1.], 0.5), None);
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_distribution_sample_and_verify_match_reference() {
        let Some(dev) = super::super::cuda_test_device() else {
            return;
        };
        let (r, v) = (3, 2051);
        let x = Tensor::from_vec(
            (0..r * v)
                .map(|i| ((i * 31) % 997) as f32 / 97. - 4.)
                .collect::<Vec<_>>(),
            (r, v),
            &Device::Cpu,
        )
        .unwrap();
        let counts = Tensor::from_vec(
            (0..r * v)
                .map(|i| u32::from(i % 7 == 0) * 3)
                .collect::<Vec<_>>(),
            (r, v),
            &Device::Cpu,
        )
        .unwrap();
        let u = Tensor::new(&[0.01f32, 0.35, 0.99], &Device::Cpu).unwrap();
        let cpu = distributions(
            &x,
            &counts,
            &SamplingParams {
                top_k: 137,
                top_p: 0.9,
                min_p: 0.02,
                repetition_penalty: 1.2,
                presence_penalty: 0.3,
                frequency_penalty: 0.1,
                ..SamplingParams::default()
            },
        )
        .unwrap();
        let gpu = distributions(
            &x.to_device(&dev).unwrap(),
            &counts.to_device(&dev).unwrap(),
            &SamplingParams {
                top_k: 137,
                top_p: 0.9,
                min_p: 0.02,
                repetition_penalty: 1.2,
                presence_penalty: 0.3,
                frequency_penalty: 0.1,
                ..SamplingParams::default()
            },
        )
        .unwrap();
        for (a, e) in gpu
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        {
            assert!((a - e).abs() < 1e-5);
        }
        assert_eq!(
            sample_rows(&cpu, &u).unwrap().to_vec2::<u32>().unwrap(),
            sample_rows(&gpu, &u.to_device(&dev).unwrap())
                .unwrap()
                .to_vec2::<u32>()
                .unwrap()
        );
        assert_eq!(
            greedy_rows(&x).unwrap().to_vec2::<u32>().unwrap(),
            greedy_rows(&x.to_device(&dev).unwrap())
                .unwrap()
                .to_vec2::<u32>()
                .unwrap()
        );
        let p = Tensor::new(&[[0.4f32, 0.6], [0.9, 0.1], [0.5, 0.5]], &Device::Cpu).unwrap();
        let q = Tensor::new(&[[0.8f32, 0.2], [0.5, 0.5], [0.2, 0.8]], &Device::Cpu).unwrap();
        let proposals = Tensor::new(&[0u32, 1, 1], &Device::Cpu).unwrap();
        let draws = Tensor::new(&[[0.75f32, 0.3], [0.1, 0.9], [0.7, 0.2]], &Device::Cpu).unwrap();
        let expected = verify_rows(&p, &q, &proposals, &draws)
            .unwrap()
            .to_vec2::<u32>()
            .unwrap();
        let actual = verify_rows(
            &p.to_device(&dev).unwrap(),
            &q.to_device(&dev).unwrap(),
            &proposals.to_device(&dev).unwrap(),
            &draws.to_device(&dev).unwrap(),
        )
        .unwrap()
        .to_vec2::<u32>()
        .unwrap();
        assert_eq!(expected, actual);
    }
}
