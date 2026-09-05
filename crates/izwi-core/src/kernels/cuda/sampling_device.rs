use super::SamplingParams;
use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::{
    backend::BackendStorage,
    cuda_backend::{CudaStorageSlice, WrapErr},
    CudaStorage, Layout, Result, Shape,
};
fn view<'a>(s: &'a CudaStorage, l: &Layout) -> Result<CudaView<'a, f32>> {
    let CudaStorageSlice::F32(s) = &s.slice else {
        candle_core::bail!("F32 sampling storage")
    };
    let (a, b) = l
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("contiguous sampling required".into()))?;
    Ok(s.slice(a..b))
}
fn config(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
pub(super) fn greedy(x: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
    let (r, v) = l.shape().dims2()?;
    let d = x.device();
    let x = view(x, l)?;
    // SAFETY: the kernel writes both result words per row.
    let mut out = unsafe { d.alloc::<u32>(r * 2)? };
    let f = d.get_or_load_custom_func(
        "qwen38_greedy",
        "izwi_qwen38_sampling",
        super::super::cuda_ptx::SAMPLING,
    )?;
    let mut b = f.builder();
    b.arg(&x);
    b.arg(&mut out);
    candle_core::builder_arg!(b, v as i32);
    unsafe { b.launch(config(r)) }.w()?;
    Ok((
        CudaStorage {
            slice: CudaStorageSlice::U32(out),
            device: d.clone(),
        },
        Shape::from((r, 2)),
    ))
}
pub(super) fn distribution(
    x: &CudaStorage,
    l: &Layout,
    c: &CudaStorage,
    cl: &Layout,
    p: SamplingParams,
) -> Result<(CudaStorage, Shape)> {
    let (r, v) = l.shape().dims2()?;
    let n = r * v;
    let d = x.device();
    let x = view(x, l)?;
    let CudaStorageSlice::U32(c) = &c.slice else {
        candle_core::bail!("U32 sampling counts")
    };
    let c = c.slice(cl.start_offset()..cl.start_offset() + n);
    // SAFETY: prepare initializes the first pair; every merge fills the other pair.
    let mut a = unsafe { d.alloc::<f32>(n)? };
    let mut ai = unsafe { d.alloc::<u32>(n)? };
    let mut z = unsafe { d.alloc::<f32>(n)? };
    let mut zi = unsafe { d.alloc::<u32>(n)? };
    let mut out = unsafe { d.alloc::<f32>(n)? };
    let f = d.get_or_load_custom_func(
        "qwen38_sampling_prepare",
        "izwi_qwen38_sampling",
        super::super::cuda_ptx::SAMPLING,
    )?;
    let mut b = f.builder();
    b.arg(&x);
    b.arg(&c);
    b.arg(&mut a);
    b.arg(&mut ai);
    candle_core::builder_arg!(
        b,
        v as i32,
        n as i32,
        p.temperature,
        p.repetition_penalty,
        p.presence_penalty,
        p.frequency_penalty
    );
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    let f = d.get_or_load_custom_func(
        "qwen38_sampling_merge",
        "izwi_qwen38_sampling",
        super::super::cuda_ptx::SAMPLING,
    )?;
    let mut width = 1;
    while width < v {
        let mut b = f.builder();
        b.arg(&a);
        b.arg(&ai);
        b.arg(&mut z);
        b.arg(&mut zi);
        candle_core::builder_arg!(b, v as i32, n as i32, width as i32);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        std::mem::swap(&mut a, &mut z);
        std::mem::swap(&mut ai, &mut zi);
        width *= 2;
    }
    let f = d.get_or_load_custom_func(
        "qwen38_sampling_probs",
        "izwi_qwen38_sampling",
        super::super::cuda_ptx::SAMPLING,
    )?;
    let mut b = f.builder();
    b.arg(&a);
    b.arg(&ai);
    b.arg(&mut out);
    candle_core::builder_arg!(
        b,
        v as i32,
        p.top_k.min(v) as i32,
        p.top_p.clamp(1e-6, 1.0),
        p.min_p,
        i32::from(p.temperature <= 1e-5)
    );
    unsafe { b.launch(config(r)) }.w()?;
    Ok((
        CudaStorage {
            slice: CudaStorageSlice::F32(out),
            device: d.clone(),
        },
        l.shape().clone(),
    ))
}
pub(super) fn select(
    p: &CudaStorage,
    l: &Layout,
    q: Option<(&CudaStorage, &Layout)>,
    meta: (&CudaStorage, &Layout),
) -> Result<(CudaStorage, Shape)> {
    let (r, v) = l.shape().dims2()?;
    let d = p.device();
    let p = view(p, l)?;
    let q = q.map(|(q, l)| view(q, l)).transpose()?;
    let meta = view(meta.0, meta.1)?;
    let cols = if q.is_some() { 3 } else { 2 };

    let n = r * v;
    // Stable rank preserves the existing seeded descending-p/token-ID CDF order.
    let mut sorted = unsafe { d.alloc::<f32>(n)? };
    let mut order = unsafe { d.alloc::<u32>(n)? };
    let mut tmp = unsafe { d.alloc::<f32>(n)? };
    let mut ti = unsafe { d.alloc::<u32>(n)? };
    let f = d.get_or_load_custom_func(
        "qwen38_sampling_order",
        "izwi_qwen38_sampling",
        super::super::cuda_ptx::SAMPLING,
    )?;
    let mut b = f.builder();
    b.arg(&p);
    b.arg(&mut sorted);
    b.arg(&mut order);
    candle_core::builder_arg!(b, v as i32, n as i32);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    let f = d.get_or_load_custom_func(
        "qwen38_sampling_merge",
        "izwi_qwen38_sampling",
        super::super::cuda_ptx::SAMPLING,
    )?;
    let mut width = 1;
    while width < v {
        let mut b = f.builder();
        b.arg(&sorted);
        b.arg(&order);
        b.arg(&mut tmp);
        b.arg(&mut ti);
        candle_core::builder_arg!(b, v as i32, n as i32, width as i32);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        std::mem::swap(&mut sorted, &mut tmp);
        std::mem::swap(&mut order, &mut ti);
        width *= 2;
    }
    // SAFETY: select initializes the complete result even for invalid rows.
    let mut out = unsafe { d.alloc::<u32>(r * cols)? };
    let f = d.get_or_load_custom_func(
        if q.is_some() {
            "qwen38_verify"
        } else {
            "qwen38_sample"
        },
        "izwi_qwen38_sampling",
        super::super::cuda_ptx::SAMPLING,
    )?;
    let mut b = f.builder();
    b.arg(&p);
    if let Some(q) = &q {
        b.arg(q);
    }
    b.arg(&meta);
    b.arg(&order);
    b.arg(&mut out);
    candle_core::builder_arg!(b, v as i32);
    unsafe { b.launch(config(r)) }.w()?;
    Ok((
        CudaStorage {
            slice: CudaStorageSlice::U32(out),
            device: d.clone(),
        },
        Shape::from((r, cols)),
    ))
}
