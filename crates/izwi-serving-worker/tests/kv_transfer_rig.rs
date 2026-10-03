//! KV page-transfer mock-transport rig (DS10 groundwork B3).
//!
//! Two threads of one test binary exchange DS10 page-transfer frames over a
//! real loopback TCP socket: the producer captures pages from arena-shaped
//! backing tensors with the genuine DS4 codec, frames them per the DS10
//! page-transfer spec (IZKV1 header, geometry, tenant
//! namespace, SHA-256 digest chain), and streams them; the consumer verifies
//! the chain, re-keys the pages into its own fresh backing, and the rig
//! demands bit-identical bytes on both sides. A tampered payload must be
//! rejected by the chain check.
//!
//! This is contract evidence for the framing, not a serving path: no engine
//! code, scheduler state, or protocol types participate. The engine stays
//! node-local per the serving-plan exclusion (ADR 0008).

use candle_core::{DType, Device, Tensor};
use izwi_core::backends::kv::{capture_block, restore_block};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpListener;

const MAGIC: &[u8; 5] = b"IZKV1";

/// One layer's paged geometry: `num_kv_heads`, `key_head_dim`, `value_head_dim`.
type LayerGeometry = (u32, u32, u32);

fn frame(
    namespace: &[u8; 32],
    previous_chain: &[u8; 32],
    position_base: u64,
    page_tokens: u32,
    geometry: &[LayerGeometry],
    dtype: u8,
    payload: &[u8],
) -> Vec<u8> {
    let mut frame = Vec::with_capacity(MAGIC.len() + 4 + 2 + 4 + 2 + payload.len() + 64);
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.push(dtype); // F32 = 0 (the only dtype this rig exercises)
    frame.push(0); // layout: PageTokenHeadDim
    frame.extend_from_slice(&page_tokens.to_le_bytes());
    frame.extend_from_slice(&(geometry.len() as u16).to_le_bytes());
    for (heads, key_dim, value_dim) in geometry {
        frame.extend_from_slice(&heads.to_le_bytes());
        frame.extend_from_slice(&key_dim.to_le_bytes());
        frame.extend_from_slice(&value_dim.to_le_bytes());
    }
    frame.extend_from_slice(&position_base.to_le_bytes());
    frame.extend_from_slice(namespace);
    let chain = digest_chain(previous_chain, position_base, namespace, payload);
    frame.extend_from_slice(&chain);
    frame.extend_from_slice(payload);
    frame
}

fn digest_chain(
    previous: &[u8; 32],
    position_base: u64,
    namespace: &[u8; 32],
    payload: &[u8],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(previous);
    hasher.update(position_base.to_le_bytes());
    hasher.update(namespace);
    hasher.update(payload);
    hasher.finalize().into()
}

/// Deterministic page backing shaped exactly like a CPU KV arena's per-layer
/// tensors: `[capacity_pages, page_tokens, kv_heads, head_dim]`.
fn layer_backing(
    capacity_pages: usize,
    page_tokens: usize,
    heads: usize,
    head_dim: usize,
    seed: u64,
) -> Tensor {
    let count = capacity_pages * page_tokens * heads * head_dim;
    let values = (0..count)
        .map(|index| {
            let raw = ((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed) % 4096;
            (raw as f32 / 1024.0) - 2.0
        })
        .collect();
    Tensor::from_vec(values, (capacity_pages, page_tokens, heads, head_dim), &Device::Cpu)
        .unwrap()
}

fn page_payload(
    backing: &[Tensor],
    geometry: &[LayerGeometry],
    page_tokens: usize,
    page: usize,
) -> Vec<u8> {
    // Per layer: key block then value block, each a contiguous run in the
    // arena dtype (page_transfer.rs layout contract).
    let mut payload = Vec::new();
    for (layer, (heads, key_dim, value_dim)) in geometry.iter().enumerate() {
        let key = backing[layer * 2].narrow(0, page, 1).unwrap();
        let value = backing[layer * 2 + 1].narrow(0, page, 1).unwrap();
        let _ = (heads, key_dim, value_dim, page_tokens);
        let mut buffer = vec![0_u8; key.elem_count() * 4];
        let written = capture_block(&key, 0, &mut buffer).unwrap();
        assert_eq!(written, buffer.len());
        payload.extend_from_slice(&buffer);
        let mut buffer = vec![0_u8; value.elem_count() * 4];
        let written = capture_block(&value, 0, &mut buffer).unwrap();
        assert_eq!(written, buffer.len());
        payload.extend_from_slice(&buffer);
    }
    payload
}

#[test]
fn loopback_page_transfer_preserves_pages_and_enforces_digest_chain() {
    let page_tokens = 4_usize;
    let capacity_pages = 2_usize;
    let geometry: Vec<LayerGeometry> = vec![(1, 2, 2), (2, 2, 1)];
    let dtype_byte = 0_u8; // F32

    // Arena-shaped backing: per layer, one key tensor and one value tensor.
    let mut backing = Vec::new();
    for (layer, &(heads, key_dim, value_dim)) in geometry.iter().enumerate() {
        backing.push(layer_backing(capacity_pages, page_tokens, heads as usize, key_dim as usize, 0x100 + layer as u64));
        backing.push(layer_backing(capacity_pages, page_tokens, heads as usize, value_dim as usize, 0x200 + layer as u64));
    }

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();

    let producer_backing = backing.clone();
    let producer_geometry = geometry.clone();
    let producer = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut previous_chain = [0_u8; 32];
        for page in 0..capacity_pages {
            let payload = page_payload(&producer_backing, &producer_geometry, page_tokens, page);
            let framed = frame(
                &[7_u8; 32], // tenant namespace (DINV-02 scope)
                &previous_chain,
                (page * page_tokens) as u64,
                page_tokens as u32,
                &producer_geometry,
                dtype_byte,
                &payload,
            );
            socket.write_all(&(framed.len() as u32).to_le_bytes()).unwrap();
            socket.write_all(&framed).unwrap();
            previous_chain = digest_chain(&previous_chain, (page * page_tokens) as u64, &[7_u8; 32], &payload);
        }
    });

    let mut stream = std::net::TcpStream::connect(address).unwrap();
    let mut namespace = [0_u8; 32];
    namespace.fill(7);
    let mut previous_chain = [0_u8; 32];
    // Consumer-side re-keyed backing: fresh zero tensors owned by the
    // "receiving process", with identical declared geometry.
    let mut consumer_backing: Vec<Option<Tensor>> = backing
        .iter()
        .map(|tensor| {
            Some(Tensor::zeros(tensor.dims(), DType::F32, &Device::Cpu).unwrap())
        })
        .collect();
    let mut received_payloads = Vec::new();

    for page in 0..capacity_pages {
        let mut length_bytes = [0_u8; 4];
        stream.read_exact(&mut length_bytes).unwrap();
        let length = u32::from_le_bytes(length_bytes) as usize;
        let mut framed = vec![0_u8; length];
        stream.read_exact(&mut framed).unwrap();

        // Header parse + validation (fail-closed before any payload use).
        assert_eq!(&framed[..5], MAGIC, "magic gate");
        let payload_len = u32::from_le_bytes(framed[5..9].try_into().unwrap()) as usize;
        // magic(5) + len(4) + dtype(1) + layout(1) + page_tokens(4) + layers(2)
        // + geometry(12/layer) + position(8) + namespace(32) + chain(32) + payload
        assert_eq!(
            framed.len(),
            9 + 2 + 4 + 2 + geometry.len() * 12 + 8 + 32 + 32 + payload_len
        );
        assert_eq!(framed[9], dtype_byte, "dtype gate");
        assert_eq!(u32::from_le_bytes(framed[11..15].try_into().unwrap()), page_tokens as u32);
        assert_eq!(
            u16::from_le_bytes(framed[15..17].try_into().unwrap()) as usize,
            geometry.len(),
            "layer-count gate"
        );
        let chain_offset = 17 + geometry.len() * 12 + 8 + 32;
        let claimed_chain: [u8; 32] = framed[chain_offset..chain_offset + 32].try_into().unwrap();
        let payload = &framed[chain_offset + 32..];
        let computed = digest_chain(&previous_chain, (page * page_tokens) as u64, &namespace, payload);
        assert_eq!(claimed_chain, computed, "digest chain must verify before attach");
        previous_chain = computed;
        received_payloads.push(payload.to_vec());

        // Re-key: scatter the received page into the consumer's own backing
        // through the same restore codec the arenas use.
        let mut offset = 0_usize;
        for index in 0..backing.len() {
            let destination = consumer_backing[index].take().unwrap();
            let consumed = restore_block(&destination, page, &payload[offset..]).unwrap();
            offset += consumed;
            consumer_backing[index] = Some(destination);
        }
        assert_eq!(offset, payload.len(), "payload must be fully consumed");
    }
    producer.join().unwrap();

    // Bitwise proof: recapturing the re-keyed consumer pages must reproduce
    // the producer payloads byte-for-byte.
    for page in 0..capacity_pages {
        assert_eq!(
            page_payload(
                &consumer_backing
                    .iter()
                    .map(|slot| slot.clone().unwrap())
                    .collect::<Vec<_>>(),
                &geometry,
                page_tokens,
                page,
            ),
            received_payloads[page],
            "consumer page {page} must be bit-identical to the producer's"
        );
    }
}

#[test]
fn tampered_page_payload_is_rejected_by_digest_chain() {
    let namespace = [9_u8; 32];
    let previous = [0_u8; 32];
    let geometry: Vec<LayerGeometry> = vec![(1, 2, 2)];
    let mut framed = frame(
        &namespace,
        &previous,
        0,
        4,
        &geometry,
        0,
        &vec![0.5_f32.to_le_bytes(), 1.5_f32.to_le_bytes()].concat(),
    );
    // Flip one payload byte at the very end of the frame.
    let last = framed.len() - 1;
    framed[last] ^= 0xFF;
    let chain_offset = 9 + 2 + geometry.len() * 12 + 8 + 32;
    let claimed: [u8; 32] = framed[chain_offset..chain_offset + 32].try_into().unwrap();
    let payload = &framed[chain_offset + 32..];
    let computed = digest_chain(&previous, 0, &namespace, payload);
    assert_ne!(
        claimed, computed,
        "a tampered page must never pass the attach gate"
    );
}
