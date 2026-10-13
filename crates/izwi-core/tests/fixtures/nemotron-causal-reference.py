"""Regenerate nemotron-causal-reference.json.

Ports, with torch 2.9.1, the NeMo code izwi's Nemotron 3.5 ASR must match
(NVIDIA/NeMo main, fetched 2026-10-13):
  * subsampling.py ConvSubsampling(dw_striding, is_causal=True) forward through
    MaskedConvSequential._forward_torch, with causal_convs.py CausalConv2D
    (F.pad(x, (2, 1, 2, 1)) then a padding-0 convolution);
  * conformer_encoder.py ConformerEncoder._create_masks, att_context_style
    "chunked_limited".
Weights and inputs come from a counter hash that izwi's test reproduces
bit-exactly, at the checkpoint's real shapes (256 channels, out [1024, 4352]).

    python -I crates/izwi-core/tests/fixtures/nemotron-causal-reference.py \
        crates/izwi-core/tests/fixtures/nemotron-causal-reference.json
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F


def hashed(seed, count, scale):
    i = np.arange(count, dtype=np.uint64)
    h = (i * 0x9E3779B1 + seed * 0x85EBCA77) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    v = (h >> 8).astype(np.float32) * np.float32(2.0 / 16777216.0) - np.float32(1.0)
    return torch.from_numpy(v * np.float32(scale))


C, D, MELS = 256, 1024, 128
W = {
    "conv.0.weight": hashed(1, C * 9, 0.5).reshape(C, 1, 3, 3),
    "conv.0.bias": hashed(2, C, 0.125),
    "conv.2.weight": hashed(3, C * 9, 0.5).reshape(C, 1, 3, 3),
    "conv.2.bias": hashed(4, C, 0.125),
    "conv.3.weight": hashed(5, C * C, 1 / 16).reshape(C, C, 1, 1),
    "conv.3.bias": hashed(6, C, 0.125),
    "conv.5.weight": hashed(7, C * 9, 0.5).reshape(C, 1, 3, 3),
    "conv.5.bias": hashed(8, C, 0.125),
    "conv.6.weight": hashed(9, C * C, 1 / 16).reshape(C, C, 1, 1),
    "conv.6.bias": hashed(10, C, 0.125),
    "out.weight": hashed(11, D * C * 17, 1 / 64).reshape(D, C * 17),
    "out.bias": hashed(12, D, 0.125),
}
FRAMES, VALID = 37, 35
# (batch, time, mels); frames past VALID hold values NeMo's mask must hide.
features = hashed(13, FRAMES * MELS, 2.0).reshape(1, FRAMES, MELS)


def causal(x, w, b, groups):
    return F.conv2d(F.pad(x, (2, 1, 2, 1)), w, b, stride=2, groups=groups)


def time_mask(x, length):
    keep = (torch.arange(x.shape[2]) < length).to(x.dtype)
    return x * keep[None, None, :, None]


layers = [
    ("strided", lambda x: causal(x, W["conv.0.weight"], W["conv.0.bias"], 1)),
    ("act", torch.relu),
    ("strided", lambda x: causal(x, W["conv.2.weight"], W["conv.2.bias"], C)),
    ("point", lambda x: F.conv2d(x, W["conv.3.weight"], W["conv.3.bias"])),
    ("act", torch.relu),
    ("strided", lambda x: causal(x, W["conv.5.weight"], W["conv.5.bias"], C)),
    ("point", lambda x: F.conv2d(x, W["conv.6.weight"], W["conv.6.bias"])),
    ("act", torch.relu),
]
with torch.no_grad():
    x, length = features.unsqueeze(1), VALID
    for kind, layer in layers:
        x = layer(time_mask(x, length))
        if kind == "strided":
            length = (length + 3 - 3) // 2 + 1
    x = time_mask(x, length)
    b, c, t, f = x.shape
    assert f == 17, f
    out = (x.transpose(1, 2).reshape(b, t, c * f) @ W["out.weight"].T + W["out.bias"])[0, :length]
    print("pre_encode frames", t, "valid", length, "bins", f)


def chunked_limited(max_len, left, right):
    att_mask = torch.ones(1, max_len, max_len, dtype=torch.bool)
    chunk_size = right + 1
    left_chunks_num = left // chunk_size if left >= 0 else 10000
    chunk_idx = torch.div(torch.arange(0, max_len, dtype=torch.int), chunk_size, rounding_mode="trunc")
    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
    chunked = torch.logical_and(torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0))
    return torch.logical_and(att_mask, chunked.unsqueeze(0))[0]


masks = {}
MASK_LEN = 150
for left, right in [(56, 13), (56, 6), (56, 3), (56, 1), (56, 0)]:
    m = chunked_limited(MASK_LEN, left, right)
    rows = []
    for q in range(MASK_LEN):
        visible = torch.nonzero(m[q]).flatten().tolist()
        assert visible == list(range(visible[0], visible[-1] + 1))
        rows.append([visible[0], visible[-1]])
    masks[f"{left},{right}"] = rows

cols = list(range(0, D, 8))
json.dump(
    {
        "source": "NeMo ConvSubsampling(dw_striding, causal) + ConformerEncoder._create_masks(chunked_limited), ported to torch 2.9.1",
        "subsampling": {
            "feature_frames": FRAMES,
            "valid_frames": VALID,
            "encoded_frames": int(length),
            "columns": cols,
            "rows": [[round(float(out[r, c_]), 6) for c_ in cols] for r in range(length)],
            "row_sums": [float(out[r].double().sum()) for r in range(length)],
            "row_sq_sums": [float((out[r].double() ** 2).sum()) for r in range(length)],
        },
        "mask_len": MASK_LEN,
        "masks": masks,
    },
    open(sys.argv[1], "w"),
    separators=(",", ":"),
)
