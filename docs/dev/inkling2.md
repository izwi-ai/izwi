# Inkling 2

izwi serves Thinking Machines' Inkling 2 Mini as a text chat model on the
CPU, Metal and CUDA. It is its own model family (`inkling2`): a looped
sparse-expert decoder whose expert weights are published quantized and stay
packed in memory. It imports no other family;
`the_family_imports_no_other_architecture` holds that line.

User-facing behaviour is in the [models guide](../user/models/index.md) and
the [chat guide](../user/features/chat.md).

| Variant | Files | Family |
| --- | --- | --- |
| `Inkling-2-Mini` | one 19.5 GB weight file (NVFP4 and MXFP8 experts, BF16 dense tensors) | `Inkling2Chat` |

## Pinned checkpoint

| Repository | Revision | Licence |
| --- | --- | --- |
| `thinkingmachines/Inkling-2-Mini` | `c9d028302cd0f035fdf8e7e91f089a4ce94d2195` | Apache-2.0 |

`tests/fixtures/inkling2/` holds the published configs, the chat template
and the census of all 1,298 tensors of both safetensors files, read from
their headers over HTTP range requests
(`scripts/fixtures/fetch_inkling2_slices.py`). The downloader fetches the
weight file at its exact size and the configs. `mtp.safetensors` (the
multi-token-prediction head) is not fetched: nothing loads it.

## The model

The reference is `models/inkling2_loop` in Hugging Face Transformers.

- **Loop.** 24 layers of weights. Layers 6 to 17 run three times with the
  same weights, so a token passes through 48 layer applications. Passes
  after the first restart from the state before the loop and read
  "messages" from the previous pass through learned queries and low-rank
  tanh gates. Every pass has its own keys, values and convolution state.
- **Depth residual.** There is no plain residual stream. Layers are grouped
  in blocks of six; the input of each attention and each MLP is an attention
  over the block inputs so far with a learned query. This is per position:
  nothing is carried across tokens.
- **Attention.** 24 heads over 8 key/value heads. Layers 5, 11, 17 and 23
  attend to everything; the other 20 see a 1,024-token window. There is no
  rotary embedding: each query adds a bias computed from its own hidden
  state and a trained bank of bias-by-distance profiles, and a sink logit
  that only absorbs probability. Queries and keys are L2-clipped, values are
  RMS-normalised, and the output is gated by a sigmoid of the layer input.
- **Short convolutions.** Keys, values, the attention output and the MLP
  output each pass a four-tap causal depthwise convolution. On the
  full-attention layers the key and value taps are static weights plus
  per-token weights.
- **Experts.** 128 experts, 8 per token. Scores are sigmoids; the choice
  adds a bias, the weights do not. The activation is a soft-capped SwiGLU.

## Modules

| Module | What it holds |
| --- | --- |
| `config` | `config.json` and the derived geometry; settings the trunk has no code for fail the load |
| `checkpoint` | the memory-mapped weight file; its census is derived from the config and equals the published header |
| `quant` | NVFP4 and MXFP8, decoded and multiplied straight from the packed bytes |
| `experts` | routing and the expert feed-forward on the host |
| `trunk` | the trunk composed from plain tensor operations; runs on its own dense state or on engine state through `Inkling2Context` |
| `state` | the engine state contract: 48 paged attention layers (40 windowed) and a ring of the four convolutions' last three inputs per layer application |
| `chat` | the served model: prefill spans, decode steps, rollback, replay |
| `prompt`, `reply` | the chat template by hand, and the model's reply messages as the text the server reads |
| `memory` | the footprint from the header, for load admission |

### Checkpoint conventions

Each has a test, and breaking each fails the parity test:

- `experts.w13_weight` rows interleave gate and up (`gate0, up0, gate1, …`).
- NVFP4 packs two 4-bit codes per byte, the even element in the low nibble,
  with one FP8 scale per 16 values and one `amax / outer_scale_denominator`
  factor per expert. MXFP8 has one power-of-two scale per 32 values.
- `attn_res.attn_queries.{i}` belongs to layer `i + 1`.
- The two `middle_cycle_cross_pass_*` tensors hold (query, mlp) and (key,
  value) message queries; messages are used as query, key, value, mlp.
- `unembed.weight` is padded to the embedding's rows; 200,058 are used.
- `k_sconv.static_w` is indexed by backward distance.

## What a loaded model holds

| Part | CPU | Metal | CUDA |
| --- | --- | --- | --- |
| Experts, as stored (14.6 GiB) | mapped file | mapped file | mapped file (host) |
| Dense text tensors, F32 (6.0 GiB) | host | device | device |
| Routers, F32 | host | host | host |
| Keys and values | F32 | F16 | BF16 |

F16 keys and values on Metal move the tiny random fixture's logits by about
0.1 (it routes experts on near-ties, so its greedy tokens then differ). That
has not been measured on the real checkpoint; `state::kv_dtype` is the one
place to change if it matters there.

Load admission prices this from the checkpoint header before a weight is
read (`runtime/lifecycle/inkling2_memory.rs`). Each token a conversation
holds costs keys and values in all 48 layer applications (192 KiB in BF16),
so the served context is capped at 32,768 tokens
(`chat::DEFAULT_MAX_CONTEXT_TOKENS`); the checkpoint's own limit is 1M.

## Serving path today

The served path is the composed trunk on every backend:

- activations are F32;
- attention is composed from tensor operations over the context gathered
  from the paged cache (`KvArena::gather_context`), because the paged
  attention kernels take no additive bias or sink;
- the experts run on the host from the mapped file, also on Metal and CUDA,
  which costs one device-to-host read and one upload per layer application;
- rows of a batch run one after another.

Not implemented yet: fused attention, convolution and expert kernels, CUDA
graphs, dense tensors in the checkpoint's BF16, image input (the vision
tower is not loaded), prefix sharing, releasing windowed pages, the MTP
head, and grammar-constrained tool calls and JSON output.

## Chat surface

- The prompt (`prompt.rs`) is compared byte for byte with conversations
  rendered by the published template (`template_cases.json`, written by
  `scripts/fixtures/generate_inkling2_template_goldens.py`).
- Thinking effort: `Xhigh`, `Medium` and `Low` map to 0.99, 0.7 and 0.2;
  thinking switched off is effort 0; unset is the template's 0.9.
- The model answers in messages (thinking, text, tool calls). `reply.rs`
  writes thinking as a `<think>` block and each call as a Hermes
  `<tool_call>` block, which the server already parses.
- Sampling defaults to the request's own settings; the published
  `generation_config.json` samples at temperature 1.0.

## Verification

| What | Where | Status |
| --- | --- | --- |
| Census equals the published header | CPU test | passes |
| Published expert rows decode bit for bit | CPU test, `real_slices.safetensors` | passes |
| Trunk activations and logits against the reference fixture, in one span and token by token | CPU test | passes |
| Serving on the paged cache: prefill logits, greedy decode, rollback, replay | CPU test | passes |
| Load, chat alone and concurrently, unload through the runtime | CPU test, synthetic checkpoint | passes |
| Prompt against the published template | CPU test | passes |
| Trunk on the Metal device; serving on a Metal cache with F32 keys (logit error 1.4e-4, greedy tokens equal) and with F16 keys as served (logit error 0.10 on the random fixture) | Metal tests, M-series Mac | passes |
| The reference fixture against Transformers itself | `generate_inkling2_golden.py --check-transformers` | not run: needs a Transformers release with `inkling2_loop` |
| Real checkpoint on any backend | a machine with 24 GB+ | not run |
| CUDA build and run | | not run |

The reference fixture (`tests/fixtures/inkling2_golden`) is a tiny random
model in the published layout whose activations come from a plain-PyTorch
transcription of the Transformers forward pass. Until that transcription is
checked against Transformers, parity is with the transcription.
