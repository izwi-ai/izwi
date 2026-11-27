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
| Experts, as stored (14.6 GiB) | mapped file | mapped file | device, uploaded as stored (the mapped file if they run on the host) |
| Dense text tensors, F32 (6.0 GiB) | host | device (3.0 GiB with F16 projections on request) | device (3.0 GiB with BF16 projections on request) |
| Vision tower, F32 (1.2 GiB) | host | device (0.6 GiB with F16) | device (0.6 GiB with BF16) |
| Routers, F32 | host | host | host, and a device copy with the experts |
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
- on the CPU and Metal the experts run on the host from the mapped file
  (on Metal that costs one device-to-host read and one upload per layer
  application);
- on CUDA the experts run on the device (`kernels::moe_packed`): the packed
  bytes are uploaded as stored, routing and the expert products stay on the
  GPU, and the expert ids never reach the host. The load compares that path
  with the host path on one layer of each format and falls back to the host
  when they differ by more than 1e-3 relative. `IZWI_INKLING2_MOE_BACKEND`
  (`auto`, `host`, `device`) overrides the choice; the loaded-model
  diagnostics report the backend, the reason and the check's error;
- on Metal the same kernels exist (ported one for one, fast math off) and
  `IZWI_INKLING2_MOE_BACKEND=device` selects them, with the same load-time
  check. They are not the default there: the upload is a second copy of the
  14.6 GiB of experts in unified memory beside the mapped file, and one
  layer's experts take about 4 ms for one token at the published geometry
  on an M-series GPU (measured in a debug build), which has not been
  compared with the host path on the real model;
- on CUDA, with the experts on the device, a decode step replays its pure
  stages as CUDA graphs (`kernels::cuda::stretch_graphs`): two per layer
  application (before and after the attention core) and the final read, 97
  for the published model. Each graph is warmed, captured, checked bit for
  bit against the eager stage twice, then replayed; any failure turns the
  graphs off for the process. The attention core of every layer application
  stays eager, because it builds per-step index and mask tensors on the host
  and its shapes grow with the context. `IZWI_INKLING2_DECODE_GRAPHS=off`
  (or `IZWI_CUDA_DECODE_GRAPHS=off`) runs the stages eagerly; the diagnostics
  report warm-ups, captures, verifications and replays. Prefill is eager;
- rows of a batch run one after another.

Not implemented yet: a fused paged attention kernel with the position bias
and sink (which a whole-step graph needs: until then 48 attention cores per
token run eagerly), fused convolution kernels, and the MTP head.

## Dense weights in 16 bits

`precision.rs` decides the dtype of the large matrices (embedding,
unembedding, attention projections, the vision tower's linears) and holds
the one product they go through. Everything is F32 by default.
`IZWI_INKLING2_DENSE_DTYPE` asks for 16 bits on an accelerator:

- `f16` on Metal: the weights are converted at load (checked per tensor; one
  that does not fit stays F32), activations and accumulation stay F32
  through the mixed-dtype mat-vec kernel, and the numbers equal the F32
  path's up to summation order (logits within 1.4e-3 on the fixture, the
  same as F32 against the reference). Dense memory halves and a decode
  step's products are faster (97 against 166 microseconds for a layer
  projection on an M1 Pro). It is not the default because a prefill span
  wider than 16 rows widens each weight for the call, which makes that
  product about a quarter slower than with F32 weights; end-to-end prefill
  time on the real model has not been measured.
- `bf16` on CUDA: the stored bytes are uploaded as they are and each
  projection rounds its activations to BF16. That is the reference's own
  arithmetic, but on the random fixture it flips expert routing and moves
  logits by whole units, so it stays a request until a product of F32
  activations with a BF16 weight exists there (`Projection` is the place).
  Type-checked only.

Load admission prices the dense tensors at the dtype they will be held in.

## Releasing window pages

Forty of the 48 layer applications look back 1,024 tokens. When prefix
sharing is off, their keys and values are a state domain of their own
(`INKLING2_WINDOWED_STATE_DOMAIN`), all with one window, and the engine
releases that domain's pages once no query can reach them. A long
conversation then holds whole-history keys and values only for the eight
full-attention applications: about a sixth of the 192 KiB a token costs in
BF16 beyond the window. The model takes two caches (`state::Inkling2Caches`)
and routes each application to its cache; the engine passes the second one
where it passes another family's second domain.

Prefix sharing and page release do not combine. The engine shares snapshot
prefixes only for a contract with one paged domain, and frees window pages
only for a domain whose layers share one window. So the contract has one
shape or the other, chosen at load: prefix sharing where the catalog engages
it (CPU, Metal) or the operator forces it, page release otherwise (CUDA by
default, and anywhere with `enable_prefix_caching` off and catalog
selection off). Load-time arena sizing is not window-aware yet, so the gain
is pages left for other sequences, not a smaller arena.

## Prefix sharing

With prefix sharing on, the keys and values publish their committed pages
and the convolution ring a snapshot at each page. A prompt declares its
history boundary at the start of its last message (`history_boundary`): the
conversation's next turn renders everything before it again, so it attaches
those pages and the ring's snapshot instead of prefilling them. A forced
tool call or a JSON reply ends the prompt in a message the next turn does
not have, which is why the boundary is before the last message and not at
the prompt's end. A prompt with images shares nothing: two images' slots are
the same tokens.

It engages by the catalog's prefix-reuse cell: on the CPU and Metal, where
the attach test ran (an attached prefix prefills and decodes exactly as a
fresh run; on the CPU also through the runtime, where a second turn attaches
the first and replies as it does without sharing), and declared but off on
CUDA until the test runs on a GPU. `enable_prefix_caching` forces it on.

## Image input

`vision.rs` follows transformers' `Inkling2ImageProcessor` and
`Inkling2VisionModel`.

- An image is used at its own resolution: one image token per 40 x 40 pixels,
  rounded up per side. Transparency is composited over a checkerboard, and the
  bottom and right are padded to whole cells with a value just below black.
- The tower (23 layers, width 1024) embeds 20-pixel patches, adds a learned
  32 x 32 position grid resampled bilinearly to the image's patch grid, rotates
  queries and keys by patch row and column, and projects each cell's 2 x 2
  patches to one language-model embedding. Images are encoded one at a time;
  attention heads are taken a few at a time so the scores of a large image
  stay within 256 MiB.
- The prompt writes each image as its own message (`role`,
  `<|content_image|>`, one `<|unused_200054|>` slot per image token,
  `<|end_message|>`). The image's embeddings replace the slots' token
  embeddings after the embedding norm.
- The server marks an image's place in a message's text with
  `CHAT_IMAGE_PLACEHOLDER`; the prompt renderer splits user and tool messages
  on it. Fetching and decoding (data URL, guarded public URL, a file under
  `IZWI_MEDIA_LOCAL_ROOT`) is `models/shared/media`, shared with Qwen3.5.
- A prompt with images travels as a prepared artifact
  (`NativeChatPreparedPrompt::Inkling2`): its tokens and the image features on
  the host. The decode state keeps the features, so prefill in spans and a
  replay after suspension put them back in the same slots.

Two limits are this server's, not the model's: a request carries at most 4
images, and an image that would take more than 1,600 image tokens
(1,600 x 1,600 pixels) is scaled down, keeping its shape, until it fits. The
reference never resizes; the tower's attention is quadratic in the patch
count. The tower is loaded with the model in F32 (1.2 GiB as published) and
is part of the load's admission figures.

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
- Tool calls are constrained on the model's own tokens
  (`ToolCallSyntax::InklingMessage` in the catalog): the grammar starts at
  `<|content_invoke_tool_json|>` and describes `{"name":"NAME","args":{...}}`
  with each tool's `parameters` schema. A named `tool_choice` starts the
  sampler inside the arguments the prompt opened. With `required` the prompt
  cannot open the call (the header names the function before the opener), so
  the reply is held to a call and the body of the call the model opens to its
  schema.
- `response_format: json_object`: the prompt opens the model's text message
  with thinking off and the sampler holds it to one JSON object, then lets
  only the message and the turn end. The shared JSON state machine is not
  used here; it would constrain the message headers as well.

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
| The reference fixture against Transformers itself, every layer application and the logits | `generate_inkling2_golden.py --check-transformers`, Transformers 5.24.1 | passes: F64 logits within 1e-12 |
| Image cells, vision features and the logits of a prompt with two images against the reference fixture; prefill in spans that cut an image; replay with images | CPU test | passes |
| An image chat through the runtime: PNG data URLs, two requests at once, failures that leave the model serving | CPU test, synthetic checkpoint | passes |
| Prefix sharing: an attached prefix against a fresh run (CPU, and Metal with F16 keys), and a second turn through the runtime | CPU and Metal tests | passes |
| The split layout: same logits and tokens as one cache, and on a windowed view that has released its early pages; through the runtime, replies equal the single-domain run's | CPU test | passes |
| Tool calls and JSON replies held to their grammar on random weights | CPU test | passes |
| A separate worker process serving the synthetic checkpoint over HTTP and through the gateway | CPU process test | passes |
| Real checkpoint on any backend | a machine with 24 GB+ | not run |
| Device experts inside the served model, on the kernels' CPU reference: load check, logits, greedy tokens equal to the host path | CPU test | passes |
| The Metal expert kernels against the CPU reference (ids equal, outputs within 3e-7 relative, a token's result bit-identical alone and in a batch), and inside the served model with the load check | Metal tests, M-series Mac | passes |
| The CUDA expert kernels' source, run on the host one thread at a time against an independent evaluation | `scripts/cuda-host-emulation/moe_packed.cpp` | passes |
| CUDA build type check with a stand-in `nvcc` | this Mac | passes |
| Anything on a GPU | | not run |

The reference fixture (`tests/fixtures/inkling2_golden`) is a tiny random
model in the published layout whose activations come from a plain-PyTorch
transcription of the Transformers forward pass. The transcription was checked
against Transformers 5.24.1 on that checkpoint: run in F64 the two agree to
1e-12 on the logits and 6e-11 on every layer application. In F32 they differ
by 2e-3 on the logits, which is rounding that the tiny random model amplifies
(Transformers' own F32 run is 1e-3 from its F64 one). Transformers takes the
full-attention layers from `layer_types`, so the fixture's config carries it
next to `local_layer_ids`; without it every sixth layer is a full one, as
published.
