# Understanding LLM Inference: A Journey Through the Izwi Audio Engine

## Introduction

If you're new to LLM inference, this guide will walk you through how a prompt transforms into output by following its journey through the inference engine. We'll use the Izwi Audio engine as our example—a production-grade inference system designed for audio AI workloads like text-to-speech (TTS), automatic speech recognition (ASR), and voice chat.

## The Big Picture: What Happens When You Send a Prompt?

When you send a request to generate speech from text (or transcribe audio to text), your prompt doesn't just go straight to a model. It travels through several stages, each with a specific job:

```
Your Request → Runtime Layer → Engine Core → Scheduler → Executor → Model → Output
```

Let's follow this journey step by step.

---

## Stage 1: Request Arrival and Validation (The Runtime Layer)

### What Happens Here

When your request first arrives—whether through the CLI, HTTP API, or desktop app—it enters the **Runtime Service**. Think of this as the reception desk of a busy office.

The Runtime Service does several things:

1. **Validates your request**: Checks that you provided valid inputs (text for TTS, audio for ASR, etc.)
2. **Routes to the right model**: Determines which model family and variant should handle your request (Qwen3-TTS, Kokoro-82M, Qwen3-ASR, etc.)
3. **Sets up streaming**: If you want real-time output, it prepares the channels for delivering chunks as they're generated
4. **Tracks telemetry**: Records metrics like queue wait time and processing latency

### Key Concept: Request Lifecycle

A request isn't just data—it's a living object that carries state through the system. It includes:
- Your input (text, audio, chat messages)
- Generation parameters (temperature, max tokens, voice settings)
- Priority level (normal, high, critical)
- A unique ID for tracking

### The Handoff

Once validated, the Runtime Service hands your request to the **Engine**—the heart of the inference system.

---

## Stage 2: The Engine - Your Request Joins the Queue

### What is the Engine?

The Engine is the main orchestrator. It manages the entire inference pipeline and ensures fair access to the GPU/CPU for all concurrent requests. It's like an air traffic controller for model inference.

### Request States

Your request enters the Engine and immediately goes into one of two states:

1. **Waiting**: Your request is in line, waiting for resources
2. **Running**: Your request is actively being processed

### The Continuous Loop

The Engine runs a continuous loop (often called the "inference loop") that:
1. Checks for new requests
2. Decides which requests to process this "step"
3. Runs the model forward pass
4. Handles the output
5. Repeats

This loop runs dozens or hundreds of times per second, processing requests in small chunks rather than one at a time.

---

## Stage 3: The Scheduler - Deciding What to Process

### The Scheduling Challenge

Imagine you're a chef in a busy kitchen. You have multiple orders (requests), limited stove space (GPU memory), and each dish takes different amounts of time. How do you decide what to cook next?

This is the Scheduler's job. Every "step" of the inference loop, the Scheduler must decide:

- Which requests get processed this step?
- Should we run "prefill" or "decode" for each request?
- How many tokens should we process?

### Key Concept: Prefill vs. Decode

This is one of the most important concepts in LLM inference. Processing happens in two distinct phases:

#### Prefill Phase

**What it does**: Processes your entire input prompt at once.

**Analogy**: Reading and understanding a book before discussing it.

**Characteristics**:
- Happens once per request, at the beginning
- Processes all prompt tokens in parallel
- Computationally intensive (matrix multiplications on the full sequence)
- Populates the KV cache with context
- Must complete before any output can be generated

**Example**: For TTS, this is where the model reads your entire text input and builds an internal representation.

#### Decode Phase

**What it does**: Generates output tokens one at a time, autoregressively.

**Analogy**: Writing a story one word at a time, where each new word depends on everything written before.

**Characteristics**:
- Happens repeatedly until generation is complete
- Processes only ONE new token per step (per request)
- Uses cached information from previous tokens (KV cache)
- Less computationally intensive per step, but many steps
- Latency-sensitive—users want each token quickly

**Example**: For TTS, this is where the model generates audio samples frame by frame.

### Scheduling Policies

The Scheduler can use different strategies:

1. **FCFS (First-Come, First-Served)**: Simple queue. Fair but can cause head-of-line blocking if a long request arrives first.

2. **Priority-Based**: Higher priority requests jump ahead. Critical for real-time voice applications where interruptions need immediate response.

3. **Adaptive Batching**: Dynamically adjusts how many tokens to process based on current load and latency targets.

### Chunked Prefill

For very long prompts, the Scheduler might break prefill into chunks. This prevents one long request from blocking all others.

---

## Stage 4: The KV Cache Manager - Memory Architecture

### The Problem: Attention is Expensive

Transformer models use "attention" to relate different tokens to each other. Without optimization, each new token would need to re-compute attention with ALL previous tokens—making generation slower and slower as output grows.

### The Solution: KV Caching

The KV (Key-Value) cache stores intermediate results from previous tokens so they don't need to be recomputed. This is like keeping your notes open while writing instead of re-reading the entire book for each new paragraph.

### Paged Attention: Memory Efficiency

Izwi uses "paged attention" (inspired by vLLM), which divides the KV cache into fixed-size **blocks**:

```
Request 1: [Block 0][Block 1][Block 2]     
Request 2: [Block 3][Block 1]  ← Shares Block 1 with Request 1!
Request 3: [Block 4][Block 5]
```

**Benefits**:
- **No wasted memory**: Unlike contiguous allocation, we only allocate what we need
- **Shared prefixes**: Multiple requests with the same prompt start can share blocks
- **Copy-on-Write**: When a shared block needs modification, it's copied only then
- **Efficient eviction**: Individual blocks can be freed when done

### Block Allocator

The KV Cache Manager maintains a pool of blocks:
- **Free List**: Available blocks ready for allocation
- **Allocated Blocks**: Currently in use by active requests
- **Reference Counting**: Tracks how many requests share each block

---

## Stage 5: The Executor - Running the Model

### What is the Executor?

The Executor is where the actual neural network computation happens. It takes the scheduled requests and runs them through the model.

### Two Execution Paths

The Executor has separate paths for prefill and decode:

#### Prefill Execution

1. Tokenizes the input prompt into token IDs
2. Runs the full transformer forward pass on all tokens at once
3. Computes attention across the entire sequence
4. Stores K and V values in the KV cache blocks
5. Returns the last token's hidden state for the first decode step

#### Decode Execution

1. Takes only the NEW token(s) to generate (usually just 1)
2. Runs a partial forward pass using cached K/V from previous steps
3. Performs "paged decode attention"—efficient attention using the cached blocks
4. Samples the next token from the output distribution
5. Appends new K/V to the cache
6. Returns the generated token

### Model State Management

The Executor maintains state for each active request:
- **Chat Decode State**: For text generation, tracks conversation history
- **TTS Decode State**: For speech synthesis, tracks audio frames generated
- **ASR Decode State**: For transcription, tracks recognized text

This state allows requests to be paused and resumed (for preemption) and enables streaming output.

### Parallel Execution

On CPU, multiple requests can run in parallel across threads. On Metal (Apple Silicon), execution is typically serialized to avoid GPU command queue contention, though this is configurable.

---

## Stage 6: The Model - Neural Network Forward Pass

### Architecture Overview

Izwi supports multiple model families (Qwen3, LFM2, Gemma). While architectures differ, they share common patterns:

#### Transformer Layers

Models consist of multiple identical layers (e.g., 24 layers). Each layer has:

1. **Self-Attention**: Relates tokens to each other
   - Query (Q): "What am I looking for?"
   - Key (K): "What do I contain?"
   - Value (V): "What information do I provide?"

2. **Feed-Forward Network**: Transforms representations

3. **Layer Normalization**: Stabilizes training/inference

### Attention Mechanisms

#### Full Attention (Prefill)

During prefill, the model computes attention across the entire sequence:

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

This creates a complete attention matrix showing how each token relates to every other token.

#### Paged Decode Attention

During decode, we only have one new query token but many cached K/V tokens. The paged attention implementation:

1. Iterates through cached pages incrementally
2. Uses "online softmax" (FlashAttention-style) to avoid materializing the full attention matrix
3. Accumulates attention scores across pages
4. Handles both dense and quantized (Int8) cache pages

### Grouped Query Attention (GQA)

Many modern models use GQA to reduce memory bandwidth:
- Traditional: Every query head has its own K/V heads
- GQA: Multiple query heads share the same K/V heads
- This reduces the KV cache size and memory bandwidth requirements

### RoPE (Rotary Position Embedding)

Models need to know token positions. RoPE encodes position information by rotating query and key vectors based on their position in the sequence. This allows the model to understand "token A comes before token B."

---

## Stage 7: Output Processing and Streaming

### From Tokens to Output

The model generates tokens, but users want text or audio:

1. **Text Generation**: Tokens are detokenized back into text
2. **TTS**: Tokens represent audio codec codes, which are decoded into waveform samples
3. **ASR**: Tokens represent text, detokenized into the final transcript

### Streaming Output

For real-time applications, output is streamed chunk by chunk:

1. As soon as the first decode step completes, the first chunk is sent
2. Each subsequent decode step generates more output
3. Chunks flow through channels back to the user
4. This creates the experience of "real-time" generation

### Completion Detection

The engine stops when:
- Maximum tokens reached (user-specified limit)
- End-of-sequence token generated
- Stop sequence encountered
- User manually cancels

---

## Putting It All Together: A Complete Example

Let's trace a TTS request: **"Hello, how are you today?"**

### Step-by-Step Journey

1. **Request Arrival**
   - User sends: `POST /v1/audio/speech` with text
   - Runtime validates: Text is present, voice is valid
   - Creates `EngineCoreRequest` with TTS task type

2. **Engine Reception**
   - Request added to Engine with unique ID
   - Goes to Scheduler's waiting queue
   - Telemetry records arrival time

3. **First Scheduling Step**
   - Scheduler selects this request for prefill
   - Allocates KV cache blocks (e.g., 2 blocks for prompt)
   - No decode scheduled yet (prefill must complete first)

4. **Prefill Execution**
   - Executor tokenizes: "Hello, how are you today?" → [15496, 11, 703, 389, 345, 30]
   - Runs forward pass through all 24 transformer layers
   - Computes attention over full 6-token sequence
   - Stores K/V tensors in allocated blocks
   - Returns hidden state for last token

5. **Second Scheduling Step**
   - Scheduler: prefill complete, now schedule decode
   - Request moves from "prefill" to "decode" queue
   - Allocates additional block for new tokens

6. **Decode Execution (Repeated)**
   - Step 1: Generate first audio frame token
     - Input: Last token's hidden state
     - Run attention using cached K/V + new query
     - Sample next token from output distribution
     - Store new K/V in cache
     - Convert token to audio samples
     - Stream first chunk to user
   
   - Step 2-50: Continue generating frames
     - Each step uses growing KV cache
     - Each step produces ~12-20ms of audio
     - User hears audio in real-time

7. **Completion**
   - Generation reaches natural pause or max length
   - Stop token detected
   - Final output chunk sent
   - KV cache blocks freed
   - Request marked complete
   - Telemetry records total time

### Latency Breakdown

The user experiences:
- **Queue Wait**: Time waiting for other requests (0-100ms)
- **Time to First Token (TTFT)**: Prefill time (50-200ms)
- **Time Per Output Token (TPOT)**: Decode time per step (10-50ms)
- **Total Time**: Sum of all decode steps

---

## Key Architectural Patterns

### 1. Separation of Concerns

Each component has a focused responsibility:
- **Runtime**: API-facing, request lifecycle
- **Engine**: Orchestration, loop management
- **Scheduler**: Resource allocation decisions
- **KV Cache**: Memory management
- **Executor**: Model computation
- **Model**: Neural network logic

### 2. Continuous Batching

Unlike batch systems that wait to fill a batch, the engine:
- Processes work every step (many times per second)
- Dynamically mixes prefill and decode work
- Adapts to arriving and completing requests
- Maximizes GPU utilization

### 3. Streaming-First Design

The entire pipeline supports streaming:
- Results flow through async channels
- No waiting for complete generation
- Ideal for real-time audio applications

### 4. Memory-Efficient Caching

Paged KV cache enables:
- Higher batch sizes (more concurrent requests)
- Longer sequences (more context)
- Memory sharing (common prefixes)
- Fine-grained memory management

---

## Performance Considerations

### What Makes Inference Fast or Slow?

1. **Prefill Time**: Depends on prompt length. Longer prompts = longer prefill.

2. **Decode Throughput**: Depends on model size and batch size. More concurrent requests = better GPU utilization (up to a point).

3. **Memory Bandwidth**: Often the bottleneck. Reading/writing the KV cache dominates compute time for large models.

4. **Attention Computation**: Scales quadratically with sequence length during prefill, but linearly during decode (thanks to caching).

### Optimization Strategies

1. **Chunked Prefill**: Break long prompts into chunks to reduce latency spikes.

2. **Priority Scheduling**: Ensure critical requests (like voice interruption) get processed first.

3. **KV Cache Quantization**: Store cache in Int8 instead of Float16 to reduce memory by 50%.

4. **Flash Attention**: Use optimized kernels that fuse attention operations and reduce memory reads.

5. **Prefix Caching**: Cache common prompt beginnings (system prompts, conversation starters) for reuse.

---

## Summary

Understanding LLM inference means understanding how a prompt travels through multiple stages:

1. **Runtime** validates and routes your request
2. **Engine** orchestrates the inference loop
3. **Scheduler** decides when to prefill vs decode
4. **KV Cache** stores and manages intermediate results
5. **Executor** runs the neural network
6. **Model** performs attention and transformations
7. **Output** flows back through the pipeline

The key insight is that inference is not a single operation but a continuous process of **prefill** (understanding input) followed by many **decode** steps (generating output), with careful memory management to make it efficient.

For audio AI specifically, this architecture enables real-time speech synthesis and recognition by streaming output as it's generated, while efficiently managing the memory needed to maintain conversation context.

---

## Further Reading

To dive deeper into specific areas:

- **vLLM Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **Transformer Architecture**: "Attention Is All You Need"
- **Continuous Batching**: Research on "Orca" and "Splitwise" serving systems

The Izwi Audio engine implements these research advances in a production system optimized for audio AI workloads on Apple Silicon and CPUs.
