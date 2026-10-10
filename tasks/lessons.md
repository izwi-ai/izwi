# Lessons

- When the user narrows a planning scope by saying to skip a workstream, remove that workstream from implementation phases instead of carrying it forward as planned work; keep prior analysis only as explicitly out-of-scope context when useful.
- When closing review findings, turn documented runtime contracts into executable checks: readiness warnings must affect readiness status, and structured logging promises must be verified against the actual emitted JSON shape rather than inferred from span construction.
- When a user says Linux installers, cover every shipped Linux surface (`.deb`, AppImage/updater, terminal tarball, and future Linux bundle paths) unless they explicitly narrow to Ubuntu.
- When a user says one installer or binary must support CPU and CUDA, preserve the public binary names unless they explicitly approve renamed CUDA commands; use private runtime layout or loader work, not public `*-cuda` names.
- When requirements include hardware support scope (for example Docker-only vs all distributions), treat scope as a hard contract and reconfirm before finalizing implementation plans.
- For CUDA-related fixes, always align runtime behavior, packaging/release workflow, Docker builds, CLI reporting, and docs together to avoid "CUDA installed but not actually enabled" confusion.
- When introducing lower-precision activation policies around Candle `QMatMul`, verify the actual weight expansion path first; some GGUF tensors materialize as dense `F32`, so projection helpers must normalize input dtypes before inference is considered fixed.
- When lowering Gemma4 activations to `F16`, audit `RmsNorm` boundaries too; Candle's quantized norm weights dequantize to `F32`, and the norm kernel rejects mixed `F16`/`F32` inputs unless we promote the activation first.
- When building attention masks with shape `[1, 1, q, k]` for reuse across heads, always apply them with the broadcast-aware tensor ops; plain `+` will fail against score tensors shaped `[batch, heads, q, k]`.
- When using temporary debug harnesses to diagnose model regressions, keep them untracked and remove them before staging; never assume the user wants debugging files committed.
- For chat decode bugs that emit role/control artifacts (for example empty text or `"user"`), do not resample from the same logits after rejecting a token. Consume the sampled token into model state first (llama.cpp-style sample/accept/process), then decide whether to surface it to users.
- When a user reports "still broken" on streaming, always verify both non-stream and SSE `/messages` paths against a locally started patched server before claiming the fix is complete.
- When startup appears frozen on model loading, verify crash vs hang first (PID stability + DiagnosticReports), then remove network-dependent metadata calls from the startup critical path.
- To fix Gemma4 returning only `"user"`, keep generation stateful: sampled hidden control/role-prefix tokens must advance the KV/cache position even when they are suppressed from output.
- For the Gemma4 `"user"` output bug, keep the fix scoped to `gemma4/chat.rs`: consume hidden control/role-prefix tokens by advancing decode state, do not resample the same logits, and avoid unrelated subsystem edits.
- On Apple Metal, Gemma4 `F16` compute can yield all-NaN prefill logits; default Gemma4 compute dtype to `F32` on Metal unless explicitly overridden.
- If Gemma4 emits repeated hidden prefix markers (`<turn|>`, `<|turn>`, channel markers) with no visible output, block hidden IDs during prefix resampling to force a visible token candidate before giving up.
- For UI redesign follow-ups, treat user visual corrections as strict constraints: remove decorative container treatments when asked, preserve the existing color system, and enforce explicit text behavior requests (for example `whitespace-nowrap` on required single-line labels).
- For onboarding completion screens, bias toward a simple single-column structure; avoid introducing extra panel splits when the user asks for a cleaner, lighter finish.
- For copy-only completion lists, prefer plain text rows over boxed treatments unless grouping materially improves scan speed.
- When rendering row metadata like model size, use an explicit trailing column (`grid-cols-[minmax(0,1fr)_auto]` + `whitespace-nowrap`) instead of wrap-enabled flex so right-aligned values never fall below long copy.
- After adding cursor pagination to history/library tables, validate bottom breathing room on every affected route; if the final block is flush with the viewport, add explicit route shell bottom padding instead of relying on last-child margins.
- If bottom spacing changes on route pages appear to have no effect, inspect shared wrapper structure first: child margins can collapse inside non-flex page containers, so fix layout semantics (`section`/`header` + flex column) before stacking route-specific spacing overrides.
- When TTFT is the primary target, prioritize prefill-path compute reductions first; decode-loop and stream-chunk optimizations mostly affect TPS/end-to-end and are unlikely to move first-token latency.
- For market-research follow-ups on Izwi, treat user requests for validation as potentially commercialization-oriented by default; explicitly analyze packaging, monetization, and the open-source-to-paid conversion path rather than stopping at market trends and competitors.
- For Whisper streaming QA, always validate both the SSE final transcript and non-stream transcript on a long real file (not only `fox.wav`); chunked streaming can look fine on short clips while still producing overlap-duplication regressions.
- For Whisper non-incremental streaming, avoid chunking audio that already fits the model context window unless explicitly overridden; overlap merge can degrade transcript quality and add avoidable latency.
- For long-form ASR, avoid hard wall-clock timeout wrappers around active transcription generation; global request timeout values are not reliable proxies for valid transcription completion time.
- When a model migration request says "only support X" for a specific route (for example diarization ASR), remove legacy model visibility from route model filters/preferences and also enforce the same constraint in runtime/store normalization so hidden fallbacks cannot persist.
- When the user asks to match a newer modal readiness UX, replicate the exact interaction pattern (single full-width action + discrete status labels) and remove verbose progress/count copy rather than preserving previous explanatory text.
- For UI alignment corrections, prefer changing the row container semantics (`justify-between` + content order) instead of adding ad-hoc spacing utilities so "left label / right status" remains stable across breakpoints.
- When extending shared primitives like `TabsTrigger`, audit inherited layout utilities such as `whitespace-nowrap`; multiline tab or rail content must explicitly opt back into wrapping and responsive stacking.
- For left-rail tab navigation in dense settings modals, avoid inline status pills unless they are essential to navigation; they add clutter quickly and can destabilize vertical rhythm on narrow widths.
- In dense settings modals, remove explanatory sidebars and overview sections when the user asks for a cleaner surface; keep only controls or context that directly supports the next action.
- For operational model-management tabs, prefer one bulk action and a plain status list over repeating per-model action buttons and explanatory copy when the stack is fixed and predictable.
- When simplifying grouped model-management UI, keep one action surface per model group if the user still expects that group to own loading/unloading; reduce duplication, but do not collapse away functional ownership.
- In compact model inventory rows, do not repeat the raw variant string under an already clear formatted model label unless the user explicitly asks for technical identifiers.
- In editor-style tabs, avoid dedicating a second column to generic writing guidance when the user asks for a cleaner modal; keep status badges inline with the primary editor header instead.
- In compact setup modals, remove secondary tuning panels like playback controls when the user asks to strip the surface back; do not preserve them out of habit if they are not central to setup.
- For simplified left-rail tabs, keep a faint inactive border so non-active items still read as clickable navigation rather than plain text.
- When landing support for a new model family, the user expects it visible and
  usable by default. An internal activation-gating posture (catalog-disabled
  until hardware evidence) hides the model from every surface — CLI list,
  desktop models/chat lists, download paths — and reads as "not supported".
  Gate certification CLAIMS, not VISIBILITY: enable the catalog, keep the
  load-admission math truthful per backend, and record what remains
  uncertified in the handoff doc. When a dedicated-family plan chooses
  default-off visibility, surface that tradeoff to the user explicitly at
  ship time instead of assuming the earlier gating decision carries over.
- A CI gate whose legs have never all executed is an UNVERIFIED gate: earlier
  failing gates masked hygiene's clippy leg for the branch's entire life, and
  when the gate finally reached clippy it failed on ~50 latent violations.
  Before pushing, run the ENTIRE lane locally (or mirror it), not just the
  legs you changed. Also: `cargo clippy --fix` records lint results in
  cargo's cache WITHOUT -D warnings, so a follow-up `cargo clippy -- -D
  warnings` can reuse them and hide remaining errors — touch sources (or
  clean) between fix passes and verification runs.
- Container-image CI lanes that check out the whole repo inherit workspace
  `.cargo/config.toml` (here: a python3 rustc-wrapper) — a minimal image
  without the interpreter kills every cargo invocation at startup, faster
  than any compile error. When one lane dies instantly while identical code
  compiles green elsewhere, diff the three environments (runner / container
  with checkout / Dockerfile without config) before reading code.
- "Linux-only" CI failures with no log access may be TOOLCHAIN drift: CI's
  floating `stable` can be several releases ahead of a stale local rustup.
  Install the CI version locally as a secondary toolchain
  (`rustup toolchain install <ver>` + `cargo +<ver> clippy ...`) before
  assuming platform-specific code. Corollary: `clippy::incompatible_msrv`
  guards both directions — never adopt renamed std APIs (fetch_update ->
  try_update, stable 1.95) while the workspace MSRV and the Dockerfile
  builder pin an older toolchain (1.88); exempt the deprecation group in the
  gate instead (-A deprecated, with the MSRV justification in a comment).
- A green CI lane can be a CACHE-SKIP FALSE GREEN: cargo caches lint
  results, so a restored cache can skip re-linting unchanged-but-dirty
  crates (run 194 hygiene passed in 3m01s right after failing 3 runs in a
  row). After lint-affecting changes, trust only runs that actually
  re-linted (check the step duration against a cold baseline) or force it
  locally with touch/clean.
- When CI logs are auth-gated but the repo is public, check-run ANNOTATIONS
  are still readable via the API — and a workflow can write arbitrary
  content into them with `::error title=...::<url-encoded payload>` steps.
  Tee the failing lane's output and re-emit its error lines as annotations;
  each push iteration costs ~4 minutes.
- A docker mirror of a CI lane needs `rustup component add clippy` when the
  toolchain is installed with `--profile minimal`, and the workspace
  `.cargo/config.toml` travels with the checkout — a python3 rustc-wrapper
  requires python3 in any minimal container image (see the run-191 CUDA fix).
- A manifest authored by mirroring another family is a HYPOTHESIS, not a contract:
  fail-closed validation of an unverified manifest must not ship default-on (2nd
  occurrence of the Oct-4 trunk contract-drift class — this time the MTP manifest
  took down every native qwen36moe load on the H100). Before enabling manifest
  validation by default, either census the published checkpoint (HTTP range-request
  header fetch, no download needed) or gate the feature off until the handoff.
  Corollary: a fixture generated FROM the plan fn can never catch plan-vs-published
  drift — pin an executable census of the PUBLISHED tensors for every validated
  plan, trunk and MTP alike.
- A test that asserts on a single request after a state change is a POLL-CADENCE
  RACE, not a deterministic check: a gateway learns a peer's durable claim and an
  in-flight invocation's released credit on its own status-poller cadence
  (90-110% jitter), so a fixed sleep or one-shot assertion races an observation
  the product never promised synchronously. Assert on the CONVERGED outcome
  (bounded retry, fail after a generous window), and keep the strict guard: only
  served-or-shed may be retried, anything else fails immediately. Corollary: a
  shared/database-side signal is NOT a valid proxy for another process's in-memory
  registry — a peer's row can be staler than a third party's view, so it cannot
  gate the race. Diagnose these by running the lane under realistic CPU load
  (the repo's CI runners are 2 vCPU): an idle machine hid a 1-in-6 flake.
- A loader that materializes a tensor correctly can still be wrong about its VALUE
  CONVENTION, and no shape/dtype/finiteness check will ever see it. Qwen3.5/3.6-MoE
  stores RMSNorm gains zero-centered (HF applies `1.0 + weight` at runtime); the
  qwen36moe native loader used them raw, attenuating 100 per-layer gains ~10-25x, so
  the residual stream degenerated to the untouched embedding and the model emitted the
  unconditional unigram prior as multilingual salad — a symptom that reads exactly like
  a broken CUDA kernel. Two sessions of kernel/layout/routing audits cleared every
  prime suspect while the bug sat one `+ 1.0` away in a loader. Rules that follow:
  (1) when a native safetensors loader and its sibling disagree on a *value transform*
  (`qwen38/text.rs::load_native_zero_centered_norm` vs `qwen36moe/native_model.rs::rms_norm`),
  that disagreement is a defect until proven otherwise — diff the transforms, not just
  the paths; (2) "the trunk works in production" is worthless as coverage when the
  production path is a different FORMAT (llama.cpp bakes the `+1` in at GGUF
  conversion, so the GGUF fixture cannot detect its absence); (3) a fixture generated
  from the plan, or asserting only FINITENESS, cannot catch a gain error — assert
  effective gain == 1 + w; (4) census the PUBLISHED VALUES, which is cheap: safetensors
  headers are readable over HTTP Range, so a 37 GB checkpoint's value census costs a
  few hundred KB. Validate the decoder against 2-3 known-plausible tensors (A_log,
  dt_bias, conv1d) before believing any reading, and use a sibling checkpoint whose
  loader is known-good as the control. Two decoder traps: BF16 is the HIGH half of F32
  (low-half reconstruction yields all-denormal zeros), and `data_offsets` are relative
  to `8 + header_length`, not byte 0.
- When a shared trunk was written for GGUF and a NATIVE loader feeds it, enumerate EVERY
  value transform the llama.cpp converter applies for that arch (`conversion/<family>.py`
  `modify_tensors` + any mixin bases) and check that the native loader mirrors each one.
  Qwen3.5/3.6 has four transforms: A_log→-exp, norm +1, conv squeeze, and the V-head grouped→tiled
  reorder. Two of them were missed. "The GGUF path works on this trunk" proves the
  convention of the CONVERTED file, never the published one. That fallacy hid the norm
  bug and then "refuted" the V-head bug. Also: a norm's convention comes from its module
  CLASS in the reference implementation, not from a magnitude census of its stored values
  (a trained zero-centered gain can drift to 1.6). And "no weight permutation can
  reconcile layouts X and Y" claims need a written proof; permuting the other side
  usually works.
- When a checkpoint convention disagrees with a shared trunk's assumption, first ask whether the
  TRUNK can honor the checkpoint's convention cheaply before rewriting weights at load. The
  Qwen3.6 value-head order fix was ~30 lines as a source-declared expansion order
  (`linear_v_head_order`) versus byte-level permutation across five residency forms (packed Q8,
  tiled Q8, expanded, raw FP8, streaming loads). Weight surgery is right only when every
  consumer (kernels included) hard-codes the convention. Then gate the kernels that do (the
  compact Metal DeltaNet kernel hard-codes tiled pairing on un-expanded heads).
- Prove a value-parity fixture is RED before trusting it green: revert each fix in turn and
  confirm the golden test fails by a margin far above its tolerance (here 2-3 logits vs 2e-3),
  and validate the reference itself (HF cached decode == full forward) before using it.
- Do not run rustfmt over whole files to "keep them clean" unless HEAD is verifiably clean
  in-repo (`git show HEAD:f | rustfmt --check` via stdin can report clean while the in-repo
  run reformats ~150 lines); unrelated reformat churn buries a fix. Only the files listed in
  `scripts/ci/check-backend-truth.sh` are format-gated.
- Splitting a dirty tree into logical commits: stage exact content with
  `git hash-object -w` + `git update-index --cacheinfo`, test the index state under
  `git stash push --keep-index`, then restore with `git checkout stash@{0} -- <files>` and
  drop the stash — `git stash pop` conflicts once the staged hunks are committed.
- Do not make a plan's first phase depend on measurement access the user may not have
  (profilers, benchmark lanes, side-by-side runs of other engines on the production GPU).
  Before gating work on "measure first", confirm what hardware and tooling the user can reach.
  When the answer is "none beyond the deployed app", verify progress with what exists: the app's
  own throughput readout on a fixed prompt, CPU-testable structural invariants (count
  device-to-host readbacks through one helper), path counters in diagnostics, and load-time
  self-checks that compare each new GPU fast path to the legacy path and disable it on
  mismatch. Start with the phase whose targets are COUNTED in code (for example, 80 syncs per
  token), not estimated, so it is safe without a profile. Restate any earlier promotion rule
  that assumed hardware evidence instead of silently ignoring it.
- Never gate a commit on `cargo test ... | grep ... | head && git commit`: a pipeline's exit
  status is the LAST command's (`head` exits 0 even when the build failed and grep matched
  nothing), so a non-compiling tree got committed. Capture output to a file, keep each
  command's `$?`, and commit only inside `if [ $T -eq 0 ] && [ $C -eq 0 ]; then ... fi`. A
  silent/empty test summary is a failure, not a pass.
- When applying rustfmt only to "my" hunks of a file with pre-existing format drift, rustfmt's
  import re-sorting splits one logical move into a removal hunk and an insertion hunk at
  different lines; applying only the hunk that overlaps my edits silently DELETES imports.
  Re-compile after any partial-format pass, and prefer hand-formatting the few new hunks.
- Without nvcc/GPU locally (macOS), CUDA work is still verifiable before deploy: Apple clang
  parses CUDA device code with `-x cuda --cuda-device-only -fsyntax-only -nocudainc` plus a small
  stub header (validate the stubs by checking the repo's existing .cu files first); a
  std::thread-per-CUDA-thread emulator (block barriers + warp-shuffle exchange) runs the real
  kernel source against a float64 reference; and `cargo check/clippy --features cuda` works with
  a fake `nvcc` that prints a 5-line `--version` (cudarc reads line 4) and touches the PTX/object
  outputs the build scripts expect.
- When porting fast paths to a new device, test each path's load-time resolution on that device through production loading, not only its kernels and a test hook. The Metal kernels passed their GPU tests, but `resolve_fused_qk` probed support with a hard-coded BF16 that the F16-only Metal kernel rejects, so the path silently stayed legacy. A device leg asserting every diagnostics summary is `fused` caught it at once.
- A load-time self-check that compares a fused path with a "reference" built from the same transformed weights (stacked views, packs) cannot see a bad transform. Check the transform against the original tensors before dropping them.
- On failure paths around a native resource (stream capture, graph teardown), read the wrapper library's source for hidden pre-checks before relying on its cleanup call. cudarc's `end_capture` first runs `bind_to_thread` → `check_err`, which returns a stale error saved by an unrelated `Drop` without ever ending the capture, leaving the stream stuck in capture mode. Use the raw driver call inside a drop guard for cleanup that must always run.
