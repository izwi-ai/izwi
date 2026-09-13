# Fish S2 long audio serving and qualification

Fish speech planning applies to streaming and final-file generation. It preserves
input order and uses one segment for short input. Long input is segmented before
per-segment output budgets are assigned. The loaded tokenizer/reference/context
check can split a segment further before it publishes any audio. Frame exhaustion
is incomplete speech, never a successful recording. A model EOS is necessary for
success but does not establish transcript completeness or voice quality.

These changes have automated fixture coverage. They have **not** been qualified
on a deployed CUDA GPU for hours of audio, sustained concurrency, voice continuity
or restart/failure behavior. Do not infer a production capacity or quality claim
from fixture test results. Keep the rollout disabled on an unqualified deployment
until its acceptance gates pass.

## Limits and operational choices

| Setting | Default | Meaning |
| --- | --- | --- |
| `IZWI_FISH_LONG_FORM_ENABLED` | `true` | `false` rejects multi-segment Fish input before acceptance; short single-segment input remains available. This is separate from native batching. |
| `IZWI_TTS_MAX_TEXT_BYTES` | 1 MiB | UTF-8 input admission ceiling, not an audio duration estimate. |
| `IZWI_TTS_LONG_FORM_CHUNK_MAX_CHARS` | 480 | Preliminary segment target; Fish uses at most 480 characters, with exact loaded-context validation afterward. |
| `IZWI_TTS_SEGMENT_TIMEOUT_SECS` | 1800 | Inference deadline for an individual segment. |
| `IZWI_TTS_MAX_JOB_SECONDS` | 86400 | Whole-job wall deadline measured from durable creation, including queueing, retries and cooperative continuations. |
| `IZWI_TTS_MAX_ACTIVE_JOBS` | 256 | Database-wide admitted live TTS jobs, including queued, running, paused and retrying jobs. |
| `IZWI_TTS_MAX_ACTIVE_JOBS_PER_TENANT` | 32 | Live TTS jobs per trusted server-authored tenant identity; legacy/unidentified jobs share the anonymous bucket. |
| `IZWI_TTS_REPLAY_RETENTION_SECONDS` | 86400 | Retention interval for replay PCM belonging to terminal jobs. |
| `IZWI_TTS_MAX_JOURNAL_BYTES` | 8 GiB | Per-job replay journal ceiling; not an aggregate storage reservation. |
| `IZWI_TTS_TOTAL_SPOOL_BYTES` | 1 GiB | Process-wide concurrent temporary WAV reservation. Increase only with measured disk capacity and concurrency headroom. |

Each PCM replay object is additionally capped at 1 MiB, and one job can retain
at most 131,072 replay entries. The entry ceiling covers the two-hour
qualification target at the default 4,800-sample Fish output chunk while keeping
pathological tiny-chunk journals finite.

Explicit caller `max_tokens` / `max_output_tokens` remains a **whole-job** output
budget. It is not multiplied by the number of segments. Omit a cap for
complete-text generation subject to deployment quotas. Exhausting an explicit
cap, context, disk, replay or time limit produces a non-success outcome; it must
not present partial audio as ready. Raising the input limit alone cannot raise
storage or HTTP limits.

At 44,100 Hz, mono PCM16 needs 88,200 bytes per audio second: approximately
303 MiB for one hour or 606 MiB for two hours, excluding metadata, replay journal,
storage copies and overlapping finalization. Replay and temporary final WAV may
coexist. Final assembly waits for aggregate spool capacity when the artifact can
fit the configured pool; an artifact larger than the entire pool is rejected.
The wait remains cancellable and within the whole-job deadline. A 1 GiB aggregate
spool budget therefore does not assemble arbitrarily many two-hour recordings
simultaneously. RIFF/WAV has a format-size ceiling; a larger
storage quota is not permission to create an invalid WAV. Inspect real artifact
sizes and reservations under concurrent finalization before raising quotas.

New durable Fish PCM is written through reserved-write protocol v1 as
tenant-scoped opaque artifacts. The checkpoint publication marker, opaque media
row, exact-attempt replay reference, and reservation consumption commit in one
transaction; runtime replay rows contain no provider key. Reads verify tenant,
canonical `audio/pcm-f32le`, size, and SHA-256 before decoding. Tombstone-first
deletion retains a durable cleanup intent across provider failure. Existing
raw-key replay rows remain supported for local upgrade compatibility. Speech
history can consume and stream an exact tenant-scoped opaque artifact reference,
but completion and deletion stay fenced until artifact/history settlement is
transactional. The Fish producer does not publish the final WAV through that
path yet.

The provider contract includes an explicit reserved-file capability for that
future final-WAV migration. It copies and hashes finalized spools with fixed
64 KiB buffers and retains the same write-ID expiry/recovery fence. A provider
that supports reserved byte writes but not reserved files must keep final-WAV
fleet readiness disabled; there is no fallback to an unreserved file upload.

Media providers must implement reserved writes plus the bounded file publication
and streaming read interfaces used by long speech. A legacy whole-byte
upload/download adapter is not sufficient for new replay entries. Test provider
failures, quota exhaustion, partial files, cleanup and restart recovery using the
deployed adapter. Plan storage retention and cleanup for replay artifacts and
failed jobs as well as successful recordings.

Local speech scratch files live below a lazily created, process-owned directory
under `IZWI_SPEECH_SPOOL_DIR` (by default the system temporary directory's
`izwi/speech-spool` child). The root must be a non-symlink directory accessible
only to its owner. Each process holds an exclusive lock for its UUID-named child;
first-use cleanup removes a recognized sibling only after acquiring that exact
lock. It never infers death from a PID, age, or expired lease. Live, malformed,
symlinked, or scan-limit-exceeding entries are left untouched. Temporary files
are also atomically capped at 255 per process so every normally produced owner
directory remains within the recovery scan limit. They remain RAII-owned, so
success, rejection, cancellation, timeout, and response drop release their
individual files while the existing aggregate spool-byte budget remains
authoritative. Unix mode bits and Windows protected DACLs restrict both the
root and process directory to the current owner.

## Database migration and model identity

Local SQLite applies the compatibility migration automatically. Provider-managed
schemas must add these objects **before** starting the updated server:

```sql
CREATE TABLE IF NOT EXISTS runtime_admission_locks (
    id TEXT PRIMARY KEY,
    lock_value INTEGER NOT NULL DEFAULT 1
);
ALTER TABLE runtime_jobs ADD COLUMN admission_tenant TEXT NULL;
INSERT INTO runtime_admission_locks (id, lock_value)
VALUES ('tts', 1) ON CONFLICT (id) DO NOTHING;
```

The `ALTER TABLE` applies once; use the provider's migration ledger or supported
`IF NOT EXISTS` syntax on subsequent deploys. Provider schema validation requires
the table and both columns plus `runtime_jobs.admission_tenant`. The store also
idempotently seeds the `tts` row on admission, then updates it inside the same
transaction as counting and creating/retrying jobs. This row lock serializes
concurrent producers across processes; an unlocked count check would race.
Quota limits bound queued job storage/work, independently of GPU batch size.
Terminal jobs stop consuming slots without a separate release counter. Apply the
same configured limits to every process sharing the database.

Provider-managed schemas must also retain the unique
`idx_runtime_artifacts_attempt_publication` index over `(stage_id,
producer_attempt_token, publication_key)`. Startup verifies its uniqueness and
exact column order because opaque attempt publication relies on that fence.

Fish checkpoints seal a content fingerprint of required configuration, tokenizer,
codec and weight shard files, together with reference/settings identity. The
fingerprint is computed once per native model load using bounded reads. This
adds a full pass over model artifacts (roughly 10 GB for a typical S2 Pro load;
use the deployed artifact sizes for the actual cost), so include extra disk I/O
and startup time in deployment measurements. Serve model files immutably during
loading and throughout that model's lifetime; never replace files in place under
a loaded process. Deploy a new revision/path and load it as a new model identity.
A checkpoint created with a different fingerprint cannot resume under the new
weights/tokenizer/codec.

## Durable jobs, HTTP requests and partial recordings

Use speech history for long-lived work. A history listener disconnect detaches
playback from the durable job; it does not mean that the user cancelled inference.
Reconnect to the record's events route with the last committed sequence cursor.
After a reload, use **Listen from beginning** on the pending recording to attach
without submitting the text again. Explicit cancel remains the operation that stops the job. Clients must deduplicate
replayed sequence IDs and keep the original PCM sample rate. Navigation and
reconnection must not regenerate independently sampled audio.

Completed segment boundaries can be checkpointed and resumed with the same
model/reference/settings identity. Between segments, the durable stage yields
its worker claim and rejoins the queue so short jobs can obtain worker slots as
well as GPU permits. Continuations keep the failure retry budget and receive a
fresh attempt token; old workers cannot publish using the reused retry count. A worker interruption inside a published
segment cannot safely regenerate that segment: report an incomplete outcome
instead of silently duplicating samples. Attempt fences must prevent a stale
worker from updating progress, publishing new PCM, or completing the recording.
A final ready state requires all segments and final artifact publication.

Before deleting expired replay, GC atomically fences the terminal job with
`speech_replay_expired`. That job can no longer be retried: submit a new job if
regeneration is needed. This prevents a concurrent retry from racing deletion of
its checkpoint audio. Interrupted deletion remains eligible for cleanup, and the
original failure message is retained. The saved final artifact and the replay
journal have distinct retention responsibilities; test both with the actual
media provider.

OpenAI-compatible synchronous HTTP requests retain disconnect-cancels semantics.
Streaming keepalives address idle transport periods but do not extend a proxy's
hard request deadline. Configure and test browser/proxy/load-balancer deadlines
separately from job and segment deadlines; use the durable interface for narration
that may outlive one HTTP request.

## Repeatable evidence

Create a JSON template containing `model`, `saved_voice_id` (or an authorized
reference), and desired generation settings. The generator removes inherited
output caps for complete-text qualification. Estimated minute labels size text;
actual speech duration depends on language, text, speed and model output.

Generate workloads without contacting a server:

```sh
python3 scripts/bench/run-fish-long-form-qualification.py \
  --template /tmp/fish-template.json --output /tmp/fish-long-workloads
```

The workload matrix covers English and Chinese at approximately 1, 10, 30, 60
and 120 minutes, interleaving short requests with long ones. Numbered passages
help locate omitted or repeated text during transcript review. Also supply real
licensed narration material, punctuation-poor input and long voice references;
repetitive synthetic text alone is not representative quality evidence.

Run against an authorized test deployment (creates real speech records):

```sh
python3 scripts/bench/run-fish-long-form-qualification.py \
  --template /tmp/fish-template.json --metadata /tmp/fish-deployment.json \
  --url https://YOUR-TEST-SERVER/v1/text-to-speech \
  --concurrency 3 --output /tmp/fish-long-c3
```

Concurrency three is an experiment, not a capacity recommendation. Repeat with
one request, irregular widths, the measured capacity, and overload, across CPU
and representative GPU memory tiers. Metadata uses the same exact deployment,
checkpoint, provider, context and cache identifiers as the existing
[concurrent serving benchmark](FISH_S2_MULTI_USER_SERVING.md). Record server
telemetry and generator disk/network headroom alongside these client results.
The socket timeout is an idle-operation timeout. The runner does not retry jobs.

Each request retains its PCM, event metadata, terminal result and downloaded saved
WAV. The tool compares that job's streamed PCM to its saved WAV in bounded blocks,
checking format, sample rate, count, exact sample bytes and SHA-256. It rejects
missing final output, failed jobs, changed record identities, sample mismatch and
oversized downloads. It does not compare independently generated requests.
`--max-download-bytes` defaults to 1 GiB per artifact; increase deliberately for
larger experiments. Existing output directories are never overwritten.

`long-form-report.json` exposes transport pass/fail and per-workload failures.
It always keeps `production_certified: false`: transcript coverage, listening,
resource soak and fault injection require separate evidence. Failures remain in
the report even when other requests complete.

## Required release gates beyond transport

- Review ASR transcript coverage and human listening for every language and
  duration class. Predeclare missing/repeated sentence, voice similarity, boundary
  click/prosody and playback-gap thresholds. Keep the input hashes and reviewers'
  decisions with the run. Exact sample equality does not prove spoken coverage.
- Compare short-request p95/p99 first PCM and audible playback latency against the
  current baseline while long jobs run. Measure fairness by tenant, queue delay,
  rejection rate, decode occupancy and runtime reservation cleanup.
- Verify that RAM, GPU allocation, spool reservation and replay storage are bounded
  during generation, upload, download and concurrent finalization, including
  artifacts above 256 MiB. Monitor disk over a multi-hour soak, not just inference.
- Inject listener disconnect/reconnect and explicit cancel during preparation,
  first PCM, segment boundaries, partial segments, upload and final commit.
  Inject worker restart, stale-worker completion, provider failure and disk quota
  exhaustion. Verify no duplicate committed PCM, false ready state or leaked files.
- Run explicit tiny whole-job output caps and verify a non-success outcome. Test
  text, journal, spool, timeout and RIFF boundary rejection without allocating
  multi-gigabyte fixtures. Record intentional quota failures separately from
  complete-text workloads; never relabel an unexpected failure as a pass.
- Exercise streaming and non-streaming history plus OpenAI routes. Review browser
  scheduling/underruns, cursor validation and duplicate suppression. Server PCM
  receipt alone does not prove successful playback.

Only enable the rollout on a deployment after these gates pass and the measured
operating envelope, storage retention and failure runbook are reviewed.
