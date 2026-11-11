# Speech API end-to-end checks

Stock OpenAI SDK clients, and SGLang-Omni/vLLM-Omni style socket clients, against a
running izwi-server. These checks cover `/v1/audio/speech` (every format, plus chunked
`audio` and SSE streams), model aliases, the outcome lookup, `/v1/audio/translations`,
voices and consents, batch, the `/v1/audio/speech/stream` socket, and the Files and
Batch APIs (`client.files`, `client.batches`).

## Run a server

Run it with the aliases the alias and translation scenarios use. The command below
uses Metal; `--backend cpu` and `--backend cuda` run the same checks.

```bash
IZWI_DB_PATH=/tmp/izwi-speech-e2e.sqlite3 IZWI_MEDIA_DIR=/tmp/izwi-speech-e2e-media \
IZWI_OPENAI_MODEL_ALIASES=tts-1=Kokoro-82M,whisper-1=Granite-Speech-5.0-2B-GGUF \
  target/release/izwi-server --port 18080 --backend metal
```

The scenarios need these models installed:

- Kokoro-82M
- Qwen3-TTS-12Hz-0.6B-Base (voice cloning)
- Chatterbox-Multilingual-500M (French speech for the translation check)
- Granite-Speech-5.0-2B-GGUF (translation)

## openai-python

```bash
uv venv /tmp/speech-sdk && uv pip install --python /tmp/speech-sdk/bin/python openai "websockets>=13" httpx
/tmp/speech-sdk/bin/python scripts/e2e/speech-api/run_sdk.py --url http://127.0.0.1:18080
```

`--only 1 3 9` runs some of the scenarios. S11 runs right after S6.

`/v1/audio/speech` pins each model it loads (Kokoro, Chatterbox Multilingual,
then the Qwen3-TTS clone model). On a 16 GB Mac, Granite Speech has no room
to load again once all three are pinned. A scenario rerun with `--only` that
needs it then fails with `service_unavailable`: unload a TTS model first
(`POST /v1/admin/models/{model}/unload`).

Don't set the aliases on a server you use for `scripts/e2e/openai-realtime`.
There, `whisper-1` would load Granite Speech beside the voice stack.

| # | Scenario |
| --- | --- |
| 1 | `mp3`, `opus`, `flac` and `wav` decode with ffprobe as the codec their Content-Type names; `pcm` is 24 kHz |
| 2 | `stream_format="audio"` through `with_streaming_response.iter_bytes()`: chunked, first byte before the end, one decodable MP3 |
| 3 | `stream_format="sse"` with Opus: the deltas concatenate into one Ogg Opus file; `speech.audio.done.usage` has all three counts |
| 4 | `model="tts-1"` speaks through `IZWI_OPENAI_MODEL_ALIASES` |
| 5 | `sample_rate` (16 kHz FLAC), then `GET /v1/audio/speech/{X-Izwi-Speech-Id}` is `completed` |
| 6 | French speech (Chatterbox Multilingual) into `translations.create(model="whisper-1")` in all five formats; `verbose_json.task == "translate"` |
| 7 | `voice_consents.create/list/update/delete`, `voices.create(consent=…)`, `speech.create(voice={"id"})`; deleting the consent makes the voice a 403 |
| 8 | Batch: three items, two succeed, the third's error names `items.2.response_format` |
| 9 | Socket (izwi dialect): `session.configured`, matched `audio.start`/`audio.done`, each segment a decodable Ogg Opus file, `Dr.` and `3.5` not split; the utterance's lookup is `completed` with usage |
| 10 | Socket (sglang-omni dialect) closes after `session.done`; `input.cancel` ends the utterance (vllm-omni dialect) |
| 11 | openai-node: `speech.create` returns MP3 bytes and `translations.create` returns an object (JSON Content-Type) |
| 12 | SGLang-Omni's voice upload as its docs send it (`audio_sample`, free-text `consent`), `names_only`, speech by the uploaded name, delete by name |
| 13 | `files.create/retrieve/list/content/delete` round-trip a `batch` file; a deleted file is a 404 and purpose `assistants` a 400 |
| 14 | `batches.create(endpoint="/v1/audio/transcriptions")`: one line names its audio by `file_id`, one carries `audio_base64` with `verbose_json`, one names an unknown model; the batch is `completed` with counts 3/2/1, the output file in input order and the refused line in the error file |
| 15 | `batches.create(endpoint="/v1/audio/speech")`: each line's `audio_file_id` is a decodable MP3 named `<custom_id>.mp3`; a duplicate `custom_id` and `stream: true` fail a batch with line errors; `batches.cancel` ends `cancelled` |

## openai-node

Scenario 11 needs openai-node installed in a directory:

```bash
mkdir -p /tmp/speech-node && (cd /tmp/speech-node && npm init -y && npm install openai)
/tmp/speech-sdk/bin/python scripts/e2e/speech-api/run_sdk.py --only 11 --node-dir /tmp/speech-node
```

## CUDA (H100)

Run the same `run_sdk.py` against a CUDA server. Then measure what the encoders
cost under load with `scripts/bench/run-audio-load.py`. `--tts-format` asks for a
format, and decodes it with ffmpeg for parity. `--tts-stream-format audio` also
reports the chunked body's time to first byte:

```bash
for format in wav pcm_i16 mp3 opus flac; do
  python3 scripts/bench/run-audio-load.py --task tts --model Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --voice ryan --tts-format "$format" --tts-stream-format audio \
    --request-rate inf --max-concurrency 32 --num-requests 64 --require-parity \
    --output "target/speech-h100-$format.json"
done
```

Compare these between formats:

- `audio_throughput`;
- `first_byte_s`;
- `rtf`;
- the host CPU while the run is busy.

The plan's gate is that encoding all streams costs under one core and adds under
5 ms to first audio. Translations on Nemotron 3.5 Transcribe (FP8, CUDA only) run with
`IZWI_OPENAI_MODEL_ALIASES=whisper-1=Nemotron-3.5-Transcribe-30B-A4B-FP8` and
`run_sdk.py --only 6`.
