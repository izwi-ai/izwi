# OpenAI Realtime end-to-end checks

Stock OpenAI Realtime clients against a running izwi-server's `/v1/realtime`.
Each script changes only the client's base URL.

| Script | Client | What it checks |
| --- | --- | --- |
| `run_sdk.py` | openai-python (`openai[realtime]`, tested with 3.46.0) | Eight scenarios: a server-VAD spoken turn, a text response, a function-call round trip, barge-in and truncate, manual turns, an out-of-band response, G.711 μ-law, and LiveKit-style history replay. Every server event is validated against the SDK's own GA `RealtimeServerEvent` types; `--record` keeps the traffic as JSONL. |
| `run_livekit.py` | LiveKit Agents `RealtimeModel` (livekit-agents 1.8.11) | `generate_reply` binding, `update_chat_ctx`/`update_tools` acknowledgements, a tool call on the function stream, a server-VAD turn, `interrupt()`. |
| `run_pipecat.py` | Pipecat `OpenAIRealtimeLLMService` (pipecat-ai 1.15.0) | The `session.created` → `session.updated` handshake, a tool round trip through the context aggregator, a spoken turn, barge-in with Pipecat's truncate. Fails on any error event Pipecat would treat as fatal. |

Run a server (here on Metal with the installed models, and a scratch database):

```bash
IZWI_DB_PATH=/tmp/izwi-e2e.sqlite3 IZWI_MEDIA_DIR=/tmp/izwi-e2e-media \
  target/release/izwi-server --port 18080 --backend metal
```

`--backend cpu` (or `cuda`) runs the same checks on another backend; nothing in
the scripts depends on it.

Then, each client in its own environment:

```bash
uv venv /tmp/rt-sdk && uv pip install --python /tmp/rt-sdk/bin/python "openai[realtime]"
/tmp/rt-sdk/bin/python scripts/e2e/openai-realtime/run_sdk.py --model izwi-voice
/tmp/rt-sdk/bin/python scripts/e2e/openai-realtime/run_sdk.py --model izwi-voice-compact
```

```bash
uv venv /tmp/rt-livekit && uv pip install --python /tmp/rt-livekit/bin/python livekit-agents==1.8.11 livekit-plugins-openai==1.8.11
/tmp/rt-livekit/bin/python scripts/e2e/openai-realtime/run_livekit.py --model gpt-realtime
```

```bash
uv venv /tmp/rt-pipecat && uv pip install --python /tmp/rt-pipecat/bin/python "pipecat-ai[openai]==1.15.0"
/tmp/rt-pipecat/bin/python scripts/e2e/openai-realtime/run_pipecat.py --model gpt-realtime
```

## Telephony media stream

`run_media_stream.py` is the carrier side of `/v1/realtime/media-stream`: it
plays a WAV as a phone call (`connected`, `start`, 20 ms frames of base64
8 kHz μ-law in real time, `stop`), prints the `transcript` frames and saves the
speech the call returns. It needs only `websockets`, which `openai[realtime]`
installs.

```bash
# A call that is only transcribed, with a model that transcribes live.
/tmp/rt-sdk/bin/python scripts/e2e/openai-realtime/run_media_stream.py \
    --intent transcription --transcription-model Voxtral-Mini-4B-Realtime-2602
# A call a voice agent answers; its speech is saved as media-stream-reply.wav.
/tmp/rt-sdk/bin/python scripts/e2e/openai-realtime/run_media_stream.py --model izwi-voice --voice cedar
```

`--custom-parameters` sends the call's parameters in the `start` frame instead
of the query string, and `--api-key` (or `IZWI_API_KEY`) authenticates. The
run passes when a transcription call returns a completed transcript, or a
conversation call returns speech, and no `error` frame arrives.

`--url` points at another server (`http://host:port` for the SDK and LiveKit,
`ws://host:port` for Pipecat). The scripts use `data/fox.wav` (24 kHz) as the
user's speech. Models load on first use, so a cold server's first scenario is
slow.
