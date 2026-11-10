---
title: "Phone calls and call centres"
description: "Connect phone calls to Izwi for live transcription and voice agents, through a media server, an agent framework or a carrier's media stream."
sidebarTitle: "Phone calls"
icon: "phone"
---
Izwi does not terminate phone calls itself. A call reaches it through something that does: a SIP or WebRTC media server, an agent framework, or a carrier or call-centre platform that streams a call's audio over a WebSocket. Izwi then transcribes the caller live, or answers as a voice agent, with models running on your own hardware.

---

## Ways to connect a call

| How the call arrives | Izwi endpoint | Use it when |
|----------------------|---------------|-------------|
| A SIP or WebRTC media server, or an agent framework, that speaks the OpenAI Realtime protocol | `GET /v1/realtime` | You already run a media server or agent framework, or need SIP trunking, transfers or DTMF menus. They stay in that layer. |
| A carrier or call-centre platform that streams call audio over a WebSocket (a "media stream") | `GET /v1/realtime/media-stream` | The platform can fork or connect a call to a WebSocket URL and you want no other component in between. |

Not supported:

- **Native SIP.** Izwi has no SIP endpoint and registers with no trunk.
- **WebRTC.** `/v1/realtime/calls` is not served, and `/v1/realtime?call_id=` returns `400`.
- **DTMF.** Keypad digits are not detected in audio, and `dtmf` frames of a media stream are ignored. Handle menus in the telephony layer.

Both endpoints run the same session, so models, voices, turn detection and limits are those of the [OpenAI Realtime API](/api#openai-realtime-api). Each call takes one realtime session slot (`IZWI_MAX_REALTIME_SESSIONS`).

---

## Through a media server or agent framework

Anything with an OpenAI Realtime client can be pointed at Izwi. The repository's end-to-end suite (`scripts/e2e/openai-realtime/`) runs the unmodified Realtime clients of openai-python, LiveKit Agents and Pipecat against `/v1/realtime`; with LiveKit or Pipecat, the framework's own SIP or telephony transport carries the call and its Realtime client talks to Izwi.

Configure the framework's OpenAI Realtime client with:

| Setting | Value |
|---------|-------|
| Base URL | `http://<izwi-host>:8080/v1` (clients that take a WebSocket URL: `ws://<izwi-host>:8080/v1/realtime`) |
| API key | The server's `IZWI_API_KEY`, or a [client secret](/api#browser-clients-and-client-secrets) minted for the call |
| Model | A realtime profile such as `izwi-voice`; OpenAI model names select the default profile |
| Audio format | `audio/pcmu` (G.711 μ-law) or `audio/pcma` (A-law) at 8 kHz in both directions when the framework passes telephone audio through unchanged; `audio/pcm` at 24 kHz when it resamples |
| Turn detection | `server_vad`, so Izwi cuts the caller's turns and a caller who talks over the agent interrupts it |

```python
# LiveKit Agents
openai.realtime.RealtimeModel(base_url="http://izwi-host:8080/v1", api_key=IZWI_API_KEY, model="izwi-voice")

# Pipecat
OpenAIRealtimeLLMService(base_url="ws://izwi-host:8080/v1/realtime", api_key=IZWI_API_KEY, ...)
```

---

## Through a media stream

`GET /v1/realtime/media-stream` speaks the WebSocket framing carriers and call-centre platforms use to hand a call's audio to an application: JSON text frames carrying base64 8 kHz μ-law audio. Give the platform the stream URL; one socket carries one call.

```
wss://<izwi-host>/v1/realtime/media-stream?model=izwi-voice&voice=cedar
```

| Parameter | Meaning |
|-----------|---------|
| `model` | The realtime profile, as on `/v1/realtime`. |
| `intent` | `transcription` for a call that is only transcribed. Without it the call is a conversation and Izwi answers. |
| `voice` | The agent's voice. |
| `instructions` | The agent's system prompt. |
| `transcription_model` | The model that transcribes the caller. |
| `language` | The caller's language, for the transcription model. |
| `transcripts` | `false` turns the `transcript` frames off. |

Some platforms drop the query string of a stream URL. Send the same names as the `start` frame's custom parameters instead; a custom parameter overrides the query string.

Authentication is the server's own: an `Authorization: Bearer` header carrying `IZWI_API_KEY` or a [client secret](/api#browser-clients-and-client-secrets), or the client secret as the `openai-insecure-api-key.<key>` WebSocket subprotocol. A client secret's session config applies to the call, so an application can mint one per call with the instructions and voice that call needs. Keys are not read from the query string. If your platform can set neither a header nor a subprotocol, put a proxy that adds the header between the platform and Izwi.

The frames, the events Izwi adds, and what is ignored are in the [API reference](/api#telephony-media-streams).

---

## Example: live transcription for agent assist

A supervisor or assist tool wants the caller's words while they are spoken.

1. Pick a transcription model that transcribes live. `GET /v1/realtime/capabilities` lists each model under `transcription_models` with `transcription_mode`: `live` models send partial text while the caller speaks; `turn` models send each utterance after the caller pauses.
2. Have the platform stream the caller's audio to a transcription call:

   ```
   wss://<izwi-host>/v1/realtime/media-stream?intent=transcription&transcription_model=Voxtral-Mini-4B-Realtime-2602
   ```

3. Read the `transcript` frames from the same socket:

   ```json
   {"event": "transcript", "streamSid": "MZ...", "item_id": "item_...", "type": "delta", "text": "I'd like to"}
   {"event": "transcript", "streamSid": "MZ...", "item_id": "item_...", "type": "completed", "text": "I'd like to change my booking."}
   ```

   `delta` frames add up to the utterance and `completed` carries it whole. A `turn` model sends `completed` frames only.

With a media server or agent framework instead, open `/v1/realtime?intent=transcription`, set `audio.input.transcription.model` and read `conversation.item.input_audio_transcription.delta` and `.completed` events.

To transcribe both parties, stream each party's audio to its own transcription call.

---

## Example: a voice agent answers the call

1. Choose a profile: `izwi-voice` runs a transcription, chat and speech model in sequence; `izwi-voice-compact` runs one speech-to-speech model. `IZWI_REALTIME_PROFILES` defines your own.
2. Connect the call to a conversation:

   ```
   wss://<izwi-host>/v1/realtime/media-stream?model=izwi-voice&voice=cedar
   ```

   Send `instructions` as a custom parameter, or bind them to a client secret, to give the agent its script.
3. Izwi cuts the caller's turns, answers each one, and sends the speech back as `media` frames for the platform to play. When the caller talks over the agent, Izwi stops the reply and sends `clear` so the platform drops the audio it has not played yet.

Function tools, and anything else that needs the full Realtime protocol during the call, need a Realtime client: connect through an agent framework, or your own client on `/v1/realtime` with G.711 audio.

`scripts/e2e/openai-realtime/run_media_stream.py` plays a WAV file as a call of either kind and saves the agent's speech.

---

## Post-call transcription beside live calls

One server can transcribe recordings in bulk while it serves live calls.

- Send recordings to `POST /v1/audio/transcriptions` with `service_tier: "flex"`. Flex requests are admitted and scheduled behind live work. See [Audio Service Tier](/api#audio-service-tier).
- While a model has live demand, batch work on that model leaves part of its rows free so a live request starts without waiting. `IZWI_REALTIME_ROW_RESERVE` sets how many (by default a quarter of the model's rows, at least one).
- A flex request that finds the server full waits up to `IZWI_ADMISSION_QUEUE_TIMEOUT_SECS` (30 seconds) and is then refused with `429`, code `resource_unavailable` and a `Retry-After` header. Retry it later.
- Long recordings are transcribed in bounded steps, with live requests served between them. Two models are the exception: VibeVoice-ASR and Nemotron 3.5 Transcribe read a whole recording in one pass and hold the device until it ends, so a live request of the same model waits. If you need those models for recordings and strict live latency, run them on a separate server or device.

How many live calls a GPU carries, alone and beside a batch load, depends on the models and the hardware and has to be measured on yours. `scripts/bench/run-mixed-load.py` holds a set of live sessions while it raises a batch load and reports what the batch load does to live latency; see `scripts/bench/README.md`.

---

## See also

- [OpenAI Realtime API](/api#openai-realtime-api)
- [Voice Mode](/features/voice)
- [Transcription](/features/transcription)
