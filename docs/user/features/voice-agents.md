---
title: "Voice agents"
description: "Give Izwi's voice assistant tools, connect tool servers, split a conversation between agents, and tune how it takes turns."
sidebarTitle: "Voice agents"
icon: "bot"
---
A voice conversation in Izwi is run by an agent: it listens, decides what to say, and can look things up or act before it answers. This guide covers what you can give that agent, in the app and over the API. Everything runs on your own machine.

For starting a conversation and choosing an engine, see [Voice Mode](/features/voice).

---

## What an agent can do

| Capability | What it means | In the app | Over the API |
|------------|---------------|------------|--------------|
| **Built-in tools** | The assistant can check the date and time | Voice settings › Tools | `builtin_tools` |
| **Your own tools** | The assistant calls a function and your program answers | — | `tools` |
| **Tool servers (MCP)** | The assistant calls the tools of an MCP server (a web service or a program on this computer), and Izwi runs the calls | Voice settings › Tools | `mcp_servers` |
| **Agents and handoffs** | Several agents, each with its own job, pass the conversation between them | Voice settings › Agents | `agents` |
| **Tasks** | An agent collects set details (a name, a date) and records them | — | `agents[].task` |
| **Turn taking** | How quickly it answers, and what counts as an interruption | Voice settings › Conversation | see [Turn taking](#turn-taking) |

Tools need a model that can call them. The **Standard** engine can, with its `Qwen3.5-4B` assistant. Of the **Compact** engine's models, `LFM3-Audio-2B-GGUF` can and `LFM2.5-Audio-1.5B-GGUF` cannot. Agents and tasks need the Standard engine.

---

## Tools

Open **Voice settings › Tools**.

### Date and time

Turn on **Date and time** and the assistant can answer "what's the date?" or "what time is it?" from the real clock.

On the Standard engine this is off until you turn it on. Offering any tool makes the assistant read a longer prompt. With the Qwen3.5 assistant Izwi reuses the part it has read before, so on an M1 Pro the cost is well under a second a reply, where it used to be about two; the first reply after Izwi starts still pays it in full. A reply that calls a tool takes about as long as one that does not. On the Compact engine with LFM3 Audio it is on by default.

### Tool servers

A tool server is a program that offers tools over the [Model Context Protocol](https://modelcontextprotocol.io) (MCP): a stock lookup, a calendar, a ticket system. Add its name and address, and an access token if it needs one. The assistant can then call its tools, and Izwi runs the calls.

Izwi's server makes the connection, not your browser. For safety it only connects to hosts you have allowed. Set `IZWI_MCP_ALLOWED_HOSTS` to a comma-separated list of hosts (`mcp.example.com`, `127.0.0.1:9000`, or `*` for any) before starting Izwi. Until it is set, a conversation that names a tool server by address fails to start and says so.

The access token is kept in this device's browser storage and sent to Izwi when a conversation starts.

### Tool servers that run as a program

Many MCP servers are not web services but programs you run: a file browser, a database reader, a calendar bridge. Izwi starts these itself. Because a program can do anything your account can, only a file on the computer Izwi runs on can name one; nothing typed into the app or sent over the API can.

Create `mcp.json` in Izwi's data folder, next to `izwi.sqlite3`: `~/Library/Application Support/izwi` on macOS, `~/.local/share/izwi` on Linux, `%LOCALAPPDATA%\izwi` on Windows (`IZWI_MCP_SERVERS_FILE` names another place). It takes the same form other MCP apps use:

```json
{
  "mcpServers": {
    "files": {
      "command": "/opt/homebrew/bin/npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"]
    },
    "docs": {
      "url": "https://mcp.example.com/mcp",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

Restart Izwi. Each server then appears in **Voice settings › Tools** under **Set up on this server**, with a switch. A remote server listed in this file needs no entry in `IZWI_MCP_ALLOWED_HOSTS`, and its token stays in the file.

- **Give the full path to the command.** The desktop app does not see the `PATH` your terminal has, so `npx` or `uvx` alone is often not found. `env` adds environment variables and `cwd` sets the folder the program starts in.
- **One copy runs for everyone.** A program starts the first time a conversation uses it, is shared by every conversation after, and is started again if it exits. Its first start may take a while if it has to install itself; Izwi waits up to a minute.
- **Anyone who can use this Izwi server can use these tools.** List only servers you would let every caller of the API reach.

### What you see

When the assistant uses a tool, the transcript shows a short line ahead of its reply: "Using stock level…", then "Used stock level". A tool that fails or takes too long says so, and the assistant tells you it could not check.

If a tool keeps the assistant silent for more than a second and a half, it says "One moment." first.

---

## Agents

One assistant with one set of instructions suits most conversations. Agents help when a conversation has distinct jobs: a front desk that greets and routes, a billing agent that handles refunds.

Open **Voice settings › Agents** and add at least two.

| Field | What to put there |
|-------|-------------------|
| **Name** | What the agent is called. It shows in the transcript when the conversation is handed to it. |
| **What this agent does** | Its instructions. Your main instructions in Voice settings still apply to every agent. |
| **When to hand over to it** | One line the other agents read to decide when to pass the conversation on, such as "For invoices and refunds". |
| **Can hand over to** | The agents this one may pass the conversation to. |

The first agent in the list answers first. When an agent decides the conversation belongs to another, it hands over; the transcript shows "Handed over to Billing", and that agent answers from then on, until it hands back or the conversation ends.

A small model will sometimes say "I'll transfer you" without doing it, or ask again for something you already said. Clear, short instructions ("For anything about invoices, hand over to billing at once") help.

---

## Turn taking

Open **Voice settings**, under **Conversation**.

| Switch | What it does | Default |
|--------|--------------|---------|
| **Play on through noise** | A cough, a hum or a noise in the room pauses the reply for a moment and then lets it carry on, instead of cutting it off. Real words still interrupt. | On |
| **Follow my pace** | Izwi normally waits a little under a second of silence before it answers. With this on it learns how long you pause mid-sentence and waits less if your pauses are short. If it cuts in on you, it backs off again. | Off |
| **Start replies sooner** | Izwi begins working out its answer during that wait, and drops it if you carry on talking. Replies start roughly half a second sooner, at the cost of extra work your computer does on every pause. Standard engine only. | Off |

Izwi decides you have finished from silence alone. It does not use a separate model to guess whether a sentence is complete.

---

## For developers

The voice WebSocket (`/v1/voice/realtime/ws`) takes all of this in `input_stream_start`. A session with the date and time tool, one tool of your own, and a tool server:

```json
{
  "type": "input_stream_start",
  "mode": "modular",
  "asr_model_id": "Parakeet-TDT-0.6B-v3",
  "text_model_id": "Qwen3.5-4B",
  "tts_model_id": "Kokoro-82M",
  "builtin_tools": true,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
          "type": "object",
          "properties": { "city": { "type": "string" } },
          "required": ["city"]
        }
      }
    }
  ],
  "mcp_servers": [
    { "label": "stock", "url": "http://127.0.0.1:9000/mcp" },
    { "label": "files" }
  ]
}
```

An `mcp_servers` entry without a `url` names a server from `mcp.json`; `GET /v1/voice/tool-servers` lists them.

When the reply calls `get_weather`, the server sends `function_call` with a `call_id`. Your client runs it and answers:

```json
{ "type": "function_call_output", "call_id": "call_…", "output": { "temperature_c": 21 } }
```

A tool marked `"background": true` does not hold the reply: the assistant says it has started, and the output you send later is told on the next turn.

Two agents, the second with a task to collect:

```json
{
  "type": "input_stream_start",
  "mode": "modular",
  "asr_model_id": "Parakeet-TDT-0.6B-v3",
  "text_model_id": "Qwen3.5-4B",
  "tts_model_id": "Kokoro-82M",
  "agents": [
    {
      "id": "front",
      "name": "Front desk",
      "instructions": "Greet callers. For invoices or refunds, hand over to billing at once.",
      "handoffs": ["billing"]
    },
    {
      "id": "billing",
      "name": "Billing",
      "handoff_description": "Handles invoices and refunds.",
      "instructions": "Collect the invoice number and the reason, then record the refund request.",
      "task": {
        "name": "refund",
        "description": "Record a refund request once you have the invoice number and the reason.",
        "parameters": {
          "type": "object",
          "properties": {
            "invoice": { "type": "string" },
            "reason": { "type": "string" }
          },
          "required": ["invoice", "reason"]
        }
      }
    }
  ]
}
```

The server sends `agent_handoff` when the conversation moves, and `task_completed` with the collected fields once the agent has them all.

The turn-taking switches are `resume_false_interruptions`, `adaptive_silence` and `preemptive_generation`. A client that sets `resume_false_interruptions` must stop playback on `turn_paused` without discarding what it has queued, and play on at `turn_resumed`.

Clients that speak the OpenAI Realtime protocol (`/v1/realtime`) get the same agent behind it: function tools as usual, and MCP servers as `{"type": "mcp", "server_label", "server_url"}` entries in `tools`. There a tool call waits for your client's approval unless the entry says `"require_approval": "never"`: the response ends with an `mcp_approval_request` item, and an `mcp_approval_response` item runs the call or declines it.

The [API reference](/api#voice-realtime) lists every field and event. To test an agent before people talk to it, `scripts/bench/run-voice-agent-eval.py` plays a caller against it and scores the calls.

---

## Limits

- On the Standard engine a conversation remembers what its tools found, so a later question can build on an earlier lookup without calling the tool again. A long result is kept in shortened form (about 1,500 characters). On the Compact engine tool calls and their results last for one turn.
- A tool server named by address must be allowed in `IZWI_MCP_ALLOWED_HOSTS`; one that runs as a program must be in `mcp.json`. A voice conversation does not ask you before a tool runs.
- Agents and tasks are not available with the Compact engine. Tasks can only be set over the API.
- A hum or a cough resumes the reply only where Izwi can tell it had no words. A short "yes" or "okay" while Izwi is speaking still interrupts it.
