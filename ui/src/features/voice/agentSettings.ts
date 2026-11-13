/**
 * What a voice conversation's agent may do, as chosen in Voice settings and
 * remembered on this device: the tools it may call, the agents it is made
 * of, and how it takes turns.
 */

/** An MCP server whose tools the agent may call. */
export interface VoiceToolServer {
  /** Names the server; letters, digits, `_` or `-`. */
  label: string;
  url: string;
  /** Sent as a bearer token, when the server needs one. */
  token: string;
}

/** One agent of a conversation that several share. */
export interface VoiceAgent {
  /** Letters, digits, `_` or `-`; made from the name. */
  id: string;
  name: string;
  instructions: string;
  /** When another agent should hand the conversation to this one. */
  handoffDescription: string;
  /** The ids of the agents this one may hand the conversation to. */
  handoffs: string[];
}

export interface VoiceAgentSettings {
  /** The built-in tools (date and time); `null` is the engine's default. */
  builtinTools: boolean | null;
  toolServers: VoiceToolServer[];
  /**
   * Labels of the servers set up on the Izwi server (its MCP servers file)
   * that are turned on.
   */
  configuredToolServers: string[];
  /** With two or more, the first answers and they hand over to each other. */
  agents: VoiceAgent[];
  /** Follow the speaker's own pauses when deciding they have finished. */
  adaptiveSilence: boolean;
  /** Start writing the reply at the pause that may end the turn. */
  replySooner: boolean;
  /** Pause, and play on, when a sound over the reply was not speech. */
  resumeFalseInterruptions: boolean;
}

export const DEFAULT_VOICE_AGENT_SETTINGS: VoiceAgentSettings = {
  builtinTools: null,
  toolServers: [],
  configuredToolServers: [],
  agents: [],
  adaptiveSilence: false,
  replySooner: false,
  resumeFalseInterruptions: true,
};

export const MAX_VOICE_TOOL_SERVERS = 8;
export const MAX_VOICE_AGENTS = 16;

const ID_PATTERN = /^[A-Za-z0-9_-]{1,48}$/;

/** "Front desk" → "front-desk"; made distinct from `taken`. */
export function voiceAgentId(name: string, taken: string[]): string {
  const base =
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40) || "agent";
  let id = base;
  for (let n = 2; taken.includes(id); n += 1) {
    id = `${base}-${n}`;
  }
  return id;
}

/** Why a tool server cannot be added, or `null` when it can. */
export function voiceToolServerProblem(
  server: VoiceToolServer,
  others: VoiceToolServer[],
): string | null {
  if (!ID_PATTERN.test(server.label)) {
    return "Give it a short name: letters, digits, - or _.";
  }
  if (others.some((other) => other.label === server.label)) {
    return "Another tool server has that name.";
  }
  try {
    const url = new URL(server.url);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return "The address must start with http:// or https://.";
    }
  } catch {
    return "Enter the server's full address, such as https://example.com/mcp.";
  }
  return null;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** Settings read back from storage: anything malformed falls to its default. */
export function parseVoiceAgentSettings(raw: string | null): VoiceAgentSettings {
  let stored: Record<string, unknown> = {};
  try {
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      stored = parsed as Record<string, unknown>;
    }
  } catch {
    // Unreadable settings are the defaults.
  }
  const flag = (key: keyof VoiceAgentSettings): boolean =>
    typeof stored[key] === "boolean"
      ? (stored[key] as boolean)
      : (DEFAULT_VOICE_AGENT_SETTINGS[key] as boolean);

  const toolServers: VoiceToolServer[] = [];
  for (const item of Array.isArray(stored.toolServers) ? stored.toolServers : []) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const server = {
      label: asString(record.label),
      url: asString(record.url),
      token: asString(record.token),
    };
    if (
      toolServers.length < MAX_VOICE_TOOL_SERVERS &&
      voiceToolServerProblem(server, toolServers) === null
    ) {
      toolServers.push(server);
    }
  }

  const agents: VoiceAgent[] = [];
  for (const item of Array.isArray(stored.agents) ? stored.agents : []) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const id = asString(record.id);
    if (
      agents.length >= MAX_VOICE_AGENTS ||
      !ID_PATTERN.test(id) ||
      agents.some((agent) => agent.id === id)
    ) {
      continue;
    }
    agents.push({
      id,
      name: asString(record.name) || id,
      instructions: asString(record.instructions),
      handoffDescription: asString(record.handoffDescription),
      handoffs: Array.isArray(record.handoffs)
        ? record.handoffs.filter((to): to is string => typeof to === "string")
        : [],
    });
  }
  // A handoff names an agent that is still there, and not its own.
  for (const agent of agents) {
    agent.handoffs = agent.handoffs.filter(
      (to, index, all) =>
        to !== agent.id &&
        all.indexOf(to) === index &&
        agents.some((other) => other.id === to),
    );
  }

  const configuredToolServers = [
    ...new Set(
      (Array.isArray(stored.configuredToolServers)
        ? stored.configuredToolServers
        : []
      ).filter(
        (label): label is string =>
          typeof label === "string" && ID_PATTERN.test(label),
      ),
    ),
  ];

  return {
    builtinTools:
      typeof stored.builtinTools === "boolean" ? stored.builtinTools : null,
    toolServers,
    configuredToolServers,
    agents,
    adaptiveSilence: flag("adaptiveSilence"),
    replySooner: flag("replySooner"),
    resumeFalseInterruptions: flag("resumeFalseInterruptions"),
  };
}

/** Remove an agent, and every handoff to it. */
export function withoutVoiceAgent(agents: VoiceAgent[], id: string): VoiceAgent[] {
  return agents
    .filter((agent) => agent.id !== id)
    .map((agent) => ({
      ...agent,
      handoffs: agent.handoffs.filter((to) => to !== id),
    }));
}

/**
 * Whether the built-in tools are on: the choice made, else off on both
 * engines. Offering a tool lengthens every standard turn, and the compact
 * engine's model calls a lone date-and-time tool on a greeting and reads
 * the date out.
 */
export function voiceBuiltinToolsOn(settings: VoiceAgentSettings): boolean {
  return settings.builtinTools ?? false;
}

/**
 * The `input_stream_start` fields these settings add for a conversation on
 * an engine of `mode`. Agents and an early reply need the standard
 * (modular) engine. `callsTools` is whether the engine's model can call
 * tools at all: without it nothing tool-related is sent. `configured` are
 * the labels of the servers set up on the Izwi server: one that is turned
 * on but no longer there is left out, as is one whose name an added server
 * has taken.
 */
export function voiceStreamOptions(
  settings: VoiceAgentSettings,
  mode: "modular" | "unified",
  callsTools: boolean,
  configured: string[] = [],
): Record<string, unknown> {
  const options: Record<string, unknown> = {
    resume_false_interruptions: settings.resumeFalseInterruptions,
  };
  if (settings.adaptiveSilence) {
    options.adaptive_silence = true;
  }
  if (mode === "modular" && settings.replySooner) {
    options.preemptive_generation = true;
  }
  if (!callsTools) {
    return options;
  }
  if (settings.builtinTools !== null) {
    options.builtin_tools = settings.builtinTools;
  }
  const servers: Record<string, unknown>[] = settings.toolServers.map(
    (server) => ({
      label: server.label,
      url: server.url,
      ...(server.token.trim()
        ? { headers: { Authorization: `Bearer ${server.token.trim()}` } }
        : {}),
    }),
  );
  for (const label of settings.configuredToolServers) {
    if (
      configured.includes(label) &&
      !settings.toolServers.some((server) => server.label === label)
    ) {
      servers.push({ label });
    }
  }
  if (servers.length > 0) {
    options.mcp_servers = servers.slice(0, MAX_VOICE_TOOL_SERVERS);
  }
  if (mode === "modular" && settings.agents.length >= 2) {
    options.agents = settings.agents.map((agent) => ({
      id: agent.id,
      name: agent.name,
      ...(agent.instructions.trim() ? { instructions: agent.instructions.trim() } : {}),
      ...(agent.handoffDescription.trim()
        ? { handoff_description: agent.handoffDescription.trim() }
        : {}),
      handoffs: agent.handoffs,
    }));
  }
  return options;
}
