import { describe, expect, it } from "vitest";

import {
  DEFAULT_VOICE_AGENT_SETTINGS,
  parseVoiceAgentSettings,
  voiceAgentId,
  voiceBuiltinToolsOn,
  voiceStreamOptions,
  voiceToolServerProblem,
  withoutVoiceAgent,
  type VoiceAgentSettings,
} from "@/features/voice/agentSettings";

const front = {
  id: "front-desk",
  name: "Front desk",
  instructions: "Greet callers.",
  handoffDescription: "",
  handoffs: ["billing"],
};
const billing = {
  id: "billing",
  name: "Billing",
  instructions: "",
  handoffDescription: "For invoices.",
  handoffs: [],
};

describe("voice agent settings", () => {
  it("fall back to the defaults when storage is empty or broken", () => {
    expect(parseVoiceAgentSettings(null)).toEqual(DEFAULT_VOICE_AGENT_SETTINGS);
    expect(parseVoiceAgentSettings("{nope")).toEqual(DEFAULT_VOICE_AGENT_SETTINGS);
    expect(parseVoiceAgentSettings("[1]")).toEqual(DEFAULT_VOICE_AGENT_SETTINGS);
  });

  it("keep what is well formed and drop what is not", () => {
    const settings = parseVoiceAgentSettings(
      JSON.stringify({
        builtinTools: true,
        replySooner: true,
        resumeFalseInterruptions: false,
        toolServers: [
          { label: "stock", url: "http://127.0.0.1:9000/mcp", token: "t" },
          { label: "stock", url: "http://other/mcp" },
          { label: "bad name", url: "http://x/mcp" },
          { label: "ftp", url: "ftp://x/mcp" },
        ],
        agents: [
          front,
          { ...billing, handoffs: ["billing", "gone", "front-desk", "front-desk"] },
          { id: "front-desk", name: "Twice" },
          { id: "no good" },
        ],
      }),
    );
    expect(settings.builtinTools).toBe(true);
    expect(settings.replySooner).toBe(true);
    expect(settings.resumeFalseInterruptions).toBe(false);
    expect(settings.adaptiveSilence).toBe(false);
    expect(settings.toolServers).toEqual([
      { label: "stock", url: "http://127.0.0.1:9000/mcp", token: "t" },
    ]);
    expect(settings.agents.map((agent) => agent.id)).toEqual(["front-desk", "billing"]);
    // No handoff to itself, to an agent that is gone, or twice.
    expect(settings.agents[1].handoffs).toEqual(["front-desk"]);
  });

  it("name agents by a distinct id", () => {
    expect(voiceAgentId("Front desk", [])).toBe("front-desk");
    expect(voiceAgentId("Front desk!", ["front-desk", "front-desk-2"])).toBe("front-desk-3");
    expect(voiceAgentId("…", [])).toBe("agent");
  });

  it("say why a tool server cannot be added", () => {
    const server = { label: "stock", url: "https://example.com/mcp", token: "" };
    expect(voiceToolServerProblem(server, [])).toBeNull();
    expect(voiceToolServerProblem(server, [server])).toMatch(/that name/);
    expect(voiceToolServerProblem({ ...server, label: "" }, [])).toMatch(/short name/);
    expect(voiceToolServerProblem({ ...server, url: "example.com" }, [])).toMatch(/full address/);
    expect(voiceToolServerProblem({ ...server, url: "file:///etc" }, [])).toMatch(/http/);
  });

  it("remove an agent together with the handoffs to it", () => {
    expect(withoutVoiceAgent([front, billing], "billing")).toEqual([
      { ...front, handoffs: [] },
    ]);
  });

  it("send only what a conversation's engine can use", () => {
    const settings: VoiceAgentSettings = {
      builtinTools: true,
      toolServers: [
        { label: "stock", url: "http://h/mcp", token: " secret " },
        { label: "open", url: "http://o/mcp", token: "" },
      ],
      configuredToolServers: [],
      agents: [front, billing],
      adaptiveSilence: true,
      replySooner: true,
      resumeFalseInterruptions: true,
    };
    expect(voiceStreamOptions(settings, "modular", true)).toEqual({
      resume_false_interruptions: true,
      adaptive_silence: true,
      preemptive_generation: true,
      builtin_tools: true,
      mcp_servers: [
        { label: "stock", url: "http://h/mcp", headers: { Authorization: "Bearer secret" } },
        { label: "open", url: "http://o/mcp" },
      ],
      agents: [
        { id: "front-desk", name: "Front desk", instructions: "Greet callers.", handoffs: ["billing"] },
        { id: "billing", name: "Billing", handoff_description: "For invoices.", handoffs: [] },
      ],
    });
    // The compact engine has no agents and no early reply.
    const unified = voiceStreamOptions(settings, "unified", true);
    expect(unified.agents).toBeUndefined();
    expect(unified.preemptive_generation).toBeUndefined();
    expect(unified.mcp_servers).toBeDefined();
    // A model that cannot call tools is sent none.
    expect(voiceStreamOptions(settings, "unified", false)).toEqual({
      resume_false_interruptions: true,
      adaptive_silence: true,
    });
    // A server set up on the Izwi server is named by its label alone, while
    // it is still there and no added server has taken its name.
    const local = {
      ...DEFAULT_VOICE_AGENT_SETTINGS,
      toolServers: [{ label: "stock", url: "http://h/mcp", token: "" }],
      configuredToolServers: ["files", "gone", "stock"],
    };
    expect(
      voiceStreamOptions(local, "modular", true, ["files", "stock"]).mcp_servers,
    ).toEqual([{ label: "stock", url: "http://h/mcp" }, { label: "files" }]);
    expect(
      parseVoiceAgentSettings(
        JSON.stringify({ configuredToolServers: ["files", "files", "a b", 3] }),
      ).configuredToolServers,
    ).toEqual(["files"]);
    // One agent alone is just the instructions; the defaults send nothing extra.
    expect(
      voiceStreamOptions({ ...DEFAULT_VOICE_AGENT_SETTINGS, agents: [front] }, "modular", true),
    ).toEqual({ resume_false_interruptions: true });
  });

  it("turn the built-in tools on by default only where they cost nothing", () => {
    expect(voiceBuiltinToolsOn(DEFAULT_VOICE_AGENT_SETTINGS, "unified")).toBe(true);
    expect(voiceBuiltinToolsOn(DEFAULT_VOICE_AGENT_SETTINGS, "modular")).toBe(false);
    expect(
      voiceBuiltinToolsOn({ ...DEFAULT_VOICE_AGENT_SETTINGS, builtinTools: true }, "modular"),
    ).toBe(true);
  });
});
