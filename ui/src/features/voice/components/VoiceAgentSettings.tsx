import { useId, useState, type ReactNode } from "react";
import { ChevronRight, Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  MAX_VOICE_AGENTS,
  MAX_VOICE_TOOL_SERVERS,
  voiceAgentId,
  voiceBuiltinToolsOn,
  voiceToolServerProblem,
  withoutVoiceAgent,
  type VoiceAgent,
  type VoiceAgentSettings,
} from "@/features/voice/agentSettings";
import type { VoiceConfiguredToolServer } from "@/api";
import { cn } from "@/lib/utils";

const sectionTitle = "mb-1.5 text-[13px] font-medium text-(--text-muted)";
const rowLink =
  "flex min-h-[52px] w-full items-center justify-between gap-4 text-left text-sm text-(--text-primary) transition-colors hover:text-(--text-secondary)";
const field =
  "w-full rounded-[10px] border border-(--border-default) bg-(--bg-surface-2) px-3 py-2 text-sm text-(--text-primary) outline-hidden transition-colors placeholder:text-(--text-subtle) focus:border-(--border-strong) disabled:opacity-60";
const fieldLabel = "text-xs font-medium text-(--text-muted)";
const note = "text-[13px] leading-relaxed text-(--text-muted)";

interface AgentSettingsProps {
  settings: VoiceAgentSettings;
  onChange: (change: Partial<VoiceAgentSettings>) => void;
  /** The engine in use: agents and an early reply need the standard one. */
  mode: "modular" | "unified";
  /** Whether the engine's model can call tools at all. */
  callsTools: boolean;
  /** The MCP servers set up on the Izwi server. */
  configured: VoiceConfiguredToolServer[];
}

function SwitchRow({
  title,
  detail,
  checked,
  disabled,
  onCheckedChange,
}: {
  title: string;
  detail: ReactNode;
  checked: boolean;
  disabled?: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const id = useId();
  return (
    <div className="flex min-h-[52px] items-center justify-between gap-4 py-1.5">
      <label htmlFor={id} className="flex flex-col gap-0.5">
        <span className="text-sm font-medium text-(--text-primary)">{title}</span>
        <span className="text-xs text-(--text-muted)">{detail}</span>
      </label>
      <Switch
        id={id}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onCheckedChange}
      />
    </div>
  );
}

/** How the conversation takes turns: three switches on the main view. */
export function ConversationFlowSection({ settings, onChange, mode }: AgentSettingsProps) {
  return (
    <section className="flex flex-col">
      <h3 className={sectionTitle}>Conversation</h3>
      <SwitchRow
        title="Play on through noise"
        detail="A cough or a hum pauses the reply instead of ending it"
        checked={settings.resumeFalseInterruptions}
        onCheckedChange={(checked) => onChange({ resumeFalseInterruptions: checked })}
      />
      <SwitchRow
        title="Follow my pace"
        detail="Waits less after you stop if you rarely pause mid-sentence"
        checked={settings.adaptiveSilence}
        onCheckedChange={(checked) => onChange({ adaptiveSilence: checked })}
      />
      <SwitchRow
        title="Start replies sooner"
        detail={
          mode === "modular"
            ? "Begins the answer while checking you have finished. Uses more power."
            : "Available with the Standard engine"
        }
        checked={mode === "modular" && settings.replySooner}
        disabled={mode !== "modular"}
        onCheckedChange={(checked) => onChange({ replySooner: checked })}
      />
    </section>
  );
}

/** The two rows on the main view that open the tools and agents views. */
export function ToolsAndAgentsSection({
  settings,
  mode,
  callsTools,
  configured,
  onOpenTools,
  onOpenAgents,
}: AgentSettingsProps & { onOpenTools: () => void; onOpenAgents: () => void }) {
  const servers =
    settings.toolServers.length +
    configured.filter((server) =>
      settings.configuredToolServers.includes(server.label),
    ).length;
  const tools = [
    callsTools && voiceBuiltinToolsOn(settings) ? "Date and time" : null,
    callsTools && servers > 0
      ? `${servers} tool server${servers === 1 ? "" : "s"}`
      : null,
  ].filter(Boolean);
  const agents =
    mode !== "modular"
      ? "Standard engine only"
      : settings.agents.length >= 2
        ? `${settings.agents.length} agents`
        : "One assistant";
  return (
    <section className="flex flex-col">
      <h3 className={sectionTitle}>Tools and agents</h3>
      <button type="button" onClick={onOpenTools} className={rowLink}>
        <span className="font-medium">Tools</span>
        <span className="flex items-center gap-1.5 text-[13px] text-(--text-muted)">
          {callsTools ? tools.join(" · ") || "None" : "Not with this model"}
          <ChevronRight className="h-4 w-4" />
        </span>
      </button>
      <button
        type="button"
        onClick={onOpenAgents}
        className={cn(rowLink, "border-t border-(--border-muted)")}
      >
        <span className="font-medium">Agents</span>
        <span className="flex items-center gap-1.5 text-[13px] text-(--text-muted)">
          {agents}
          <ChevronRight className="h-4 w-4" />
        </span>
      </button>
    </section>
  );
}

/** The tools the assistant may call: the built-in ones and tool servers. */
export function ToolsView({
  settings,
  onChange,
  mode,
  callsTools,
  configured,
}: AgentSettingsProps) {
  const nameId = useId();
  const addressId = useId();
  const tokenId = useId();
  const [draft, setDraft] = useState({ label: "", url: "", token: "" });
  const [problem, setProblem] = useState<string | null>(null);
  const atLimit = settings.toolServers.length >= MAX_VOICE_TOOL_SERVERS;

  const add = () => {
    const server = {
      label: draft.label.trim(),
      url: draft.url.trim(),
      token: draft.token.trim(),
    };
    const found = voiceToolServerProblem(server, settings.toolServers);
    setProblem(found);
    if (found) return;
    onChange({ toolServers: [...settings.toolServers, server] });
    setDraft({ label: "", url: "", token: "" });
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-7 overflow-y-auto px-6 pb-8 pt-5">
      {!callsTools && (
        <p className={note}>
          This model can’t use tools. Switch to LFM3 Audio or the Standard
          engine to turn them on.
        </p>
      )}
      <section className="flex flex-col">
        <h3 className={sectionTitle}>Built in</h3>
        <SwitchRow
          title="Date and time"
          detail={
            mode === "modular"
              ? "Lets the assistant check today’s date and the time. Replies take a little longer while any tool is on."
              : "Rarely needed: the Compact engine is already told the date and time. With this on, it may read them out when you say hello."
          }
          checked={callsTools && voiceBuiltinToolsOn(settings)}
          disabled={!callsTools}
          onCheckedChange={(checked) => onChange({ builtinTools: checked })}
        />
      </section>

      {configured.length > 0 && (
        <section className="flex flex-col">
          <h3 className={sectionTitle}>Set up on this server</h3>
          {configured.map((server) => (
            <SwitchRow
              key={server.label}
              title={server.label}
              detail={
                server.transport === "stdio"
                  ? "Runs as a program on the computer Izwi runs on"
                  : "A tool server Izwi connects to"
              }
              checked={
                callsTools && settings.configuredToolServers.includes(server.label)
              }
              disabled={!callsTools}
              onCheckedChange={(checked) =>
                onChange({
                  configuredToolServers: checked
                    ? [...settings.configuredToolServers, server.label]
                    : settings.configuredToolServers.filter(
                        (label) => label !== server.label,
                      ),
                })
              }
            />
          ))}
        </section>
      )}

      <section className="flex flex-col gap-3">
        <h3 className={cn(sectionTitle, "mb-0")}>Tool servers</h3>
        <p className={note}>
          Connect an MCP server and the assistant can call its tools. Izwi
          connects to it from the server, which must allow the address in{" "}
          <code className="text-xs">IZWI_MCP_ALLOWED_HOSTS</code>. A server
          that runs as a program is set up in Izwi’s{" "}
          <code className="text-xs">mcp.json</code> file instead.
        </p>
        {settings.toolServers.length > 0 && (
          <ul className="flex flex-col">
            {settings.toolServers.map((server) => (
              <li
                key={server.label}
                className="flex min-h-[52px] items-center justify-between gap-3 border-b border-(--border-muted) last:border-b-0"
              >
                <span className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-sm font-medium text-(--text-primary)">
                    {server.label}
                  </span>
                  <span className="truncate text-xs text-(--text-muted)">{server.url}</span>
                </span>
                <button
                  type="button"
                  aria-label={`Remove ${server.label}`}
                  onClick={() =>
                    onChange({
                      toolServers: settings.toolServers.filter(
                        (other) => other.label !== server.label,
                      ),
                    })
                  }
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[10px] text-(--text-muted) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary)"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </li>
            ))}
          </ul>
        )}
        {!atLimit && (
          <form
            className="flex flex-col gap-2.5"
            onSubmit={(event) => {
              event.preventDefault();
              add();
            }}
          >
            <label htmlFor={nameId} className={fieldLabel}>
              Name
            </label>
            <input
              id={nameId}
              value={draft.label}
              placeholder="stock"
              onChange={(event) => setDraft({ ...draft, label: event.target.value })}
              className={field}
            />
            <label htmlFor={addressId} className={fieldLabel}>
              Address
            </label>
            <input
              id={addressId}
              value={draft.url}
              inputMode="url"
              placeholder="https://example.com/mcp"
              onChange={(event) => setDraft({ ...draft, url: event.target.value })}
              className={field}
            />
            <label htmlFor={tokenId} className={fieldLabel}>
              Access token (optional)
            </label>
            <input
              id={tokenId}
              type="password"
              autoComplete="off"
              value={draft.token}
              onChange={(event) => setDraft({ ...draft, token: event.target.value })}
              className={field}
            />
            {problem && (
              <p role="alert" className="text-xs text-(--danger-text)">
                {problem}
              </p>
            )}
            <Button type="submit" variant="outline" size="sm" className="mt-1 self-start">
              <Plus className="h-3.5 w-3.5" />
              Add tool server
            </Button>
          </form>
        )}
      </section>
      <p className="text-xs text-(--text-subtle)">Applies to your next conversation</p>
    </div>
  );
}

function AgentCard({
  agent,
  index,
  others,
  onChange,
  onRemove,
}: {
  agent: VoiceAgent;
  index: number;
  others: VoiceAgent[];
  onChange: (change: Partial<VoiceAgent>) => void;
  onRemove: () => void;
}) {
  const nameId = useId();
  const instructionsId = useId();
  const whenId = useId();
  return (
    <li className="flex flex-col gap-2.5 rounded-xl border border-(--border-default) p-3.5">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs text-(--text-subtle)">
          {index === 0 ? "Answers first" : `Agent ${index + 1}`}
        </span>
        <button
          type="button"
          aria-label={`Remove ${agent.name}`}
          onClick={onRemove}
          className="flex h-8 w-8 items-center justify-center rounded-[10px] text-(--text-muted) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary)"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
      <label htmlFor={nameId} className={fieldLabel}>
        Name
      </label>
      <input
        id={nameId}
        value={agent.name}
        onChange={(event) => onChange({ name: event.target.value })}
        className={field}
      />
      <label htmlFor={instructionsId} className={fieldLabel}>
        What this agent does
      </label>
      <textarea
        id={instructionsId}
        rows={3}
        value={agent.instructions}
        placeholder="You handle refunds. Ask for the invoice number."
        onChange={(event) => onChange({ instructions: event.target.value })}
        className={cn(field, "resize-y leading-relaxed")}
      />
      {index > 0 && (
        <>
          <label htmlFor={whenId} className={fieldLabel}>
            When to hand over to it
          </label>
          <input
            id={whenId}
            value={agent.handoffDescription}
            placeholder="For invoices and refunds"
            onChange={(event) => onChange({ handoffDescription: event.target.value })}
            className={field}
          />
        </>
      )}
      {others.length > 0 && (
        <fieldset className="flex flex-col gap-1.5">
          <legend className={cn(fieldLabel, "mb-1")}>Can hand over to</legend>
          {others.map((other) => {
            const checked = agent.handoffs.includes(other.id);
            return (
              <label
                key={other.id}
                className="flex min-h-9 items-center gap-2.5 text-sm text-(--text-primary)"
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() =>
                    onChange({
                      handoffs: checked
                        ? agent.handoffs.filter((to) => to !== other.id)
                        : [...agent.handoffs, other.id],
                    })
                  }
                  className="h-4 w-4 accent-(--accent-solid)"
                />
                {other.name || other.id}
              </label>
            );
          })}
        </fieldset>
      )}
    </li>
  );
}

/** The agents a conversation is split between, and who hands over to whom. */
export function AgentsView({ settings, onChange, mode }: AgentSettingsProps) {
  const { agents } = settings;
  const update = (id: string, change: Partial<VoiceAgent>) =>
    onChange({
      agents: agents.map((agent) => (agent.id === id ? { ...agent, ...change } : agent)),
    });
  const add = () => {
    const name = agents.length === 0 ? "Assistant" : `Agent ${agents.length + 1}`;
    onChange({
      agents: [
        ...agents,
        {
          id: voiceAgentId(name, agents.map((agent) => agent.id)),
          name,
          instructions: "",
          handoffDescription: "",
          handoffs: [],
        },
      ],
    });
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 pb-8 pt-5">
      <p className={note}>
        Split a conversation between agents, each with its own job. The first
        one answers, and an agent can hand the conversation to the ones you
        tick. Your instructions in Voice settings apply to all of them.
      </p>
      {mode !== "modular" && (
        <p className={note}>
          Agents need the Standard engine. With the Compact engine the
          conversation uses your instructions alone.
        </p>
      )}
      {agents.length === 1 && (
        <p className={note}>Add a second agent for them to take effect.</p>
      )}
      {agents.length > 0 && (
        <ol className="flex flex-col gap-3">
          {agents.map((agent, index) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              index={index}
              others={agents.filter((other) => other.id !== agent.id)}
              onChange={(change) => update(agent.id, change)}
              onRemove={() => onChange({ agents: withoutVoiceAgent(agents, agent.id) })}
            />
          ))}
        </ol>
      )}
      {agents.length < MAX_VOICE_AGENTS && (
        <Button type="button" variant="outline" size="sm" className="self-start" onClick={add}>
          <Plus className="h-3.5 w-3.5" />
          Add agent
        </Button>
      )}
      <p className="text-xs text-(--text-subtle)">Applies to your next conversation</p>
    </div>
  );
}
