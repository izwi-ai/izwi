import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ChatThread } from "@/api";
import {
  ChatThreadList,
  threadDateGroup,
} from "@/features/chat/components/ChatThreadList";

const DAY = 24 * 60 * 60 * 1000;

function thread(id: string, title: string, updatedAt: number, extra = {}): ChatThread {
  return {
    id,
    title,
    model_id: null,
    system_prompt: null,
    created_at: updatedAt,
    updated_at: updatedAt,
    last_message_preview: null,
    message_count: 2,
    ...extra,
  };
}

describe("threadDateGroup", () => {
  it("buckets by recency, then by month", () => {
    const now = new Date(2026, 9, 18, 15);
    expect(threadDateGroup(now.getTime() - 60_000, now)).toBe("Today");
    expect(threadDateGroup(now.getTime() - DAY, now)).toBe("Yesterday");
    expect(threadDateGroup(now.getTime() - 4 * DAY, now)).toBe("Previous 7 days");
    expect(threadDateGroup(now.getTime() - 20 * DAY, now)).toBe("Previous 30 days");
    expect(threadDateGroup(new Date(2026, 6, 2).getTime(), now)).toMatch(/July/);
    expect(threadDateGroup(new Date(2025, 6, 2).getTime(), now)).toMatch(/2025/);
  });
});

describe("ChatThreadList", () => {
  it("groups, searches, selects and deletes threads", async () => {
    const onSelect = vi.fn();
    const onDelete = vi.fn();
    const now = Date.now();
    render(
      <ChatThreadList
        threads={[
          thread("a", "Screenshot question", now, { has_images: true }),
          thread("b", "Podcast intro", now - 3 * DAY),
        ]}
        activeThreadId="a"
        loading={false}
        busy={false}
        onSelect={onSelect}
        onDelete={onDelete}
        onNewChat={vi.fn()}
      />,
    );

    const chats = screen.getByRole("navigation", { name: "Chats" });
    expect(within(chats).getByText("Today")).toBeVisible();
    expect(within(chats).getByText("Previous 7 days")).toBeVisible();
    expect(
      within(chats).getByRole("button", { name: /^Screenshot question\s*\(has images\)$/ }),
    ).toHaveAttribute("aria-current", "page");

    fireEvent.click(within(chats).getByRole("button", { name: "Podcast intro" }));
    expect(onSelect).toHaveBeenCalledWith("b");
    fireEvent.click(
      within(chats).getByRole("button", { name: "Delete Podcast intro" }),
    );
    expect(onDelete).toHaveBeenCalledWith("b");

    fireEvent.change(screen.getByRole("searchbox", { name: "Search chats" }), {
      target: { value: "podcast" },
    });
    await waitFor(() =>
      expect(
        within(chats).queryByRole("button", { name: /^Screenshot question\s*\(has images\)$/ }),
      ).not.toBeInTheDocument(),
    );
  });
});
