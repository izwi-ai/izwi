import { act, fireEvent, render, screen } from "@testing-library/react";
import { Trash2 } from "lucide-react";
import { describe, expect, it, vi } from "vitest";

import { HistoryTable, type HistoryColumn } from "@/components/history/HistoryTable";
import { SearchField, SegmentedFilter } from "@/components/history/HistoryToolbar";
import { RowMenu } from "@/components/history/RowMenu";

interface Row {
  id: string;
  name: string;
}

const columns: HistoryColumn<Row>[] = [
  { id: "name", header: "Name", cell: (row) => row.name },
];

function renderTable(
  props: Partial<Parameters<typeof HistoryTable<Row>>[0]> = {},
) {
  const onOpenRow = vi.fn();
  const onDelete = vi.fn();
  render(
    <HistoryTable<Row>
      rows={[
        { id: "1", name: "fox.wav" },
        { id: "2", name: "nasa.wav" },
      ]}
      columns={columns}
      getRowId={(row) => row.id}
      getRowLabel={(row) => row.name}
      onOpenRow={onOpenRow}
      rowActions={(row) => (
        <RowMenu
          label={row.name}
          items={[
            {
              id: "delete",
              label: "Delete",
              icon: Trash2,
              danger: true,
              onSelect: () => onDelete(row.id),
            },
          ]}
        />
      )}
      loading={false}
      error={null}
      empty={{ title: "No files yet", description: "Files appear here." }}
      total={9}
      noun="files"
      {...props}
    />,
  );
  return { onOpenRow, onDelete };
}

describe("HistoryTable", () => {
  it("opens rows by click and keyboard and counts against the total", () => {
    const { onOpenRow } = renderTable();

    fireEvent.click(screen.getByText("fox.wav"));
    fireEvent.keyDown(screen.getByRole("row", { name: "Open nasa.wav" }), {
      key: "Enter",
    });

    expect(onOpenRow.mock.calls.map(([row]) => row.id)).toEqual(["1", "2"]);
    expect(screen.getByText("Showing 2 of 9 files")).toBeVisible();
  });

  it("runs row menu items without opening the row", async () => {
    const { onOpenRow, onDelete } = renderTable();

    const trigger = screen.getByRole("button", {
      name: "More actions for fox.wav",
    });
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false });
    fireEvent.click(trigger);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Delete" }));

    expect(onDelete).toHaveBeenCalledWith("1");
    expect(onOpenRow).not.toHaveBeenCalled();
  });

  it("tells an empty list from an empty search", () => {
    renderTable({ rows: [], total: 0 });
    expect(screen.getByText("No files yet")).toBeVisible();
  });

  it("says when filters match nothing", () => {
    renderTable({ rows: [], total: 0, filtered: true });
    expect(screen.getByText("No matching files")).toBeVisible();
  });
});

describe("SearchField", () => {
  it("reports the search after a pause and clears on Escape", () => {
    vi.useFakeTimers();
    try {
      const onChange = vi.fn();
      render(
        <SearchField value="" onChange={onChange} placeholder="Search files…" />,
      );
      const input = screen.getByRole("searchbox", { name: "Search files" });

      fireEvent.change(input, { target: { value: "fo" } });
      fireEvent.change(input, { target: { value: "fox" } });
      expect(onChange).not.toHaveBeenCalled();
      act(() => {
        vi.advanceTimersByTime(300);
      });
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(onChange).toHaveBeenLastCalledWith("fox");

      fireEvent.keyDown(input, { key: "Escape" });
      expect(onChange).toHaveBeenLastCalledWith("");
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("SegmentedFilter", () => {
  it("selects options and shows the count on the selected one", () => {
    const onChange = vi.fn();
    render(
      <SegmentedFilter
        label="Status"
        options={[
          { id: "all", label: "All" },
          { id: "failed", label: "Failed" },
        ]}
        value="all"
        onChange={onChange}
        selectedCount={63}
      />,
    );

    expect(screen.getByRole("radio", { name: "All 63" })).toBeChecked();
    fireEvent.click(screen.getByRole("radio", { name: "Failed" }));
    expect(onChange).toHaveBeenCalledWith("failed");
  });
});
