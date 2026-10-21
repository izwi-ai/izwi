import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

import { usePageTitle } from "@/app/router/DocumentTitle";
import { RowMenu, type RowMenuItem } from "@/components/history/RowMenu";
import { PageHeader, type Breadcrumb } from "@/components/PageShell";

export interface RecordMetaItem {
  icon: LucideIcon;
  label: string;
  /** Accessible name when the label alone is ambiguous ("Model"). */
  title?: string;
}

interface RecordPageProps {
  title: string;
  breadcrumbs: Breadcrumb[];
  /** Facts under the title: date, length, language, model, … */
  meta: RecordMetaItem[];
  /** Header buttons (Copy, Export). */
  actions?: ReactNode;
  /** The header's "…" menu (regenerate, cancel, models, delete). */
  menuItems?: RowMenuItem[];
  /** Status, warnings and errors above the content. */
  alerts?: ReactNode;
  children: ReactNode;
}

/** Layout of a record page: one header with the record's facts, then content. */
export function RecordPage({
  title,
  breadcrumbs,
  meta,
  actions,
  menuItems = [],
  alerts,
  children,
}: RecordPageProps) {
  usePageTitle(title);
  return (
    <>
      <PageHeader
        breadcrumbs={breadcrumbs}
        title={title}
        className="mb-5"
        description={
          meta.length > 0 ? (
            <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-(--text-muted)">
              {meta.map((item) => (
                <span
                  key={`${item.title ?? ""}${item.label}`}
                  className="inline-flex items-center gap-1.5"
                  title={item.title}
                >
                  <item.icon
                    aria-hidden="true"
                    className="h-3.5 w-3.5 text-(--text-subtle)"
                  />
                  {item.title ? (
                    <span className="sr-only">{item.title}: </span>
                  ) : null}
                  {item.label}
                </span>
              ))}
            </span>
          ) : undefined
        }
        actions={
          actions || menuItems.length > 0 ? (
            <>
              {actions}
              {menuItems.length > 0 ? (
                <RowMenu
                  label={title}
                  items={menuItems}
                  triggerClassName="h-9 w-9 rounded-[10px] border border-(--border-default)"
                />
              ) : null}
            </>
          ) : undefined
        }
      />
      <div className="space-y-5">
        {alerts}
        {children}
      </div>
    </>
  );
}
