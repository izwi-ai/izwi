import type { ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface StudioProjectUtilitiesProps {
  profilePanel: ReactNode;
  pronunciationPanel: ReactNode;
  snapshotsPanel: ReactNode;
}

export function StudioProjectUtilities({
  profilePanel,
  pronunciationPanel,
  snapshotsPanel,
}: StudioProjectUtilitiesProps) {
  return (
    <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none sm:p-6">
      <div className="flex flex-col gap-1">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Project Utilities
        </div>
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">
          Configure and maintain this project
        </h3>
      </div>

      <Tabs defaultValue="profile" className="mt-4">
        <TabsList className="h-9 w-full justify-start gap-1 overflow-x-auto rounded-xl border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-1">
          <TabsTrigger
            value="profile"
            className="h-7 rounded-lg px-2 text-xs font-semibold"
          >
            Profile
          </TabsTrigger>
          <TabsTrigger
            value="pronunciation"
            className="h-7 rounded-lg px-2 text-xs font-semibold"
          >
            Pronunciations
          </TabsTrigger>
          <TabsTrigger
            value="snapshots"
            className="h-7 rounded-lg px-2 text-xs font-semibold"
          >
            Snapshots
          </TabsTrigger>
        </TabsList>

        <TabsContent value="profile" className="mt-4">
          {profilePanel}
        </TabsContent>
        <TabsContent value="pronunciation" className="mt-4">
          {pronunciationPanel}
        </TabsContent>
        <TabsContent value="snapshots" className="mt-4">
          {snapshotsPanel}
        </TabsContent>
      </Tabs>
    </Card>
  );
}
