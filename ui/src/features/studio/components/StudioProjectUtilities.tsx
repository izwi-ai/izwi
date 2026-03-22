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
    <Card className="rounded-2xl border-0 bg-transparent p-0 shadow-none">
      <div className="flex flex-col gap-1">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Project Utilities
        </div>
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">
          Configure and maintain this project
        </h3>
      </div>

      <Tabs defaultValue="profile" className="mt-4">
        <TabsList className="h-auto w-full justify-start gap-1 rounded-xl border-0 bg-[var(--bg-surface-1)] p-1">
          <TabsTrigger
            value="profile"
            className="h-7 min-w-[92px] rounded-lg px-2 text-xs font-semibold"
          >
            Profile
          </TabsTrigger>
          <TabsTrigger
            value="pronunciation"
            className="h-7 min-w-[110px] rounded-lg px-2 text-xs font-semibold"
          >
            Pronunciations
          </TabsTrigger>
          <TabsTrigger
            value="snapshots"
            className="h-7 min-w-[96px] rounded-lg px-2 text-xs font-semibold"
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
