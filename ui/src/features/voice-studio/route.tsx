import { useState } from "react";
import { Plus } from "lucide-react";
import type { ModelInfo } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { VoicesPage } from "@/features/voices/route";

interface VoiceStudioPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
}

export function VoiceStudioPage({
  models,
  selectedModel,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onSelect,
  onError,
}: VoiceStudioPageProps) {
  const [isCreationModalOpen, setIsCreationModalOpen] = useState(false);

  return (
    <PageShell>
      <PageHeader
        title="Voice Studio"
        description="Manage saved and built-in voices, then create new cloned or designed voices from one place."
        actions={
          <Button
            onClick={() => setIsCreationModalOpen(true)}
            className="h-9 rounded-[var(--radius-pill)] px-4 text-sm"
          >
            <Plus className="h-4 w-4" />
            New Voice
          </Button>
        }
      />

      <div className="mt-5 pb-4 sm:pb-5">
        <VoicesPage
          models={models}
          selectedModel={selectedModel}
          loading={loading}
          downloadProgress={downloadProgress}
          onDownload={onDownload}
          onCancelDownload={onCancelDownload}
          onLoad={onLoad}
          onUnload={onUnload}
          onDelete={onDelete}
          onSelect={onSelect}
          onError={onError}
          embedded
          onAddNewVoice={() => setIsCreationModalOpen(true)}
        />
      </div>

      <Dialog open={isCreationModalOpen} onOpenChange={setIsCreationModalOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>New Voice</DialogTitle>
            <DialogDescription>
              Choose a workflow to create a new voice profile for Text to Speech.
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3 text-sm text-[var(--text-secondary)]">
            Voice creation flow setup continues in the next rollout step.
          </div>
        </DialogContent>
      </Dialog>
    </PageShell>
  );
}
