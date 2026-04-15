import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const RECORD_POLL_INTERVAL_MS = 2500;

interface SpeechTextRecordBase {
  id: string;
  processing_status: "pending" | "processing" | "ready" | "failed";
  summary_status?: "not_requested" | "pending" | "ready" | "failed";
}

export interface UseSpeechTextRecordResult<TRecord extends SpeechTextRecordBase> {
  record: TRecord | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

interface UseSpeechTextRecordOptions<TRecord extends SpeechTextRecordBase> {
  recordId: string | null | undefined;
  getRecord: (recordId: string) => Promise<TRecord>;
  loadErrorMessage: string;
  enablePolling?: boolean;
}

function defaultRecordPollingPredicate(record: SpeechTextRecordBase): boolean {
  if (
    record.processing_status === "pending" ||
    record.processing_status === "processing"
  ) {
    return true;
  }

  return record.summary_status === "pending";
}

export function useSpeechTextRecord<TRecord extends SpeechTextRecordBase>(
  options: UseSpeechTextRecordOptions<TRecord>,
): UseSpeechTextRecordResult<TRecord> {
  const { recordId, getRecord, loadErrorMessage, enablePolling = true } = options;

  const [record, setRecord] = useState<TRecord | null>(null);
  const [loading, setLoading] = useState(Boolean(recordId));
  const [error, setError] = useState<string | null>(null);
  const recordRef = useRef<TRecord | null>(null);

  useEffect(() => {
    recordRef.current = record;
  }, [record]);

  const loadRecord = useCallback(
    async (background = false) => {
      if (!recordId) {
        setRecord(null);
        setLoading(false);
        setError(null);
        return;
      }

      const hasVisibleRecord = recordRef.current !== null;
      if (!background || !hasVisibleRecord) {
        setLoading(true);
        setError(null);
      }

      try {
        const nextRecord = await getRecord(recordId);
        setRecord(nextRecord);
        setError(null);
      } catch (err) {
        if (!background || !hasVisibleRecord) {
          setRecord(null);
        }
        setError(err instanceof Error ? err.message : loadErrorMessage);
      } finally {
        setLoading(false);
      }
    },
    [getRecord, loadErrorMessage, recordId],
  );

  const refresh = useCallback(async () => {
    await loadRecord(recordRef.current !== null);
  }, [loadRecord]);

  useEffect(() => {
    recordRef.current = null;
    setRecord(null);
    setLoading(Boolean(recordId));
    setError(null);
    void loadRecord(false);
  }, [loadRecord, recordId]);

  const pollingRequired = useMemo(
    () => (record ? defaultRecordPollingPredicate(record) : false),
    [record],
  );

  useEffect(() => {
    if (!recordId || !enablePolling || !pollingRequired) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void loadRecord(true);
    }, RECORD_POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [enablePolling, loadRecord, pollingRequired, recordId]);

  return {
    record,
    loading,
    error,
    refresh,
  };
}
