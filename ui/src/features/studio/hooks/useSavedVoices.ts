import { useEffect, useState } from "react";

import { api, type SavedVoiceSummary } from "@/api";

/** The saved voices Studio can render with; empty until they load. */
export function useSavedVoices(): SavedVoiceSummary[] {
  const [voices, setVoices] = useState<SavedVoiceSummary[]>([]);
  useEffect(() => {
    let current = true;
    api
      .listSavedVoices()
      .then((loaded) => {
        if (current) setVoices(loaded);
      })
      .catch(() => undefined);
    return () => {
      current = false;
    };
  }, []);
  return voices;
}
