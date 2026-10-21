import { useEffect, useSyncExternalStore } from "react";
import { useLocation } from "react-router";

import {
  documentTitleForLocation,
  withAppName,
} from "@/app/router/routePresentation";

// A page can name itself more precisely than its route (a record's title).
let pageTitle: string | null = null;
const listeners = new Set<() => void>();

function setPageTitle(next: string | null) {
  if (pageTitle !== next) {
    pageTitle = next;
    listeners.forEach((listener) => listener());
  }
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Name the window after this page's content while it is mounted. */
export function usePageTitle(title: string | null | undefined) {
  useEffect(() => {
    const trimmed = title?.trim() || null;
    setPageTitle(trimmed);
    return () => setPageTitle(null);
  }, [title]);
}

export function DocumentTitle() {
  const location = useLocation();
  const override = useSyncExternalStore(subscribe, () => pageTitle);

  useEffect(() => {
    document.title = override
      ? withAppName(override)
      : documentTitleForLocation(location.pathname, location.search);
  }, [location.pathname, location.search, override]);

  return null;
}
