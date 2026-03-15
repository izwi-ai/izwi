import type { ReactNode } from "react";
import { ModelCatalogProvider } from "@/app/providers/ModelCatalogProvider";
import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { ThemeProvider } from "@/app/providers/ThemeProvider";

interface AppProvidersProps {
  children: ReactNode;
}

export function AppProviders({ children }: AppProvidersProps) {
  return (
    <ThemeProvider>
      <NotificationProvider>
        <ModelCatalogProvider>{children}</ModelCatalogProvider>
      </NotificationProvider>
    </ThemeProvider>
  );
}
