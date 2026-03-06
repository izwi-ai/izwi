import type { ReactNode } from "react";
import { ModelCatalogProvider } from "@/app/providers/ModelCatalogProvider";
import { ThemeProvider } from "@/app/providers/ThemeProvider";

interface AppProvidersProps {
  children: ReactNode;
}

export function AppProviders({ children }: AppProvidersProps) {
  return (
    <ThemeProvider>
      <ModelCatalogProvider>{children}</ModelCatalogProvider>
    </ThemeProvider>
  );
}
