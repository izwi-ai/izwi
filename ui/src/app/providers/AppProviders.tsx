import type { ReactNode } from "react";
import { MotionConfig } from "framer-motion";
import { AnalyticsBootstrapProvider } from "@/app/analytics/AnalyticsBootstrapProvider";
import { AppUpdateProvider } from "@/app/providers/AppUpdateProvider";
import { ModelCatalogProvider } from "@/app/providers/ModelCatalogProvider";
import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { ThemeProvider } from "@/app/providers/ThemeProvider";

interface AppProvidersProps {
  children: ReactNode;
}

export function AppProviders({ children }: AppProvidersProps) {
  return (
    <MotionConfig reducedMotion="user">
      <AnalyticsBootstrapProvider>
        <ThemeProvider>
          <NotificationProvider>
            <AppUpdateProvider>
              <ModelCatalogProvider>{children}</ModelCatalogProvider>
            </AppUpdateProvider>
          </NotificationProvider>
        </ThemeProvider>
      </AnalyticsBootstrapProvider>
    </MotionConfig>
  );
}
