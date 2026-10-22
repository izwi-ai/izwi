import { useEffect } from "react";
import { BrowserRouter, useNavigate } from "react-router";
import { useAppUpdates } from "@/app/providers/AppUpdateProvider";
import { AppRoutes } from "@/app/router/AppRoutes";

interface TrayRouteDetail {
  path?: unknown;
}

function DesktopTrayBridge() {
  const navigate = useNavigate();
  const { checkForUpdates } = useAppUpdates();

  useEffect(() => {
    const onTrayRoute = (event: Event) => {
      const routeEvent = event as CustomEvent<TrayRouteDetail>;
      const path = routeEvent.detail?.path;
      if (typeof path !== "string" || !path.startsWith("/")) {
        return;
      }
      navigate(path);
    };
    const onTrayCheckUpdates = () => {
      void checkForUpdates(true);
    };

    window.addEventListener("izwi:tray-route", onTrayRoute as EventListener);
    window.addEventListener("izwi:tray-check-updates", onTrayCheckUpdates);

    return () => {
      window.removeEventListener("izwi:tray-route", onTrayRoute as EventListener);
      window.removeEventListener("izwi:tray-check-updates", onTrayCheckUpdates);
    };
  }, [checkForUpdates, navigate]);

  return null;
}

export function AppRouter() {
  return (
    <BrowserRouter>
      <DesktopTrayBridge />
      <AppRoutes />
    </BrowserRouter>
  );
}
