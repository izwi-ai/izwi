import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "@/app/router/AppRoutes";

export function AppRouter() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
