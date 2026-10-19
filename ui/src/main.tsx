import React from "react";
import ReactDOM from "react-dom/client";
// Bundled locally so the UI renders in Inter on every OS and offline,
// instead of falling back to each platform's system font.
import "@fontsource-variable/inter";
import { bootstrapDocumentIcons } from "@/app/bootstrap/icons";
import { bootstrapDocumentTheme } from "@/app/bootstrap/theme";
import { AppProviders } from "@/app/providers/AppProviders";
import { AppRouter } from "@/app/router/AppRouter";
import "./index.css";

bootstrapDocumentIcons();
bootstrapDocumentTheme();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppProviders>
      <AppRouter />
    </AppProviders>
  </React.StrictMode>,
);
