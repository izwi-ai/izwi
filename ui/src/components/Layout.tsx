import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Users,
  Wand2,
  FileText,
  MessageSquare,
  AudioLines,
  Box,
  Github,
  AlertCircle,
  X,
  Menu,
  Sun,
  Moon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const appIconUrl = `/app-icon.png?v=${Date.now()}`;
const APP_VERSION = `v${__APP_VERSION__}`;

interface LayoutProps {
  error: string | null;
  onErrorDismiss: () => void;
  readyModelsCount: number;
  resolvedTheme: "light" | "dark";
  themePreference: "system" | "light" | "dark";
  onThemePreferenceChange: (preference: "system" | "light" | "dark") => void;
}

interface NavItem {
  id: string;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  path: string;
}

const TOP_NAV_ITEMS: NavItem[] = [
  {
    id: "voice",
    label: "Voice",
    description: "Flagship realtime interaction",
    icon: AudioLines,
    path: "/voice",
  },
  {
    id: "chat",
    label: "Chat",
    description: "Standard AI interaction hub",
    icon: MessageSquare,
    path: "/chat",
  },
  {
    id: "transcription",
    label: "Transcription",
    description: "Input utility for audio workflows",
    icon: FileText,
    path: "/transcription",
  },
  {
    id: "diarization",
    label: "Diarization",
    description: "Speaker segmentation with timestamps",
    icon: Users,
    path: "/diarization",
  },
];

const CREATION_NAV_ITEMS: NavItem[] = [
  {
    id: "text-to-speech",
    label: "Text to Speech",
    description: "Output speech from text",
    icon: Mic,
    path: "/text-to-speech",
  },
  {
    id: "voice-cloning",
    label: "Voice Cloning",
    description: "Identity personalization from reference audio",
    icon: Users,
    path: "/voice-cloning",
  },
  {
    id: "voice-design",
    label: "Voice Design",
    description: "Create voices from descriptions",
    icon: Wand2,
    path: "/voice-design",
  },
];

const BOTTOM_NAV_ITEMS: NavItem[] = [
  {
    id: "models",
    label: "Models",
    description: "Manage your downloaded models",
    icon: Box,
    path: "/models",
  },
];

export function Layout({
  error,
  onErrorDismiss,
  readyModelsCount,
  resolvedTheme,
  themePreference,
  onThemePreferenceChange,
}: LayoutProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return window.localStorage.getItem("izwi.sidebar.collapsed") === "1";
  });

  useEffect(() => {
    window.localStorage.setItem(
      "izwi.sidebar.collapsed",
      isSidebarCollapsed ? "1" : "0",
    );
  }, [isSidebarCollapsed]);

  const loadedText =
    readyModelsCount > 0
      ? `${readyModelsCount} model${readyModelsCount !== 1 ? "s" : ""} loaded`
      : "No models loaded";

  const switchTheme = () => {
    onThemePreferenceChange(resolvedTheme === "dark" ? "light" : "dark");
  };

  const handleNavClick = (path: string) => {
    setMobileMenuOpen(false);
    if (
      path === "/chat" &&
      typeof window !== "undefined" &&
      window.innerWidth >= 1024
    ) {
      setIsSidebarCollapsed(true);
    }
  };

  return (
    <div className="h-dvh flex overflow-hidden bg-transparent text-foreground selection:bg-primary/25 selection:text-foreground">
      {/* Mobile header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 border-b border-border/80 bg-background/78 backdrop-blur-xl">
        <div className="flex items-center justify-between px-4 py-3.5">
          <div className="flex items-center gap-3">
            <div className="relative w-8 h-8 rounded-xl overflow-hidden border border-border/80 bg-card shadow-sm">
              <img
                src={appIconUrl}
                alt="Izwi logo"
                className="w-full h-full object-cover p-0.5 brightness-125 contrast-125"
              />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-foreground">Izwi</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={switchTheme}
              className="h-8 w-8 rounded-lg"
              title={
                resolvedTheme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
            >
              {resolvedTheme === "dark" ? (
                <Sun className="w-4 h-4 text-foreground" />
              ) : (
                <Moon className="w-4 h-4 text-foreground" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="h-8 w-8 rounded-lg"
            >
              <Menu className="w-5 h-5 text-foreground" />
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile menu overlay */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setMobileMenuOpen(false)}
            className="lg:hidden fixed inset-0 bg-background/68 backdrop-blur-sm z-40"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside
        className={cn(
          "w-[16rem] border-r border-border/40 flex flex-col fixed h-full z-50 bg-background/95 backdrop-blur-xl transition-all duration-300",
          "lg:translate-x-0",
          isSidebarCollapsed ? "lg:w-[72px]" : "lg:w-[16rem]",
          mobileMenuOpen ? "translate-x-0 shadow-2xl" : "-translate-x-full",
        )}
      >
        {/* Logo - hidden on mobile since it's in the header */}
        <div
          className={cn(
            "hidden lg:flex h-16 shrink-0",
            isSidebarCollapsed
              ? "flex-col items-center justify-center gap-2"
              : "items-center justify-between px-5",
          )}
        >
          <div
            className={cn(
              "flex items-center gap-3",
              isSidebarCollapsed && "hidden",
            )}
          >
            <div className="relative w-8 h-8 rounded-lg overflow-hidden border border-border/40 shadow-sm flex-shrink-0 bg-background">
              <img
                src={appIconUrl}
                alt="Izwi logo"
                className="w-full h-full object-cover p-0.5 brightness-110"
              />
            </div>
            <div>
              <h1 className="text-base font-medium text-foreground tracking-tight">
                Izwi
              </h1>
            </div>
          </div>
          {isSidebarCollapsed && (
            <div className="relative w-8 h-8 rounded-lg overflow-hidden border border-border/40 shadow-sm flex-shrink-0 bg-background mb-2 mt-2">
              <img
                src={appIconUrl}
                alt="Izwi logo"
                className="w-full h-full object-cover p-0.5 brightness-110"
              />
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsSidebarCollapsed((collapsed) => !collapsed)}
            className="h-8 w-8 rounded-md text-muted-foreground hover:text-foreground"
            title={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <Menu className="w-4 h-4" />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 overflow-y-auto flex flex-col scrollbar-thin gap-6">
          <div className="space-y-1">
            <h4
              className={cn(
                "px-3 text-[11px] font-medium text-muted-foreground mb-2",
                isSidebarCollapsed && "sr-only",
              )}
            >
              Workspace
            </h4>
            {TOP_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => handleNavClick(item.path)}
                className={({ isActive }) =>
                  cn(
                    "sidebar-link flex items-center rounded-lg border transition-all group",
                    isSidebarCollapsed
                      ? "justify-center p-2.5 mx-auto w-10 h-10"
                      : "gap-3 px-3 py-2.5 mx-2",
                    isActive
                      ? "sidebar-link-active text-foreground font-medium shadow-sm"
                      : "sidebar-link-idle text-muted-foreground hover:text-foreground",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={cn(
                        "sidebar-link-icon flex items-center justify-center transition-colors",
                        isActive
                          ? "sidebar-link-icon-active"
                          : "sidebar-link-icon-idle group-hover:sidebar-link-icon-hover",
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                    </div>
                    <div
                      className={cn(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
                      <div
                        className={cn(
                          "text-sm truncate leading-tight",
                          isActive
                            ? "sidebar-link-title-active"
                            : "sidebar-link-title-idle group-hover:sidebar-link-title-hover",
                        )}
                      >
                        {item.label}
                      </div>
                    </div>
                  </>
                )}
              </NavLink>
            ))}
          </div>

          <div className="space-y-1">
            <h4
              className={cn(
                "px-3 text-[11px] font-medium text-muted-foreground mb-2",
                isSidebarCollapsed && "sr-only",
              )}
            >
              Creation
            </h4>
            {CREATION_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => handleNavClick(item.path)}
                className={({ isActive }) =>
                  cn(
                    "sidebar-link flex items-center rounded-lg border transition-all group",
                    isSidebarCollapsed
                      ? "justify-center p-2.5 mx-auto w-10 h-10"
                      : "gap-3 px-3 py-2.5 mx-2",
                    isActive
                      ? "sidebar-link-active text-foreground font-medium shadow-sm"
                      : "sidebar-link-idle text-muted-foreground hover:text-foreground",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={cn(
                        "sidebar-link-icon flex items-center justify-center transition-colors",
                        isActive
                          ? "sidebar-link-icon-active"
                          : "sidebar-link-icon-idle group-hover:sidebar-link-icon-hover",
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                    </div>
                    <div
                      className={cn(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
                      <div
                        className={cn(
                          "text-sm truncate leading-tight",
                          isActive
                            ? "sidebar-link-title-active"
                            : "sidebar-link-title-idle group-hover:sidebar-link-title-hover",
                        )}
                      >
                        {item.label}
                      </div>
                    </div>
                  </>
                )}
              </NavLink>
            ))}
          </div>

          {/* Bottom navigation section */}
          <div className="mt-auto space-y-1 pt-4 border-t border-border/40">
            {BOTTOM_NAV_ITEMS.map((item) => (
              <NavLink
                key={item.id}
                to={item.path}
                title={isSidebarCollapsed ? item.label : undefined}
                onClick={() => handleNavClick(item.path)}
                className={({ isActive }) =>
                  cn(
                    "sidebar-link flex items-center rounded-lg border transition-all group",
                    isSidebarCollapsed
                      ? "justify-center p-2.5 mx-auto w-10 h-10"
                      : "gap-3 px-3 py-2.5 mx-2",
                    isActive
                      ? "sidebar-link-active text-foreground font-medium shadow-sm"
                      : "sidebar-link-idle text-muted-foreground hover:text-foreground",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div
                      className={cn(
                        "sidebar-link-icon flex items-center justify-center transition-colors",
                        isActive
                          ? "sidebar-link-icon-active"
                          : "sidebar-link-icon-idle group-hover:sidebar-link-icon-hover",
                      )}
                    >
                      <item.icon className="w-4 h-4" />
                    </div>
                    <div
                      className={cn(
                        "flex-1 min-w-0",
                        isSidebarCollapsed && "hidden",
                      )}
                    >
                      <div
                        className={cn(
                          "text-sm truncate leading-tight",
                          isActive
                            ? "sidebar-link-title-active"
                            : "sidebar-link-title-idle group-hover:sidebar-link-title-hover",
                        )}
                      >
                        {item.label}
                      </div>
                    </div>
                  </>
                )}
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Footer */}
        <div
          className={cn(
            "border-t border-border/40 bg-background/50",
            isSidebarCollapsed ? "p-3" : "p-4",
          )}
        >
          <div
            className={cn(
              "flex items-center",
              isSidebarCollapsed
                ? "flex-col items-center gap-3"
                : "justify-between",
            )}
          >
            <div
              className={cn(
                "flex flex-col",
                isSidebarCollapsed ? "items-center gap-1.5" : "min-w-0 gap-1",
              )}
            >
              <div
                className={cn(
                  "text-[11px] font-medium",
                  readyModelsCount > 0
                    ? "text-foreground"
                    : "text-muted-foreground",
                  isSidebarCollapsed && "text-center",
                )}
                title={loadedText}
              >
                {isSidebarCollapsed ? (
                  <span
                    className={cn(
                      "inline-flex w-2 h-2 rounded-full",
                      readyModelsCount > 0
                        ? "bg-green-500"
                        : "bg-muted-foreground/30",
                    )}
                  />
                ) : readyModelsCount > 0 ? (
                  <span className="flex items-center gap-2">
                    <span className="relative flex h-1.5 w-1.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-green-500"></span>
                    </span>
                    {loadedText}
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/30" />
                    {loadedText}
                  </span>
                )}
              </div>
            </div>
            <div
              className={cn(
                "flex items-center",
                isSidebarCollapsed ? "flex-col gap-2" : "gap-3",
              )}
            >
              <div
                className={cn(
                  "text-[10px] font-medium text-muted-foreground tracking-wider",
                  isSidebarCollapsed && "text-center",
                )}
                title={`App version ${APP_VERSION}`}
              >
                {APP_VERSION}
              </div>
              <a
                href="https://github.com/agentem-ai/izwi"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-md p-1 text-muted-foreground hover:text-foreground transition-colors hover:bg-accent"
                title="Izwi on GitHub"
              >
                <Github className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div
        className={cn(
          "flex-1 flex flex-col pt-16 lg:pt-0 transition-all duration-300 min-w-0 overflow-hidden",
          isSidebarCollapsed ? "lg:ml-[88px]" : "lg:ml-[18rem]",
        )}
      >
        <div className="hidden lg:flex justify-end px-6 lg:px-8 pt-4 shrink-0">
          <div className="flex flex-col items-end">
            <Button
              variant="outline"
              size="sm"
              onClick={switchTheme}
              className="gap-2 rounded-full px-4 bg-card/65 backdrop-blur-sm"
              title={
                resolvedTheme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
            >
              {resolvedTheme === "dark" ? (
                <>
                  <Sun className="w-4 h-4" />
                  <span className="text-xs font-medium">Light</span>
                </>
              ) : (
                <>
                  <Moon className="w-4 h-4" />
                  <span className="text-xs font-medium">Dark</span>
                </>
              )}
            </Button>
            {themePreference === "system" && (
              <div className="mt-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider pr-2">
                System
              </div>
            )}
          </div>
        </div>

        {/* Error toast */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              className="fixed top-4 left-1/2 -translate-x-1/2 z-50"
            >
              <div className="flex items-center gap-3 px-4 py-3 rounded-xl border border-destructive/45 bg-destructive/92 text-destructive-foreground shadow-[0_16px_40px_-24px_rgba(190,24,93,0.75)] font-medium text-sm">
                <AlertCircle className="w-4 h-4" />
                <span>{error}</span>
                <button
                  onClick={onErrorDismiss}
                  className="p-1 rounded-md hover:bg-white/20 transition-colors ml-2"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto w-full p-6 sm:p-8 lg:px-12 lg:pb-12 lg:pt-8 flex flex-col">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
