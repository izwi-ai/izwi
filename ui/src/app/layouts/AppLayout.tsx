import { useEffect, useRef, useState, type CSSProperties } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  Download,
  Github,
  Menu,
  PanelLeft,
  RotateCcw,
  TriangleAlert,
  X,
} from "lucide-react";
import { APP_ICON_URL } from "@/shared/config/runtime";
import { useAppUpdates } from "@/app/providers/AppUpdateProvider";
import { RouteErrorBoundary } from "@/app/router/RouteErrorBoundary";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { StatePanel } from "@/components/ui/state-panel";
import { FirstRunOnboarding } from "@/app/onboarding/FirstRunOnboarding";
import {
  NAV_GROUPS,
  SETTINGS_NAV_ITEM,
  type NavItem,
} from "@/app/layouts/navigation";
import { RuntimeCard } from "@/app/layouts/RuntimeCard";

const appIconUrl = APP_ICON_URL;
const DESKTOP_NAVIGATION_QUERY = "(min-width: 1024px)";
const SIDEBAR_WIDTH = "15rem";
const SIDEBAR_COLLAPSED_WIDTH = "72px";

function isDesktopNavigationLayout() {
  if (typeof window === "undefined") {
    return true;
  }
  if (typeof window.matchMedia === "function") {
    return window.matchMedia(DESKTOP_NAVIGATION_QUERY).matches;
  }
  return window.innerWidth >= 1024;
}

interface LayoutProps {
  readyModelsCount: number;
  selectedModelLabel: string | null;
  catalogError: string | null;
  onRetryModelCatalog: () => Promise<boolean>;
}

function SidebarLink({
  item,
  collapsed,
  interactive,
  badge,
  onNavigate,
}: {
  item: NavItem;
  collapsed: boolean;
  interactive: boolean;
  badge?: { label: string } | null;
  onNavigate: (path: string) => void;
}) {
  return (
    <NavLink
      to={item.path}
      title={collapsed ? item.label : undefined}
      onClick={() => onNavigate(item.path)}
      data-app-nav-link
      tabIndex={interactive ? undefined : -1}
      className={({ isActive }) =>
        cn(
          "sidebar-link relative flex items-center rounded-lg border text-[13.5px] transition-colors group",
          collapsed ? "mx-auto h-10 w-10 justify-center" : "h-9 gap-2.5 px-2.5",
          isActive
            ? "sidebar-link-active font-medium"
            : "sidebar-link-idle",
        )
      }
    >
      {({ isActive }) => (
        <>
          <item.icon
            className={cn(
              "h-4 w-4 shrink-0 transition-colors",
              isActive
                ? "sidebar-link-icon-active"
                : "sidebar-link-icon-idle group-hover:sidebar-link-icon-hover",
            )}
          />
          <span
            className={cn(
              "min-w-0 flex-1 truncate",
              isActive
                ? "sidebar-link-title-active"
                : "sidebar-link-title-idle group-hover:sidebar-link-title-hover",
              collapsed && "sr-only",
            )}
          >
            {item.label}
          </span>
          {badge ? (
            <span
              title={badge.label}
              aria-label={badge.label}
              className={cn(
                "h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--status-warning-text)]",
                collapsed && "absolute right-1.5 top-1.5",
              )}
            />
          ) : null}
        </>
      )}
    </NavLink>
  );
}

export function AppLayout({
  readyModelsCount,
  selectedModelLabel,
  catalogError,
  onRetryModelCatalog,
}: LayoutProps) {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isDesktopNavigation, setIsDesktopNavigation] = useState(
    isDesktopNavigationLayout,
  );
  const [isRetryingModelCatalog, setIsRetryingModelCatalog] = useState(false);
  const mobileMenuTriggerRef = useRef<HTMLButtonElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
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

  useEffect(() => {
    if (typeof window.matchMedia !== "function") {
      return;
    }

    const mediaQuery = window.matchMedia(DESKTOP_NAVIGATION_QUERY);
    const handleLayoutChange = (event: MediaQueryListEvent) => {
      setIsDesktopNavigation(event.matches);
      if (event.matches) {
        setMobileMenuOpen(false);
      }
    };

    setIsDesktopNavigation(mediaQuery.matches);
    mediaQuery.addEventListener("change", handleLayoutChange);
    return () => mediaQuery.removeEventListener("change", handleLayoutChange);
  }, []);

  const sidebarInteractive = isDesktopNavigation || mobileMenuOpen;

  useEffect(() => {
    const sidebar = sidebarRef.current;
    if (!sidebar) {
      return;
    }

    if (sidebarInteractive) {
      sidebar.removeAttribute("inert");
    } else {
      sidebar.setAttribute("inert", "");
    }
  }, [sidebarInteractive]);

  useEffect(() => {
    if (!mobileMenuOpen || isDesktopNavigation) {
      return;
    }

    sidebarRef.current
      ?.querySelector<HTMLElement>("[data-app-nav-link]")
      ?.focus();
  }, [isDesktopNavigation, mobileMenuOpen]);

  const loadedText =
    readyModelsCount > 0
      ? `${readyModelsCount} model${readyModelsCount !== 1 ? "s" : ""} loaded`
      : "No models loaded";

  const {
    availableUpdate,
    status: updateStatus,
    isPromptOpen,
    progressPercent,
    errorMessage: updateErrorMessage,
    dismissPrompt,
    installUpdate,
    restartToApply,
  } = useAppUpdates();

  const closeMobileMenu = () => {
    setMobileMenuOpen(false);
    mobileMenuTriggerRef.current?.focus();
  };

  const handleNavClick = (path: string) => {
    if (isDesktopNavigation) {
      setMobileMenuOpen(false);
    } else {
      closeMobileMenu();
    }
    if (
      path === "/chat" &&
      typeof window !== "undefined" &&
      window.innerWidth >= 1024
    ) {
      setIsSidebarCollapsed(true);
    }
  };

  const handleSidebarKeyDown = (event: React.KeyboardEvent<HTMLElement>) => {
    if (isDesktopNavigation || !mobileMenuOpen) {
      return;
    }

    if (event.key === "Escape") {
      event.preventDefault();
      closeMobileMenu();
      return;
    }

    if (event.key !== "Tab") {
      return;
    }

    const focusableElements = Array.from(
      sidebarRef.current?.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ) ?? [],
    ).filter((element) => element.tabIndex >= 0);
    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];

    if (!firstFocusable || !lastFocusable) {
      return;
    }
    if (event.shiftKey && document.activeElement === firstFocusable) {
      event.preventDefault();
      lastFocusable.focus();
    } else if (!event.shiftKey && document.activeElement === lastFocusable) {
      event.preventDefault();
      firstFocusable.focus();
    }
  };

  const handleRetryModelCatalog = async () => {
    if (isRetryingModelCatalog) {
      return;
    }
    setIsRetryingModelCatalog(true);
    try {
      await onRetryModelCatalog();
    } finally {
      setIsRetryingModelCatalog(false);
    }
  };

  return (
    <div className="h-dvh flex overflow-hidden bg-transparent text-foreground selection:bg-primary/25 selection:text-foreground">
      <FirstRunOnboarding />
      {/* Mobile header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 border-b border-border/80 bg-background/80 backdrop-blur-xl">
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
              <div className="mt-0.5 text-[11px] text-muted-foreground">
                {selectedModelLabel ?? loadedText}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              ref={mobileMenuTriggerRef}
              variant="ghost"
              size="icon"
              onClick={() =>
                mobileMenuOpen ? closeMobileMenu() : setMobileMenuOpen(true)
              }
              aria-label={
                mobileMenuOpen ? "Close navigation" : "Open navigation"
              }
              aria-expanded={mobileMenuOpen}
              aria-controls="app-sidebar-navigation"
              data-testid="mobile-navigation-trigger"
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
            onClick={closeMobileMenu}
            aria-hidden="true"
            className="lg:hidden fixed inset-0 bg-background/70 backdrop-blur-sm z-40"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside
        ref={sidebarRef}
        aria-hidden={sidebarInteractive ? undefined : true}
        aria-label={!isDesktopNavigation && mobileMenuOpen ? "Main navigation" : undefined}
        aria-modal={!isDesktopNavigation && mobileMenuOpen ? true : undefined}
        role={!isDesktopNavigation && mobileMenuOpen ? "dialog" : undefined}
        onKeyDown={handleSidebarKeyDown}
        data-testid="app-sidebar"
        className={cn(
          "w-[15rem] border-r border-[var(--border-subtle)] flex flex-col fixed h-full z-50 bg-[var(--surface-canvas)] transition-all duration-300",
          "lg:translate-x-0",
          isSidebarCollapsed ? "lg:w-[72px]" : "lg:w-[15rem]",
          mobileMenuOpen ? "translate-x-0 shadow-2xl" : "-translate-x-full",
        )}
      >
        {/* Logo - hidden on mobile since it's in the header */}
        <div
          className={cn(
            "hidden lg:flex shrink-0 pt-3.5",
            isSidebarCollapsed
              ? "flex-col items-center gap-2 px-3"
              : "items-center gap-2.5 px-5 pb-1",
          )}
        >
          <img
            src={appIconUrl}
            alt="Izwi logo"
            className="h-[26px] w-[26px] shrink-0 rounded-[7px]"
          />
          <span
            className={cn(
              "text-[15px] font-semibold tracking-tight text-foreground",
              isSidebarCollapsed && "sr-only",
            )}
          >
            Izwi
          </span>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsSidebarCollapsed((collapsed) => !collapsed)}
            aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-expanded={!isSidebarCollapsed}
            aria-controls="app-sidebar-navigation"
            data-testid="sidebar-collapse-toggle"
            tabIndex={isDesktopNavigation ? undefined : -1}
            className={cn(
              "h-8 w-8 rounded-md text-[var(--text-subtle)] hover:text-foreground",
              !isSidebarCollapsed && "ml-auto",
            )}
            title={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <PanelLeft className="h-4 w-4" />
          </Button>
        </div>

        {/* Navigation */}
        <nav
          id="app-sidebar-navigation"
          aria-label="Primary"
          className="flex flex-1 flex-col overflow-y-auto px-3 py-3 scrollbar-thin"
        >
          {NAV_GROUPS.map((group, index) => (
            <div
              key={group.label}
              role="group"
              aria-label={group.label}
              className={cn(
                isSidebarCollapsed &&
                  index > 0 &&
                  "mx-2 mt-2 border-t border-[var(--border-subtle)] pt-2",
              )}
            >
              <h4
                className={cn(
                  "px-2.5 pb-1 pt-3 text-[11px] font-medium tracking-[0.02em] text-[var(--text-subtle)]",
                  isSidebarCollapsed && "sr-only",
                )}
              >
                {group.label}
              </h4>
              <div className="space-y-0.5">
                {group.items.map((item) => (
                  <SidebarLink
                    key={item.id}
                    item={item}
                    collapsed={isSidebarCollapsed}
                    interactive={sidebarInteractive}
                    onNavigate={handleNavClick}
                  />
                ))}
              </div>
            </div>
          ))}

          <div className="mt-auto space-y-0.5 pt-4">
            <SidebarLink
              item={SETTINGS_NAV_ITEM}
              collapsed={isSidebarCollapsed}
              interactive={sidebarInteractive}
              badge={availableUpdate ? { label: "Update available" } : null}
              onNavigate={handleNavClick}
            />
          </div>
        </nav>

        {/* Footer */}
        <div className="shrink-0 px-3 pb-3">
          <RuntimeCard
            readyModelsCount={readyModelsCount}
            catalogError={catalogError}
            collapsed={isSidebarCollapsed}
          />
          {!isSidebarCollapsed ? (
            <a
              href="https://github.com/izwi-ai/izwi"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Izwi on GitHub"
              tabIndex={sidebarInteractive ? undefined : -1}
              className="mt-2 inline-flex items-center gap-1.5 rounded-md px-1 py-0.5 text-[11px] text-[var(--text-subtle)] transition-colors hover:text-foreground"
            >
              <Github className="h-3.5 w-3.5" />
              GitHub
            </a>
          ) : null}
        </div>
      </aside>

      {/* Main content */}
      <div
        className={cn(
          "flex-1 flex flex-col pt-16 lg:pt-0 transition-all duration-300 min-w-0 overflow-hidden",
          isSidebarCollapsed ? "lg:ml-[72px]" : "lg:ml-[15rem]",
        )}
        style={
          {
            "--app-shell-left": isSidebarCollapsed
              ? SIDEBAR_COLLAPSED_WIDTH
              : SIDEBAR_WIDTH,
          } as CSSProperties
        }
      >
        {/* Page content */}
        <main className="flex-1 overflow-y-auto w-full p-6 sm:p-8 lg:px-12 lg:pb-12 lg:pt-8 flex flex-col">
          {catalogError ? (
            <div role="alert" className="mb-5 shrink-0">
              <StatePanel
                title="Model service unavailable"
                description={catalogError}
                icon={TriangleAlert}
                tone="danger"
                actions={
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={isRetryingModelCatalog}
                    onClick={() => void handleRetryModelCatalog()}
                  >
                    {isRetryingModelCatalog ? "Retrying…" : "Retry models"}
                  </Button>
                }
              />
            </div>
          ) : null}
          <RouteErrorBoundary resetKey={location.pathname}>
            <Outlet />
          </RouteErrorBoundary>
        </main>
      </div>
      {isPromptOpen && availableUpdate ? (
        <div className="fixed bottom-4 right-4 z-[70] w-[min(30rem,calc(100vw-1.5rem))] rounded-2xl border border-border/80 bg-background/90 p-4 shadow-2xl backdrop-blur-xl">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                In-App Update
              </p>
              <h3 className="mt-1 text-sm font-semibold text-foreground">
                {updateStatus === "downloaded"
                  ? "Update ready to apply"
                  : `Update available: ${availableUpdate.version}`}
              </h3>
            </div>
            <button
              type="button"
              className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
              onClick={dismissPrompt}
              aria-label="Dismiss update prompt"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <p className="mt-2 text-xs leading-5 text-muted-foreground">
            {availableUpdate.platformBehavior.appExitsDuringInstall
              ? "Windows installs close Izwi automatically and run the updater."
              : "Install the update now and restart when you’re ready."}
          </p>

          {availableUpdate.notes ? (
            <p className="mt-2 max-h-20 overflow-y-auto rounded-lg border border-border/70 bg-card/50 px-3 py-2 text-xs leading-5 text-muted-foreground">
              {availableUpdate.notes}
            </p>
          ) : null}

          {updateStatus === "downloading" ? (
            <div className="mt-3 space-y-1">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Downloading update…</span>
                <span>{progressPercent !== null ? `${progressPercent}%` : "..."}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-muted/50">
                <div
                  className={cn(
                    "h-full bg-primary transition-[width] duration-200",
                    progressPercent === null && "w-1/2 animate-pulse",
                  )}
                  style={
                    progressPercent !== null
                      ? { width: `${progressPercent}%` }
                      : undefined
                  }
                />
              </div>
            </div>
          ) : null}

          {updateErrorMessage ? (
            <p className="mt-3 rounded-lg border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3 py-2 text-xs text-[var(--status-warning-text)]">
              {updateErrorMessage}
            </p>
          ) : null}

          <div className="mt-3 flex flex-wrap items-center gap-2">
            {updateStatus === "downloaded" &&
            availableUpdate.platformBehavior.supportsRestartLater ? (
              <Button
                type="button"
                size="sm"
                className="gap-2"
                onClick={() => void restartToApply()}
              >
                <RotateCcw className="h-4 w-4" />
                Restart Now
              </Button>
            ) : (
              <Button
                type="button"
                size="sm"
                className="gap-2"
                disabled={updateStatus === "downloading"}
                onClick={() => void installUpdate()}
              >
                <Download className="h-4 w-4" />
                {updateStatus === "downloading" ? "Installing…" : "Install Update"}
              </Button>
            )}

            <Button type="button" size="sm" variant="ghost" onClick={dismissPrompt}>
              Later
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
