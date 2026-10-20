import { useEffect, useRef, useState, type CSSProperties } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { CircleAlert, Github, Menu, PanelLeft, Search } from "lucide-react";
import { APP_ICON_URL } from "@/shared/config/runtime";
import { useAppUpdates } from "@/app/providers/AppUpdateProvider";
import { RouteErrorBoundary } from "@/app/router/RouteErrorBoundary";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { FirstRunGate } from "@/app/onboarding/FirstRunGate";
import {
  NAV_GROUPS,
  SETTINGS_NAV_ITEM,
  type NavItem,
} from "@/app/layouts/navigation";
import { RuntimeCard } from "@/app/layouts/RuntimeCard";
import { UpdateCard } from "@/app/layouts/UpdateCard";
import { CommandPalette } from "@/app/command/CommandPalette";
import { isMacPlatform } from "@/lib/platform";

const appIconUrl = APP_ICON_URL;
const DESKTOP_NAVIGATION_QUERY = "(min-width: 1024px)";
const SIDEBAR_WIDTH = "15rem";
const SIDEBAR_COLLAPSED_WIDTH = "72px";
const SIDEBAR_COLLAPSED_KEY = "izwi.sidebar.collapsed";

function readSidebarPreference(): boolean {
  try {
    return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

function isChatPath(pathname: string) {
  return pathname === "/chat" || pathname.startsWith("/chat/");
}

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
  onNavigate: () => void;
}) {
  return (
    <NavLink
      to={item.path}
      title={collapsed ? item.label : undefined}
      onClick={onNavigate}
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

/** The app, or first run in its place while that's due. */
export function AppLayout(props: LayoutProps) {
  return (
    <FirstRunGate>
      <AppShell {...props} />
    </FirstRunGate>
  );
}

function AppShell({
  readyModelsCount,
  catalogError,
  onRetryModelCatalog,
}: LayoutProps) {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isDesktopNavigation, setIsDesktopNavigation] = useState(
    isDesktopNavigationLayout,
  );
  const [isRetryingModelCatalog, setIsRetryingModelCatalog] = useState(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const mobileMenuTriggerRef = useRef<HTMLButtonElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const [prefersCollapsedSidebar, setPrefersCollapsedSidebar] = useState(
    readSidebarPreference,
  );
  // Chat collapses the sidebar for room. Toggling it there lasts until you
  // leave Chat and never changes the saved preference.
  const onChat = isChatPath(location.pathname);
  const [isChatSidebarCollapsed, setIsChatSidebarCollapsed] = useState(true);

  useEffect(() => {
    if (!onChat) {
      setIsChatSidebarCollapsed(true);
    }
  }, [onChat]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.key.toLowerCase() === "k" &&
        (event.metaKey || event.ctrlKey) &&
        !event.altKey &&
        !event.shiftKey
      ) {
        event.preventDefault();
        setIsCommandPaletteOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

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
  // The compact drawer always shows labels.
  const isSidebarCollapsed =
    isDesktopNavigation &&
    (onChat ? isChatSidebarCollapsed : prefersCollapsedSidebar);

  const toggleSidebar = () => {
    if (onChat) {
      setIsChatSidebarCollapsed(!isSidebarCollapsed);
      return;
    }
    const next = !prefersCollapsedSidebar;
    setPrefersCollapsedSidebar(next);
    try {
      window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, next ? "1" : "0");
    } catch {
      // Unavailable storage only loses the preference across launches.
    }
  };

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

  const { availableUpdate } = useAppUpdates();

  const closeMobileMenu = () => {
    setMobileMenuOpen(false);
    mobileMenuTriggerRef.current?.focus();
  };

  const handleNavClick = () => {
    if (isDesktopNavigation) {
      setMobileMenuOpen(false);
    } else {
      closeMobileMenu();
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
      <CommandPalette
        open={isCommandPaletteOpen}
        onOpenChange={setIsCommandPaletteOpen}
      />
      {/* Compact header: the sidebar's brand row, for narrow windows */}
      <header className="fixed inset-x-0 top-0 z-40 flex h-14 items-center gap-2.5 border-b border-[var(--border-subtle)] bg-[var(--surface-canvas)] pl-5 pr-3 lg:hidden">
        <img
          src={appIconUrl}
          alt=""
          className="h-[26px] w-[26px] shrink-0 rounded-[7px]"
        />
        <span className="text-[15px] font-semibold tracking-tight text-[var(--text-primary)]">
          Izwi
        </span>
        <Button
          ref={mobileMenuTriggerRef}
          variant="ghost"
          size="icon"
          onClick={() =>
            mobileMenuOpen ? closeMobileMenu() : setMobileMenuOpen(true)
          }
          aria-label={mobileMenuOpen ? "Close navigation" : "Open navigation"}
          aria-expanded={mobileMenuOpen}
          aria-controls="app-sidebar-navigation"
          data-testid="mobile-navigation-trigger"
          className="ml-auto h-8 w-8 rounded-md text-[var(--text-subtle)] hover:text-foreground"
        >
          <Menu className="h-4 w-4" />
        </Button>
      </header>

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
        {/* Brand row; the open drawer covers the compact header, so it shows there too */}
        <div
          className={cn(
            "flex shrink-0 pt-3.5",
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
            onClick={toggleSidebar}
            aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-expanded={!isSidebarCollapsed}
            aria-controls="app-sidebar-navigation"
            data-testid="sidebar-collapse-toggle"
            tabIndex={isDesktopNavigation ? undefined : -1}
            className={cn(
              "hidden h-8 w-8 rounded-md text-[var(--text-subtle)] hover:text-foreground lg:inline-flex",
              !isSidebarCollapsed && "ml-auto",
            )}
            title={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <PanelLeft className="h-4 w-4" />
          </Button>
        </div>

        <div className={cn("shrink-0 px-3 pt-3", isSidebarCollapsed && "flex justify-center")}>
          <button
            type="button"
            onClick={() => setIsCommandPaletteOpen(true)}
            tabIndex={sidebarInteractive ? undefined : -1}
            aria-label="Search or jump to"
            aria-keyshortcuts={isMacPlatform() ? "Meta+K" : "Control+K"}
            title={isSidebarCollapsed ? "Search or jump to" : undefined}
            className={cn(
              "flex items-center gap-2 rounded-[10px] border border-[var(--border-subtle)] bg-[var(--surface-panel)] text-[13px] text-[var(--text-subtle)] transition-colors hover:text-[var(--text-primary)]",
              isSidebarCollapsed ? "h-10 w-10 justify-center" : "h-[34px] w-full px-2.5",
            )}
          >
            <Search className="h-4 w-4 shrink-0" />
            {!isSidebarCollapsed ? (
              <>
                <span className="truncate">Search or jump to…</span>
                <kbd className="ml-auto rounded-[5px] border border-[var(--border-default)] px-1.5 font-sans text-[11px] leading-[18px] text-[var(--text-subtle)]">
                  {isMacPlatform() ? "⌘K" : "Ctrl K"}
                </kbd>
              </>
            ) : null}
          </button>
        </div>

        {/* Navigation */}
        <nav
          id="app-sidebar-navigation"
          aria-label="Primary"
          className="flex flex-1 flex-col overflow-y-auto px-3 pb-3 pt-1 scrollbar-thin"
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
          "flex-1 flex flex-col pt-14 lg:pt-0 transition-all duration-300 min-w-0 overflow-hidden",
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
            <div
              role="alert"
              className="mb-5 flex shrink-0 items-start gap-3 rounded-[12px] border border-[var(--border-subtle)] bg-[var(--surface-panel)] px-3.5 py-3"
            >
              <CircleAlert
                aria-hidden="true"
                className="mt-0.5 h-4 w-4 shrink-0 text-[var(--danger-text)]"
              />
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                <p className="text-[13.5px] font-medium leading-5 text-[var(--text-primary)]">
                  Couldn’t load models
                </p>
                <p className="break-words text-[12.5px] leading-[1.45] text-[var(--text-muted)]">
                  {catalogError}
                </p>
              </div>
              <button
                type="button"
                disabled={isRetryingModelCatalog}
                onClick={() => void handleRetryModelCatalog()}
                className="inline-flex h-8 shrink-0 items-center rounded-full border border-[var(--border-default)] px-3.5 text-[13px] text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)] disabled:opacity-50"
              >
                {isRetryingModelCatalog ? "Retrying…" : "Retry"}
              </button>
            </div>
          ) : null}
          <RouteErrorBoundary resetKey={location.pathname}>
            <Outlet />
          </RouteErrorBoundary>
        </main>
      </div>
      <UpdateCard />
    </div>
  );
}
