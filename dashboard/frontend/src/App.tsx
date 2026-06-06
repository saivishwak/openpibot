import {
  Activity,
  Bot,
  Camera,
  Cpu,
  Database,
  FileText,
  Gauge,
  GraduationCap,
  Headphones,
  Menu,
  Monitor,
  Moon,
  SlidersHorizontal,
  Stethoscope,
  Sun,
  X,
} from "lucide-react";
import { NavLink, Route, Routes } from "react-router-dom";
import useSWR from "swr";
import { VRStatus, fetcher } from "./api";
import { Badge, Button, SelectField, cn } from "./components/ui";
import { Cameras } from "./pages/Cameras";
import { Calibration } from "./pages/Calibration";
import { Dashboard } from "./pages/Dashboard";
import { Diagnostics } from "./pages/Diagnostics";
import { Inference } from "./pages/Inference";
import { Logs } from "./pages/Logs";
import { Recording } from "./pages/Recording";
import { Training } from "./pages/Training";
import { VRTeleop } from "./pages/VRTeleop";
import { useEffect, useState } from "react";

const nav = [
  { to: "/", label: "Dashboard", icon: Gauge },
  { to: "/control", label: "Control", icon: Headphones },
  { to: "/calibration", label: "Calibration", icon: SlidersHorizontal },
  { to: "/recording", label: "Recording", icon: Database },
  { to: "/training", label: "Training", icon: GraduationCap },
  { to: "/inference", label: "Inference", icon: Cpu },
  { to: "/cameras", label: "Cameras", icon: Camera },
  { to: "/diagnostics", label: "Diagnostics", icon: Stethoscope },
  { to: "/logs", label: "Logs", icon: FileText },
] as const;

type ThemeMode = "system" | "light" | "dark";

const THEME_KEY = "openpibot.theme";

function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") return "system";
  let saved: string | null = null;
  try {
    saved = window.localStorage.getItem(THEME_KEY);
  } catch {
    saved = null;
  }
  return saved === "light" || saved === "dark" || saved === "system" ? saved : "system";
}

function applyTheme(mode: ThemeMode) {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.classList.toggle("dark", mode === "dark" || (mode === "system" && prefersDark));
}

function GlobalStatus() {
  const { data } = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1500 });
  const connected = data?.connected_sides ?? [];
  const anyConnected = connected.length > 0;
  const active = data?.dual_mode ? "dual" : data?.active_arm ?? "none";
  return (
    <div className="hidden items-center gap-2 md:flex">
      <Badge tone={anyConnected ? "success" : "neutral"}>{anyConnected ? connected.join(" + ") : "no robot"}</Badge>
      <Badge tone={data?.engaged ? "danger" : anyConnected ? "warning" : "neutral"}>
        {data?.engaged ? "engaged" : anyConnected ? "armed" : "idle"}
      </Badge>
      <Badge tone={data?.recording ? "info" : "neutral"}>{data?.recording ? "recording" : `active: ${active}`}</Badge>
    </div>
  );
}

function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <aside
      className={cn(
        "fixed inset-y-0 left-0 z-30 w-64 border-r border-border bg-card transition-transform lg:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full",
      )}
    >
      <div className="flex h-14 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-2">
          <Bot size={22} />
          <div>
            <div className="text-sm font-semibold">OpenPiBot</div>
            <div className="text-xs text-muted-foreground">robot console</div>
          </div>
        </div>
        <button className="rounded-md p-1 hover:bg-muted lg:hidden" onClick={onClose} aria-label="Close navigation">
          <X size={18} />
        </button>
      </div>
      <nav className="space-y-1 p-3">
        {nav.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            onClick={onClose}
            className={({ isActive }) =>
              cn(
                "flex h-9 items-center gap-2 rounded-md px-3 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground",
                isActive && "bg-muted text-foreground",
              )
            }
          >
            <item.icon size={17} />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}

function ThemeSelector() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);

  useEffect(() => {
    applyTheme(theme);
    try {
      window.localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Theme still applies for this session when persistence is unavailable.
    }
    if (theme !== "system") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => applyTheme("system");
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [theme]);

  const iconClass = "hidden text-muted-foreground sm:block";
  const Icon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;

  return (
    <div className="flex items-center gap-2">
      <Icon className={iconClass} size={16} />
      <div className="w-32">
        <SelectField
          value={theme}
          onValueChange={(value) => setTheme(value as ThemeMode)}
          ariaLabel="Theme"
          options={[
            { value: "system", label: "System" },
            { value: "light", label: "Light" },
            { value: "dark", label: "Dark" },
          ]}
        />
      </div>
    </div>
  );
}

export function App() {
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-full bg-background text-foreground">
      <Sidebar open={open} onClose={() => setOpen(false)} />
      {open ? <div className="fixed inset-0 z-20 bg-black/40 lg:hidden" onClick={() => setOpen(false)} /> : null}
      <div className="lg:pl-64">
        <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-border bg-background/95 px-4 backdrop-blur sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <Button variant="ghost" className="px-2 lg:hidden" onClick={() => setOpen(true)}>
              <Menu size={18} />
            </Button>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Activity size={17} />
              <span>OpenPiBot dashboard</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <GlobalStatus />
            <ThemeSelector />
          </div>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/control" element={<VRTeleop />} />
            <Route path="/calibration" element={<Calibration />} />
            <Route path="/recording" element={<Recording />} />
            <Route path="/training" element={<Training />} />
            <Route path="/inference" element={<Inference />} />
            <Route path="/cameras" element={<Cameras />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
            <Route path="/logs" element={<Logs />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
