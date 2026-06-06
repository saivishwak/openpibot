import { Activity, Bot, Camera, Database, Headphones, Stethoscope } from "lucide-react";
import useSWR from "swr";
import { RobotsResponse, VRStatus, fetcher } from "../api";
import { Badge, Card, Page, SelectField } from "../components/ui";

function Stat({ label, value, icon: Icon, tone = "neutral" }: {
  label: string;
  value: string;
  icon: typeof Activity;
  tone?: "neutral" | "success" | "warning" | "danger" | "info";
}) {
  return (
    <Card>
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase text-muted-foreground">{label}</p>
          <p className="mt-1 text-2xl font-semibold">{value}</p>
        </div>
        <div className="rounded-md border border-border bg-muted p-2">
          <Icon size={20} />
        </div>
      </div>
      <div className="mt-3"><Badge tone={tone}>{label}</Badge></div>
    </Card>
  );
}

export function Dashboard() {
  const { data: vr } = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1000 });
  const { data: robots } = useSWR<RobotsResponse>("/api/config/robots", fetcher);
  const { data: doctor } = useSWR<{ checks: Array<{ status: string }> }>("/api/doctor", fetcher);
  const activeRobot = robots?.active_robot ?? "xlerobot";
  const failures = doctor?.checks.filter((c) => c.status === "fail").length ?? 0;
  const warnings = doctor?.checks.filter((c) => c.status === "warn").length ?? 0;

  return (
    <Page
      title="OpenPiBot Dashboard"
      description="Bimanual SO-101 control, VR teleoperation, dataset recording, PI0.5 training, and inference in one local console."
      action={
        <div className="w-56">
          <SelectField
            value={activeRobot}
            onValueChange={() => alert("Robot switching is read-only until multi-profile YAML writes are enabled.")}
            options={(robots?.robots ?? [{ id: activeRobot, name: "XLeRobot", active: true }]).map((r) => ({
              value: r.id,
              label: r.name ? `${r.name} (${r.id})` : r.id,
            }))}
          />
        </div>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Robot" value={(vr?.connected_sides ?? []).join(" + ") || "offline"} icon={Bot} tone={(vr?.connected_sides?.length ?? 0) ? "success" : "neutral"} />
        <Stat label="VR" value={vr?.engaged ? "engaged" : vr?.active_arm ?? "idle"} icon={Headphones} tone={vr?.engaged ? "danger" : "neutral"} />
        <Stat label="Recording" value={vr?.recording ? `${vr.recording_info.frames_in_current_episode} frames` : `${vr?.recording_info.episodes_saved ?? 0} episodes`} icon={Database} tone={vr?.recording ? "info" : "neutral"} />
        <Stat label="Diagnostics" value={failures ? `${failures} fail` : warnings ? `${warnings} warn` : "ok"} icon={Stethoscope} tone={failures ? "danger" : warnings ? "warning" : "success"} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <h2 className="text-sm font-semibold">Robot Readiness</h2>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {(["left", "right"] as const).map((side) => {
              const arm = vr?.arms?.[side];
              return (
                <div key={side} className="rounded-md border border-border p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium capitalize">{side} arm</span>
                    <Badge tone={arm?.connected ? "success" : "neutral"}>{arm?.connected ? "connected" : "offline"}</Badge>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Badge tone={arm?.torque_enabled ? "success" : "warning"}>{arm?.torque_enabled ? "torque on" : "torque off"}</Badge>
                    <Badge tone={arm?.calibrated ? "success" : "neutral"}>{arm?.calibrated ? "anchored" : "not anchored"}</Badge>
                    <Badge tone={arm?.home?.captured ? "success" : "neutral"}>{arm?.home?.captured ? "home saved" : "no home"}</Badge>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <Card>
          <h2 className="text-sm font-semibold">Camera Roles</h2>
          <div className="mt-3 grid gap-2">
            {["head", "left_wrist", "right_wrist"].map((role) => (
              <div key={role} className="flex items-center justify-between rounded-md border border-border p-3">
                <span className="text-sm">{role}</span>
                <Camera size={16} className="text-muted-foreground" />
              </div>
            ))}
          </div>
        </Card>
      </div>
    </Page>
  );
}

