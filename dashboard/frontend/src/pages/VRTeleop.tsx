import { lazy, Suspense, useEffect, useState } from "react";
import { Copy, Hand, Home, Lock, Plug, PlugZap, Power, Unlock } from "lucide-react";
import useSWR from "swr";
import { ArmSide, VRStatus, api, fetcher } from "../api";
import { CameraPreviewGrid } from "../components/CameraPreviewGrid";
import { Badge, Button, Card, ConfirmDialog, Page, Range, Toggle } from "../components/ui";

const RobotUrdfViewer = lazy(() => import("../components/RobotUrdfViewer"));

function safeCall<T>(work: () => Promise<T>, refresh: () => Promise<unknown>) {
  return work().then(refresh).catch((err) => alert(String(err)));
}

function ArmControl({ side, status, refresh, busy, setBusy }: {
  side: ArmSide;
  status?: VRStatus;
  refresh: () => Promise<unknown>;
  busy: boolean;
  setBusy: (busy: boolean) => void;
}) {
  const arm = status?.arms?.[side];
  const connected = !!arm?.connected;
  const active = status?.active_arm === side || !!status?.dual_mode;
  const quality = arm?.calibration.quality;
  const speedCm = quality ? quality.offset_speed_ema_mps * 100 : 0;
  const ikRejectPct = quality ? quality.ik_reject_fraction * 100 : 0;
  const run = async (fn: () => Promise<VRStatus>) => {
    setBusy(true);
    try { await safeCall(fn, refresh); } finally { setBusy(false); }
  };
  return (
    <Card className={active ? "border-danger" : ""}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold capitalize">{side} arm</h2>
          <div className="mt-2 flex flex-wrap gap-2">
            <Badge tone={connected ? "success" : "neutral"}>{connected ? "connected" : "offline"}</Badge>
            <Badge tone={arm?.torque_enabled ? "success" : "warning"}>{arm?.torque_enabled ? "torque on" : "torque off"}</Badge>
            <Badge tone={arm?.calibrated ? "success" : "neutral"}>{arm?.calibrated ? "anchored" : "grip to anchor"}</Badge>
            <Badge tone={arm?.calibration.vr_ctrl_to_ee_ready ? "success" : "neutral"}>
              {arm?.calibration.vr_ctrl_to_ee_ready ? "wrist aligned" : "wrist unaligned"}
            </Badge>
          </div>
        </div>
        {!connected ? (
          <Button disabled={busy} onClick={() => run(() => api.vrConnect(side))}><Plug size={16} />Connect</Button>
        ) : (
          <Button variant="secondary" disabled={busy} onClick={() => run(() => api.vrDisconnect(side))}><PlugZap size={16} />Disconnect</Button>
        )}
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button variant="secondary" disabled={!connected || busy} onClick={() => run(() => api.vrTorqueRelease(side))}><Unlock size={16} />Release</Button>
        <Button variant="secondary" disabled={!connected || busy} onClick={() => run(() => api.vrTorqueLock(side))}><Lock size={16} />Lock</Button>
        <Button variant="secondary" disabled={!connected || busy} onClick={() => run(() => api.vrHomeCapture(side))}><Hand size={16} />Capture home</Button>
        <Button variant="secondary" disabled={!connected || busy || !arm?.home?.captured || !arm?.torque_enabled} onClick={() => run(() => api.vrHomeGo(side))}><Home size={16} />Go home</Button>
      </div>
      <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
        <div className="rounded-md border border-border bg-muted/40 p-2">
          <div className="text-muted-foreground">EE speed EMA</div>
          <div className="mt-1 text-sm font-medium">{quality?.samples ? `${speedCm.toFixed(1)} cm/s` : "—"}</div>
        </div>
        <div className="rounded-md border border-border bg-muted/40 p-2">
          <div className="text-muted-foreground">IK rejects</div>
          <div className="mt-1 text-sm font-medium">{quality?.samples ? `${ikRejectPct.toFixed(1)}%` : "—"}</div>
        </div>
      </div>
    </Card>
  );
}

function Engagement({ status, refresh }: { status?: VRStatus; refresh: () => Promise<unknown> }) {
  const [scale, setScale] = useState(0.5);
  useEffect(() => { if (status?.scale) setScale(status.scale); }, [status?.scale]);
  const connected = status?.connected_sides ?? [];
  const activeArm = status?.active_arm ?? connected[0] ?? "right";
  const engaged = !!status?.engaged;
  return (
    <Card className={engaged ? "border-danger bg-danger/5" : ""}>
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-sm font-semibold">Engagement</h2>
          <p className="mt-1 text-xs text-muted-foreground">Motors only move while engaged and anchored from the Quest grip reset.</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">{engaged ? "ENGAGED" : "Disarmed"}</span>
          <Toggle
            checked={engaged}
            disabled={!connected.length}
            onCheckedChange={(next) => safeCall(() => api.vrEngage(next, scale, activeArm), refresh)}
          />
        </div>
      </div>
      <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_auto] lg:items-center">
        <div>
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-medium">Speed scale</span>
            <span className="mono">{scale.toFixed(2)}</span>
          </div>
          <Range value={scale} min={0.1} max={1} step={0.05} onValueChange={setScale} disabled={!connected.length} />
        </div>
        <Button variant="secondary" disabled={!connected.length} onClick={() => safeCall(() => api.vrEngage(engaged, scale, activeArm), refresh)}>
          Apply scale
        </Button>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <Badge tone={status?.dual_mode ? "danger" : "neutral"}>{status?.dual_mode ? "dual mode" : `active ${activeArm}`}</Badge>
        <Badge tone={status?.last_tick_age_ms && status.last_tick_age_ms < 500 ? "success" : "neutral"}>tick {status?.last_tick_age_ms ?? "—"} ms</Badge>
      </div>
    </Card>
  );
}

export function VRTeleop() {
  const { data, mutate, isLoading } = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1000 });
  const [busy, setBusy] = useState(false);
  const [confirmStop, setConfirmStop] = useState(false);
  const refresh = () => mutate();
  const endpoint = data?.vr_endpoint;

  return (
    <Page title="Control" description="Quest VR teleoperation with explicit connect, engage, grip anchoring, watchdog, and emergency release.">
      <div className="grid gap-4 xl:grid-cols-2">
        <ArmControl side="left" status={data} refresh={refresh} busy={busy || isLoading} setBusy={setBusy} />
        <ArmControl side="right" status={data} refresh={refresh} busy={busy || isLoading} setBusy={setBusy} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.8fr]">
        <Engagement status={data} refresh={refresh} />
        <Card>
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold">Quest Endpoint</h2>
              <p className="mt-1 text-xs text-muted-foreground">Open this URL in the Quest browser after connecting an arm.</p>
            </div>
            <Button variant="danger" onClick={() => setConfirmStop(true)}><Power size={16} />Emergency</Button>
          </div>
          <div className="mt-3 flex items-center gap-2 rounded-md border border-border bg-muted p-3">
            <code className="mono flex-1 overflow-hidden text-ellipsis text-sm">{endpoint ?? "connect an arm to start VR servers"}</code>
            <Button variant="secondary" disabled={!endpoint} onClick={() => endpoint && navigator.clipboard.writeText(endpoint)}>
              <Copy size={16} />Copy
            </Button>
          </div>
          {data?.last_error ? <p className="mt-3 text-sm text-danger">{data.last_error}</p> : null}
        </Card>
      </div>

      <Suspense fallback={<Card>Loading 3D view...</Card>}>
        <RobotUrdfViewer status={data} />
      </Suspense>

      <Card>
        <h2 className="mb-3 text-sm font-semibold">Camera Preview</h2>
        <CameraPreviewGrid />
      </Card>

      <ConfirmDialog
        open={confirmStop}
        onOpenChange={setConfirmStop}
        title="Emergency stop?"
        description="Torque will be released on both arms immediately. The robot will not move to a safe pose."
        confirmLabel="Stop now"
        onConfirm={() => {
          setConfirmStop(false);
          safeCall(api.vrEmergencyStop, refresh);
        }}
      />
    </Page>
  );
}
