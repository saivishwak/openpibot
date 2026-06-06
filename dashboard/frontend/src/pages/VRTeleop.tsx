import { lazy, Suspense, useEffect, useState } from "react";
import { Copy, Hand, Home, Lock, Plug, PlugZap, Power, Unlock } from "lucide-react";
import useSWR from "swr";
import { ArmSide, QuestBridgeStatus, QuestVideoStatus, VRStatus, api, fetcher } from "../api";
import { CameraPreviewGrid } from "../components/CameraPreviewGrid";
import { Badge, Button, Card, ConfirmDialog, Page, Range, Toggle } from "../components/ui";

const RobotUrdfViewer = lazy(() => import("../components/RobotUrdfViewer"));

function operatorStageLabel(stage?: string) {
  switch (stage) {
    case "connect_required": return "connection";
    case "mirror_waiting_robot": return "robot check";
    case "mirror_ready": return "mirror ready";
    case "teleop_head_only": return "head camera only";
    case "teleop_arms": return "teleop active";
    case "suspended": return "suspended";
    default: return stage?.replaceAll("_", " ") ?? "loading";
  }
}

function OperatorFlow({ status }: { status?: VRStatus }) {
  const operator = status?.operator;
  const blockers = [
    ...(operator?.ready_blockers ?? []),
    ...(operator?.recording_blockers ?? []),
  ];
  const headCamera = operator?.head_camera_url;
  return (
    <Card className={operator?.stage === "suspended" ? "border-danger bg-danger/5" : ""}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-sm font-semibold">Operator flow</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {operator?.guidance ?? "Waiting for the VR backend operator state."}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge tone={operator?.stage === "teleop_arms" ? "success" : operator?.stage === "suspended" ? "danger" : "info"}>
            {operatorStageLabel(operator?.stage)}
          </Badge>
          <Badge tone={operator?.connection?.websocket_clients ? "success" : "neutral"}>
            {operator?.connection?.websocket_clients ?? 0} headset client{operator?.connection?.websocket_clients === 1 ? "" : "s"}
          </Badge>
          <Badge tone={headCamera ? "success" : "warning"}>{headCamera ? "head camera ready" : "head camera missing"}</Badge>
          <Badge tone={operator?.recording?.active ? "danger" : operator?.recording?.ready ? "success" : "warning"}>
            {operator?.recording?.active ? `recording ${operator.recording.frames} frames` : operator?.recording?.ready ? "recording ready" : "recording blocked"}
          </Badge>
        </div>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-4">
        {(["connection", "mirror", "teleop", "suspend"] as const).map((step) => {
          const active = (
            (step === "connection" && operator?.stage === "connect_required") ||
            (step === "mirror" && (operator?.stage === "mirror_waiting_robot" || operator?.stage === "mirror_ready")) ||
            (step === "teleop" && (operator?.stage === "teleop_head_only" || operator?.stage === "teleop_arms")) ||
            (step === "suspend" && operator?.stage === "suspended")
          );
          return (
            <div key={step} className={active ? "rounded-md border border-primary/50 bg-primary/10 p-3" : "rounded-md border border-border bg-muted/40 p-3"}>
              <div className="text-xs font-semibold uppercase text-muted-foreground">{step}</div>
              <p className="mt-1 text-xs text-muted-foreground">
                {step === "connection" ? "Backend, WebSocket, arms, and cameras visible in headset." :
                  step === "mirror" ? "Operator faces the workspace and confirms Ready." :
                  step === "teleop" ? "Grip anchors the arms; trigger controls gripper." :
                  "Warnings, stale tracking, and emergency stop stay visible."}
              </p>
            </div>
          );
        })}
      </div>
      {blockers.length ? (
        <p className="mt-3 text-sm text-warning">Operator blockers: {blockers.join("; ")}</p>
      ) : null}
    </Card>
  );
}

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

function QuestNativeBridge({ status, bridge, video, refreshVideo }: {
  status?: VRStatus;
  bridge?: QuestBridgeStatus;
  video?: QuestVideoStatus;
  refreshVideo: () => Promise<QuestVideoStatus | undefined>;
}) {
  const [questHost, setQuestHost] = useState("");
  const [questToken, setQuestToken] = useState("");
  const [videoBusy, setVideoBusy] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const host = typeof window !== "undefined" ? window.location.host : "<workstation>:5000";
  const scheme = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss" : "ws";
  const baseEndpoint = bridge?.ws_url ?? `${scheme}://${host}${bridge?.endpoint ?? "/api/vr/quest/ws"}`;
  const endpoint = questToken.trim()
    ? `${baseEndpoint}?token=${encodeURIComponent(questToken.trim())}`
    : baseEndpoint;
  const nativeReady = !!status?.operator?.connection.native_quest_ready;
  const token = questToken.trim();
  const startVideo = async () => {
    if (!questHost.trim() || !token) return;
    setVideoBusy(true);
    setVideoError(null);
    try {
      await api.questVideoStart(questHost.trim(), token);
      await refreshVideo();
    } catch (e) {
      setVideoError(String(e));
    } finally {
      setVideoBusy(false);
    }
  };
  const stopVideo = async () => {
    if (!token) return;
    setVideoBusy(true);
    setVideoError(null);
    try {
      await api.questVideoStop(token);
      await refreshVideo();
    } catch (e) {
      setVideoError(String(e));
    } finally {
      setVideoBusy(false);
    }
  };
  return (
    <Card className={nativeReady ? "border-success/60" : ""}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-sm font-semibold">Native Quest app</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Use the standalone Quest 3 app. It streams Unity/OpenXR controller poses to the backend; WebXR is no longer required for normal teleop.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge tone={nativeReady ? "success" : "neutral"}>
            {bridge?.clients ?? status?.operator?.connection.native_quest_clients ?? 0} native client{(bridge?.clients ?? 0) === 1 ? "" : "s"}
          </Badge>
          <Badge tone={video?.ready ? "success" : video?.gst_available === false ? "danger" : "warning"}>
            {video?.ready ? "video running" : video?.gst_available === false ? "gst missing" : "video stopped"}
          </Badge>
          <Badge tone="info">{bridge?.coordinate_frame ?? "quest_operator_frame"}</Badge>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2 rounded-md border border-border bg-muted p-3">
        <code className="mono flex-1 overflow-hidden text-ellipsis text-sm">{endpoint}</code>
        <Button variant="secondary" onClick={() => navigator.clipboard.writeText(endpoint)}>
          <Copy size={16} />Copy
        </Button>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Pairing token is not exposed by the public status API. Set `XLE_QUEST_PAIRING_TOKEN` before starting the backend and enter the same value here and in the Quest app.
      </p>
      <input
        className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
        placeholder="Quest pairing token"
        value={questToken}
        onChange={(e) => setQuestToken(e.target.value)}
      />
      <div className="mt-3 grid gap-2 md:grid-cols-[1fr_auto_auto]">
        <input
          className="rounded-md border border-border bg-background px-3 py-2 text-sm"
          placeholder="Quest IP for video, e.g. 192.168.1.55"
          value={questHost}
          onChange={(e) => setQuestHost(e.target.value)}
        />
        <Button variant="secondary" disabled={videoBusy || !questHost.trim() || !token} onClick={startVideo}>Start video</Button>
        <Button variant="secondary" disabled={videoBusy || !token || !video?.running} onClick={stopVideo}>Stop video</Button>
      </div>
      {video?.running_roles?.length ? (
        <p className="mt-2 text-xs text-muted-foreground">
          Streaming to {video.quest_host}: {video.running_roles.join(", ")}
        </p>
      ) : null}
      {video ? (
        <p className="mt-2 text-xs text-muted-foreground">
          Quest video config: UDP base {video.base_port}, {video.bitrate_kbps} kbps.
        </p>
      ) : null}
      {videoError ? <p className="mt-2 text-xs text-danger">{videoError}</p> : null}
      {bridge?.last_seen_ms != null ? (
        <p className="mt-2 text-xs text-muted-foreground">Last native packet: {bridge.last_seen_ms} ms ago.</p>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">Waiting for the Quest app to connect.</p>
      )}
      {video?.missing_roles?.length ? (
        <p className="mt-2 text-xs text-warning">Missing camera roles for headset video: {video.missing_roles.join(", ")}</p>
      ) : null}
      {video?.streams?.length ? (
        <div className="mt-3 grid gap-2 md:grid-cols-3">
          {video.streams.map((stream) => (
            <div key={stream.role} className="rounded-md border border-border bg-muted/40 p-2 text-xs">
              {(() => {
                const health = video.receive_health?.[stream.role];
                return health ? (
                  <div className="mb-1 text-success">
                    headset {health.state} · {health.fps.toFixed(1)} fps · {Math.round(health.latency_ms)} ms
                  </div>
                ) : null;
              })()}
              <div className="font-medium">{stream.role}</div>
              <div className="mt-1 text-muted-foreground">
                {stream.camera_name} · {stream.width}×{stream.height}@{stream.fps} · UDP {stream.udp_port}
              </div>
              <div className={stream.running ? "mt-1 text-success" : "mt-1 text-muted-foreground"}>
                {stream.running ? `running pid ${stream.pid}` : "stopped"}
              </div>
              {stream.last_error ? <div className="mt-1 text-danger">{stream.last_error}</div> : null}
            </div>
          ))}
        </div>
      ) : null}
    </Card>
  );
}

export function VRTeleop() {
  const { data, mutate, isLoading } = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1000 });
  const { data: questBridge } = useSWR<QuestBridgeStatus>("/api/vr/quest/status", fetcher, { refreshInterval: 1000 });
  const { data: questVideo, mutate: mutateQuestVideo } = useSWR<QuestVideoStatus>("/api/vr/quest/video/status", fetcher, { refreshInterval: 3000 });
  const [busy, setBusy] = useState(false);
  const [confirmStop, setConfirmStop] = useState(false);
  const refresh = () => mutate();
  const endpoint = data?.vr_endpoint;

  return (
    <Page title="VR Operator" description="Reachy-style Quest workflow: connection, mirror/ready, teleop, and suspension state with backend-enforced recording blockers.">
      <OperatorFlow status={data} />
      <QuestNativeBridge status={data} bridge={questBridge} video={questVideo} refreshVideo={mutateQuestVideo} />

      <div className="grid gap-4 xl:grid-cols-2">
        <ArmControl side="left" status={data} refresh={refresh} busy={busy || isLoading} setBusy={setBusy} />
        <ArmControl side="right" status={data} refresh={refresh} busy={busy || isLoading} setBusy={setBusy} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.8fr]">
        <Engagement status={data} refresh={refresh} />
        <Card>
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold">Quest operator room</h2>
              <p className="mt-1 text-xs text-muted-foreground">Open this URL in the Quest browser. The headset shows the same operator stage and blockers as this card.</p>
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
