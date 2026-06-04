import { CheckCircle2, Home, Play, RotateCcw, SlidersHorizontal, Square, Trash2, X } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";
import { ArmSide, VRStatus, api, fetcher } from "../api";
import { Badge, Button, Card, Field, Page, SelectField, TextInput } from "../components/ui";

const arms: ArmSide[] = ["left", "right"];
const directions = ["forward", "back", "left", "right", "up", "down"];
const wristStates = new Set([
  "awaiting_anchor_wrist_verify",
  "motioning_wrist_verify",
  "awaiting_anchor_wrist_pitch",
  "motioning_wrist_pitch",
  "awaiting_anchor_wrist_roll",
  "motioning_wrist_roll",
]);

function run(work: () => Promise<unknown>, refresh: () => Promise<unknown>) {
  return work().then(refresh).catch((err) => alert(String(err)));
}

function wizardPrompt(state: string) {
  switch (state) {
    case "awaiting_anchor_fwd":
      return { title: "Forward", detail: "Squeeze and hold grip, move the controller forward, then release." };
    case "motioning_fwd":
      return { title: "Forward", detail: "Keep grip held while moving forward. Release after the distance passes the minimum." };
    case "awaiting_anchor_up":
      return { title: "Up", detail: "Squeeze and hold grip, move the controller up, then release." };
    case "motioning_up":
      return { title: "Up", detail: "Keep grip held while moving up. Release after the distance passes the minimum." };
    case "awaiting_anchor_left":
      return { title: "Left", detail: "Squeeze and hold grip, move the controller left, then release." };
    case "motioning_left":
      return { title: "Left", detail: "Keep grip held while moving left. Release after the distance passes the minimum." };
    case "awaiting_anchor_wrist_verify":
    case "awaiting_anchor_wrist_pitch":
      return { title: "Wrist pitch", detail: "Optional: squeeze grip, pitch wrist up, then release. You can skip this step." };
    case "motioning_wrist_verify":
    case "motioning_wrist_pitch":
      return { title: "Wrist pitch", detail: "Keep grip held while pitching up. Release after the angle passes the minimum." };
    case "awaiting_anchor_wrist_roll":
      return { title: "Wrist roll", detail: "Optional: squeeze grip, roll wrist right, then release. You can skip this step." };
    case "motioning_wrist_roll":
      return { title: "Wrist roll", detail: "Keep grip held while rolling right. Release after the angle passes the minimum." };
    default:
      return { title: "Idle", detail: "Start the wizard, then use the selected Quest controller grip for each prompted motion." };
  }
}

function pct(value: number, target: number) {
  if (!Number.isFinite(value) || !Number.isFinite(target) || target <= 0) return 0;
  return Math.max(0, Math.min(100, (value / target) * 100));
}

function cmVec(vec?: [number, number, number] | null) {
  return vec ? vec.map((v) => `${(v * 100).toFixed(1)}`).join(" / ") : "—";
}

function ProfileCard({ status, refresh }: { status?: VRStatus; refresh: () => Promise<unknown> }) {
  const [newProfile, setNewProfile] = useState("");
  const profiles = status?.calibration_profiles?.profiles ?? [];
  const active = status?.calibration_profiles?.active_profile ?? "default";
  return (
    <Card>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
        <Field label="Calibration save profile">
          <SelectField
            value={active}
            onValueChange={(profile) => run(() => api.vrSelectCalibrationProfile(profile), refresh)}
            options={(profiles.length ? profiles : [{ name: active }]).map((p) => ({ value: p.name, label: p.name === active ? `${p.name} (active)` : p.name }))}
          />
        </Field>
        <Field label="New profile">
          <TextInput value={newProfile} onChange={(e) => setNewProfile(e.currentTarget.value)} placeholder="user_or_table_setup" />
        </Field>
        <Button disabled={!newProfile.trim()} onClick={() => run(() => api.vrCreateCalibrationProfile(newProfile.trim(), true), refresh).then(() => setNewProfile(""))}>
          Create
        </Button>
        <Button variant="secondary" disabled={profiles.length <= 1} onClick={() => run(() => api.vrDeleteCalibrationProfile(active), refresh)}>
          <Trash2 size={16} />Delete active
        </Button>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {profiles.map((p) => (
          <Badge key={p.name} tone={p.name === active ? "info" : "neutral"}>
            {p.name}: left {p.left_robot_verified ? "verified" : p.left_saved ? "saved" : "empty"} · right {p.right_robot_verified ? "verified" : p.right_saved ? "saved" : "empty"}
          </Badge>
        ))}
      </div>
    </Card>
  );
}

function ArmCalibration({ side, status, refresh }: { side: ArmSide; status?: VRStatus; refresh: () => Promise<unknown> }) {
  const arm = status?.arms?.[side];
  const calibration = arm?.calibration;
  const wizardState = calibration?.wizard_state ?? "idle";
  const wizardActive = wizardState !== "idle";
  const wizard = wizardPrompt(wizardState);
  const isWristStep = wristStates.has(wizardState);
  const wristDeg = wizardState.includes("roll")
    ? calibration?.wizard_wrist_roll_verify_deg ?? 0
    : calibration?.wizard_wrist_pitch_verify_deg ?? calibration?.wizard_wrist_verify_deg ?? 0;
  const liveValue = isWristStep ? wristDeg : (calibration?.wizard_motion_m ?? 0) * 100;
  const minValue = isWristStep ? calibration?.wizard_wrist_verify_min_deg ?? 0 : (calibration?.wizard_min_m ?? 0) * 100;
  const targetValue = isWristStep ? calibration?.wizard_wrist_verify_target_deg ?? 1 : (calibration?.wizard_target_m ?? 0.1) * 100;
  const progress = pct(liveValue, targetValue);
  const controllerAge = arm?.controller.age_ms;
  const controllerLive = typeof controllerAge === "number" && controllerAge < 2000;
  const verification = arm?.calibration.robot_verification;
  const live = verification?.live;
  const alignmentTone = live?.ready
    ? "success"
    : live?.state === "adjust" || live?.state === "robot_motion_too_small" || live?.state === "stale_controller"
      ? "warning"
      : "neutral";
  const readiness = verification?.readiness ?? "stage1_only";
  const readinessLabel = readiness === "ready_to_record"
    ? "ready to record"
    : readiness === "verified_test_pending"
      ? "verified, test pending"
      : "stage 1 only";
  const [label, setLabel] = useState("forward");
  return (
    <Card>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-sm font-semibold capitalize">{side} calibration</h2>
          <div className="mt-2 flex flex-wrap gap-2">
            <Badge tone={arm?.connected ? "success" : "neutral"}>{arm?.connected ? "connected" : "offline"}</Badge>
            <Badge tone={arm?.calibrated ? "success" : "neutral"}>{arm?.calibrated ? "vr calibrated" : "not calibrated"}</Badge>
            <Badge tone={verification?.quality === "good" ? "success" : verification?.quality === "warn" ? "warning" : "neutral"}>{verification?.quality ?? "unverified"}</Badge>
            <Badge tone={readiness === "ready_to_record" ? "success" : readiness === "verified_test_pending" ? "warning" : "neutral"}>
              {readinessLabel}
            </Badge>
            <Badge tone={verification?.test_completed ? "success" : verification?.test_active ? "info" : "neutral"}>
              {verification?.test_completed ? "low-scale tested" : verification?.test_active ? "low-scale active" : "low-scale pending"}
            </Badge>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button disabled={!arm?.connected} onClick={() => run(() => api.vrCalibrateStart(side), refresh)}>
            <SlidersHorizontal size={16} />{wizardActive ? "Restart VR wizard" : "Start VR wizard"}
          </Button>
          {wizardActive ? (
            <Button variant="secondary" onClick={() => run(() => api.vrCalibrateCancel(side), refresh)}>
              <X size={16} />Cancel
            </Button>
          ) : null}
        </div>
      </div>

      <div className={wizardActive ? "mt-4 rounded-md border border-blue-500/30 bg-blue-500/10 p-3" : "mt-4 rounded-md border border-border p-3"}>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h3 className="text-xs font-semibold uppercase text-muted-foreground">VR wizard</h3>
            <p className="mt-1 text-sm font-medium">{wizard.title}</p>
            <p className="mt-1 text-xs text-muted-foreground">{wizard.detail}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge tone={wizardActive ? "info" : calibration?.persisted?.saved ? "success" : "neutral"}>
              {wizardActive ? wizardState.replaceAll("_", " ") : calibration?.persisted?.saved ? "saved" : "not started"}
            </Badge>
            <Badge tone={controllerLive ? "success" : "warning"}>
              {controllerLive ? `controller ${controllerAge} ms` : "controller waiting"}
            </Badge>
            <Badge tone={arm?.controller.mode === "reset" ? "info" : "neutral"}>{arm?.controller.mode ?? "idle"}</Badge>
          </div>
        </div>
        {wizardActive ? (
          <div className="mt-3 space-y-2">
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
              <span>{isWristStep ? "Angle" : "Motion"}: {liveValue.toFixed(1)}{isWristStep ? " deg" : " cm"}</span>
              <span>Minimum: {minValue.toFixed(1)}{isWristStep ? " deg" : " cm"}</span>
              <span>Target: {targetValue.toFixed(1)}{isWristStep ? " deg" : " cm"}</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge tone={calibration?.wizard_fwd_captured ? "success" : wizardState.includes("fwd") ? "info" : "neutral"}>forward</Badge>
              <Badge tone={calibration?.wizard_up_captured ? "success" : wizardState.includes("up") ? "info" : "neutral"}>up</Badge>
              <Badge tone={calibration?.wizard_left_captured ? "success" : wizardState.includes("left") ? "info" : "neutral"}>left</Badge>
              <Badge tone={calibration?.wizard_wrist_pitch_captured ? "success" : wizardState.includes("wrist_pitch") || wizardState.includes("wrist_verify") ? "info" : "neutral"}>wrist pitch</Badge>
              <Badge tone={calibration?.wizard_wrist_roll_captured ? "success" : wizardState.includes("wrist_roll") ? "info" : "neutral"}>wrist roll</Badge>
              {isWristStep ? (
                <Button variant="secondary" onClick={() => run(() => api.vrCalibrateSkipWristVerify(side), refresh)}>
                  Skip wrist verify
                </Button>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>

      <div className="mt-4 grid gap-3 xl:grid-cols-2">
        <div className="rounded-md border border-border p-3">
          <h3 className="text-xs font-semibold uppercase text-muted-foreground">Home pose</h3>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrHomeCapture(side), refresh)}><Home size={16} />Capture</Button>
            <Button variant="secondary" disabled={!arm?.connected || !arm?.home?.captured} onClick={() => run(() => api.vrHomeGo(side), refresh)}><RotateCcw size={16} />Go home</Button>
          </div>
        </div>

        <div className="rounded-md border border-border p-3">
          <h3 className="text-xs font-semibold uppercase text-muted-foreground">Robot verification</h3>
          <div className="mt-3 grid gap-2 sm:grid-cols-[1fr_auto]">
            <SelectField value={label} onValueChange={setLabel} options={directions.map((d) => ({ value: d, label: d }))} />
            <Button variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyStart(side), refresh)}>
              Start
            </Button>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyRobotPose(side, "start"), refresh)}>Robot start</Button>
            <Button variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyVrPose(side, "start", label), refresh)}>VR start</Button>
            <Button variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyRobotPose(side, "end"), refresh)}>Robot end</Button>
            <Button variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyVrPose(side, "end", label), refresh)}>VR end</Button>
            <Button disabled={!arm?.connected || (verification?.sample_count ?? 0) < (verification?.min_samples ?? 6)} onClick={() => run(() => api.vrRobotVerifySolve(side), refresh)}>
              <CheckCircle2 size={16} />Solve
            </Button>
            <Button variant="secondary" disabled={!arm?.connected || verification?.quality !== "good" || verification?.test_active} onClick={() => run(() => api.vrRobotVerifyTestStart(side), refresh)}>
              <Play size={16} />Start low-scale test
            </Button>
            <Button variant="secondary" disabled={!verification?.test_active} onClick={() => run(() => api.vrRobotVerifyTestStop(side), refresh)}>
              <Square size={16} />Stop test
            </Button>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Samples: {verification?.sample_count ?? 0}/{verification?.min_samples ?? 6} · fit {verification?.fit_error_cm?.toFixed(2) ?? "—"} cm
          </p>
          <div className="mt-3 rounded-md border border-border bg-muted/40 p-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <h4 className="text-xs font-semibold uppercase text-muted-foreground">VR alignment</h4>
                <p className="mt-1 text-sm">{live?.message ?? "Start robot verification to see live alignment."}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge tone={alignmentTone}>{live?.ready ? "match" : live?.state?.replaceAll("_", " ") ?? "waiting"}</Badge>
                <Badge tone={live?.preview_source === "provisional_lstsq" ? "info" : live?.preview_source === "verified" ? "success" : "neutral"}>
                  {live?.preview_source === "provisional_lstsq" ? "provisional fit" : live?.preview_source === "verified" ? "verified map" : "stage 1 preview"}
                </Badge>
                <Badge tone={live?.controller_age_ms != null && live.controller_age_ms < 2000 ? "success" : "warning"}>
                  {live?.controller_age_ms != null ? `controller ${live.controller_age_ms} ms` : "controller waiting"}
                </Badge>
              </div>
            </div>
            <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
              <div className="rounded-md border border-border bg-background p-2">
                <div className="text-muted-foreground">Direction error</div>
                <div className="mt-1 text-sm font-medium">{live?.direction_error_deg != null ? `${live.direction_error_deg.toFixed(1)} deg` : "—"}</div>
              </div>
              <div className="rounded-md border border-border bg-background p-2">
                <div className="text-muted-foreground">Scale ratio</div>
                <div className="mt-1 text-sm font-medium">{live?.magnitude_ratio != null ? `${live.magnitude_ratio.toFixed(2)}x` : "—"}</div>
              </div>
              <div className="rounded-md border border-border bg-background p-2">
                <div className="text-muted-foreground">Position error</div>
                <div className="mt-1 text-sm font-medium">{live?.position_error_cm != null ? `${live.position_error_cm.toFixed(1)} cm` : "—"}</div>
              </div>
            </div>
            <div className="mt-3 grid gap-2 text-xs lg:grid-cols-2">
              <div className="rounded-md border border-border bg-background p-2">
                <div className="text-muted-foreground">Robot target delta X/Y/Z cm</div>
                <div className="mono mt-1 text-sm">{cmVec(live?.target_robot_delta)}</div>
              </div>
              <div className="rounded-md border border-border bg-background p-2">
                <div className="text-muted-foreground">VR predicted robot delta X/Y/Z cm</div>
                <div className="mono mt-1 text-sm">{cmVec(live?.predicted_robot_delta)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

export function Calibration() {
  const { data, mutate } = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1000 });
  return (
    <Page title="Calibration" description="Robot dropdowns and calibration profiles are loaded from YAML/API state, with separate VR direction calibration and robot-verified refinement.">
      <ProfileCard status={data} refresh={() => mutate()} />
      <div className="grid gap-4">
        {arms.map((side) => <ArmCalibration key={side} side={side} status={data} refresh={() => mutate()} />)}
      </div>
    </Page>
  );
}
