import { CheckCircle2, Home, Lock, Play, RotateCcw, ShieldCheck, SkipForward, SlidersHorizontal, Square, Trash2, Unlock, X } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";
import { ArmSide, VRArmState, VRStatus, api, fetcher } from "../api";
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

function pct(value: number, target: number) {
  if (!Number.isFinite(value) || !Number.isFinite(target) || target <= 0) return 0;
  return Math.max(0, Math.min(100, (value / target) * 100));
}

function cmVec(vec?: [number, number, number] | null) {
  return vec ? vec.map((v) => `${(v * 100).toFixed(1)}`).join(" / ") : "-";
}

function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

function wizardStep(state: string) {
  switch (state) {
    case "awaiting_anchor_up":
    case "motioning_up":
      return 2;
    case "awaiting_anchor_left":
    case "motioning_left":
      return 3;
    case "awaiting_anchor_wrist_verify":
    case "awaiting_anchor_wrist_pitch":
    case "motioning_wrist_verify":
    case "motioning_wrist_pitch":
      return 4;
    case "awaiting_anchor_wrist_roll":
    case "motioning_wrist_roll":
      return 5;
    default:
      return 1;
  }
}

function wizardPrompt(state: string, side: ArmSide) {
  switch (state) {
    case "awaiting_anchor_fwd":
      return { title: "Set forward start", detail: `Hold ${side} grip where your hand starts. Keep the controller steady before moving.` };
    case "motioning_fwd":
      return { title: "Move forward", detail: "Keep grip held, move straight forward about 10 cm, then release." };
    case "awaiting_anchor_up":
      return { title: "Set upward start", detail: `Hold ${side} grip again at a comfortable neutral position.` };
    case "motioning_up":
      return { title: "Move up", detail: "Keep grip held, move straight upward about 10 cm, then release." };
    case "awaiting_anchor_left":
      return { title: "Set leftward start", detail: `Hold ${side} grip again. This step detects mirrored left/right mapping.` };
    case "motioning_left":
      return { title: "Move left", detail: "Keep grip held, move to your left about 10 cm, then release." };
    case "awaiting_anchor_wrist_verify":
    case "awaiting_anchor_wrist_pitch":
      return { title: "Optional wrist pitch", detail: "Hold grip, then pitch the wrist upward clearly. Skip if you want the default Quest wrist axes." };
    case "motioning_wrist_verify":
    case "motioning_wrist_pitch":
      return { title: "Pitch wrist up", detail: "Keep grip held, pitch upward 20-45 degrees, then release." };
    case "awaiting_anchor_wrist_roll":
      return { title: "Optional wrist roll", detail: "Hold grip, then roll the wrist to the right. Skip to keep the default roll axis." };
    case "motioning_wrist_roll":
      return { title: "Roll wrist right", detail: "Keep grip held, roll right 20-45 degrees, then release." };
    default:
      return { title: "Ready to calibrate", detail: "Start the VR wizard, then follow the large instruction shown here and inside the Quest app." };
  }
}

function WizardSteps({ arm }: { arm?: VRArmState }) {
  const cal = arm?.calibration;
  const state = cal?.wizard_state ?? "idle";
  const activeStep = state === "idle" ? 0 : wizardStep(state);
  const steps = [
    { label: "Forward", done: !!cal?.wizard_fwd_captured },
    { label: "Up", done: !!cal?.wizard_up_captured },
    { label: "Left", done: !!cal?.wizard_left_captured },
    { label: "Pitch", done: !!cal?.wizard_wrist_pitch_captured },
    { label: "Roll", done: !!cal?.wizard_wrist_roll_captured },
  ];
  return (
    <div className="grid gap-2 md:grid-cols-5">
      {steps.map((step, idx) => {
        const number = idx + 1;
        const active = activeStep === number;
        return (
          <div key={step.label} className={active ? "rounded-md border border-primary/60 bg-primary/10 p-3" : "rounded-md border border-border bg-muted/35 p-3"}>
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold uppercase text-muted-foreground">Step {number}</span>
              <Badge tone={step.done ? "success" : active ? "info" : "neutral"}>{step.done ? "done" : active ? "now" : "next"}</Badge>
            </div>
            <div className="mt-2 text-sm font-medium">{step.label}</div>
          </div>
        );
      })}
    </div>
  );
}

function ProfileCard({ status, refresh }: { status?: VRStatus; refresh: () => Promise<unknown> }) {
  const [newProfile, setNewProfile] = useState("");
  const profiles = status?.calibration_profiles?.profiles ?? [];
  const active = status?.calibration_profiles?.active_profile ?? "default";
  const activeProfile = profiles.find((p) => p.name === active);
  const hasAnySaved = !!(activeProfile?.left_saved || activeProfile?.right_saved);
  return (
    <Card>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
        <Field label="Setup profile">
          <SelectField
            value={active}
            onValueChange={(profile) => run(() => api.vrSelectCalibrationProfile(profile), refresh)}
            options={(profiles.length ? profiles : [{ name: active }]).map((p) => ({ value: p.name, label: p.name === active ? `${p.name} (active)` : p.name }))}
          />
        </Field>
        <Field label="Create profile">
          <TextInput value={newProfile} onChange={(e) => setNewProfile(e.currentTarget.value)} placeholder="desk_left_camera_v1" />
        </Field>
        <Button disabled={!newProfile.trim()} onClick={() => run(() => api.vrCreateCalibrationProfile(newProfile.trim(), true), refresh).then(() => setNewProfile(""))}>
          Create
        </Button>
        <Button variant="secondary" disabled={profiles.length <= 1} onClick={() => run(() => api.vrDeleteCalibrationProfile(active), refresh)}>
          <Trash2 size={16} />Delete active
        </Button>
      </div>
      <div className={hasAnySaved ? "mt-4 rounded-md border border-success/30 bg-success/10 p-3" : "mt-4 rounded-md border border-warning/30 bg-warning/10 p-3"}>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h2 className="text-sm font-semibold">Active profile: {active}</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {hasAnySaved
                ? "Saved calibration exists for this profile. Starting a new VR wizard will overwrite the selected arm after completion."
                : "No saved arm calibration exists in this profile yet."}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge tone={activeProfile?.left_robot_verified ? "success" : activeProfile?.left_saved ? "info" : "warning"}>
              Left: {activeProfile?.left_robot_verified ? "robot verified" : activeProfile?.left_saved ? "VR saved" : "not saved"}
            </Badge>
            <Badge tone={activeProfile?.right_robot_verified ? "success" : activeProfile?.right_saved ? "info" : "warning"}>
              Right: {activeProfile?.right_robot_verified ? "robot verified" : activeProfile?.right_saved ? "VR saved" : "not saved"}
            </Badge>
            {activeProfile?.updated_at ? <Badge tone="neutral">updated {activeProfile.updated_at}</Badge> : null}
          </div>
        </div>
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

function ArmPicker({ status, selected, onSelect }: { status?: VRStatus; selected: ArmSide; onSelect: (side: ArmSide) => void }) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {arms.map((side) => {
        const arm = status?.arms?.[side];
        const verify = arm?.calibration.robot_verification;
        const selectedClass = selected === side ? "border-primary bg-primary/10" : "border-border bg-card";
        return (
          <button key={side} onClick={() => onSelect(side)} className={`rounded-lg border p-4 text-left transition hover:bg-muted/60 ${selectedClass}`}>
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-base font-semibold capitalize">{side} arm</div>
                <div className="mt-1 text-xs text-muted-foreground">{arm?.connected ? "Connected" : "Offline"}</div>
              </div>
              <Badge tone={selected === side ? "info" : "neutral"}>{selected === side ? "selected" : "select"}</Badge>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <Badge tone={arm?.connected ? "success" : "neutral"}>{arm?.connected ? "connected" : "offline"}</Badge>
              <Badge tone={arm?.calibration.persisted?.saved ? "success" : "neutral"}>{arm?.calibration.persisted?.saved ? "VR saved" : "VR missing"}</Badge>
              <Badge tone={verify?.readiness === "ready_to_record" ? "success" : verify?.quality === "good" ? "warning" : "neutral"}>
                {verify?.readiness === "ready_to_record" ? "recording ready" : verify?.quality === "good" ? "test pending" : verify?.quality ?? "unverified"}
              </Badge>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function VRWizardPanel({ side, arm, refresh }: { side: ArmSide; arm?: VRArmState; refresh: () => Promise<unknown> }) {
  const cal = arm?.calibration;
  const state = cal?.wizard_state ?? "idle";
  const active = state !== "idle";
  const prompt = wizardPrompt(state, side);
  const isWristStep = wristStates.has(state);
  const wristDeg = state.includes("roll")
    ? cal?.wizard_wrist_roll_verify_deg ?? 0
    : cal?.wizard_wrist_pitch_verify_deg ?? cal?.wizard_wrist_verify_deg ?? 0;
  const liveValue = isWristStep ? wristDeg : (cal?.wizard_motion_m ?? 0) * 100;
  const minValue = isWristStep ? cal?.wizard_wrist_verify_min_deg ?? 0 : (cal?.wizard_min_m ?? 0) * 100;
  const targetValue = isWristStep ? cal?.wizard_wrist_verify_target_deg ?? 1 : (cal?.wizard_target_m ?? 0.1) * 100;
  const progress = pct(liveValue, targetValue);
  const controllerAge = arm?.controller.age_ms;
  const controllerLive = typeof controllerAge === "number" && controllerAge < 2000;
  const saved = !!cal?.persisted?.saved;
  return (
    <Card className={active ? "border-primary/60" : ""}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-sm font-semibold">Stage 1: VR direction calibration</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Captures your forward, up, and left motions so Quest controller movement maps to the robot frame.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge tone={arm?.connected ? "success" : "warning"}>{arm?.connected ? "arm connected" : "connect arm first"}</Badge>
          <Badge tone={controllerLive ? "success" : "warning"}>{controllerLive ? `controller ${controllerAge} ms` : "move controller"}</Badge>
          <Badge tone={saved ? "success" : "neutral"}>{saved ? "saved" : "not saved"}</Badge>
        </div>
      </div>

      <div className="mt-4 rounded-md border border-border bg-muted/30 p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="text-xs font-semibold uppercase text-muted-foreground">{active ? `Current step ${wizardStep(state)}/5` : "Not running"}</div>
            <div className="mt-1 text-xl font-semibold">{prompt.title}</div>
            <p className="mt-2 max-w-3xl text-sm text-muted-foreground">{prompt.detail}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button disabled={!arm?.connected} onClick={() => run(() => api.vrCalibrateStart(side), refresh)}>
              <SlidersHorizontal size={16} />{active ? "Restart" : "Start"}
            </Button>
            {active ? (
              <Button variant="secondary" onClick={() => run(() => api.vrCalibrateCancel(side), refresh)}>
                <X size={16} />Cancel
              </Button>
            ) : null}
            {active && isWristStep ? (
              <Button variant="secondary" onClick={() => run(() => api.vrCalibrateSkipWristVerify(side), refresh)}>
                <SkipForward size={16} />Skip wrist
              </Button>
            ) : null}
          </div>
        </div>

        <div className="mt-4">
          <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
            <span>{isWristStep ? "Rotation" : "Motion"}</span>
            <span>{liveValue.toFixed(1)} / {targetValue.toFixed(1)} {isWristStep ? "deg" : "cm"} · minimum {minValue.toFixed(1)}</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-background">
            <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </div>

      <div className="mt-4">
        <WizardSteps arm={arm} />
      </div>

      {cal?.confidence === "poor" ? (
        <p className="mt-3 text-sm text-warning">Calibration quality is poor. Re-run with straighter, more separated forward/up/left motions.</p>
      ) : null}
    </Card>
  );
}

function RobotVerificationPanel({ side, arm, refresh }: { side: ArmSide; arm?: VRArmState; refresh: () => Promise<unknown> }) {
  const [label, setLabel] = useState("forward");
  const verification = arm?.calibration.robot_verification;
  const live = verification?.live;
  const sampleCount = verification?.sample_count ?? 0;
  const minSamples = verification?.min_samples ?? 6;
  const missing = verification?.missing_labels ?? [];
  const passErrorCm = verification?.pass_error_cm ?? 3;
  const residuals = [...(verification?.sample_residuals ?? [])].sort((a, b) => b.residual_cm - a.residual_cm);
  const badResidualLabels = residuals
    .filter((item) => item.residual_cm > passErrorCm)
    .map((item) => item.label)
    .filter((value, index, values) => values.indexOf(value) === index)
    .slice(0, 3);
  const canSolve = sampleCount >= minSamples && missing.length === 0;
  const ready = verification?.readiness === "ready_to_record";
  const collecting = verification?.quality === "collecting" || verification?.state !== "idle";
  const alignmentTone = live?.ready
    ? "success"
    : live?.state === "adjust" || live?.state === "robot_motion_too_small" || live?.state === "stale_controller"
      ? "warning"
      : "neutral";
  return (
    <Card>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-sm font-semibold">Stage 2: Robot verification</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Collect paired robot and VR motions. This verifies direction, estimates scale, and unlocks dataset recording.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge tone={ready ? "success" : verification?.quality === "good" ? "warning" : collecting ? "info" : "neutral"}>
            {ready ? "ready to record" : verification?.quality === "good" ? "low-scale test pending" : verification?.quality ?? "unverified"}
          </Badge>
          <Badge tone={sampleCount >= minSamples ? "success" : "warning"}>{sampleCount}/{minSamples} samples</Badge>
          <Badge tone={verification?.test_active ? "info" : verification?.test_completed ? "success" : "neutral"}>
            {verification?.test_active ? "test active" : verification?.test_completed ? "test passed" : "test pending"}
          </Badge>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[0.8fr_1.2fr]">
        <div className="rounded-md border border-border bg-muted/30 p-4">
          <Field label="Sample direction">
            <SelectField value={label} onValueChange={setLabel} options={directions.map((d) => ({ value: d, label: titleCase(d) }))} />
          </Field>
          <div className="mt-4 space-y-2">
            <Button className="w-full justify-start" variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyStart(side), refresh)}>
              <Play size={16} />Start / reset verification
            </Button>
            <div className="grid gap-2 sm:grid-cols-2">
              <Button className="justify-start" variant="secondary" disabled={!arm?.connected || !collecting || !arm?.torque_enabled} onClick={() => run(() => api.vrTorqueRelease(side), refresh)}>
                <Unlock size={16} />Release torque
              </Button>
              <Button className="justify-start" variant="secondary" disabled={!arm?.connected || arm?.torque_enabled} onClick={() => run(() => api.vrTorqueLock(side), refresh)}>
                <Lock size={16} />Lock torque
              </Button>
            </div>
            <Button className="w-full justify-start" variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyRobotPose(side, "start", label), refresh)}>
              1. Capture robot start
            </Button>
            <Button className="w-full justify-start" variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyRobotPose(side, "end", label), refresh)}>
              2. Move robot, capture robot end
            </Button>
            <Button className="w-full justify-start" variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyVrPose(side, "start", label), refresh)}>
              3. Press A/X in Quest for VR start
            </Button>
            <Button className="w-full justify-start" variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrRobotVerifyVrPose(side, "end", label), refresh)}>
              4. Move with grip, press A/X for VR end
            </Button>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Start verification keeps torque on. Support the arm before pressing Release torque; robot-end capture locks torque again if it was off. VR-end capture keeps torque holding the arm.
            Browser buttons remain as fallback. In the headset, A captures VR start/end on the right arm; X does the same for the left arm.
          </p>
        </div>

        <div className="rounded-md border border-border bg-muted/30 p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h3 className="text-sm font-semibold">Live alignment</h3>
              <p className="mt-1 text-sm text-muted-foreground">{live?.message ?? "Start robot verification to see live alignment guidance."}</p>
            </div>
            <Badge tone={alignmentTone}>{live?.ready ? "matched" : live?.state?.replaceAll("_", " ") ?? "waiting"}</Badge>
          </div>
          <div className="mt-4 grid gap-2 text-xs md:grid-cols-3">
            <div className="rounded-md border border-border bg-background p-2">
              <div className="text-muted-foreground">Direction error</div>
              <div className="mt-1 text-sm font-medium">{live?.direction_error_deg != null ? `${live.direction_error_deg.toFixed(1)} deg` : "-"}</div>
            </div>
            <div className="rounded-md border border-border bg-background p-2">
              <div className="text-muted-foreground">Scale ratio</div>
              <div className="mt-1 text-sm font-medium">{live?.magnitude_ratio != null ? `${live.magnitude_ratio.toFixed(2)}x` : "-"}</div>
            </div>
            <div className="rounded-md border border-border bg-background p-2">
              <div className="text-muted-foreground">Position error</div>
              <div className="mt-1 text-sm font-medium">{live?.position_error_cm != null ? `${live.position_error_cm.toFixed(1)} cm` : "-"}</div>
            </div>
          </div>
          <div className="mt-3 grid gap-2 text-xs lg:grid-cols-2">
            <div className="rounded-md border border-border bg-background p-2">
              <div className="text-muted-foreground">Robot target X/Y/Z cm</div>
              <div className="mono mt-1 text-sm">{cmVec(live?.target_robot_delta)}</div>
            </div>
            <div className="rounded-md border border-border bg-background p-2">
              <div className="text-muted-foreground">VR predicted X/Y/Z cm</div>
              <div className="mono mt-1 text-sm">{cmVec(live?.predicted_robot_delta)}</div>
            </div>
          </div>
          {missing.length ? <p className="mt-3 text-sm text-warning">Missing directions: {missing.join(", ")}</p> : null}
          {residuals.length ? (
            <div className="mt-3 rounded-md border border-border bg-background p-3">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <h4 className="text-xs font-semibold uppercase text-muted-foreground">Solve residuals</h4>
                  <p className="mt-1 text-xs text-muted-foreground">
                    RMS must be {"<="} {passErrorCm.toFixed(1)} cm. Recapture the worst direction first, then solve again.
                  </p>
                </div>
                {badResidualLabels.length ? (
                  <Badge tone="warning">Recapture {badResidualLabels.join(", ")}</Badge>
                ) : (
                  <Badge tone="success">Within sample limit</Badge>
                )}
              </div>
              {verification?.residual_hint ? <p className="mt-2 text-xs text-warning">{verification.residual_hint}</p> : null}
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[36rem] text-left text-xs">
                  <thead className="border-b border-border text-muted-foreground">
                    <tr>
                      <th className="py-1 pr-3 font-medium">Direction</th>
                      <th className="py-1 pr-3 font-medium">Residual</th>
                      <th className="py-1 pr-3 font-medium">Angle</th>
                      <th className="py-1 pr-3 font-medium">Robot</th>
                      <th className="py-1 pr-3 font-medium">VR</th>
                      <th className="py-1 font-medium">Error X/Y/Z cm</th>
                    </tr>
                  </thead>
                  <tbody>
                    {residuals.map((item) => (
                      <tr key={`${item.index}-${item.label}`} className="border-b border-border/60 last:border-0">
                        <td className="py-1.5 pr-3 font-medium">{titleCase(item.label)}</td>
                        <td className={item.residual_cm > passErrorCm ? "py-1.5 pr-3 font-medium text-warning" : "py-1.5 pr-3"}>
                          {item.residual_cm.toFixed(1)} cm
                        </td>
                        <td className="py-1.5 pr-3">
                          {item.direction_error_deg != null ? `${item.direction_error_deg.toFixed(0)} deg` : "-"}
                        </td>
                        <td className="py-1.5 pr-3">{item.robot_motion_cm.toFixed(1)} cm</td>
                        <td className="py-1.5 pr-3">{item.vr_motion_cm.toFixed(1)} cm</td>
                        <td className="mono py-1.5">
                          {item.error_vector_cm ? item.error_vector_cm.map((v) => v.toFixed(1)).join(" / ") : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
          <div className="mt-4 flex flex-wrap gap-2">
            <Button disabled={!arm?.connected || !canSolve} onClick={() => run(() => api.vrRobotVerifySolve(side), refresh)}>
              <CheckCircle2 size={16} />Solve verification
            </Button>
            <Button variant="secondary" disabled={!arm?.connected || verification?.quality !== "good" || verification?.test_active} onClick={() => run(() => api.vrRobotVerifyTestStart(side), refresh)}>
              <ShieldCheck size={16} />Start low-scale test
            </Button>
            <Button variant="secondary" disabled={!verification?.test_active} onClick={() => run(() => api.vrRobotVerifyTestStop(side), refresh)}>
              <Square size={16} />Stop test
            </Button>
            <Button variant="secondary" disabled={!collecting} onClick={() => run(() => api.vrRobotVerifyCancel(side), refresh)}>
              <X size={16} />Cancel
            </Button>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">Fit error: {verification?.fit_error_cm?.toFixed(2) ?? "-"} cm RMS.</p>
        </div>
      </div>
    </Card>
  );
}

function SecondaryControls({ side, arm, refresh }: { side: ArmSide; arm?: VRArmState; refresh: () => Promise<unknown> }) {
  return (
    <Card>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-sm font-semibold">Home pose</h2>
          <p className="mt-1 text-sm text-muted-foreground">Save and return to a known physical starting pose for this arm.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" disabled={!arm?.connected} onClick={() => run(() => api.vrHomeCapture(side), refresh)}><Home size={16} />Capture home</Button>
          <Button variant="secondary" disabled={!arm?.connected || !arm?.home?.captured} onClick={() => run(() => api.vrHomeGo(side), refresh)}><RotateCcw size={16} />Go home</Button>
        </div>
      </div>
    </Card>
  );
}

export function Calibration() {
  const { data, mutate } = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1000 });
  const [selectedSide, setSelectedSide] = useState<ArmSide>("left");
  const refresh = () => mutate();
  const arm = data?.arms?.[selectedSide];
  return (
    <Page title="Calibration" description="Calibrate one arm at a time: first align Quest motion to the robot frame, then verify the mapping with paired robot and VR motions.">
      <ProfileCard status={data} refresh={refresh} />
      <ArmPicker status={data} selected={selectedSide} onSelect={setSelectedSide} />
      <VRWizardPanel side={selectedSide} arm={arm} refresh={refresh} />
      <RobotVerificationPanel side={selectedSide} arm={arm} refresh={refresh} />
      <SecondaryControls side={selectedSide} arm={arm} refresh={refresh} />
    </Page>
  );
}
