import { Check, Database, Home, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import useSWR from "swr";
import { VRStatus, api, fetcher } from "../api";
import { CameraPreviewGrid } from "../components/CameraPreviewGrid";
import { Badge, Button, Card, ConfirmDialog, Field, Page, TextInput } from "../components/ui";

type BadgeTone = "neutral" | "success" | "warning" | "danger" | "info";

function recordingGateBadge(info?: VRStatus["recording_info"]): { label: string; tone: BadgeTone } {
  if (!info) return { label: "loading", tone: "neutral" };
  if (info.transition_active) {
    return {
      label: info.transition_target === false ? "stopping recording" : "starting recording",
      tone: "info",
    };
  }
  if (info.active) return { label: "recording", tone: "info" };
  if (info.armed) return { label: "armed", tone: "warning" };
  if (info.start_allowed) {
    return info.anchor_pending
      ? { label: "anchor refresh pending", tone: "success" }
      : { label: "start allowed", tone: "success" };
  }

  const hardBlockers = info.start_blockers ?? info.calibration_blockers ?? [];
  const verificationBlockers = info.verification_blockers ?? [];
  const blockers = [...hardBlockers, ...verificationBlockers].map((item) => item.toLowerCase());
  const has = (needle: string) => blockers.some((item) => item.includes(needle));

  if (has("connect both")) return { label: "connect arms", tone: "warning" };
  if (has("quest app")) return { label: "Quest required", tone: "warning" };
  if (has("camera")) return { label: "camera setup required", tone: "warning" };
  if (has("home pose")) return { label: "home required", tone: "warning" };
  if (has("torque")) return { label: "torque required", tone: "warning" };
  if (has("low-scale calibration test is still active")) return { label: "test active", tone: "info" };
  if (has("low-scale calibration test not completed")) return { label: "low-scale test required", tone: "warning" };
  if (verificationBlockers.length || has("robot verification")) return { label: "verification required", tone: "warning" };
  if (hardBlockers.length) return { label: "recording blocked", tone: "warning" };
  return { label: "checking readiness", tone: "neutral" };
}

export function Recording() {
  const { data, mutate } = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1000 });
  const info = data?.recording_info;
  const [task, setTask] = useState("");
  const [root, setRoot] = useState("");
  const [repoId, setRepoId] = useState("");
  const [taskError, setTaskError] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [busyRecording, setBusyRecording] = useState(false);
  const [busyHome, setBusyHome] = useState(false);
  const [savingDatasetConfig, setSavingDatasetConfig] = useState(false);
  const active = !!data?.recording;
  const armed = !!data?.recording_armed || !!info?.armed;
  const recordingLocked = active || armed;
  const transitionActive = !!info?.transition_active;
  const recordingBusy = recordingLocked || transitionActive;
  const effectiveTask = task.trim() || info?.last_task?.trim() || info?.task_default?.trim() || "";
  const hardBlockers = info?.start_blockers ?? info?.calibration_blockers ?? [];
  const anchorBlockers = info?.anchor_blockers ?? [];
  const startAllowed = !!info?.start_allowed;
  const gateBadge = recordingGateBadge(info);
  const connectedSides = data?.connected_sides ?? [];
  const readinessLabel = (readiness?: string) => readiness === "ready_to_record"
    ? "ready"
    : readiness === "verified_test_pending"
      ? "test pending"
      : "stage 1 only";
  const homeReady = connectedSides.length > 0 && connectedSides.every((side) => {
    const arm = data?.arms?.[side];
    return !!arm?.connected && !!arm?.home?.captured && !!arm?.torque_enabled;
  });

  useEffect(() => {
    if (!task && info?.last_task) setTask(info.last_task);
  }, [info?.last_task, task]);

  const toggle = async (enabled: boolean) => {
    if (enabled && !effectiveTask) {
      setTaskError("Task description is required before starting an episode.");
      return;
    }
    setBusyRecording(true);
    try {
      await api.vrSetRecording(enabled, effectiveTask);
      await mutate();
    } catch (err) {
      alert(String(err));
    } finally {
      setBusyRecording(false);
    }
  };

  const saveDatasetConfig = async () => {
    if (recordingBusy || savingDatasetConfig) return;
    const nextRoot = root.trim();
    const nextRepoId = repoId.trim();
    if (!nextRoot && !nextRepoId) return;
    setSavingDatasetConfig(true);
    try {
      await api.vrSetRecordingRoot(nextRoot || undefined, nextRepoId || undefined);
      setRoot("");
      setRepoId("");
      await mutate();
    } catch (err) {
      alert(String(err));
    } finally {
      setSavingDatasetConfig(false);
    }
  };

  const goHome = async () => {
    setBusyHome(true);
    try {
      await api.vrHomeGo();
      await mutate();
    } catch (err) {
      alert(String(err));
    } finally {
      setBusyHome(false);
    }
  };

  return (
    <Page title="Dataset Recording" description="Capture bimanual VR demonstrations as LeRobot episodes for PI0.5 fine-tuning.">
      <Card className={active ? "border-blue-500 bg-blue-500/5" : ""}>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h2 className="text-sm font-semibold">Episode Control</h2>
            <div className="mt-2 flex flex-wrap gap-2">
              <Badge tone={active ? "info" : armed ? "warning" : "neutral"}>{active ? "recording" : armed ? "armed" : "idle"}</Badge>
              <Badge tone="neutral">{info?.episodes_saved ?? 0} episodes saved</Badge>
              {!recordingLocked || transitionActive ? <Badge tone={gateBadge.tone}>{gateBadge.label}</Badge> : null}
              {active ? <Badge tone="info">{info?.frames_in_current_episode ?? 0} frames</Badge> : null}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={busyRecording || transitionActive || (!recordingLocked && !startAllowed)}
              onClick={() => toggle(!recordingLocked)}
            >
              <Database size={16} />{active ? "Stop recording" : armed ? "Cancel recording" : "Start recording"}
            </Button>
            <Button variant="secondary" disabled={recordingBusy || busyHome || !homeReady} onClick={goHome}>
              <Home size={16} />Go home
            </Button>
            <Button variant="secondary" disabled={recordingBusy || (info?.episodes_saved ?? 0) < 1} onClick={() => setConfirmDelete(true)}>
              <Trash2 size={16} />Delete last
            </Button>
          </div>
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          <Field label="Task description" hint="Required; stored in LeRobot metadata and used as the language instruction.">
            <TextInput value={task} onChange={(e) => {
              const next = e.currentTarget.value;
              setTask(next);
              setTaskError("");
              if (!recordingBusy) api.vrSetRecordingTask(next.trim()).catch(() => undefined);
            }} disabled={recordingBusy} placeholder={info?.task_default || "Pick the red block and place it in the bin"} />
          </Field>
          <Field label="Dataset repo ID" hint="Blank keeps the configured repo id. Enter user/dataset-name to update dataset.repo_id.">
            <TextInput
              value={repoId}
              onChange={(e) => setRepoId(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  saveDatasetConfig();
                }
              }}
              disabled={recordingBusy || savingDatasetConfig}
              placeholder={info?.repo_id ?? "user/dataset-name"}
            />
          </Field>
          <Field label="Storage root" hint="Blank shows the configured/default root. Save writes dataset.root and dataset.repo_id to config/xlerobot.yaml.">
            <div className="flex gap-2">
              <TextInput
                value={root}
                onChange={(e) => setRoot(e.currentTarget.value)}
                onBlur={saveDatasetConfig}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.currentTarget.blur();
                  }
                }}
                disabled={recordingBusy || savingDatasetConfig}
                placeholder={info?.root ?? ""}
              />
              <Button
                type="button"
                variant="secondary"
                disabled={recordingBusy || savingDatasetConfig || (!root.trim() && !repoId.trim())}
                onMouseDown={(e) => e.preventDefault()}
                onClick={saveDatasetConfig}
              >
                <Check size={16} />Save
              </Button>
            </div>
          </Field>
        </div>
        {taskError ? (
          <p className="mt-3 text-sm text-warning">{taskError}</p>
        ) : null}
        {hardBlockers.length ? (
          <p className="mt-3 text-sm text-warning">Recording blocked: {hardBlockers.join("; ")}</p>
        ) : anchorBlockers.length ? (
          <p className="mt-3 text-sm text-muted-foreground">Start recording is available. The backend will refresh VR anchors from the latest controller poses: {anchorBlockers.join("; ")}</p>
        ) : info?.notice ? (
          <p className="mt-3 text-sm text-muted-foreground">{info.notice}</p>
        ) : null}
        <div className="mt-3 flex flex-wrap gap-2">
          {(["left", "right"] as const).map((side) => {
            const arm = data?.arms?.[side];
            const readiness = arm?.calibration.robot_verification.readiness;
            return (
              <Badge
                key={side}
                tone={readiness === "ready_to_record" ? "success" : readiness === "verified_test_pending" ? "warning" : "neutral"}
              >
                {side}: {arm?.connected ? readinessLabel(readiness) : "offline"}
              </Badge>
            );
          })}
        </div>
      </Card>

      <Card>
        <h2 className="mb-3 text-sm font-semibold">Recording Cameras</h2>
        <CameraPreviewGrid />
      </Card>

      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title="Delete last episode?"
        description={`Episode ${info?.last_episode_index ?? "—"} with ${info?.last_episode_frames ?? 0} frames will be removed.`}
        confirmLabel="Delete episode"
        onConfirm={() => {
          setConfirmDelete(false);
          api.vrDeleteLastRecordingEpisode().then(() => mutate()).catch((err) => alert(String(err)));
        }}
      />
    </Page>
  );
}
