import { Check, Database, Home, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import useSWR from "swr";
import { VRStatus, api, fetcher } from "../api";
import { CameraPreviewGrid } from "../components/CameraPreviewGrid";
import { Badge, Button, Card, ConfirmDialog, Field, Page, TextInput } from "../components/ui";

export function Recording() {
  const { data, mutate } = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1000 });
  const info = data?.recording_info;
  const [task, setTask] = useState("");
  const [root, setRoot] = useState("");
  const [repoId, setRepoId] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [busyHome, setBusyHome] = useState(false);
  const [savingDatasetConfig, setSavingDatasetConfig] = useState(false);
  const active = !!data?.recording;
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
    try {
      await api.vrSetRecording(enabled, task.trim());
      await mutate();
    } catch (err) {
      alert(String(err));
    }
  };

  const saveDatasetConfig = async () => {
    if (active || savingDatasetConfig) return;
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
              <Badge tone={active ? "info" : "neutral"}>{active ? "recording" : "idle"}</Badge>
              <Badge tone="neutral">{info?.episodes_saved ?? 0} episodes saved</Badge>
              <Badge tone={info?.calibration_ready ? "success" : "warning"}>{info?.calibration_ready ? "calibration ready" : "verification required"}</Badge>
              {active ? <Badge tone="info">{info?.frames_in_current_episode ?? 0} frames</Badge> : null}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button disabled={!active && !task.trim()} onClick={() => toggle(!active)}>
              <Database size={16} />{active ? "Stop recording" : "Start recording"}
            </Button>
            <Button variant="secondary" disabled={active || busyHome || !homeReady} onClick={goHome}>
              <Home size={16} />Go home
            </Button>
            <Button variant="secondary" disabled={active || (info?.episodes_saved ?? 0) < 1} onClick={() => setConfirmDelete(true)}>
              <Trash2 size={16} />Delete last
            </Button>
          </div>
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          <Field label="Task description" hint="Required; stored in LeRobot metadata and used as the language instruction.">
            <TextInput value={task} onChange={(e) => {
              const next = e.currentTarget.value;
              setTask(next);
              if (!active) api.vrSetRecordingTask(next.trim()).catch(() => undefined);
            }} disabled={active} placeholder="Pick the red block and place it in the bin" />
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
              disabled={active || savingDatasetConfig}
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
                disabled={active || savingDatasetConfig}
                placeholder={info?.root ?? ""}
              />
              <Button
                type="button"
                variant="secondary"
                disabled={active || savingDatasetConfig || (!root.trim() && !repoId.trim())}
                onMouseDown={(e) => e.preventDefault()}
                onClick={saveDatasetConfig}
              >
                <Check size={16} />Save
              </Button>
            </div>
          </Field>
        </div>
        {(info?.calibration_blockers ?? []).length ? (
          <p className="mt-3 text-sm text-warning">Recording blocked: {info?.calibration_blockers.join("; ")}</p>
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
