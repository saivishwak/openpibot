import { Camera as CameraIcon } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";
import { CameraSpec, api, fetcher } from "../api";
import { Badge, Card, Page, SelectField } from "../components/ui";

function CameraTile({ cam, roles, refresh }: { cam: CameraSpec; roles: string[]; refresh: () => Promise<unknown> }) {
  const [busy, setBusy] = useState(false);
  const [errored, setErrored] = useState(false);
  const assign = async (role: string) => {
    if (!cam.by_path) return;
    setBusy(true);
    try {
      await api.assign(cam.by_path, role === "__none__" ? null : role);
      await refresh();
    } catch (err) {
      alert(String(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card>
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <CameraIcon size={16} />
            <h2 className="truncate text-sm font-semibold">{cam.name}</h2>
            {cam.role ? <Badge tone="info">{cam.role}</Badge> : null}
          </div>
          <p className="mt-1 break-all text-xs text-muted-foreground">{cam.card || "unknown card"}</p>
          <p className="mono mt-1 break-all text-xs text-muted-foreground">{cam.by_path || cam.path}</p>
        </div>
      </div>
      <div className="cam-tile">
        {!errored ? (
          <img src={`/camera/${encodeURIComponent(cam.name)}/stream?t=${encodeURIComponent(cam.path)}`} alt={cam.name} onError={() => setErrored(true)} />
        ) : (
          <div className="grid h-full place-items-center p-4 text-center text-xs text-muted-foreground">stream unavailable</div>
        )}
      </div>
      <div className="mt-3">
        <SelectField
          value={cam.role || "__none__"}
          disabled={!cam.by_path || busy}
          onValueChange={assign}
          options={[{ value: "__none__", label: "(none)" }, ...roles.map((role) => ({ value: role, label: role }))]}
        />
      </div>
    </Card>
  );
}

export function Cameras() {
  const { data, error, mutate } = useSWR<{ cameras: CameraSpec[]; roles: string[] }>("/api/cameras", fetcher);
  return (
    <Page title="Cameras" description="Assign USB camera devices to OpenPIBot roles. Assignments are saved to YAML and reused for recording and inference.">
      {error ? <Card className="border-danger text-danger">{String(error)}</Card> : null}
      <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
        {(data?.cameras ?? []).map((cam) => (
          <CameraTile key={`${cam.name}-${cam.path}`} cam={cam} roles={data?.roles ?? []} refresh={() => mutate()} />
        ))}
      </div>
      {data && data.cameras.length === 0 ? <p className="text-sm text-muted-foreground">No capture-capable cameras detected.</p> : null}
    </Page>
  );
}

