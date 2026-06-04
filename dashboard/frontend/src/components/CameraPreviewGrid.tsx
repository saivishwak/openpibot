import useSWR from "swr";
import { CameraSpec, fetcher } from "../api";

const DEFAULT_ROLES = ["head", "left_wrist", "right_wrist"];

interface CamerasResponse {
  cameras: CameraSpec[];
  roles: string[];
}

function CameraPreviewTile({ role, cam }: { role: string; cam?: CameraSpec }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-2 text-xs font-medium text-muted-foreground">
        <span>{role}</span>
        {cam?.path ? <span className="mono max-w-[70%] truncate">{cam.path}</span> : null}
      </div>
      <div className="cam-tile">
        {cam ? (
          <img src={`/camera/${encodeURIComponent(cam.name)}/stream`} alt={role} />
        ) : (
          <div className="grid h-full place-items-center p-4 text-center text-xs text-muted-foreground">
            role not configured
          </div>
        )}
      </div>
    </div>
  );
}

export function CameraPreviewGrid({ roles = DEFAULT_ROLES }: { roles?: string[] }) {
  const { data, error } = useSWR<CamerasResponse>("/api/cameras", fetcher, { refreshInterval: 5000 });
  const cameras = data?.cameras ?? [];

  if (error) {
    return <p className="text-sm text-danger">{String(error)}</p>;
  }

  return (
    <div className="grid gap-3 md:grid-cols-3">
      {roles.map((role) => {
        const cam = cameras.find((item) => item.name === role || item.role === role);
        return <CameraPreviewTile key={role} role={role} cam={cam} />;
      })}
    </div>
  );
}
