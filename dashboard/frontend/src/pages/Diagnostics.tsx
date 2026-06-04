import { RefreshCw } from "lucide-react";
import useSWR from "swr";
import { CheckStatus, DoctorCheck, api, fetcher } from "../api";
import { Badge, Button, Card, Page } from "../components/ui";

const tone: Record<CheckStatus, "neutral" | "success" | "warning" | "danger" | "info"> = {
  ok: "success",
  warn: "warning",
  fail: "danger",
  info: "info",
};

export function Diagnostics() {
  const { data, error, isLoading, mutate } = useSWR<{ checks: DoctorCheck[] }>("/api/doctor", fetcher);
  return (
    <Page
      title="Diagnostics"
      description="USB devices, camera nodes, serial access, configured paths, Python imports, and RealSense availability."
      action={<Button variant="secondary" onClick={() => mutate()} disabled={isLoading}><RefreshCw size={16} />Run checks</Button>}
    >
      {error ? <Card className="border-danger text-danger">{String(error)}</Card> : null}
      <div className="grid gap-3 lg:grid-cols-2">
        {(data?.checks ?? []).map((check, index) => (
          <Card key={`${check.name}-${index}`}>
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <h2 className="text-sm font-semibold">{check.name}</h2>
                <pre className="mono mt-2 whitespace-pre-wrap break-words text-xs text-muted-foreground">{check.detail}</pre>
              </div>
              <Badge tone={tone[check.status]}>{check.status}</Badge>
            </div>
          </Card>
        ))}
      </div>
      {isLoading && !data ? <p className="text-sm text-muted-foreground">Loading diagnostics...</p> : null}
    </Page>
  );
}

