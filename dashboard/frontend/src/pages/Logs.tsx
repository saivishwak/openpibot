import { RefreshCw } from "lucide-react";
import useSWR from "swr";
import { api, fetcher } from "../api";
import { Button, Card, Page } from "../components/ui";

export function Logs() {
  const { data, mutate, isLoading } = useSWR<{ path: string; lines: string[] }>("/api/logs/recent?lines=300", fetcher, { refreshInterval: 3000 });
  return (
    <Page
      title="Logs"
      description="Recent backend logs with request IDs and hardware/debug messages."
      action={<Button variant="secondary" disabled={isLoading} onClick={() => mutate()}><RefreshCw size={16} />Refresh</Button>}
    >
      <Card>
        <div className="mb-3 text-xs text-muted-foreground">{data?.path ?? "server log"}</div>
        <pre className="mono max-h-[70vh] overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted p-4 text-xs">
          {(data?.lines ?? []).join("\n") || "No log lines yet."}
        </pre>
      </Card>
    </Page>
  );
}

