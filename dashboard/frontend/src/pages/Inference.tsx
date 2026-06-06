import { Play, Server } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";
import { Job, api, fetcher } from "../api";
import { Badge, Button, Card, Field, Page, TextInput } from "../components/ui";

function JobRows({ jobs }: { jobs: Job[] }) {
  return (
    <div className="grid gap-3">
      {jobs.map((job) => (
        <Card key={job.id}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold">{job.id}</span>
                <Badge tone={job.status === "running" ? "info" : job.status === "succeeded" ? "success" : job.status === "failed" ? "danger" : "neutral"}>{job.status}</Badge>
              </div>
              <pre className="mono mt-2 overflow-hidden text-ellipsis whitespace-nowrap text-xs text-muted-foreground">{job.command.join(" ")}</pre>
            </div>
          </div>
          <pre className="mono mt-3 max-h-44 overflow-auto rounded-md bg-muted p-3 text-xs">{job.log.slice(-20).join("\n") || "No logs yet."}</pre>
        </Card>
      ))}
    </div>
  );
}

export function Inference() {
  const { data: jobs, mutate } = useSWR<{ jobs: Job[] }>("/api/jobs", fetcher, { refreshInterval: 1500 });
  const [policyPath, setPolicyPath] = useState("outputs/pi05_finetune/checkpoints/last/pretrained_model");
  const [task, setTask] = useState("");

  const startInference = async () => {
    const args = [
      policyPath.trim() ? `--policy-path=${policyPath.trim()}` : "",
      task.trim() ? `--task=${task.trim()}` : "",
    ].filter(Boolean);
    try {
      await api.startInference(args);
      await mutate();
    } catch (err) {
      alert(String(err));
    }
  };

  const visibleJobs = (jobs?.jobs ?? []).filter((j) =>
    j.command.some((p) => p.includes("infer_pi05") || p.includes("openpi_policy_server")),
  ).reverse();

  return (
    <Page title="Inference" description="Run the package-managed PI0.5 WebSocket server or local finetuned checkpoint inference from OpenPiBot.">
      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <h2 className="text-sm font-semibold">OpenPiBot policy server</h2>
          <p className="mt-1 text-sm text-muted-foreground">Starts OpenPI from the package-managed runtime. No vendored third-party checkout is used.</p>
          <Button className="mt-4" onClick={() => api.startPi05Server().then(() => mutate()).catch((err) => alert(String(err)))}>
            <Server size={16} />Start PI0.5 server
          </Button>
        </Card>
        <Card>
          <h2 className="text-sm font-semibold">Local finetuned inference</h2>
          <div className="mt-3 grid gap-3">
            <Field label="Policy path">
              <TextInput value={policyPath} onChange={(e) => setPolicyPath(e.currentTarget.value)} />
            </Field>
            <Field label="Task override">
              <TextInput value={task} onChange={(e) => setTask(e.currentTarget.value)} placeholder="optional" />
            </Field>
          </div>
          <Button className="mt-4" onClick={startInference}><Play size={16} />Start inference</Button>
        </Card>
      </div>
      <JobRows jobs={visibleJobs} />
    </Page>
  );
}
