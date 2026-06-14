import { Database, Play } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";
import { Job, api, fetcher } from "../api";
import { Badge, Button, Card, Field, Page, SelectField, TextInput } from "../components/ui";

type TrainingModel = "pi05" | "molmoact2";

const modelOptions = [
  { value: "pi05", label: "PI0.5" },
  { value: "molmoact2", label: "MolmoAct2" },
];

function Jobs({ jobs }: { jobs: Job[] }) {
  return (
    <div className="grid gap-3">
      {jobs.map((job) => (
        <Card key={job.id}>
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold">{job.id}</h2>
                <Badge tone={job.status === "succeeded" ? "success" : job.status === "failed" ? "danger" : job.status === "running" ? "info" : "neutral"}>{job.status}</Badge>
              </div>
              <pre className="mono mt-2 overflow-hidden text-ellipsis whitespace-nowrap text-xs text-muted-foreground">{job.command.join(" ")}</pre>
              <pre className="mono mt-3 max-h-44 overflow-auto rounded-md bg-muted p-3 text-xs">{job.log.slice(-20).join("\n") || "No logs yet."}</pre>
            </div>
          </div>
        </Card>
      ))}
      {!jobs.length ? <p className="text-sm text-muted-foreground">No jobs yet.</p> : null}
    </div>
  );
}

export function Training() {
  const { data: jobs, mutate } = useSWR<{ jobs: Job[] }>("/api/jobs", fetcher, { refreshInterval: 1500 });
  const [model, setModel] = useState<TrainingModel>("pi05");
  const [datasetRepo, setDatasetRepo] = useState("");
  const [outputDir, setOutputDir] = useState("outputs/pi05_finetune");
  const [steps, setSteps] = useState("20000");
  const [busy, setBusy] = useState(false);

  const changeModel = (next: string) => {
    const selected = next as TrainingModel;
    setModel(selected);
    setOutputDir(selected === "molmoact2" ? "outputs/molmoact2_finetune" : "outputs/pi05_finetune");
  };

  const start = async () => {
    const args = [
      datasetRepo.trim() ? `--dataset-repo-id=${datasetRepo.trim()}` : "",
      outputDir.trim() ? `--output-dir=${outputDir.trim()}` : "",
      steps.trim() ? `--steps=${steps.trim()}` : "",
    ].filter(Boolean);
    setBusy(true);
    try {
      if (model === "molmoact2") {
        await api.startMolmoAct2Training(args);
      } else {
        await api.startTraining(args);
      }
      await mutate();
    } catch (err) {
      alert(String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Page title="AI Training" description="Start fine-tuning jobs against the configured LeRobot dataset and follow command output from the dashboard.">
      <Card>
        <div className="grid gap-4 lg:grid-cols-3">
          <Field label="Model">
            <SelectField value={model} onValueChange={changeModel} options={modelOptions} ariaLabel="Training model" />
          </Field>
          <Field label="Dataset repo id">
            <TextInput value={datasetRepo} onChange={(e) => setDatasetRepo(e.currentTarget.value)} placeholder="from config when blank" />
          </Field>
          <Field label="Output directory">
            <TextInput value={outputDir} onChange={(e) => setOutputDir(e.currentTarget.value)} />
          </Field>
          <Field label="Steps">
            <TextInput value={steps} onChange={(e) => setSteps(e.currentTarget.value)} />
          </Field>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <Button onClick={start} disabled={busy}><Play size={16} />Start training</Button>
          <Button variant="secondary" onClick={() => api.pushDataset([]).then(() => mutate()).catch((err) => alert(String(err)))}><Database size={16} />Push dataset</Button>
        </div>
      </Card>
      <Jobs jobs={(jobs?.jobs ?? []).filter((j) => j.command.some((p) => p.includes("finetune") || p.includes("push_dataset"))).reverse()} />
    </Page>
  );
}
