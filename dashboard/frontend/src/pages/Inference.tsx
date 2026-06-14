import { Play, Server } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";
import { Job, api, fetcher } from "../api";
import { Badge, Button, Card, Field, Page, SelectField, TextInput } from "../components/ui";

type InferenceModel = "pi05" | "molmoact2";
type MolmoSource = "finetuned" | "base";

const modelOptions = [
  { value: "pi05", label: "PI0.5" },
  { value: "molmoact2", label: "MolmoAct2" },
];

const molmoSourceOptions = [
  { value: "finetuned", label: "Finetuned checkpoint" },
  { value: "base", label: "Base checkpoint" },
];

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
  const [model, setModel] = useState<InferenceModel>("pi05");
  const [molmoSource, setMolmoSource] = useState<MolmoSource>("finetuned");
  const [policyPath, setPolicyPath] = useState("outputs/pi05_finetune/checkpoints/last/pretrained_model");
  const [molmoCheckpointPath, setMolmoCheckpointPath] = useState("allenai/MolmoAct2");
  const [datasetRepoId, setDatasetRepoId] = useState("");
  const [task, setTask] = useState("");

  const changeModel = (next: string) => {
    const selected = next as InferenceModel;
    setModel(selected);
    setPolicyPath(
      selected === "molmoact2"
        ? "outputs/molmoact2_finetune/checkpoints/last/pretrained_model"
        : "outputs/pi05_finetune/checkpoints/last/pretrained_model",
    );
  };

  const changeMolmoSource = (next: string) => {
    setMolmoSource(next as MolmoSource);
  };

  const startInference = async () => {
    let args: string[];
    if (model === "molmoact2" && molmoSource === "base") {
      const checkpoint = molmoCheckpointPath.trim();
      if (!checkpoint) {
        alert("MolmoAct2 base checkpoint is required.");
        return;
      }
      args = [
        `--checkpoint-path=${checkpoint}`,
        datasetRepoId.trim() ? `--dataset-repo-id=${datasetRepoId.trim()}` : "",
        task.trim() ? `--task=${task.trim()}` : "",
      ].filter(Boolean);
    } else {
      args = [
        policyPath.trim() ? `--policy-path=${policyPath.trim()}` : "",
        task.trim() ? `--task=${task.trim()}` : "",
      ].filter(Boolean);
    }
    try {
      if (model === "molmoact2") {
        await api.startMolmoAct2Inference(args);
      } else {
        await api.startInference(args);
      }
      await mutate();
    } catch (err) {
      alert(String(err));
    }
  };

  const visibleJobs = (jobs?.jobs ?? []).filter((j) =>
    j.command.some((p) => p.includes("infer_pi05") || p.includes("infer_molmoact2") || p.includes("openpi_policy_server")),
  ).reverse();

  return (
    <Page title="Inference" description="Run package-managed PI0.5 serving, finetuned local inference, or MolmoAct2 base-model baseline inference.">
      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <h2 className="text-sm font-semibold">OpenPiBot policy server</h2>
          <p className="mt-1 text-sm text-muted-foreground">Starts OpenPI from the package-managed runtime. No vendored third-party checkout is used.</p>
          <Button className="mt-4" onClick={() => api.startPi05Server().then(() => mutate()).catch((err) => alert(String(err)))}>
            <Server size={16} />Start PI0.5 server
          </Button>
        </Card>
        <Card>
          <h2 className="text-sm font-semibold">Local VLA inference</h2>
          <div className="mt-3 grid gap-3">
            <Field label="Model">
              <SelectField value={model} onValueChange={changeModel} options={modelOptions} ariaLabel="Inference model" />
            </Field>
            {model === "molmoact2" ? (
              <Field label="MolmoAct2 source">
                <SelectField value={molmoSource} onValueChange={changeMolmoSource} options={molmoSourceOptions} ariaLabel="MolmoAct2 source" />
              </Field>
            ) : null}
            {model === "molmoact2" && molmoSource === "base" ? (
              <>
                <Field label="Checkpoint path">
                  <TextInput value={molmoCheckpointPath} onChange={(e) => setMolmoCheckpointPath(e.currentTarget.value)} />
                </Field>
                <Field label="Dataset repo id">
                  <TextInput value={datasetRepoId} onChange={(e) => setDatasetRepoId(e.currentTarget.value)} placeholder="config default" />
                </Field>
              </>
            ) : (
              <Field label="Policy path">
                <TextInput value={policyPath} onChange={(e) => setPolicyPath(e.currentTarget.value)} />
              </Field>
            )}
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
