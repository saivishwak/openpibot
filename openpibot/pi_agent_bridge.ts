import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import { Type, getModel } from "@earendil-works/pi-ai";

type BridgeHistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

type BridgePayload = {
  query: string;
  system_prompt: string;
  model: string;
  thinking_level: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
  timeout_ms: number;
  max_steps: number;
  image: {
    mime_type: string;
    data: string;
  };
  history: BridgeHistoryMessage[];
  system1_contract: Record<string, unknown>;
  execution_state?: Record<string, unknown>;
};

type StagedPlan = {
  summary: string;
  steps: Array<{ task: string; reason?: string }>;
  safety_notes: string[];
};

type PlanProgress = {
  message: string;
  next_action: "continue" | "retry" | "done" | "blocked";
  item_statuses: Array<{ index: number; status: string; note: string }>;
  updated_tasks: Array<{ index: number; task: string }>;
};

class HttpError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function readStdin(): Promise<string> {
  const chunks: Uint8Array[] = [];
  for await (const chunk of Bun.stdin.stream()) {
    chunks.push(chunk);
  }
  return new TextDecoder().decode(Buffer.concat(chunks));
}

function parsePayload(input: unknown): BridgePayload {
  const payload = input as BridgePayload;
  if (!payload || typeof payload !== "object") throw new HttpError(400, "JSON body is required");
  if (!payload.query?.trim()) throw new HttpError(400, "query is required");
  if (!payload.image?.data) throw new HttpError(400, "image is required");
  if (!payload.system_prompt?.trim()) throw new HttpError(400, "system_prompt is required");
  validateCustomSystemPrompt(payload.system_prompt);
  return payload;
}

function validateCustomSystemPrompt(systemPrompt: string) {
  const required = [
    "OpenPiBot System 2",
    "get_pi05_system1_contract",
    "get_execution_state",
    "preview_pi05_execution",
    "stage_pi05_execution_plan",
    "track_execution_plan_progress",
    "The CLI can execute only a plan staged through",
  ];
  const missing = required.filter((needle) => !systemPrompt.includes(needle));
  if (missing.length > 0) {
    throw new HttpError(
      400,
      `custom OpenPiBot system prompt is required; missing: ${missing.join(", ")}`,
    );
  }
}

function textResult(details: unknown, terminate = false) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(details) }],
    details,
    terminate,
  };
}

function parseStatusLine(line: string) {
  const [rawIndex, rawStatus, ...noteParts] = line.split("|");
  const index = Number.parseInt((rawIndex ?? "").trim(), 10);
  const status = (rawStatus ?? "").trim().toLowerCase();
  const note = noteParts.join("|").trim();
  return { index, status, note };
}

function parseTaskLine(line: string) {
  const [rawIndex, ...taskParts] = line.split("|");
  const index = Number.parseInt((rawIndex ?? "").trim(), 10);
  const task = taskParts.join("|").trim();
  return { index, task };
}

function buildTools(payload: BridgePayload, staged: { plan?: StagedPlan; progress?: PlanProgress }): AgentTool[] {
  return [
    {
      name: "get_pi05_system1_contract",
      label: "System 1 Contract",
      description:
        "Return the PI0.5 System 1 controller contract, safety constraints, and runtime defaults. Use before planning robot motion.",
      parameters: Type.Object({}),
      execute: async () => textResult(payload.system1_contract),
      executionMode: "sequential",
    },
    {
      name: "get_execution_state",
      label: "Execution State",
      description:
        "Return the active physical execution plan, current item, attempts, last exit code, and known item statuses. Use before tracking progress when execution_state is present.",
      parameters: Type.Object({}),
      execute: async () => textResult(payload.execution_state ?? null),
      executionMode: "sequential",
    },
    {
      name: "preview_pi05_execution",
      label: "Preview PI0.5 Execution",
      description:
        "Preview proposed PI0.5 task prompts. This does not move the robot and exists so the agent can inspect the execution shape before staging.",
      parameters: Type.Object({
        tasks: Type.Array(Type.String({ minLength: 1 }), {
          minItems: 1,
          maxItems: payload.max_steps,
        }),
      }),
      execute: async (_toolCallId, params) => {
        const tasks = params.tasks
          .map((task: string) => task.trim())
          .filter(Boolean)
          .slice(0, payload.max_steps);
        return textResult({
          moves_robot: false,
          requires_user_confirmation: true,
          execution_defaults: payload.system1_contract.execution_defaults,
          previews: tasks.map((task: string) => ({ task })),
        });
      },
      executionMode: "sequential",
    },
    {
      name: "stage_pi05_execution_plan",
      label: "Stage PI0.5 Plan",
      description:
        "Stage the final PI0.5 execution plan for the CLI. This does not move the robot. Use only after checking the contract and previewing tasks.",
      parameters: Type.Object({
        summary: Type.String({ minLength: 1 }),
        tasks: Type.Array(Type.String({ minLength: 1 }), {
          minItems: 1,
          maxItems: payload.max_steps,
        }),
        safety_notes: Type.Array(Type.String(), { default: [] }),
      }),
      execute: async (_toolCallId, params) => {
        const steps = params.tasks
          .map((task: string) => ({ task: task.trim() }))
          .filter((step: { task: string }) => step.task)
          .slice(0, payload.max_steps);
        staged.plan = {
          summary: params.summary.trim(),
          steps,
          safety_notes: params.safety_notes
            .map((note: string) => note.trim())
            .filter(Boolean),
        };
        return textResult({
          staged: steps.length > 0,
          moves_robot: false,
          requires_user_confirmation: true,
          plan: staged.plan,
        });
      },
      executionMode: "sequential",
    },
    {
      name: "track_execution_plan_progress",
      label: "Track Execution Progress",
      description:
        "Update and replan progress for an active multi-step physical plan after looking at the current camera image. Use only when execution_state is present. item_statuses lines must be 'step_number|pending|note', 'step_number|completed|note', or 'step_number|blocked|note'. updated_tasks lines must be 'step_number|new PI0.5 task prompt' for any pending item whose task should change, or the next new step number to append a newly discovered item. next_action must be continue, retry, done, or blocked.",
      parameters: Type.Object({
        item_statuses: Type.Array(Type.String({ minLength: 1 }), {
          minItems: 1,
          maxItems: payload.max_steps,
        }),
        updated_tasks: Type.Array(Type.String(), { default: [] }),
        next_action: Type.String({ minLength: 1 }),
        message: Type.String({ minLength: 1 }),
      }),
      execute: async (_toolCallId, params) => {
        const validStatuses = new Set(["pending", "completed", "blocked"]);
        const validActions = new Set(["continue", "retry", "done", "blocked"]);
        const item_statuses = params.item_statuses
          .map((line: string) => parseStatusLine(line))
          .filter((item: { index: number; status: string }) =>
            Number.isFinite(item.index) && item.index >= 1 && validStatuses.has(item.status),
          )
          .slice(0, payload.max_steps);
        const next_action = String(params.next_action || "").trim().toLowerCase();
        const updated_tasks = (params.updated_tasks ?? [])
          .map((line: string) => parseTaskLine(line))
          .filter((item: { index: number; task: string }) =>
            Number.isFinite(item.index) && item.index >= 1 && item.task.length > 0,
          )
          .slice(0, payload.max_steps);
        staged.progress = {
          message: String(params.message || "").trim(),
          next_action: validActions.has(next_action)
            ? (next_action as PlanProgress["next_action"])
            : "continue",
          item_statuses,
          updated_tasks,
        };
        return textResult({
          tracked: item_statuses.length > 0,
          progress: staged.progress,
          execution_state: payload.execution_state ?? null,
        });
      },
      executionMode: "sequential",
    },
  ];
}

function getOpenAIModel(modelId: string) {
  return getModel("openai" as any, modelId as any);
}

function extractAssistantText(agent: Agent): string {
  for (let i = agent.state.messages.length - 1; i >= 0; i -= 1) {
    const message = agent.state.messages[i] as any;
    if (message.role !== "assistant") continue;
    if (typeof message.content === "string") return message.content.trim();
    if (!Array.isArray(message.content)) continue;
    return message.content
      .filter((block: any) => block.type === "text" && typeof block.text === "string")
      .map((block: any) => block.text)
      .join("")
      .trim();
  }
  return "";
}

function buildInitialMessages(history: BridgeHistoryMessage[]) {
  return history
    .filter((message) => message.content?.trim())
    .map((message) => ({
      role: message.role,
      content: [{ type: "text" as const, text: message.content }],
      timestamp: Date.now(),
    }));
}

async function runTurnPayload(payload: BridgePayload) {
  const staged: { plan?: StagedPlan; progress?: PlanProgress } = {};
  const agent = new Agent({
    initialState: {
      systemPrompt: payload.system_prompt,
      model: getOpenAIModel(payload.model),
      thinkingLevel: payload.thinking_level,
      tools: buildTools(payload, staged),
      messages: buildInitialMessages(payload.history),
    },
    getApiKey: (provider: string) => {
      if (provider === "openai") return process.env.OPENAI_API_KEY;
      return undefined;
    },
    toolExecution: "sequential",
  });

  await agent.prompt(payload.query, [
    {
      type: "image",
      data: payload.image.data,
      mimeType: payload.image.mime_type,
    },
  ]);

  const message = extractAssistantText(agent);
  if (staged.progress && staged.progress.item_statuses.length > 0) {
    return {
      mode: "progress",
      message: message || staged.progress.message,
      progress: staged.progress,
    };
  }
  if (staged.plan && staged.plan.steps.length > 0) {
    return {
      mode: "plan",
      message: message || "I staged a PI0.5 plan for operator review.",
      plan: staged.plan,
    };
  }
  return {
    mode: "chat",
    message: message || "I am Tess. What would you like me to do?",
  };
}

function jsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);
  if (request.method === "GET" && url.pathname === "/health") {
    return jsonResponse(200, { ok: true, service: "openpibot-pi-agent" });
  }
  if (request.method !== "POST" || url.pathname !== "/turn") {
    return jsonResponse(404, { error: "not found" });
  }
  try {
    const payload = parsePayload(await request.json());
    return jsonResponse(200, await runTurnPayload(payload));
  } catch (error) {
    if (error instanceof HttpError) {
      return jsonResponse(error.status, { error: error.message });
    }
    const message = error instanceof Error ? error.message : String(error);
    return jsonResponse(500, { error: message });
  }
}

function argValue(name: string, fallback: string) {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  return process.argv[index + 1] ?? fallback;
}

async function main() {
  if (process.argv.includes("--serve")) {
    const hostname = argValue("--host", "127.0.0.1");
    const port = Number.parseInt(argValue("--port", "8765"), 10);
    Bun.serve({
      hostname,
      port,
      fetch: handleRequest,
    });
    console.error(`OpenPiBot Pi agent listening on http://${hostname}:${port}`);
    return;
  }
  const payload = parsePayload(JSON.parse(await readStdin()));
  process.stdout.write(JSON.stringify(await runTurnPayload(payload)));
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  console.error(message);
  process.exit(1);
});
