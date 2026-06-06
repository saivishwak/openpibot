# OpenPIBot

OpenPIBot is a dashboard and CLI for bimanual SO-101/XLeRobot workflows: native Quest/OpenXR VR teleoperation, LeRobot dataset recording, PI0.5 fine-tuning, and policy inference.

It combines a FastAPI backend, a React dashboard, hardware-aware robot runtime code, and focused scripts for training, dataset upload/visualization, and inference.

## Highlights

- **Native Quest/OpenXR VR teleoperation** with Meta Quest controllers, per-arm engagement, calibration profiles, and safety limits.
- **Dataset recording** in LeRobot format with synchronized present joint state, same-tick commanded actions, and camera observations.
- **PI0.5 training and inference** through local finetuned checkpoints using the same config, robot loader, camera roles, joint order, and VR-style command shaping as recording.
- **Robot calibration tools** for home pose capture, VR frame calibration, and robot-verified calibration before dataset-quality recording.
- **Production-oriented backend** with structured API routes, job logs, diagnostics, and a single OpenPIBot server process.
- **Clean dashboard** for control, calibration, recording, training, inference, cameras, diagnostics, and logs.

## Prerequisites

- Linux recommended for robot control. USB motor access requires membership in the `dialout` group.
- Python `>=3.12,<3.13` and [`uv`](https://docs.astral.sh/uv/) for the backend, CLI, and training tools.
- Node.js `>=20` and `pnpm` for building the dashboard.
- Initialized `vendor/lerobot` submodule; it is used as the editable LeRobot workspace for XLeRobot support.
- Calibrated SO-101/XLeRobot arms, USB cameras, and a Meta Quest headset for VR teleop.
- A Hugging Face account/token if you plan to push datasets or model checkpoints.

## Quick Start

```bash
git submodule update --init vendor/lerobot
uv sync
uv run openpibot run
```

`openpibot run` builds `dashboard/frontend` with `pnpm` before starting the backend. Use `--no-build-dashboard` if you are running a separate frontend dev server.

Open the dashboard at `http://localhost:5000`.

Typical first run:

1. Open `config/xlerobot.yaml` and set the serial ports, dataset repo, and robot profile.
2. Build the dashboard and start the backend with the commands above.
3. Assign `head`, `left_wrist`, and `right_wrist` camera roles from the Cameras page.
4. Connect each arm from the Control page.
5. Capture a home pose from Calibration.
6. Run VR calibration, robot verification, and the low-scale test before recording finetuning data.
7. Open the native Quest app and start teleop/recording.

## Common Commands

| Command | Purpose |
|---|---|
| `uv run openpibot run --host 0.0.0.0` | Build the dashboard and run the OpenPIBot backend on port 5000 |
| `uv run openpibot run --reload --no-build-dashboard` | Run the backend with reload on port 5000 |
| `pnpm --dir dashboard/frontend dev` | Run the Vite frontend dev server |
| `pnpm --dir dashboard/frontend build` | Build the dashboard manually |
| `uv run openpibot info` | Print project and robot configuration |
| `uv run openpibot doctor` | Run diagnostics |
| `uv run python scripts/finetune_pi05.py` | Fine-tune PI0.5 on the configured dataset |
| `uv run python scripts/infer_pi05_finetuned.py ...` | Run local finetuned inference |
| `bash scripts/run_openpi_server.sh` | Start the optional package-managed OpenPI policy server |
| `uv run python scripts/push_dataset.py` | Push a recorded dataset |

## Dashboard

| Page | Purpose |
|---|---|
| Dashboard | Robot, VR, recording, and diagnostics overview |
| Control | VR teleop, engagement, Quest URL, 3D view, and camera preview |
| Calibration | Robot/profile dropdowns, home pose, VR calibration, and robot verification |
| Recording | Task prompt, dataset path, and episode controls |
| Training | PI0.5 fine-tuning jobs |
| Inference | Package-managed OpenPI server and local checkpoint inference jobs |
| Cameras | Live previews and role assignment |
| Diagnostics | USB, serial, camera, import, and RealSense checks |
| Logs | Backend logs with request IDs |

## Project Layout

```text
.
├── openpibot/             FastAPI backend, CLI, runtime services
├── dashboard/             React/Tailwind dashboard source and build output
├── scripts/               Training, dataset, push, and inference utilities
├── config/                Robot profiles, camera config, calibration, home pose
├── docs/                  Architecture, setup, teleop, calibration, recording, training, inference
├── reference/             Local reference docs/assets used by the app
└── vendor/                LeRobot submodule plus local XLeRobot URDF assets
```

## Documentation

| Document | Covers |
|---|---|
| [docs/setup.md](docs/setup.md) | Installation, hardware configuration, motor calibration |
| [docs/architecture.md](docs/architecture.md) | Current Quest/OpenXR -> recording -> finetuning -> inference architecture |
| [docs/teleop.md](docs/teleop.md) | VR teleop workflow and controls |
| [docs/calibration.md](docs/calibration.md) | Home pose, VR calibration, robot-verified calibration |
| [docs/recording.md](docs/recording.md) | LeRobot dataset recording and upload |
| [docs/training.md](docs/training.md) | PI0.5 fine-tuning and evaluation |
| [docs/inference.md](docs/inference.md) | Local inference and optional OpenPI WebSocket server |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common hardware and runtime issues |

## Safety

OpenPIBot is designed for supervised robot operation.

- Connecting an arm enables holding torque only; it does not home or move automatically.
- The Control page includes an emergency stop that disables torque immediately.
- Motion commands are gated by controller engagement, per-tick joint caps, and stale-goal watchdogs.
- The only dashboard action that initiates autonomous motion is the user-triggered **Go to Home** command.

## Notes

- The main backend is `openpibot.server`; `dashboard/` is frontend source and build output only.
- Robot-side PI0.5 WebSocket client execution uses the locked `openpi-client` package.
- OpenPI policy-server dependencies are isolated from the robot/dashboard environment with `uv --no-project --with "$OPENPI_PACKAGE"`.
