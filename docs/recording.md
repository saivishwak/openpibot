# Dataset Recording

Captures bimanual SO-101 teleop demonstrations to a LeRobot v2 dataset. Use the recorded data to fine-tune pi0.5 or another VLA.

## What gets recorded

Each frame contains:
- `action` — absolute commanded joint positions for both arms after the same VR
  IK, per-joint caps, deadbands, and command filtering used by live teleop
  (12-vector: 6 joints × 2 arms)
- `observation.state` — present joint positions read from the robot at that
  tick (same 12-vector)
- `observation.images.<role>` — video stream per camera with a role assigned
  (`head`, `left_wrist`, `right_wrist`)

At 30 Hz, with 3 cameras at 640×480 RGB.

## Configure

In `config/xlerobot.yaml`:

```yaml
dataset:
  repo_id: <hf-user>/<dataset-name>   # e.g. saivishwak/xlerobot-vr-teleop
  root: /optional/local/dataset/root   # optional; omit/null for HF cache default
  fps: 30
  push_to_hub: false                  # set true to push to HF after finalizing
  home_before_episode: true           # auto-home all arms at start of each episode
```

Set camera roles on the **Cameras page** (not the Teleop page). Each camera needs a role for it to be included as an `observation.images.*` feature.

The Recording page's **Dataset repo ID** and **Storage root** fields are editable
while recording is idle. Blank fields keep the configured values and show the
current values as placeholders. Entering a repo id and/or root path and saving
writes `dataset.repo_id` and/or `dataset.root` into `config/xlerobot.yaml`;
recording then uses those persisted values. Config changes are rejected while
recording is active or armed.

## Recording flow

1. Select or create the correct **Calibration Profile** for the current user/setup.
2. Connect both arms, run stage 1 VR calibration, then run robot verification for every connected arm (see [calibration.md](calibration.md)).
3. Run **Start low-scale test** on each verified arm and confirm the robot moves forward/left/up the same way your controller moves. Stop the test before recording.
4. Capture home pose (once, see [calibration.md](calibration.md)).
5. Set a task description. Either via API or the dataset config's `task_default`.
6. **Press B** on the right controller — or click *Start recording* in the UI.
   - If `home_before_episode: true`, all arms slowly move to home first.
   - If robot verification is missing/poor, or the low-scale test has not been completed for a connected arm, recording is blocked.
   - Then a new episode opens.
7. Squeeze grip + perform the demonstration.
8. **Press B again** — episode is saved to disk.
9. Repeat for as many episodes as you want.

Frames are recorded **every drive-loop tick** (30 Hz) while recording is active, regardless of whether you're actively teleoperating that tick. Passive arms still contribute their `observation.state`.

For quick manual testing, VR-only calibration can still drive the robot. For VLA training data, use robot-verified calibration so the recorded joint actions match the intended end-effector motion.

## Where it lives

Episodes are written to `dataset.root` when set, otherwise to
`$HF_LEROBOT_HOME/<repo_id>/` (default
`~/.cache/huggingface/lerobot/<repo_id>/`).

Root resolution order:

1. `dataset.root` in `config/xlerobot.yaml`
2. `$HF_LEROBOT_HOME/<repo_id>`
3. `~/.cache/huggingface/lerobot/<repo_id>`

If `push_to_hub: true`, the dashboard recorder pushes to the Hub when recording finalizes (e.g. on emergency stop). With the default `push_to_hub: false`, upload manually after recording (see below).

## Push to Hugging Face Hub

Use `scripts/push_dataset.py` to upload a finished local dataset. It reads `dataset.repo_id` (and optional `dataset.root`) from `config/xlerobot.yaml` by default.

**Prerequisites**

- Recording is finished and flushed (episodes on disk under the local root).
- The folder is a valid LeRobot dataset (`meta/info.json` must exist).
- Hugging Face CLI login with write access to the target dataset repo:

```bash
hf auth login
```

**Default upload** (uses `repo_id` from `config/xlerobot.yaml`, e.g. `saivishwak/xlerobot-vr-teleop-medicine-bowl`):

```bash
cd /path/to/xlerobot

# Show resolved repo id and local path without uploading
uv run python scripts/push_dataset.py --dry-run

# Upload
uv run python scripts/push_dataset.py
```

Local root resolution (when `--root` is omitted):

1. `dataset.root` in `config/xlerobot.yaml`, if set
2. Else `$HF_LEROBOT_HOME/<repo_id>`
3. Else `~/.cache/huggingface/lerobot/<repo_id>`

**Explicit repo or path**

```bash
uv run python scripts/push_dataset.py \
  --repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl

uv run python scripts/push_dataset.py \
  --repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --root /custom/path/to/dataset
```

On success you should see:

```text
Uploaded dataset to https://huggingface.co/datasets/<repo_id>
```

**Large datasets (~500 MiB+, many videos)** — The script auto-selects `upload_large_folder` (resumable, parallel). For a ~3 GiB dataset this avoids `upload_folder` stalling around 50–70%.

```bash
uv run python scripts/push_dataset.py \
  --repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --root ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --large \
  --num-workers 4
```

If an upload **hangs or stalls** (e.g. stuck at 62%):

1. **Ctrl+C** the stuck process.
2. Re-run the **same** command — large-folder mode skips files already on the Hub.
3. Use `--dry-run` first to confirm `Upload mode: upload_large_folder (resumable)`.

| Flag | Default | Description |
|------|---------|-------------|
| `--large` / `--no-large` | auto (>500 MiB) | Resumable `upload_large_folder` vs plain `upload_folder` |
| `--num-workers` | `4` | Parallel uploads (large mode only) |
| `--private` | off | Create private Hub dataset |
| `--no-videos` | off | Skip `videos/` (parquet only; faster test) |
| `--dry-run` | off | Print paths, size, upload mode; no upload |

**Task labels** — Before training, confirm episode prompts in `meta/tasks.parquet` and per-episode `tasks` in `meta/episodes/` match what you intend (e.g. one consistent string per episode for medicine→bowl demos).

## Stopping cleanly

- **Toggle off recording (B button or UI)** to save the current episode.
- **Emergency Stop** also flushes the in-flight episode and finalizes the dataset.
- Disconnecting an arm does *not* save — recording can continue with the remaining arm(s) (their joints contribute, the disconnected arm's joints come through as zeros).

### View

Quick sanity check before viewing

```bash
# Set REPO to match dataset.repo_id in config/xlerobot.yaml
REPO=saivishwak/xlerobot-vr-teleop-medicine-bowl

# Confirm episodes are on disk
ls ~/.cache/huggingface/lerobot/$REPO/data/chunk-000/
# -> file-000.parquet

# Confirm v3 episode metadata exists
ls ~/.cache/huggingface/lerobot/$REPO/meta/episodes/
# -> chunk-000/file-000.parquet

# Open viewer for the most recent
uv run python scripts/lerobot_dataset_viz_main.py --repo-id $REPO \
  --episode-index $(ls ~/.cache/huggingface/lerobot/$REPO/data/chunk-000/ |
wc -l | awk '{print $1-1}')
```

Use the project wrapper instead of calling `lerobot-dataset-viz` directly. The vendored LeRobot checkout defaults to a version tag lookup for v3 datasets; the wrapper leaves the vendor code untouched, loads the Hub `main` revision instead, and defaults the dataloader to `--num-workers 0` to avoid shared-memory issues on small machines. To view another revision:

```bash
LEROBOT_DATASET_REVISION=my-branch uv run python scripts/lerobot_dataset_viz_main.py \
  --repo-id saivishwak/xlerobot-vr-teleop-medicine-bowl \
  --episode-index 0
```
