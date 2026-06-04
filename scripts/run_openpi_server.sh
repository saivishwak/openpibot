#!/usr/bin/env bash
# Start the package-managed OpenPI PI0.5 policy WebSocket server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${OPENPI_CONFIG:-pi05_bimanual_so101_lora}"
CKPT_DIR="${OPENPI_CHECKPOINT_DIR:-gs://openpi-assets/checkpoints/pi05_base}"
PORT="${OPENPI_PORT:-8000}"
OPENPI_PACKAGE="${OPENPI_PACKAGE:-openpi @ git+https://github.com/Physical-Intelligence/openpi.git}"

echo "[openpi-server] config=$CONFIG"
echo "[openpi-server] ckpt  =$CKPT_DIR"
echo "[openpi-server] port  =$PORT"
echo "[openpi-server] pkg   =$OPENPI_PACKAGE"

exec uv run --no-project --with "$OPENPI_PACKAGE" \
    python scripts/openpi_policy_server.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT_DIR"
