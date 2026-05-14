#!/usr/bin/env bash
#
# Boot the Dash dashboard app and the MyST dev site together for local
# development. The article (paper.md) iframes panels from the dashboard, so
# both processes need to be running for the page to render fully.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT_DASH=${PORT_DASH:-8050}
PORT_MYST=${PORT_MYST:-3000}

DASH_PID=""
MYST_PID=""

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$DASH_PID" ]]; then kill "$DASH_PID" 2>/dev/null || true; fi
  if [[ -n "$MYST_PID" ]]; then kill "$MYST_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

if ! command -v uv >/dev/null 2>&1; then
  echo "[start] uv not on PATH. Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if ! command -v myst >/dev/null 2>&1; then
  echo "[start] mystmd CLI not on PATH. Install with: npm install -g mystmd"
  exit 1
fi

# `uv run` will create the .venv from pyproject.toml + uv.lock on first call
# and reuse it afterwards. No separate setup step needed.
echo "[start] Dash on  http://localhost:$PORT_DASH"
PORT="$PORT_DASH" uv run python app/app.py >/tmp/kan-dash.log 2>&1 &
DASH_PID=$!

echo "[start] MyST on  http://localhost:$PORT_MYST"
myst start --port "$PORT_MYST" &
MYST_PID=$!

echo "[start] Dash log: /tmp/kan-dash.log"
echo "[start] Open http://localhost:$PORT_MYST and scroll to the iframe panel."
echo "[start] Ctrl-C to stop both."

wait
