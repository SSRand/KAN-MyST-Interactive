#!/usr/bin/env bash
#
# Boot the local Jupyter server and the MyST dev site together. The token here
# matches `project.thebe.server.token` in myst.yml — only the localhost server
# is exposed, so the public-looking token is fine.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT_JUPYTER=${PORT_JUPYTER:-8888}
PORT_MYST=${PORT_MYST:-3000}
TOKEN=${TOKEN:-kan-demo-local}

JUPYTER_PID=""
MYST_PID=""

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$JUPYTER_PID" ]]; then kill "$JUPYTER_PID" 2>/dev/null || true; fi
  if [[ -n "$MYST_PID" ]]; then kill "$MYST_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  echo "[start] No .venv yet — running 'uv sync' first..."
  UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache} uv sync
  PY=".venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
  echo "[start] Warning: no .venv and uv not found; using system $PY"
fi

if ! command -v myst >/dev/null 2>&1; then
  echo "[start] mystmd CLI not on PATH. Install with: npm install -g mystmd"
  exit 1
fi

echo "[start] Jupyter on http://localhost:$PORT_JUPYTER  (token: $TOKEN)"
"$PY" -m jupyter server \
  --ServerApp.port="$PORT_JUPYTER" \
  --ServerApp.token="$TOKEN" \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.disable_check_xsrf=True \
  --no-browser >/tmp/kan-myst-jupyter.log 2>&1 &
JUPYTER_PID=$!

echo "[start] MyST on    http://localhost:$PORT_MYST"
myst start --port "$PORT_MYST" &
MYST_PID=$!

echo "[start] Jupyter log: /tmp/kan-myst-jupyter.log"
echo "[start] Open http://localhost:$PORT_MYST and click 'Run' on a code cell."
echo "[start] Ctrl-C to stop both."

wait
