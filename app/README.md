# KAN Dashboard

The Python web app that powers the interactive panels embedded by the KAN
paper. Each route is one focused panel; the article in the parent directory
iframes them individually.

| Route | Status | Purpose |
|---|---|---|
| `/` | v0 | Landing page with links to the panels |
| `/coarse` | v0 | Coarse `[2, 5, 1]` KAN fit on a fixed target |
| `/refine` | planned | Grid refinement after the coarse fit |
| `/sparsify` | planned | Entropy-penalty sparsification |
| `/prune` | planned | Edge pruning |
| `/symbolic` | planned | Symbolic snapping of surviving edges |

## Local development

Python deps are managed by `uv` at the **repo root** (`../pyproject.toml`),
not in this folder. From the repo root:

```bash
uv sync                           # creates ../.venv and installs everything
uv run python app/app.py          # boots Dash on port 8050
```

Then open `http://localhost:8050/coarse`. The first training run takes a few
seconds (loading torch, building the KAN, LBFGS). Subsequent runs reuse the
in-process model state.

If you'd rather start both the dashboard and the MyST dev site in one
command, run `../start.sh` from this folder's parent — that's the dev path
used during article work.

## Container image

The Dockerfile lives here but the build context is the repo root (so the
container can see `pyproject.toml` + `uv.lock`):

```bash
cd ..                                              # repo root
docker build -f app/Dockerfile -t kan-dashboard .
docker run --rm -p 8050:8050 kan-dashboard
```

The image uses the official statically-linked `uv` binary to install deps
inside the container (one `uv sync --frozen` step), then runs `gunicorn` from
the venv on PATH. No surprises.

## Code layout

```
app/
├── app.py              # Dash multi-page entry point
├── kan_core.py         # train_coarse(grid, steps) → loss + KAN diagram
├── pages/
│   ├── home.py         # /
│   └── coarse.py       # /coarse
├── Dockerfile          # built with `docker build -f app/Dockerfile .` from root
└── README.md           # you are here
```

`pages/<name>.py` is auto-discovered by Dash and registered to `/<name>` (the
file calls `dash.register_page(__name__, path="/coarse", ...)`). Adding a new
panel is one new file in `pages/` plus one `:::{iframe}` directive in the
sibling `paper.md`.
