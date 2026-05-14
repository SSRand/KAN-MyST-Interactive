# KAN Dashboard

The Python web app that powers the interactive panels embedded by `paper.md`.
Each route is one focused panel; the article iframes them individually.

| Route | Purpose |
|---|---|
| `/` | Index of panels |
| `/coarse` | Coarse `[2, 5, 1]` KAN fit |
| `/refine` | Coarse fit + spline grid refinement |
| `/sparsify` | Coarse fit + continued training under L1 + entropy penalty |
| `/prune` | Sparsify + `model.prune()` + refit on the surviving topology |
| `/symbolic` | Full pipeline + auto-symbolic snap; renders a closed-form formula |

## Local development

Python deps are managed by `uv` at the repo root.

```bash
uv sync                    # one-time, creates ../.venv
uv run python app/app.py   # boots Dash on http://localhost:8050
```

## Container image

Build context is the **repo root** so the container sees `pyproject.toml`
and `uv.lock`:

```bash
docker build -f app/Dockerfile -t kan-dashboard .
docker run --rm -p 8050:8050 kan-dashboard
```

## Adding a panel

1. Add `pages/<name>.py` with a `dash.register_page(__name__, path="/<name>", ...)`
   call. Dash auto-discovers it; the route is live on the next process start.
2. Add a `:::{iframe} http://localhost:8050/<name>` block in `../paper.md`.

`kan_core.py` exposes small training helpers (`_setup`, `_dataset`,
`_new_kan`, `_render_diagram`) so new `train_*` functions can compose them
without duplication.
