# KAN-MyST-Interactive

Interactive Kolmogorov–Arnold Network paper. The article (`paper.md`) is
rendered with MyST; each interactive panel is a route on a sibling Dash app
(`app/`) embedded inline via `<iframe>`.

Repo: https://github.com/SSRand/KAN-MyST-Interactive

## Run locally

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and `mystmd`
(`npm install -g mystmd`).

```bash
./start.sh
```

This boots the Dash app on `http://localhost:8050` and the MyST dev site on
`http://localhost:3000`. Open the MyST URL.

## Container deployment

```bash
docker build -f app/Dockerfile -t kan-dashboard .
docker run --rm -p 8050:8050 kan-dashboard
```

The Dockerfile is in `app/` but the build context is the repo root so the
container can see `pyproject.toml` and `uv.lock`. Inside the image,
`uv sync --frozen` reproduces the exact dependency versions.

## Layout

```
.
├── paper.md          # the article: prose + iframe directives
├── myst.yml
├── paper.bib
├── pyproject.toml    # uv-managed Python deps
├── uv.lock
├── app/              # Dash dashboard — one route per panel
│   ├── app.py
│   ├── kan_core.py
│   ├── figures.py         # shared loss_figure + error_panel helpers
│   ├── pages/
│   │   ├── home.py        # /
│   │   ├── coarse.py      # /coarse
│   │   ├── refine.py      # /refine
│   │   ├── sparsify.py    # /sparsify
│   │   ├── prune.py       # /prune
│   │   └── symbolic.py    # /symbolic
│   ├── Dockerfile
│   └── README.md
├── start.sh
└── content/kan_demo.ipynb   # supporting notebook (not embedded)
```

Adding a new panel is one file in `app/pages/` plus one `:::{iframe}`
directive in `paper.md`.
