# KAN-MyST-Interactive

Interactive Kolmogorov–Arnold Network paper, structured for the Evidence
publishing platform. Two pieces:

1. **A MyST article** (`paper.md`, rendered with the Evidence `article-theme`)
   that carries the prose, equations, citations, and figures.
2. **A Plotly Dash dashboard** (`app/`) that exposes each interactive panel
   at its own URL. The article embeds each URL with an `:::{iframe}` directive
   next to the relevant section.

This split is the architecture Agah asked for in the 2026-04-30 meeting:
URL-resolved modular panels, containerised so Evidence can deploy them, and
interleaved with prose via iframe. See
[`docs/direction-2026-04-30.md`](docs/direction-2026-04-30.md) for the
rationale and the v0/v1 boundary.

Repo: https://github.com/SSRand/KAN-MyST-Interactive

## v0 scope

Only one panel is implemented (`/coarse` — coarse KAN fit on a fixed target).
The architecture works end to end: visiting the article triggers the iframe,
the iframe loads the Dash route, the route trains a KAN on demand and returns
the diagram and loss curve. Subsequent panels (`/refine`, `/sparsify`, `/prune`,
`/symbolic`) and a target-function picker are v1 work.

## Quick start (local dev)

```bash
git clone https://github.com/SSRand/KAN-MyST-Interactive.git
cd KAN-MyST-Interactive

# one-time tooling
#   uv:  https://docs.astral.sh/uv/getting-started/installation/
#   mystmd:
npm install -g mystmd

# boot both: Dash on 8050, MyST on 3000
./start.sh
```

`start.sh` calls `uv run` for the Dash side — uv creates `.venv/` and installs
everything from `pyproject.toml` + `uv.lock` on first call, reuses it
afterwards. No separate `pip install` step.

Open `http://localhost:3000` for the article. The iframe in the "A baseline
training run" section loads `http://localhost:8050/coarse` directly.

## Layout

```
KAN-MyST-Interactive/
├── paper.md              # the article (prose, equations, iframe directives)
├── paper.bib             # bibliography
├── myst.yml              # MyST project config (theme, bibliography, abbreviations)
├── pyproject.toml        # uv-managed Python deps (Dash + pykan + torch)
├── uv.lock               # pinned dependency resolution; committed
├── start.sh              # boots app/ + MyST together for dev
├── content/
│   └── kan_demo.ipynb    # supporting notebook (not embedded in the article)
├── app/                  # the Dash dashboard
│   ├── app.py            # multi-page entry point
│   ├── kan_core.py       # train_coarse(grid, steps)
│   ├── pages/
│   │   ├── home.py       # /
│   │   └── coarse.py     # /coarse
│   ├── Dockerfile        # build context = repo root (sees pyproject + uv.lock)
│   └── README.md
└── docs/
    ├── direction-2026-04-30.md      # why this architecture
    └── ../../meetings/2026-04-30.md # raw meeting notes (workspace-level)
```

## Production deployment

The dashboard is shipped as a Docker image. The Dockerfile lives in
`app/Dockerfile` but the build context is the **repo root** — that's how the
container sees `pyproject.toml` and `uv.lock`:

```bash
docker build -f app/Dockerfile -t kan-dashboard .
docker run --rm -p 8050:8050 kan-dashboard
```

Inside the image, `uv sync --frozen` installs the exact versions recorded in
`uv.lock`, so the deployed runtime matches your local one.

The article remains a static MyST build. The `:::{iframe}` URL in `paper.md`
is the public URL of the dashboard container; for local dev it is
`http://localhost:8050/coarse`, for a published deployment it would be (for
example) `https://kan-app.evidencepub.io/coarse`. Swap the URL in `paper.md`
and rebuild the article.

## v1 backlog

- One Dash page per remaining training stage (`/refine`, `/sparsify`,
  `/prune`, `/symbolic`).
- Target-function picker (Evidence-validated AST allowlist, reused from
  `kan-fulltext-demo/scripts/kan_engine.py`).
- Port the full KAN paper text from `kan-fulltext-demo/scripts/build_fulltext.py`
  into MyST chapters so the demo carries the whole paper, not just a stub.
- Decide on production hosting for the dashboard container (Evidence infra
  vs. self-hosted) and pin a stable URL in `paper.md`.
