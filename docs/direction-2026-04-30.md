# Direction Analysis after 2026-04-30 Meeting

> See `../../meetings/2026-04-30.md` (workspace-level `meetings/` folder) for
> the raw notes and the slide deck + PDFs referenced in the call. This
> document compares the action item Agah assigned against the current v0 path
> so we can decide what to keep and what to pivot.

## What Agah is actually asking for

Re-reading the action item carefully, four constraints are stacked:

1. **Restructure as a dashboard / data app.** Not a Jupyter-style article;
   a real web app with its own UI surface. Plotly Dash is one example; the
   framework is negotiable.
2. **Containerizable.** It has to ship as a self-contained image (Docker).
   Evidence's pipeline can host containers; it cannot host arbitrary Python
   processes on the reader's machine.
3. **URL-resolved modularity.** Interactive sections live at different URLs
   (`/coarse`, `/refine`, `/sparsify`, …). Each URL is one focused piece of
   the demo, not "the whole app on one page".
4. **Interleaved with prose in MyST via iframe.** The MyST article is the
   reader's path through the paper; each interactive panel appears at the
   right spot in the narrative as an `<iframe>` pointing at one of the
   dashboard URLs.

The mental picture:

```
MyST article (paper.md)
├── "Chapter 1 prose..."
├── <iframe src="https://kan-app.example/coarse">   ← coarse-fit panel
├── "Chapter 2 prose..."
├── <iframe src="https://kan-app.example/refine">   ← grid-refinement panel
├── "Chapter 3 prose..."
├── <iframe src="https://kan-app.example/sparsify"> ← sparsification panel
└── …
```

Two systems, one front door (MyST) — connected by URLs, not by a Jupyter
kernel.

## What v0 currently does

The v0 just committed in `kan-myst-demo/` does this:

| Element | v0 implementation |
|---|---|
| Article | MyST `paper.md`, `article-theme` |
| Interactive cell | `{code-cell} python` block inline in `paper.md` |
| Kernel | Thebe attaches to a local `jupyter server` on `localhost:8888` (booted by `start.sh`) |
| Compute | `pykan` running in that local kernel |
| Modularity | None — one monolithic cell does everything |
| Deployable as a container | No |

## Where v0 aligns, where it doesn't

**Aligns ✓**

- The article surface is MyST. Whatever pivot happens above, this stays.
- The article-theme styling, citations, equations, cross-references are the
  Evidence-recognised reading surface.
- Pinned Python deps (`pyproject.toml`) and the choice to keep compute off
  Evidence's infrastructure both port forward.

**Does not align ✗**

- **Wrong execution layer.** Thebe + Jupyter is a "code-in-the-article"
  pattern; Agah is asking for a "dashboard-behind-an-iframe" pattern. These
  are different architectures, not different configurations of the same one.
- **No URL modularity.** v0 has one inline cell. The ask is many small
  URL-addressable panels.
- **Not containerizable.** v0's "Python backend" is the reader's own Jupyter
  process. Agah wants something we can wrap in a Docker image and hand to
  Evidence's deployment.
- **No iframe boundary.** MyST and the compute live in the same browser
  process; iframe isolation (which is what Evidence's editorial flow expects
  for embedded apps) is missing.

Roughly: v0 solved the wrong problem (how to keep Thebe working without
Binder), even though it solved it well. The right problem is how to ship a
containerized data app with modular URLs and let MyST embed slices of it.

## Pivot plan

Two layers, both new:

### Layer A — the dashboard app (`app/` subdirectory)

A Python web app (recommend Plotly Dash to start; FastAPI + a small React/Vue
front end is the alternative). Multiple routes, each a focused panel:

| Route | What it does | What the reader manipulates |
|---|---|---|
| `/target` | Pick or type a 2-D target function; preview as a heatmap | Expression text input, example dropdown |
| `/coarse` | Train a `[2,5,1]` grid-5 KAN, show edge functions and residual | Width, grid, steps sliders |
| `/refine` | Take a fitted model and refine its grid (grid extension) | Grid size slider, refinement steps |
| `/sparsify` | Apply entropy penalty, watch edges thin out | Penalty weight, steps |
| `/prune` | Manually or automatically prune low-magnitude edges | Threshold slider |
| `/symbolic` | Snap surviving edges to symbolic forms; show the formula | Per-edge symbol picker |

Each panel is **self-contained at its URL** — visit `/coarse` directly and you
get a working tool, not a page that complains about missing global state.
State that needs to flow between panels (a fitted model, for example)
travels via URL params or a tiny shared session store; this is part of the
"URL-resolved" point — the URL is the API.

The reusable training core already exists: `kan-fulltext-demo/scripts/kan_engine.py`
has the multi-stage training as Python functions and `serve.py` has a working
job-queue pattern. The dashboard wraps those, replaces the bespoke HTML
frontend with Dash components, splits the single-page UI into per-route panels,
and ships a `Dockerfile`.

### Layer B — the MyST article (this repo's `paper.md`)

Replace each `{code-cell}` block with an `{iframe}` directive pointing at the
corresponding dashboard URL:

```markdown
:::{iframe} https://kan-app.evidencepub.io/coarse
:label: fig-coarse
:width: 100%
:height: 600px
Train a coarse KAN on a 2-D target. Adjust the grid and watch the edge
splines respond.
:::
```

The text around it is normal prose. The cross-reference target
(`#fig-coarse`) lets the rest of the paper refer back to the interaction by
number.

A reader of the published article sees the dashboard panel inline; a reader
of the source markdown sees a clean iframe directive and can chase the URL.

## What of v0 to keep, what to drop

| Asset | Verdict |
|---|---|
| `paper.md` (the article shell, citations, equations, layout) | **Keep**, rewrite the interactive sections |
| `paper.bib` | **Keep** |
| `myst.yml` (theme, bibliography, abbreviations) | **Keep**, but **delete** the `thebe.server` block (no Thebe in the pivoted design) |
| `content/kan_demo.ipynb` | **Optional**: keep as a "supporting documents" download for readers who want to reproduce the run locally without the dashboard. Not part of the rendered story. |
| `pyproject.toml`, `start.sh` | **Drop from this repo** — they belong to the dashboard repo/subdirectory, not the article. |
| `.gitignore` | **Keep**, extend with the dashboard build outputs |

The MyST repo becomes article-only. The dashboard either lives at
`kan-myst-demo/app/` (mono-repo) or in a separate repo
(`KAN-Interactive-App`). My recommendation: **subdirectory inside the same
repo**, because the article and the dashboard ship together and the iframe
URLs are coordinates that must stay in sync.

## Open questions to bring back to Agah / the team

1. **Hosting.** Where does the container actually run? Evidence-supplied
   infra? Our own (fly.io, render, a small VM)? This decides whether the
   iframe URLs are evidencepub-owned or ours — and whether CORS / X-Frame-
   Options need negotiation.
2. **Cold start vs. warm session.** If multiple panels need to share a
   fitted model (you fit on `/coarse`, refine on `/refine`), do we expect a
   server-side session store, or do we serialize the model into URL params?
3. **Auth / rate limiting.** A public dashboard that runs `model.fit()` on
   every request is a DoS waiting to happen. Is the deployment scoped to
   approved readers (links from the article) or open?
4. **The Jupyter notebook story.** Evidence's other templates
   (mystical-article, mooc, mriscope) all show notebooks; should the
   dashboard route panels also be downloadable as `.ipynb`, or is the
   notebook deliberately dropped in favour of the dashboard?

## Recommended next step

A 1-2 day spike, in this order:

1. Stand up a single-route Dash app under `app/` that wraps the existing
   `kan_engine.py` and serves `/coarse`. Containerize it with a minimal
   Dockerfile.
2. Replace exactly one `{code-cell}` in `paper.md` with an `{iframe}` to
   that route, point at `localhost:8050` for local dev.
3. Show Agah the result on the next call. Get confirmation on the iframe
   pattern before generalising to all five panels.

If the spike confirms the direction, layer in the other routes one at a time;
the upfront pivot risk is contained to the first panel.
