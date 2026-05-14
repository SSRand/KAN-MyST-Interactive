# KAN-MyST-Interactive

An interactive Kolmogorov–Arnold Network paper rendered as a MyST article with
Evidence's `article-theme`, executed against a **local** Jupyter kernel. The
page presents itself as an Evidence preprint and uses Thebe-style "Run" buttons
on every code cell — but the compute runs on your own machine, so the loop is
fast and does not depend on Evidence's Binder ever booting.

> Upstream of `kan-fulltext-demo/`: this is the MyST-native rewrite Evidence
> asked for. Compute stays local; the MyST + Thebe surface is the part Evidence
> recognises as theirs.

Maintained at: https://github.com/SSRand/KAN-MyST-Interactive

## Why local execution

The Evidence Binder (`terrarium.evidencepub.io`) cannot currently build
arbitrary repositories on demand, and even when it does, pykan + torch + numerics
yields a ~10 minute cold start. For a demo whose value is interactive
iteration on the training, that is a non-starter. So:

- **Frontend (Evidence's part)**: MyST article-theme, citation rendering,
  cross-references, Thebe "Run" UI, downloadable notebook.
- **Backend (your part)**: a Jupyter server on `localhost:8888`, started by
  `./start.sh`. MyST's Thebe config points at it directly.

## Quick start

```bash
git clone https://github.com/SSRand/KAN-MyST-Interactive.git
cd KAN-MyST-Interactive

# one-time setup
UV_CACHE_DIR=/tmp/uv-cache uv sync         # installs pykan + torch + jupyter
npm install -g mystmd                       # if not already on PATH

# boot Jupyter + MyST together
./start.sh
```

Open `http://localhost:3000` and click "Run" on any `{code-cell}` block — the
first click takes ~2 s to attach to the local kernel, subsequent runs are
instant.

## Layout

```
kan-myst-demo/                       # folder; GitHub repo is KAN-MyST-Interactive
├── myst.yml                         # MyST project + thebe.server → localhost
├── paper.md                         # the article (narrative + inline code cells)
├── paper.bib                        # bibliography
├── pyproject.toml                   # uv-managed Python deps (pykan, torch, jupyter)
├── start.sh                         # boots Jupyter on 8888 + MyST on 3000
├── content/
│   └── kan_demo.ipynb               # labelled cells (#kan-fit, #kan-loss)
└── README.md
```

## How the wiring works

| Surface | Mechanism |
|---|---|
| Article text, figures, equations, citations | Standard MyST markdown rendered with `article-theme` |
| `:::{figure} #kan-fit` and `#kan-loss` | Cell-label references resolved by MyST-NB into the notebook outputs |
| `{code-cell} python` in `paper.md` | Inline executable cell; Thebe attaches it to the local kernel on first "Run" click |
| Local kernel | `jupyter server` on `localhost:8888`, token `kan-demo-local`, CORS open to MyST's `localhost:3000` |
| MyST → kernel binding | `project.thebe.server` block in `myst.yml` |

The token in `myst.yml` is intentionally fixed and committed — the server is
only reachable on localhost, so the token authorises nothing outside the
reader's own machine.

## v0 acceptance criteria

- `myst start` renders `paper.md` without warnings about missing labels.
- `./start.sh` brings up both servers; visiting `http://localhost:3000` shows
  the article with "Run" affordances on the code cells.
- Clicking "Run" on `kan-interactive` produces the `model.plot()` matplotlib
  figure inline within ~10 s on a typical laptop.

## v1 backlog

- Port `kan-fulltext-demo/scripts/build_fulltext.py` to emit MyST chapters in
  `content/` so the full paper text travels with the demo.
- Wrap `kan-interactive` in an `{anywidget}` widget that mirrors the
  `kan-fulltext-demo` scrubber UI and dispatches its training through the same
  local Thebe kernel via `thebe-core`'s programmatic API.
- Pre-execute the notebook with `myst build --execute` so the article still
  shows useful default outputs when `./start.sh` is not running.
