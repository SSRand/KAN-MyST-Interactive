---
title: KAN as a Dynamic Paper
subtitle: An interactive Evidence-style paper backed by a containerised Dash app
numbering:
  heading_2: false
  figure:
    template: Fig. %s
---

## From a static PDF to a runnable artifact

Kolmogorov-Arnold Networks (KANs) replace the fixed activation functions on the
nodes of a multi-layer perceptron with learnable splines on the edges
@KAN2024. The change is small to state and substantial in effect: the network
is no longer a black box mapping `[input → activation → output]`, but an
explicit composition of one-dimensional functions whose shapes can be inspected
after training.

This page is the v0 of the dashboard-based rebuild Evidence asked for in the
2026-04-30 meeting. The interactive panel below is **not** a Jupyter cell:
it is an `<iframe>` into one route of a sibling Plotly Dash app (see `app/`).
Each panel is independently addressable by URL, lives in its own container,
and is meant to be embedded next to the prose that explains the step it
demonstrates. The KAN paper has natural sections (coarse fit, grid refinement,
sparsification, pruning, symbolic snapping); v1 will add one panel per section.

## A baseline training run

The reference run trains a tiny `[2, 5, 1]` KAN on the surface

```{math}
:label: target
f(x, y) \;=\; \exp\!\bigl(\sin(\pi x) + y^{2}\bigr).
```

The panel below is `http://localhost:8050/coarse` from the dashboard app. Move
the grid slider to set the spline resolution, optionally adjust the LBFGS step
budget, then click **Train**. The learned KAN diagram and the loss curve update
in place.

:::{iframe} http://localhost:8050/coarse
:label: panel-coarse
:width: 100%
:height: 720
Coarse KAN fit. Identical training loop to the upstream `kan-fulltext-demo`,
exposed as a single URL-addressable panel so it can be embedded here and
nowhere else of the article.
:::

## What v0 deliberately leaves out

This page implements only the first of the planned panels (`/coarse`). The
remaining panels (`/refine`, `/sparsify`, `/prune`, `/symbolic`) and the
target-function picker live in the v1 backlog. Both the dashboard app
(`app/pages/`) and the article (this file) are designed so a new panel is one
new file in `app/pages/` plus one `:::{iframe}` directive here.

A Jupyter notebook with the same default training run is kept under
`content/kan_demo.ipynb` for readers who want to reproduce locally without
running the dashboard container.
