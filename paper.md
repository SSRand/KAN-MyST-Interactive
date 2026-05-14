---
title: KAN as a Dynamic Paper
subtitle: Interactive Kolmogorov–Arnold Networks
numbering:
  heading_2: false
  figure:
    template: Fig. %s
---

## Kolmogorov–Arnold Networks

A Kolmogorov–Arnold Network places learnable spline activations on the edges
of an MLP instead of fixed activations on the nodes @KAN2024. The change is
small to state and substantial in effect: the network is no longer a black
box mapping inputs to outputs, but an explicit composition of one-dimensional
functions whose shapes can be inspected after training.

This page interleaves the paper's prose with the training loop itself. Each
section introduces a step of KAN training and is followed by an interactive
panel that performs that step on a sample target. Every panel is a separate
URL on the sibling Dash app and is embedded here as an `<iframe>`.

## Coarse fit

We start with a tiny `[2, 5, 1]` KAN on the target surface

```{math}
:label: target
f(x, y) \;=\; \exp\!\bigl(\sin(\pi x) + y^{2}\bigr).
```

The panel sets the spline grid resolution and the LBFGS step budget, runs
the fit, and shows the learned KAN diagram together with the loss curve.

:::{iframe} http://localhost:8050/coarse
:label: panel-coarse
:width: 100%
:height: 720px
:::

## Grid refinement

Once the coarse fit converges, `model.refine()` upsamples each edge spline
to a finer grid and resumes training. The cost is modest; the accuracy gain
can be substantial. The next panel runs the coarse fit and the refinement
back to back; the dashed line on the loss curve marks the transition.

:::{iframe} http://localhost:8050/refine
:label: panel-refine
:width: 100%
:height: 760px
:::

## Sparsification

Accuracy is one half of KAN's value proposition; interpretability is the
other. Adding an L1 + entropy penalty during training shrinks edges that
carry little signal, so what remains is a smaller, easier-to-read network
that approximates the same function. The next panel resumes training under
the penalty and renders the thinned diagram.

:::{iframe} http://localhost:8050/sparsify
:label: panel-sparsify
:width: 100%
:height: 760px
:::
