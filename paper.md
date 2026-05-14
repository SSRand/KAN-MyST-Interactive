---
title: KAN as a Dynamic Paper
subtitle: An interactive Evidence-style paper, run locally
kernelspec:
  name: python3
  display_name: Python 3
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

This page is the same KAN interaction that powered the original HTML demo,
re-skinned as a MyST article rendered with Evidence's `article-theme`. Live
training still happens through a Thebe-style "Run" button on the cells below,
but the kernel attaches to a **Jupyter server running on your own machine**
(booted by `./start.sh`) — Evidence's Binder is not in the loop. The result
looks and reads like an Evidence preprint while keeping the snappy single-digit
second turnaround of local pykan training.

## A baseline training run

The reference run trains a tiny `[2, 5, 1]` KAN on the surface

```{math}
:label: target
f(x, y) \;=\; \exp\!\bigl(\sin(\pi x) + y^{2}\bigr).
```

The notebook cell labelled `kan-fit` in `content/kan_demo.ipynb` does the
training and renders the learned edge functions via `model.plot()`.

:::{figure} #kan-fit
:label: fig-kan
The KAN graph after 20 LBFGS steps on the target surface
[](#target). Each edge is a learnable spline; thicker edges
carry larger contributions to the output.
:::

:::{figure} #kan-loss
:label: fig-loss
Training and test loss for the same run. The drop on the first few steps comes
from LBFGS fitting the coarse 5-grid splines; later refinement and pruning
stages (not shown in v0) flatten the curve further.
:::

## A cell you can rerun

The block below is the same training loop, exposed inline so a reader can edit
the target expression or the network width and rerun it. Thebe attaches to the
local Jupyter server started by `./start.sh`, so the first run reuses an
already-warm kernel and a 20-step LBFGS fit finishes in a handful of seconds.

```{code-cell} python
:label: kan-interactive
import math
import matplotlib.pyplot as plt
import torch
from kan import KAN, create_dataset

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

def target(x):
    # Edit this expression to explore a different surface.
    return torch.exp(torch.sin(math.pi * x[:, [0]]) + x[:, [1]] ** 2)

dataset = create_dataset(target, n_var=2, train_num=300, test_num=300)
model = KAN(width=[2, 5, 1], grid=5, k=3, seed=1)
model.fit(dataset, opt="LBFGS", steps=20)
model.plot(beta=10, scale=0.5)
plt.gcf()
```

## What v0 deliberately skips

This page is the smallest viable migration. The richer interactive UI from
`kan-fulltext-demo` — typed expression input, target preview heatmap,
stage-by-stage scrubber, symbolic readout — is **not** ported yet. v0 only
proves the architecture:

- the article reads like an Evidence preprint (article-theme, citations,
  cross-references, "Run" buttons);
- the inline `{code-cell}` is a real Jupyter cell that attaches to a local
  kernel via Thebe;
- `pykan` runs on the reader's machine, so the loop is fast and predictable
  regardless of any remote build queue.

The v1 plan wraps the inline cell in an `{anywidget}` widget that preserves
the original scrubber UX while dispatching its training calls through the same
local kernel — see the project README.
