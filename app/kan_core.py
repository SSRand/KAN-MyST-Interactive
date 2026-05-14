"""Minimal pykan wrapper used by the Dash routes.

Designed so each Dash page calls one function and gets back data + a rendered
PNG.  No disk side effects, no global state — every request is independent.
That matches Agah's "URL-resolved modular panels" framing: the kernel of the
demo (training) is a pure function from (knob settings) to (artifact).
"""

from __future__ import annotations

import base64
import io
import math
import os
import tempfile
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt  # noqa: E402  must follow MPLBACKEND
import torch  # noqa: E402
from kan import KAN, create_dataset  # noqa: E402


DEFAULT_EXPRESSION = "exp(sin(pi*x) + y**2)"


def _default_target(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.sin(math.pi * x[:, [0]]) + x[:, [1]] ** 2)


def train_coarse(grid: int = 5, steps: int = 20) -> dict[str, Any]:
    """Train one [2, 5, 1] KAN with the given grid / LBFGS steps.

    Returns:
        train_loss, test_loss: lists of floats
        graph_png_b64: data URL ready for <img src=...>
    """
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)

    dataset = create_dataset(
        _default_target,
        n_var=2,
        train_num=300,
        test_num=300,
        seed=17,
    )
    model = KAN(
        width=[2, 5, 1],
        grid=int(grid),
        k=3,
        seed=0,
        device=torch.device("cpu"),
        auto_save=False,
    )

    history = model.fit(dataset, opt="LBFGS", steps=int(steps), log=10**9)
    train_loss = [float(x) for x in history.get("train_loss", [])]
    test_loss = [float(x) for x in history.get("test_loss", [])]

    plt.close("all")
    with tempfile.TemporaryDirectory() as tmp:
        model(dataset["train_input"])
        model.plot(
            folder=tmp,
            beta=8,
            metric="forward_n",
            scale=0.72,
            in_vars=[r"$x$", r"$y$"],
            out_vars=[r"$f(x,y)$"],
            title=f"KAN  grid={grid}  steps={steps}",
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="white")
        plt.close("all")

    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("ascii")

    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "graph_png_b64": f"data:image/png;base64,{png_b64}",
    }
