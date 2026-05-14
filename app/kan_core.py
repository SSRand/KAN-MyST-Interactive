"""Minimal pykan wrappers used by the Dash routes.

Every `train_*` function is pure: same inputs → same outputs, no global
state, no disk side effects. Each Dash page calls one of these and gets back
loss arrays + a base64-encoded KAN diagram PNG.
"""

from __future__ import annotations

import base64
import io
import math
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from kan import KAN, create_dataset  # noqa: E402


DEFAULT_EXPRESSION = "exp(sin(pi*x) + y**2)"


def _default_target(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.sin(math.pi * x[:, [0]]) + x[:, [1]] ** 2)


def _setup() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)


def _dataset() -> dict[str, torch.Tensor]:
    return create_dataset(
        _default_target,
        n_var=2,
        train_num=300,
        test_num=300,
        seed=17,
    )


def _new_kan(grid: int, ckpt_path: str) -> KAN:
    """Build a fresh KAN. `ckpt_path` must point at an existing directory:
    pykan's `fit()` calls `log_history` which opens `<ckpt_path>/history.txt`
    in append mode and that errors if the directory is missing, even when
    `auto_save=False`.
    """
    return KAN(
        width=[2, 5, 1],
        grid=int(grid),
        k=3,
        seed=0,
        device=torch.device("cpu"),
        auto_save=False,
        ckpt_path=ckpt_path,
    )


def _render_diagram(model: KAN, dataset: dict, title: str) -> str:
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
            title=title,
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="white")
        plt.close("all")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _losses(history: dict) -> tuple[list[float], list[float]]:
    train = [float(x) for x in history.get("train_loss", [])]
    test = [float(x) for x in history.get("test_loss", [])]
    return train, test


def train_coarse(grid: int = 5, steps: int = 20) -> dict[str, Any]:
    """One [2, 5, 1] KAN fit on the default target."""
    _setup()
    dataset = _dataset()
    with tempfile.TemporaryDirectory(prefix="kan-ckpt-") as ckpt_dir:
        model = _new_kan(grid, ckpt_path=ckpt_dir)
        history = model.fit(dataset, opt="LBFGS", steps=int(steps), log=10**9)
        train_loss, test_loss = _losses(history)
        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "graph_png_b64": _render_diagram(
                model, dataset, f"KAN  grid={grid}  steps={steps}"
            ),
        }


def train_sparsify(
    coarse_steps: int = 20,
    lamb: float = 0.002,
    lamb_entropy: float = 1.0,
    sparse_steps: int = 20,
) -> dict[str, Any]:
    """Coarse fit, then a continued fit under L1 + entropy penalties.

    The first phase fits the KAN normally. The second phase resumes training
    with `lamb` (overall regularisation) and `lamb_entropy` (entropy weight),
    which together push activations toward "few large edges" — visibly thins
    the diagram.
    """
    _setup()
    dataset = _dataset()
    with tempfile.TemporaryDirectory(prefix="kan-ckpt-") as ckpt_dir:
        model = _new_kan(grid=5, ckpt_path=ckpt_dir)
        coarse_history = model.fit(
            dataset, opt="LBFGS", steps=int(coarse_steps), log=10**9
        )
        coarse_train, coarse_test = _losses(coarse_history)

        sparse_history = model.fit(
            dataset,
            opt="LBFGS",
            steps=int(sparse_steps),
            lamb=float(lamb),
            lamb_entropy=float(lamb_entropy),
            log=10**9,
        )
        sparse_train, sparse_test = _losses(sparse_history)

        return {
            "train_loss": coarse_train + sparse_train,
            "test_loss": coarse_test + sparse_test,
            "split_at": len(coarse_train),
            "graph_png_b64": _render_diagram(
                model,
                dataset,
                f"sparsified  λ={lamb:.4g}  H={lamb_entropy:g}",
            ),
        }


def train_refine(
    coarse_grid: int = 5,
    coarse_steps: int = 20,
    refined_grid: int = 10,
    refined_steps: int = 20,
) -> dict[str, Any]:
    """Coarse fit, then `model.refine()` to a finer spline grid, then continued training."""
    _setup()
    dataset = _dataset()

    with tempfile.TemporaryDirectory(prefix="kan-ckpt-") as ckpt_dir:
        model = _new_kan(coarse_grid, ckpt_path=ckpt_dir)
        coarse_history = model.fit(dataset, opt="LBFGS", steps=int(coarse_steps), log=10**9)
        coarse_train, coarse_test = _losses(coarse_history)

        model = model.refine(int(refined_grid))
        # `refine()` returns a fresh KAN whose ckpt_path resets to pykan's
        # default ('./model'), which may not exist. Pin it back to our tempdir.
        model.ckpt_path = ckpt_dir

        refined_history = model.fit(dataset, opt="LBFGS", steps=int(refined_steps), log=10**9)
        refined_train, refined_test = _losses(refined_history)

        return {
            "train_loss": coarse_train + refined_train,
            "test_loss": coarse_test + refined_test,
            "split_at": len(coarse_train),
            "graph_png_b64": _render_diagram(
                model, dataset, f"refined  grid={coarse_grid} → {refined_grid}"
            ),
        }
