"""Minimal pykan wrappers used by the Dash routes.

Every `train_*` function is pure: same inputs → same outputs, no global
state, no disk side effects. Each Dash page calls one of these and gets back
loss arrays + a base64-encoded KAN diagram PNG.
"""

from __future__ import annotations

import ast
import base64
import io
import os
import tempfile
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from kan import KAN, create_dataset  # noqa: E402


DEFAULT_EXPRESSION = "exp(sin(pi*x) + y**2)"

# Allowed identifiers, functions, and AST node types for user-supplied target
# expressions. Anything outside the allowlist (`__import__`, attribute access,
# comprehensions, …) is rejected before the expression is ever evaluated.
SAFE_FUNCTIONS = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
}
NUMPY_SAFE_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}
SAFE_NAMES = set(SAFE_FUNCTIONS) | {"x", "y", "pi", "e"}
SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def validate_expression(expression: str) -> str:
    """Parse and statically validate a user target expression.

    Returns the normalised form (caret → double-asterisk). Raises ValueError
    if any token, name, or AST node falls outside the allowlist.
    """
    normalised = expression.strip().replace("^", "**")
    if not normalised:
        raise ValueError("Target expression is empty.")
    tree = ast.parse(normalised, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_NODES):
            raise ValueError(f"Unsupported expression element: {node.__class__.__name__}")
        if isinstance(node, ast.Name) and node.id not in SAFE_NAMES:
            raise ValueError(f"Unsupported name: {node.id}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls like sin(x) are supported.")
        if isinstance(node, ast.Call) and node.func.id not in SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function: {node.func.id}")
    return normalised


def expression_to_latex_body(expression: str) -> str:
    """Format a (validated) user expression as LaTeX.

    Returns just the right-hand side; callers wrap it in `f(x, y) = ...` if
    they want the full identity. Raises ValueError if the expression isn't
    in the AST allowlist; any sympy-level conversion error is propagated as
    a generic Exception.
    """
    normalised = validate_expression(expression)
    import sympy

    x_sym, y_sym = sympy.symbols("x y")
    expr = sympy.sympify(
        normalised,
        locals={"x": x_sym, "y": y_sym, "pi": sympy.pi, "e": sympy.E},
    )
    return sympy.latex(expr)


def make_target(expression: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compile a validated expression to a torch-friendly target function."""
    normalised = validate_expression(expression)
    code = compile(ast.parse(normalised, mode="eval"), "<kan-expression>", "eval")

    def target_fn(inputs: torch.Tensor) -> torch.Tensor:
        env = {
            **SAFE_FUNCTIONS,
            "x": inputs[:, [0]],
            "y": inputs[:, [1]],
            "pi": torch.tensor(np.pi, dtype=inputs.dtype, device=inputs.device),
            "e": torch.tensor(np.e, dtype=inputs.dtype, device=inputs.device),
        }
        result = eval(code, {"__builtins__": {}}, env)  # noqa: S307 — env is locked down
        if not torch.is_tensor(result):
            result = torch.as_tensor(result, dtype=inputs.dtype, device=inputs.device)
        if result.ndim == 0:
            result = result.expand(inputs.shape[0], 1)
        if result.ndim == 1:
            result = result[:, None]
        if result.shape != (inputs.shape[0], 1):
            raise ValueError("Expression must evaluate to one scalar value per (x, y) point.")
        if not torch.isfinite(result).all():
            raise ValueError("Expression produced non-finite values on the training grid.")
        return result

    target_fn.expression = normalised  # type: ignore[attr-defined]
    return target_fn


try:
    DEFAULT_LATEX = expression_to_latex_body(DEFAULT_EXPRESSION)
except Exception:
    DEFAULT_LATEX = DEFAULT_EXPRESSION


def evaluate_on_grid(
    expression: str, n: int = 48
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate `expression` numerically on an n x n grid over [-1, 1]^2.

    Returns `(xs, ys, zs)` where `xs` and `ys` are 1D length-n arrays and
    `zs` has shape `(n, n)`. Non-finite values become NaN so heatmaps render
    them as transparent gaps; constants broadcast to the full grid.

    Validation is the same AST allowlist used by `make_target`, so this is
    safe to call on raw user input. Used by `/coarse`'s live target preview.
    """
    normalised = validate_expression(expression)
    code = compile(ast.parse(normalised, mode="eval"), "<kan-preview>", "eval")

    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    env = {
        **NUMPY_SAFE_FUNCTIONS,
        "x": xx,
        "y": yy,
        "pi": np.pi,
        "e": np.e,
    }
    with np.errstate(all="ignore"):
        raw = eval(code, {"__builtins__": {}}, env)  # noqa: S307 — env is locked down

    zs = np.broadcast_to(np.asarray(raw, dtype=float), xx.shape).copy()
    zs[~np.isfinite(zs)] = np.nan
    return xs, ys, zs


def _ensure_finite(stage: str, train: list[float], test: list[float]) -> None:
    """Raise a friendly RuntimeError if any loss is NaN or inf.

    LBFGS can diverge at high `lamb` / `lamb_entropy` settings; if we don't
    catch it here the rest of the pipeline propagates NaNs into the diagram
    and the user only sees an opaque matplotlib failure.
    """
    if any(not np.isfinite(v) for v in train + test):
        raise RuntimeError(
            f"Training diverged at the '{stage}' stage (loss became non-finite). "
            "Try a smaller λ or H, or fewer steps."
        )


def _setup() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)


def _dataset(target_fn: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, torch.Tensor]:
    return create_dataset(
        target_fn,
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


def _png_to_data_url(buf: io.BytesIO) -> str:
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


def _render_placeholder(title: str, exc: BaseException) -> str:
    """Fallback PNG shown when pykan's plot can't render this model."""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.text(
        0.5, 0.7,
        f"{title} — diagram unavailable",
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=13, fontweight="bold", color="#475569",
    )
    ax.text(
        0.5, 0.35,
        f"pykan plot raised {exc.__class__.__name__}: {str(exc)[:160]}",
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=9, color="#94a3b8",
    )
    ax.text(
        0.5, 0.12,
        "(loss curve and metrics below are still valid)",
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=8, color="#cbd5e1",
    )
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="white")
    plt.close("all")
    return _png_to_data_url(buf)


def _render_diagram(model: KAN, dataset: dict, title: str) -> str:
    """Render the KAN diagram with a graceful fallback.

    pykan's `plot()` divides by `acts_scale_spline.sum()` to normalise edge
    transparency. After aggressive pruning that sum can be zero, which makes
    alpha NaN and matplotlib then refuses to draw. We try `metric="forward_n"`
    first (informative when it works), fall back to the default metric on
    failure, and finally render a text placeholder if both fail — so a single
    bad configuration never blanks the whole panel.
    """
    plt.close("all")
    model(dataset["train_input"])  # populate forward-pass stats for plot()

    last_exc: BaseException | None = None
    for overrides in ({"metric": "forward_n"}, {}):
        plt.close("all")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                model.plot(
                    folder=tmp,
                    beta=8,
                    scale=0.72,
                    in_vars=[r"$x$", r"$y$"],
                    out_vars=[r"$f(x,y)$"],
                    title=title,
                    **overrides,
                )
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=160, facecolor="white")
                plt.close("all")
                return _png_to_data_url(buf)
        except Exception as exc:  # noqa: BLE001 — every diagram path is best-effort
            last_exc = exc

    assert last_exc is not None
    return _render_placeholder(title, last_exc)


def _losses(history: dict) -> tuple[list[float], list[float]]:
    train = [float(x) for x in history.get("train_loss", [])]
    test = [float(x) for x in history.get("test_loss", [])]
    return train, test


def train_coarse(
    expression: str = DEFAULT_EXPRESSION,
    grid: int = 5,
    steps: int = 20,
) -> dict[str, Any]:
    """One [2, 5, 1] KAN fit on the given target expression."""
    _setup()
    target = make_target(expression)
    dataset = _dataset(target)
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
    expression: str = DEFAULT_EXPRESSION,
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
    target = make_target(expression)
    dataset = _dataset(target)
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
    expression: str = DEFAULT_EXPRESSION,
    coarse_grid: int = 5,
    coarse_steps: int = 20,
    refined_grid: int = 10,
    refined_steps: int = 20,
) -> dict[str, Any]:
    """Coarse fit, then `model.refine()` to a finer spline grid, then continued training."""
    _setup()
    target = make_target(expression)
    dataset = _dataset(target)

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


def _try_prune(model: KAN, ckpt_dir: str) -> KAN:
    """Prune the model; fall back to the unpruned model on failure.

    pykan's `prune()` returns a new model, so we pin its ckpt_path to our
    tempdir the same way we do for refine(). It can raise when no edge
    qualifies for removal, which the upstream demo also tolerates.
    """
    try:
        pruned = model.prune()
        pruned.ckpt_path = ckpt_dir
        return pruned
    except Exception:
        return model


def train_prune(
    expression: str = DEFAULT_EXPRESSION,
    lamb: float = 0.002,
    lamb_entropy: float = 1.0,
    sparse_steps: int = 20,
    prune_steps: int = 15,
) -> dict[str, Any]:
    """Coarse → sparsify → prune → refit.

    Defaults mirror the upstream `kan-fulltext-demo` recipe. Higher λ / H
    push training to diverge (loss → NaN), so we also defensively check
    for finite losses after each `model.fit()`.
    """
    _setup()
    target = make_target(expression)
    dataset = _dataset(target)
    with tempfile.TemporaryDirectory(prefix="kan-ckpt-") as ckpt_dir:
        model = _new_kan(grid=5, ckpt_path=ckpt_dir)

        coarse_history = model.fit(dataset, opt="LBFGS", steps=20, log=10**9)
        coarse_train, coarse_test = _losses(coarse_history)
        _ensure_finite("coarse", coarse_train, coarse_test)

        sparse_history = model.fit(
            dataset,
            opt="LBFGS",
            steps=int(sparse_steps),
            lamb=float(lamb),
            lamb_entropy=float(lamb_entropy),
            log=10**9,
        )
        sparse_train, sparse_test = _losses(sparse_history)
        _ensure_finite("sparsify", sparse_train, sparse_test)

        pruned = _try_prune(model, ckpt_dir)

        prune_history = pruned.fit(
            dataset, opt="LBFGS", steps=int(prune_steps), log=10**9
        )
        prune_train, prune_test = _losses(prune_history)
        _ensure_finite("prune-refit", prune_train, prune_test)

        return {
            "train_loss": coarse_train + sparse_train + prune_train,
            "test_loss": coarse_test + sparse_test + prune_test,
            "split_at_sparse": len(coarse_train),
            "split_at_prune": len(coarse_train) + len(sparse_train),
            "graph_png_b64": _render_diagram(
                pruned, dataset, f"pruned  λ={lamb:.4g}"
            ),
        }


def _symbolic_formula(model: KAN) -> dict[str, str | None]:
    """Run pykan's auto_symbolic + symbolic_formula and format the result.

    Returns a dict with either `latex` (LaTeX-rendered formula) populated or
    `error` (a short error message) populated.
    """
    try:
        lib = ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "cos", "abs"]
        model.auto_symbolic(lib=lib, verbose=0, r2_threshold=0.0)
        formula = model.symbolic_formula()[0][0]
    except Exception as exc:  # noqa: BLE001
        return {"latex": None, "error": f"{exc.__class__.__name__}: {str(exc)[:200]}"}

    try:
        from kan.utils import ex_round  # type: ignore

        formula = ex_round(formula, 4)
    except Exception:
        pass

    try:
        import sympy

        return {"latex": sympy.latex(formula), "error": None}
    except Exception:
        return {"latex": str(formula), "error": None}


def train_symbolic(
    expression: str = DEFAULT_EXPRESSION,
    lamb: float = 0.002,
    lamb_entropy: float = 1.0,
    sparse_steps: int = 25,
) -> dict[str, Any]:
    """Full pipeline: coarse → sparsify → prune → refit → auto-symbolic snap.

    Defaults mirror upstream and avoid the NaN divergence the previous
    aggressive defaults hit.
    """
    _setup()
    target = make_target(expression)
    dataset = _dataset(target)
    with tempfile.TemporaryDirectory(prefix="kan-ckpt-") as ckpt_dir:
        model = _new_kan(grid=5, ckpt_path=ckpt_dir)

        coarse_history = model.fit(dataset, opt="LBFGS", steps=20, log=10**9)
        coarse_train, coarse_test = _losses(coarse_history)
        _ensure_finite("coarse", coarse_train, coarse_test)

        sparse_history = model.fit(
            dataset,
            opt="LBFGS",
            steps=int(sparse_steps),
            lamb=float(lamb),
            lamb_entropy=float(lamb_entropy),
            log=10**9,
        )
        sparse_train, sparse_test = _losses(sparse_history)
        _ensure_finite("sparsify", sparse_train, sparse_test)

        pruned = _try_prune(model, ckpt_dir)
        refit_history = pruned.fit(dataset, opt="LBFGS", steps=15, log=10**9)
        refit_train, refit_test = _losses(refit_history)
        _ensure_finite("prune-refit", refit_train, refit_test)

        formula = _symbolic_formula(pruned)

        return {
            "train_loss": coarse_train + sparse_train + refit_train,
            "test_loss": coarse_test + sparse_test + refit_test,
            "split_at_sparse": len(coarse_train),
            "split_at_prune": len(coarse_train) + len(sparse_train),
            "formula": formula,
            "graph_png_b64": _render_diagram(pruned, dataset, "symbolic snap"),
        }
