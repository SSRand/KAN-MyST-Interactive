"""`/coarse` panel — the entry point of the training pipeline.

This is the only panel where the target function is editable. The expression
is typed into a text field whose contents are rendered as MathJax in real
time, then the Train button fits a `[2, 5, 1]` KAN at the chosen grid /
LBFGS steps and shows the diagram + loss curve.

Subsequent panels (`/refine`, `/sparsify`, `/prune`, `/symbolic`) do not let
the reader change the target; they train on the same default expression.
That mirrors how a printed paper would read: the target is declared once.
"""

from __future__ import annotations

import sys
import traceback

import dash
from dash import Input, Output, State, callback, dcc, html

from figures import empty_figure, error_panel, loss_figure, target_surface_figure
from kan_core import (
    DEFAULT_EXPRESSION,
    evaluate_on_grid,
    expression_to_latex_body,
    train_coarse,
)

dash.register_page(__name__, path="/coarse", title="Coarse KAN fit", name="Coarse fit")


_INPUT_STYLE = {
    "width": "100%",
    "padding": "0.45rem 0.7rem",
    "fontFamily": "ui-monospace, SFMono-Regular, Menlo, monospace",
    "fontSize": "0.92rem",
    "borderRadius": "5px",
    "border": "1px solid #cbd5e1",
    "boxSizing": "border-box",
}


layout = html.Div(
    style={"maxWidth": "780px", "margin": "0 auto"},
    children=[
        html.H2("Coarse KAN fit", style={"marginTop": 0}),
        html.P(
            "Edit the target function and click Train. Allowed names: ",
            style={"marginBottom": "0.2rem"},
        ),
        html.P(
            html.Code("sin, cos, tan, tanh, exp, log, sqrt, abs, x, y, pi, e"),
            style={"color": "#475569", "fontSize": "0.85rem", "marginTop": 0},
        ),
        # Target block: label, editable input, live LaTeX preview.
        html.Div(
            style={"marginTop": "1rem", "marginBottom": "1.2rem"},
            children=[
                html.Label(
                    "Target  f(x, y)",
                    style={"display": "block", "marginBottom": "0.4rem", "fontWeight": 500},
                ),
                dcc.Input(
                    id="coarse-expr",
                    type="text",
                    value=DEFAULT_EXPRESSION,
                    placeholder="exp(sin(pi*x) + y**2)",
                    debounce=False,  # update on every keystroke for live render
                    style=_INPUT_STYLE,
                ),
                dcc.Markdown(
                    id="coarse-expr-rendered",
                    mathjax=True,
                    style={
                        "marginTop": "0.55rem",
                        "minHeight": "1.6rem",
                        "color": "#0f172a",
                        "fontSize": "1.02rem",
                    },
                ),
                dcc.Graph(
                    id="coarse-expr-surface",
                    figure=empty_figure("(target preview)"),
                    config={"displayModeBar": False},
                    style={"marginTop": "0.4rem"},
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "120px 1fr", "rowGap": "0.6rem", "columnGap": "1rem", "alignItems": "center"},
            children=[
                html.Label("Grid"),
                dcc.Slider(
                    id="coarse-grid",
                    min=3,
                    max=20,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in (3, 5, 10, 15, 20)},
                ),
                html.Label("LBFGS steps"),
                dcc.Slider(
                    id="coarse-steps",
                    min=5,
                    max=50,
                    step=5,
                    value=20,
                    marks={i: str(i) for i in (5, 10, 20, 30, 40, 50)},
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "1rem"},
            children=[
                html.Button(
                    "Train",
                    id="coarse-train",
                    n_clicks=0,
                    style={
                        "padding": "0.5rem 1.4rem",
                        "borderRadius": "6px",
                        "border": "0",
                        "background": "#2563eb",
                        "color": "white",
                        "cursor": "pointer",
                        "fontWeight": 600,
                    },
                ),
            ],
        ),
        dcc.Loading(
            type="default",
            children=html.Div(
                id="coarse-output",
                style={"marginTop": "1.2rem"},
                children=html.P(
                    "Click Train to fit a KAN with the chosen target, grid, and step count.",
                    style={"color": "#666", "fontStyle": "italic"},
                ),
            ),
        ),
    ],
)


@callback(
    Output("coarse-expr-rendered", "children"),
    Output("coarse-expr-surface", "figure"),
    Input("coarse-expr", "value"),
)
def _live_preview(expression: str | None):
    """Re-render the LaTeX form and the 2-D heatmap on every keystroke."""
    if not expression or not expression.strip():
        return "", empty_figure("Type a target expression to preview.")

    try:
        latex = f"$$f(x, y) = {expression_to_latex_body(expression)}$$"
    except ValueError as exc:
        latex = f"⚠️ {exc}"
    except Exception as exc:  # noqa: BLE001 — sympy can raise many shapes
        latex = f"⚠️ {exc.__class__.__name__}: {exc}"

    try:
        xs, ys, zs = evaluate_on_grid(expression)
        figure = target_surface_figure(xs, ys, zs)
    except ValueError as exc:
        figure = empty_figure(str(exc))
    except Exception as exc:  # noqa: BLE001
        figure = empty_figure(f"{exc.__class__.__name__}: {exc}")

    return latex, figure


@callback(
    Output("coarse-output", "children"),
    Input("coarse-train", "n_clicks"),
    State("coarse-expr", "value"),
    State("coarse-grid", "value"),
    State("coarse-steps", "value"),
    prevent_initial_call=True,
)
def _on_train(_n_clicks: int, expression: str, grid: int, steps: int):
    print(f"[coarse] expr={expression!r} grid={grid} steps={steps}", file=sys.stderr, flush=True)
    try:
        result = train_coarse(expression=expression, grid=grid, steps=steps)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return error_panel(exc)

    train_loss = result["train_loss"]
    test_loss = result["test_loss"]
    final_train = train_loss[-1] if train_loss else float("nan")
    final_test = test_loss[-1] if test_loss else float("nan")

    return [
        html.Div(
            style={"display": "flex", "gap": "1.5rem", "fontSize": "0.92rem", "color": "#222"},
            children=[
                html.Span([html.Strong("final train loss: "), f"{final_train:.4g}"]),
                html.Span([html.Strong("final test loss: "), f"{final_test:.4g}"]),
                html.Span([html.Strong("grid: "), str(grid)]),
                html.Span([html.Strong("steps: "), str(steps)]),
            ],
        ),
        html.Img(
            src=result["graph_png_b64"],
            alt="KAN learned-edge diagram",
            style={"width": "100%", "marginTop": "1rem", "border": "1px solid #e5e7eb", "borderRadius": "8px"},
        ),
        dcc.Graph(
            figure=loss_figure(train_loss, test_loss),
            config={"displayModeBar": False},
            style={"marginTop": "0.5rem"},
        ),
    ]
