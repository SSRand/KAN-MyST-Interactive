"""`/symbolic` panel — auto-snap surviving edges to closed-form expressions.

Runs the full pipeline (coarse → sparsify → prune → refit) and then asks
pykan's `auto_symbolic` to identify the closest symbolic form for each
surviving edge. The result is a closed-form approximation of the target
function rendered as inline math.
"""

from __future__ import annotations

import sys
import traceback

import dash
from dash import Input, Output, State, callback, dcc, html

from figures import error_panel, loss_figure, target_input
from kan_core import DEFAULT_EXPRESSION, train_symbolic

dash.register_page(__name__, path="/symbolic", title="Symbolic snap", name="Symbolic snap")


def _formula_block(formula: dict) -> html.Div:
    """Render the discovered formula (LaTeX) or the error message."""
    if formula.get("error"):
        return html.Div(
            [
                html.Strong("symbolic fit failed: "),
                html.Code(formula["error"]),
            ],
            style={"color": "#b45309", "padding": "0.5rem 0"},
        )

    latex = formula.get("latex") or "?"
    return html.Div(
        [
            html.Div(
                "Discovered symbolic approximation",
                style={"fontWeight": 600, "marginBottom": "0.4rem", "color": "#0f172a"},
            ),
            dcc.Markdown(
                f"$$f(x, y) \\approx {latex}$$",
                mathjax=True,
                style={"fontSize": "1.05rem"},
            ),
        ],
        style={
            "background": "#f8fafc",
            "border": "1px solid #e2e8f0",
            "padding": "0.9rem 1.1rem",
            "borderRadius": "8px",
            "marginTop": "1rem",
        },
    )


layout = html.Div(
    style={"maxWidth": "780px", "margin": "0 auto"},
    children=[
        html.H2("Symbolic snap", style={"marginTop": 0}),
        html.P(
            [
                "After pruning, pykan tries to match each surviving edge to a closed-form function from ",
                html.Code("[x, x², x³, exp, log, sqrt, tanh, sin, cos, abs]"),
                "; the recovered formula appears below.",
            ],
            style={"marginBottom": "0.2rem"},
        ),
        html.P(
            [
                "Allowed names in the target expression: ",
                html.Code("sin, cos, tan, tanh, exp, log, sqrt, abs, x, y, pi, e"),
            ],
            style={"color": "#475569", "fontSize": "0.85rem", "marginTop": 0},
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "150px 1fr",
                "rowGap": "0.6rem",
                "columnGap": "1rem",
                "alignItems": "center",
            },
            children=[
                *target_input("symbolic-expr", DEFAULT_EXPRESSION),
                html.Label("λ (overall)"),
                dcc.Slider(
                    id="symbolic-lamb",
                    min=0.001,
                    max=0.02,
                    step=0.001,
                    value=0.005,
                    marks={
                        0.001: "0.001",
                        0.005: "0.005",
                        0.01: "0.01",
                        0.02: "0.02",
                    },
                ),
                html.Label("H (entropy)"),
                dcc.Slider(
                    id="symbolic-entropy",
                    min=0.0,
                    max=3.0,
                    step=0.25,
                    value=2.0,
                    marks={i: f"{i}" for i in (0, 1, 2, 3)},
                ),
                html.Label("Sparse steps"),
                dcc.Slider(
                    id="symbolic-sparse-steps",
                    min=10,
                    max=50,
                    step=5,
                    value=30,
                    marks={i: str(i) for i in (10, 20, 30, 40, 50)},
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "1rem"},
            children=[
                html.Button(
                    "Train",
                    id="symbolic-train",
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
                id="symbolic-output",
                style={"marginTop": "1.2rem"},
                children=html.P(
                    "Click Train to run the full pipeline and attempt symbolic snapping (~30-60 s).",
                    style={"color": "#666", "fontStyle": "italic"},
                ),
            ),
        ),
    ],
)


@callback(
    Output("symbolic-output", "children"),
    Input("symbolic-train", "n_clicks"),
    State("symbolic-expr", "value"),
    State("symbolic-lamb", "value"),
    State("symbolic-entropy", "value"),
    State("symbolic-sparse-steps", "value"),
    prevent_initial_call=True,
)
def _on_train(_n: int, expression: str, lamb: float, lamb_entropy: float, sparse_steps: int):
    print(
        f"[symbolic] expr={expression!r} lamb={lamb} entropy={lamb_entropy} sparse_steps={sparse_steps}",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = train_symbolic(
            expression=expression,
            lamb=lamb,
            lamb_entropy=lamb_entropy,
            sparse_steps=sparse_steps,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return error_panel(exc)

    train_loss = result["train_loss"]
    test_loss = result["test_loss"]
    final_train = train_loss[-1] if train_loss else float("nan")
    final_test = test_loss[-1] if test_loss else float("nan")

    return [
        _formula_block(result["formula"]),
        html.Div(
            style={"display": "flex", "gap": "1.5rem", "fontSize": "0.92rem", "color": "#222", "marginTop": "1rem", "flexWrap": "wrap"},
            children=[
                html.Span([html.Strong("final train loss: "), f"{final_train:.4g}"]),
                html.Span([html.Strong("final test loss: "), f"{final_test:.4g}"]),
                html.Span([html.Strong("λ: "), f"{lamb:.4g}"]),
            ],
        ),
        html.Img(
            src=result["graph_png_b64"],
            alt="symbolic-snapped KAN diagram",
            style={
                "width": "100%",
                "marginTop": "1rem",
                "border": "1px solid #e5e7eb",
                "borderRadius": "8px",
            },
        ),
        dcc.Graph(
            figure=loss_figure(
                train_loss,
                test_loss,
                splits=[
                    (result["split_at_sparse"], "penalty →"),
                    (result["split_at_prune"], "prune →"),
                ],
            ),
            config={"displayModeBar": False},
            style={"marginTop": "0.5rem"},
        ),
    ]
