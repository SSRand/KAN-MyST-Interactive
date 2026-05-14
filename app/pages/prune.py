"""`/prune` panel — sparsify, prune low-magnitude edges, refit.

The pipeline is coarse fit → fit under L1 + entropy → `model.prune()` →
short refit on the surviving topology. The KAN diagram afterwards is a
strictly smaller network than the original `[2, 5, 1]`; the loss curve has
two dashed transitions (penalty on, prune).
"""

from __future__ import annotations

import sys
import traceback

import dash
from dash import Input, Output, State, callback, dcc, html

from figures import error_panel, loss_figure
from kan_core import DEFAULT_LATEX, train_prune

dash.register_page(__name__, path="/prune", title="Pruning", name="Pruning")


layout = html.Div(
    style={"maxWidth": "780px", "margin": "0 auto"},
    children=[
        html.H2("Pruning", style={"marginTop": 0}),
        html.P(
            "Sparsification thins the edges; pruning removes the weak ones structurally, then a short refit recovers any accuracy lost.",
            style={"marginBottom": "0.6rem"},
        ),
        dcc.Markdown(
            f"**Target:** $f(x, y) = {DEFAULT_LATEX}$",
            mathjax=True,
            style={"color": "#475569", "marginBottom": "1rem"},
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
                html.Label("λ (overall)"),
                dcc.Slider(
                    id="prune-lamb",
                    min=0.001,
                    max=0.02,
                    step=0.001,
                    value=0.002,
                    marks={
                        0.001: "0.001",
                        0.002: "0.002",
                        0.005: "0.005",
                        0.01: "0.01",
                        0.02: "0.02",
                    },
                ),
                html.Label("H (entropy)"),
                dcc.Slider(
                    id="prune-entropy",
                    min=0.0,
                    max=3.0,
                    step=0.25,
                    value=1.0,
                    marks={i: f"{i}" for i in (0, 1, 2, 3)},
                ),
                html.Label("Sparse steps"),
                dcc.Slider(
                    id="prune-sparse-steps",
                    min=10,
                    max=40,
                    step=5,
                    value=20,
                    marks={i: str(i) for i in (10, 20, 30, 40)},
                ),
                html.Label("Refit steps"),
                dcc.Slider(
                    id="prune-refit-steps",
                    min=5,
                    max=30,
                    step=5,
                    value=15,
                    marks={i: str(i) for i in (5, 10, 15, 20, 30)},
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "1rem"},
            children=[
                html.Button(
                    "Train",
                    id="prune-train",
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
                id="prune-output",
                style={"marginTop": "1.2rem"},
                children=html.P(
                    "Click Train to run the full coarse → sparsify → prune → refit pipeline (~20-30 s).",
                    style={"color": "#666", "fontStyle": "italic"},
                ),
            ),
        ),
    ],
)


@callback(
    Output("prune-output", "children"),
    Input("prune-train", "n_clicks"),
    State("prune-lamb", "value"),
    State("prune-entropy", "value"),
    State("prune-sparse-steps", "value"),
    State("prune-refit-steps", "value"),
    prevent_initial_call=True,
)
def _on_train(_n: int, lamb: float, lamb_entropy: float, sparse_steps: int, refit_steps: int):
    print(
        f"[prune] lamb={lamb} entropy={lamb_entropy} sparse_steps={sparse_steps} refit_steps={refit_steps}",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = train_prune(
            lamb=lamb,
            lamb_entropy=lamb_entropy,
            sparse_steps=sparse_steps,
            prune_steps=refit_steps,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return error_panel(exc)

    train_loss = result["train_loss"]
    test_loss = result["test_loss"]
    final_train = train_loss[-1] if train_loss else float("nan")
    final_test = test_loss[-1] if test_loss else float("nan")

    return [
        html.Div(
            style={"display": "flex", "gap": "1.5rem", "fontSize": "0.92rem", "color": "#222", "flexWrap": "wrap"},
            children=[
                html.Span([html.Strong("final train loss: "), f"{final_train:.4g}"]),
                html.Span([html.Strong("final test loss: "), f"{final_test:.4g}"]),
                html.Span([html.Strong("λ: "), f"{lamb:.4g}"]),
            ],
        ),
        html.Img(
            src=result["graph_png_b64"],
            alt="pruned KAN diagram",
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
