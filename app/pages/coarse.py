"""`/coarse` panel — minimum-viable interactive KAN fit.

Visible knobs: spline grid size, LBFGS step count. One button triggers a fresh
training run; the result is a KAN graph image plus a loss curve.

This page is designed to render fine inside an `<iframe>` — no top nav,
self-contained controls, and a fixed maximum width so it doesn't overflow the
host article column.
"""

from __future__ import annotations

import sys
import traceback

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from kan_core import DEFAULT_EXPRESSION, train_coarse

dash.register_page(__name__, path="/coarse", title="Coarse KAN fit", name="Coarse fit")


def _loss_figure(train: list[float], test: list[float]) -> go.Figure:
    fig = go.Figure()
    if train:
        fig.add_trace(go.Scatter(y=train, name="train", mode="lines+markers"))
    if test:
        fig.add_trace(go.Scatter(y=test, name="test", mode="lines+markers"))
    fig.update_layout(
        xaxis_title="step",
        yaxis_title="loss",
        yaxis_type="log",
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
        legend=dict(orientation="h", y=1.1, x=0),
    )
    return fig


layout = html.Div(
    style={"maxWidth": "780px", "margin": "0 auto"},
    children=[
        html.H2("Coarse KAN fit", style={"marginTop": 0}),
        html.P(
            [
                "Target: ",
                html.Code(f"f(x, y) = {DEFAULT_EXPRESSION}"),
                ". The slider sets the spline grid resolution; the step slider sets the LBFGS iteration budget.",
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
                    "Click Train to fit a KAN with the chosen grid and step count.",
                    style={"color": "#666", "fontStyle": "italic"},
                ),
            ),
        ),
    ],
)


@callback(
    Output("coarse-output", "children"),
    Input("coarse-train", "n_clicks"),
    State("coarse-grid", "value"),
    State("coarse-steps", "value"),
    prevent_initial_call=True,
)
def _on_train(_n_clicks: int, grid: int, steps: int):
    print(f"[coarse] grid={grid} steps={steps}", file=sys.stderr, flush=True)
    try:
        result = train_coarse(grid=grid, steps=steps)
    except Exception as exc:  # noqa: BLE001 — we want every error visible
        traceback.print_exc(file=sys.stderr)
        return html.Pre(
            f"{exc.__class__.__name__}: {exc}\n\n{traceback.format_exc()}",
            style={
                "color": "#b91c1c",
                "background": "#fef2f2",
                "padding": "1rem",
                "borderRadius": "6px",
                "border": "1px solid #fecaca",
                "fontSize": "0.78rem",
                "overflow": "auto",
                "whiteSpace": "pre-wrap",
            },
        )
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
            figure=_loss_figure(train_loss, test_loss),
            config={"displayModeBar": False},
            style={"marginTop": "0.5rem"},
        ),
    ]
