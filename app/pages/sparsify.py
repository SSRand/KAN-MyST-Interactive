"""`/sparsify` panel — sparsification under L1 + entropy penalties.

A coarse `[2, 5, 1]` KAN is fit normally for a baseline, then training
continues with `lamb` (overall regularisation weight) and `lamb_entropy`
(entropy penalty). The combined pressure drives unused edges toward zero —
the visible effect on `model.plot()` is a noticeably thinner diagram.
"""

from __future__ import annotations

import sys
import traceback

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from kan_core import DEFAULT_EXPRESSION, train_sparsify

dash.register_page(__name__, path="/sparsify", title="Sparsification", name="Sparsification")


def _loss_figure(train: list[float], test: list[float], split_at: int) -> go.Figure:
    fig = go.Figure()
    if train:
        fig.add_trace(go.Scatter(y=train, name="train", mode="lines+markers"))
    if test:
        fig.add_trace(go.Scatter(y=test, name="test", mode="lines+markers"))
    if split_at and 0 < split_at < len(train):
        fig.add_vline(x=split_at - 0.5, line=dict(color="#9ca3af", dash="dash"))
        fig.add_annotation(
            x=split_at - 0.5,
            y=1,
            yref="paper",
            text="penalty on →",
            showarrow=False,
            xshift=8,
            font=dict(size=11, color="#6b7280"),
        )
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
        html.H2("Sparsification", style={"marginTop": 0}),
        html.P(
            [
                "Target: ",
                html.Code(f"f(x, y) = {DEFAULT_EXPRESSION}"),
                ". A coarse fit converges, then training continues with an L1 + entropy penalty so weak edges fade out.",
            ]
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
                    id="sparsify-lamb",
                    min=0.0,
                    max=0.01,
                    step=0.0005,
                    value=0.002,
                    marks={
                        0: "0",
                        0.002: "0.002",
                        0.005: "0.005",
                        0.01: "0.01",
                    },
                ),
                html.Label("H (entropy)"),
                dcc.Slider(
                    id="sparsify-entropy",
                    min=0.0,
                    max=3.0,
                    step=0.25,
                    value=1.0,
                    marks={i: f"{i}" for i in (0, 1, 2, 3)},
                ),
                html.Label("Sparse steps"),
                dcc.Slider(
                    id="sparsify-steps",
                    min=5,
                    max=40,
                    step=5,
                    value=20,
                    marks={i: str(i) for i in (5, 10, 20, 30, 40)},
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "1rem"},
            children=[
                html.Button(
                    "Train",
                    id="sparsify-train",
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
                id="sparsify-output",
                style={"marginTop": "1.2rem"},
                children=html.P(
                    "Click Train to fit a coarse KAN and then continue under the penalty.",
                    style={"color": "#666", "fontStyle": "italic"},
                ),
            ),
        ),
    ],
)


@callback(
    Output("sparsify-output", "children"),
    Input("sparsify-train", "n_clicks"),
    State("sparsify-lamb", "value"),
    State("sparsify-entropy", "value"),
    State("sparsify-steps", "value"),
    prevent_initial_call=True,
)
def _on_train(_n: int, lamb: float, lamb_entropy: float, sparse_steps: int):
    print(
        f"[sparsify] lamb={lamb} lamb_entropy={lamb_entropy} sparse_steps={sparse_steps}",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = train_sparsify(
            coarse_steps=20,
            lamb=lamb,
            lamb_entropy=lamb_entropy,
            sparse_steps=sparse_steps,
        )
    except Exception as exc:  # noqa: BLE001 — every error visible
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
                html.Span([html.Strong("λ: "), f"{lamb:.4g}"]),
                html.Span([html.Strong("H: "), f"{lamb_entropy:g}"]),
            ],
        ),
        html.Img(
            src=result["graph_png_b64"],
            alt="sparsified KAN diagram",
            style={
                "width": "100%",
                "marginTop": "1rem",
                "border": "1px solid #e5e7eb",
                "borderRadius": "8px",
            },
        ),
        dcc.Graph(
            figure=_loss_figure(train_loss, test_loss, result["split_at"]),
            config={"displayModeBar": False},
            style={"marginTop": "0.5rem"},
        ),
    ]
