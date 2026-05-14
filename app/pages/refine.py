"""`/refine` panel — grid refinement after a coarse fit.

The panel trains a coarse `[2, 5, 1]` KAN, then upsamples its spline grid via
`model.refine()` and continues training. The loss curve shows both stages
back to back; a dashed line marks the refinement transition.
"""

from __future__ import annotations

import sys
import traceback

import dash
from dash import Input, Output, State, callback, dcc, html

from figures import error_panel, loss_figure
from kan_core import DEFAULT_EXPRESSION, train_refine

dash.register_page(__name__, path="/refine", title="Grid refinement", name="Grid refinement")


layout = html.Div(
    style={"maxWidth": "780px", "margin": "0 auto"},
    children=[
        html.H2("Grid refinement", style={"marginTop": 0}),
        html.P(
            [
                "Target: ",
                html.Code(f"f(x, y) = {DEFAULT_EXPRESSION}"),
                ". The panel first trains a coarse KAN, then refines its spline grid and continues.",
            ]
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "140px 1fr",
                "rowGap": "0.6rem",
                "columnGap": "1rem",
                "alignItems": "center",
            },
            children=[
                html.Label("Coarse grid"),
                dcc.Slider(
                    id="refine-coarse-grid",
                    min=3,
                    max=15,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in (3, 5, 10, 15)},
                ),
                html.Label("Refined grid"),
                dcc.Slider(
                    id="refine-refined-grid",
                    min=5,
                    max=30,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in (5, 10, 15, 20, 25, 30)},
                ),
                html.Label("Steps each"),
                dcc.Slider(
                    id="refine-steps",
                    min=5,
                    max=30,
                    step=5,
                    value=15,
                    marks={i: str(i) for i in (5, 10, 15, 20, 25, 30)},
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "1rem"},
            children=[
                html.Button(
                    "Train",
                    id="refine-train",
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
                id="refine-output",
                style={"marginTop": "1.2rem"},
                children=html.P(
                    "Click Train to fit a KAN and refine its spline grid.",
                    style={"color": "#666", "fontStyle": "italic"},
                ),
            ),
        ),
    ],
)


@callback(
    Output("refine-output", "children"),
    Input("refine-train", "n_clicks"),
    State("refine-coarse-grid", "value"),
    State("refine-refined-grid", "value"),
    State("refine-steps", "value"),
    prevent_initial_call=True,
)
def _on_train(_n: int, coarse_grid: int, refined_grid: int, steps: int):
    print(
        f"[refine] coarse_grid={coarse_grid} refined_grid={refined_grid} steps={steps}",
        file=sys.stderr,
        flush=True,
    )
    if refined_grid <= coarse_grid:
        return html.Div(
            f"Refined grid ({refined_grid}) must be strictly greater than coarse grid ({coarse_grid}).",
            style={"color": "#b91c1c", "padding": "0.5rem 0"},
        )

    try:
        result = train_refine(
            coarse_grid=coarse_grid,
            coarse_steps=steps,
            refined_grid=refined_grid,
            refined_steps=steps,
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
            style={"display": "flex", "gap": "1.5rem", "fontSize": "0.92rem", "color": "#222"},
            children=[
                html.Span([html.Strong("final train loss: "), f"{final_train:.4g}"]),
                html.Span([html.Strong("final test loss: "), f"{final_test:.4g}"]),
                html.Span([html.Strong("grid: "), f"{coarse_grid} → {refined_grid}"]),
            ],
        ),
        html.Img(
            src=result["graph_png_b64"],
            alt="refined KAN diagram",
            style={
                "width": "100%",
                "marginTop": "1rem",
                "border": "1px solid #e5e7eb",
                "borderRadius": "8px",
            },
        ),
        dcc.Graph(
            figure=loss_figure(train_loss, test_loss, splits=[(result["split_at"], "refine →")]),
            config={"displayModeBar": False},
            style={"marginTop": "0.5rem"},
        ),
    ]
