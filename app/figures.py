"""Shared Plotly + Dash display helpers for the panels.

Every panel renders the same two things: a stacked loss curve with optional
vertical "stage transition" markers, and a styled traceback panel when the
training callback raises. Both live here so adding a panel is a single new
file under `pages/`, not a tour of boilerplate to copy.
"""

from __future__ import annotations

import traceback
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from dash import html


def loss_figure(
    train: list[float],
    test: list[float],
    splits: Iterable[tuple[int, str]] = (),
) -> go.Figure:
    """Loss curve with optional vertical dashed transitions.

    Args:
        train, test: per-step loss lists (concatenated across stages if multi-stage)
        splits: list of (step_index, label) — a dashed line is drawn before each
            index with the label rendered as a small annotation above it.
    """
    fig = go.Figure()
    if train:
        fig.add_trace(go.Scatter(y=train, name="train", mode="lines+markers"))
    if test:
        fig.add_trace(go.Scatter(y=test, name="test", mode="lines+markers"))

    n = len(train)
    for index, label in splits:
        if 0 < index < n:
            fig.add_vline(x=index - 0.5, line=dict(color="#9ca3af", dash="dash"))
            fig.add_annotation(
                x=index - 0.5,
                y=1,
                yref="paper",
                text=label,
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


def target_surface_figure(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> go.Figure:
    """Heatmap of f(x, y) over a 2-D grid — used by /coarse's live preview."""
    fig = go.Figure(
        data=go.Heatmap(
            z=zs,
            x=xs,
            y=ys,
            colorscale="Viridis",
            zsmooth="best",
            colorbar=dict(thickness=12, len=0.9, outlinewidth=0),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{z:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=10, t=10, b=40),
        height=320,
        xaxis_title="x",
        yaxis=dict(title="y", scaleanchor="x"),
        plot_bgcolor="white",
    )
    return fig


def empty_figure(message: str) -> go.Figure:
    """Plotly figure with no data, just a centered explanatory note."""
    fig = go.Figure()
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=10, t=10, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="#94a3b8", size=12),
            )
        ],
    )
    return fig


def error_panel(exc: BaseException) -> html.Pre:
    """Pretty-print a traceback inline so a failed callback is debuggable."""
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
