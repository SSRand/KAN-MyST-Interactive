"""Landing route. Lets someone visiting the bare app URL find the panels.

Not embedded by paper.md — only /coarse is iframed today. This page exists so
that http://localhost:8050/ doesn't 404 during development.
"""

import dash
from dash import dcc, html

dash.register_page(__name__, path="/", title="KAN dashboard", name="Home")

layout = html.Div(
    [
        html.H1("KAN dashboard"),
        html.P(
            "This is the dashboard backend for the KAN MyST paper. "
            "Each route is one focused interactive panel; the paper iframes them "
            "individually next to the relevant prose."
        ),
        html.H3("Panels"),
        html.Ul(
            [
                html.Li(
                    dcc.Link(
                        "/coarse — coarse KAN fit on a fixed target",
                        href="/coarse",
                    )
                ),
                html.Li(
                    html.Span(
                        "/refine, /sparsify, /prune, /symbolic — planned (v1)",
                        style={"color": "#777"},
                    )
                ),
            ]
        ),
    ]
)
