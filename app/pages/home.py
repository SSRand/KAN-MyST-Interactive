"""Landing route for the dashboard. Lists the available panels.

Not embedded by paper.md — only the individual panel URLs are iframed.
"""

import dash
from dash import dcc, html

dash.register_page(__name__, path="/", title="KAN dashboard", name="Home")

layout = html.Div(
    style={"maxWidth": "640px", "margin": "0 auto"},
    children=[
        html.H1("KAN dashboard", style={"marginTop": 0}),
        html.P("Each route is one focused interactive panel from the KAN paper."),
        html.Ul(
            [
                html.Li(dcc.Link("/coarse — coarse KAN fit", href="/coarse")),
                html.Li(dcc.Link("/refine — grid refinement", href="/refine")),
            ]
        ),
    ],
)
