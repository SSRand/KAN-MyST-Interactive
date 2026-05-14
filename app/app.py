"""KAN dashboard — multi-page Dash app entry point.

Each interactive panel of the KAN paper lives at its own URL under
`pages/`. The MyST article in the parent directory embeds those URLs via
`:::{iframe}` directives. The app is otherwise plain Dash; the routing layer
is "what file is in the `pages/` folder".
"""

from __future__ import annotations

import os

import dash
from dash import html

app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

# Expose the Flask app for production WSGI servers (gunicorn, uWSGI).
server = app.server

app.layout = html.Div(
    [dash.page_container],
    style={
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "padding": "1.5rem",
        "color": "#0f172a",
    },
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=False)
