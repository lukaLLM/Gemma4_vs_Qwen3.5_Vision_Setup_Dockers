"""Entrypoint for the local multimodal compare lab."""

from __future__ import annotations

import uvicorn

from visual_experimentation_app.api import create_app
from visual_experimentation_app.config import get_settings


def main() -> None:
    """Run the compare lab API + Gradio UI server."""
    settings = get_settings()
    app = create_app(include_ui=True)
    print(
        (
            "[compare-lab] "
            f"Running on http://{settings.host}:{settings.port} "
            f"(api={settings.api_prefix}, ui={settings.ui_path})"
        ),
        flush=True,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
