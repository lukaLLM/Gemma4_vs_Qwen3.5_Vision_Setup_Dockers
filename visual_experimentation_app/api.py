"""FastAPI routes and app factory for the multimodal compare lab."""

from __future__ import annotations

from dataclasses import asdict
from typing import cast

import gradio as gr
from fastapi import APIRouter, FastAPI, HTTPException

from visual_experimentation_app.compare_service import execute_and_persist_compare
from visual_experimentation_app.config import get_settings
from visual_experimentation_app.result_store import (
    ensure_results_layout,
    list_compare_history,
    load_compare_result,
)
from visual_experimentation_app.schemas import CompareHistoryItem, CompareRequest, CompareResult

router = APIRouter()


@router.get("/health")
def health() -> dict[str, object]:
    """Basic health endpoint for local tooling and smoke checks."""
    settings = get_settings()
    ensure_results_layout()
    return {
        "status": "ok",
        "api_prefix": settings.api_prefix,
        "ui_path": settings.ui_path,
        "model_a_default": {
            "label": settings.model_a.label,
            "base_url": settings.model_a.base_url,
            "model": settings.model_a.model,
            "request_defaults": asdict(settings.model_a.request_defaults),
        },
        "model_b_default": {
            "label": settings.model_b.label,
            "base_url": settings.model_b.base_url,
            "model": settings.model_b.model,
            "request_defaults": asdict(settings.model_b.request_defaults),
        },
        "results_dir": str(settings.results_dir),
    }


@router.post("/compare", response_model=CompareResult)
def compare(request: CompareRequest) -> CompareResult:
    """Execute one compare run and persist its result."""
    return execute_and_persist_compare(request)


@router.get("/compares", response_model=list[CompareHistoryItem])
def compares(limit: int = 200) -> list[CompareHistoryItem]:
    """List recent persisted compare summaries."""
    clean_limit = max(1, min(limit, 1000))
    return list_compare_history(limit=clean_limit)


@router.get("/compares/{compare_id}", response_model=CompareResult)
def compare_detail(compare_id: str) -> CompareResult:
    """Fetch one persisted compare by ID."""
    result = load_compare_result(compare_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown compare_id: {compare_id}")
    return result


def create_app(*, include_ui: bool = True) -> FastAPI:
    """Create API app and optionally mount Gradio UI."""
    settings = get_settings()
    ensure_results_layout()

    app = FastAPI(title="Multimodal Compare Lab", version="0.1.0")
    app.include_router(router, prefix=settings.api_prefix)

    if (not include_ui) or settings.ui_path != "/":

        @app.get("/")
        def root() -> dict[str, str]:
            return {
                "status": "ok",
                "api": settings.api_prefix,
                "ui": settings.ui_path if include_ui else "",
            }

    if not include_ui:
        return app

    from visual_experimentation_app.ui import build_ui_blocks, ui_css, ui_head, ui_theme

    blocks = build_ui_blocks()
    return cast(
        FastAPI,
        gr.mount_gradio_app(
            app,
            blocks,
            path=settings.ui_path,
            theme=ui_theme(),
            css=ui_css(),
            head=ui_head(),
        ),
    )
