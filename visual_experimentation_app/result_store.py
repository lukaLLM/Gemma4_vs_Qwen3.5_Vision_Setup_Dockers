"""Persistence helpers for compare-run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visual_experimentation_app.config import get_settings
from visual_experimentation_app.schemas import CompareHistoryItem, CompareResult


def _results_root() -> Path:
    return get_settings().results_dir


def _compares_dir() -> Path:
    return _results_root() / "compares"


def _compare_history_path() -> Path:
    return _results_root() / "compare_history.jsonl"


def ensure_results_layout() -> None:
    """Create compare result directories if they do not yet exist."""
    _results_root().mkdir(parents=True, exist_ok=True)
    _compares_dir().mkdir(parents=True, exist_ok=True)


def save_compare_result(result: CompareResult) -> Path:
    """Persist one compare result as JSON and append to compare history."""
    ensure_results_layout()
    compare_path = _compares_dir() / f"{result.compare_id}.json"
    compare_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    with _compare_history_path().open("a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json())
        handle.write("\n")

    return compare_path


def load_compare_result(compare_id: str) -> CompareResult | None:
    """Load a compare result by ID, if present."""
    compare_path = _compares_dir() / f"{compare_id}.json"
    if not compare_path.exists():
        return None
    payload = json.loads(compare_path.read_text(encoding="utf-8"))
    return CompareResult.model_validate(payload)


def _history_to_item(payload: dict[str, Any]) -> CompareHistoryItem:
    model_a_result = payload.get("model_a_result", {})
    model_b_result = payload.get("model_b_result", {})
    request = payload.get("request", {})
    timings = payload.get("timings", {})
    model_a_request = request.get("model_a", {})
    model_b_request = request.get("model_b", {})

    model_a_effective = model_a_result.get("effective_params", {})
    model_b_effective = model_b_result.get("effective_params", {})

    return CompareHistoryItem(
        compare_id=str(payload.get("compare_id", "")),
        created_at=str(payload.get("created_at", "")),
        status=str(payload.get("status", "error")),  # type: ignore[arg-type]
        model_a_label=str(model_a_result.get("label") or model_a_request.get("label") or ""),
        model_a_model=str(model_a_effective.get("model") or model_a_request.get("model") or ""),
        model_b_label=str(model_b_result.get("label") or model_b_request.get("label") or ""),
        model_b_model=str(model_b_effective.get("model") or model_b_request.get("model") or ""),
        total_ms=float(timings.get("total_ms", 0.0)),
    )


def list_compare_history(*, limit: int = 200) -> list[CompareHistoryItem]:
    """Read compare history entries in reverse chronological order."""
    history_path = _compare_history_path()
    if not history_path.exists():
        return []

    lines = history_path.read_text(encoding="utf-8").splitlines()
    items: list[CompareHistoryItem] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        payload = json.loads(line)
        items.append(_history_to_item(payload))
        if len(items) >= limit:
            break
    return items
