"""Shared compare-run orchestration for API and UI callers."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from visual_experimentation_app.result_store import save_compare_result
from visual_experimentation_app.schemas import CompareRequest, CompareResult, CompareTiming
from visual_experimentation_app.vllm_client import execute_compare


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def execute_and_persist_compare(request: CompareRequest) -> CompareResult:
    """Execute one compare request, persist it, and return the stored result."""
    compare_id = f"compare_{uuid4().hex}"
    execution = execute_compare(request)
    result = CompareResult(
        compare_id=compare_id,
        status=execution.status,
        created_at=_utc_now_iso(),
        request=request,
        timings=CompareTiming(
            preprocess_ms=execution.preprocess_ms,
            total_ms=execution.total_ms,
        ),
        media_metadata=execution.media_metadata,
        model_a_result=execution.model_a_result,
        model_b_result=execution.model_b_result,
    )
    save_compare_result(result)
    return result
