"""Tests for compare-lab API routes."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.api import compare, compare_detail, compares, health  # noqa: E402
from visual_experimentation_app.config import clear_settings_cache  # noqa: E402
from visual_experimentation_app.result_store import save_compare_result  # noqa: E402
from visual_experimentation_app.schemas import (  # noqa: E402
    CompareRequest,
    CompareResult,
    CompareTargetResult,
    CompareTiming,
    RunTiming,
    TokenUsageStats,
)


def _build_compare_result(*, compare_id: str = "compare_test") -> CompareResult:
    request = CompareRequest(prompt="hello")
    return CompareResult(
        compare_id=compare_id,
        status="ok",
        created_at="2026-04-12T00:00:00+00:00",
        request=request,
        timings=CompareTiming(preprocess_ms=10.0, total_ms=50.0),
        media_metadata={"images": [], "videos": [], "video": None},
        model_a_result=CompareTargetResult(
            label="Model A",
            status="ok",
            output_text="alpha",
            error=None,
            timings=RunTiming(preprocess_ms=0.0, request_ms=20.0, total_ms=20.0, ttft_ms=5.0),
            token_usage=TokenUsageStats(prompt_tokens=10, output_tokens=20, total_tokens=30),
            effective_params={"model": "model-a", "base_url": "http://127.0.0.1:8000/v1"},
        ),
        model_b_result=CompareTargetResult(
            label="Model B",
            status="ok",
            output_text="beta",
            error=None,
            timings=RunTiming(preprocess_ms=0.0, request_ms=25.0, total_ms=25.0, ttft_ms=6.0),
            token_usage=TokenUsageStats(prompt_tokens=11, output_tokens=21, total_tokens=32),
            effective_params={"model": "model-b", "base_url": "http://127.0.0.1:8001/v1"},
        ),
    )


class CompareLabApiTest(unittest.TestCase):
    """Covers core local API behavior."""

    def setUp(self) -> None:
        """Build isolated temp results storage."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(
            os.environ,
            {"MM_LAB_RESULTS_DIR": self.tmp_dir.name},
            clear=False,
        )
        self.env_patch.start()
        clear_settings_cache()

    def tearDown(self) -> None:
        """Release temp resources and clear cached settings."""
        self.env_patch.stop()
        self.tmp_dir.cleanup()
        clear_settings_cache()

    def test_health(self) -> None:
        """Health endpoint returns compare API metadata and defaults."""
        payload = health()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model_a_default"]["model"], "Qwen/Qwen3.5-27B-FP8")
        self.assertEqual(
            payload["model_a_default"]["request_defaults"]["max_tokens"],
            10000,
        )
        self.assertEqual(
            payload["model_a_default"]["request_defaults"]["top_k"],
            20,
        )
        self.assertEqual(
            payload["model_b_default"]["model"],
            "RedHatAI/gemma-4-31B-it-FP8-block",
        )
        self.assertEqual(
            payload["model_b_default"]["request_defaults"]["max_completion_tokens"],
            9500,
        )
        self.assertEqual(
            payload["model_b_default"]["request_defaults"]["top_k"],
            64,
        )

    def test_compare_success(self) -> None:
        """Compare endpoint returns successful payload when service succeeds."""
        result = _build_compare_result()
        with mock.patch(
            "visual_experimentation_app.api.execute_and_persist_compare",
            return_value=result,
        ):
            payload = compare(CompareRequest(prompt="hello"))

        self.assertEqual(payload.status, "ok")
        self.assertEqual(payload.model_a_result.output_text, "alpha")
        self.assertEqual(payload.model_b_result.output_text, "beta")

    def test_compare_detail_404(self) -> None:
        """Compare detail endpoint returns 404 for unknown compare IDs."""
        with self.assertRaises(HTTPException) as exc_info:
            compare_detail("not_found")
        self.assertEqual(exc_info.exception.status_code, 404)

    def test_compares_list_and_detail_use_persisted_results(self) -> None:
        """Compare history and detail routes should read persisted compare artifacts."""
        result = _build_compare_result(compare_id="compare_saved")
        save_compare_result(result)

        items = compares()
        self.assertEqual(items[0].compare_id, "compare_saved")
        self.assertEqual(items[0].model_a_label, "Model A")
        self.assertEqual(items[0].model_b_label, "Model B")

        payload = compare_detail("compare_saved")
        self.assertEqual(payload.compare_id, "compare_saved")
        self.assertEqual(payload.model_a_result.output_text, "alpha")


if __name__ == "__main__":
    unittest.main()
