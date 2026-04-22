"""Tests for compare result persistence."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.config import clear_settings_cache  # noqa: E402
from visual_experimentation_app.result_store import (  # noqa: E402
    list_compare_history,
    load_compare_result,
    save_compare_result,
)
from visual_experimentation_app.schemas import (  # noqa: E402
    CompareRequest,
    CompareResult,
    CompareTargetResult,
    CompareTiming,
    RunTiming,
    TokenUsageStats,
)


def _build_compare_result(*, compare_id: str = "compare_store") -> CompareResult:
    return CompareResult(
        compare_id=compare_id,
        status="partial",
        created_at="2026-04-12T00:00:00+00:00",
        request=CompareRequest(prompt="hello"),
        timings=CompareTiming(preprocess_ms=12.0, total_ms=75.0),
        media_metadata={"images": [], "videos": [], "video": None},
        model_a_result=CompareTargetResult(
            label="Qwen",
            status="ok",
            output_text="alpha",
            error=None,
            timings=RunTiming(preprocess_ms=0.0, request_ms=30.0, total_ms=30.0, ttft_ms=8.0),
            token_usage=TokenUsageStats(prompt_tokens=10, output_tokens=20, total_tokens=30),
            effective_params={"model": "Qwen/Qwen3.5-27B-FP8", "base_url": "http://127.0.0.1:8000/v1"},
        ),
        model_b_result=CompareTargetResult(
            label="Gemma",
            status="error",
            output_text="",
            error="boom",
            timings=RunTiming(preprocess_ms=0.0, request_ms=28.0, total_ms=28.0, ttft_ms=None),
            token_usage=TokenUsageStats(),
            effective_params={
                "model": "RedHatAI/gemma-4-31B-it-FP8-block",
                "base_url": "http://127.0.0.1:8001/v1",
            },
        ),
    )


class ResultStoreCompareArtifactsTest(unittest.TestCase):
    """Covers compare JSON artifact persistence and history listing."""

    def setUp(self) -> None:
        """Create isolated temp results directory per test."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(
            os.environ,
            {"MM_LAB_RESULTS_DIR": self.tmp_dir.name},
            clear=False,
        )
        self.env_patch.start()
        clear_settings_cache()

    def tearDown(self) -> None:
        """Release test resources and clear settings cache."""
        self.env_patch.stop()
        self.tmp_dir.cleanup()
        clear_settings_cache()

    def test_save_compare_result_writes_json_and_history(self) -> None:
        """Persisted compare artifacts should be readable from disk and history."""
        result = _build_compare_result()

        path = save_compare_result(result)

        self.assertTrue(Path(path).exists())
        loaded = load_compare_result("compare_store")
        assert loaded is not None
        self.assertEqual(loaded.compare_id, "compare_store")
        self.assertEqual(loaded.model_a_result.output_text, "alpha")
        self.assertEqual(loaded.model_b_result.error, "boom")

        history = list_compare_history(limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].compare_id, "compare_store")
        self.assertEqual(history[0].model_a_label, "Qwen")
        self.assertEqual(history[0].model_b_label, "Gemma")
        self.assertEqual(history[0].status, "partial")


if __name__ == "__main__":
    unittest.main()
