"""Tests for compare request schema normalization and limits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.schemas import CompareRequest, CompareTargetConfig  # noqa: E402


class CompareRequestSchemaTest(unittest.TestCase):
    """Covers compare-request normalization and per-target defaults mode."""

    def test_legacy_video_fields_are_normalized(self) -> None:
        """Legacy singular video fields should still populate plural fields."""
        request = CompareRequest(
            prompt="hello",
            video_path="/tmp/a.mp4",
            video_cache_uuid="vid-a",
        )
        self.assertEqual(request.video_paths, ["/tmp/a.mp4"])
        self.assertEqual(request.video_path, "/tmp/a.mp4")
        self.assertEqual(request.video_cache_uuids, ["vid-a"])
        self.assertEqual(request.video_cache_uuid, "vid-a")

    def test_supports_two_videos(self) -> None:
        """Schema should accept exactly two videos."""
        request = CompareRequest(
            prompt="hello",
            video_paths=["/tmp/a.mp4", "/tmp/b.mp4"],
            video_cache_uuids=["vid-a", "vid-b"],
        )
        self.assertEqual(len(request.video_paths), 2)
        self.assertEqual(request.video_path, "/tmp/a.mp4")
        self.assertEqual(request.video_cache_uuid, "vid-a")

    def test_rejects_more_than_two_videos(self) -> None:
        """Schema should reject requests that include more than two videos."""
        with self.assertRaises(ValidationError):
            CompareRequest(
                prompt="hello",
                video_paths=["/tmp/a.mp4", "/tmp/b.mp4", "/tmp/c.mp4"],
            )

    def test_text_only_mode_clears_media_and_cache_fields(self) -> None:
        """Text-only requests should ignore provided image/video and cache UUID payloads."""
        request = CompareRequest(
            prompt="hello",
            text_only=True,
            image_paths=["/tmp/a.png"],
            video_paths=["/tmp/a.mp4"],
            image_cache_uuids=["img-a"],
            video_cache_uuids=["vid-a"],
        )
        self.assertEqual(request.image_paths, [])
        self.assertEqual(request.video_paths, [])
        self.assertIsNone(request.video_path)
        self.assertEqual(request.image_cache_uuids, [])
        self.assertEqual(request.video_cache_uuids, [])
        self.assertIsNone(request.video_cache_uuid)

    def test_rejects_segmentation_with_multiple_videos(self) -> None:
        """Segmentation is only supported when a single video is present."""
        with self.assertRaises(ValidationError):
            CompareRequest(
                prompt="hello",
                video_paths=["/tmp/a.mp4", "/tmp/b.mp4"],
                segment_max_duration_s=10.0,
            )

    def test_target_config_use_model_defaults_clears_generation_params(self) -> None:
        """Per-target defaults mode should null explicit generation settings."""
        target = CompareTargetConfig(
            use_model_defaults=True,
            max_tokens=256,
            max_completion_tokens=128,
            temperature=0.1,
            top_p=0.5,
            top_k=10,
            presence_penalty=0.2,
            frequency_penalty=0.3,
        )
        self.assertIsNone(target.max_tokens)
        self.assertIsNone(target.max_completion_tokens)
        self.assertIsNone(target.temperature)
        self.assertIsNone(target.top_p)
        self.assertIsNone(target.top_k)
        self.assertIsNone(target.presence_penalty)
        self.assertIsNone(target.frequency_penalty)

    def test_target_config_accepts_supported_gemma_max_soft_tokens(self) -> None:
        """Gemma max_soft_tokens override should accept known supported values."""
        target = CompareTargetConfig(gemma_max_soft_tokens=560)
        self.assertEqual(target.gemma_max_soft_tokens, 560)

    def test_target_config_accepts_positive_thinking_token_budget(self) -> None:
        """Thinking token budget should accept positive integer values."""
        target = CompareTargetConfig(thinking_token_budget=2048)
        self.assertEqual(target.thinking_token_budget, 2048)

    def test_target_config_rejects_unsupported_gemma_max_soft_tokens(self) -> None:
        """Gemma max_soft_tokens override should reject unsupported values."""
        with self.assertRaises(ValidationError):
            CompareTargetConfig(gemma_max_soft_tokens=300)


if __name__ == "__main__":
    unittest.main()
