"""Tests for MM lab media preprocessing helpers."""

from __future__ import annotations

import sys
import unittest
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.media_preprocess import (  # noqa: E402
    _extract_segment,
    build_segment_ranges,
    should_downscale,
)


class MediaPreprocessHelpersTest(unittest.TestCase):
    """Covers pure segmentation and resize helper logic."""

    def test_should_downscale(self) -> None:
        """Downscale helper should only trigger when source exceeds target."""
        self.assertTrue(should_downscale(source_height=720, target_height=480))
        self.assertFalse(should_downscale(source_height=360, target_height=480))

    def test_build_segment_ranges_without_split(self) -> None:
        """Short clips should return a single [0, duration] segment."""
        ranges = build_segment_ranges(duration_s=12.0, max_duration_s=30.0, overlap_s=2.0)
        self.assertEqual(ranges, [(0.0, 12.0)])

    def test_build_segment_ranges_with_overlap(self) -> None:
        """Segment planner should emit overlapped ranges for long clips."""
        ranges = build_segment_ranges(duration_s=65.0, max_duration_s=30.0, overlap_s=2.0)
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0], (0.0, 32.0))
        self.assertEqual(ranges[1], (28.0, 62.0))
        self.assertEqual(ranges[2], (58.0, 65.0))

    def test_extract_segment_always_transcodes_for_metadata_stability(self) -> None:
        """Segment extraction should use transcode path instead of stream copy."""
        fake_output = Path("/tmp/mm_lab_segment_test.mp4")
        with (
            mock.patch(
                "visual_experimentation_app.media_preprocess._mktemp_path",
                return_value=fake_output,
            ),
            mock.patch("visual_experimentation_app.media_preprocess._run_ffmpeg") as mocked_ffmpeg,
        ):
            result = _extract_segment(Path("/tmp/input.mp4"), start_s=1.0, end_s=5.0)

        self.assertEqual(result, fake_output)
        mocked_ffmpeg.assert_called_once()
        cmd = mocked_ffmpeg.call_args.args[0]
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)
        self.assertIn("-vf", cmd)
        self.assertNotIn("copy", cmd)
        self.assertIn("1.000", cmd)
        self.assertIn("4.000", cmd)


if __name__ == "__main__":
    unittest.main()
