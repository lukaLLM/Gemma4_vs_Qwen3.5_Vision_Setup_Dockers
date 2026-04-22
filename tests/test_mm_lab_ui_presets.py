"""Tests for compare-lab prompt/segmentation UI presets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.ui_presets import (  # noqa: E402
    DEFAULT_TAG_CATEGORIES,
    PROMPT_MODE_CLASSIFIER,
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_OBJECT_DETECTION,
    PROMPT_MODE_SEARCH_INDEXING,
    PROMPT_MODE_TAGGING,
    PROMPT_MODE_VISIBLE_CHUNK_SUMMARY,
    SEGMENTATION_PROFILE_BALANCED,
    SEGMENTATION_PROFILE_FINE_GRAINED,
    SEGMENTATION_PROFILE_OFF,
    build_prompt_for_mode,
    parse_tag_categories,
    segmentation_values_for_profile,
)


class UiPresetsTest(unittest.TestCase):
    """Covers prompt preset rendering and segmentation profile mapping."""

    def test_parse_tag_categories_uses_default_when_empty(self) -> None:
        """Empty CSV should resolve to default categories."""
        parsed = parse_tag_categories("")
        self.assertEqual(parsed, [item.strip() for item in DEFAULT_TAG_CATEGORIES.split(",")])

    def test_parse_tag_categories_deduplicates_and_trims(self) -> None:
        """CSV parser should remove empty values and deduplicate entries."""
        parsed = parse_tag_categories(" anime, drama, anime , , Documentary ")
        self.assertEqual(parsed, ["anime", "drama", "Documentary"])

    def test_build_prompt_for_custom_mode_returns_editable_text(self) -> None:
        """Custom mode should preserve user-provided prompt text."""
        prompt = build_prompt_for_mode(
            mode=PROMPT_MODE_CUSTOM,
            current_prompt="  Explain this frame by frame.  ",
            tag_categories_csv="",
        )
        self.assertEqual(prompt, "Explain this frame by frame.")

    def test_build_prompt_for_search_mode_enforces_json(self) -> None:
        """Search preset should include structured JSON-only instruction."""
        prompt = build_prompt_for_mode(
            mode=PROMPT_MODE_SEARCH_INDEXING,
            current_prompt="ignored",
            tag_categories_csv="",
        )
        self.assertIn("Return ONLY valid JSON", prompt)
        self.assertIn('"segments"', prompt)

    def test_build_prompt_for_visible_chunk_summary_matches_requested_shape(self) -> None:
        """Visible chunk summary preset should preserve the requested output constraints."""
        prompt = build_prompt_for_mode(
            mode=PROMPT_MODE_VISIBLE_CHUNK_SUMMARY,
            current_prompt="ignored",
            tag_categories_csv="",
        )
        self.assertIn("Exactly 4 sentences", prompt)
        self.assertIn("Exactly 6 bullet points", prompt)
        self.assertIn("Exactly 8 keywords", prompt)
        self.assertIn("Use only visible evidence", prompt)

    def test_build_prompt_for_object_detection_uses_query_placeholder(self) -> None:
        """Object detection preset should include schema and `{query}` placeholder."""
        prompt = build_prompt_for_mode(
            mode=PROMPT_MODE_OBJECT_DETECTION,
            current_prompt="ignored",
            tag_categories_csv="",
        )
        self.assertIn("Locate the requested object(s) in the image", prompt)
        self.assertIn('"detections"', prompt)
        self.assertIn('"bbox_norm"', prompt)
        self.assertIn("Requested: {query}", prompt)

    def test_build_prompt_for_tagging_injects_categories(self) -> None:
        """Tagging preset should inject allowed categories into prompt."""
        prompt = build_prompt_for_mode(
            mode=PROMPT_MODE_TAGGING,
            current_prompt="ignored",
            tag_categories_csv="anime, drama",
        )
        self.assertIn('Allowed categories: ["anime", "drama"]', prompt)
        self.assertIn('"primary_category"', prompt)

    def test_build_prompt_for_classifier_requires_single_category(self) -> None:
        """Classifier preset should enforce single-category JSON response."""
        prompt = build_prompt_for_mode(
            mode=PROMPT_MODE_CLASSIFIER,
            current_prompt="ignored",
            tag_categories_csv="anime, drama",
        )
        self.assertIn("assign one category only", prompt)
        self.assertIn('Allowed categories: ["anime", "drama"]', prompt)
        self.assertIn('"category"', prompt)
        self.assertNotIn('"secondary_categories"', prompt)

    def test_segmentation_profile_mapping_defaults(self) -> None:
        """Preset profiles should map to expected duration and overlap values."""
        self.assertEqual(
            segmentation_values_for_profile(
                profile=SEGMENTATION_PROFILE_BALANCED,
                current_duration=9.0,
                current_overlap=3.0,
            ),
            (30.0, 2.0),
        )
        self.assertEqual(
            segmentation_values_for_profile(
                profile=SEGMENTATION_PROFILE_FINE_GRAINED,
                current_duration=9.0,
                current_overlap=3.0,
            ),
            (2.0, 0.5),
        )
        self.assertEqual(
            segmentation_values_for_profile(
                profile=SEGMENTATION_PROFILE_OFF,
                current_duration=9.0,
                current_overlap=3.0,
            ),
            (0.0, 0.0),
        )

    def test_segmentation_profile_custom_uses_current_values(self) -> None:
        """Custom profile should preserve current duration/overlap values."""
        self.assertEqual(
            segmentation_values_for_profile(
                profile="Custom",
                current_duration=12.5,
                current_overlap=1.5,
            ),
            (12.5, 1.5),
        )
        self.assertEqual(
            segmentation_values_for_profile(
                profile="Custom",
                current_duration=-1.0,
                current_overlap=-2.0,
            ),
            (0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
