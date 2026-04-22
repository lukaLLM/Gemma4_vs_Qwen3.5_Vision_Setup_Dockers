"""Tests for local detection preview rendering."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_coins_bbox import (  # noqa: E402
    BBOX_NORM,
    main as plot_coins_bbox_main,
    write_coins_bbox_preview,
)
from visual_experimentation_app.detection_preview import (  # noqa: E402
    parse_detection_payload,
    write_detection_preview,
)


class DetectionPreviewTest(unittest.TestCase):
    """Covers validation and HTML preview rendering for detection boxes."""

    def test_write_detection_preview_renders_expected_overlay(self) -> None:
        """Preview should reflect the normalized box passed in the detection JSON."""
        detections = parse_detection_payload(
            {
                "detections": [
                    {
                        "label": "50 cents reversed",
                        "bbox_norm": [0.68, 0.421, 0.908, 0.533],
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "coins_preview.html"
            result = write_detection_preview(
                image_path=ROOT / "Benchmarks/coins.jpeg",
                detections=detections,
                output_path=output_path,
                title="Coins Preview",
            )

            self.assertEqual(result, output_path.resolve())
            html_text = result.read_text(encoding="utf-8")

        self.assertIn("50 cents reversed", html_text)
        self.assertIn("coins.jpeg", html_text)
        self.assertIn("left: 68.0000%;", html_text)
        self.assertIn("top: 42.1000%;", html_text)
        self.assertIn("width: 22.8000%;", html_text)
        self.assertIn("height: 11.2000%;", html_text)

    def test_parse_detection_payload_rejects_invalid_box_order(self) -> None:
        """Invalid normalized boxes should be rejected before rendering."""
        with self.assertRaises(ValueError):
            parse_detection_payload(
                {
                    "detections": [
                        {
                            "label": "50 cents reversed",
                            "bbox_norm": [0.9, 0.421, 0.68, 0.533],
                        }
                    ]
                }
            )

    def test_plot_coins_bbox_script_uses_module_bbox_variable(self) -> None:
        """Coins plot script should use the module bbox variable."""
        with mock.patch(
            "scripts.plot_coins_bbox.write_coins_bbox_preview",
            return_value=Path("/tmp/coins_bbox_preview.jpeg"),
        ) as mocked_write:
            exit_code = plot_coins_bbox_main()

        self.assertEqual(exit_code, 0)
        mocked_write.assert_called_once()
        self.assertEqual(mocked_write.call_args.args[0], BBOX_NORM)

    def test_write_coins_bbox_preview_saves_annotated_image(self) -> None:
        """Coins bbox writer should save a real image file in the requested location."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "coins_bbox_preview.jpeg"
            result = write_coins_bbox_preview((0.68, 0.421, 0.908, 0.533), output_path=output_path)

            self.assertEqual(result, output_path.resolve())
            self.assertTrue(result.is_file())

            with Image.open(result) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertGreater(image.size[0], 0)
                self.assertGreater(image.size[1], 0)


if __name__ == "__main__":
    unittest.main()
