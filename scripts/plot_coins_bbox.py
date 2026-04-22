"""Save one hard-coded normalized bbox on the fixed benchmark coins image."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.detection_preview import _validate_bbox_norm

COINS_IMAGE = ROOT / "Benchmarks/coins.jpeg"
OUTPUT_IMAGE = ROOT / "visual_experimentation_app/results/previews/coins_bbox_preview.jpeg"
BBOX_NORM = (
    0.472, 0.468, 0.703, 0.575
)


def write_coins_bbox_preview(
    bbox_norm: tuple[float, float, float, float],
    *,
    output_path: Path = OUTPUT_IMAGE,
) -> Path:
    """Draw one bbox on the benchmark coins image and save it as a JPEG."""
    x0, y0, x1, y1 = _validate_bbox_norm(bbox_norm)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(COINS_IMAGE) as source_image:
        image = source_image.convert("RGB")

    width, height = image.size
    left = round(x0 * width)
    top = round(y0 * height)
    right = round(x1 * width)
    bottom = round(y1 * height)
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(width, height) * 0.01))

    for offset in range(line_width):
        draw.rectangle(
            (left - offset, top - offset, right + offset, bottom + offset),
            outline=(220, 20, 60),
        )

    image.save(output_path, format="JPEG", quality=95)
    return output_path.resolve()


def main() -> int:
    """Render the preview image from the hard-coded bbox and print the output path."""
    output_path = write_coins_bbox_preview(BBOX_NORM)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
