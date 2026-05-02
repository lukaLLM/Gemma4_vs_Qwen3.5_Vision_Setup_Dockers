"""Helpers for rendering local HTML previews of detection boxes."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


@dataclass(frozen=True)
class Detection:
    """One labeled normalized bounding box."""

    label: str
    bbox_norm: tuple[float, float, float, float]
    color: str | None = None


def _validate_bbox_norm(raw_bbox: object) -> tuple[float, float, float, float]:
    """Validate and normalize a bbox expressed in [0, 1] coordinates."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("bbox_norm must be a list of exactly 4 numbers.")
    x0, y0, x1, y1 = tuple(float(value) for value in raw_bbox)
    if not all(0.0 <= value <= 1.0 for value in (x0, y0, x1, y1)):
        raise ValueError("bbox_norm values must be floats in [0, 1].")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox_norm must satisfy x1 > x0 and y1 > y0.")
    return (x0, y0, x1, y1)


def _parse_box2d(raw_box: object) -> tuple[float, float, float, float]:
    """Convert a model-native box_2d [y1, x1, y2, x2] on the 0–1000 grid to [x0, y0, x1, y1] in [0, 1].

    Gemma 4 (and compatible models) output bounding boxes in row-first order on a
    1000×1000 normalized grid. This helper descales and reorders to the internal
    (x0, y0, x1, y1) convention used by :class:`Detection`.
    """
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError("box_2d must be a list of exactly 4 numbers.")
    y1_raw, x1_raw, y2_raw, x2_raw = tuple(float(v) for v in raw_box)
    if not all(0.0 <= v <= 1000.0 for v in (y1_raw, x1_raw, y2_raw, x2_raw)):
        raise ValueError("box_2d values must be in [0, 1000].")
    if y2_raw <= y1_raw or x2_raw <= x1_raw:
        raise ValueError("box_2d must satisfy y2 > y1 and x2 > x1.")
    return (x1_raw / 1000.0, y1_raw / 1000.0, x2_raw / 1000.0, y2_raw / 1000.0)


def _parse_polygon_2d(raw_polygon: object) -> list[tuple[float, float]]:
    """Convert model-native polygon_2d [[y, x], ...] on the 0–1000 grid to [(x, y), ...] in [0, 1].

    Gemma 4 outputs polygon points in row-first (y, x) order on a 1000×1000 grid.
    This helper descales and reorders to the internal (x, y) convention used by
    :class:`SegmentationResult`.
    """
    if not isinstance(raw_polygon, (list, tuple)) or len(raw_polygon) < 3:
        raise ValueError("polygon_2d must be a list of at least 3 [y, x] pairs.")
    points: list[tuple[float, float]] = []
    for i, point in enumerate(raw_polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"polygon_2d[{i}] must be a [y, x] pair.")
        y, x = float(point[0]), float(point[1])
        if not (0.0 <= y <= 1000.0 and 0.0 <= x <= 1000.0):
            raise ValueError(f"polygon_2d[{i}] values must be in [0, 1000].")
        points.append((x / 1000.0, y / 1000.0))
    return points


def parse_detection_payload(payload: object) -> list[Detection]:
    """Parse the benchmark detection payload into validated detection rows.

    Accepts both the new ``box_2d`` format (model-native 0–1000 grid, row-first
    ``[y1, x1, y2, x2]``) and the legacy ``bbox_norm`` format (``[x0, y0, x1, y1]``
    in ``[0, 1]``).  ``box_2d`` takes precedence when both keys are present.
    """
    if not isinstance(payload, dict):
        raise ValueError("Detection payload must be a JSON object.")
    raw_detections = payload.get("detections")
    if not isinstance(raw_detections, list):
        raise ValueError("Detection payload must contain a detections list.")

    detections: list[Detection] = []
    for index, raw_detection in enumerate(raw_detections, start=1):
        if not isinstance(raw_detection, dict):
            raise ValueError(f"detections[{index}] must be an object.")
        raw_label = raw_detection.get("label")
        label = str(raw_label or "").strip()
        if not label:
            raise ValueError(f"detections[{index}].label must be a non-empty string.")
        if raw_detection.get("box_2d") is not None:
            bbox = _parse_box2d(raw_detection["box_2d"])
        else:
            bbox = _validate_bbox_norm(raw_detection.get("bbox_norm"))
        detections.append(
            Detection(
                label=label,
                bbox_norm=bbox,
                color=str(raw_detection.get("color") or "").strip() or None,
            )
        )
    return detections


def load_detection_payload_json(payload_json: str) -> list[Detection]:
    """Parse detection JSON text into validated detection rows."""
    return parse_detection_payload(json.loads(payload_json))


def _slugify(value: str) -> str:
    """Build a filesystem-friendly slug."""
    collapsed = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return collapsed or "detections"


def _default_output_path(image_path: Path, detections: Sequence[Detection]) -> Path:
    """Choose a stable preview location when the caller does not provide one."""
    preview_dir = Path("visual_experimentation_app/results/previews")
    suffix = _slugify(detections[0].label if detections else "detections")
    return preview_dir / f"{image_path.stem}_{suffix}.html"


def _html_image_src(image_path: Path, output_path: Path) -> str:
    """Render the image path relative to the preview file for local browsing."""
    return os.path.relpath(image_path, start=output_path.parent)


def write_detection_preview(
    image_path: Path,
    detections: Sequence[Detection],
    output_path: Path | None = None,
    *,
    title: str | None = None,
) -> Path:
    """Write a self-contained HTML preview that overlays detection boxes on an image."""
    resolved_image_path = image_path.expanduser().resolve()
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"Image not found: {resolved_image_path}")

    resolved_output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else _default_output_path(resolved_image_path, detections).resolve()
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_html = []
    for detection in detections:
        x0, y0, x1, y1 = detection.bbox_norm
        overlay_html.append(
            """
            <div class="box" style="left: {left:.4f}%; top: {top:.4f}%; width: {width:.4f}%; height: {height:.4f}%;">
              <span class="box-label">{label}</span>
            </div>
            """.strip().format(
                left=x0 * 100.0,
                top=y0 * 100.0,
                width=(x1 - x0) * 100.0,
                height=(y1 - y0) * 100.0,
                label=html.escape(detection.label),
            )
        )

    payload = {
        "detections": [
            {"label": detection.label, "bbox_norm": list(detection.bbox_norm)}
            for detection in detections
        ]
    }
    html_text = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f5f1e8;
        --panel: #fffdf8;
        --text: #1b1b1b;
        --accent: #c0392b;
        --accent-soft: rgba(192, 57, 43, 0.16);
        --border: #d8cfc4;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        padding: 24px;
        font-family: "Georgia", "Times New Roman", serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(214, 177, 109, 0.28), transparent 28%),
          linear-gradient(135deg, #f8f3eb, var(--bg));
      }}

      .layout {{
        display: grid;
        gap: 24px;
        grid-template-columns: minmax(320px, 1.2fr) minmax(260px, 0.8fr);
        align-items: start;
      }}

      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 20px 40px rgba(54, 43, 27, 0.08);
        overflow: hidden;
      }}

      .image-panel {{
        padding: 20px;
      }}

      .image-wrap {{
        position: relative;
        display: inline-block;
        width: 100%;
      }}

      img {{
        display: block;
        width: 100%;
        height: auto;
        border-radius: 12px;
      }}

      .box {{
        position: absolute;
        border: 3px solid var(--accent);
        background: var(--accent-soft);
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.35) inset;
      }}

      .box-label {{
        position: absolute;
        top: 0;
        left: 0;
        transform: translateY(-100%);
        padding: 6px 10px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        background: var(--accent);
        color: #fff;
        font-size: 14px;
        line-height: 1;
        white-space: nowrap;
      }}

      .meta-panel {{
        padding: 20px 22px;
      }}

      h1 {{
        margin: 0 0 16px;
        font-size: 28px;
        line-height: 1.1;
      }}

      h2 {{
        margin: 22px 0 10px;
        font-size: 18px;
      }}

      p,
      li {{
        font-size: 16px;
        line-height: 1.5;
      }}

      code,
      pre {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      }}

      ul {{
        margin: 0;
        padding-left: 20px;
      }}

      pre {{
        margin: 0;
        padding: 14px;
        overflow-x: auto;
        border-radius: 12px;
        background: #201d1a;
        color: #f6f2eb;
        font-size: 14px;
      }}

      .file-path {{
        margin: 0 0 14px;
        word-break: break-all;
      }}

      @media (max-width: 900px) {{
        body {{
          padding: 16px;
        }}

        .layout {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="layout">
      <section class="panel image-panel">
        <div class="image-wrap">
          <img src="{image_src}" alt="{title}">
          {overlay_html}
        </div>
      </section>

      <aside class="panel meta-panel">
        <h1>{title}</h1>
        <p class="file-path"><strong>Image:</strong> <code>{image_path}</code></p>
        <h2>Detections</h2>
        <ul>
          {detection_list}
        </ul>
        <h2>Detection JSON</h2>
        <pre>{payload_json}</pre>
      </aside>
    </div>
  </body>
</html>
""".format(
        title=html.escape(title or f"Detection Preview: {resolved_image_path.name}"),
        image_src=html.escape(_html_image_src(resolved_image_path, resolved_output_path)),
        image_path=html.escape(str(resolved_image_path)),
        overlay_html="\n          ".join(overlay_html) or "<!-- no detections -->",
        detection_list="\n          ".join(
            f"<li><strong>{html.escape(detection.label)}</strong>: "
            f"<code>{html.escape(str(list(detection.bbox_norm)))}</code></li>"
            for detection in detections
        )
        or "<li>No detections</li>",
        payload_json=html.escape(json.dumps(payload, indent=2)),
    )
    resolved_output_path.write_text(html_text, encoding="utf-8")
    return resolved_output_path


def draw_detections_on_image(
    image_path: Path,
    detections: Sequence[Detection],
) -> "PILImage":
    """Draw detection bounding boxes on an image and return the annotated PIL Image.

    Args:
        image_path: Path to the source image file.
        detections: Sequence of validated Detection objects.

    Returns:
        A new PIL Image with red bounding boxes and labels drawn.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    from PIL import Image, ImageDraw, ImageFont

    resolved = image_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")

    with Image.open(resolved) as src:
        image = src.convert("RGB")

    width, height = image.size
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(width, height) * 0.008))
    outline_color = (220, 20, 60)
    label_bg = (220, 20, 60)
    label_fg = (255, 255, 255)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(14, round(min(width, height) * 0.022)))
    except (OSError, IOError):
        font = ImageFont.load_default()

    for detection in detections:
        x0, y0, x1, y1 = detection.bbox_norm
        left = round(x0 * width)
        top = round(y0 * height)
        right = round(x1 * width)
        bottom = round(y1 * height)

        for offset in range(line_width):
            draw.rectangle(
                (left - offset, top - offset, right + offset, bottom + offset),
                outline=outline_color,
            )

        label = detection.label
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        pad = 4
        label_top = max(0, top - text_h - pad * 2)
        draw.rectangle(
            (left, label_top, left + text_w + pad * 2, label_top + text_h + pad * 2),
            fill=label_bg,
        )
        draw.text((left + pad, label_top + pad), label, fill=label_fg, font=font)

    return image


_MASK_PALETTE = [
    (220, 20, 60),
    (30, 144, 255),
    (50, 205, 50),
    (255, 165, 0),
    (148, 103, 189),
    (0, 206, 209),
    (255, 20, 147),
    (255, 215, 0),
]


def _css_color_to_rgb(name: str) -> tuple[int, int, int] | None:
    """Convert a CSS color name to an RGB tuple using Pillow."""
    from PIL import ImageColor

    try:
        r, g, b = ImageColor.getrgb(name)
        return (r, g, b)
    except (ValueError, AttributeError):
        return None


def draw_colored_masks_on_image(
    image_path: Path,
    detections: Sequence[Detection],
) -> "PILImage":
    """Draw semi-transparent colored mask overlays on an image.

    Args:
        image_path: Path to the source image file.
        detections: Sequence of validated Detection objects with optional color hints.

    Returns:
        A new PIL Image with colored mask overlays and labels drawn.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    from PIL import Image, ImageDraw, ImageFont

    resolved = image_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")

    with Image.open(resolved) as src:
        image = src.convert("RGBA")

    width, height = image.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    label_fg = (255, 255, 255)
    mask_alpha = 160
    border_alpha = 220

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(14, round(min(width, height) * 0.022)))
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, detection in enumerate(detections):
        if detection.color:
            rgb = _css_color_to_rgb(detection.color)
        else:
            rgb = None
        if rgb is None:
            rgb = _MASK_PALETTE[i % len(_MASK_PALETTE)]

        x0, y0, x1, y1 = detection.bbox_norm
        left = round(x0 * width)
        top = round(y0 * height)
        right = round(x1 * width)
        bottom = round(y1 * height)

        overlay_draw.rectangle(
            (left, top, right, bottom),
            fill=(*rgb, mask_alpha),
            outline=(*rgb, border_alpha),
            width=max(2, round(min(width, height) * 0.005)),
        )

        label = detection.label
        bbox_text = overlay_draw.textbbox((0, 0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        pad = 4
        label_top = max(0, top - text_h - pad * 2)
        overlay_draw.rectangle(
            (left, label_top, left + text_w + pad * 2, label_top + text_h + pad * 2),
            fill=(*rgb, 220),
        )
        overlay_draw.text((left + pad, label_top + pad), label, fill=(*label_fg, 255), font=font)

    composited = Image.alpha_composite(image, overlay)
    return composited.convert("RGB")


@dataclass(frozen=True)
class SegmentationResult:
    """One labeled polygon segmentation with bounding box."""

    label: str
    bbox_norm: tuple[float, float, float, float]
    polygon_norm: list[tuple[float, float]]
    color: str | None = None


def _validate_polygon_norm(raw_polygon: object) -> list[tuple[float, float]]:
    """Validate normalized polygon coordinate pairs."""
    if not isinstance(raw_polygon, (list, tuple)) or len(raw_polygon) < 3:
        raise ValueError("polygon_norm must be a list of at least 3 [x, y] pairs.")
    points: list[tuple[float, float]] = []
    for i, point in enumerate(raw_polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"polygon_norm[{i}] must be an [x, y] pair.")
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"polygon_norm[{i}] values must be floats in [0, 1].")
        points.append((x, y))
    return points


def parse_segmentation_payload(payload: object) -> list[SegmentationResult]:
    """Parse a segmentation JSON payload into validated segmentation results.

    Accepts both the new ``box_2d`` / ``polygon_2d`` format (model-native 0–1000
    grid, row-first coordinates) and the legacy ``bbox_norm`` / ``polygon_norm``
    format (``[0, 1]`` range).  New keys take precedence when both are present.
    """
    if not isinstance(payload, dict):
        raise ValueError("Segmentation payload must be a JSON object.")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Segmentation payload must contain a segments list.")

    results: list[SegmentationResult] = []
    for index, raw_seg in enumerate(raw_segments, start=1):
        if not isinstance(raw_seg, dict):
            raise ValueError(f"segments[{index}] must be an object.")
        raw_label = raw_seg.get("label")
        label = str(raw_label or "").strip()
        if not label:
            raise ValueError(f"segments[{index}].label must be a non-empty string.")
        if raw_seg.get("box_2d") is not None:
            bbox = _parse_box2d(raw_seg["box_2d"])
        else:
            bbox = _validate_bbox_norm(raw_seg.get("bbox_norm"))
        if raw_seg.get("polygon_2d") is not None:
            polygon = _parse_polygon_2d(raw_seg["polygon_2d"])
        else:
            polygon = _validate_polygon_norm(raw_seg.get("polygon_norm"))
        results.append(
            SegmentationResult(
                label=label,
                bbox_norm=bbox,
                polygon_norm=polygon,
                color=str(raw_seg.get("color") or "").strip() or None,
            )
        )
    return results


def load_segmentation_payload_json(payload_json: str) -> list[SegmentationResult]:
    """Parse segmentation JSON text into validated segmentation results."""
    return parse_segmentation_payload(json.loads(payload_json))


def draw_segmentation_masks_on_image(
    image_path: Path,
    segments: Sequence[SegmentationResult],
) -> "PILImage":
    """Draw filled polygon masks on an image from segmentation results.

    Args:
        image_path: Path to the source image file.
        segments: Sequence of validated SegmentationResult objects.

    Returns:
        A new PIL Image with polygon masks and labels drawn.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    from PIL import Image, ImageDraw, ImageFont

    resolved = image_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")

    with Image.open(resolved) as src:
        image = src.convert("RGBA")

    width, height = image.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    label_fg = (255, 255, 255)
    mask_alpha = 120
    outline_alpha = 230

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=max(14, round(min(width, height) * 0.022)),
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, seg in enumerate(segments):
        if seg.color:
            rgb = _css_color_to_rgb(seg.color)
        else:
            rgb = None
        if rgb is None:
            rgb = _MASK_PALETTE[i % len(_MASK_PALETTE)]

        pixel_polygon = [(round(x * width), round(y * height)) for x, y in seg.polygon_norm]

        overlay_draw.polygon(
            pixel_polygon,
            fill=(*rgb, mask_alpha),
            outline=(*rgb, outline_alpha),
        )
        # Draw thick outline
        line_width = max(2, round(min(width, height) * 0.004))
        for offset in range(line_width):
            overlay_draw.polygon(
                [(px + offset, py + offset) for px, py in pixel_polygon],
                outline=(*rgb, outline_alpha),
            )

        # Label at top of bounding box
        x0, y0, _, _ = seg.bbox_norm
        left = round(x0 * width)
        top = round(y0 * height)
        label = seg.label
        bbox_text = overlay_draw.textbbox((0, 0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        pad = 4
        label_top = max(0, top - text_h - pad * 2)
        overlay_draw.rectangle(
            (left, label_top, left + text_w + pad * 2, label_top + text_h + pad * 2),
            fill=(*rgb, 220),
        )
        overlay_draw.text((left + pad, label_top + pad), label, fill=(*label_fg, 255), font=font)

    composited = Image.alpha_composite(image, overlay)
    return composited.convert("RGB")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for manual preview generation."""
    parser = argparse.ArgumentParser(
        description="Render a local HTML preview for normalized detection boxes."
    )
    parser.add_argument("--image", required=True, type=Path, help="Image path to inspect.")
    parser.add_argument(
        "--detections-json",
        required=True,
        help='Detection JSON string like {"detections":[{"label":"coin","bbox_norm":[...]}]}',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional HTML output path. Defaults under visual_experimentation_app/results/previews.",
    )
    parser.add_argument("--title", default=None, help="Optional page title.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for preview generation."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    detections = load_detection_payload_json(args.detections_json)
    output_path = write_detection_preview(
        image_path=args.image,
        detections=detections,
        output_path=args.output,
        title=args.title,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
