"""Helpers for rendering local HTML previews of detection boxes."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Detection:
    """One labeled normalized bounding box."""

    label: str
    bbox_norm: tuple[float, float, float, float]


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


def parse_detection_payload(payload: object) -> list[Detection]:
    """Parse the benchmark detection payload into validated detection rows."""
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
        detections.append(
            Detection(label=label, bbox_norm=_validate_bbox_norm(raw_detection.get("bbox_norm")))
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
