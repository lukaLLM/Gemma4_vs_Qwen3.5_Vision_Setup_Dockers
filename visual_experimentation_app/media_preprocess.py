"""Media probing, downscaling, segmentation, and data-url encoding utilities."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MediaInfo:
    """Media metadata extracted via ffprobe."""

    width: int
    height: int
    duration_s: float
    fps: float
    codec: str
    pix_fmt: str


@dataclass(frozen=True)
class SegmentClip:
    """One segment of a source video."""

    path: Path
    start_s: float
    end_s: float
    is_temp: bool


@dataclass
class PreparedMedia:
    """Output of local preprocessing for one experiment run."""

    image_paths: list[Path]
    video_paths: list[Path]
    cleanup_paths: list[Path]
    metadata: dict[str, Any]


def _resolve_existing_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def _parse_fps(raw_fps: object) -> float:
    if not isinstance(raw_fps, str):
        return 0.0
    value = raw_fps.strip()
    if not value or value in {"N/A", "0/0"}:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return 0.0
        if den == 0:
            return 0.0
        return num / den
    try:
        return float(value)
    except ValueError:
        return 0.0


def _run_ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed for {path.name}: {stderr}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("ffprobe payload is not a JSON object.")
    return payload


def probe_media(path: Path) -> MediaInfo:
    """Probe media dimensions/fps/duration using ffprobe."""
    payload = _run_ffprobe(path)
    streams = payload.get("streams", [])
    video_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if not isinstance(video_stream, dict):
        raise RuntimeError(f"No video stream found for: {path}")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    codec = str(video_stream.get("codec_name") or "").lower()
    pix_fmt = str(video_stream.get("pix_fmt") or "").lower()
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))

    duration_s = 0.0
    fmt = payload.get("format")
    if isinstance(fmt, dict):
        try:
            duration_s = float(fmt.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0
    if duration_s <= 0:
        try:
            duration_s = float(video_stream.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0

    return MediaInfo(
        width=max(width, 0),
        height=max(height, 0),
        duration_s=max(duration_s, 0.0),
        fps=max(fps, 0.0),
        codec=codec,
        pix_fmt=pix_fmt,
    )


def should_downscale(*, source_height: int, target_height: int) -> bool:
    """Return whether media should be downscaled for target max-height policy."""
    return source_height > max(target_height, 1)


def _mktemp_path(*, prefix: str, suffix: str) -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(path)


def _run_ffmpeg(cmd: list[str], *, context: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"{context}: {stderr}")


def _downscale_image(source: Path, *, target_height: int) -> Path:
    output_path = _mktemp_path(prefix="mm_lab_img_", suffix=".png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale=-2:{target_height}:force_original_aspect_ratio=decrease",
        str(output_path),
    ]
    _run_ffmpeg(cmd, context=f"Failed to downscale image {source.name}")
    return output_path


def _preprocess_video(
    source: Path,
    *,
    target_height: int,
    target_video_fps: float | None,
) -> Path:
    info = probe_media(source)
    filter_parts: list[str] = []

    if should_downscale(source_height=info.height, target_height=target_height):
        filter_parts.append(f"scale=-2:{target_height}:force_original_aspect_ratio=decrease")
    if target_video_fps is not None and info.fps > (target_video_fps + 0.05):
        filter_parts.append(f"fps={target_video_fps:g}")

    if not filter_parts:
        return source

    filter_parts.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    filter_parts.append("format=yuv420p")
    output_path = _mktemp_path(prefix="mm_lab_video_", suffix=".mp4")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(filter_parts),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_ffmpeg(cmd, context=f"Failed to preprocess video {source.name}")
    return output_path


def prepare_media(
    *,
    image_paths: list[str],
    video_paths: list[str],
    preprocess_images: bool,
    preprocess_video: bool,
    target_height: int,
    target_video_fps: float | None,
) -> PreparedMedia:
    """Resolve and optionally preprocess image/video inputs."""
    resolved_images: list[Path] = []
    cleanup_paths: list[Path] = []
    image_meta: list[dict[str, Any]] = []

    for raw_path in image_paths:
        source = _resolve_existing_path(raw_path)
        original_info = probe_media(source)
        processed = source

        if preprocess_images and should_downscale(
            source_height=original_info.height,
            target_height=target_height,
        ):
            processed = _downscale_image(source, target_height=target_height)
            cleanup_paths.append(processed)
        processed_info = probe_media(processed)

        resolved_images.append(processed)
        image_meta.append(
            {
                "source_path": str(source),
                "processed_path": str(processed),
                "original_height": original_info.height,
                "processed_height": processed_info.height,
                "original_width": original_info.width,
                "processed_width": processed_info.width,
            }
        )

    resolved_videos: list[Path] = []
    video_meta: list[dict[str, Any]] = []
    for raw_video_path in video_paths:
        source_video = _resolve_existing_path(raw_video_path)
        original_video_info = probe_media(source_video)
        processed_video = source_video
        if preprocess_video:
            processed_video = _preprocess_video(
                source_video,
                target_height=target_height,
                target_video_fps=target_video_fps,
            )
            if processed_video != source_video:
                cleanup_paths.append(processed_video)
        processed_video_info = probe_media(processed_video)
        resolved_videos.append(processed_video)
        video_meta.append(
            {
                "source_path": str(source_video),
                "processed_path": str(processed_video),
                "original_height": original_video_info.height,
                "processed_height": processed_video_info.height,
                "original_width": original_video_info.width,
                "processed_width": processed_video_info.width,
                "original_fps": original_video_info.fps,
                "processed_fps": processed_video_info.fps,
                "duration_s": processed_video_info.duration_s,
            }
        )

    return PreparedMedia(
        image_paths=resolved_images,
        video_paths=resolved_videos,
        cleanup_paths=cleanup_paths,
        metadata={
            "images": image_meta,
            "videos": video_meta,
            # Keep legacy key for compatibility with older result viewers.
            "video": video_meta[0] if video_meta else None,
        },
    )


def build_segment_ranges(
    *,
    duration_s: float,
    max_duration_s: float,
    overlap_s: float,
) -> list[tuple[float, float]]:
    """Split duration into [start, end] ranges with optional overlap."""
    total_duration = max(duration_s, 0.0)
    if total_duration <= 0:
        return [(0.0, 0.0)]
    if max_duration_s <= 0 or total_duration <= max_duration_s:
        return [(0.0, total_duration)]

    overlap = max(overlap_s, 0.0)
    ranges: list[tuple[float, float]] = []
    base_start = 0.0

    while base_start < total_duration - 1e-6:
        base_end = min(total_duration, base_start + max_duration_s)
        start = 0.0 if base_start <= 0 else max(0.0, base_start - overlap)
        end = (
            total_duration
            if base_end >= total_duration
            else min(total_duration, base_end + overlap)
        )
        ranges.append((start, end))
        if base_end >= total_duration:
            break
        base_start += max_duration_s

    return ranges


def _extract_segment(source: Path, *, start_s: float, end_s: float) -> Path:
    output_path = _mktemp_path(prefix="mm_lab_segment_", suffix=".mp4")
    duration = max(end_s - start_s, 0.05)

    transcode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_ffmpeg(transcode_cmd, context=f"Failed to extract segment from {source.name}")
    return output_path


def extract_video_segments(
    *,
    video_path: Path,
    max_duration_s: float,
    overlap_s: float,
) -> list[SegmentClip]:
    """Create temporary video clips for segment-level experiments."""
    info = probe_media(video_path)
    ranges = build_segment_ranges(
        duration_s=info.duration_s,
        max_duration_s=max_duration_s,
        overlap_s=overlap_s,
    )
    if len(ranges) <= 1:
        start, end = ranges[0]
        return [SegmentClip(path=video_path, start_s=start, end_s=end, is_temp=False)]

    clips: list[SegmentClip] = []
    try:
        for start_s, end_s in ranges:
            segment = _extract_segment(video_path, start_s=start_s, end_s=end_s)
            clips.append(SegmentClip(path=segment, start_s=start_s, end_s=end_s, is_temp=True))
    except Exception:
        cleanup_paths([clip.path for clip in clips if clip.is_temp])
        raise
    return clips


def encode_file_to_data_url(path: Path) -> str:
    """Encode a local file into a data URL."""
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "video/mp4" if path.suffix.lower() in {".mp4", ".mkv", ".mov"} else "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def cleanup_paths(paths: list[Path]) -> None:
    """Best-effort cleanup for temporary files."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
