"""Preset helpers for compare-lab prompt and segmentation controls."""

from __future__ import annotations

import json

DEFAULT_CUSTOM_PROMPT = "Describe what is happening."
DEFAULT_TAG_CATEGORIES = "anime, documentary, action/adventure, drama"

PROMPT_MODE_CUSTOM = "Custom"
PROMPT_MODE_SEARCH_INDEXING = "Search/Indexing"
PROMPT_MODE_SUMMARIZATION = "Understanding/Summarization"
PROMPT_MODE_VISIBLE_CHUNK_SUMMARY = "Visible Chunk Summary"
PROMPT_MODE_OBJECT_DETECTION = "Object Detection (Boxes)"
PROMPT_MODE_TAGGING = "Tagging"
PROMPT_MODE_CLASSIFIER = "Classifier (Single Category)"
PROMPT_MODE_VIDEO_TYPE_ONE_WORD = "Video Type (One Word)"

PROMPT_MODE_CHOICES = [
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_SEARCH_INDEXING,
    PROMPT_MODE_SUMMARIZATION,
    PROMPT_MODE_VISIBLE_CHUNK_SUMMARY,
    PROMPT_MODE_OBJECT_DETECTION,
    PROMPT_MODE_TAGGING,
    PROMPT_MODE_CLASSIFIER,
    PROMPT_MODE_VIDEO_TYPE_ONE_WORD,
]

SEGMENTATION_PROFILE_BALANCED = "Balanced (30s / 2s)"
SEGMENTATION_PROFILE_FINE_GRAINED = "Fine-grained (2s / 0.5s)"
SEGMENTATION_PROFILE_OFF = "Off (0s / 0s)"
SEGMENTATION_PROFILE_CUSTOM = "Custom"

SEGMENTATION_PROFILE_CHOICES = [
    SEGMENTATION_PROFILE_BALANCED,
    SEGMENTATION_PROFILE_FINE_GRAINED,
    SEGMENTATION_PROFILE_OFF,
    SEGMENTATION_PROFILE_CUSTOM,
]

_SEGMENTATION_PROFILE_VALUES = {
    SEGMENTATION_PROFILE_BALANCED: (30.0, 2.0),
    SEGMENTATION_PROFILE_FINE_GRAINED: (2.0, 0.5),
    SEGMENTATION_PROFILE_OFF: (0.0, 0.0),
}

_SEARCH_INDEXING_PROMPT = """Analyze the provided media and build a searchable video index.

Return ONLY valid JSON (no markdown) with this schema:
{
  "segments": [
    {
      "time_start_s": number,
      "time_end_s": number,
      "scene_summary": "string",
      "keywords": ["string"],
      "entities": ["string"],
      "actions": ["string"]
    }
  ],
  "global_keywords": ["string"],
  "notable_objects": ["string"],
  "notable_people": ["string"],
  "search_queries": ["string"],
  "confidence_notes": ["string"]
}

Requirements:
- Use seconds for all timestamps.
- Keep keywords short and high-signal.
- Include uncertainty in confidence_notes.
"""

_SUMMARIZATION_PROMPT = """
Analyze the provided media and produce understanding-focused summarization.

Return ONLY valid JSON (no markdown) with this schema:
{
  "summary": {
    "short": "string",
    "detailed": "string"
  },
  "key_events": [
    {
      "time_start_s": number,
      "time_end_s": number,
      "event": "string",
      "importance": "low|medium|high"
    }
  ],
  "main_characters_or_subjects": ["string"],
  "topics": ["string"],
  "scene_flow": ["string"],
  "open_questions": ["string"]
}

Requirements:
- Use concise, factual language.
- Keep key_events chronologically ordered.
"""

_VISIBLE_CHUNK_SUMMARY_PROMPT = """Analyze this video chunk and produce:
1. Exactly 4 sentences summarizing the main visible events.
2. Exactly 6 bullet points listing important actions or scene changes.
3. Exactly 8 keywords describing the content.

Use only visible evidence. Keep the output concise and factual."""

_OBJECT_DETECTION_PROMPT = """Task: Locate the requested object(s) in the image.

Return JSON that matches this schema exactly:
{
  "detections": [
    {
      "label": string,
      "bbox_norm": [x0, y0, x1, y1]
    }
  ]
}

Requested: {query}

Requirements:
- Return ONLY valid JSON. No markdown.
- bbox_norm values must be floats in [0,1].
- Ensure x1 > x0 and y1 > y0.
- If the requested object is not visible, return {"detections": []}.
- Use concise labels that match the requested object(s).
"""

_VIDEO_TYPE_ONE_WORD_PROMPT = """Classify the primary video type.

Return exactly one lowercase word.
No punctuation. No explanation. No extra tokens.

Allowed words:
tutorial, documentary, interview, lecture, gameplay, vlog, ad, news, sports, animation, movie, other
"""


def parse_tag_categories(raw_value: str) -> list[str]:
    """Parse and deduplicate comma-separated tag categories."""
    cleaned = raw_value.strip()
    source = cleaned if cleaned else DEFAULT_TAG_CATEGORIES

    categories: list[str] = []
    seen: set[str] = set()
    for chunk in source.split(","):
        item = chunk.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        categories.append(item)

    if categories:
        return categories
    return [item.strip() for item in DEFAULT_TAG_CATEGORIES.split(",") if item.strip()]


def build_prompt_for_mode(*, mode: str, current_prompt: str, tag_categories_csv: str) -> str:
    """Return prompt text for the selected mode while keeping custom mode editable."""
    cleaned_current = current_prompt.strip()

    if mode == PROMPT_MODE_CUSTOM:
        return cleaned_current or DEFAULT_CUSTOM_PROMPT
    if mode == PROMPT_MODE_SEARCH_INDEXING:
        return _SEARCH_INDEXING_PROMPT
    if mode == PROMPT_MODE_SUMMARIZATION:
        return _SUMMARIZATION_PROMPT
    if mode == PROMPT_MODE_VISIBLE_CHUNK_SUMMARY:
        return _VISIBLE_CHUNK_SUMMARY_PROMPT
    if mode == PROMPT_MODE_OBJECT_DETECTION:
        return _OBJECT_DETECTION_PROMPT
    if mode == PROMPT_MODE_VIDEO_TYPE_ONE_WORD:
        return _VIDEO_TYPE_ONE_WORD_PROMPT
    if mode not in {PROMPT_MODE_TAGGING, PROMPT_MODE_CLASSIFIER}:
        return cleaned_current or DEFAULT_CUSTOM_PROMPT

    allowed_categories = parse_tag_categories(tag_categories_csv)
    categories_json = json.dumps(allowed_categories, ensure_ascii=True)
    if mode == PROMPT_MODE_CLASSIFIER:
        return (
            "Analyze the provided media and assign one category only.\n\n"
            f"Allowed categories: {categories_json}\n\n"
            "Return ONLY valid JSON (no markdown) with this schema:\n"
            "{\n"
            '  "category": "string",\n'
            '  "confidence": number,\n'
            '  "rationale": "string"\n'
            "}\n\n"
            "Requirements:\n"
            "- category must be exactly one value from allowed categories.\n"
            "- confidence must be between 0 and 1.\n"
            "- Keep rationale concise and evidence-based.\n"
        )
    return (
        "Analyze the provided media and assign category tags.\n\n"
        f"Allowed categories: {categories_json}\n\n"
        "Return ONLY valid JSON (no markdown) with this schema:\n"
        "{\n"
        '  "primary_category": "string",\n'
        '  "secondary_categories": ["string"],\n'
        '  "category_confidence": {"string": number},\n'
        '  "rationale": "string",\n'
        '  "content_flags": ["string"],\n'
        '  "evidence": ["string"]\n'
        "}\n\n"
        "Requirements:\n"
        "- primary_category must be one of the allowed categories.\n"
        "- secondary_categories must be a subset of allowed categories.\n"
        "- Use category_confidence values between 0 and 1.\n"
        "- If uncertain, explain in rationale and evidence.\n"
    )


def segmentation_values_for_profile(
    *,
    profile: str,
    current_duration: float,
    current_overlap: float,
) -> tuple[float, float]:
    """Map a UI segmentation profile to duration/overlap defaults."""
    mapped = _SEGMENTATION_PROFILE_VALUES.get(profile)
    if mapped is not None:
        return mapped

    duration = max(0.0, float(current_duration))
    overlap = max(0.0, float(current_overlap))
    return duration, overlap
