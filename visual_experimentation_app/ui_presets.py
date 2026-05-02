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
PROMPT_MODE_COLORED_MASKS = "Colored Masks"
PROMPT_MODE_SEGMENTATION_MASKS = "Segmentation Masks (Polygons)"
PROMPT_MODE_TAGGING = "Tagging"
PROMPT_MODE_CLASSIFIER = "Classifier (Single Category)"
PROMPT_MODE_VIDEO_TYPE_ONE_WORD = "Video Type (One Word)"
PROMPT_MODE_DETAILED_DESCRIPTION = "Detailed Image Description"
PROMPT_MODE_HANDWRITTEN_NOTES = "Handwritten Notes / OCR"
PROMPT_MODE_FOOD_RECOGNITION = "Food Recognition"
PROMPT_MODE_LOCATION_ID = "Location Identification"
PROMPT_MODE_GEOGUESSR = "GeoGuessr"
PROMPT_MODE_PRICE_EXTRACTION = "Price Extraction"
PROMPT_MODE_PRICE_EXTRACTION_STRICT = "Price Extraction (Strict)"
PROMPT_MODE_MEME_RECOGNITION = "Meme Recognition"
PROMPT_MODE_TRADLE = "Tradle"
PROMPT_MODE_TABLE_EXTRACTION = "Table Extraction & Math"
PROMPT_MODE_ACTION_ID_VIDEO = "Action ID (Video)"
PROMPT_MODE_WORKOUT_TRACKING = "Workout Tracking (Video)"
PROMPT_MODE_EVENT_ID_VIDEO = "Event ID (Video)"
PROMPT_MODE_FIRE_DETECTION = "Fire Detection"
PROMPT_MODE_REAL_VS_AI_VIDEO = "Real vs AI Video"
PROMPT_MODE_COMPLEX_COUNTING = "Complex Counting"
PROMPT_MODE_VIDEO_NUMBER_SUMMING = "Video Number Summing"
PROMPT_MODE_ANIMAL_ID = "Animal Identification"
PROMPT_MODE_ANIMAL_ID_UNCERTAINTY = "Animal ID (With Uncertainty)"
PROMPT_MODE_WILDLIFE_DATASET = "Wildlife Dataset Description"
PROMPT_MODE_UNCERTAINTY_CALIBRATION = "Uncertainty Calibration"
PROMPT_MODE_UNCERTAINTY_CONFIDENCE = "Uncertainty Calibration (Confidence)"
PROMPT_MODE_UNCERTAINTY_STRICT_JSON = "Uncertainty Calibration (Strict JSON)"
PROMPT_MODE_UI_UPLOAD_FILE = "UI Understanding: Upload File"
PROMPT_MODE_UI_NEW_SPREADSHEET = "UI Understanding: New Spreadsheet"
PROMPT_MODE_UI_SHARE_PROJECT = "UI Understanding: Share Project"
PROMPT_MODE_UI_OPEN_FILE = "UI Understanding: Open File"

PROMPT_MODE_CHOICES = [
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_SEARCH_INDEXING,
    PROMPT_MODE_SUMMARIZATION,
    PROMPT_MODE_VISIBLE_CHUNK_SUMMARY,
    PROMPT_MODE_OBJECT_DETECTION,
    PROMPT_MODE_COLORED_MASKS,
    PROMPT_MODE_SEGMENTATION_MASKS,
    PROMPT_MODE_TAGGING,
    PROMPT_MODE_CLASSIFIER,
    PROMPT_MODE_VIDEO_TYPE_ONE_WORD,
    PROMPT_MODE_DETAILED_DESCRIPTION,
    PROMPT_MODE_HANDWRITTEN_NOTES,
    PROMPT_MODE_FOOD_RECOGNITION,
    PROMPT_MODE_LOCATION_ID,
    PROMPT_MODE_GEOGUESSR,
    PROMPT_MODE_PRICE_EXTRACTION,
    PROMPT_MODE_PRICE_EXTRACTION_STRICT,
    PROMPT_MODE_MEME_RECOGNITION,
    PROMPT_MODE_TRADLE,
    PROMPT_MODE_TABLE_EXTRACTION,
    PROMPT_MODE_ACTION_ID_VIDEO,
    PROMPT_MODE_WORKOUT_TRACKING,
    PROMPT_MODE_EVENT_ID_VIDEO,
    PROMPT_MODE_FIRE_DETECTION,
    PROMPT_MODE_REAL_VS_AI_VIDEO,
    PROMPT_MODE_COMPLEX_COUNTING,
    PROMPT_MODE_VIDEO_NUMBER_SUMMING,
    PROMPT_MODE_ANIMAL_ID,
    PROMPT_MODE_ANIMAL_ID_UNCERTAINTY,
    PROMPT_MODE_WILDLIFE_DATASET,
    PROMPT_MODE_UNCERTAINTY_CALIBRATION,
    PROMPT_MODE_UNCERTAINTY_CONFIDENCE,
    PROMPT_MODE_UNCERTAINTY_STRICT_JSON,
    PROMPT_MODE_UI_UPLOAD_FILE,
    PROMPT_MODE_UI_NEW_SPREADSHEET,
    PROMPT_MODE_UI_SHARE_PROJECT,
    PROMPT_MODE_UI_OPEN_FILE,
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
      "box_2d": [y1, x1, y2, x2]
    }
  ]
}

Requested: {query}

Requirements:
- Return ONLY valid JSON. No markdown.
- box_2d values are integers on a 0–1000 scale (row-first: y1, x1, y2, x2).
- Ensure y2 > y1 and x2 > x1.
- If the requested object is not visible, return {"detections": []}.
- Use concise labels that match the requested object(s).
"""

_COLORED_MASKS_PROMPT = """Task: Identify and segment the requested object(s) in the image.

Return JSON that matches this schema exactly:
{
  "detections": [
    {
      "label": string,
      "box_2d": [y1, x1, y2, x2],
      "color": string
    }
  ]
}

Requested: {query}

Requirements:
- Return ONLY valid JSON. No markdown.
- box_2d values are integers on a 0–1000 scale (row-first: y1, x1, y2, x2).
- Ensure y2 > y1 and x2 > x1.
- If the requested object is not visible, return {"detections": []}.
- Use concise labels that match the requested object(s).
- Assign a distinct CSS color name to each detected object (e.g. "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow").
- Use different colors for different objects.
"""

_SEGMENTATION_MASKS_PROMPT = """Task: Perform pixel-level instance segmentation on the requested object(s) in the image.

Return JSON that matches this schema exactly:
{
  "segments": [
    {
      "label": string,
      "box_2d": [y1, x1, y2, x2],
      "polygon_2d": [[y, x], [y, x], ...],
      "color": string
    }
  ]
}

Requested: {query}

Requirements:
- Return ONLY valid JSON. No markdown.
- box_2d values are integers on a 0–1000 scale (row-first: y1, x1, y2, x2). Ensure y2 > y1 and x2 > x1.
- polygon_2d is a list of [y, x] coordinate pairs on a 0–1000 scale tracing the object contour.
  Provide at least 6 points per polygon.
  Trace the actual visible boundary of the object, not just the bounding box corners.
- If the requested object is not visible, return {"segments": []}.
- Use concise labels that match the requested object(s).
- Assign a distinct CSS color name to each segment (e.g. "red", "blue", "green", "orange", "purple", "cyan").
- Use different colors for different objects.
"""

_VIDEO_TYPE_ONE_WORD_PROMPT = """Classify the primary video type.

Return exactly one lowercase word.
No punctuation. No explanation. No extra tokens.

Allowed words:
tutorial, documentary, interview, lecture, gameplay, vlog, ad, news, sports, animation, movie, other
"""

_DETAILED_DESCRIPTION_PROMPT = "Describe the image with as much detail as possible."

_HANDWRITTEN_NOTES_PROMPT = (
    "Write the text from the paper. The result should be in English."
)

_FOOD_RECOGNITION_PROMPT = "Describe this dish and predict where it comes from."

_LOCATION_ID_PROMPT = "Where was this photo taken?"

_GEOGUESSR_PROMPT = """You are playing GeoGuessr from a single image. Your goal is to make the best possible map pin guess.

Important rules:
- Be evidence-based.
- Do not hallucinate readable text. If text is unclear, say it is unclear.
- Distinguish strong clues from weak clues.
- If exact location is impossible, choose the most likely representative location.
- Give coordinates even if uncertain, but clearly state confidence.
- Avoid overconfidence.

Image analysis task:
Describe the scene, list all geographic clues, compare possible countries, eliminate unlikely options, then choose the best pin.

Output:
1. Scene summary
2. Key clues
3. Country ranking with probabilities
4. Region/city guess
5. Final pin coordinates
6. Confidence score
7. Google Map coordinates
"""

_PRICE_EXTRACTION_PROMPT = "How much matcha latte cost on this photo."

_PRICE_EXTRACTION_STRICT_PROMPT = (
    "How much matcha latte cost on this photo. Provide only price as output and currency."
)

_MEME_RECOGNITION_PROMPT = (
    "Recognize this meme template. "
    "Provide its name and explain how it become popular."
)

_TRADLE_PROMPT = """You are an expert Tradle solver.

Analyze the Tradle screenshot and guess the mystery country from its export treemap.

Rules:
- Use only visible information.
- Do not invent unreadable text or percentages.
- Focus on the biggest export blocks first.
- Use previous guess feedback if shown: distance, direction, and proximity.

Steps:
1. Read the visible exports and percentages.
2. Identify the main export type:
   oil/gas, minerals, agriculture, textiles, cars, electronics, machinery, chemicals, or mixed.
3. Compare countries that match this export profile.
4. Use distance and direction clues from previous guesses to narrow the location.
5. Give the best final country guess.

Output format:

{
  "visible_exports": ["...", "..."],
  "main_export_profile": "...",
  "top_candidates": [
    {
      "country": "...",
      "reason": "..."
    }
  ],
  "best_guess": "...",
  "confidence": 0-100,
  "next_guess_if_wrong": "..."
}
"""

_TABLE_EXTRACTION_PROMPT = """\
Extract the table from this image exactly as markdown. Do not summarize. \
Preserve row and col order. Then compute total yearly revenue for each region, \
rank regions from highest to lowest, show calculations, and do not skip arithmetic."""

_ACTION_ID_VIDEO_PROMPT = "Which player's jersey number scored the points?"

_WORKOUT_TRACKING_PROMPT = (
    "Count the sets and reps of the squats. "
    "Estimate the weight on the bar and assess the technique."
)

_EVENT_ID_VIDEO_PROMPT = "What number is on the shorts of the man?"

_FIRE_DETECTION_PROMPT = "Detect if fire is happening. Answer only yes or no."

_REAL_VS_AI_VIDEO_PROMPT = """\
Analyze the video for physical inconsistencies, temporal artifacts, lighting issues, \
motion problems, and object consistency. Decide if the video is real or AI generated. \
Respond with a single word only: "Real" or "AI generated". In the next line, specify why."""

_COMPLEX_COUNTING_PROMPT = """\
Analyze the image and identify every visible coin. \
Classify each coin as either euro or Polish Zloty. \
Sum the values within each currency separately, \
then count the total number of visible coins. Do not include explanations."""

_VIDEO_NUMBER_SUMMING_PROMPT = "Write each number shown and then sum."

_ANIMAL_ID_PROMPT = (
    "Identify the animal in this image. Give the most specific answer possible."
)

_ANIMAL_ID_UNCERTAINTY_PROMPT = (
    "Identify the animal in this image. "
    "If species is uncertain, give top 3 possibilities with confidence."
)

_WILDLIFE_DATASET_PROMPT = (
    "Describe the image for a wildlife monitoring dataset. "
    "Include animal type, visible features, image quality issues, and uncertainty."
)

_UNCERTAINTY_CALIBRATION_PROMPT = "What animal is shown in this image?"

_UNCERTAINTY_CONFIDENCE_PROMPT = (
    "Identify the animal in this image. "
    "If uncertain, give top 3 possibilities with confidence percentages."
)

_UNCERTAINTY_STRICT_JSON_PROMPT = """\
Answer only in this JSON format:
{"confident": true_or_false, "most_likely": "...", "confidence": 0.0, \
"alternatives": ["...", "..."], "reason": "..."}"""

_UI_UPLOAD_FILE_PROMPT = (
    "The user wants to upload a file. "
    "What should they click next? Answer with the exact UI element name only."
)

_UI_NEW_SPREADSHEET_PROMPT = (
    "The user wants to create a spreadsheet. "
    "What should they click next? Answer with the exact UI element name only."
)

_UI_SHARE_PROJECT_PROMPT = (
    "The user wants to share the current project with a teammate. "
    "What should they click? Answer with the exact UI element name only."
)

_UI_OPEN_FILE_PROMPT = (
    "The user wants to open the file named Homepage_v2.fig. "
    "What should they click? Answer with the exact file name only."
)


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
    if mode == PROMPT_MODE_COLORED_MASKS:
        return _COLORED_MASKS_PROMPT
    if mode == PROMPT_MODE_SEGMENTATION_MASKS:
        return _SEGMENTATION_MASKS_PROMPT
    if mode == PROMPT_MODE_VIDEO_TYPE_ONE_WORD:
        return _VIDEO_TYPE_ONE_WORD_PROMPT
    if mode == PROMPT_MODE_DETAILED_DESCRIPTION:
        return _DETAILED_DESCRIPTION_PROMPT
    if mode == PROMPT_MODE_HANDWRITTEN_NOTES:
        return _HANDWRITTEN_NOTES_PROMPT
    if mode == PROMPT_MODE_FOOD_RECOGNITION:
        return _FOOD_RECOGNITION_PROMPT
    if mode == PROMPT_MODE_LOCATION_ID:
        return _LOCATION_ID_PROMPT
    if mode == PROMPT_MODE_GEOGUESSR:
        return _GEOGUESSR_PROMPT
    if mode == PROMPT_MODE_PRICE_EXTRACTION:
        return _PRICE_EXTRACTION_PROMPT
    if mode == PROMPT_MODE_PRICE_EXTRACTION_STRICT:
        return _PRICE_EXTRACTION_STRICT_PROMPT
    if mode == PROMPT_MODE_MEME_RECOGNITION:
        return _MEME_RECOGNITION_PROMPT
    if mode == PROMPT_MODE_TRADLE:
        return _TRADLE_PROMPT
    if mode == PROMPT_MODE_TABLE_EXTRACTION:
        return _TABLE_EXTRACTION_PROMPT
    if mode == PROMPT_MODE_ACTION_ID_VIDEO:
        return _ACTION_ID_VIDEO_PROMPT
    if mode == PROMPT_MODE_WORKOUT_TRACKING:
        return _WORKOUT_TRACKING_PROMPT
    if mode == PROMPT_MODE_EVENT_ID_VIDEO:
        return _EVENT_ID_VIDEO_PROMPT
    if mode == PROMPT_MODE_FIRE_DETECTION:
        return _FIRE_DETECTION_PROMPT
    if mode == PROMPT_MODE_REAL_VS_AI_VIDEO:
        return _REAL_VS_AI_VIDEO_PROMPT
    if mode == PROMPT_MODE_COMPLEX_COUNTING:
        return _COMPLEX_COUNTING_PROMPT
    if mode == PROMPT_MODE_VIDEO_NUMBER_SUMMING:
        return _VIDEO_NUMBER_SUMMING_PROMPT
    if mode == PROMPT_MODE_ANIMAL_ID:
        return _ANIMAL_ID_PROMPT
    if mode == PROMPT_MODE_ANIMAL_ID_UNCERTAINTY:
        return _ANIMAL_ID_UNCERTAINTY_PROMPT
    if mode == PROMPT_MODE_WILDLIFE_DATASET:
        return _WILDLIFE_DATASET_PROMPT
    if mode == PROMPT_MODE_UNCERTAINTY_CALIBRATION:
        return _UNCERTAINTY_CALIBRATION_PROMPT
    if mode == PROMPT_MODE_UNCERTAINTY_CONFIDENCE:
        return _UNCERTAINTY_CONFIDENCE_PROMPT
    if mode == PROMPT_MODE_UNCERTAINTY_STRICT_JSON:
        return _UNCERTAINTY_STRICT_JSON_PROMPT
    if mode == PROMPT_MODE_UI_UPLOAD_FILE:
        return _UI_UPLOAD_FILE_PROMPT
    if mode == PROMPT_MODE_UI_NEW_SPREADSHEET:
        return _UI_NEW_SPREADSHEET_PROMPT
    if mode == PROMPT_MODE_UI_SHARE_PROJECT:
        return _UI_SHARE_PROJECT_PROMPT
    if mode == PROMPT_MODE_UI_OPEN_FILE:
        return _UI_OPEN_FILE_PROMPT
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
