"""Gradio UI for side-by-side multimodal model comparison."""

from __future__ import annotations

from typing import Any, Literal, cast

import gradio as gr

from visual_experimentation_app.compare_service import execute_and_persist_compare
from visual_experimentation_app.config import TargetDefaults, get_settings
from visual_experimentation_app.payload_builder import parse_json_object
from visual_experimentation_app.result_store import list_compare_history, load_compare_result
from visual_experimentation_app.schemas import (
    CompareRequest,
    CompareResult,
    CompareTargetConfig,
    CompareTargetResult,
)
from visual_experimentation_app.ui_presets import (
    DEFAULT_CUSTOM_PROMPT,
    DEFAULT_TAG_CATEGORIES,
    PROMPT_MODE_CHOICES,
    PROMPT_MODE_CLASSIFIER,
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_TAGGING,
    SEGMENTATION_PROFILE_CHOICES,
    SEGMENTATION_PROFILE_OFF,
    build_prompt_for_mode,
    segmentation_values_for_profile,
)

APP_THEME = gr.themes.Ocean()

CUSTOM_CSS = """
.gradio-container {
  max-width: 1800px !important;
  padding-top: 1.5rem !important;
  color: #e8edf6;
  background:
    radial-gradient(circle at top left, rgba(249, 115, 22, 0.2), transparent 30rem),
    radial-gradient(circle at 85% 5%, rgba(56, 189, 248, 0.14), transparent 28rem),
    linear-gradient(180deg, #060a12 0%, #0b1321 44%, #0b1424 100%);
}

#mm-header {
  padding: 2rem 2.2rem;
  border-radius: 28px;
  border: 1px solid rgba(110, 130, 164, 0.26);
  background: linear-gradient(
    135deg,
    rgba(16, 24, 39, 0.94),
    rgba(14, 30, 50, 0.95)
  );
  box-shadow: 0 22px 46px rgba(0, 0, 0, 0.36);
  margin-bottom: 1rem;
}

#mm-header .kicker {
  margin: 0 0 0.4rem 0;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 0.76rem;
  color: #f6ad55;
}

#mm-header h1 {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(2.2rem, 4vw, 3.5rem);
  line-height: 1.05;
  color: #f8fafc;
}

#mm-header .lead {
  max-width: 68rem;
  margin: 0.8rem 0 0 0;
  font-size: 1.06rem;
  line-height: 1.65;
  color: #c9d7ec;
}

.panel,
.compare-card {
  border: 1px solid rgba(110, 130, 164, 0.24) !important;
  border-radius: 24px !important;
  background: rgba(13, 20, 34, 0.86) !important;
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.3);
}

.control-panel {
  padding: 1.2rem 1.35rem !important;
}

.settings-shell {
  margin-top: 1.1rem;
}

.compare-card {
  padding: 1rem 1.1rem 1.2rem 1.1rem !important;
  background:
    linear-gradient(180deg, rgba(13, 20, 34, 0.95), rgba(10, 16, 28, 0.95)) !important;
}

.compare-card h2,
.compare-card h3,
.compare-card h4,
#compare-summary h2,
#compare-summary h3,
#compare-summary h4 {
  margin: 0.2rem 0 0.45rem 0 !important;
}

#compare-summary {
  padding: 0.95rem 1.15rem !important;
  border-radius: 18px;
  border: 1px solid rgba(95, 114, 146, 0.28);
  color: #e8edf6 !important;
  background: linear-gradient(
    135deg,
    rgba(30, 41, 59, 0.95),
    rgba(15, 23, 42, 0.95)
  );
}

.results-heading {
  margin-bottom: 0.4rem;
}

.results-heading h3 {
  margin: 0 !important;
}

.output-pane {
  min-height: 44rem;
  max-height: 62rem;
  overflow: auto;
  padding: 1.1rem 1.2rem !important;
  border-radius: 18px;
  border: 1px solid rgba(90, 107, 135, 0.35);
  color: #f8fafc !important;
  background: rgba(9, 14, 24, 0.96);
}

.output-pane,
.output-pane *,
.readable-output {
  color: #f8fafc !important;
}

.readable-output p,
.readable-output li {
  font-size: 1.04rem !important;
  line-height: 1.68 !important;
}

.readable-output code {
  font-size: 0.94em;
}

.output-pane h3 {
  display: inline-flex;
  align-items: center;
  margin: 0.1rem 0 0.7rem 0 !important;
  padding: 0.25rem 0.8rem;
  border-radius: 999px;
  font-size: 0.9rem !important;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #f8fafc !important;
  background: linear-gradient(135deg, rgba(249, 115, 22, 0.22), rgba(56, 189, 248, 0.16));
  border: 1px solid rgba(148, 163, 184, 0.24);
}

.output-pane hr {
  margin: 1.1rem 0 1.3rem 0;
  border: none;
  border-top: 1px solid rgba(110, 130, 164, 0.28);
}

.output-pane .reasoning-shell {
  margin: 0 0 1.15rem 0;
  padding: 0.95rem 1rem 1rem 1rem;
  border-radius: 18px;
  border: 1px solid rgba(110, 130, 164, 0.26);
  background: linear-gradient(180deg, rgba(18, 27, 45, 0.96), rgba(11, 19, 33, 0.98));
}

.output-pane .reasoning-shell summary {
  cursor: pointer;
  list-style: none;
  font-size: 0.9rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #ffd59c;
}

.output-pane .reasoning-shell summary::-webkit-details-marker {
  display: none;
}

.output-pane .reasoning-shell summary::after {
  margin-left: 0.75rem;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: none;
  color: #94a3b8;
}

.output-pane .reasoning-shell[open] summary::after {
  content: "Click to collapse";
}

.output-pane .reasoning-shell:not([open]) summary::after {
  content: "Click to expand";
}

.output-pane .reasoning-shell pre {
  margin: 0.9rem 0 0 0;
  padding: 1rem 1.05rem;
  border-radius: 16px;
  border: 1px solid rgba(110, 130, 164, 0.22);
  background: rgba(7, 12, 21, 0.8);
  color: #e8edf6;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-x: auto;
  font-size: 0.96rem;
  line-height: 1.65;
}

.compact-note p {
  margin: 0.15rem 0 0 0 !important;
  color: #c7d2e5 !important;
}

@media (max-width: 1100px) {
  .output-pane {
    min-height: 26rem;
    max-height: none;
  }
}
"""

GEMMA_MAX_SOFT_TOKEN_CHOICES: list[tuple[str, str]] = [
    ("Auto (no override)", ""),
    ("70", "70"),
    ("140", "140"),
    ("280", "280"),
    ("560", "560"),
    ("1120", "1120"),
]
GEMMA_MAX_SOFT_TOKEN_ALLOWED = {70, 140, 280, 560, 1120}


def ui_theme() -> object:
    """Return the Gradio theme used by the compare lab."""
    return APP_THEME


def ui_css() -> str:
    """Return custom CSS for the compare lab Gradio app."""
    return CUSTOM_CSS


def _extract_paths(upload_value: Any) -> list[str]:
    if upload_value is None:
        return []
    if isinstance(upload_value, str):
        return [upload_value]
    if isinstance(upload_value, list):
        paths: list[str] = []
        for item in upload_value:
            paths.extend(_extract_paths(item))
        return paths
    if isinstance(upload_value, dict):
        for key in ("path", "name"):
            value = upload_value.get(key)
            if isinstance(value, str) and value.strip():
                return [value]
        return []

    name = getattr(upload_value, "name", None)
    if isinstance(name, str) and name.strip():
        return [name]
    return []


def _image_preview_value(upload_value: Any) -> list[str]:
    """Return image upload paths for inline preview."""
    return _extract_paths(upload_value)


def _video_preview_value(upload_value: Any) -> str | None:
    """Return first uploaded video path for inline preview."""
    paths = _extract_paths(upload_value)
    return paths[0] if paths else None


def _clean_text(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        return raw_value
    return str(raw_value)


def _csv_to_str_list(raw_value: Any) -> list[str]:
    cleaned = _clean_text(raw_value)
    if not cleaned.strip():
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _parse_gemma_max_soft_tokens(raw_value: Any) -> int | None:
    cleaned = _clean_text(raw_value).strip()
    if not cleaned:
        return None
    try:
        value = int(cleaned)
    except ValueError as exc:
        raise ValueError("Gemma max_soft_tokens must be one of 70/140/280/560/1120.") from exc
    if value not in GEMMA_MAX_SOFT_TOKEN_ALLOWED:
        raise ValueError("Gemma max_soft_tokens must be one of 70/140/280/560/1120.")
    return value


def _parse_thinking_token_budget(raw_value: Any) -> int | None:
    cleaned = _clean_text(raw_value).strip()
    if not cleaned:
        return None
    try:
        value = int(float(cleaned))
    except ValueError as exc:
        raise ValueError("thinking_token_budget must be a positive integer.") from exc
    if value <= 0:
        return None
    return value


def _apply_prompt_mode(mode: str, current_prompt: str, tag_categories_csv: str) -> str:
    """Return prompt text based on the selected UI preset mode."""
    return build_prompt_for_mode(
        mode=_clean_text(mode).strip(),
        current_prompt=_clean_text(current_prompt),
        tag_categories_csv=_clean_text(tag_categories_csv),
    )


def _refresh_prompt_for_tagging(
    mode: str,
    current_prompt: str,
    tag_categories_csv: str,
) -> str:
    """Refresh prompt when category-driven preset modes are active."""
    clean_mode = _clean_text(mode).strip()
    if clean_mode not in {PROMPT_MODE_TAGGING, PROMPT_MODE_CLASSIFIER}:
        return _clean_text(current_prompt)
    return build_prompt_for_mode(
        mode=clean_mode,
        current_prompt=_clean_text(current_prompt),
        tag_categories_csv=_clean_text(tag_categories_csv),
    )


def _apply_segmentation_profile(
    profile: str,
    current_duration: float,
    current_overlap: float,
) -> tuple[float, float]:
    """Apply segmentation defaults from the selected profile."""
    return segmentation_values_for_profile(
        profile=_clean_text(profile).strip(),
        current_duration=float(current_duration),
        current_overlap=float(current_overlap),
    )


def _build_target_config(
    *,
    label: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    use_model_defaults: bool,
    max_tokens: int,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    thinking_mode: str,
    show_reasoning: bool,
    measure_ttft: bool,
    thinking_token_budget: int,
    gemma_max_soft_tokens: str,
    extra_body_json: str,
    extra_headers_json: str,
) -> CompareTargetConfig:
    mode = thinking_mode.strip().lower()
    if mode not in {"auto", "on", "off"}:
        mode = "auto"
    thinking_mode_value = cast(Literal["auto", "on", "off"], mode)

    extra_body = parse_json_object(_clean_text(extra_body_json), field_name="Extra Body JSON")
    extra_headers = parse_json_object(
        _clean_text(extra_headers_json),
        field_name="Extra Headers JSON",
    )

    return CompareTargetConfig(
        label=_clean_text(label).strip(),
        base_url=_clean_text(base_url).strip() or None,
        model=_clean_text(model).strip() or None,
        api_key=_clean_text(api_key).strip() or None,
        timeout_seconds=float(timeout_seconds),
        use_model_defaults=bool(use_model_defaults),
        max_tokens=int(max_tokens),
        max_completion_tokens=int(max_completion_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
        thinking_mode=thinking_mode_value,
        show_reasoning=bool(show_reasoning),
        measure_ttft=bool(measure_ttft),
        thinking_token_budget=_parse_thinking_token_budget(thinking_token_budget),
        gemma_max_soft_tokens=_parse_gemma_max_soft_tokens(gemma_max_soft_tokens),
        request_extra_body=extra_body,
        request_extra_headers={str(key): str(value) for key, value in extra_headers.items()},
    )


def _build_compare_request(
    *,
    prompt: str,
    text_input: str,
    text_only: bool,
    image_upload: Any,
    video_upload: Any,
    preprocess_images: bool,
    preprocess_video: bool,
    target_height: int,
    target_video_fps: float,
    safe_video_sampling: bool,
    video_sampling_fps: float,
    segment_max_duration_s: float,
    segment_overlap_s: float,
    segment_workers: int,
    image_cache_uuids_text: str,
    video_cache_uuids_text: str,
    disable_caching: bool,
    model_a: CompareTargetConfig,
    model_b: CompareTargetConfig,
) -> CompareRequest:
    return CompareRequest(
        prompt=_clean_text(prompt).strip(),
        text_input=_clean_text(text_input).strip() or None,
        text_only=bool(text_only),
        image_paths=_extract_paths(image_upload),
        video_paths=_extract_paths(video_upload),
        preprocess_images=bool(preprocess_images),
        preprocess_video=bool(preprocess_video),
        target_height=int(target_height),
        target_video_fps=float(target_video_fps),
        safe_video_sampling=bool(safe_video_sampling),
        video_sampling_fps=(
            None if bool(safe_video_sampling) else float(video_sampling_fps)
        ),
        segment_max_duration_s=float(segment_max_duration_s),
        segment_overlap_s=float(segment_overlap_s),
        segment_workers=int(segment_workers),
        image_cache_uuids=_csv_to_str_list(image_cache_uuids_text),
        video_cache_uuids=_csv_to_str_list(video_cache_uuids_text),
        disable_caching=bool(disable_caching),
        model_a=model_a,
        model_b=model_b,
    )


def _build_compare_summary(result: CompareResult) -> str:
    return (
        f"## Compare `{result.compare_id}`\n"
        f"- Status: **{result.status}**\n"
        f"- Shared preprocess: `{result.timings.preprocess_ms:.1f} ms`\n"
        f"- Total wall time: `{result.timings.total_ms:.1f} ms`\n"
        f"- Models: `{result.model_a_result.label}` vs `{result.model_b_result.label}`"
    )


def _build_target_status_markdown(result: CompareTargetResult) -> str:
    effective = result.effective_params
    model = effective.get("model", "")
    base_url = effective.get("base_url", "")
    lines = [
        f"### {result.label}",
        f"- Status: **{result.status}**",
    ]
    if model:
        lines.append(f"- Model: `{model}`")
    if base_url:
        lines.append(f"- Base URL: `{base_url}`")
    lines.append(f"- Request latency: `{result.timings.request_ms:.1f} ms`")
    if result.timings.ttft_ms is not None:
        lines.append(f"- TTFT: `{result.timings.ttft_ms:.1f} ms`")
    if result.token_usage.prompt_tokens is not None:
        lines.append(f"- Prompt tokens: `{result.token_usage.prompt_tokens}`")
    if result.token_usage.output_tokens is not None:
        lines.append(f"- Output tokens: `{result.token_usage.output_tokens}`")
    if result.token_usage.tokens_per_second is not None:
        lines.append(f"- Output tok/sec: `{result.token_usage.tokens_per_second:.3f}`")
    if result.token_usage.total_tokens is not None:
        lines.append(f"- Total tokens: `{result.token_usage.total_tokens}`")
    if result.error:
        lines.append(f"- Error: {result.error}")
    return "\n".join(lines)


def _build_effective_request_markdown(result: CompareTargetResult) -> str:
    """Render a concise summary of effective request parameters."""
    effective = result.effective_params
    sent = effective.get("sent_generation_params", {})
    omitted = effective.get("omitted_for_model_defaults", [])
    defaults_info = effective.get("model_defaults_info", {})
    error_details = effective.get("error_details")

    if not isinstance(sent, dict):
        sent = {}
    if not isinstance(omitted, list):
        omitted = []

    sent_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in sent.items())
    if not sent_lines:
        sent_lines = "- _none_"

    omitted_lines = "\n".join(f"- `{item}`" for item in omitted)
    if not omitted_lines:
        omitted_lines = "- _none_"

    defaults_md = ""
    if effective.get("use_model_defaults"):
        info_source = (
            defaults_info.get("source", "unknown")
            if isinstance(defaults_info, dict)
            else "unknown"
        )
        info_path = defaults_info.get("path", "") if isinstance(defaults_info, dict) else ""
        info_message = (
            defaults_info.get("message", "") if isinstance(defaults_info, dict) else ""
        )
        sampling_values = (
            defaults_info.get("sampling_values", {})
            if isinstance(defaults_info, dict)
            else {}
        )
        if isinstance(sampling_values, dict) and sampling_values:
            sampling_lines = "\n".join(
                f"- `{key}`: `{value}`" for key, value in sampling_values.items()
            )
        else:
            sampling_lines = "- _not discoverable from local generation_config_"

        path_line = f"- `generation_config_path`: `{info_path}`\n" if info_path else ""
        message_line = f"- note: {info_message}\n" if info_message else ""
        defaults_md = (
            "\n\n**Model/Server Defaults (Best Effort)**\n"
            f"- `source`: `{info_source}`\n"
            f"{path_line}"
            f"{message_line}"
            "\n**Resolved Sampling Defaults**\n"
            f"{sampling_lines}"
        )

    error_md = ""
    if isinstance(error_details, dict) and error_details:
        error_md = (
            "\n\n**Error Details**\n"
            f"- `error_type`: `{error_details.get('error_type', '')}`\n"
            "- `is_video_processor_error`: "
            f"`{error_details.get('is_video_processor_error', False)}`"
        )

    return (
        "### Effective Request\n"
        f"- `use_model_defaults`: `{effective.get('use_model_defaults', False)}`\n"
        f"- `text_only`: `{effective.get('text_only', False)}`\n"
        f"- `disable_caching`: `{effective.get('disable_caching', False)}`\n"
        f"- `model`: `{effective.get('model', '')}`\n"
        f"- `base_url`: `{effective.get('base_url', '')}`\n"
        "- `thinking_token_budget_supported`: "
        f"`{effective.get('thinking_token_budget_supported', False)}`\n"
        "- `thinking_token_budget_applied`: "
        f"`{effective.get('thinking_token_budget_applied', None)}`\n"
        "- `gemma_max_soft_tokens_applied`: "
        f"`{effective.get('gemma_max_soft_tokens_applied', None)}`\n\n"
        "**Sent Generation Params**\n"
        f"{sent_lines}\n\n"
        "**Omitted For Model Defaults**\n"
        f"{omitted_lines}"
        f"{defaults_md}"
        f"{error_md}"
    )


def _run_compare(
    prompt: str,
    text_input: str,
    text_only: bool,
    image_upload: Any,
    video_upload: Any,
    preprocess_images: bool,
    preprocess_video: bool,
    target_height: int,
    target_video_fps: float,
    safe_video_sampling: bool,
    video_sampling_fps: float,
    segment_max_duration_s: float,
    segment_overlap_s: float,
    segment_workers: int,
    image_cache_uuids: str,
    video_cache_uuids: str,
    disable_caching: bool,
    model_a_label: str,
    model_a_base_url: str,
    model_a_model: str,
    model_a_api_key: str,
    model_a_timeout_seconds: float,
    model_a_use_model_defaults: bool,
    model_a_max_tokens: int,
    model_a_max_completion_tokens: int,
    model_a_temperature: float,
    model_a_top_p: float,
    model_a_top_k: int,
    model_a_presence_penalty: float,
    model_a_frequency_penalty: float,
    model_a_thinking_mode: str,
    model_a_show_reasoning: bool,
    model_a_measure_ttft: bool,
    model_a_thinking_token_budget: int,
    model_a_gemma_max_soft_tokens: str,
    model_a_extra_body_json: str,
    model_a_extra_headers_json: str,
    model_b_label: str,
    model_b_base_url: str,
    model_b_model: str,
    model_b_api_key: str,
    model_b_timeout_seconds: float,
    model_b_use_model_defaults: bool,
    model_b_max_tokens: int,
    model_b_max_completion_tokens: int,
    model_b_temperature: float,
    model_b_top_p: float,
    model_b_top_k: int,
    model_b_presence_penalty: float,
    model_b_frequency_penalty: float,
    model_b_thinking_mode: str,
    model_b_show_reasoning: bool,
    model_b_measure_ttft: bool,
    model_b_thinking_token_budget: int,
    model_b_gemma_max_soft_tokens: str,
    model_b_extra_body_json: str,
    model_b_extra_headers_json: str,
) -> tuple[str, dict[str, Any], str, str, str, str, str, str]:
    try:
        target_a = _build_target_config(
            label=model_a_label,
            base_url=model_a_base_url,
            model=model_a_model,
            api_key=model_a_api_key,
            timeout_seconds=float(model_a_timeout_seconds),
            use_model_defaults=bool(model_a_use_model_defaults),
            max_tokens=int(model_a_max_tokens),
            max_completion_tokens=int(model_a_max_completion_tokens),
            temperature=float(model_a_temperature),
            top_p=float(model_a_top_p),
            top_k=int(model_a_top_k),
            presence_penalty=float(model_a_presence_penalty),
            frequency_penalty=float(model_a_frequency_penalty),
            thinking_mode=model_a_thinking_mode,
            show_reasoning=bool(model_a_show_reasoning),
            measure_ttft=bool(model_a_measure_ttft),
            thinking_token_budget=int(model_a_thinking_token_budget),
            gemma_max_soft_tokens=model_a_gemma_max_soft_tokens,
            extra_body_json=model_a_extra_body_json,
            extra_headers_json=model_a_extra_headers_json,
        )
        target_b = _build_target_config(
            label=model_b_label,
            base_url=model_b_base_url,
            model=model_b_model,
            api_key=model_b_api_key,
            timeout_seconds=float(model_b_timeout_seconds),
            use_model_defaults=bool(model_b_use_model_defaults),
            max_tokens=int(model_b_max_tokens),
            max_completion_tokens=int(model_b_max_completion_tokens),
            temperature=float(model_b_temperature),
            top_p=float(model_b_top_p),
            top_k=int(model_b_top_k),
            presence_penalty=float(model_b_presence_penalty),
            frequency_penalty=float(model_b_frequency_penalty),
            thinking_mode=model_b_thinking_mode,
            show_reasoning=bool(model_b_show_reasoning),
            measure_ttft=bool(model_b_measure_ttft),
            thinking_token_budget=int(model_b_thinking_token_budget),
            gemma_max_soft_tokens=model_b_gemma_max_soft_tokens,
            extra_body_json=model_b_extra_body_json,
            extra_headers_json=model_b_extra_headers_json,
        )
        request = _build_compare_request(
            prompt=prompt,
            text_input=text_input,
            text_only=bool(text_only),
            image_upload=image_upload,
            video_upload=video_upload,
            preprocess_images=preprocess_images,
            preprocess_video=preprocess_video,
            target_height=int(target_height),
            target_video_fps=float(target_video_fps),
            safe_video_sampling=bool(safe_video_sampling),
            video_sampling_fps=float(video_sampling_fps),
            segment_max_duration_s=float(segment_max_duration_s),
            segment_overlap_s=float(segment_overlap_s),
            segment_workers=int(segment_workers),
            image_cache_uuids_text=image_cache_uuids,
            video_cache_uuids_text=video_cache_uuids,
            disable_caching=bool(disable_caching),
            model_a=target_a,
            model_b=target_b,
        )
    except Exception as exc:  # noqa: BLE001
        message = f"**Input error:** {exc}"
        return (
            message,
            {},
            message,
            "_Invalid request._",
            message,
            message,
            "_Invalid request._",
            message,
        )

    result = execute_and_persist_compare(request)
    return (
        _build_compare_summary(result),
        result.model_dump(),
        _build_target_status_markdown(result.model_a_result),
        result.model_a_result.output_text.strip() or "_No output text returned._",
        _build_effective_request_markdown(result.model_a_result),
        _build_target_status_markdown(result.model_b_result),
        result.model_b_result.output_text.strip() or "_No output text returned._",
        _build_effective_request_markdown(result.model_b_result),
    )


def _refresh_history() -> tuple[list[list[Any]], dict[str, Any]]:
    items = list_compare_history(limit=200)
    rows = [
        [
            item.compare_id,
            item.status,
            item.created_at,
            item.model_a_label,
            item.model_b_label,
            round(item.total_ms, 2),
        ]
        for item in items
    ]
    dropdown = gr.update(
        choices=[item.compare_id for item in items],
        value=items[0].compare_id if items else None,
    )
    return rows, dropdown


def _load_history_detail(compare_id: str) -> dict[str, Any]:
    if not compare_id:
        return {}
    result = load_compare_result(compare_id)
    if result is None:
        return {"error": f"Compare not found: {compare_id}"}
    return result.model_dump()


def build_ui_blocks() -> gr.Blocks:
    """Build and return compare-lab Gradio Blocks."""
    settings = get_settings()

    def build_target_panel(
        *,
        title: str,
        defaults: TargetDefaults,
    ) -> list[gr.components.Component]:
        with gr.Accordion(title, open=True, elem_classes=["panel"]):
            gr.Markdown(
                (
                    "These values are preloaded from environment defaults and only need "
                    "changes when you want to deviate for a specific compare."
                ),
                elem_classes=["compact-note"],
            )
            label = gr.Textbox(label="Label", value=defaults.label)
            base_url = gr.Textbox(label="vLLM Base URL", value=defaults.base_url)
            model = gr.Textbox(label="Model", value=defaults.model)
            api_key = gr.Textbox(label="API Key", value=defaults.api_key, type="password")
            timeout_seconds = gr.Slider(
                minimum=10,
                maximum=600,
                step=5,
                value=settings.default_timeout_seconds,
                label="Request Timeout (seconds)",
            )
            use_model_defaults = gr.Checkbox(
                label="Use model/server defaults",
                value=defaults.request_defaults.use_model_defaults,
            )
            max_tokens = gr.Slider(
                minimum=1,
                maximum=131072,
                step=1,
                value=defaults.request_defaults.max_tokens,
                label="max_tokens",
            )
            max_completion_tokens = gr.Slider(
                minimum=1,
                maximum=131072,
                step=1,
                value=defaults.request_defaults.max_completion_tokens,
                label="max_completion_tokens",
            )
            temperature = gr.Slider(
                minimum=0,
                maximum=2,
                step=0.05,
                value=defaults.request_defaults.temperature,
                label="temperature",
            )
            top_p = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=defaults.request_defaults.top_p,
                label="top_p",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=200,
                step=1,
                value=defaults.request_defaults.top_k,
                label="top_k",
            )
            presence_penalty = gr.Slider(
                minimum=-2,
                maximum=2,
                step=0.1,
                value=defaults.request_defaults.presence_penalty,
                label="presence_penalty",
            )
            frequency_penalty = gr.Slider(
                minimum=-2,
                maximum=2,
                step=0.1,
                value=defaults.request_defaults.frequency_penalty,
                label="frequency_penalty",
            )
            thinking_mode = gr.Radio(
                choices=["auto", "on", "off"],
                label="thinking_mode",
                value=defaults.request_defaults.thinking_mode,
            )
            show_reasoning = gr.Checkbox(
                label="Show reasoning output",
                value=defaults.request_defaults.show_reasoning,
            )
            measure_ttft = gr.Checkbox(
                label="Measure TTFT via streaming",
                value=defaults.request_defaults.measure_ttft,
            )
            thinking_token_budget = gr.Slider(
                minimum=0,
                maximum=8192,
                step=1,
                value=0,
                label="thinking_token_budget",
                info=(
                    "Qwen3-family reasoning budget override. "
                    "Set 0 to leave it unset. Gemma 4 ignores this; "
                    "use Gemma max_soft_tokens below."
                ),
            )
            gemma_max_soft_tokens = gr.Dropdown(
                label="Gemma max_soft_tokens override",
                choices=GEMMA_MAX_SOFT_TOKEN_CHOICES,
                value="",
                info="Gemma-only vision token budget override: 70/140/280/560/1120.",
            )
            extra_body_json = gr.Textbox(
                label="Extra Body JSON",
                lines=5,
                value="{}",
            )
            extra_headers_json = gr.Textbox(
                label="Extra Headers JSON",
                lines=3,
                value="{}",
            )
        return [
            label,
            base_url,
            model,
            api_key,
            timeout_seconds,
            use_model_defaults,
            max_tokens,
            max_completion_tokens,
            temperature,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            thinking_mode,
            show_reasoning,
            measure_ttft,
            thinking_token_budget,
            gemma_max_soft_tokens,
            extra_body_json,
            extra_headers_json,
        ]

    with gr.Blocks(title="Multimodal Compare Lab") as app:
        gr.HTML(
            """
            <section id="mm-header">
              <p class="kicker">COMPARE LAB</p>
              <h1>Multimodal Compare Lab</h1>
              <p class="lead">
                Upload media once, keep preprocessing shared, and compare Qwen and Gemma
                side by side with independent generation settings.
              </p>
            </section>
            """
        )

        with gr.Tab("Compare"):
            gr.Markdown(
                (
                    "Configure shared media once, then send the same prompt and payload "
                    "to both models."
                ),
                elem_classes=["tab-note"],
            )
            with gr.Column(elem_classes=["panel", "control-panel"]):
                with gr.Row(equal_height=False):
                    prompt_mode = gr.Radio(
                        choices=PROMPT_MODE_CHOICES,
                        label="Prompt Mode",
                        value=PROMPT_MODE_CUSTOM,
                    )
                    tag_categories = gr.Textbox(
                        label="Tag Categories (CSV)",
                        value=DEFAULT_TAG_CATEGORIES,
                        info="Used by Tagging and Classifier presets.",
                    )
                prompt = gr.Textbox(label="Prompt", lines=5, value=DEFAULT_CUSTOM_PROMPT)
                text_input = gr.Textbox(
                    label="Additional Text Input / Query (optional)",
                    lines=3,
                    info="If the prompt contains {query}, this value replaces it.",
                )
                text_only = gr.Checkbox(
                    label="Text-only compare (ignore uploaded images/videos)",
                    value=False,
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        image_upload = gr.File(label="Images (multiple)", file_count="multiple")
                        image_preview = gr.Gallery(
                            label="Image Preview",
                            columns=4,
                            object_fit="contain",
                            height=220,
                        )
                    with gr.Column(scale=2):
                        video_upload = gr.File(label="Videos (up to 2)", file_count="multiple")
                        video_preview = gr.Video(label="Video Preview")
                    with gr.Column(scale=2, elem_classes=["run-column"]):
                        compare_button = gr.Button("Run Compare", variant="primary")
                        compare_summary = gr.Markdown(
                            "Idle.",
                            elem_id="compare-summary",
                            elem_classes=["readable-output"],
                        )

                with gr.Accordion("Shared Preprocessing + Segmentation", open=False):
                    preprocess_images = gr.Checkbox(label="Preprocess images", value=True)
                    preprocess_video = gr.Checkbox(label="Preprocess video", value=True)
                    target_height = gr.Slider(
                        minimum=128,
                        maximum=1080,
                        step=16,
                        value=settings.default_target_height,
                        label="Target Max Height (px)",
                    )
                    target_video_fps = gr.Slider(
                        minimum=0.1,
                        maximum=30.0,
                        step=0.1,
                        value=settings.default_video_fps,
                        label="Target Video FPS",
                    )
                    safe_video_sampling = gr.Checkbox(
                        label="Safe video processor defaults (do_sample_frames=false)",
                        value=settings.default_safe_video_sampling,
                    )
                    video_sampling_fps = gr.Slider(
                        minimum=0.1,
                        maximum=30.0,
                        step=0.1,
                        value=settings.default_video_fps,
                        label="Video sampling fps when safe mode is OFF",
                    )
                    segmentation_profile = gr.Radio(
                        choices=SEGMENTATION_PROFILE_CHOICES,
                        label="Segmentation Profile",
                        value=SEGMENTATION_PROFILE_OFF,
                    )
                    segment_max_duration_s = gr.Slider(
                        minimum=0.0,
                        maximum=3600.0,
                        step=1.0,
                        value=0.0,
                        label="Segment max duration (seconds, 0 disables segmentation)",
                    )
                    segment_overlap_s = gr.Slider(
                        minimum=0.0,
                        maximum=300.0,
                        step=0.5,
                        value=0.0,
                        label="Segment overlap (seconds)",
                    )
                    segment_workers = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=1,
                        label="Chunk Parallel Requests",
                    )

                with gr.Accordion("Shared Cache Controls", open=False):
                    image_cache_uuids = gr.Textbox(
                        label="Image cache UUIDs (comma-separated)",
                        placeholder="img-uuid-1,img-uuid-2",
                    )
                    video_cache_uuids = gr.Textbox(
                        label="Video cache UUIDs (comma-separated, optional)",
                        placeholder="vid-uuid-1,vid-uuid-2",
                    )
                    disable_caching = gr.Checkbox(
                        label="Disable cache reuse for measurement",
                        value=False,
                    )

            with gr.Accordion(
                "Model Settings",
                open=True,
                elem_classes=["panel", "settings-shell"],
            ):
                gr.Markdown(
                    (
                        "Keep the defaults for quick side-by-side tests, or expand a model "
                        "below to tune its request settings."
                    ),
                    elem_classes=["compact-note"],
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        model_a_inputs = build_target_panel(
                            title=settings.model_a.label,
                            defaults=settings.model_a,
                        )

                    with gr.Column(scale=1):
                        model_b_inputs = build_target_panel(
                            title=settings.model_b.label,
                            defaults=settings.model_b,
                        )

            with gr.Row(equal_height=False):
                with gr.Column(elem_classes=["panel", "compare-card"]):
                    gr.Markdown(
                        f"### {settings.model_a.label} Output",
                        elem_classes=["results-heading"],
                    )
                    model_a_output = gr.Markdown(
                        f"{settings.model_a.label} output will appear here.",
                        elem_id="model-a-output",
                        elem_classes=["readable-output", "output-pane"],
                    )
                    with gr.Accordion("Status + Tokens", open=False):
                        model_a_status = gr.Markdown(
                            f"{settings.model_a.label} status will appear here.",
                            elem_classes=["readable-output"],
                        )
                    with gr.Accordion("Effective Request", open=False):
                        model_a_effective = gr.Markdown(
                            f"{settings.model_a.label} effective request will appear here.",
                            elem_classes=["readable-output"],
                        )

                with gr.Column(elem_classes=["panel", "compare-card"]):
                    gr.Markdown(
                        f"### {settings.model_b.label} Output",
                        elem_classes=["results-heading"],
                    )
                    model_b_output = gr.Markdown(
                        f"{settings.model_b.label} output will appear here.",
                        elem_id="model-b-output",
                        elem_classes=["readable-output", "output-pane"],
                    )
                    with gr.Accordion("Status + Tokens", open=False):
                        model_b_status = gr.Markdown(
                            f"{settings.model_b.label} status will appear here.",
                            elem_classes=["readable-output"],
                        )
                    with gr.Accordion("Effective Request", open=False):
                        model_b_effective = gr.Markdown(
                            f"{settings.model_b.label} effective request will appear here.",
                            elem_classes=["readable-output"],
                        )

            with gr.Accordion(
                "Raw Compare JSON",
                open=False,
                elem_classes=["panel", "settings-shell"],
            ):
                compare_json = gr.JSON(
                    label="Compare Result JSON",
                    elem_classes=["readable-output"],
                )

        with gr.Tab("History"):
            gr.Markdown(
                "Browse prior compare sessions and reload full JSON details.",
                elem_classes=["tab-note"],
            )
            refresh_history_btn = gr.Button("Refresh History")
            history_table = gr.Dataframe(
                headers=[
                    "compare_id",
                    "status",
                    "created_at",
                    "left_label",
                    "right_label",
                    "total_ms",
                ],
                datatype=["str", "str", "str", "str", "str", "number"],
                interactive=False,
                elem_classes=["readable-output"],
            )
            history_compare_id = gr.Dropdown(label="Select compare_id", choices=[])
            load_detail_btn = gr.Button("Load Compare Detail")
            history_json = gr.JSON(
                label="Compare Detail",
                elem_classes=["readable-output"],
            )

        shared_inputs = [
            prompt,
            text_input,
            text_only,
            image_upload,
            video_upload,
            preprocess_images,
            preprocess_video,
            target_height,
            target_video_fps,
            safe_video_sampling,
            video_sampling_fps,
            segment_max_duration_s,
            segment_overlap_s,
            segment_workers,
            image_cache_uuids,
            video_cache_uuids,
            disable_caching,
        ]
        compare_inputs = shared_inputs + model_a_inputs + model_b_inputs

        compare_button.click(
            fn=_run_compare,
            inputs=compare_inputs,
            outputs=[
                compare_summary,
                compare_json,
                model_a_status,
                model_a_output,
                model_a_effective,
                model_b_status,
                model_b_output,
                model_b_effective,
            ],
        )
        prompt_mode.change(
            fn=_apply_prompt_mode,
            inputs=[prompt_mode, prompt, tag_categories],
            outputs=[prompt],
        )
        tag_categories.change(
            fn=_refresh_prompt_for_tagging,
            inputs=[prompt_mode, prompt, tag_categories],
            outputs=[prompt],
        )
        segmentation_profile.change(
            fn=_apply_segmentation_profile,
            inputs=[segmentation_profile, segment_max_duration_s, segment_overlap_s],
            outputs=[segment_max_duration_s, segment_overlap_s],
        )
        image_upload.change(
            fn=_image_preview_value,
            inputs=[image_upload],
            outputs=[image_preview],
        )
        video_upload.change(
            fn=_video_preview_value,
            inputs=[video_upload],
            outputs=[video_preview],
        )

        refresh_history_btn.click(
            fn=_refresh_history,
            inputs=[],
            outputs=[history_table, history_compare_id],
        )
        load_detail_btn.click(
            fn=_load_history_detail,
            inputs=[history_compare_id],
            outputs=[history_json],
        )
        app.load(
            fn=_refresh_history,
            inputs=[],
            outputs=[history_table, history_compare_id],
        )

    return cast(gr.Blocks, app)
