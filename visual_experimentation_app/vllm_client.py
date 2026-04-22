"""Execution engine for shared-media compare runs against OpenAI-compatible vLLM servers."""

from __future__ import annotations

import html
import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai import OpenAI

from visual_experimentation_app.config import (
    LabSettings,
    TargetDefaults,
    TargetRequestDefaults,
    get_settings,
)
from visual_experimentation_app.media_preprocess import (
    PreparedMedia,
    SegmentClip,
    cleanup_paths,
    encode_file_to_data_url,
    extract_video_segments,
    prepare_media,
)
from visual_experimentation_app.payload_builder import (
    build_messages,
    coerce_text,
    extract_message_parts,
    merge_extra_body,
    model_supports_thinking_token_budget,
    normalize_base_url,
)
from visual_experimentation_app.schemas import (
    CompareRequest,
    CompareStatus,
    CompareTargetConfig,
    CompareTargetResult,
    RunTiming,
    TokenUsageStats,
)


@dataclass(frozen=True)
class CompareExecution:
    """Execution payload returned by the compare engine."""

    status: CompareStatus
    preprocess_ms: float
    total_ms: float
    media_metadata: dict[str, Any]
    model_a_result: CompareTargetResult
    model_b_result: CompareTargetResult


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting captured from API usage metadata."""

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ResolvedTargetConfig:
    """Resolved target settings after applying environment defaults."""

    label: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float


@dataclass(frozen=True)
class PreparedComparePayload:
    """Shared preprocessed media and prebuilt messages for both targets."""

    prepared_media: PreparedMedia
    shared_messages: list[dict[str, object]] | None
    segment_messages: list[list[dict[str, object]]]
    segments: list[SegmentClip]
    cleanup_paths: list[Path]


def _effective_setting(value: str | None, fallback: str) -> str:
    selected = (value or "").strip()
    return selected if selected else fallback


def _effective_timeout(value: float | None, fallback: float) -> float:
    if value is None:
        return fallback
    return max(1.0, float(value))


def _apply_target_request_defaults(
    *,
    target: CompareTargetConfig,
    defaults: TargetRequestDefaults,
) -> CompareTargetConfig:
    """Fill omitted target request fields from configured per-model defaults."""
    payload: dict[str, Any] = target.model_dump()
    explicit_fields = set(target.model_fields_set)
    generation_field_names = {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
    }
    has_explicit_generation_override = bool(explicit_fields & generation_field_names)

    use_model_defaults = (
        payload["use_model_defaults"]
        if "use_model_defaults" in explicit_fields
        else (False if has_explicit_generation_override else defaults.use_model_defaults)
    )
    payload["use_model_defaults"] = use_model_defaults

    generation_defaults = {
        "max_tokens": defaults.max_tokens,
        "max_completion_tokens": defaults.max_completion_tokens,
        "temperature": defaults.temperature,
        "top_p": defaults.top_p,
        "top_k": defaults.top_k,
        "presence_penalty": defaults.presence_penalty,
        "frequency_penalty": defaults.frequency_penalty,
    }
    toggles = {
        "thinking_mode": defaults.thinking_mode,
        "show_reasoning": defaults.show_reasoning,
        "measure_ttft": defaults.measure_ttft,
    }

    if use_model_defaults:
        for field_name in generation_defaults:
            payload[field_name] = None
    else:
        for field_name, default_value in generation_defaults.items():
            if (field_name not in explicit_fields) or (payload[field_name] is None):
                payload[field_name] = default_value

    for field_name, toggle_value in toggles.items():
        if (field_name not in explicit_fields) or (payload[field_name] is None):
            payload[field_name] = toggle_value

    return CompareTargetConfig.model_validate(payload)


_VIDEO_PROCESSOR_HINT = (
    "vLLM video processor rejected this request (Qwen3VLProcessor). "
    "Try lowering video sampling FPS, enabling Safe video sampling, "
    "reducing segment workers, or using shorter segment duration."
)


def _exception_text(exc: Exception) -> str:
    parts: list[str] = [str(exc)]

    body = getattr(exc, "body", None)
    if body:
        try:
            parts.append(json.dumps(body, ensure_ascii=True, default=str))
        except (TypeError, ValueError):
            parts.append(str(body))

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            response_text = getattr(response, "text", "")
            if response_text:
                parts.append(str(response_text))
        except Exception:  # noqa: BLE001
            pass

    return "\n".join(part for part in parts if part).lower()


def is_video_processor_error(exc: Exception) -> bool:
    """Return whether exception text matches known video processor failures."""
    text = _exception_text(exc)
    return (
        "failed to apply qwen3vlprocessor" in text
        or ("error in preprocessing prompt inputs" in text and "video" in text)
        or (
            "video_processing_utils" in text
            and "out of bounds" in text
            and "index" in text
        )
    )


def summarize_execution_error(exc: Exception) -> str:
    """Return concise, actionable run error text for UI/API responses."""
    if is_video_processor_error(exc):
        return _VIDEO_PROCESSOR_HINT

    raw_error = str(exc).strip() or exc.__class__.__name__
    if len(raw_error) <= 700:
        return raw_error
    return f"{raw_error[:700]}..."


def build_execution_error_details(exc: Exception) -> dict[str, Any]:
    """Build rich error details while keeping UI-facing message concise."""
    details: dict[str, Any] = {
        "error_type": exc.__class__.__name__,
        "is_video_processor_error": is_video_processor_error(exc),
        "raw_error": str(exc).strip() or repr(exc),
    }
    if details["is_video_processor_error"]:
        details["hint"] = _VIDEO_PROCESSOR_HINT
    return details


def _format_assistant_output(
    *,
    content: str,
    reasoning: str,
    include_reasoning: bool,
) -> str:
    clean_content = content.strip()
    clean_reasoning = reasoning.strip()
    if include_reasoning and clean_reasoning:
        reasoning_block = (
            '<details class="reasoning-shell" open>'
            '<summary>Reasoning</summary>'
            f"<pre>{html.escape(clean_reasoning)}</pre>"
            "</details>"
        )
        if clean_content:
            return f"{reasoning_block}\n\n### Final Answer\n\n{clean_content}"
        return reasoning_block
    if clean_content:
        return clean_content
    return clean_reasoning


def _segment_header(segment: SegmentClip, index: int, total: int) -> str:
    return f"### Segment {index}/{total} [{segment.start_s:.2f}s - {segment.end_s:.2f}s]"


def _model_cache_dir(model: str) -> Path:
    """Map Hugging Face model id to local cache directory name."""
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model.replace('/', '--')}"


def _load_generation_config_defaults(model: str) -> dict[str, Any]:
    """Best-effort read of local Hugging Face generation_config for this model."""
    model_dir = _model_cache_dir(model)
    if not model_dir.exists():
        return {
            "found": False,
            "source": "not_found",
            "message": "Model cache directory was not found locally.",
        }

    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        candidates = sorted(snapshots_dir.glob("*/generation_config.json"), reverse=True)
        if candidates:
            path = candidates[0]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                return {
                    "found": False,
                    "source": "parse_error",
                    "path": str(path),
                    "message": f"Found generation_config.json but failed to parse: {exc}",
                }

            sampling_keys = [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "max_new_tokens",
            ]
            sampling_values = {
                key: payload[key]
                for key in sampling_keys
                if key in payload and payload[key] is not None
            }
            return {
                "found": True,
                "source": "generation_config_json",
                "path": str(path),
                "sampling_values": sampling_values,
            }

    no_exist_dir = model_dir / ".no_exist"
    no_exist_markers = list(no_exist_dir.glob("*/generation_config.json"))
    if no_exist_markers:
        return {
            "found": False,
            "source": "missing_in_model_repo",
            "message": "Model repository appears to have no generation_config.json.",
        }

    return {
        "found": False,
        "source": "unknown",
        "message": (
            "Could not locate generation_config.json in local cache. "
            "vLLM will use model defaults if available, otherwise its internal defaults."
        ),
    }


def _chat_completion_kwargs(
    *,
    target: CompareTargetConfig,
    model: str,
    messages: list[dict[str, object]],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "extra_body": extra_body,
    }
    if target.max_tokens is not None:
        kwargs["max_tokens"] = target.max_tokens
    if target.max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = target.max_completion_tokens
    if target.temperature is not None:
        kwargs["temperature"] = target.temperature
    if target.top_p is not None:
        kwargs["top_p"] = target.top_p
    if target.presence_penalty is not None:
        kwargs["presence_penalty"] = target.presence_penalty
    if target.frequency_penalty is not None:
        kwargs["frequency_penalty"] = target.frequency_penalty
    return kwargs


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _extract_usage_tokens(payload: object) -> TokenUsage:
    usage = getattr(payload, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=_int_or_none(getattr(usage, "prompt_tokens", None)),
        output_tokens=_int_or_none(getattr(usage, "completion_tokens", None)),
        total_tokens=_int_or_none(getattr(usage, "total_tokens", None)),
    )


def _sum_token_usage(usages: Iterable[TokenUsage]) -> TokenUsage:
    usage_list = list(usages)

    prompt_values = [
        value for value in (item.prompt_tokens for item in usage_list) if value is not None
    ]
    output_values = [
        value for value in (item.output_tokens for item in usage_list) if value is not None
    ]
    total_values = [
        value for value in (item.total_tokens for item in usage_list) if value is not None
    ]

    return TokenUsage(
        prompt_tokens=sum(prompt_values) if prompt_values else None,
        output_tokens=sum(output_values) if output_values else None,
        total_tokens=sum(total_values) if total_values else None,
    )


def _tokens_per_second(*, output_tokens: int | None, request_ms: float) -> float | None:
    """Return output-token throughput for a request in tokens/second."""
    if output_tokens is None or output_tokens <= 0:
        return None
    if request_ms <= 0:
        return None
    return round((output_tokens * 1000.0) / request_ms, 3)


def _invoke_completion(
    *,
    client: OpenAI,
    target: CompareTargetConfig,
    model: str,
    messages: list[dict[str, object]],
    extra_body: dict[str, Any],
) -> tuple[str, float | None, TokenUsage]:
    kwargs = _chat_completion_kwargs(
        target=target,
        model=model,
        messages=messages,
        extra_body=extra_body,
    )

    if not target.measure_ttft:
        response = client.chat.completions.create(**kwargs)
        if not response.choices:
            return "", None, _extract_usage_tokens(response)
        content, reasoning = extract_message_parts(response.choices[0].message)
        return (
            _format_assistant_output(
                content=content,
                reasoning=reasoning,
                include_reasoning=bool(target.show_reasoning),
            ),
            None,
            _extract_usage_tokens(response),
        )

    start = time.perf_counter()
    first_chunk_time: float | None = None
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    final_usage = TokenUsage()

    stream = client.chat.completions.create(
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )
    for event in stream:
        event_usage = _extract_usage_tokens(event)
        if (
            event_usage.prompt_tokens is not None
            or event_usage.output_tokens is not None
            or event_usage.total_tokens is not None
        ):
            final_usage = event_usage
        if not event.choices:
            continue
        delta = event.choices[0].delta
        content_delta = coerce_text(getattr(delta, "content", None))
        reasoning_delta = coerce_text(getattr(delta, "reasoning_content", None))
        if not reasoning_delta:
            reasoning_delta = coerce_text(getattr(delta, "reasoning", None))

        changed = False
        if content_delta:
            content_chunks.append(content_delta)
            changed = True
        if reasoning_delta:
            reasoning_chunks.append(reasoning_delta)
            changed = True
        if changed and first_chunk_time is None:
            first_chunk_time = time.perf_counter()

    output = _format_assistant_output(
        content="".join(content_chunks),
        reasoning="".join(reasoning_chunks),
        include_reasoning=bool(target.show_reasoning),
    )
    ttft_ms = ((first_chunk_time - start) * 1000.0) if first_chunk_time is not None else None
    return output, ttft_ms, final_usage


def _build_client(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    extra_headers: dict[str, str],
) -> OpenAI:
    kwargs: dict[str, Any] = {
        "base_url": normalize_base_url(base_url),
        "api_key": api_key,
        "timeout": timeout_seconds,
    }
    if extra_headers:
        kwargs["default_headers"] = extra_headers
    return OpenAI(**kwargs)


def _build_segments(prepared: PreparedMedia, request: CompareRequest) -> list[SegmentClip]:
    if len(prepared.video_paths) != 1:
        return []
    video_path = prepared.video_paths[0]
    if request.segment_max_duration_s <= 0:
        return [SegmentClip(path=video_path, start_s=0.0, end_s=0.0, is_temp=False)]
    return extract_video_segments(
        video_path=video_path,
        max_duration_s=request.segment_max_duration_s,
        overlap_s=request.segment_overlap_s,
    )


def _cache_uuids(
    *,
    values: list[str],
    item_count: int,
    prefix: str,
    disable_caching: bool,
) -> list[str]:
    if not disable_caching:
        return values
    return [f"nocache-{prefix}-{uuid4().hex}" for _ in range(item_count)]


def _prepare_compare_payload(request: CompareRequest) -> PreparedComparePayload:
    prepared = prepare_media(
        image_paths=request.image_paths,
        video_paths=request.video_paths,
        preprocess_images=request.preprocess_images,
        preprocess_video=request.preprocess_video,
        target_height=request.target_height,
        target_video_fps=request.target_video_fps,
    )

    segments = _build_segments(prepared, request)
    cleanup_paths_for_run = prepared.cleanup_paths + [
        segment.path for segment in segments if segment.is_temp
    ]

    image_cache_uuids = _cache_uuids(
        values=request.image_cache_uuids,
        item_count=len(prepared.image_paths),
        prefix="img",
        disable_caching=request.disable_caching,
    )
    video_cache_uuids = _cache_uuids(
        values=request.video_cache_uuids,
        item_count=len(prepared.video_paths),
        prefix="vid",
        disable_caching=request.disable_caching,
    )
    image_data_urls = [encode_file_to_data_url(path) for path in prepared.image_paths]

    if not segments:
        shared_messages = build_messages(
            prompt=request.prompt,
            text_input=request.text_input,
            image_data_urls=image_data_urls,
            video_data_urls=[encode_file_to_data_url(path) for path in prepared.video_paths],
            image_cache_uuids=image_cache_uuids,
            video_cache_uuids=video_cache_uuids,
        )
        return PreparedComparePayload(
            prepared_media=prepared,
            shared_messages=shared_messages,
            segment_messages=[],
            segments=[],
            cleanup_paths=cleanup_paths_for_run,
        )

    if len(segments) == 1:
        segment_messages = [
            build_messages(
                prompt=request.prompt,
                text_input=request.text_input,
                image_data_urls=image_data_urls,
                video_data_urls=[encode_file_to_data_url(segments[0].path)],
                image_cache_uuids=image_cache_uuids,
                video_cache_uuids=video_cache_uuids[:1],
            )
        ]
    else:
        segment_messages = []
        for index, segment in enumerate(segments, start=1):
            segment_messages.append(
                build_messages(
                    prompt=request.prompt,
                    text_input=request.text_input,
                    image_data_urls=image_data_urls,
                    video_data_urls=[encode_file_to_data_url(segment.path)],
                    image_cache_uuids=image_cache_uuids,
                    video_cache_uuids=(
                        [f"nocache-segment-{index}-{uuid4().hex}"]
                        if request.disable_caching
                        else []
                    ),
                )
            )

    return PreparedComparePayload(
        prepared_media=prepared,
        shared_messages=None,
        segment_messages=segment_messages,
        segments=segments,
        cleanup_paths=cleanup_paths_for_run,
    )


def _resolve_target_config(
    target: CompareTargetConfig,
    defaults: TargetDefaults,
    *,
    timeout_seconds: float,
) -> ResolvedTargetConfig:
    return ResolvedTargetConfig(
        label=_effective_setting(target.label, defaults.label),
        base_url=_effective_setting(target.base_url, defaults.base_url),
        model=_effective_setting(target.model, defaults.model),
        api_key=_effective_setting(target.api_key, defaults.api_key),
        timeout_seconds=_effective_timeout(target.timeout_seconds, timeout_seconds),
    )


def _run_segmented_messages(
    *,
    target: CompareTargetConfig,
    resolved: ResolvedTargetConfig,
    extra_body: dict[str, Any],
    messages_by_segment: list[list[dict[str, object]]],
    segments: list[SegmentClip],
    max_workers: int,
) -> tuple[str, float | None, TokenUsage]:
    results: list[tuple[str, float | None, TokenUsage]] = [
        ("", None, TokenUsage()) for _ in messages_by_segment
    ]

    def run_one(index: int) -> tuple[int, str, float | None, TokenUsage]:
        threaded_client = _build_client(
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            timeout_seconds=resolved.timeout_seconds,
            extra_headers=target.request_extra_headers,
        )
        text, ttft, usage = _invoke_completion(
            client=threaded_client,
            target=target,
            model=resolved.model,
            messages=messages_by_segment[index],
            extra_body=dict(extra_body),
        )
        return index, text, ttft, usage

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, index) for index in range(len(messages_by_segment))]
        for future in futures:
            index, text, ttft, usage = future.result()
            results[index] = (text, ttft, usage)

    sections: list[str] = []
    ttft_values: list[float] = []
    usage_values: list[TokenUsage] = []
    for index, segment in enumerate(segments, start=1):
        text, ttft, usage = results[index - 1]
        section_body = text.strip() or "_No output._"
        sections.append(f"{_segment_header(segment, index, len(segments))}\n{section_body}")
        if ttft is not None:
            ttft_values.append(ttft)
        usage_values.append(usage)
    combined = "\n\n".join(sections)
    aggregate_ttft = min(ttft_values) if ttft_values else None
    aggregate_usage = _sum_token_usage(usage_values)
    return combined, aggregate_ttft, aggregate_usage


def _success_effective_params(
    *,
    request: CompareRequest,
    target: CompareTargetConfig,
    resolved: ResolvedTargetConfig,
    extra_body: dict[str, Any],
    prepared_payload: PreparedComparePayload,
    cache_salt: str | None,
) -> dict[str, Any]:
    segments = prepared_payload.segments
    mm_kwargs = extra_body.get("mm_processor_kwargs")
    if not isinstance(mm_kwargs, dict):
        mm_kwargs = {}

    effective_params: dict[str, Any] = {
        "label": resolved.label,
        "base_url": normalize_base_url(resolved.base_url),
        "model": resolved.model,
        "use_model_defaults": target.use_model_defaults,
        "text_only": request.text_only,
        "timeout_seconds": resolved.timeout_seconds,
        "target_height": request.target_height,
        "target_video_fps": request.target_video_fps,
        "safe_video_sampling": request.safe_video_sampling,
        "video_sampling_fps": request.video_sampling_fps,
        "video_count": len(prepared_payload.prepared_media.video_paths),
        "segment_max_duration_s": request.segment_max_duration_s,
        "segment_overlap_s": request.segment_overlap_s,
        "segment_workers": request.segment_workers,
        "segment_count": len(segments) if segments else 0,
        "disable_caching": request.disable_caching,
        "cache_salt": cache_salt,
        "extra_body": extra_body,
        "sent_generation_params": {
            "max_tokens": target.max_tokens,
            "max_completion_tokens": target.max_completion_tokens,
            "temperature": target.temperature,
            "top_p": target.top_p,
            "top_k": target.top_k,
            "presence_penalty": target.presence_penalty,
            "frequency_penalty": target.frequency_penalty,
            "thinking_mode": target.thinking_mode,
            "thinking_token_budget": target.thinking_token_budget,
            "gemma_max_soft_tokens": target.gemma_max_soft_tokens,
        },
        "thinking_token_budget_supported": model_supports_thinking_token_budget(
            resolved.model
        ),
        "thinking_token_budget_applied": extra_body.get("thinking_token_budget"),
        "gemma_max_soft_tokens_applied": mm_kwargs.get("max_soft_tokens"),
        "omitted_for_model_defaults": (
            [
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "frequency_penalty",
            ]
            if target.use_model_defaults
            else []
        ),
        "shared_preprocessing_reused": True,
    }
    if target.use_model_defaults:
        effective_params["model_defaults_info"] = _load_generation_config_defaults(
            resolved.model
        )
    return effective_params


def _error_target_result(
    *,
    label: str,
    base_url: str,
    model: str,
    request_ms: float,
    error_message: str,
    error_details: dict[str, Any],
) -> CompareTargetResult:
    return CompareTargetResult(
        label=label,
        status="error",
        output_text="",
        error=error_message,
        timings=RunTiming(
            preprocess_ms=0.0,
            request_ms=request_ms,
            total_ms=request_ms,
            ttft_ms=None,
        ),
        token_usage=TokenUsageStats(),
        effective_params={
            "label": label,
            "base_url": normalize_base_url(base_url),
            "model": model,
            "error_details": error_details,
        },
    )


def _execute_target(
    *,
    request: CompareRequest,
    target: CompareTargetConfig,
    defaults: TargetDefaults,
    settings: LabSettings,
    prepared_payload: PreparedComparePayload,
) -> CompareTargetResult:
    effective_target = _apply_target_request_defaults(
        target=target,
        defaults=defaults.request_defaults,
    )
    resolved = _resolve_target_config(
        effective_target,
        defaults,
        timeout_seconds=settings.default_timeout_seconds,
    )
    request_started = time.perf_counter()
    cache_salt: str | None = None

    try:
        extra_body = merge_extra_body(
            user_extra_body=effective_target.request_extra_body,
            include_video=bool(prepared_payload.prepared_media.video_paths),
            safe_video_sampling=request.safe_video_sampling,
            video_sampling_fps=request.video_sampling_fps,
            thinking_mode=effective_target.thinking_mode,
            top_k=effective_target.top_k,
            model_name=resolved.model,
            thinking_token_budget=effective_target.thinking_token_budget,
            gemma_max_soft_tokens=effective_target.gemma_max_soft_tokens,
        )
        if request.disable_caching:
            cache_salt = f"nocache-{uuid4().hex}"
            extra_body["cache_salt"] = cache_salt

        client = _build_client(
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            timeout_seconds=resolved.timeout_seconds,
            extra_headers=effective_target.request_extra_headers,
        )

        if prepared_payload.shared_messages is not None:
            output_text, ttft_ms, token_usage = _invoke_completion(
                client=client,
                target=effective_target,
                model=resolved.model,
                messages=prepared_payload.shared_messages,
                extra_body=extra_body,
            )
        elif len(prepared_payload.segment_messages) == 1:
            output_text, ttft_ms, token_usage = _invoke_completion(
                client=client,
                target=effective_target,
                model=resolved.model,
                messages=prepared_payload.segment_messages[0],
                extra_body=extra_body,
            )
        else:
            output_text, ttft_ms, token_usage = _run_segmented_messages(
                target=effective_target,
                resolved=resolved,
                extra_body=extra_body,
                messages_by_segment=prepared_payload.segment_messages,
                segments=prepared_payload.segments,
                max_workers=max(
                    1,
                    min(request.segment_workers, len(prepared_payload.segment_messages)),
                ),
            )
    except Exception as exc:  # noqa: BLE001
        request_ms = (time.perf_counter() - request_started) * 1000.0
        return _error_target_result(
            label=resolved.label,
            base_url=resolved.base_url,
            model=resolved.model,
            request_ms=request_ms,
            error_message=summarize_execution_error(exc),
            error_details=build_execution_error_details(exc),
        )

    request_ms = (time.perf_counter() - request_started) * 1000.0
    return CompareTargetResult(
        label=resolved.label,
        status="ok",
        output_text=output_text,
        error=None,
        timings=RunTiming(
            preprocess_ms=0.0,
            request_ms=request_ms,
            total_ms=request_ms,
            ttft_ms=ttft_ms,
        ),
        token_usage=TokenUsageStats(
            prompt_tokens=token_usage.prompt_tokens,
            output_tokens=token_usage.output_tokens,
            total_tokens=token_usage.total_tokens,
            tokens_per_second=_tokens_per_second(
                output_tokens=token_usage.output_tokens,
                request_ms=request_ms,
            ),
        ),
        effective_params=_success_effective_params(
            request=request,
            target=effective_target,
            resolved=resolved,
            extra_body=extra_body,
            prepared_payload=prepared_payload,
            cache_salt=cache_salt,
        ),
    )


def _compare_status(
    model_a_result: CompareTargetResult,
    model_b_result: CompareTargetResult,
) -> CompareStatus:
    statuses = {model_a_result.status, model_b_result.status}
    if statuses == {"ok"}:
        return "ok"
    if "ok" in statuses:
        return "partial"
    return "error"


def execute_compare(request: CompareRequest) -> CompareExecution:
    """Execute one compare request and return shared media + per-target results."""
    settings: LabSettings = get_settings()
    started_at = time.perf_counter()
    prepared_payload: PreparedComparePayload | None = None
    preprocess_started = time.perf_counter()

    try:
        prepared_payload = _prepare_compare_payload(request)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_model_a = executor.submit(
                _execute_target,
                request=request,
                target=request.model_a,
                defaults=settings.model_a,
                settings=settings,
                prepared_payload=prepared_payload,
            )
            future_model_b = executor.submit(
                _execute_target,
                request=request,
                target=request.model_b,
                defaults=settings.model_b,
                settings=settings,
                prepared_payload=prepared_payload,
            )
            model_a_result = future_model_a.result()
            model_b_result = future_model_b.result()
    except Exception as exc:  # noqa: BLE001
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        total_ms = (time.perf_counter() - started_at) * 1000.0
        resolved_a = _resolve_target_config(
            request.model_a,
            settings.model_a,
            timeout_seconds=settings.default_timeout_seconds,
        )
        resolved_b = _resolve_target_config(
            request.model_b,
            settings.model_b,
            timeout_seconds=settings.default_timeout_seconds,
        )
        error_message = f"Shared preprocessing failed: {summarize_execution_error(exc)}"
        error_details = build_execution_error_details(exc)
        model_a_result = _error_target_result(
            label=resolved_a.label,
            base_url=resolved_a.base_url,
            model=resolved_a.model,
            request_ms=0.0,
            error_message=error_message,
            error_details=error_details,
        )
        model_b_result = _error_target_result(
            label=resolved_b.label,
            base_url=resolved_b.base_url,
            model=resolved_b.model,
            request_ms=0.0,
            error_message=error_message,
            error_details=error_details,
        )
        return CompareExecution(
            status="error",
            preprocess_ms=preprocess_ms,
            total_ms=total_ms,
            media_metadata={},
            model_a_result=model_a_result,
            model_b_result=model_b_result,
        )
    finally:
        if prepared_payload is not None:
            cleanup_paths(prepared_payload.cleanup_paths)

    total_ms = (time.perf_counter() - started_at) * 1000.0
    return CompareExecution(
        status=_compare_status(model_a_result, model_b_result),
        preprocess_ms=preprocess_ms,
        total_ms=total_ms,
        media_metadata=prepared_payload.prepared_media.metadata,
        model_a_result=model_a_result,
        model_b_result=model_b_result,
    )
