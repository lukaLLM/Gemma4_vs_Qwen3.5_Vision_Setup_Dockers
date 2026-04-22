"""Helpers for OpenAI-compatible multimodal payload assembly."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


def normalize_base_url(value: str) -> str:
    """Normalize OpenAI-compatible base URL to include `/v1` suffix."""
    base = value.strip().rstrip("/")
    if not base:
        return "http://127.0.0.1:8000/v1"
    if not base.endswith("/v1"):
        return f"{base}/v1"
    return base


def coerce_text(content: object) -> str:
    """Convert rich response content payloads into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                for key in ("text", "content", "reasoning_content", "reasoning", "output_text"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        return "".join(parts)
    return str(content)


def extract_message_parts(message: object) -> tuple[str, str]:
    """Extract assistant content and optional reasoning text."""
    content = coerce_text(getattr(message, "content", None)).strip()
    reasoning = coerce_text(getattr(message, "reasoning_content", None)).strip()
    if not reasoning:
        reasoning = coerce_text(getattr(message, "reasoning", None)).strip()
    if not reasoning and content:
        content, reasoning = split_out_think_tags(content)
    return content, reasoning


_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def split_out_think_tags(content: str) -> tuple[str, str]:
    """Split inline <think>...</think> tags into final content and reasoning text."""
    text = content.strip()
    if not text:
        return "", ""

    matches = list(_THINK_TAG_PATTERN.finditer(text))
    if not matches:
        return text, ""

    reasoning_parts = [match.group(1).strip() for match in matches if match.group(1).strip()]
    clean_content = _THINK_TAG_PATTERN.sub("", text).strip()
    return clean_content, "\n\n".join(reasoning_parts).strip()


def inject_prompt_query(prompt: str, text_input: str | None) -> str:
    """Replace `{query}` placeholders or append free-form text input."""
    prompt_text = prompt.strip()
    supplemental_text = (text_input or "").strip()
    if "{query}" in prompt_text:
        return prompt_text.replace("{query}", supplemental_text)
    if supplemental_text:
        return f"{prompt_text}\n\n{supplemental_text}"
    return prompt_text


def build_messages(
    *,
    prompt: str,
    text_input: str | None,
    image_data_urls: list[str],
    video_data_urls: list[str],
    image_cache_uuids: list[str],
    video_cache_uuids: list[str],
) -> list[dict[str, object]]:
    """Build OpenAI chat `messages` payload for mixed text/image/video input."""
    content: list[dict[str, object]] = []

    prompt_text = inject_prompt_query(prompt, text_input)
    content.append({"type": "text", "text": prompt_text})

    for index, data_url in enumerate(image_data_urls):
        image_payload: dict[str, object] = {"url": data_url}
        if index < len(image_cache_uuids):
            uuid_value = image_cache_uuids[index].strip()
            if uuid_value:
                image_payload["uuid"] = uuid_value
        content.append({"type": "image_url", "image_url": image_payload})

    for index, data_url in enumerate(video_data_urls):
        video_payload: dict[str, object] = {"url": data_url}
        if index < len(video_cache_uuids):
            uuid_value = video_cache_uuids[index].strip()
            if uuid_value:
                video_payload["uuid"] = uuid_value
        content.append({"type": "video_url", "video_url": video_payload})

    return [{"role": "user", "content": content}]


def model_supports_thinking_token_budget(model_name: str | None) -> bool:
    """Return whether the resolved model family supports vLLM thinking budgets."""
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return False
    supported_markers = (
        "qwen3",
        "qwen3.5",
        "deepseek",
        "nemotron",
    )
    return any(marker in normalized for marker in supported_markers)


def merge_extra_body(
    *,
    user_extra_body: Mapping[str, Any],
    include_video: bool,
    safe_video_sampling: bool,
    video_sampling_fps: float | None,
    thinking_mode: str | None,
    top_k: int | None,
    model_name: str | None = None,
    thinking_token_budget: int | None = None,
    gemma_max_soft_tokens: int | None = None,
) -> dict[str, Any]:
    """Merge user-supplied `extra_body` with safe multimodal defaults."""
    merged: dict[str, Any] = dict(user_extra_body)

    if include_video:
        raw_mm = merged.get("mm_processor_kwargs")
        mm_kwargs: dict[str, object] = dict(raw_mm) if isinstance(raw_mm, dict) else {}

        if safe_video_sampling:
            # Safe mode must win over any user-provided sampling config.
            mm_kwargs["do_sample_frames"] = False
            mm_kwargs.pop("fps", None)
        elif video_sampling_fps is not None:
            mm_kwargs["do_sample_frames"] = True
            mm_kwargs["fps"] = video_sampling_fps

        if mm_kwargs:
            merged["mm_processor_kwargs"] = mm_kwargs

    if gemma_max_soft_tokens is not None and "gemma" in (model_name or "").lower():
        raw_mm = merged.get("mm_processor_kwargs")
        mm_kwargs = dict(raw_mm) if isinstance(raw_mm, dict) else {}
        mm_kwargs["max_soft_tokens"] = int(gemma_max_soft_tokens)
        merged["mm_processor_kwargs"] = mm_kwargs

    if (
        thinking_token_budget is not None
        and model_supports_thinking_token_budget(model_name)
    ):
        merged["thinking_token_budget"] = int(thinking_token_budget)

    if top_k is not None:
        merged.setdefault("top_k", int(top_k))

    mode = (thinking_mode or "").strip().lower()
    if mode in {"on", "off"}:
        raw_chat = merged.get("chat_template_kwargs")
        chat_kwargs: dict[str, object] = dict(raw_chat) if isinstance(raw_chat, dict) else {}
        chat_kwargs.setdefault("enable_thinking", mode == "on")
        merged["chat_template_kwargs"] = chat_kwargs

    return merged


def parse_json_object(raw_value: str, *, field_name: str) -> dict[str, Any]:
    """Parse a JSON object string and raise descriptive errors on failure."""
    trimmed = raw_value.strip()
    if not trimmed:
        return {}

    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return parsed
