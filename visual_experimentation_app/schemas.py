"""Typed request/response schemas for the multimodal compare lab."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ThinkingMode = Literal["auto", "on", "off"]
RunStatus = Literal["ok", "error"]
CompareStatus = Literal["ok", "partial", "error"]
GEMMA_MAX_SOFT_TOKENS_ALLOWED = {70, 140, 280, 560, 1120}


class CompareTargetConfig(BaseModel):
    """Per-model request settings for one side of a compare run."""

    model_config = {"extra": "forbid"}

    label: str = ""
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)

    use_model_defaults: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    thinking_mode: ThinkingMode | None = None
    show_reasoning: bool | None = None
    measure_ttft: bool | None = None
    thinking_token_budget: int | None = Field(default=None, ge=1)
    gemma_max_soft_tokens: int | None = Field(default=None)
    request_extra_body: dict[str, Any] = Field(default_factory=dict)
    request_extra_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_target_settings(self) -> CompareTargetConfig:
        """Trim simple string fields and collapse explicit params in defaults mode."""
        self.label = self.label.strip()
        self.base_url = (self.base_url or "").strip() or None
        self.model = (self.model or "").strip() or None
        self.api_key = (self.api_key or "").strip() or None
        normalized_thinking_mode = (self.thinking_mode or "").strip().lower()
        if not normalized_thinking_mode:
            self.thinking_mode = None
        elif normalized_thinking_mode == "auto":
            self.thinking_mode = "auto"
        elif normalized_thinking_mode == "on":
            self.thinking_mode = "on"
        elif normalized_thinking_mode == "off":
            self.thinking_mode = "off"
        else:
            raise ValueError("thinking_mode must be one of: auto, on, off.")
        self.request_extra_headers = {
            str(key): str(value) for key, value in self.request_extra_headers.items()
        }
        if (
            self.gemma_max_soft_tokens is not None
            and self.gemma_max_soft_tokens not in GEMMA_MAX_SOFT_TOKENS_ALLOWED
        ):
            allowed = ", ".join(str(value) for value in sorted(GEMMA_MAX_SOFT_TOKENS_ALLOWED))
            raise ValueError(f"gemma_max_soft_tokens must be one of: {allowed}.")

        if self.use_model_defaults:
            self.max_tokens = None
            self.max_completion_tokens = None
            self.temperature = None
            self.top_p = None
            self.top_k = None
            self.presence_penalty = None
            self.frequency_penalty = None
        return self


class CompareRequest(BaseModel):
    """Input payload for a shared-media, dual-model compare run."""

    model_config = {"extra": "forbid"}

    prompt: str = Field(..., min_length=1)
    text_input: str | None = None
    text_only: bool = False
    image_paths: list[str] = Field(default_factory=list)
    video_paths: list[str] = Field(default_factory=list)
    video_path: str | None = None

    preprocess_images: bool = True
    preprocess_video: bool = True
    target_height: int = Field(default=480, ge=64, le=4096)
    target_video_fps: float | None = Field(default=1.0, gt=0.0, le=240.0)
    safe_video_sampling: bool = True
    video_sampling_fps: float | None = Field(default=None, gt=0.0, le=240.0)

    segment_max_duration_s: float = Field(default=0.0, ge=0.0, le=3600.0)
    segment_overlap_s: float = Field(default=0.0, ge=0.0, le=300.0)
    segment_workers: int = Field(default=1, ge=1, le=64)

    image_cache_uuids: list[str] = Field(default_factory=list)
    video_cache_uuids: list[str] = Field(default_factory=list)
    video_cache_uuid: str | None = None
    disable_caching: bool = False

    model_a: CompareTargetConfig = Field(default_factory=CompareTargetConfig)
    model_b: CompareTargetConfig = Field(default_factory=CompareTargetConfig)

    @model_validator(mode="after")
    def validate_prompt_and_segments(self) -> CompareRequest:
        """Normalize prompt/media fields and enforce segment constraints."""
        prompt = self.prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty after trimming.")
        self.prompt = prompt
        self.text_input = (self.text_input or "").strip() or None

        normalized_video_paths = [
            str(path).strip() for path in self.video_paths if str(path).strip()
        ]
        legacy_video_path = (self.video_path or "").strip()
        if legacy_video_path and legacy_video_path not in normalized_video_paths:
            normalized_video_paths.insert(0, legacy_video_path)
        if len(normalized_video_paths) > 2:
            raise ValueError("At most 2 videos are supported per compare request.")
        self.video_paths = normalized_video_paths
        self.video_path = self.video_paths[0] if self.video_paths else None

        self.image_paths = [str(path).strip() for path in self.image_paths if str(path).strip()]

        normalized_video_uuids = [
            str(uuid_value).strip()
            for uuid_value in self.video_cache_uuids
            if str(uuid_value).strip()
        ]
        legacy_video_uuid = (self.video_cache_uuid or "").strip()
        if legacy_video_uuid and legacy_video_uuid not in normalized_video_uuids:
            normalized_video_uuids.insert(0, legacy_video_uuid)
        self.video_cache_uuids = normalized_video_uuids
        self.video_cache_uuid = (
            self.video_cache_uuids[0] if self.video_cache_uuids else None
        )
        self.image_cache_uuids = [
            str(uuid_value).strip()
            for uuid_value in self.image_cache_uuids
            if str(uuid_value).strip()
        ]

        if self.text_only:
            self.image_paths = []
            self.video_paths = []
            self.video_path = None
            self.image_cache_uuids = []
            self.video_cache_uuids = []
            self.video_cache_uuid = None

        if self.segment_max_duration_s <= 0:
            self.segment_overlap_s = 0.0
        if self.segment_overlap_s >= self.segment_max_duration_s > 0:
            raise ValueError("segment_overlap_s must be less than segment_max_duration_s.")
        if self.segment_max_duration_s > 0 and len(self.video_paths) > 1:
            raise ValueError("Segmentation supports only one video at a time.")
        return self


class RunTiming(BaseModel):
    """Timing breakdown for one target result."""

    preprocess_ms: float = 0.0
    request_ms: float = 0.0
    total_ms: float = 0.0
    ttft_ms: float | None = None


class TokenUsageStats(BaseModel):
    """Token accounting for one target result."""

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tokens_per_second: float | None = None


class CompareTargetResult(BaseModel):
    """Per-model result details for one compare run."""

    label: str
    status: RunStatus
    output_text: str = ""
    error: str | None = None
    timings: RunTiming
    token_usage: TokenUsageStats = Field(default_factory=TokenUsageStats)
    effective_params: dict[str, Any] = Field(default_factory=dict)


class CompareTiming(BaseModel):
    """Shared timing metadata for a compare run."""

    preprocess_ms: float = 0.0
    total_ms: float = 0.0


class CompareResult(BaseModel):
    """Persisted envelope for a full compare run."""

    compare_id: str
    status: CompareStatus
    created_at: str
    request: CompareRequest
    timings: CompareTiming
    media_metadata: dict[str, Any] = Field(default_factory=dict)
    model_a_result: CompareTargetResult
    model_b_result: CompareTargetResult


class CompareHistoryItem(BaseModel):
    """Compact compare summary used by history listing endpoints."""

    compare_id: str
    created_at: str
    status: CompareStatus
    model_a_label: str
    model_a_model: str
    model_b_label: str
    model_b_model: str
    total_ms: float
