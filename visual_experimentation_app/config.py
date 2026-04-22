"""Settings for the multimodal compare lab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

ThinkingMode = Literal["auto", "on", "off"]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_thinking_mode(name: str, default: ThinkingMode) -> ThinkingMode:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized == "on":
        return "on"
    if normalized == "off":
        return "off"
    return default


def _normalize_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        return "http://127.0.0.1:8000/v1"
    if not base.endswith("/v1"):
        return f"{base}/v1"
    return base


def _normalize_path(value: str, default: str) -> str:
    cleaned = value.strip() if value.strip() else default
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    if len(cleaned) > 1:
        cleaned = cleaned.rstrip("/")
    return cleaned


def _load_dotenv_defaults(dotenv_path: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    path = dotenv_path or root / ".env"
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
            if "=" not in line:
                continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        parsed = value.strip()
        if (parsed.startswith('"') and parsed.endswith('"')) or (
            parsed.startswith("'") and parsed.endswith("'")
        ):
            parsed = parsed[1:-1]
        os.environ[key] = parsed


@dataclass(frozen=True)
class TargetRequestDefaults:
    """Environment-resolved request defaults for one compare target."""

    use_model_defaults: bool
    max_tokens: int
    max_completion_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    frequency_penalty: float
    thinking_mode: ThinkingMode
    show_reasoning: bool
    measure_ttft: bool


@dataclass(frozen=True)
class TargetDefaults:
    """Environment-resolved defaults for one compare target."""

    label: str
    base_url: str
    model: str
    api_key: str
    request_defaults: TargetRequestDefaults


@dataclass(frozen=True)
class LabSettings:
    """Environment-resolved runtime defaults for the compare lab."""

    host: str
    port: int
    ui_path: str
    api_prefix: str
    results_dir: Path
    default_timeout_seconds: float
    default_target_height: int
    default_video_fps: float
    default_safe_video_sampling: bool
    model_a: TargetDefaults
    model_b: TargetDefaults


def _build_target_request_defaults(
    *,
    prefix: str,
    top_k_default: int,
) -> TargetRequestDefaults:
    """Return per-target request defaults from environment variables."""
    return TargetRequestDefaults(
        use_model_defaults=_env_bool(f"{prefix}_DEFAULT_USE_MODEL_DEFAULTS", False),
        max_tokens=max(1, _env_int(f"{prefix}_DEFAULT_MAX_TOKENS", 10000)),
        max_completion_tokens=max(
            1,
            _env_int(f"{prefix}_DEFAULT_MAX_COMPLETION_TOKENS", 9500),
        ),
        temperature=min(2.0, max(0.0, _env_float(f"{prefix}_DEFAULT_TEMPERATURE", 1.0))),
        top_p=min(1.0, max(0.05, _env_float(f"{prefix}_DEFAULT_TOP_P", 0.95))),
        top_k=max(1, _env_int(f"{prefix}_DEFAULT_TOP_K", top_k_default)),
        presence_penalty=min(
            2.0,
            max(-2.0, _env_float(f"{prefix}_DEFAULT_PRESENCE_PENALTY", 1.5)),
        ),
        frequency_penalty=min(
            2.0,
            max(-2.0, _env_float(f"{prefix}_DEFAULT_FREQUENCY_PENALTY", 0.0)),
        ),
        thinking_mode=_env_thinking_mode(f"{prefix}_DEFAULT_THINKING_MODE", "off"),
        show_reasoning=_env_bool(f"{prefix}_DEFAULT_SHOW_REASONING", False),
        measure_ttft=_env_bool(f"{prefix}_DEFAULT_MEASURE_TTFT", True),
    )


@lru_cache(maxsize=1)
def get_settings() -> LabSettings:
    """Load and return cached compare-lab settings."""
    _load_dotenv_defaults()
    root = Path(__file__).resolve().parents[1]

    results_dir_raw = os.getenv("MM_LAB_RESULTS_DIR", "").strip()
    if results_dir_raw:
        results_dir = Path(results_dir_raw).expanduser()
    else:
        results_dir = root / "visual_experimentation_app" / "results"

    return LabSettings(
        host=os.getenv("MM_LAB_HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=max(1, _env_int("MM_LAB_PORT", 7870)),
        ui_path=_normalize_path(os.getenv("MM_LAB_UI_PATH", "/"), "/"),
        api_prefix=_normalize_path(os.getenv("MM_LAB_API_PREFIX", "/api"), "/api"),
        results_dir=results_dir,
        default_timeout_seconds=max(
            1.0,
            _env_float("MM_LAB_DEFAULT_TIMEOUT_SECONDS", 180.0),
        ),
        default_target_height=max(64, _env_int("MM_LAB_DEFAULT_TARGET_HEIGHT", 480)),
        default_video_fps=max(0.1, _env_float("MM_LAB_DEFAULT_VIDEO_FPS", 1.0)),
        default_safe_video_sampling=_env_bool("MM_LAB_SAFE_VIDEO_SAMPLING", False),
        model_a=TargetDefaults(
            label=os.getenv("MM_LAB_MODEL_A_LABEL", "Qwen 3.5 27B FP8").strip()
            or "Qwen 3.5 27B FP8",
            base_url=_normalize_base_url(
                os.getenv("MM_LAB_MODEL_A_BASE_URL", "http://127.0.0.1:8000/v1")
            ),
            model=os.getenv("MM_LAB_MODEL_A_MODEL", "Qwen/Qwen3.5-27B-FP8").strip()
            or "Qwen/Qwen3.5-27B-FP8",
            api_key=os.getenv("MM_LAB_MODEL_A_API_KEY", "EMPTY").strip() or "EMPTY",
            request_defaults=_build_target_request_defaults(
                prefix="MM_LAB_MODEL_A",
                top_k_default=20,
            ),
        ),
        model_b=TargetDefaults(
            label=os.getenv("MM_LAB_MODEL_B_LABEL", "Gemma 4 31B FP8").strip()
            or "Gemma 4 31B FP8",
            base_url=_normalize_base_url(
                os.getenv("MM_LAB_MODEL_B_BASE_URL", "http://127.0.0.1:8001/v1")
            ),
            model=os.getenv(
                "MM_LAB_MODEL_B_MODEL",
                "RedHatAI/gemma-4-31B-it-FP8-block",
            ).strip()
            or "RedHatAI/gemma-4-31B-it-FP8-block",
            api_key=os.getenv("MM_LAB_MODEL_B_API_KEY", "EMPTY").strip() or "EMPTY",
            request_defaults=_build_target_request_defaults(
                prefix="MM_LAB_MODEL_B",
                top_k_default=64,
            ),
        ),
    )


def clear_settings_cache() -> None:
    """Clear memoized settings for tests that mutate the environment."""
    get_settings.cache_clear()
