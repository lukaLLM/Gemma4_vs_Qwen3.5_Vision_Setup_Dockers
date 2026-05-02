"""Tests for compare-lab environment settings."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.config import clear_settings_cache, get_settings  # noqa: E402


class CompareLabConfigTest(unittest.TestCase):
    """Covers dual-model env loading and normalization."""

    def tearDown(self) -> None:
        """Clear cached settings between tests."""
        clear_settings_cache()

    def test_dual_model_defaults_are_loaded_from_env(self) -> None:
        """Model A and B env vars should populate compare defaults."""
        with mock.patch.dict(
            os.environ,
            {
                "MM_LAB_MODEL_A_LABEL": "Qwen A",
                "MM_LAB_MODEL_A_BASE_URL": "http://127.0.0.1:9000",
                "MM_LAB_MODEL_A_MODEL": "org/model-a",
                "MM_LAB_MODEL_A_API_KEY": "KEY_A",
                "MM_LAB_MODEL_A_DEFAULT_MAX_TOKENS": "10000",
                "MM_LAB_MODEL_A_DEFAULT_MAX_COMPLETION_TOKENS": "9500",
                "MM_LAB_MODEL_A_DEFAULT_TEMPERATURE": "1.0",
                "MM_LAB_MODEL_A_DEFAULT_TOP_P": "0.95",
                "MM_LAB_MODEL_A_DEFAULT_TOP_K": "20",
                "MM_LAB_MODEL_A_DEFAULT_PRESENCE_PENALTY": "1.5",
                "MM_LAB_MODEL_A_DEFAULT_FREQUENCY_PENALTY": "0.0",
                "MM_LAB_MODEL_A_DEFAULT_THINKING_MODE": "off",
                "MM_LAB_MODEL_A_DEFAULT_SHOW_REASONING": "0",
                "MM_LAB_MODEL_A_DEFAULT_MEASURE_TTFT": "1",
                "MM_LAB_MODEL_B_LABEL": "Gemma B",
                "MM_LAB_MODEL_B_BASE_URL": "http://127.0.0.1:9001",
                "MM_LAB_MODEL_B_MODEL": "org/model-b",
                "MM_LAB_MODEL_B_API_KEY": "KEY_B",
                "MM_LAB_MODEL_B_DEFAULT_USE_MODEL_DEFAULTS": "1",
                "MM_LAB_MODEL_B_DEFAULT_MAX_TOKENS": "12000",
                "MM_LAB_MODEL_B_DEFAULT_MAX_COMPLETION_TOKENS": "8700",
                "MM_LAB_MODEL_B_DEFAULT_TEMPERATURE": "0.8",
                "MM_LAB_MODEL_B_DEFAULT_TOP_P": "0.9",
                "MM_LAB_MODEL_B_DEFAULT_TOP_K": "64",
                "MM_LAB_MODEL_B_DEFAULT_PRESENCE_PENALTY": "1.2",
                "MM_LAB_MODEL_B_DEFAULT_FREQUENCY_PENALTY": "0.3",
                "MM_LAB_MODEL_B_DEFAULT_THINKING_MODE": "on",
                "MM_LAB_MODEL_B_DEFAULT_SHOW_REASONING": "1",
                "MM_LAB_MODEL_B_DEFAULT_MEASURE_TTFT": "0",
                "MM_LAB_DEFAULT_TIMEOUT_SECONDS": "210",
                "MM_LAB_DEFAULT_TARGET_HEIGHT": "640",
                "MM_LAB_DEFAULT_VIDEO_FPS": "1.5",
                "MM_LAB_SAFE_VIDEO_SAMPLING": "1",
            },
            clear=False,
        ):
            clear_settings_cache()
            settings = get_settings()

        self.assertEqual(settings.model_a.label, "Qwen A")
        self.assertEqual(settings.model_a.base_url, "http://127.0.0.1:9000/v1")
        self.assertEqual(settings.model_a.model, "org/model-a")
        self.assertEqual(settings.model_a.api_key, "KEY_A")
        self.assertFalse(settings.model_a.request_defaults.use_model_defaults)
        self.assertEqual(settings.model_a.request_defaults.max_tokens, 10000)
        self.assertEqual(settings.model_a.request_defaults.max_completion_tokens, 9500)
        self.assertEqual(settings.model_a.request_defaults.temperature, 1.0)
        self.assertEqual(settings.model_a.request_defaults.top_p, 0.95)
        self.assertEqual(settings.model_a.request_defaults.top_k, 20)
        self.assertEqual(settings.model_a.request_defaults.presence_penalty, 1.5)
        self.assertEqual(settings.model_a.request_defaults.frequency_penalty, 0.0)
        self.assertEqual(settings.model_a.request_defaults.thinking_mode, "off")
        self.assertFalse(settings.model_a.request_defaults.show_reasoning)
        self.assertTrue(settings.model_a.request_defaults.measure_ttft)
        self.assertEqual(settings.model_b.label, "Gemma B")
        self.assertEqual(settings.model_b.base_url, "http://127.0.0.1:9001/v1")
        self.assertEqual(settings.model_b.model, "org/model-b")
        self.assertEqual(settings.model_b.api_key, "KEY_B")
        self.assertTrue(settings.model_b.request_defaults.use_model_defaults)
        self.assertEqual(settings.model_b.request_defaults.max_tokens, 12000)
        self.assertEqual(settings.model_b.request_defaults.max_completion_tokens, 8700)
        self.assertEqual(settings.model_b.request_defaults.temperature, 0.8)
        self.assertEqual(settings.model_b.request_defaults.top_p, 0.9)
        self.assertEqual(settings.model_b.request_defaults.top_k, 64)
        self.assertEqual(settings.model_b.request_defaults.presence_penalty, 1.2)
        self.assertEqual(settings.model_b.request_defaults.frequency_penalty, 0.3)
        self.assertEqual(settings.model_b.request_defaults.thinking_mode, "on")
        self.assertTrue(settings.model_b.request_defaults.show_reasoning)
        self.assertFalse(settings.model_b.request_defaults.measure_ttft)
        self.assertEqual(settings.default_timeout_seconds, 210.0)
        self.assertEqual(settings.default_target_height, 640)
        self.assertEqual(settings.default_video_fps, 1.5)
        self.assertTrue(settings.default_safe_video_sampling)

    def test_unset_generation_env_defaults_remain_unset(self) -> None:
        """Missing generation env vars should stay None for vLLM fallback behavior."""
        with (
            mock.patch.dict(
                os.environ,
                {
                    "MM_LAB_MODEL_A_LABEL": "Qwen A",
                    "MM_LAB_MODEL_B_LABEL": "Gemma B",
                },
                clear=True,
            ),
            mock.patch("visual_experimentation_app.config._load_dotenv_defaults"),
        ):
            clear_settings_cache()
            settings = get_settings()

        self.assertIsNone(settings.model_a.request_defaults.max_tokens)
        self.assertIsNone(settings.model_a.request_defaults.temperature)
        self.assertIsNone(settings.model_a.request_defaults.top_k)
        self.assertIsNone(settings.model_a.request_defaults.presence_penalty)
        self.assertIsNone(settings.model_b.request_defaults.max_completion_tokens)
        self.assertIsNone(settings.model_b.request_defaults.top_p)
        self.assertIsNone(settings.model_b.request_defaults.frequency_penalty)


if __name__ == "__main__":
    unittest.main()
