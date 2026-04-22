"""Tests for MM lab payload assembly helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.payload_builder import (  # noqa: E402
    build_messages,
    extract_message_parts,
    inject_prompt_query,
    merge_extra_body,
    model_supports_thinking_token_budget,
    normalize_base_url,
    parse_json_object,
    split_out_think_tags,
)


class PayloadBuilderTest(unittest.TestCase):
    """Covers URL normalization, JSON parsing, and multimodal message construction."""

    def test_normalize_base_url_appends_v1(self) -> None:
        """Base URL helper should normalize to `/v1` suffix."""
        self.assertEqual(normalize_base_url("http://localhost:8000"), "http://localhost:8000/v1")
        self.assertEqual(normalize_base_url("http://localhost:8000/v1"), "http://localhost:8000/v1")

    def test_build_messages_supports_images_videos_and_cache_uuids(self) -> None:
        """Message payload should include all multimodal items and UUID hints."""
        messages = build_messages(
            prompt="Describe scene",
            text_input="extra context",
            image_data_urls=["data:image/png;base64,a", "data:image/png;base64,b"],
            video_data_urls=["data:video/mp4;base64,c", "data:video/mp4;base64,d"],
            image_cache_uuids=["img-1", "img-2"],
            video_cache_uuids=["vid-1", "vid-2"],
        )
        content = messages[0]["content"]
        assert isinstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["uuid"], "img-1")
        self.assertEqual(content[3]["type"], "video_url")
        self.assertEqual(content[3]["video_url"]["uuid"], "vid-1")
        self.assertEqual(content[4]["type"], "video_url")
        self.assertEqual(content[4]["video_url"]["uuid"], "vid-2")

    def test_build_messages_replaces_query_placeholder_with_text_input(self) -> None:
        """Object-detection prompts should substitute `{query}` from text input."""
        messages = build_messages(
            prompt="Requested: {query}",
            text_input="tabby cat",
            image_data_urls=[],
            video_data_urls=[],
            image_cache_uuids=[],
            video_cache_uuids=[],
        )
        content = messages[0]["content"]
        assert isinstance(content, list)
        self.assertEqual(content[0]["text"], "Requested: tabby cat")

    def test_inject_prompt_query_appends_text_when_no_placeholder_exists(self) -> None:
        """Non-placeholder prompts should retain legacy text-input appending behavior."""
        self.assertEqual(
            inject_prompt_query("Describe scene", "extra context"),
            "Describe scene\n\nextra context",
        )

    def test_split_out_think_tags_separates_reasoning_from_content(self) -> None:
        """Inline think tags should be split into visible answer and reasoning."""
        content, reasoning = split_out_think_tags(
            "<think>hidden reasoning</think>\n\nFinal answer here."
        )
        self.assertEqual(content, "Final answer here.")
        self.assertEqual(reasoning, "hidden reasoning")

    def test_extract_message_parts_uses_think_tags_when_reasoning_field_missing(self) -> None:
        """Assistant content with think tags should still populate reasoning output."""
        message = type(
            "Message",
            (),
            {"content": "<think>step by step</think>\nVisible answer", "reasoning": None},
        )()
        content, reasoning = extract_message_parts(message)
        self.assertEqual(content, "Visible answer")
        self.assertEqual(reasoning, "step by step")

    def test_merge_extra_body_applies_safe_defaults(self) -> None:
        """Extra body merge should inject safe video and thinking defaults."""
        merged = merge_extra_body(
            user_extra_body={},
            include_video=True,
            safe_video_sampling=True,
            video_sampling_fps=2.0,
            thinking_mode="off",
            top_k=20,
        )
        self.assertEqual(merged["top_k"], 20)
        self.assertEqual(merged["mm_processor_kwargs"]["do_sample_frames"], False)
        self.assertEqual(merged["chat_template_kwargs"]["enable_thinking"], False)

    def test_merge_extra_body_preserves_explicit_top_k_override(self) -> None:
        """User-specified top_k in extra_body should win over slider default injection."""
        merged = merge_extra_body(
            user_extra_body={"top_k": 77},
            include_video=False,
            safe_video_sampling=False,
            video_sampling_fps=None,
            thinking_mode="off",
            top_k=20,
        )
        self.assertEqual(merged["top_k"], 77)

    def test_merge_extra_body_sets_thinking_flag_when_enabled(self) -> None:
        """Thinking mode should propagate enable_thinking into chat template kwargs."""
        merged = merge_extra_body(
            user_extra_body={},
            include_video=False,
            safe_video_sampling=False,
            video_sampling_fps=None,
            thinking_mode="on",
            top_k=None,
        )
        self.assertTrue(merged["chat_template_kwargs"]["enable_thinking"])

    def test_merge_extra_body_sets_gemma_max_soft_tokens_for_gemma_models(self) -> None:
        """Gemma runs should inject max_soft_tokens into mm_processor_kwargs."""
        merged = merge_extra_body(
            user_extra_body={},
            include_video=False,
            safe_video_sampling=False,
            video_sampling_fps=None,
            thinking_mode="off",
            top_k=None,
            model_name="RedHatAI/gemma-4-31B-it-FP8-block",
            gemma_max_soft_tokens=560,
        )
        self.assertEqual(merged["mm_processor_kwargs"]["max_soft_tokens"], 560)

    def test_merge_extra_body_sets_thinking_token_budget_for_supported_models(self) -> None:
        """Qwen3-family models should forward thinking_token_budget."""
        merged = merge_extra_body(
            user_extra_body={},
            include_video=False,
            safe_video_sampling=False,
            video_sampling_fps=None,
            thinking_mode="on",
            top_k=None,
            model_name="Qwen/Qwen3.5-27B-FP8",
            thinking_token_budget=2048,
        )
        self.assertEqual(merged["thinking_token_budget"], 2048)

    def test_merge_extra_body_skips_thinking_token_budget_for_unsupported_models(self) -> None:
        """Unsupported model families should not receive thinking_token_budget."""
        merged = merge_extra_body(
            user_extra_body={},
            include_video=False,
            safe_video_sampling=False,
            video_sampling_fps=None,
            thinking_mode="on",
            top_k=None,
            model_name="RedHatAI/gemma-4-31B-it-FP8-block",
            thinking_token_budget=2048,
        )
        self.assertNotIn("thinking_token_budget", merged)

    def test_merge_extra_body_does_not_set_gemma_budget_for_non_gemma_models(self) -> None:
        """Non-Gemma models should ignore gemma_max_soft_tokens override."""
        merged = merge_extra_body(
            user_extra_body={},
            include_video=False,
            safe_video_sampling=False,
            video_sampling_fps=None,
            thinking_mode="off",
            top_k=None,
            model_name="Qwen/Qwen3.5-27B-FP8",
            gemma_max_soft_tokens=560,
        )
        self.assertNotIn("mm_processor_kwargs", merged)

    def test_model_supports_thinking_token_budget_detects_qwen3_family(self) -> None:
        """Qwen3-family model names should be recognized as budget-capable."""
        self.assertTrue(model_supports_thinking_token_budget("Qwen/Qwen3.5-27B-FP8"))
        self.assertFalse(
            model_supports_thinking_token_budget("RedHatAI/gemma-4-31B-it-FP8-block")
        )

    def test_parse_json_object_rejects_non_object(self) -> None:
        """JSON parser should reject payloads that are not object-shaped."""
        with self.assertRaises(ValueError):
            parse_json_object('["a"]', field_name="bad_json")


if __name__ == "__main__":
    unittest.main()
