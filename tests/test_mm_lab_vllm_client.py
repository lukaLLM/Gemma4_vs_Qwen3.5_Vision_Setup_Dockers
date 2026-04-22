"""Tests for compare-lab vLLM client helpers and orchestration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visual_experimentation_app.config import (  # noqa: E402
    LabSettings,
    TargetDefaults,
    TargetRequestDefaults,
)
from visual_experimentation_app.media_preprocess import PreparedMedia  # noqa: E402
from visual_experimentation_app.schemas import CompareRequest, CompareTargetConfig  # noqa: E402
from visual_experimentation_app.vllm_client import (  # noqa: E402
    TokenUsage,
    _extract_usage_tokens,
    _format_assistant_output,
    _sum_token_usage,
    _tokens_per_second,
    build_execution_error_details,
    execute_compare,
    is_video_processor_error,
    summarize_execution_error,
)


class _FakeExc(Exception):
    """Exception type with optional response/body attributes for tests."""

    def __init__(self, message: str, *, body: object | None = None) -> None:
        super().__init__(message)
        self.body = body


class VllmClientErrorHelpersTest(unittest.TestCase):
    """Covers concise error summarization and processor error detection."""

    def test_detects_qwen3vl_processor_error(self) -> None:
        """Known Qwen3VLProcessor errors should be classified as processor failures."""
        exc = _FakeExc("Failed to apply Qwen3VLProcessor on request payload")
        self.assertTrue(is_video_processor_error(exc))

    def test_summary_returns_actionable_hint_for_processor_error(self) -> None:
        """Processor failures should return concise actionable guidance."""
        exc = _FakeExc("Error code: 400 - Failed to apply Qwen3VLProcessor")
        summary = summarize_execution_error(exc)
        self.assertIn("Qwen3VLProcessor", summary)
        self.assertIn("Safe video sampling", summary)

    def test_build_details_keeps_raw_error_and_flag(self) -> None:
        """Detailed payload should preserve raw error text for diagnostics."""
        exc = _FakeExc(
            "Error code: 400",
            body={"error": {"message": "Failed to apply Qwen3VLProcessor"}},
        )
        details = build_execution_error_details(exc)
        self.assertEqual(details["error_type"], "_FakeExc")
        self.assertTrue(details["is_video_processor_error"])
        self.assertIn("Error code: 400", details["raw_error"])
        self.assertIn("hint", details)

    def test_summary_truncates_long_non_processor_errors(self) -> None:
        """Generic very long errors should be truncated for UI readability."""
        long_error = "x" * 1000
        exc = _FakeExc(long_error)
        summary = summarize_execution_error(exc)
        self.assertTrue(summary.endswith("..."))
        self.assertLess(len(summary), len(long_error))

    def test_extract_usage_tokens_from_non_stream_response(self) -> None:
        """Usage extraction should read prompt/completion/total token fields."""
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=111, completion_tokens=222, total_tokens=333)
        )
        usage = _extract_usage_tokens(response)
        self.assertEqual(usage.prompt_tokens, 111)
        self.assertEqual(usage.output_tokens, 222)
        self.assertEqual(usage.total_tokens, 333)

    def test_extract_usage_tokens_from_stream_chunk(self) -> None:
        """Stream chunks with usage payload should be parsed identically."""
        chunk = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
        usage = _extract_usage_tokens(chunk)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)

    def test_extract_usage_tokens_missing_usage_returns_none_fields(self) -> None:
        """Missing usage should map to a fully-null usage struct."""
        usage = _extract_usage_tokens(SimpleNamespace(usage=None))
        self.assertIsNone(usage.prompt_tokens)
        self.assertIsNone(usage.output_tokens)
        self.assertIsNone(usage.total_tokens)

    def test_sum_token_usage_supports_partial_segment_data(self) -> None:
        """Summation should accumulate available segment usage and ignore missing values."""
        usage = _sum_token_usage(
            [
                TokenUsage(prompt_tokens=5, output_tokens=10, total_tokens=15),
                TokenUsage(prompt_tokens=None, output_tokens=None, total_tokens=None),
                TokenUsage(prompt_tokens=7, output_tokens=14, total_tokens=21),
            ]
        )
        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.output_tokens, 24)
        self.assertEqual(usage.total_tokens, 36)

    def test_tokens_per_second_returns_none_for_missing_or_zero_inputs(self) -> None:
        """Throughput helper should handle missing token or timing values."""
        self.assertIsNone(_tokens_per_second(output_tokens=None, request_ms=120.0))
        self.assertIsNone(_tokens_per_second(output_tokens=42, request_ms=0.0))

    def test_tokens_per_second_computes_expected_rate(self) -> None:
        """Throughput helper should compute output token rate in seconds."""
        self.assertEqual(_tokens_per_second(output_tokens=250, request_ms=1000.0), 250.0)

    def test_format_assistant_output_with_reasoning_uses_markdown_sections(self) -> None:
        """Thinking output should render as a styled reasoning panel plus answer heading."""
        rendered = _format_assistant_output(
            content="Final answer text.",
            reasoning="Chain of thought summary.",
            include_reasoning=True,
        )
        self.assertIn('<details class="reasoning-shell" open>', rendered)
        self.assertIn("<summary>Reasoning</summary>", rendered)
        self.assertIn("Chain of thought summary.", rendered)
        self.assertIn("### Final Answer", rendered)
        self.assertNotIn("<think>", rendered)


class CompareExecutionTest(unittest.TestCase):
    """Covers shared-preprocessing compare orchestration."""

    def test_execute_compare_reuses_preprocessed_payload_for_both_targets(self) -> None:
        """Shared media encoding should happen once and be reused by both targets."""
        request = CompareRequest(
            prompt="Describe the image.",
            image_paths=["/tmp/example.png"],
            model_a=CompareTargetConfig(label="Qwen", model="model-a", temperature=0.1),
            model_b=CompareTargetConfig(label="Gemma", model="model-b", temperature=0.8),
        )
        fake_prepared = PreparedMedia(
            image_paths=[Path("/tmp/example.png")],
            video_paths=[],
            cleanup_paths=[],
            metadata={"images": [{"source_path": "/tmp/example.png"}], "videos": [], "video": None},
        )
        encoded_paths: list[Path] = []
        captured_messages: list[list[dict[str, object]]] = []

        def fake_encode(path: Path) -> str:
            encoded_paths.append(path)
            return f"data://{path.name}"

        def fake_invoke_completion(
            *,
            client: object,
            target: CompareTargetConfig,
            model: str,
            messages: list[dict[str, object]],
            extra_body: dict[str, object],
        ) -> tuple[str, float | None, TokenUsage]:
            captured_messages.append(messages)
            return (
                f"output-{target.label}",
                12.5,
                TokenUsage(prompt_tokens=10, output_tokens=20, total_tokens=30),
            )

        with (
            mock.patch(
                "visual_experimentation_app.vllm_client.prepare_media",
                return_value=fake_prepared,
            ) as mock_prepare,
            mock.patch(
                "visual_experimentation_app.vllm_client.encode_file_to_data_url",
                side_effect=fake_encode,
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client._build_client",
                return_value=object(),
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client._invoke_completion",
                side_effect=fake_invoke_completion,
            ),
        ):
            result = execute_compare(request)

        self.assertEqual(mock_prepare.call_count, 1)
        self.assertEqual(encoded_paths, [Path("/tmp/example.png")])
        self.assertEqual(len(captured_messages), 2)
        self.assertIs(captured_messages[0], captured_messages[1])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.model_a_result.output_text, "output-Qwen")
        self.assertEqual(result.model_b_result.output_text, "output-Gemma")
        self.assertEqual(
            result.model_a_result.effective_params["sent_generation_params"]["temperature"],
            0.1,
        )
        self.assertEqual(
            result.model_b_result.effective_params["sent_generation_params"]["temperature"],
            0.8,
        )
        self.assertFalse(result.model_a_result.effective_params["text_only"])
        self.assertFalse(result.model_b_result.effective_params["text_only"])

    def test_execute_compare_applies_model_defaults_only_to_omitted_fields(self) -> None:
        """Configured defaults should fill omissions without overriding explicit values."""
        request = CompareRequest(
            prompt="Describe the image.",
            image_paths=["/tmp/example.png"],
            model_a=CompareTargetConfig(label="Qwen", model="model-a"),
            model_b=CompareTargetConfig(
                label="Gemma",
                model="model-b",
                temperature=0.8,
                top_k=42,
            ),
        )
        fake_settings = LabSettings(
            host="127.0.0.1",
            port=7870,
            ui_path="/",
            api_prefix="/api",
            results_dir=Path("/tmp/mm-results"),
            default_timeout_seconds=180.0,
            default_target_height=480,
            default_video_fps=1.0,
            default_safe_video_sampling=False,
            model_a=TargetDefaults(
                label="Qwen Default",
                base_url="http://127.0.0.1:8000/v1",
                model="qwen-default",
                api_key="EMPTY",
                request_defaults=TargetRequestDefaults(
                    use_model_defaults=False,
                    max_tokens=10000,
                    max_completion_tokens=9500,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                    presence_penalty=1.5,
                    frequency_penalty=0.0,
                    thinking_mode="off",
                    show_reasoning=False,
                    measure_ttft=True,
                ),
            ),
            model_b=TargetDefaults(
                label="Gemma Default",
                base_url="http://127.0.0.1:8001/v1",
                model="gemma-default",
                api_key="EMPTY",
                request_defaults=TargetRequestDefaults(
                    use_model_defaults=True,
                    max_tokens=10000,
                    max_completion_tokens=9500,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    presence_penalty=1.5,
                    frequency_penalty=0.0,
                    thinking_mode="off",
                    show_reasoning=False,
                    measure_ttft=True,
                ),
            ),
        )
        fake_prepared = PreparedMedia(
            image_paths=[Path("/tmp/example.png")],
            video_paths=[],
            cleanup_paths=[],
            metadata={"images": [{"source_path": "/tmp/example.png"}], "videos": [], "video": None},
        )
        captured_targets: dict[str, CompareTargetConfig] = {}

        def fake_invoke_completion(
            *,
            client: object,
            target: CompareTargetConfig,
            model: str,
            messages: list[dict[str, object]],
            extra_body: dict[str, object],
        ) -> tuple[str, float | None, TokenUsage]:
            captured_targets[target.label] = target
            return (
                f"output-{target.label}",
                10.0,
                TokenUsage(prompt_tokens=10, output_tokens=20, total_tokens=30),
            )

        with (
            mock.patch(
                "visual_experimentation_app.vllm_client.get_settings",
                return_value=fake_settings,
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client.prepare_media",
                return_value=fake_prepared,
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client.encode_file_to_data_url",
                return_value="data://example.png",
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client._build_client",
                return_value=object(),
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client._invoke_completion",
                side_effect=fake_invoke_completion,
            ),
        ):
            result = execute_compare(request)

        self.assertEqual(result.status, "ok")
        self.assertEqual(captured_targets["Qwen"].max_tokens, 10000)
        self.assertEqual(captured_targets["Qwen"].max_completion_tokens, 9500)
        self.assertEqual(captured_targets["Qwen"].temperature, 1.0)
        self.assertEqual(captured_targets["Qwen"].top_k, 20)
        self.assertFalse(captured_targets["Gemma"].use_model_defaults)
        self.assertEqual(captured_targets["Gemma"].temperature, 0.8)
        self.assertEqual(captured_targets["Gemma"].top_k, 42)
        self.assertEqual(
            result.model_a_result.effective_params["sent_generation_params"]["max_tokens"],
            10000,
        )
        self.assertEqual(
            result.model_b_result.effective_params["sent_generation_params"]["top_k"],
            42,
        )
        self.assertEqual(
            result.model_b_result.effective_params["sent_generation_params"]["max_tokens"],
            10000,
        )
        self.assertFalse(result.model_a_result.effective_params["text_only"])
        self.assertFalse(result.model_b_result.effective_params["text_only"])

    def test_execute_compare_applies_thinking_token_budget_only_to_supported_models(self) -> None:
        """Thinking budget should reach Qwen-family targets but not unsupported models."""
        request = CompareRequest(
            prompt="Describe the image.",
            image_paths=["/tmp/example.png"],
            model_a=CompareTargetConfig(
                label="Qwen",
                model="Qwen/Qwen3.5-27B-FP8",
                thinking_mode="on",
                thinking_token_budget=2048,
            ),
            model_b=CompareTargetConfig(
                label="Gemma",
                model="RedHatAI/gemma-4-31B-it-FP8-block",
                thinking_mode="on",
                thinking_token_budget=2048,
                gemma_max_soft_tokens=560,
            ),
        )
        fake_prepared = PreparedMedia(
            image_paths=[Path("/tmp/example.png")],
            video_paths=[],
            cleanup_paths=[],
            metadata={"images": [{"source_path": "/tmp/example.png"}], "videos": [], "video": None},
        )
        captured_extra_body: dict[str, dict[str, object]] = {}

        def fake_invoke_completion(
            *,
            client: object,
            target: CompareTargetConfig,
            model: str,
            messages: list[dict[str, object]],
            extra_body: dict[str, object],
        ) -> tuple[str, float | None, TokenUsage]:
            captured_extra_body[target.label] = dict(extra_body)
            return (
                f"output-{target.label}",
                12.0,
                TokenUsage(prompt_tokens=10, output_tokens=20, total_tokens=30),
            )

        with (
            mock.patch(
                "visual_experimentation_app.vllm_client.prepare_media",
                return_value=fake_prepared,
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client.encode_file_to_data_url",
                return_value="data://example.png",
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client._build_client",
                return_value=object(),
            ),
            mock.patch(
                "visual_experimentation_app.vllm_client._invoke_completion",
                side_effect=fake_invoke_completion,
            ),
        ):
            result = execute_compare(request)

        self.assertEqual(result.status, "ok")
        self.assertEqual(captured_extra_body["Qwen"]["thinking_token_budget"], 2048)
        self.assertNotIn("thinking_token_budget", captured_extra_body["Gemma"])
        self.assertEqual(
            captured_extra_body["Gemma"]["mm_processor_kwargs"]["max_soft_tokens"],
            560,
        )
        self.assertTrue(result.model_a_result.effective_params["thinking_token_budget_supported"])
        self.assertFalse(result.model_b_result.effective_params["thinking_token_budget_supported"])
        self.assertEqual(
            result.model_a_result.effective_params["thinking_token_budget_applied"],
            2048,
        )
        self.assertIsNone(result.model_b_result.effective_params["thinking_token_budget_applied"])
        self.assertEqual(
            result.model_b_result.effective_params["gemma_max_soft_tokens_applied"],
            560,
        )

if __name__ == "__main__":
    unittest.main()
