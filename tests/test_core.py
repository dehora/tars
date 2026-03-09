import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("openai", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import core
from tars.search import SearchResult


class ChatRoutingTests(unittest.TestCase):
    def test_chat_routes_to_anthropic(self) -> None:
        with mock.patch.object(core, "chat_anthropic", return_value="hi") as m:
            result = core.chat([{"role": "user", "content": "x"}], "claude", "sonnet")
        self.assertEqual(result, "hi")
        m.assert_called_once()

    def test_chat_routes_to_ollama(self) -> None:
        with mock.patch.object(core, "chat_ollama", return_value="hi") as m:
            result = core.chat([{"role": "user", "content": "x"}], "ollama", "llama3")
        self.assertEqual(result, "hi")
        m.assert_called_once()

    def test_chat_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            core.chat([], "unknown", "model")

    def test_chat_stream_routes_to_anthropic(self) -> None:
        with mock.patch.object(core, "chat_anthropic_stream", return_value=iter(["a"])) as m:
            result = list(core.chat_stream([{"role": "user", "content": "x"}], "claude", "sonnet"))
        self.assertEqual(result, ["a"])
        m.assert_called_once()

    def test_chat_stream_routes_to_ollama(self) -> None:
        with mock.patch.object(core, "chat_ollama_stream", return_value=iter(["b"])) as m:
            result = list(core.chat_stream([{"role": "user", "content": "x"}], "ollama", "llama3"))
        self.assertEqual(result, ["b"])
        m.assert_called_once()

    def test_chat_stream_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(core.chat_stream([], "unknown", "model"))

    def test_chat_passes_search_context(self) -> None:
        with mock.patch.object(core, "chat_ollama", return_value="ok") as m:
            core.chat([], "ollama", "m", search_context="ctx")
        m.assert_called_once_with([], "m", search_context="ctx", use_tools=True, tool_hints=None)


class SystemPromptContentTests(unittest.TestCase):
    def test_prompt_contains_routing_confidence(self) -> None:
        self.assertIn("ambiguous", core.SYSTEM_PROMPT)
        self.assertIn("clarifying question", core.SYSTEM_PROMPT)

    def test_prompt_no_blanket_must_call(self) -> None:
        self.assertNotIn("You MUST call", core.SYSTEM_PROMPT)


class BuildSystemPromptTests(unittest.TestCase):
    def test_without_context(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertEqual(prompt, core.SYSTEM_PROMPT)
        self.assertNotIn("<memory>", prompt)

    def test_with_memory(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value="- fact"),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("<memory>", prompt)
        self.assertIn("- fact", prompt)

    def test_with_search_context(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
        ):
            prompt = core._build_system_prompt(search_context="recent stuff")
        self.assertIn("<relevant-context>", prompt)
        self.assertIn("recent stuff", prompt)

    def test_tool_hints_in_prompt(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
        ):
            prompt = core._build_system_prompt(tool_hints=["todoist_add_task", "weather_now"])
        self.assertIn("<tool-hints>", prompt)
        self.assertIn("todoist_add_task", prompt)
        self.assertIn("weather_now", prompt)

    def test_procedural_in_prompt(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value="- always confirm tasks"),
            mock.patch.object(core, "_load_pinned", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("<procedural-rules>", prompt)
        self.assertIn("- always confirm tasks", prompt)

    def test_procedural_empty_excluded(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertNotIn("<procedural-rules>", prompt)

    def test_with_pinned(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value="- watching Severance S2"),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("<pinned>", prompt)
        self.assertIn("watching Severance S2", prompt)
        self.assertIn(core.MEMORY_PROMPT_PREFACE, prompt)

    def test_pinned_empty_excluded(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
        ):
            prompt = core._build_system_prompt()
        self.assertNotIn("<pinned>", prompt)

    def test_tool_hints_before_untrusted(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value="- fact"),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
        ):
            prompt = core._build_system_prompt(tool_hints=["weather_now"])
        hints_pos = prompt.index("<tool-hints>")
        preface_pos = prompt.index(core.MEMORY_PROMPT_PREFACE)
        self.assertLess(hints_pos, preface_pos)


class DailyContextCapTests(unittest.TestCase):
    def test_daily_context_capped(self) -> None:
        lines = [f"- {i:02d}:00 tool:weather — ok" for i in range(100)]
        big_daily = "\n".join(lines)
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
            mock.patch.object(core, "load_daily", return_value=big_daily),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("<daily-context ", prompt)
        # Should only contain the last _MAX_DAILY_LINES lines
        daily_section = prompt[prompt.index("<daily-context "):].split(">", 1)[1].split("</daily-context>")[0]
        daily_lines = [l for l in daily_section.strip().splitlines() if l.strip()]
        self.assertLessEqual(len(daily_lines), core._MAX_DAILY_LINES)

    def test_short_daily_not_truncated(self) -> None:
        daily = "- 08:00 tool:weather — sunny\n- 09:00 session compacted"
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
            mock.patch.object(core, "load_daily", return_value=daily),
        ):
            prompt = core._build_system_prompt()
        self.assertIn("tool:weather", prompt)
        self.assertIn("session compacted", prompt)


class SearchRelevantContextTests(unittest.TestCase):
    def test_empty_results(self) -> None:
        with mock.patch("tars.search.search", return_value=[]):
            result = core._search_relevant_context("hello")
        self.assertEqual(result, "")


def _make_result(file_id, seq, score, content="short content"):
    return SearchResult(
        content=content, score=score,
        file_path=f"/memory/file{file_id}.md", file_title=f"file{file_id}",
        memory_type="semantic", start_line=seq * 10 + 1, end_line=(seq + 1) * 10,
        chunk_rowid=file_id * 100 + seq, file_id=file_id, chunk_sequence=seq,
    )


class TwoPassPackingTests(unittest.TestCase):
    def test_dedupes_to_best_per_file(self) -> None:
        anchors = [
            _make_result(1, 0, 0.9),
            _make_result(1, 1, 0.5),
            _make_result(2, 0, 0.7),
        ]
        with (
            mock.patch("tars.search.search", return_value=anchors),
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("test")
        # file1 appears once (best score 0.9), file2 appears once
        self.assertEqual(result.count("file1"), 1)
        self.assertEqual(result.count("file2"), 1)

    def test_anchor_budget_caps_breadth(self) -> None:
        big = "x" * 6000  # ~1500 tokens each, budget is 900-2100 (adaptive)
        anchors = [
            _make_result(i, 0, 0.9 - i * 0.1, content=big)
            for i in range(5)
        ]
        with (
            mock.patch("tars.search.search", return_value=anchors),
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("test")
        # At ~1500 tokens each, only 1-2 fit within adaptive budget
        file_count = sum(1 for i in range(5) if f"file{i}" in result)
        self.assertLessEqual(file_count, 2)
        self.assertGreater(file_count, 0)

    def test_expansion_replaces_anchor(self) -> None:
        anchor = _make_result(1, 2, 0.9, content="anchor text")
        expanded = _make_result(1, 1, 0.9, content="neighbor anchor text neighbor")
        with (
            mock.patch("tars.search.search", return_value=[anchor]),
            mock.patch("tars.search.expand_results", return_value=[expanded]),
        ):
            result = core._search_relevant_context("test")
        self.assertIn("neighbor anchor text neighbor", result)
        self.assertNotIn("[semantic:file1:21-30]\nanchor text", result)

    def test_expansion_respects_budget(self) -> None:
        anchor = _make_result(1, 0, 0.9, content="tiny")
        huge_expansion = _make_result(1, 0, 0.9, content="x" * 50000)
        with (
            mock.patch("tars.search.search", return_value=[anchor]),
            mock.patch("tars.search.expand_results", return_value=[huge_expansion]),
        ):
            result = core._search_relevant_context("test")
        # Should keep thin anchor since expansion blows the budget
        self.assertIn("tiny", result)
        self.assertNotIn("x" * 100, result)

    def test_large_result_skipped_smaller_still_packed(self) -> None:
        """An oversized chunk shouldn't prevent smaller chunks from fitting."""
        anchors = [
            _make_result(1, 0, 0.9, content="best hit"),      # small, packed first
            _make_result(2, 0, 0.8, content="x" * 20000),     # huge, should be skipped
            _make_result(3, 0, 0.7, content="third hit"),      # small, should still fit
        ]
        with (
            mock.patch("tars.search.search", return_value=anchors),
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("test")
        self.assertIn("best hit", result)
        self.assertIn("third hit", result)
        self.assertNotIn("x" * 100, result)

    def test_thin_and_thick_mix(self) -> None:
        anchors = [
            _make_result(1, 0, 0.9, content="best hit"),
            _make_result(2, 0, 0.7, content="secondary hit"),
        ]
        expanded_1 = _make_result(1, 0, 0.9, content="expanded best hit with neighbors")
        with (
            mock.patch("tars.search.search", return_value=anchors),
            mock.patch("tars.search.expand_results", return_value=[expanded_1]),
        ):
            result = core._search_relevant_context("test")
        # Best hit should be expanded, secondary stays thin
        self.assertIn("expanded best hit with neighbors", result)
        self.assertIn("secondary hit", result)


class AnchorBudgetRatioTests(unittest.TestCase):
    def test_single_result_returns_max(self) -> None:
        results = [_make_result(1, 0, 0.9)]
        ratio = core._anchor_budget_ratio(results)
        self.assertAlmostEqual(ratio, core._ANCHOR_BUDGET_RATIO_MAX)

    def test_equal_scores_returns_max(self) -> None:
        results = [_make_result(1, 0, 0.8), _make_result(2, 0, 0.8)]
        ratio = core._anchor_budget_ratio(results)
        self.assertAlmostEqual(ratio, core._ANCHOR_BUDGET_RATIO_MAX)

    def test_dominant_top_returns_lower_ratio(self) -> None:
        results = [_make_result(1, 0, 1.0), _make_result(2, 0, 0.1)]
        ratio = core._anchor_budget_ratio(results)
        self.assertLess(ratio, core._ANCHOR_BUDGET_RATIO_MAX)
        self.assertGreaterEqual(ratio, core._ANCHOR_BUDGET_RATIO_MIN)

    def test_full_dominance_returns_min(self) -> None:
        results = [_make_result(1, 0, 1.0), _make_result(2, 0, 0.0)]
        ratio = core._anchor_budget_ratio(results)
        self.assertAlmostEqual(ratio, core._ANCHOR_BUDGET_RATIO_MIN)

    def test_ratio_bounded(self) -> None:
        for spread in [0.0, 0.3, 0.5, 0.8, 1.0]:
            results = [_make_result(1, 0, 1.0), _make_result(2, 0, 1.0 - spread)]
            ratio = core._anchor_budget_ratio(results)
            self.assertGreaterEqual(ratio, core._ANCHOR_BUDGET_RATIO_MIN)
            self.assertLessEqual(ratio, core._ANCHOR_BUDGET_RATIO_MAX)


class ToolLoopBoundTests(unittest.TestCase):
    def test_max_tool_rounds_constant(self) -> None:
        self.assertEqual(core._MAX_TOOL_ROUNDS, 10)

    def test_anthropic_loop_bounded(self) -> None:
        mock_response = mock.Mock()
        mock_response.stop_reason = "tool_use"
        mock_block = mock.Mock()
        mock_block.type = "tool_use"
        mock_block.name = "weather_now"
        mock_block.input = {}
        mock_block.id = "t1"
        mock_response.content = [mock_block]
        mock_client = mock.Mock()
        mock_client.messages.create.return_value = mock_response
        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"name": "weather_now"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
            mock.patch("anthropic.Anthropic", return_value=mock_client),
        ):
            result = core.chat_anthropic([{"role": "user", "content": "x"}], "sonnet")
        self.assertIn("maximum number of tool calls", result)
        self.assertEqual(mock_client.messages.create.call_count, core._MAX_TOOL_ROUNDS)

    def test_ollama_loop_bounded(self) -> None:
        mock_func = mock.Mock()
        mock_func.name = "weather_now"
        mock_func.arguments = {}
        mock_tool_call = mock.Mock()
        mock_tool_call.function = mock_func
        mock_message = mock.Mock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = ""
        mock_response = mock.Mock()
        mock_response.message = mock_message
        mock_ollama = mock.Mock()
        mock_ollama.chat = mock.Mock(return_value=mock_response)
        with (
            mock.patch.object(core, "ollama", mock_ollama),
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"name": "weather_now"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
        ):
            result = core.chat_ollama([{"role": "user", "content": "x"}], "llama3")
        self.assertIn("maximum number of tool calls", result)
        self.assertEqual(mock_ollama.chat.call_count, core._MAX_TOOL_ROUNDS)

    def test_anthropic_stream_loop_bounded(self) -> None:
        mock_response = mock.Mock()
        mock_response.stop_reason = "tool_use"
        mock_block = mock.Mock()
        mock_block.type = "tool_use"
        mock_block.name = "weather_now"
        mock_block.input = {}
        mock_block.id = "t1"
        mock_response.content = [mock_block]
        mock_client = mock.Mock()
        mock_client.messages.create.return_value = mock_response
        with (
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"name": "weather_now"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
            mock.patch("anthropic.Anthropic", return_value=mock_client),
        ):
            result = list(core.chat_anthropic_stream([{"role": "user", "content": "x"}], "sonnet"))
        self.assertEqual(len(result), 1)
        self.assertIn("maximum number of tool calls", result[0])

    def test_ollama_stream_loop_bounded(self) -> None:
        mock_func = mock.Mock()
        mock_func.name = "weather_now"
        mock_func.arguments = {}
        mock_tool_call = mock.Mock()
        mock_tool_call.function = mock_func
        mock_message = mock.Mock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = ""
        mock_response = mock.Mock()
        mock_response.message = mock_message
        mock_ollama = mock.Mock()
        mock_ollama.chat = mock.Mock(return_value=mock_response)
        with (
            mock.patch.object(core, "ollama", mock_ollama),
            mock.patch.object(core, "_build_system_prompt", return_value="sys"),
            mock.patch.object(core, "_get_tools", return_value=[{"name": "weather_now"}]),
            mock.patch.object(core, "run_tool", return_value="ok"),
            mock.patch.object(core, "append_daily"),
        ):
            result = list(core.chat_ollama_stream([{"role": "user", "content": "x"}], "llama3"))
        self.assertEqual(len(result), 1)
        self.assertIn("maximum number of tool calls", result[0])


class StreamUseToolsTests(unittest.TestCase):
    def test_chat_stream_use_tools_false_anthropic(self) -> None:
        with mock.patch.object(
            core, "chat_anthropic_stream", return_value=iter(["hi"]),
        ) as m:
            list(core.chat_stream(
                [{"role": "user", "content": "x"}], "claude", "sonnet",
                use_tools=False,
            ))
        _, kwargs = m.call_args
        self.assertFalse(kwargs["use_tools"])

    def test_chat_stream_use_tools_false_ollama(self) -> None:
        with mock.patch.object(
            core, "chat_ollama_stream", return_value=iter(["hi"]),
        ) as m:
            list(core.chat_stream(
                [{"role": "user", "content": "x"}], "ollama", "llama3",
                use_tools=False,
            ))
        _, kwargs = m.call_args
        self.assertFalse(kwargs["use_tools"])


class StreamNoPreflightTests(unittest.TestCase):
    def test_anthropic_stream_no_preflight_when_no_tools(self) -> None:
        mock_client = mock.Mock()
        mock_stream_ctx = mock.MagicMock()
        mock_stream_ctx.__enter__ = mock.Mock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = mock.Mock(return_value=False)
        mock_stream_ctx.text_stream = iter(["hello"])
        mock_client.messages.stream.return_value = mock_stream_ctx

        with mock.patch("tars.core.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            result = list(core.chat_anthropic_stream(
                [{"role": "user", "content": "hi"}], "sonnet",
                use_tools=False,
            ))
        mock_client.messages.create.assert_not_called()
        self.assertEqual(result, ["hello"])

    def test_ollama_stream_no_preflight_when_no_tools(self) -> None:
        chunk = mock.Mock()
        chunk.message.content = "hello"
        mock_ollama = mock.Mock()
        mock_ollama.chat.return_value = iter([chunk])
        with mock.patch.object(core, "ollama", mock_ollama):
            result = list(core.chat_ollama_stream(
                [{"role": "user", "content": "hi"}], "llama3",
                use_tools=False,
            ))
        self.assertEqual(mock_ollama.chat.call_count, 1)
        _, kwargs = mock_ollama.chat.call_args
        self.assertTrue(kwargs.get("stream"))
        self.assertEqual(result, ["hello"])


class DailyContextProvenanceTests(unittest.TestCase):
    def test_daily_context_has_type_attribute(self) -> None:
        with (
            mock.patch.object(core, "_load_memory", return_value=""),
            mock.patch.object(core, "_load_procedural", return_value=""),
            mock.patch.object(core, "_load_pinned", return_value=""),
            mock.patch.object(core, "load_daily", return_value="- 08:00 captured: example.com"),
        ):
            prompt = core._build_system_prompt()
        self.assertIn('type="tars-generated', prompt)
        self.assertIn("summarized web content", prompt)


class WeakResultExpansionTests(unittest.TestCase):
    def test_weak_results_trigger_expansion(self) -> None:
        weak = [_make_result(1, 0, 0.20)]
        # chunk_rowid=200 is new — not in baseline
        better = [_make_result(2, 0, 0.55, content="expanded hit")]
        with (
            mock.patch("tars.search.search", return_value=weak),
            mock.patch("tars.search.search_expanded", return_value=better) as mock_se,
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("vague query about something")
        mock_se.assert_called_once()
        self.assertIn("expanded hit", result)

    def test_strong_results_skip_expansion(self) -> None:
        strong = [_make_result(1, 0, 0.80, content="strong hit")]
        with (
            mock.patch("tars.search.search", return_value=strong),
            mock.patch("tars.search.search_expanded") as mock_se,
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("specific query")
        mock_se.assert_not_called()
        self.assertIn("strong hit", result)

    def test_empty_results_trigger_expansion(self) -> None:
        better = [_make_result(1, 0, 0.50, content="found via expansion")]
        with (
            mock.patch("tars.search.search", return_value=[]),
            mock.patch("tars.search.search_expanded", return_value=better) as mock_se,
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("obscure topic")
        mock_se.assert_called_once()
        self.assertIn("found via expansion", result)

    def test_expansion_failure_falls_back(self) -> None:
        weak = [_make_result(1, 0, 0.20, content="weak but present")]
        with (
            mock.patch("tars.search.search", return_value=weak),
            mock.patch("tars.search.search_expanded", side_effect=Exception("ollama down")),
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("test query")
        self.assertIn("weak but present", result)

    def test_expansion_same_chunks_keeps_original(self) -> None:
        original = [_make_result(1, 0, 0.28, content="original hit")]
        # Same chunk_rowid (100) — expansion found nothing new
        same = [_make_result(1, 0, 0.28, content="same chunk re-scored")]
        with (
            mock.patch("tars.search.search", return_value=original),
            mock.patch("tars.search.search_expanded", return_value=same),
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("borderline query")
        self.assertIn("original hit", result)

    def test_expansion_at_limit_with_new_files_wins(self) -> None:
        baseline = [_make_result(i, 0, 0.20) for i in range(20)]
        # Includes a new file (file_id=99) — expansion adds coverage
        expanded = [_make_result(i, 0, 0.20) for i in range(19)]
        expanded.append(_make_result(99, 0, 0.50, content="new file chunk"))
        with (
            mock.patch("tars.search.search", return_value=baseline),
            mock.patch("tars.search.search_expanded", return_value=expanded),
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("test at limit")
        self.assertIn("new file chunk", result)

    def test_expansion_new_chunk_same_file_keeps_baseline(self) -> None:
        baseline = [
            _make_result(1, 0, 0.20, content="file1 baseline"),
            _make_result(2, 0, 0.20, content="file2 baseline"),
        ]
        # Expanded returns different chunk from file_id=1 but drops file_id=2
        expanded = [_make_result(1, 1, 0.25, content="file1 different chunk")]
        with (
            mock.patch("tars.search.search", return_value=baseline),
            mock.patch("tars.search.search_expanded", return_value=expanded),
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            result = core._search_relevant_context("test same file")
        self.assertIn("file1 baseline", result)
        self.assertIn("file2 baseline", result)

    def test_expansion_uses_no_min_score(self) -> None:
        weak = [_make_result(1, 0, 0.20)]
        with (
            mock.patch("tars.search.search", return_value=weak),
            mock.patch("tars.search.search_expanded", return_value=[]) as mock_se,
            mock.patch("tars.search.expand_results", return_value=[]),
        ):
            core._search_relevant_context("test")
        _, kwargs = mock_se.call_args
        self.assertEqual(kwargs["min_score"], 0.0)


class ParseToolArgumentsTests(unittest.TestCase):
    def test_dict_passthrough(self) -> None:
        self.assertEqual(core._parse_tool_arguments({"a": 1}), {"a": 1})

    def test_json_string(self) -> None:
        self.assertEqual(core._parse_tool_arguments('{"a": 1}'), {"a": 1})

    def test_invalid_json_returns_empty(self) -> None:
        self.assertEqual(core._parse_tool_arguments("not json"), {})

    def test_list_returns_empty(self) -> None:
        self.assertEqual(core._parse_tool_arguments('[1, 2, 3]'), {})

    def test_scalar_returns_empty(self) -> None:
        self.assertEqual(core._parse_tool_arguments('42'), {})

    def test_string_json_returns_empty(self) -> None:
        self.assertEqual(core._parse_tool_arguments('"hello"'), {})

    def test_null_json_returns_empty(self) -> None:
        self.assertEqual(core._parse_tool_arguments('null'), {})


class ParseModelTests(unittest.TestCase):
    def test_valid_format(self) -> None:
        provider, model = core.parse_model("ollama:llama3")
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "llama3")

    def test_invalid_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            core.parse_model("nocolon")


if __name__ == "__main__":
    unittest.main()
