import unittest

from tars.config import ModelConfig
from tars.router import RouteResult, route_message


_ESC_CONFIG = ModelConfig(
    primary_provider="ollama",
    primary_model="llama3.1:8b",
    remote_provider="claude",
    remote_model="sonnet",
    routing_policy="tool",
)


class TestRouter(unittest.TestCase):
    def test_no_escalation_configured(self):
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider=None,
            remote_model=None,
            routing_policy="tool",
        )
        result = route_message("hello", config)
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "llama3.1:8b")

    def test_already_claude(self):
        config = ModelConfig(
            primary_provider="claude",
            primary_model="sonnet",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        result = route_message("add task buy milk", config)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_todoist_keywords(self):
        for msg in ["add buy milk to todoist", "remind me to call mom", "buy eggs"]:
            result = route_message(msg, _ESC_CONFIG)
            self.assertEqual(result.provider, "claude", f"failed for: {msg}")
            self.assertEqual(result.model, "sonnet", f"failed for: {msg}")

    def test_weather_keywords(self):
        result = route_message("what's the weather", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_plain_chat(self):
        result = route_message("hello how are you", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "llama3.1:8b")

    def test_case_insensitive(self):
        result = route_message("BUY EGGS", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_memory_keywords(self):
        result = route_message("remember that I like coffee", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_note_keywords(self):
        result = route_message("note: interesting idea", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_notes_keywords(self):
        for msg in ["check my notes", "search my daily note", "find it in obsidian"]:
            result = route_message(msg, _ESC_CONFIG)
            self.assertEqual(result.provider, "claude", f"failed for: {msg}")
            self.assertEqual(result.model, "sonnet", f"failed for: {msg}")

    def test_direct_tool_name(self):
        result = route_message("use weather_now for london", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertEqual(result.model, "sonnet")

    def test_todoist_hints(self):
        result = route_message("add task buy milk", _ESC_CONFIG)
        self.assertIn("todoist_add_task", result.tool_hints)

    def test_weather_hints(self):
        result = route_message("will it rain today?", _ESC_CONFIG)
        self.assertIn("weather_now", result.tool_hints)

    def test_no_hints_for_chat(self):
        result = route_message("hello how are you", _ESC_CONFIG)
        self.assertEqual(result.tool_hints, [])

    def test_multiple_hints_deduplicated(self):
        result = route_message("add task buy groceries from todoist", _ESC_CONFIG)
        self.assertEqual(len(result.tool_hints), len(set(result.tool_hints)))

    def test_direct_tool_name_hint(self):
        result = route_message("use weather_now", _ESC_CONFIG)
        self.assertEqual(result.tool_hints, ["weather_now"])

    def test_strava_my_night_run(self):
        result = route_message("why did my night run feel easier", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_activities", result.tool_hints)

    def test_strava_direct_mention(self):
        result = route_message("strava activities", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_activities", result.tool_hints)

    def test_strava_running_keyword(self):
        result = route_message("how far did I go running this week", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_activities", result.tool_hints)

    def test_strava_routes(self):
        result = route_message("show my routes", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_routes", result.tool_hints)

    def test_strava_pace(self):
        result = route_message("what was my pace yesterday", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_activities", result.tool_hints)

    def test_route_result_not_iterable(self):
        result = route_message("hello", _ESC_CONFIG)
        self.assertIsInstance(result, RouteResult)
        with self.assertRaises(TypeError):
            _, _ = result

    def test_false_positive_run_script(self):
        result = route_message("run this script", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")

    def test_false_positive_model_training(self):
        result = route_message("model training takes time", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")

    def test_false_positive_api_routes(self):
        result = route_message("these API routes are slow", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")

    def test_false_positive_pace_yourself(self):
        result = route_message("pace yourself", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")

    def test_qualified_training_escalates(self):
        result = route_message("how is my running training going", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")

    def test_qualified_pace_escalates(self):
        result = route_message("what was my running pace", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")

    def test_qualified_routes_escalates(self):
        result = route_message("show my strava routes", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")

    def test_analyse_my_training_escalates(self):
        result = route_message("analyse my training this week", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")

    def test_favourite_segments_escalates(self):
        result = route_message("show my favourite segments", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")

    def test_favorite_segments_escalates(self):
        result = route_message("show my favorite segments", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")

    def test_training_this_week_escalates(self):
        result = route_message("how was training this week", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")

    def test_false_positive_training_this_model(self):
        result = route_message("help with training this model", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")

    def test_zone_distribution_escalates(self):
        result = route_message("show me zone distribution", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_zones", result.tool_hints)

    def test_polarised_escalates(self):
        result = route_message("am I training polarised", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_zones", result.tool_hints)

    def test_polarized_us_spelling_escalates(self):
        result = route_message("is my training polarized", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_zones", result.tool_hints)

    def test_hr_zone_number_escalates(self):
        result = route_message("how much time in hr zone 2", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_zones", result.tool_hints)

    def test_zone_number_with_training_context(self):
        result = route_message("zone 2 training this week", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_zones", result.tool_hints)

    def test_time_in_zone_escalates(self):
        result = route_message("what is my time in zone breakdown", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_zones", result.tool_hints)

    def test_hr_zone_escalates_to_zones(self):
        result = route_message("show my hr zone data", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("strava_zones", result.tool_hints)

    def test_false_positive_plant_zone(self):
        result = route_message("what plants grow in zone 5", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")

    def test_false_positive_fire_zone(self):
        result = route_message("fire zone 2 evacuation", _ESC_CONFIG)
        self.assertEqual(result.provider, "ollama")

    def test_create_note_escalates(self):
        result = route_message("create a note for my pain log", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("note_write", result.tool_hints)

    def test_read_note_escalates(self):
        result = route_message("read my note about exercise", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("note_read", result.tool_hints)

    def test_append_to_log_escalates(self):
        result = route_message("add to my exercise log", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("note_append", result.tool_hints)

    def test_pain_log_escalates(self):
        result = route_message("update my pain log", _ESC_CONFIG)
        self.assertEqual(result.provider, "claude")
        self.assertIn("note_append", result.tool_hints)


if __name__ == "__main__":
    unittest.main()
