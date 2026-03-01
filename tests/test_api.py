import json
import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import api, conversation
from tars.search import SearchResult

from fastapi.testclient import TestClient


class ChatEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        api._conversations.clear()
        api._session_files.clear()
        self.client = TestClient(api.app)

    def test_chat_creates_conversation(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="hello back"):
            resp = self.client.post("/chat", json={
                "conversation_id": "test1",
                "message": "hello",
            })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["conversation_id"], "test1")
        self.assertEqual(data["reply"], "hello back")
        self.assertIn("test1", api._conversations)

    def test_chat_continues_conversation(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="ok"):
            self.client.post("/chat", json={
                "conversation_id": "test2",
                "message": "first",
            })
            self.client.post("/chat", json={
                "conversation_id": "test2",
                "message": "second",
            })
        conv = api._conversations["test2"]
        self.assertEqual(conv.msg_count, 2)
        self.assertEqual(len(conv.messages), 4)

    def test_list_conversations(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="ok"):
            self.client.post("/chat", json={
                "conversation_id": "a",
                "message": "hi",
            })
        resp = self.client.get("/conversations")
        self.assertEqual(resp.status_code, 200)
        convos = resp.json()["conversations"]
        self.assertEqual(len(convos), 1)
        self.assertEqual(convos[0]["id"], "a")
        self.assertEqual(convos[0]["message_count"], 1)

    def test_delete_conversation(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="ok"):
            self.client.post("/chat", json={
                "conversation_id": "del",
                "message": "hi",
            })
        with mock.patch.object(conversation, "_summarize_session", return_value="s"):
            with mock.patch.object(conversation, "_save_session"):
                resp = self.client.delete("/conversations/del")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("del", api._conversations)

    def test_delete_nonexistent_conversation(self) -> None:
        resp = self.client.delete("/conversations/nope")
        self.assertEqual(resp.status_code, 404)

    def test_save_conversation(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="ok"):
            self.client.post("/chat", json={
                "conversation_id": "save1",
                "message": "hi",
            })
        with mock.patch.object(api, "save_session") as save:
            resp = self.client.post("/conversations/save1/save")
        self.assertEqual(resp.status_code, 200)
        save.assert_called_once()
        # Conversation should still exist (not deleted).
        self.assertIn("save1", api._conversations)

    def test_save_nonexistent_conversation(self) -> None:
        resp = self.client.post("/conversations/nope/save")
        self.assertEqual(resp.status_code, 404)

    def test_chat_stream_returns_sse(self) -> None:
        with mock.patch.object(conversation, "chat_stream", return_value=iter(["hel", "lo"])):
            resp = self.client.post("/chat/stream", json={
                "conversation_id": "stream1",
                "message": "hi",
            })
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers["content-type"])
        # Parse SSE events from the response body
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        # Should have two delta events, one done event, and one meta event
        self.assertEqual(events[0], {"delta": "hel"})
        self.assertEqual(events[1], {"delta": "lo"})
        self.assertEqual(events[2], {"done": True})
        self.assertIn("meta", events[-1])
        self.assertIn("model", events[-1]["meta"])
        # Conversation state should be updated
        self.assertIn("stream1", api._conversations)
        self.assertEqual(api._conversations["stream1"].msg_count, 1)


    def test_feedback_saves_correction(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="bad reply"):
            self.client.post("/chat", json={
                "conversation_id": "fb1",
                "message": "add eggs",
            })
        with mock.patch.object(api, "save_correction", return_value="feedback saved") as save:
            resp = self.client.post("/feedback", json={
                "conversation_id": "fb1",
            })
        self.assertEqual(resp.status_code, 200)
        save.assert_called_once_with("add eggs", "bad reply", "")

    def test_feedback_with_note(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="wrong"):
            self.client.post("/chat", json={
                "conversation_id": "fb2",
                "message": "hello",
            })
        with mock.patch.object(api, "save_correction", return_value="feedback saved") as save:
            resp = self.client.post("/feedback", json={
                "conversation_id": "fb2",
                "note": "should use todoist",
            })
        self.assertEqual(resp.status_code, 200)
        save.assert_called_once_with("hello", "wrong", "should use todoist")

    def test_feedback_reward(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="good reply"):
            self.client.post("/chat", json={
                "conversation_id": "fb3",
                "message": "add eggs",
            })
        with mock.patch.object(api, "save_reward", return_value="feedback saved") as save:
            resp = self.client.post("/feedback", json={
                "conversation_id": "fb3",
                "kind": "reward",
                "note": "nice",
            })
        self.assertEqual(resp.status_code, 200)
        save.assert_called_once_with("add eggs", "good reply", "nice")

    def test_feedback_no_messages(self) -> None:
        resp = self.client.post("/feedback", json={
            "conversation_id": "empty",
        })
        self.assertEqual(resp.status_code, 400)

    def test_tool_endpoint(self) -> None:
        raw = json.dumps({"results": [{"content": "task", "priority": 1, "due": {"string": "today"}, "duration": None}], "nextCursor": None})
        with mock.patch.object(api, "run_tool", return_value=raw) as rt:
            resp = self.client.post("/tool", json={
                "name": "todoist_today",
                "args": {},
            })
        self.assertEqual(resp.status_code, 200)
        # Result should be formatted, not raw JSON
        self.assertIn("[p1] task", resp.json()["result"])
        rt.assert_called_once_with("todoist_today", {}, quiet=True)

    def test_tool_endpoint_with_args(self) -> None:
        with mock.patch.object(api, "run_tool", return_value='{"ok": true}') as rt:
            resp = self.client.post("/tool", json={
                "name": "todoist_add_task",
                "args": {"content": "buy eggs", "due": "today"},
            })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["result"], "done")
        rt.assert_called_once_with("todoist_add_task", {"content": "buy eggs", "due": "today"}, quiet=True)

    def test_tool_endpoint_rejects_unknown_tool(self) -> None:
        resp = self.client.post("/tool", json={
            "name": "evil_tool",
            "args": {},
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown tool", resp.json()["detail"])

    def test_model_endpoint(self) -> None:
        resp = self.client.get("/model")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("primary", data)
        self.assertIn("remote", data)
        self.assertIn("routing_policy", data)


    def test_search_endpoint_returns_results(self) -> None:
        result = SearchResult(
            content="test content", score=0.9, file_path="/a.md",
            file_title="A", memory_type="semantic",
            start_line=1, end_line=5, chunk_rowid=1,
        )
        with mock.patch.object(api, "memory_search", return_value=[result]):
            resp = self.client.get("/search?q=test")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["content"], "test content")
        self.assertAlmostEqual(data["results"][0]["score"], 0.9)

    def test_search_endpoint_empty_query(self) -> None:
        resp = self.client.get("/search?q=")
        self.assertEqual(resp.status_code, 400)

    def test_search_endpoint_no_results(self) -> None:
        with mock.patch.object(api, "memory_search", return_value=[]):
            resp = self.client.get("/search?q=nothing")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["results"], [])

    def test_sessions_endpoint(self) -> None:
        from tars.sessions import SessionInfo
        fake = [
            SessionInfo(path=None, date="2026-02-20 15:45", title="Weather chat", filename="2026-02-20T15-45-00"),
        ]
        with mock.patch.object(api, "list_sessions", return_value=fake):
            resp = self.client.get("/sessions")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["sessions"]), 1)
        self.assertEqual(data["sessions"][0]["title"], "Weather chat")
        self.assertEqual(data["sessions"][0]["date"], "2026-02-20 15:45")

    def test_session_search_endpoint(self) -> None:
        episodic = SearchResult(
            content="session content", score=0.8, file_path="/s.md",
            file_title="S", memory_type="episodic",
            start_line=1, end_line=5, chunk_rowid=1,
        )
        semantic = SearchResult(
            content="memory", score=0.9, file_path="/m.md",
            file_title="M", memory_type="semantic",
            start_line=1, end_line=3, chunk_rowid=2,
        )
        with mock.patch.object(api, "memory_search", return_value=[episodic, semantic]):
            resp = self.client.get("/sessions/search?q=weather")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Should only return episodic results
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["content"], "session content")

    def test_session_search_empty_query(self) -> None:
        resp = self.client.get("/sessions/search?q=")
        self.assertEqual(resp.status_code, 400)

    def test_find_endpoint_returns_results(self) -> None:
        result = SearchResult(
            content="daily note", score=0.85, file_path="/notes/daily.md",
            file_title="Daily", memory_type="notes",
            start_line=1, end_line=3, chunk_rowid=1,
        )
        with mock.patch.object(api, "search_notes", return_value=[result]):
            resp = self.client.get("/find?q=daily")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["content"], "daily note")
        self.assertAlmostEqual(data["results"][0]["score"], 0.85)

    def test_find_endpoint_empty_query(self) -> None:
        resp = self.client.get("/find?q=")
        self.assertEqual(resp.status_code, 400)

    def test_find_endpoint_no_results(self) -> None:
        with mock.patch.object(api, "search_notes", return_value=[]):
            resp = self.client.get("/find?q=nothing")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["results"], [])

    def test_brief_endpoint(self) -> None:
        def fake_run_tool(name, args, *, quiet=False):
            if name == "todoist_today":
                return '{"results": [{"content": "buy eggs", "priority": 1, "due": {"string": "today"}, "duration": null}], "nextCursor": null}'
            if name == "weather_now":
                return '{"current": {"temperature_c": 10, "conditions": "Clear", "wind_speed_kmh": 5, "precipitation_mm": 0}}'
            if name == "weather_forecast":
                return '{"hourly": []}'
            return '{}'

        with mock.patch.object(api, "run_tool", side_effect=fake_run_tool):
            resp = self.client.get("/brief")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("todoist_today", data["sections"])
        self.assertIn("weather_now", data["sections"])
        self.assertIn("weather_forecast", data["sections"])

    def test_brief_endpoint_handles_failure(self) -> None:
        def fake_run_tool(name, args, *, quiet=False):
            if name == "todoist_today":
                raise FileNotFoundError("td not found")
            if name == "weather_now":
                return '{"current": {"temperature_c": 10, "conditions": "Clear", "wind_speed_kmh": 5, "precipitation_mm": 0}}'
            return '{"hourly": []}'

        with mock.patch.object(api, "run_tool", side_effect=fake_run_tool):
            resp = self.client.get("/brief")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("unavailable", data["sections"]["todoist_today"])
        self.assertNotIn("unavailable", data["sections"]["weather_now"])


    def test_mcp_endpoint_no_client(self) -> None:
        with mock.patch("tars.tools.get_mcp_client", return_value=None):
            resp = self.client.get("/mcp")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["servers"], [])

    def test_mcp_endpoint_with_servers(self) -> None:
        fake_client = mock.Mock()
        fake_client.list_servers.return_value = [
            {"name": "test", "status": "connected", "tool_count": 2, "tools": ["a", "b"]}
        ]
        with mock.patch("tars.tools.get_mcp_client", return_value=fake_client):
            resp = self.client.get("/mcp")
        self.assertEqual(resp.status_code, 200)
        servers = resp.json()["servers"]
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0]["name"], "test")

    def test_schedule_endpoint(self) -> None:
        with mock.patch("tars.scheduler.schedule_list", return_value=[]):
            with mock.patch("tars.commands.get_task_runner", return_value=None):
                resp = self.client.get("/schedule")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["os"], [])
        self.assertEqual(data["in_process"], [])

    def test_export_conversation(self) -> None:
        with mock.patch.object(conversation, "chat", return_value="hello back"):
            self.client.post("/chat", json={
                "conversation_id": "exp1",
                "message": "hello",
            })
        resp = self.client.get("/conversations/exp1/export")
        self.assertEqual(resp.status_code, 200)
        md = resp.json()["markdown"]
        self.assertIn("# Conversation exp1", md)
        self.assertIn("hello", md)
        self.assertIn("hello back", md)

    def test_export_nonexistent_conversation(self) -> None:
        resp = self.client.get("/conversations/nope/export")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
