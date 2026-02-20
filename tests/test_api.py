import json
import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import api, conversation

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
        # Should have two delta events and one done event
        self.assertEqual(events[0], {"delta": "hel"})
        self.assertEqual(events[1], {"delta": "lo"})
        self.assertEqual(events[-1], {"done": True})
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
        rt.assert_called_once_with("todoist_today", {})

    def test_tool_endpoint_with_args(self) -> None:
        with mock.patch.object(api, "run_tool", return_value='{"ok": true}') as rt:
            resp = self.client.post("/tool", json={
                "name": "todoist_add_task",
                "args": {"content": "buy eggs", "due": "today"},
            })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["result"], "done")
        rt.assert_called_once_with("todoist_add_task", {"content": "buy eggs", "due": "today"})


if __name__ == "__main__":
    unittest.main()
