"""Tests for MCP client integration."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))


class ConfigTests(unittest.TestCase):
    """Test MCP config loading and validation."""

    def test_load_from_json_file(self) -> None:
        from tars.mcp import _load_mcp_config

        config = {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp_servers.json"
            config_path.write_text(json.dumps(config))
            with mock.patch("tars.mcp._memory_dir", return_value=Path(tmpdir)):
                result = _load_mcp_config()
        self.assertEqual(result, config)

    def test_load_from_env_var(self) -> None:
        from tars.mcp import _load_mcp_config

        config = {"github": {"command": "npx", "args": ["-y", "server-github"]}}
        with mock.patch("tars.mcp._memory_dir", return_value=None):
            with mock.patch.dict(os.environ, {"TARS_MCP_SERVERS": json.dumps(config)}):
                result = _load_mcp_config()
        self.assertEqual(result, config)

    def test_file_takes_precedence_over_env(self) -> None:
        from tars.mcp import _load_mcp_config

        file_config = {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}
        env_config = {"github": {"command": "npx", "args": ["-y", "server-github"]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp_servers.json"
            config_path.write_text(json.dumps(file_config))
            with mock.patch("tars.mcp._memory_dir", return_value=Path(tmpdir)):
                with mock.patch.dict(os.environ, {"TARS_MCP_SERVERS": json.dumps(env_config)}):
                    result = _load_mcp_config()
        self.assertEqual(result, file_config)

    def test_empty_config(self) -> None:
        from tars.mcp import _load_mcp_config

        with mock.patch("tars.mcp._memory_dir", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TARS_MCP_SERVERS", None)
                result = _load_mcp_config()
        self.assertEqual(result, {})

    def test_invalid_json(self) -> None:
        from tars.mcp import _load_mcp_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp_servers.json"
            config_path.write_text("not valid json{{{")
            with mock.patch("tars.mcp._memory_dir", return_value=Path(tmpdir)):
                result = _load_mcp_config()
        self.assertEqual(result, {})

    def test_missing_command_field(self) -> None:
        from tars.mcp import _validate_config

        config = {"bad": {"args": ["something"]}}
        result = _validate_config(config)
        self.assertEqual(result, {})

    def test_args_not_a_list(self) -> None:
        from tars.mcp import _validate_config

        config = {"bad": {"command": "uvx", "args": "not-a-list"}}
        result = _validate_config(config)
        self.assertEqual(result, {})

    def test_entry_not_a_dict(self) -> None:
        from tars.mcp import _validate_config

        config = {"bad": "just a string"}
        result = _validate_config(config)
        self.assertEqual(result, {})

    def test_valid_with_env(self) -> None:
        from tars.mcp import _validate_config

        config = {
            "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
            "github": {
                "command": "npx",
                "args": ["-y", "server-github"],
                "env": {"GITHUB_TOKEN": "abc"},
            },
        }
        result = _validate_config(config)
        self.assertEqual(result, config)

    def test_mixed_valid_invalid(self) -> None:
        from tars.mcp import _validate_config

        config = {
            "good": {"command": "uvx", "args": []},
            "bad_no_command": {"args": ["x"]},
            "bad_not_dict": "nope",
        }
        result = _validate_config(config)
        self.assertEqual(len(result), 1)
        self.assertIn("good", result)


class ToolDiscoveryTests(unittest.TestCase):
    """Test MCP tool discovery and schema conversion."""

    def _make_client_with_tools(self, server_name, tools):
        """Create an MCPClient with pre-populated server info (no real connection)."""
        from tars.mcp import MCPClient, ServerInfo

        client = MCPClient({})
        client._servers[server_name] = ServerInfo(
            name=server_name,
            tools=tools,
            status="connected",
        )
        return client

    def test_discover_tools_returns_anthropic_format(self) -> None:
        tools = [
            {
                "name": "fetch.fetch_url",
                "description": "Fetch a URL",
                "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}},
                "_server": "fetch",
                "_tool_name": "fetch_url",
            }
        ]
        client = self._make_client_with_tools("fetch", tools)
        result = client.discover_tools()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "fetch.fetch_url")
        self.assertEqual(result[0]["description"], "Fetch a URL")
        self.assertIn("input_schema", result[0])
        # Internal keys should not be in the output
        self.assertNotIn("_server", result[0])
        self.assertNotIn("_tool_name", result[0])

    def test_discover_tools_prefixed_names(self) -> None:
        tools = [
            {"name": "gh.create_issue", "description": "Create issue",
             "input_schema": {}, "_server": "gh", "_tool_name": "create_issue"},
            {"name": "gh.list_repos", "description": "List repos",
             "input_schema": {}, "_server": "gh", "_tool_name": "list_repos"},
        ]
        client = self._make_client_with_tools("gh", tools)
        names = [t["name"] for t in client.discover_tools()]
        self.assertEqual(names, ["gh.create_issue", "gh.list_repos"])

    def test_discover_empty_when_no_servers(self) -> None:
        from tars.mcp import MCPClient
        client = MCPClient({})
        self.assertEqual(client.discover_tools(), [])

    def test_merge_with_native_tools(self) -> None:
        from tars.tools import ANTHROPIC_TOOLS, get_all_tools

        tools = [
            {"name": "fetch.fetch_url", "description": "Fetch",
             "input_schema": {}, "_server": "fetch", "_tool_name": "fetch_url"},
        ]
        client = self._make_client_with_tools("fetch", tools)
        anthropic_tools, ollama_tools = get_all_tools(client)
        # Should have all native tools plus the MCP tool
        self.assertEqual(len(anthropic_tools), len(ANTHROPIC_TOOLS) + 1)
        self.assertEqual(len(ollama_tools), len(ANTHROPIC_TOOLS) + 1)
        # MCP tool should be at the end
        self.assertEqual(anthropic_tools[-1]["name"], "fetch.fetch_url")
        # Ollama format should have the MCP tool too
        self.assertEqual(ollama_tools[-1]["function"]["name"], "fetch.fetch_url")


class ToolCallTests(unittest.TestCase):
    """Test MCP tool call routing."""

    def _make_client_with_session(self, server_name):
        """Create an MCPClient with a mock session and a fake event loop."""
        from tars.mcp import MCPClient, ServerInfo

        client = MCPClient({})
        mock_session = mock.Mock()
        client._sessions[server_name] = mock_session
        client._servers[server_name] = ServerInfo(
            name=server_name,
            tools=[{"name": f"{server_name}.test_tool", "description": "Test",
                    "input_schema": {}, "_server": server_name, "_tool_name": "test_tool"}],
            status="connected",
        )
        # Stub _loop so _run_async doesn't bail on the None check
        client._loop = mock.Mock()
        return client, mock_session

    def test_call_tool_routes_to_correct_server(self) -> None:
        from mcp.types import CallToolResult, TextContent

        client, mock_session = self._make_client_with_session("fetch")
        result_obj = CallToolResult(
            content=[TextContent(type="text", text="page content")],
            isError=False,
        )
        mock_session.call_tool = mock.Mock(return_value=result_obj)
        with mock.patch.object(client, "_run_async", side_effect=lambda coro: result_obj):
            result = client.call_tool("fetch.fetch_url", {"url": "https://example.com"})
        self.assertEqual(result, "page content")

    def test_call_tool_strips_prefix(self) -> None:
        from mcp.types import CallToolResult, TextContent

        client, mock_session = self._make_client_with_session("gh")
        result_obj = CallToolResult(
            content=[TextContent(type="text", text="issue created")],
            isError=False,
        )
        mock_session.call_tool = mock.Mock(return_value=result_obj)
        with mock.patch.object(client, "_run_async", side_effect=lambda coro: result_obj):
            client.call_tool("gh.create_issue", {"title": "test"})
        # Verify session.call_tool was called with stripped name
        mock_session.call_tool.assert_called_once_with("create_issue", {"title": "test"})

    def test_call_tool_server_not_connected(self) -> None:
        from tars.mcp import MCPClient
        client = MCPClient({})
        result = client.call_tool("unknown.tool", {})
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("not connected", parsed["error"])

    def test_call_tool_error_response(self) -> None:
        from mcp.types import CallToolResult, TextContent

        client, _ = self._make_client_with_session("fetch")
        result_obj = CallToolResult(
            content=[TextContent(type="text", text="404 not found")],
            isError=True,
        )
        with mock.patch.object(client, "_run_async", side_effect=lambda coro: result_obj):
            result = client.call_tool("fetch.fetch_url", {"url": "https://bad.example"})
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("404", parsed["error"])

    def test_call_tool_exception(self) -> None:
        client, _ = self._make_client_with_session("fetch")
        with mock.patch.object(client, "_run_async", side_effect=RuntimeError("connection lost")):
            result = client.call_tool("fetch.fetch_url", {})
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("connection lost", parsed["error"])

    def test_call_tool_invalid_name_no_dot(self) -> None:
        from tars.mcp import MCPClient
        client = MCPClient({})
        result = client.call_tool("nodot", {})
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("Invalid MCP tool name", parsed["error"])


class DispatchTests(unittest.TestCase):
    """Test that run_tool routes MCP tools correctly."""

    def test_mcp_tool_dispatched(self) -> None:
        from tars import tools
        mock_client = mock.Mock()
        mock_client.call_tool.return_value = '{"result": "ok"}'
        original = tools._mcp_client
        try:
            tools._mcp_client = mock_client
            result = tools.run_tool("fetch.fetch_url", {"url": "https://example.com"}, quiet=True)
            mock_client.call_tool.assert_called_once_with(
                "fetch.fetch_url", {"url": "https://example.com"}
            )
            self.assertEqual(result, '{"result": "ok"}')
        finally:
            tools._mcp_client = original

    def test_native_tools_still_work_with_mcp_client(self) -> None:
        from tars import tools
        mock_client = mock.Mock()
        original = tools._mcp_client
        try:
            tools._mcp_client = mock_client
            with mock.patch("tars.tools._run_weather_tool", return_value='{"temp": 20}'):
                result = tools.run_tool("weather_now", {}, quiet=True)
            self.assertEqual(result, '{"temp": 20}')
            mock_client.call_tool.assert_not_called()
        finally:
            tools._mcp_client = original

    def test_unknown_tool_without_mcp(self) -> None:
        from tars import tools
        original = tools._mcp_client
        try:
            tools._mcp_client = None
            result = tools.run_tool("nonexistent.tool", {}, quiet=True)
            parsed = json.loads(result)
            self.assertIn("error", parsed)
            self.assertIn("Unknown tool", parsed["error"])
        finally:
            tools._mcp_client = original


class CommandTests(unittest.TestCase):
    """Test /mcp command dispatch."""

    def test_mcp_no_client(self) -> None:
        from tars.commands import dispatch
        with mock.patch("tars.tools.get_mcp_client", return_value=None):
            result = dispatch("/mcp")
        self.assertIn("no MCP servers configured", result)

    def test_mcp_no_servers(self) -> None:
        from tars.commands import dispatch
        mock_client = mock.Mock()
        mock_client.list_servers.return_value = []
        with mock.patch("tars.tools.get_mcp_client", return_value=mock_client):
            result = dispatch("/mcp")
        self.assertIn("no MCP servers connected", result)

    def test_mcp_lists_servers(self) -> None:
        from tars.commands import dispatch
        mock_client = mock.Mock()
        mock_client.list_servers.return_value = [
            {"name": "fetch", "status": "connected", "tool_count": 2,
             "tools": ["fetch_url", "fetch_html"]},
            {"name": "github", "status": "connected", "tool_count": 3,
             "tools": ["create_issue", "list_repos", "get_pr"]},
        ]
        with mock.patch("tars.tools.get_mcp_client", return_value=mock_client):
            result = dispatch("/mcp")
        self.assertIn("fetch (2 tools)", result)
        self.assertIn("fetch_url", result)
        self.assertIn("github (3 tools)", result)

    def test_mcp_shows_error_status(self) -> None:
        from tars.commands import dispatch
        mock_client = mock.Mock()
        mock_client.list_servers.return_value = [
            {"name": "broken", "status": "error: connection refused",
             "tool_count": 0, "tools": []},
        ]
        with mock.patch("tars.tools.get_mcp_client", return_value=mock_client):
            result = dispatch("/mcp")
        self.assertIn("broken", result)
        self.assertIn("error: connection refused", result)

    def test_mcp_in_command_names(self) -> None:
        from tars.commands import command_names
        self.assertIn("/mcp", command_names())


class ClientAccessorTests(unittest.TestCase):
    """Test get/set MCP client functions."""

    def test_set_and_get(self) -> None:
        from tars.tools import get_mcp_client, set_mcp_client
        original = get_mcp_client()
        try:
            sentinel = mock.Mock()
            set_mcp_client(sentinel)
            self.assertIs(get_mcp_client(), sentinel)
            set_mcp_client(None)
            self.assertIsNone(get_mcp_client())
        finally:
            set_mcp_client(original)


class ListServersTests(unittest.TestCase):
    """Test MCPClient.list_servers()."""

    def test_list_servers_info(self) -> None:
        from tars.mcp import MCPClient, ServerInfo

        client = MCPClient({})
        client._servers["fetch"] = ServerInfo(
            name="fetch",
            tools=[
                {"name": "fetch.get", "description": "Get URL",
                 "input_schema": {}, "_server": "fetch", "_tool_name": "get"},
                {"name": "fetch.post", "description": "Post URL",
                 "input_schema": {}, "_server": "fetch", "_tool_name": "post"},
            ],
            status="connected",
        )
        servers = client.list_servers()
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0]["name"], "fetch")
        self.assertEqual(servers[0]["status"], "connected")
        self.assertEqual(servers[0]["tool_count"], 2)
        self.assertEqual(servers[0]["tools"], ["get", "post"])


class RouterIntegrationTests(unittest.TestCase):
    """Test that MCP tool names integrate with the router."""

    def test_update_tool_names(self) -> None:
        from tars.router import _TOOL_NAMES, update_tool_names

        original_size = len(_TOOL_NAMES)
        update_tool_names({"fetch.fetch_url", "github.create_issue"})
        self.assertIn("fetch.fetch_url", _TOOL_NAMES)
        self.assertIn("github.create_issue", _TOOL_NAMES)
        # Clean up
        _TOOL_NAMES.discard("fetch.fetch_url")
        _TOOL_NAMES.discard("github.create_issue")
        self.assertEqual(len(_TOOL_NAMES), original_size)


if __name__ == "__main__":
    unittest.main()
