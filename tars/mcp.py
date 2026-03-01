"""MCP (Model Context Protocol) client integration.

Connects to external MCP servers configured in mcp_servers.json or
TARS_MCP_SERVERS env var. Discovers their tools at startup and routes
tool calls through the MCP client sessions.
"""

import asyncio
import atexit
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

from tars.memory import _memory_dir


@dataclass
class ServerInfo:
    name: str
    tools: list[dict] = field(default_factory=list)
    status: str = "disconnected"


def _load_mcp_config() -> dict:
    """Load MCP server config from mcp_servers.json or TARS_MCP_SERVERS env var.

    Returns empty dict if neither exists or config is invalid.
    """
    mem_dir = _memory_dir()
    if mem_dir is not None:
        config_path = mem_dir / "mcp_servers.json"
        if config_path.exists():
            try:
                text = config_path.read_text(encoding="utf-8", errors="replace")
                config = json.loads(text)
                if isinstance(config, dict):
                    return _validate_config(config)
                print("  [mcp] config is not a JSON object, ignoring", file=sys.stderr)
                return {}
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [mcp] failed to load mcp_servers.json: {e}", file=sys.stderr)
                return {}

    env_val = os.environ.get("TARS_MCP_SERVERS", "").strip()
    if env_val:
        try:
            config = json.loads(env_val)
            if isinstance(config, dict):
                return _validate_config(config)
            print("  [mcp] TARS_MCP_SERVERS is not a JSON object, ignoring", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"  [mcp] failed to parse TARS_MCP_SERVERS: {e}", file=sys.stderr)

    return {}


def _validate_config(config: dict) -> dict:
    """Validate and filter MCP server config entries."""
    valid = {}
    for name, entry in config.items():
        if not isinstance(entry, dict):
            print(f"  [mcp] skipping {name}: not a dict", file=sys.stderr)
            continue
        if "command" not in entry:
            print(f"  [mcp] skipping {name}: missing 'command'", file=sys.stderr)
            continue
        if not isinstance(entry.get("args", []), list):
            print(f"  [mcp] skipping {name}: 'args' must be a list", file=sys.stderr)
            continue
        valid[name] = entry
    return valid


class MCPClient:
    """Manages connections to MCP servers and routes tool calls.

    Runs a dedicated asyncio event loop in a background thread.
    Sync code schedules coroutines via asyncio.run_coroutine_threadsafe().
    """

    def __init__(self, server_configs: dict) -> None:
        self._configs = server_configs
        self._servers: dict[str, ServerInfo] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._stacks: dict[str, AsyncExitStack] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Connect to all configured servers, discover their tools."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True,
        )
        self._thread.start()

        for name, config in self._configs.items():
            try:
                self._connect_server(name, config)
                tool_count = len(self._servers[name].tools)
                print(f"  [mcp] {name}: connected ({tool_count} tools)", file=sys.stderr)
            except Exception as e:
                print(f"  [mcp] {name}: failed to connect: {e}", file=sys.stderr)
                self._servers[name] = ServerInfo(name=name, status=f"error: {e}")

        atexit.register(self.stop)

    def _run_async(self, coro):
        """Run an async coroutine from sync code in the background loop."""
        if self._loop is None:
            raise RuntimeError("MCP event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    def _connect_server(self, name: str, config: dict) -> None:
        """Connect to a single MCP server and discover its tools."""
        params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
        )
        tools = self._run_async(self._async_connect(name, params))
        self._servers[name] = ServerInfo(name=name, tools=tools, status="connected")

    async def _async_connect(self, name: str, params: StdioServerParameters) -> list[dict]:
        """Async: connect to server, initialize, discover tools.

        Uses AsyncExitStack to keep the stdio transport and session alive
        across multiple tool calls. The stack is closed in stop().
        """
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(params)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            result = await session.list_tools()

            self._sessions[name] = session
            self._stacks[name] = stack

            tools = []
            for tool in result.tools:
                tools.append({
                    "name": f"{name}.{tool.name}",
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                    "_server": name,
                    "_tool_name": tool.name,
                })
            return tools
        except Exception:
            await stack.aclose()
            raise

    def stop(self) -> None:
        """Disconnect from all servers and clean up.

        Graceful aclose() of MCP stacks is unreliable because anyio's
        cancel scopes can't cross task boundaries. Since server processes
        are children that die with the parent, we just stop the loop.
        """
        atexit.unregister(self.stop)
        if self._loop is None:
            return

        self._sessions.clear()
        self._stacks.clear()
        self._servers.clear()

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None

    def discover_tools(self) -> list[dict]:
        """Return all MCP tools in Anthropic tool schema format."""
        tools = []
        for info in self._servers.values():
            for tool in info.tools:
                tools.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                })
        return tools

    def call_tool(self, prefixed_name: str, args: dict) -> str:
        """Call an MCP tool by its prefixed name (server.tool_name).

        Returns the tool result as a string (text content joined),
        or a JSON error object on failure.
        """
        if "." not in prefixed_name:
            return json.dumps({"error": f"Invalid MCP tool name: {prefixed_name}"})

        dot = prefixed_name.index(".")
        server_name = prefixed_name[:dot]
        tool_name = prefixed_name[dot + 1:]

        session = self._sessions.get(server_name)
        if session is None:
            return json.dumps({"error": f"MCP server '{server_name}' not connected"})

        try:
            result = self._run_async(session.call_tool(tool_name, args))
        except Exception as e:
            return json.dumps({"error": f"MCP tool call failed: {e}"})

        if result.isError:
            parts = []
            for block in result.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            return json.dumps({"error": " ".join(parts) or "MCP tool error"})

        parts = []
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            else:
                parts.append(f"[{block.type} content]")
        return "\n".join(parts) if parts else json.dumps({"ok": True})

    def list_servers(self) -> list[dict]:
        """Return server status info for /mcp display."""
        result = []
        for info in self._servers.values():
            tool_names = [t["_tool_name"] for t in info.tools] if info.tools else []
            result.append({
                "name": info.name,
                "status": info.status,
                "tool_count": len(info.tools),
                "tools": tool_names,
            })
        return result
