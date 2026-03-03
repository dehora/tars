"""Shared startup/teardown for MCP and task runner across all channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tars.mcp import MCPClient
    from tars.taskrunner import TaskRunner


def start_services(provider: str, model: str) -> tuple[MCPClient | None, TaskRunner]:
    """Initialize MCP client and task runner. Returns (mcp_client, runner)."""
    from tars.commands import set_task_runner
    from tars.mcp import MCPClient, _load_mcp_config
    from tars.taskrunner import TaskRunner
    from tars.tools import set_mcp_client

    mcp_client = None
    mcp_config = _load_mcp_config()
    if mcp_config:
        mcp_client = MCPClient(mcp_config)
        mcp_client.start()
        set_mcp_client(mcp_client)
        from tars.router import update_tool_names
        update_tool_names({t["name"] for t in mcp_client.discover_tools()})

    runner = TaskRunner(provider, model)
    runner.start()
    set_task_runner(runner)

    return mcp_client, runner


def stop_services(mcp_client: MCPClient | None, runner: TaskRunner) -> None:
    """Shut down MCP client and task runner."""
    from tars.commands import set_task_runner
    from tars.tools import set_mcp_client

    runner.stop()
    set_task_runner(None)
    if mcp_client:
        mcp_client.stop()
        set_mcp_client(None)
