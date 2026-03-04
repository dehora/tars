# MCP Integration

tars consumes external tool servers via the [Model Context Protocol](https://modelcontextprotocol.io/). MCP servers are configured (not coded) — tars discovers their tools at startup, merges them into the tool list, and routes calls through the MCP client. Native tools stay native; MCP is an extension point.

## Configuration

Configure via `mcp_servers.json` in the memory dir or `TARS_MCP_SERVERS` env var:

```json
{
  "fetch": {
    "command": "uvx",
    "args": ["mcp-server-fetch"]
  },
  "github": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {"GITHUB_TOKEN": "..."}
  }
}
```

Format matches Claude Code's `mcpServers` config — keyed by server name, each with `command`, `args`, optional `env`.

## Tool naming

Tool names are prefixed with the server name: `fetch.fetch`, `github.create_issue`, etc. This avoids collisions between servers and with native tools.

MCP tool names are registered with the router at startup via `update_tool_names()`, so messages mentioning MCP tools trigger escalation to the remote model when configured.

## Commands

`/mcp` lists connected servers and their discovered tools.
