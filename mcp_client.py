"""MCP client wrapper.

It handles exactly three protocol-level chores so the agent loop
does not have to care about them:
  1. Launch the MCP server as a subprocess (stdio transport) and
     complete the initialize handshake;
  2. Convert the MCP tool list into the JSON Schema format that
     OpenAI tool calling expects;
  3. Flatten a tool call result (an MCP Content list) into a string.

Tests for this module live in tests/test_mcp_client.py (pytest).
This module only configures a module-level logger; applications
should configure logging handlers/levels themselves.
"""

import logging
import shlex
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Async client that talks to an MCP server over stdio."""

    def __init__(self, command: list[str]):
        """
        command: how to launch the MCP server, e.g. ["python", "mock_k8s_server.py"]
                 or ["npx", "-y", "mcp-server-kubernetes"] (real cluster).
        """
        if not command:
            raise ValueError("MCP server command must not be empty")
        self._command = command
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._tools_cache: list[dict] | None = None

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def connect(self) -> None:
        params = StdioServerParameters(
            command=self._command[0],
            args=self._command[1:],
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        logger.info("Connected to MCP server: %s", " ".join(self._command))

    async def close(self) -> None:
        await self._stack.aclose()
        self._session = None

    # -- Tool listing -------------------------------------------------------

    async def list_tools_openai(self) -> list[dict]:
        """Return tools in the format OpenAI chat.completions expects.

        MCP tool:      {name, description, inputSchema}
        OpenAI tool:   {type: "function",
                        function: {name, description, parameters: <inputSchema>}}
        This conversion is the key glue of the demo: the LLM only
        understands the OpenAI format, the MCP server only speaks
        MCP, and this layer translates between them.
        """
        if self._tools_cache is not None:
            return self._tools_cache
        assert self._session is not None, "not connected yet"
        result = await self._session.list_tools()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in result.tools
        ]
        self._tools_cache = tools
        return tools

    async def tool_names(self) -> list[str]:
        tools = await self.list_tools_openai()
        return [t["function"]["name"] for t in tools]

    # -- Tool calling -------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Execute one MCP tool call and return the flattened text result.

        Failures do not raise: the error is returned as a string so the
        agent can feed it back to the LLM as a tool result and let the
        model decide whether to retry, pick another tool, or explain.
        """
        assert self._session is not None, "not connected yet"
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as exc:  # protocol-level / subprocess errors
            logger.warning("Tool call failed (%s): %s", name, exc)
            return f"Tool call failed ({name}): {exc}"

        parts: list[str] = []
        for block in result.content:
            # TextContent has .text; other types (images, resources, ...)
            # degrade to their string representation
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        text = "\n".join(parts)

        if getattr(result, "isError", False):
            logger.warning("Tool returned an error (%s): %s", name, text)
            return f"Tool returned an error ({name}): {text}"
        logger.debug("Tool %s returned %d chars", name, len(text))
        return text


def parse_command(command_str: str) -> list[str]:
    """Split a string like "npx -y mcp-server-kubernetes" into an argv list."""
    # On Windows, shlex needs posix=False so it does not eat backslashes;
    # simple commands work either way.
    return shlex.split(command_str, posix=(sys.platform != "win32"))
