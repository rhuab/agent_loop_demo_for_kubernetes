"""Tests for mcp_client.MCPClient against the mock K8s MCP server.

These replace the old `python mcp_client.py` self-test that used to live
inside mcp_client.py. They launch the real mock server as a subprocess
and talk to it over stdio, exactly like agent.py does -- only the LLM
is not involved.

Run with:
    pip install -r requirements.txt
    python -m pytest -v

The async client is driven with asyncio.run() inside each test, so no
pytest-asyncio plugin is needed.
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp_client import MCPClient, parse_command

ROOT = Path(__file__).resolve().parent.parent
SERVER_CMD = [sys.executable, str(ROOT / "mock_k8s_server.py")]

EXPECTED_TOOLS = {
    "list_namespaces",
    "list_pods",
    "describe_pod",
    "get_pod_logs",
    "list_nodes",
    "get_events",
}

CRASHLOOP_POD = "payment-service-6f7b8d9c0-ghi56"


def run(coro):
    return asyncio.run(coro)


async def with_client(fn):
    async with MCPClient(SERVER_CMD) as client:
        return await fn(client)


def test_parse_command_splits_arguments():
    assert parse_command("npx -y mcp-server-kubernetes") == [
        "npx",
        "-y",
        "mcp-server-kubernetes",
    ]
    assert parse_command("python mock_k8s_server.py") == [
        "python",
        "mock_k8s_server.py",
    ]


def test_empty_command_is_rejected():
    try:
        MCPClient([])
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("MCPClient([]) should raise ValueError")


def test_tools_are_converted_to_openai_format():
    async def scenario(client):
        tools = await client.list_tools_openai()
        names = await client.tool_names()
        return tools, names

    tools, names = run(with_client(scenario))

    assert set(names) == EXPECTED_TOOLS
    assert len(tools) == len(EXPECTED_TOOLS)
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] in EXPECTED_TOOLS
        assert fn["description"]  # non-empty, taken from the MCP tool
        # parameters must be the MCP inputSchema (a JSON Schema object)
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


def test_list_pods_returns_planted_anomalies():
    async def scenario(client):
        raw = await client.call_tool("list_pods", {"namespace": "all"})
        return json.loads(raw)

    data = run(with_client(scenario))

    assert data["count"] == 12
    by_name = {p["name"]: p for p in data["pods"]}
    assert by_name[CRASHLOOP_POD]["status"] == "CrashLoopBackOff"
    assert by_name[CRASHLOOP_POD]["restarts"] == 14
    assert by_name[CRASHLOOP_POD]["namespace"] == "production"
    assert (
        by_name["inventory-service-59d8f7c6a-mno90"]["status"]
        == "ImagePullBackOff"
    )
    statuses = {p["status"] for p in data["pods"]}
    assert "Pending" in statuses
    assert "Running" in statuses


def test_list_pods_filters_by_namespace():
    async def scenario(client):
        raw = await client.call_tool("list_pods", {"namespace": "production"})
        return json.loads(raw)

    data = run(with_client(scenario))

    assert data["count"] == 4
    assert {p["namespace"] for p in data["pods"]} == {"production"}


def test_describe_pod_and_logs_explain_the_crashloop():
    async def scenario(client):
        detail = json.loads(
            await client.call_tool(
                "describe_pod",
                {"name": CRASHLOOP_POD, "namespace": "production"},
            )
        )
        logs = json.loads(
            await client.call_tool(
                "get_pod_logs",
                {"name": CRASHLOOP_POD, "namespace": "production"},
            )
        )
        return detail, logs

    detail, logs = run(with_client(scenario))

    assert detail["status"] == "CrashLoopBackOff"
    assert detail["containers"][0]["restarts"] == 14
    assert any(e["reason"] == "BackOff" for e in detail["events"])
    # The fake logs must reveal the root cause the agent is expected to find
    assert "DB_PASSWORD" in logs["logs"]


def test_nodes_and_events_explain_pending_pods():
    async def scenario(client):
        nodes = json.loads(await client.call_tool("list_nodes", {}))
        events = json.loads(
            await client.call_tool("get_events", {"namespace": "kube-system"})
        )
        return nodes, events

    nodes, events = run(with_client(scenario))

    assert len(nodes) == 3
    assert all(n["status"] == "Ready" for n in nodes)
    assert any(
        e["reason"] == "FailedScheduling" for e in events["events"]
    )


def test_unknown_tool_returns_error_string_instead_of_raising():
    async def scenario(client):
        return await client.call_tool("no_such_tool", {})

    result = run(with_client(scenario))

    # call_tool must not raise: the error goes back to the LLM as text
    assert "error" in result.lower()


def test_describe_missing_pod_returns_error_json():
    async def scenario(client):
        raw = await client.call_tool(
            "describe_pod", {"name": "ghost-123", "namespace": "default"}
        )
        return json.loads(raw)

    data = run(with_client(scenario))
    assert "error" in data
