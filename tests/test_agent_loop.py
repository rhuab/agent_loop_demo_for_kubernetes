"""End-to-end test of the hand-written agent loop in agent.py.

A tiny fake OpenAI-compatible HTTP server plays the LLM: it returns
scripted tool_calls based on how many tool results it has seen, so the
test verifies -- without any real API key -- that the loop
  1. sends the MCP tools to the LLM,
  2. executes the requested tools through the real MCP client/server,
  3. feeds tool results back as role="tool" messages, and
  4. stops when the model answers without tool calls.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import agent


def _completion(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def _tool_call(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class _FakeLLMHandler(BaseHTTPRequestHandler):
    request_count = 0

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).request_count += 1

        # The OpenAI request must carry the MCP tools converted by mcp_client
        if type(self).request_count == 1:
            tool_names = {t["function"]["name"] for t in body.get("tools", [])}
            assert "list_pods" in tool_names and "list_nodes" in tool_names

        tool_messages = [m for m in body["messages"] if m.get("role") == "tool"]
        if len(tool_messages) == 0:
            payload = _completion(tool_calls=[
                _tool_call("c1", "list_pods", {"namespace": "all"}),
                _tool_call("c2", "list_nodes", {}),
            ])
        elif len(tool_messages) == 2:
            payload = _completion(tool_calls=[
                _tool_call("c3", "describe_pod", {
                    "name": "payment-service-6f7b8d9c0-ghi56",
                    "namespace": "production",
                }),
                _tool_call("c4", "get_pod_logs", {
                    "name": "payment-service-6f7b8d9c0-ghi56",
                    "namespace": "production",
                }),
            ])
        else:
            # Second-round tool results must have been fed back to the model
            joined = " ".join(m.get("content", "") for m in tool_messages)
            assert "CrashLoopBackOff" in joined
            assert "DB_PASSWORD" in joined
            payload = _completion(
                content="Final report: payment-service is in CrashLoopBackOff "
                        "because DB_PASSWORD is missing."
            )

        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_llm(monkeypatch):
    _FakeLLMHandler.request_count = 0
    server = HTTPServer(("127.0.0.1", 0), _FakeLLMHandler)  # port 0: any free port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("MODEL", "fake-model")
    monkeypatch.delenv("MCP_SERVER_COMMAND", raising=False)
    yield server
    server.shutdown()
    thread.join(timeout=5)


def test_agent_loop_multi_round_tool_calling(fake_llm):
    answer = asyncio.run(agent.run_agent("Check the cluster status"))

    assert "CrashLoopBackOff" in answer
    assert "DB_PASSWORD" in answer
    # overview round + deep-dive round + final answer round
    assert _FakeLLMHandler.request_count == 3
