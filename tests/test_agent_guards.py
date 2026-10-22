"""Guard-rail tests for the agent loop in agent.py.

A scripted fake LLM plays a "misbehaving model" in each test, so --
without any real API key -- we can verify that the loop defends itself:

  1. step 1 sends tool_choice="required" (later steps use "auto");
  2. an answer given without any tool call is pushed back;
  3. schema-invalid tool arguments are rejected before any MCP call;
  4. duplicate (tool, args) calls are served from cache, and repeated
     duplicates eventually force a summary;
  5. a report naming pods absent from tool results is pushed back;
  6. unknown (hallucinated) tool names are rejected with a whitelist.

The MCP side is real: calls go through mcp_client to the bundled
mock_k8s_server.py, exactly as in test_agent_loop.py.
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


def _tool_contents(body):
    return [m.get("content", "") for m in body["messages"] if m.get("role") == "tool"]


def _user_contents(body):
    return [m.get("content", "") for m in body["messages"] if m.get("role") == "user"]


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Fake LLM whose behaviour each test provides as `script(body, n)`."""

    script = None
    requests: list = []

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        payload = type(self).script(body, len(type(self).requests))
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_llm(monkeypatch):
    _ScriptedHandler.requests = []
    _ScriptedHandler.script = None
    server = HTTPServer(("127.0.0.1", 0), _ScriptedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("MODEL", "fake-model")
    monkeypatch.delenv("MCP_SERVER_COMMAND", raising=False)
    yield _ScriptedHandler
    server.shutdown()
    thread.join(timeout=5)


def _run(question="Check the cluster status"):
    return asyncio.run(agent.run_agent(question))


# 1. Step 1 must force tool use; later steps must not. ----------------------

def test_first_step_requires_tools(fake_llm):
    def script(body, n):
        if n == 1:
            return _completion(tool_calls=[_tool_call("c1", "list_pods", {"namespace": "all"})])
        return _completion(content="Report: payment-service-6f7b8d9c0-ghi56 is CrashLoopBackOff.")

    fake_llm.script = script
    answer = _run()

    assert "CrashLoopBackOff" in answer
    assert fake_llm.requests[0].get("tool_choice") == "required"
    assert fake_llm.requests[1].get("tool_choice") == "auto"


# 2. Answering without any investigation is pushed back. -------------------

def test_no_tool_answer_is_pushed_back(fake_llm):
    def script(body, n):
        if n == 1:  # model ignores tool_choice and answers from thin air
            return _completion(content="Everything looks fine.")
        if n == 2:
            # The pushback must be visible to the model as a user message.
            assert any("have not called any tool yet" in c for c in _user_contents(body))
            return _completion(tool_calls=[_tool_call("c1", "list_pods", {"namespace": "all"})])
        return _completion(content="Report: payment-service-6f7b8d9c0-ghi56 is CrashLoopBackOff.")

    fake_llm.script = script
    answer = _run()

    assert "CrashLoopBackOff" in answer
    assert len(fake_llm.requests) == 3


# 3. Schema-invalid arguments never reach the MCP server. ------------------

def test_invalid_arguments_are_rejected(fake_llm):
    def script(body, n):
        if n == 1:  # missing required "namespace", "tail" has wrong type
            return _completion(tool_calls=[
                _tool_call("c1", "get_pod_logs", {"name": "x", "tail": "fifty"})
            ])
        joined = " ".join(_tool_contents(body))
        assert "Invalid arguments for get_pod_logs" in joined
        assert "namespace" in joined
        return _completion(content="The invalid call was rejected before reaching the server.")

    fake_llm.script = script
    answer = _run()

    assert "rejected" in answer
    assert len(fake_llm.requests) == 2


# 4. Duplicate calls hit the cache; too many force a summary. --------------

def test_duplicate_calls_force_summary(fake_llm):
    def script(body, n):
        if not body.get("tools"):  # forced-summary request carries no tools
            joined = " ".join(_tool_contents(body))
            assert "[Duplicate call]" in joined
            return _completion(content="Forced summary: investigation stalled on duplicates.")
        return _completion(tool_calls=[
            _tool_call(f"c{n}", "list_pods", {"namespace": "all"})
        ])

    fake_llm.script = script
    answer = _run()

    assert answer.startswith("Forced summary")
    # 1 real call + MAX_REPEATS duplicate rounds + 1 forced-summary request
    assert len(fake_llm.requests) == 1 + agent.MAX_REPEATS + 1


# 5. Invented pods in the report are pushed back. --------------------------

def test_invented_pod_is_pushed_back(fake_llm):
    def script(body, n):
        if n == 1:
            return _completion(tool_calls=[_tool_call("c1", "list_pods", {"namespace": "all"})])
        if n == 2:
            return _completion(
                content="Report: ghost-app-12345-abcde and payment-service-6f7b8d9c0-ghi56 are broken."
            )
        # The pushback must name the invented pod.
        assert any("ghost-app-12345-abcde" in c for c in _user_contents(body))
        return _completion(content="Report: payment-service-6f7b8d9c0-ghi56 is CrashLoopBackOff.")

    fake_llm.script = script
    answer = _run()

    assert "ghost-app" not in answer
    assert "CrashLoopBackOff" in answer
    assert len(fake_llm.requests) == 3


# 6. Hallucinated tool names are rejected with the whitelist. --------------

def test_unknown_tool_is_rejected(fake_llm):
    def script(body, n):
        if n == 1:
            return _completion(tool_calls=[_tool_call("c1", "delete_pod", {"name": "x"})])
        joined = " ".join(_tool_contents(body))
        assert "Unknown tool 'delete_pod'" in joined
        assert "list_pods" in joined  # the whitelist is offered back
        return _completion(content="That tool does not exist; sticking to the whitelist.")

    fake_llm.script = script
    answer = _run()

    assert "whitelist" in answer
    assert len(fake_llm.requests) == 2
