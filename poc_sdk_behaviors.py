"""PoC: measure OpenAI Agents SDK behaviors relevant to migrating agent.py.

Verifies (against openai-agents as installed):
  1. What tool_choice looks like on request #2 after reset_tool_choice
  2. Invalid JSON args: raw FunctionTool vs @function_tool
  3. Unknown tool: default raise vs return_error_to_model; run_data presence
  4. tool_use_behavior sentinel + MaxTurnsExceeded run_data / to_input_list

Uses a local fake Chat Completions server (no real API key / network).
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

from openai import AsyncOpenAI

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    ToolsToFinalOutputResult,
    function_tool,
    set_tracing_disabled,
)
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from agents.tool import default_tool_error_function

set_tracing_disabled(True)

SENTINEL = "__FORCE_SUMMARY__"


# ---------------------------------------------------------------------------
# Fake LLM (Chat Completions compatible)
# ---------------------------------------------------------------------------

class _FakeHandler(BaseHTTPRequestHandler):
    script: Callable[[dict, int], dict] | None = None
    requests: list[dict] = []

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(body)
        n = len(type(self).requests)
        assert type(self).script is not None
        payload = type(self).script(body, n)
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _completion(*, content: str | None = None, tool_calls: list | None = None) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-poc",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def _tool_call(call_id: str, name: str, arguments) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def start_fake_llm():
    _FakeHandler.requests = []
    _FakeHandler.script = None
    server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, thread, f"http://127.0.0.1:{port}/v1"


def make_model(base_url: str) -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(api_key="poc-key", base_url=base_url)
    return OpenAIChatCompletionsModel(model="fake-model", openai_client=client)


def dump_run_data(exc: Exception) -> dict:
    rd = getattr(exc, "run_data", None)
    if rd is None:
        return {"run_data": None}
    items = []
    for it in rd.new_items or []:
        raw = getattr(it, "raw_item", None)
        items.append(
            {
                "type": type(it).__name__,
                "raw_type": type(raw).__name__ if raw is not None else None,
                "preview": str(raw)[:200] if raw is not None else str(it)[:200],
            }
        )
    return {
        "run_data": {
            "input_preview": str(rd.input)[:120],
            "new_items_count": len(rd.new_items or []),
            "new_items": items,
            "raw_responses_count": len(rd.raw_responses or []),
            "last_agent": getattr(rd.last_agent, "name", None),
        }
    }


def summarize_input_list(items: list) -> list[dict]:
    out = []
    for it in items:
        if isinstance(it, dict):
            role = it.get("role") or it.get("type")
            content = it.get("content") or it.get("output") or it.get("arguments")
            name = it.get("name")
            out.append(
                {
                    "kind": "dict",
                    "role_or_type": role,
                    "name": name,
                    "content_preview": str(content)[:160] if content is not None else None,
                }
            )
        else:
            out.append({"kind": type(it).__name__, "preview": str(it)[:160]})
    return out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

async def case1_tool_choice_reset(base_url: str) -> dict:
    """After a tool call, reset_tool_choice should clear tool_choice."""

    @function_tool
    def list_pods(namespace: str = "all") -> str:
        return f"pods in {namespace}: ok"

    _FakeHandler.script = lambda body, n: (
        _completion(tool_calls=[_tool_call("c1", "list_pods", {"namespace": "all"})])
        if n == 1
        else _completion(content="Report: cluster looks fine.")
    )
    _FakeHandler.requests = []

    agent = Agent(
        name="poc-tool-choice",
        instructions="Use tools then report.",
        model=make_model(base_url),
        tools=[list_pods],
        model_settings=ModelSettings(tool_choice="required"),
        reset_tool_choice=True,
    )
    result = await Runner.run(agent, "check cluster", max_turns=5)
    choices = [r.get("tool_choice", "<MISSING>") for r in _FakeHandler.requests]
    return {
        "case": "1_tool_choice_reset",
        "request_count": len(_FakeHandler.requests),
        "tool_choice_per_request": choices,
        "final_output": result.final_output,
        "note": (
            "Source resets tool_choice to None; Chat Completions converter omits the field "
            "(openai.Omit). Hand-written loop sends explicit 'auto'."
        ),
    }


async def case2_invalid_json(base_url: str) -> dict:
    """Compare @function_tool vs raw FunctionTool on bad JSON arguments."""

    invoked: list[str] = []

    @function_tool
    def decorated_echo(x: int) -> str:
        invoked.append(f"decorated:{x}")
        return f"echo {x}"

    async def raw_invoke(ctx, args_json: str) -> str:
        invoked.append(f"raw_seen:{args_json}")
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError as exc:
            return f"Tool arguments are not valid JSON: {exc}; raw arguments: {args_json}"
        return f"raw ok {args}"

    raw_tool = FunctionTool(
        name="raw_echo",
        description="raw tool that receives the JSON string",
        params_json_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        on_invoke_tool=raw_invoke,
        strict_json_schema=False,
    )

    # Subcase A: decorated tool gets invalid JSON
    _FakeHandler.requests = []
    call_n = {"n": 0}

    def script_decorated(body, n):
        call_n["n"] = n
        if n == 1:
            return _completion(tool_calls=[_tool_call("bad1", "decorated_echo", "{not-json")])
        # After tool result, finish
        return _completion(content="decorated path finished")

    _FakeHandler.script = script_decorated
    agent_a = Agent(
        name="poc-json-decorated",
        instructions="Call the tool.",
        model=make_model(base_url),
        tools=[decorated_echo],
        model_settings=ModelSettings(tool_choice="required"),
    )
    decorated_result: dict[str, Any] = {"invoked": None, "error": None, "final": None, "tool_msgs": []}
    try:
        r = await Runner.run(agent_a, "go", max_turns=4)
        decorated_result["final"] = r.final_output
        decorated_result["input_list"] = summarize_input_list(r.to_input_list())
    except Exception as exc:
        decorated_result["error"] = f"{type(exc).__name__}: {exc}"
        decorated_result["run_data"] = dump_run_data(exc)
    decorated_result["invoked"] = list(invoked)
    decorated_result["default_failure_msg"] = default_tool_error_function(None, Exception("x"))  # type: ignore[arg-type]

    # Subcase B: raw FunctionTool gets invalid JSON (handles itself)
    invoked.clear()
    _FakeHandler.requests = []

    def script_raw(body, n):
        if n == 1:
            return _completion(tool_calls=[_tool_call("bad2", "raw_echo", "{not-json")])
        return _completion(content="raw path finished")

    _FakeHandler.script = script_raw
    agent_b = Agent(
        name="poc-json-raw",
        instructions="Call the tool.",
        model=make_model(base_url),
        tools=[raw_tool],
        model_settings=ModelSettings(tool_choice="required"),
    )
    raw_result: dict[str, Any] = {"invoked": None, "error": None, "final": None}
    try:
        r = await Runner.run(agent_b, "go", max_turns=4)
        raw_result["final"] = r.final_output
        raw_result["input_list"] = summarize_input_list(r.to_input_list())
    except Exception as exc:
        raw_result["error"] = f"{type(exc).__name__}: {exc}"
        raw_result["run_data"] = dump_run_data(exc)
    raw_result["invoked"] = list(invoked)

    return {
        "case": "2_invalid_json",
        "decorated_function_tool": decorated_result,
        "raw_function_tool": raw_result,
    }


async def case3_unknown_tool(base_url: str) -> dict:
    """Unknown tool: default raise vs return_error_to_model."""

    @function_tool
    def list_pods(namespace: str = "all") -> str:
        return "ok"

    def script_unknown(body, n):
        if n == 1:
            return _completion(tool_calls=[_tool_call("u1", "delete_pod", {"name": "x"})])
        return _completion(content="recovered after unknown tool")

    # Default: raise_error
    _FakeHandler.requests = []
    _FakeHandler.script = script_unknown
    agent = Agent(
        name="poc-unknown",
        instructions="Investigate.",
        model=make_model(base_url),
        tools=[list_pods],
        model_settings=ModelSettings(tool_choice="required"),
    )
    raise_path: dict[str, Any] = {}
    try:
        await Runner.run(agent, "go", max_turns=4)
        raise_path["error"] = None
    except ModelBehaviorError as exc:
        raise_path["error"] = f"{type(exc).__name__}: {exc}"
        raise_path["run_data"] = dump_run_data(exc)
    except Exception as exc:
        raise_path["error"] = f"{type(exc).__name__}: {exc}"
        raise_path["run_data"] = dump_run_data(exc)

    # Soft: return_error_to_model + custom formatter (whitelist style)
    _FakeHandler.requests = []
    _FakeHandler.script = script_unknown

    def formatter(args):
        if args.kind == "tool_not_found":
            return (
                f"Unknown tool '{args.tool_name}'. "
                f"Available tools: list_pods. {args.default_message}"
            )
        return None

    soft_path: dict[str, Any] = {}
    try:
        r = await Runner.run(
            agent,
            "go",
            max_turns=4,
            run_config=RunConfig(
                tool_not_found_behavior="return_error_to_model",
                tool_error_formatter=formatter,
            ),
        )
        soft_path["final"] = r.final_output
        soft_path["input_list"] = summarize_input_list(r.to_input_list())
        soft_path["error"] = None
    except Exception as exc:
        soft_path["error"] = f"{type(exc).__name__}: {exc}"
        soft_path["run_data"] = dump_run_data(exc)

    return {
        "case": "3_unknown_tool",
        "default_raise_error": raise_path,
        "return_error_to_model": soft_path,
    }


async def case4_sentinel_and_max_turns(base_url: str) -> dict:
    """tool_use_behavior sentinel preserves history; MaxTurnsExceeded has run_data."""

    call_count = {"n": 0}

    @function_tool
    def list_pods(namespace: str = "all") -> str:
        call_count["n"] += 1
        return f"result-{call_count['n']}"

    async def stop_after_one(ctx, tool_results):
        # After any tool round, treat as final with sentinel
        return ToolsToFinalOutputResult(is_final_output=True, final_output=SENTINEL)

    def script_always_tool(body, n):
        # Keep calling the same tool forever (until max_turns / sentinel)
        return _completion(
            tool_calls=[_tool_call(f"c{n}", "list_pods", {"namespace": "all"})]
        )

    # 4a: sentinel via tool_use_behavior
    _FakeHandler.requests = []
    _FakeHandler.script = script_always_tool
    call_count["n"] = 0
    agent_sentinel = Agent(
        name="poc-sentinel",
        instructions="Keep investigating.",
        model=make_model(base_url),
        tools=[list_pods],
        model_settings=ModelSettings(tool_choice="required"),
        tool_use_behavior=stop_after_one,
        reset_tool_choice=True,
    )
    sentinel_path: dict[str, Any] = {}
    r = await Runner.run(agent_sentinel, "go", max_turns=5)
    sentinel_path["final_output"] = r.final_output
    sentinel_path["tool_calls_executed"] = call_count["n"]
    sentinel_path["llm_requests"] = len(_FakeHandler.requests)
    sentinel_path["input_list"] = summarize_input_list(r.to_input_list())
    sentinel_path["has_tool_output_in_history"] = any(
        (isinstance(it, dict) and (it.get("type") == "function_call_output" or it.get("role") == "tool"))
        or "function_call_output" in str(it)
        or "ToolCallOutput" in type(it).__name__
        for it in r.to_input_list()
    )

    # 4b: MaxTurnsExceeded
    _FakeHandler.requests = []
    _FakeHandler.script = script_always_tool
    call_count["n"] = 0
    agent_max = Agent(
        name="poc-max-turns",
        instructions="Keep investigating.",
        model=make_model(base_url),
        tools=[list_pods],
        model_settings=ModelSettings(tool_choice="required"),
        tool_use_behavior="run_llm_again",
        reset_tool_choice=True,
    )
    max_path: dict[str, Any] = {}
    try:
        await Runner.run(agent_max, "go", max_turns=2)
        max_path["error"] = None
    except MaxTurnsExceeded as exc:
        max_path["error"] = f"{type(exc).__name__}: {exc}"
        max_path["run_data"] = dump_run_data(exc)
        max_path["tool_calls_executed"] = call_count["n"]
        max_path["llm_requests"] = len(_FakeHandler.requests)
    except Exception as exc:
        max_path["error"] = f"{type(exc).__name__}: {exc}"
        max_path["run_data"] = dump_run_data(exc)

    return {
        "case": "4_sentinel_and_max_turns",
        "tool_use_behavior_sentinel": sentinel_path,
        "max_turns_exceeded": max_path,
    }


async def main() -> None:
    import importlib.metadata as m

    versions = {
        "openai-agents": m.version("openai-agents"),
        "openai": m.version("openai"),
        "mcp": m.version("mcp"),
    }
    server, thread, base_url = start_fake_llm()
    report = {"versions": versions, "results": []}
    try:
        for coro in (
            case1_tool_choice_reset(base_url),
            case2_invalid_json(base_url),
            case3_unknown_tool(base_url),
            case4_sentinel_and_max_turns(base_url),
        ):
            report["results"].append(await coro)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
