"""Hand-written Agent Loop: LLM + MCP tool calling, no agent framework.

The core is a single for loop (see run_agent):

    +--------------------------------------------------------+
    |  1. Send messages + tools to the LLM                   |
    |  2. Did the LLM return tool_calls?                     |
    |       No  -> that text is the final answer, exit loop  |
    |       Yes -> execute each tool via MCP, append the     |
    |              results as role="tool" messages, go to 1  |
    |  3. Still not done after MAX_STEPS -> force the model  |
    |     to summarize and exit                              |
    +--------------------------------------------------------+

Usage:
    python agent.py                                       # default question
    python agent.py "Which pods in production are unhealthy, and why?"
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp_client import MCPClient, parse_command

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure console logging. Level can be overridden via LOG_LEVEL."""
    load_dotenv()  # pick up LOG_LEVEL from .env as well
    # Windows consoles default to GBK; force UTF-8 to avoid garbled log output
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

SYSTEM_PROMPT = """You are a Kubernetes cluster operations assistant. You can only obtain cluster information through the provided tools; never invent pods, nodes, or events.

When investigating cluster health, follow this order:
1. Start with list_pods (namespace="all") and list_nodes for a global overview;
2. For pods that are not Running or have an abnormal restart count, dig deeper with describe_pod and get_pod_logs;
3. When needed, use get_events to check Warning events and confirm causes such as scheduling failures or image pull failures;
4. Once you have enough information, stop calling tools and output a structured health report in English.

The report must include: a cluster overview (counts of nodes/namespaces/pods and how many are healthy), a list of anomalies (pod, namespace, status, evidence for the root cause, recommended remediation), and a one-line overall conclusion. When stating a conclusion, cite which tool result it is based on."""

DEFAULT_QUESTION = "Give this Kubernetes cluster a full health check: find every unhealthy pod and analyze the root cause."

MAX_RESULT_PREVIEW = 400  # preview length for tool results in the log, to avoid flooding the terminal
MAX_REPEATS = 3  # identical (tool, args) calls tolerated before forcing a summary


def _preview(text: str, limit: int = MAX_RESULT_PREVIEW) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[:limit] + " ...[truncated]"


def _save_trace(messages: list, cfg: dict, question: str) -> None:
    """Save the full conversation of a run to traces/<timestamp>.jsonl.

    One JSON object per line: a leading "_meta" record, then every
    message (system / user / assistant with tool_calls / tool results),
    so a run can be replayed later to see what the model saw, called
    and concluded. Never raises: a failed trace must not fail the run.
    """
    try:
        traces_dir = Path(__file__).parent / "traces"
        traces_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:21]
        path = traces_dir / f"run-{stamp}.jsonl"
        meta = {
            "role": "_meta",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": cfg["model"],
            "question": question,
            "mcp_server": " ".join(cfg["mcp_command"]),
            "max_steps": cfg["max_steps"],
        }
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for msg in messages:
                if isinstance(msg, dict):
                    record = msg
                elif hasattr(msg, "model_dump"):
                    record = msg.model_dump()
                else:
                    record = {"role": "unknown", "content": str(msg)}
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        logger.info("Trace saved to %s", path)
    except Exception as exc:  # trace is best-effort diagnostics only
        logger.warning("Failed to save trace: %s", exc)


def _load_config() -> dict:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-your-key-here":
        logger.error(
            "OPENAI_API_KEY is not configured. Please run: "
            "Copy-Item .env.example .env (Windows) or "
            "cp .env.example .env (macOS/Linux), "
            "then edit .env and fill in your API key."
        )
        raise SystemExit(2)
    server_cmd = os.getenv("MCP_SERVER_COMMAND", "").strip()
    if server_cmd:
        command = parse_command(server_cmd)
    else:
        # Default: launch this repo's mock server; absolute path so any cwd works
        command = [sys.executable, str(Path(__file__).parent / "mock_k8s_server.py")]
    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENAI_BASE_URL", "").strip() or None,
        "model": os.getenv("MODEL", "gpt-4o-mini").strip(),
        "mcp_command": command,
        "max_steps": int(os.getenv("MAX_STEPS", "10")),
    }


async def run_agent(question: str) -> str:
    cfg = _load_config()
    llm = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    logger.info("Model: %s | MCP server: %s", cfg["model"], " ".join(cfg["mcp_command"]))
    logger.info("Question: %s", question)

    async with MCPClient(cfg["mcp_command"]) as mcp:
        tools = await mcp.list_tools_openai()
        logger.info("Connected to MCP server, available tools: %s", await mcp.tool_names())

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        seen: dict[str, str] = {}  # (tool, args) signature -> result, for duplicate detection
        repeat_count = 0
        tool_calls_made = 0

        # ------------------------- Agent Loop -------------------------
        for step in range(1, cfg["max_steps"] + 1):
            logger.info("--- Step %d: requesting LLM ---", step)
            resp = await llm.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                tools=tools,
                # Step 1 must investigate, not answer from thin air; later
                # steps use "auto" so the model is free to stop and report.
                tool_choice="required" if step == 1 else "auto",
            )
            msg = resp.choices[0].message
            # Append the assistant message as-is (it must keep tool_calls,
            # otherwise the following tool results have no matching call)
            messages.append(msg)

            if not msg.tool_calls:
                if tool_calls_made == 0 and step < cfg["max_steps"]:
                    # Model answered without investigating (can happen when
                    # a provider ignores tool_choice): push back once and
                    # let it retry instead of accepting an invented report.
                    logger.warning("Model answered without calling any tool; pushing back")
                    messages.append(
                        {
                            "role": "user",
                            "content": "You have not called any tool yet. Investigate the cluster with the provided tools before answering.",
                        }
                    )
                    continue
                answer = msg.content or ""
                logger.info("===== Final Report =====\n%s", answer)
                _save_trace(messages, cfg, question)
                return answer

            force_summary = False
            # Execute every tool call the model requested this step
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = f"Tool arguments are not valid JSON: {exc}; raw arguments: {tc.function.arguments}"
                    logger.warning("[tool] %s argument parsing failed: %s", name, exc)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
                    continue

                logger.info("[tool] %s(%s)", name, json.dumps(args, ensure_ascii=False))
                tool_calls_made += 1
                sig = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                if sig in seen:
                    repeat_count += 1
                    logger.warning(
                        "[repeat %d/%d] duplicate call %s, returning cached result",
                        repeat_count, MAX_REPEATS, sig,
                    )
                    result = (
                        "[Duplicate call] You already called this tool with identical "
                        "arguments; the result has not changed. Base your next step on "
                        "it, try a different tool, or write the final report.\n" + seen[sig]
                    )
                    if repeat_count >= MAX_REPEATS:
                        # Do not break mid-round: every tool_call in this
                        # assistant message still needs a matching tool
                        # message, or the next request would be rejected.
                        force_summary = True
                else:
                    result = await mcp.call_tool(name, args)
                    seen[sig] = result
                logger.info("[result] %s", _preview(result))
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
            if force_summary:
                logger.warning("Too many duplicate tool calls, forcing a summary...")
                break
        # ------------------------ End of loop -------------------------

        # Loop ended without converging (max steps reached or too many
        # duplicate calls): withhold tools and force the model to
        # summarize from what it already has, so we never loop forever.
        logger.warning("Forcing a summary (max_steps=%d, repeats=%d)...", cfg["max_steps"], repeat_count)
        messages.append(
            {
                "role": "user",
                "content": "You have reached the maximum number of investigation steps. Output the final health report directly from the tool results above; do not call any more tools.",
            }
        )
        resp = await llm.chat.completions.create(
            model=cfg["model"], messages=messages
        )
        answer = resp.choices[0].message.content or ""
        logger.info("===== Final Report (forced summary) =====\n%s", answer)
        _save_trace(messages, cfg, question)
        return answer


def main() -> None:
    setup_logging()
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    asyncio.run(run_agent(question))


if __name__ == "__main__":
    main()
