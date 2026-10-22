# agent_loop_demo_for_kubernetes

> Diagnosing unhealthy Kubernetes Pods usually means manually chaining `kubectl get pods`, `describe`, `logs`, and `get events` until the root cause shows up. This project automates that investigation with a hand-written Agent Loop — no agent framework, just the low-level `openai` and `mcp` SDKs — so an LLM calls Kubernetes tools over MCP, digs in over multiple rounds on its own, and produces a structured cluster health report.

## 📦 Core Features

- **Hand-written Agent Loop, zero framework** — no LangChain / AutoGen. The core loop in `run_agent()` (`agent.py`) is a single `for` loop: send `messages` + tools to the LLM, execute any `tool_calls` via MCP, append results as `role="tool"` messages, and repeat until the model answers in plain text.
- **Standard MCP tool calling** — `mcp_client.py` launches the MCP Server as a stdio subprocess, converts MCP tool definitions into OpenAI's `tools` format (`inputSchema` → `parameters`), and flattens tool results into strings. The tool list is fetched from the server at runtime, so the loop never hard-codes tool names.
- **Runs without a real cluster** — the bundled Mock K8s MCP Server (`mock_k8s_server.py`, FastMCP) serves a fake cluster (3 nodes, 4 namespaces, 12 Pods) with deliberately planted anomalies: CrashLoopBackOff, ImagePullBackOff, and Pending Pods.
- **Multi-round autonomous investigation** — the agent typically starts with `list_pods` + `list_nodes` for an overview, then digs into unhealthy Pods with `describe_pod` / `get_pod_logs` / `get_events`, citing tool evidence for every conclusion in the final report.
- **Built-in guardrails** — if the model has not converged after `MAX_STEPS` (default 10), tools are withheld and it is forced to summarize from what it has, so the loop can never spin forever burning tokens. Invalid tool-argument JSON and tool failures are fed back to the model as tool results instead of crashing the run.
- **One-line switch to a real cluster** — point `MCP_SERVER_COMMAND` in `.env` at a real Kubernetes MCP Server; `agent.py` does not change (see [Switching to a Real Cluster](#switching-to-a-real-cluster)).
- **Testable without an API key** — the pytest suite covers the MCP client against the mock server and the full agent loop against a fake LLM, so everything except the real model call is verifiable offline.
- **Clean terminal logging** — all output goes through Python's `logging` module (no bare `print`), timestamped and tagged per step, tool call, and result preview. Set `LOG_LEVEL=DEBUG` for verbose output or `LOG_LEVEL=WARNING` for a quiet run.

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (developed and tested on Python 3.13)
- **An LLM API key** for any OpenAI-compatible endpoint — OpenAI, DeepSeek, Qwen, a local Ollama server, etc. The model should support tool calling (e.g. `gpt-4o-mini`, `deepseek-chat`, `qwen-plus`)
- *(Optional, only for a real cluster)* **Node.js** (to run a real K8s MCP Server via `npx`), plus a working `kubectl` / kubeconfig

### Installation

```bash
git clone https://github.com/rhuab/agent_loop_demo_for_kubernetes.git
cd agent_loop_demo_for_kubernetes

# Install dependencies (note: mcp must be <2; 2.x renamed APIs such as FastMCP)
pip install -r requirements.txt

# Configure the LLM
cp .env.example .env        # Windows (PowerShell): Copy-Item .env.example .env
# Edit .env and fill in OPENAI_API_KEY, OPENAI_BASE_URL, and MODEL

# (Optional) Run the test suite — verifies the MCP path and the agent loop
# without any real LLM API key
python -m pytest -v

# Run the agent
python agent.py
```

## 📖 Usage

Ask a question directly, or run with no arguments to use the default full-cluster health check:

```bash
python agent.py
python agent.py "Which pods in the production namespace are unhealthy, and why?"
python agent.py "Check the pod status of this cluster"
```

### Example: Input and Output

**Input** — ask about one namespace:

```powershell
python agent.py "Which pods in the production namespace are unhealthy, and why?"
```

**Output** — excerpt from an actual run (model: `gpt-5.5`, MCP server: the bundled `mock_k8s_server.py`). Some intermediate tool calls, all `[result]` previews, and HTTP logs are omitted for brevity; in the full log every `[tool]` line is followed by a truncated result preview:

```text
18:56:21 INFO    __main__ | Model: gpt-5.5 | MCP server: ...\mock_k8s_server.py
18:56:21 INFO    __main__ | Question: Which pods in the production namespace are unhealthy, and why?
18:56:21 INFO    __main__ | Connected to MCP server, available tools: ['list_namespaces', 'list_pods', 'describe_pod', 'get_pod_logs', 'list_nodes', 'get_events']
18:56:21 INFO    __main__ | --- Step 1: requesting LLM ---
18:56:24 INFO    __main__ | [tool] list_pods({"namespace": "all"})
18:56:24 INFO    __main__ | [tool] list_nodes({})
18:56:24 INFO    __main__ | --- Step 2: requesting LLM ---
18:56:29 INFO    __main__ | [tool] describe_pod({"namespace": "production", "name": "payment-service-6f7b8d9c0-ghi56"})
18:56:29 INFO    __main__ | [tool] get_pod_logs({"namespace": "production", "name": "payment-service-6f7b8d9c0-ghi56", "tail": 50})
18:56:29 INFO    __main__ | [tool] describe_pod({"namespace": "production", "name": "inventory-service-59d8f7c6a-mno90"})
18:56:29 INFO    __main__ | [tool] get_events({"namespace": "production"})
18:56:29 INFO    __main__ | [tool] list_namespaces({})
18:56:29 INFO    __main__ | --- Step 3: requesting LLM ---
18:56:43 INFO    __main__ | ===== Final Report =====
## Cluster overview

- **Namespaces:** 4 total, **4 Active** (`list_namespaces`)
- **Nodes:** 3 total, **3 Ready** (`list_nodes`) — `node-03` is at CPU 91% / memory 93%
- **Pods:** 12 total, **8 Running and Ready** (`list_pods namespace=all`); production: 4 pods, 2 unhealthy

## Unhealthy production pods

| Pod | Status | Evidence / Root cause | Recommended remediation |
|---|---|---|---|
| `payment-service-6f7b8d9c0-ghi56` | `CrashLoopBackOff`, `0/1` ready, 14 restarts | Logs: `missing required field 'DB_PASSWORD'`, `failed to connect to database: empty password`, `FATAL exiting with code 1` | Restore the missing `DB_PASSWORD` Secret/ConfigMap entry, then roll out the deployment |
| `inventory-service-59d8f7c6a-mno90` | `ImagePullBackOff`, `0/1` ready | Events: image `myregistry/inventory-service:v0.9.9-beta` — `manifest unknown`, `ErrImagePull` | Push the missing image or correct the tag, and verify registry credentials |

## Overall conclusion

Production has **2 unhealthy pods**: `payment-service-6f7b8d9c0-ghi56` due to missing `DB_PASSWORD`, and `inventory-service-59d8f7c6a-mno90` due to a missing/unpullable image tag, based on `list_pods`, `describe_pod`, `get_pod_logs`, and `get_events`.
```

**More input examples:**

```powershell
# Default: full-cluster health check (no argument)
python agent.py

# Focus on scheduling problems
python agent.py "Are there any Pending pods, and why can't they be scheduled?"

# Focus on a single workload
python agent.py "Why does payment-service in production keep restarting?"
```

For the Pending-pods question, expect the final answer to name `metrics-server` (kube-system, `FailedScheduling: Insufficient memory`) and `checkout-test` (staging, `FailedScheduling: Insufficient cpu`) and to point at node resource pressure (node-03 at 91% CPU / 93% memory, per `list_nodes`).

While running, the terminal prints each round — which tool the LLM called, its arguments, and a result preview — followed by the structured health report (cluster overview, anomaly list with root-cause evidence and remediation, and an overall conclusion). Against the bundled mock cluster, expect the agent to identify:

- `payment-service` (production): **CrashLoopBackOff**, restarted 14 times; logs show a missing `DB_PASSWORD` config;
- `inventory-service` (production): **ImagePullBackOff**; image `v0.9.9-beta` does not exist in the registry;
- `metrics-server` (kube-system) and `checkout-test` (staging): **Pending**; scheduling fails due to insufficient node resources (node-03 is at 93% memory).

The mock server exposes 6 tools, with names intentionally close to a real Kubernetes MCP server:

`list_namespaces`, `list_pods`, `describe_pod`, `get_pod_logs`, `list_nodes`, `get_events`

### How It Works

```text
User question
   |
   v
agent.py (hand-written loop) -- chat.completions + tools --> LLM (OpenAI-compatible API)
   |  ^                                                            |
   |  `------------------------- tool_calls <----------------------'
   | MCP call_tool (stdio / JSON-RPC)
   v
mock_k8s_server.py (FastMCP) -- returns fake pod / node / event / log data
```

| File | Purpose |
| --- | --- |
| `agent.py` | The core of the demo: hand-written agent loop, multi-round tool calling, step logging, forced summary |
| `mcp_client.py` | MCP client wrapper: stdio connection, tool format conversion, `call_tool` |
| `mock_k8s_server.py` | Mock MCP Server: 3 nodes, 4 namespaces, 12 Pods, including CrashLoopBackOff / ImagePullBackOff / Pending anomalies |
| `tests/` | pytest suite: `test_mcp_client.py` (MCP client vs. mock server) and `test_agent_loop.py` (full loop against a fake LLM, no API key needed) |
| `.env.example` | Config template: API key, model, MCP server launch command |

### Switching to a Real Cluster

The agent talks to the MCP server over the standard MCP protocol, so you only replace the server — `agent.py` does not change:

1. Prepare a real cluster and make sure `kubectl` works, e.g. spin up a local cluster with [kind](https://kind.sigs.k8s.io/):

   ```bash
   kind create cluster
   kubectl get nodes
   ```

2. In `.env`, point `MCP_SERVER_COMMAND` at a real K8s MCP Server (requires Node.js):

   ```ini
   MCP_SERVER_COMMAND=npx -y mcp-server-kubernetes
   ```

3. Run `python agent.py "Check the pod status of this cluster"` again. It will reach the real cluster through your kubeconfig.

> Note: a real MCP server may expose slightly different tool names (e.g. with prefixes). That does not affect the loop — the tool list is fetched from the server at runtime and converted to the OpenAI format, and the model picks the tools itself.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'mcp.server.fastmcp'`**: you installed mcp 2.x. Run `pip install "mcp>=1.0,<2"`, or rewrite the server against the 2.x API (`MCPServer`).
- **OPENAI_API_KEY not configured error**: copy `.env.example` to `.env` as instructed and fill in a key. With a local Ollama model the key can be any value, but it must not be empty.
- **The model answers without investigating, or calls the same tool repeatedly**: switch to a model with stronger tool calling (e.g. `deepseek-chat`, `qwen-plus`, `gpt-4o-mini`), or lower `MAX_STEPS` first and read the step log to find the problematic round.


## 📄 License

This project is open-sourced under the [MIT](LICENSE) License.
