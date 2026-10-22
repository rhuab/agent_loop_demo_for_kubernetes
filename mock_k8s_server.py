"""Mock Kubernetes MCP Server (stdio).

Serves an MCP Server via FastMCP from the official mcp SDK, backed by a
static fake cluster dataset that deliberately includes anomalies such as
CrashLoopBackOff / ImagePullBackOff / Pending, so you can watch the agent
investigate them over multiple tool-calling rounds.

Tool names intentionally mirror a real k8s MCP server (e.g.
Flux159/mcp-server-kubernetes), so switching later only means pointing
agent.py's MCP_SERVER_COMMAND at the real server.

Running it standalone for debugging (it takes over stdio; do not type into it):
    python mock_k8s_server.py
In normal use it is launched as a subprocess by mcp_client.py.
"""

import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mock-k8s")

# ---------------------------------------------------------------------------
# Fake cluster data
# ---------------------------------------------------------------------------

NAMESPACES = [
    {"name": "default", "status": "Active", "age": "30d"},
    {"name": "production", "status": "Active", "age": "30d"},
    {"name": "kube-system", "status": "Active", "age": "30d"},
    {"name": "staging", "status": "Active", "age": "12d"},
]

NODES = [
    {
        "name": "node-01",
        "status": "Ready",
        "roles": "control-plane,worker",
        "version": "v1.30.2",
        "cpu": "4",
        "cpu_usage": "62%",
        "memory": "16Gi",
        "memory_usage": "71%",
        "age": "30d",
    },
    {
        "name": "node-02",
        "status": "Ready",
        "roles": "worker",
        "version": "v1.30.2",
        "cpu": "4",
        "cpu_usage": "48%",
        "memory": "16Gi",
        "memory_usage": "55%",
        "age": "30d",
    },
    {
        "name": "node-03",
        "status": "Ready",
        "roles": "worker",
        "version": "v1.30.2",
        "cpu": "2",
        "cpu_usage": "91%",
        "memory": "8Gi",
        "memory_usage": "93%",
        "age": "28d",
    },
]

# Notes on "status":
#   Running / CrashLoopBackOff / ImagePullBackOff / Pending mirror the
#   STATUS column of `kubectl get pods`.
PODS = [
    # --- default ------------------------------------------------------------
    {
        "name": "web-frontend-7d9f8b6c5-abc12",
        "namespace": "default",
        "status": "Running",
        "ready": "1/1",
        "restarts": 0,
        "node": "node-01",
        "ip": "10.244.1.12",
        "age": "5d",
        "containers": [
            {"name": "nginx", "image": "nginx:1.27", "ready": True,
             "state": "running", "restarts": 0},
        ],
        "labels": {"app": "web-frontend"},
    },
    {
        "name": "api-server-5c8d7f9b4-def34",
        "namespace": "default",
        "status": "Running",
        "ready": "1/1",
        "restarts": 1,
        "node": "node-02",
        "ip": "10.244.2.8",
        "age": "5d",
        "containers": [
            {"name": "api", "image": "myregistry/api-server:v2.3.1", "ready": True,
             "state": "running", "restarts": 1},
        ],
        "labels": {"app": "api-server"},
    },
    {
        "name": "redis-cache-0",
        "namespace": "default",
        "status": "Running",
        "ready": "1/1",
        "restarts": 0,
        "node": "node-02",
        "ip": "10.244.2.15",
        "age": "10d",
        "containers": [
            {"name": "redis", "image": "redis:7.2", "ready": True,
             "state": "running", "restarts": 0},
        ],
        "labels": {"app": "redis-cache"},
    },
    # --- production ---------------------------------------------------------
    {
        "name": "payment-service-6f7b8d9c0-ghi56",
        "namespace": "production",
        "status": "CrashLoopBackOff",
        "ready": "0/1",
        "restarts": 14,
        "node": "node-01",
        "ip": "10.244.1.20",
        "age": "2d",
        "containers": [
            {"name": "payment", "image": "myregistry/payment-service:v1.8.0",
             "ready": False, "state": "waiting (CrashLoopBackOff)", "restarts": 14},
        ],
        "labels": {"app": "payment-service"},
    },
    {
        "name": "order-service-84f6c5d7b-jkl78",
        "namespace": "production",
        "status": "Running",
        "ready": "2/2",
        "restarts": 0,
        "node": "node-02",
        "ip": "10.244.2.22",
        "age": "7d",
        "containers": [
            {"name": "order", "image": "myregistry/order-service:v3.1.0",
             "ready": True, "state": "running", "restarts": 0},
            {"name": "sidecar-proxy", "image": "envoyproxy/envoy:v1.31",
             "ready": True, "state": "running", "restarts": 0},
        ],
        "labels": {"app": "order-service"},
    },
    {
        "name": "inventory-service-59d8f7c6a-mno90",
        "namespace": "production",
        "status": "ImagePullBackOff",
        "ready": "0/1",
        "restarts": 0,
        "node": "node-03",
        "ip": "10.244.3.9",
        "age": "3h",
        "containers": [
            {"name": "inventory", "image": "myregistry/inventory-service:v0.9.9-beta",
             "ready": False, "state": "waiting (ImagePullBackOff)", "restarts": 0},
        ],
        "labels": {"app": "inventory-service"},
    },
    {
        "name": "user-service-7b6c5d8f9-pqr12",
        "namespace": "production",
        "status": "Running",
        "ready": "1/1",
        "restarts": 3,
        "node": "node-03",
        "ip": "10.244.3.11",
        "age": "15d",
        "containers": [
            {"name": "user", "image": "myregistry/user-service:v2.0.4",
             "ready": True, "state": "running", "restarts": 3},
        ],
        "labels": {"app": "user-service"},
    },
    # --- kube-system --------------------------------------------------------
    {
        "name": "coredns-5dd5756b68-stu34",
        "namespace": "kube-system",
        "status": "Running",
        "ready": "1/1",
        "restarts": 0,
        "node": "node-01",
        "ip": "10.244.1.5",
        "age": "30d",
        "containers": [
            {"name": "coredns", "image": "coredns/coredns:1.11.1", "ready": True,
             "state": "running", "restarts": 0},
        ],
        "labels": {"k8s-app": "kube-dns"},
    },
    {
        "name": "kube-proxy-vwx56",
        "namespace": "kube-system",
        "status": "Running",
        "ready": "1/1",
        "restarts": 0,
        "node": "node-02",
        "ip": "10.244.2.3",
        "age": "30d",
        "containers": [
            {"name": "kube-proxy", "image": "registry.k8s.io/kube-proxy:v1.30.2",
             "ready": True, "state": "running", "restarts": 0},
        ],
        "labels": {"k8s-app": "kube-proxy"},
    },
    {
        "name": "metrics-server-6d7f8b9c0-yzab1",
        "namespace": "kube-system",
        "status": "Pending",
        "ready": "0/1",
        "restarts": 0,
        "node": "<none>",
        "ip": "<none>",
        "age": "45m",
        "containers": [
            {"name": "metrics-server", "image": "registry.k8s.io/metrics-server/metrics-server:v0.7.1",
             "ready": False, "state": "waiting (ContainerCreating / unscheduled)", "restarts": 0},
        ],
        "labels": {"k8s-app": "metrics-server"},
    },
    # --- staging ------------------------------------------------------------
    {
        "name": "checkout-test-8c9d0e1f2-cdef3",
        "namespace": "staging",
        "status": "Pending",
        "ready": "0/1",
        "restarts": 0,
        "node": "<none>",
        "ip": "<none>",
        "age": "20m",
        "containers": [
            {"name": "checkout", "image": "myregistry/checkout-service:pr-142",
             "ready": False, "state": "waiting (unscheduled)", "restarts": 0},
        ],
        "labels": {"app": "checkout-test"},
    },
    {
        "name": "staging-api-6a7b8c9d0-ghij4",
        "namespace": "staging",
        "status": "Running",
        "ready": "1/1",
        "restarts": 0,
        "node": "node-01",
        "ip": "10.244.1.31",
        "age": "1d",
        "containers": [
            {"name": "api", "image": "myregistry/api-server:v2.4.0-rc1",
             "ready": True, "state": "running", "restarts": 0},
        ],
        "labels": {"app": "staging-api"},
    },
]

EVENTS = [
    {"namespace": "production", "type": "Warning", "reason": "BackOff",
     "object": "pod/payment-service-6f7b8d9c0-ghi56",
     "message": "Back-off restarting failed container payment in pod payment-service-6f7b8d9c0-ghi56",
     "age": "8m"},
    {"namespace": "production", "type": "Warning", "reason": "Failed",
     "object": "pod/payment-service-6f7b8d9c0-ghi56",
     "message": "Error: failed to start container: application exited with code 1 (config validation failed)",
     "age": "9m"},
    {"namespace": "production", "type": "Warning", "reason": "FailedPullImage",
     "object": "pod/inventory-service-59d8f7c6a-mno90",
     "message": 'Failed to pull image "myregistry/inventory-service:v0.9.9-beta": rpc error: code = NotFound desc = manifest unknown',
     "age": "2h"},
    {"namespace": "production", "type": "Warning", "reason": "ErrImagePull",
     "object": "pod/inventory-service-59d8f7c6a-mno90",
     "message": 'Error: ErrImagePull for image "myregistry/inventory-service:v0.9.9-beta"',
     "age": "2h"},
    {"namespace": "kube-system", "type": "Warning", "reason": "FailedScheduling",
     "object": "pod/metrics-server-6d7f8b9c0-yzab1",
     "message": "0/3 nodes are available: 2 Insufficient memory, 1 node(s) had untolerated taint(s).",
     "age": "40m"},
    {"namespace": "staging", "type": "Warning", "reason": "FailedScheduling",
     "object": "pod/checkout-test-8c9d0e1f2-cdef3",
     "message": "0/3 nodes are available: 3 Insufficient cpu.",
     "age": "18m"},
    {"namespace": "default", "type": "Normal", "reason": "Pulled",
     "object": "pod/web-frontend-7d9f8b6c5-abc12",
     "message": 'Container image "nginx:1.27" already present on machine',
     "age": "5d"},
]

# Each unhealthy pod gets fake logs that reveal its problem;
# healthy pods fall back to generic startup logs.
POD_LOGS = {
    "payment-service-6f7b8d9c0-ghi56": (
        "2026-10-21T09:12:01Z INFO  payment-service v1.8.0 starting, env=production\n"
        "2026-10-21T09:12:01Z INFO  loading config from /etc/payment/config.yaml\n"
        "2026-10-21T09:12:02Z ERROR config validation failed: missing required field 'DB_PASSWORD'\n"
        "2026-10-21T09:12:02Z ERROR failed to connect to database: empty password\n"
        "2026-10-21T09:12:02Z FATAL exiting with code 1"
    ),
    "inventory-service-59d8f7c6a-mno90": (
        "(container has not started yet: image pull failed, no logs available)"
    ),
    "metrics-server-6d7f8b9c0-yzab1": (
        "(pod is Pending and has not been scheduled to a node, no logs available)"
    ),
    "checkout-test-8c9d0e1f2-cdef3": (
        "(pod is Pending and has not been scheduled to a node, no logs available)"
    ),
    "user-service-7b6c5d8f9-pqr12": (
        "2026-10-21T08:47:11Z INFO  user-service v2.0.4 started, listening on :8080\n"
        "2026-10-21T08:59:03Z WARN  slow query detected: SELECT * FROM sessions took 2.3s\n"
        "2026-10-21T09:05:44Z INFO  health check ok, 1240 req/min"
    ),
}


def _find_pod(name: str, namespace: str) -> dict | None:
    for pod in PODS:
        if pod["name"] == name and (namespace == "all" or pod["namespace"] == namespace):
            return pod
    return None


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_namespaces() -> str:
    """List all namespaces in the cluster with their status."""
    return _dump(NAMESPACES)


@mcp.tool()
def list_pods(namespace: str = "all") -> str:
    """List pods in a namespace (or all namespaces when namespace="all").

    Returns each pod's name, namespace, status, ready containers,
    restart count, node and age. Use this first to get an overview
    of cluster health, then investigate non-Running pods.
    """
    pods = [p for p in PODS if namespace == "all" or p["namespace"] == namespace]
    summary = [
        {
            "name": p["name"],
            "namespace": p["namespace"],
            "status": p["status"],
            "ready": p["ready"],
            "restarts": p["restarts"],
            "node": p["node"],
            "age": p["age"],
        }
        for p in pods
    ]
    return _dump({"namespace": namespace, "count": len(summary), "pods": summary})


@mcp.tool()
def describe_pod(name: str, namespace: str) -> str:
    """Describe a single pod in detail: containers, images, labels,
    conditions and the events related to this pod.
    Use this to investigate a pod that is not running. The response
    already includes this pod's events, so do not call get_events
    for the same pod; use get_pod_logs next if it is CrashLoopBackOff.
    """
    pod = _find_pod(name, namespace)
    if pod is None:
        return _dump({"error": f'pod "{name}" not found in namespace "{namespace}"'})
    related_events = [
        e for e in EVENTS
        if e["namespace"] == pod["namespace"] and name in e["object"]
    ]
    detail = {
        "name": pod["name"],
        "namespace": pod["namespace"],
        "status": pod["status"],
        "ready": pod["ready"],
        "restarts": pod["restarts"],
        "node": pod["node"],
        "pod_ip": pod["ip"],
        "age": pod["age"],
        "labels": pod["labels"],
        "containers": pod["containers"],
        "events": related_events,
    }
    return _dump(detail)


@mcp.tool()
def get_pod_logs(name: str, namespace: str, tail: int = 20) -> str:
    """Get recent logs of a pod. Use this for CrashLoopBackOff or
    otherwise failing pods to find the application-level error.
    Do not use this for ImagePullBackOff or Pending pods: their
    containers never started, so no logs exist; use describe_pod
    and get_events for those instead.
    """
    pod = _find_pod(name, namespace)
    if pod is None:
        return _dump({"error": f'pod "{name}" not found in namespace "{namespace}"'})
    logs = POD_LOGS.get(
        name,
        "2026-10-21T09:00:00Z INFO  started successfully, listening on :8080\n"
        "2026-10-21T09:10:00Z INFO  health check ok",
    )
    lines = logs.splitlines()
    return _dump({
        "pod": name,
        "namespace": pod["namespace"],
        "tail": tail,
        "logs": "\n".join(lines[-tail:]),
    })


@mcp.tool()
def list_nodes() -> str:
    """List all nodes in the cluster with status and CPU/memory usage.
    Use this to check whether Pending pods are caused by resource pressure.
    """
    return _dump(NODES)


@mcp.tool()
def get_events(namespace: str = "all") -> str:
    """List cluster events (especially Warning events) in a namespace,
    or all namespaces when namespace="all". Useful for finding
    scheduling failures and image pull errors across several pods.
    For a single pod's events, prefer describe_pod, which already
    includes them.
    """
    events = [e for e in EVENTS if namespace == "all" or e["namespace"] == namespace]
    return _dump({"namespace": namespace, "count": len(events), "events": events})


if __name__ == "__main__":
    # stdio transport: stdout is reserved for JSON-RPC, never log to it.
    mcp.run()
