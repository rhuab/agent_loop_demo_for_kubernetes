"""Pytest bootstrap: make the repo root importable (agent, mcp_client, ...)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
