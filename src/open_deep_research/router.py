"""Utility for routing tools for IACT mode."""

from __future__ import annotations

from typing import Any, Dict

from .state import AgentState
from .utils import ToolRegistry


def select_tools(state: AgentState, registry: ToolRegistry) -> Dict[str, Any]:
    """Return a toolbelt for the child task using registry search."""

    task = state.get("child_task", "") if isinstance(state, dict) else ""
    tools = registry.search(task, k=5, filters=None)
    return {"toolbelt": tools, "count": len(tools)}
