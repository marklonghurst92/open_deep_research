"""Nodes for IACT supervisor and child agents."""

from __future__ import annotations

from typing import Literal, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from .configuration import Configuration
from .prompts import planner_system_prompt, child_system_prompt
from .state import AgentState
from .utils import (
    get_all_tools,
    get_api_key_for_model,
    interpret,
)


async def supervisor_node(state: AgentState, config: RunnableConfig) -> Command[Literal["iact_router", "final_report_generation"]]:
    """IACT supervisor that manages call stack and planning."""

    configurable = Configuration.from_runnable_config(config)
    stack: List[dict] = state.get("call_stack", []) or []
    notes: List[str] = state.get("notes", [])

    # Initialize root frame if call stack empty
    if not stack:
        stack.append({"agent": "root", "task": state.get("research_brief", ""), "parent": None, "scratchpad": ""})

    # Handle result returned from child
    if state.get("result") is not None:
        frame = stack.pop()
        notes.append(f"{frame['task']}: {state['result']}")
        update = {"call_stack": stack, "notes": notes, "result": None}
        if stack:
            return Command(goto="iact_supervisor", update=update)
        else:
            return Command(goto="final_report_generation", update=update)

    # Escalation - ask user for help
    if state.get("escalate") and state.get("result") is not None:
        return Command(
            goto="final_report_generation",
            update={"ask_user": state["result"], "escalate": None, "result": None, "call_stack": stack, "notes": notes},
        )

    # Plan next step
    model = init_chat_model(
        model=configurable.planner_model,
        max_tokens=configurable.per_frame_token_budget,
        api_key=get_api_key_for_model(configurable.planner_model, config),
        tags=["langsmith:nostream"],
    )
    current = stack[-1]
    scratch = current.get("scratchpad", "")
    prompt = f"Task: {current['task']}\nScratchpad: {scratch}"
    response = await model.ainvoke([SystemMessage(content=planner_system_prompt), HumanMessage(content=prompt)])
    content = response.content.strip()
    if content.upper().startswith("SPAWN:"):
        rest = content.split(":", 1)[1].strip()
        child_type, child_task = (rest.split("|", 1) + [""])[:2]
        child_type = child_type.strip() or "child"
        child_task = child_task.strip()
        stack.append({"agent": child_type, "task": child_task, "parent": current["agent"], "scratchpad": ""})
        return Command(goto="iact_router", update={"call_stack": stack, "child_task": child_task, "notes": notes})
    else:
        step = content
        if step.upper().startswith("DO:"):
            step = step.split(":", 1)[1].strip()
        current["scratchpad"] = (scratch + "\n" + step).strip()
        return Command(goto="iact_supervisor", update={"call_stack": stack, "notes": notes})


async def child_node(state: AgentState, config: RunnableConfig) -> Command[Literal["iact_supervisor"]]:
    """IACT child agent that executes early actions and returns results."""

    configurable = Configuration.from_runnable_config(config)
    task = state.get("child_task", "")
    toolbelt_meta = state.get("toolbelt", [])
    available_tools = await get_all_tools(config)
    if toolbelt_meta:
        names = {t["name"] for t in toolbelt_meta}
        tools = [t for t in available_tools if t.name in names]
    else:
        tools = available_tools

    model = init_chat_model(
        model=configurable.iact_researcher_model,
        max_tokens=configurable.per_frame_token_budget,
        api_key=get_api_key_for_model(configurable.iact_researcher_model, config),
        tags=["langsmith:nostream"],
    ).bind_tools(tools)

    messages = [SystemMessage(content=child_system_prompt), HumanMessage(content=task)]
    for _ in range(configurable.max_tool_calls):
        response = await model.ainvoke(messages)
        messages.append(response)
        action = interpret(response.content, configurable.interpreter_style)
        if not action or action["action"] == "RETURN":
            result = action["content"] if action else response.content
            return Command(goto="iact_supervisor", update={"result": result, "child_task": None, "toolbelt": []})
        if action["action"] == "ASK PARENT":
            return Command(
                goto="iact_supervisor",
                update={"escalate": True, "result": action["content"], "child_task": None, "toolbelt": []},
            )
        if action["action"] == "SEARCH":
            tool = next((t for t in tools if t.name == "tavily_search"), None)
            if tool:
                tool_res = await tool.ainvoke({"queries": [action["content"]]}, config)
                messages.append(ToolMessage(content=tool_res, name="tavily_search", tool_call_id="search"))
            continue
        if action["action"] == "CODE":
            tool = next((t for t in tools if t.name == "think_tool"), None)
            if tool:
                tool_res = await tool.ainvoke({"reflection": action["content"]}, config)
                messages.append(ToolMessage(content=tool_res, name="think_tool", tool_call_id="code"))
            continue

    # Fallback if max tool calls exceeded without return
    return Command(goto="iact_supervisor", update={"result": "", "child_task": None, "toolbelt": []})
