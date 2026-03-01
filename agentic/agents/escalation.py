
import logging
from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from agentic.tools.ticket_tools import (
    get_ticket_info,
    update_ticket_status,
    add_ticket_message,
)
from agentic.tools.cultpass_tools import get_cultpass_user_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class EscalationState(TypedDict):
    """Full state carried through the escalation graph."""
    messages: Annotated[list, add_messages]

# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------

ESCALATION_TOOLS = [
    get_ticket_info,
    update_ticket_status,
    add_ticket_message,
    get_cultpass_user_info
]

ESCALATION_SYSTEM_PROMPT = """
You are the **Escalation Agent** for UDA-Hub.

You are called when a ticket cannot be resolved automatically and requires human review.

Your responsibilities:

1. **Retrieve ticket details** with `get_ticket_info` to review the full context.

2. **Retrieve user profile** with `get_cultpass_user_info` if an external_user_id is
   available, to check for blocked status, subscription tier, or billing anomalies.

3. **Draft a structured escalation note** (for the human support lead) that includes:
   - Issue summary (1-2 sentences)
   - Root cause hypothesis
   - What was already attempted
   - Recommended action (e.g. manual refund review, account unblock, billing correction)
   - Urgency level

4. **Append the escalation note** to the ticket via `add_ticket_message` with role='system'.

5. **Write a customer-facing closing message** that:
   - Acknowledges their frustration / issue
   - Confirms a human agent will follow up within 24 hours (standard) or 4 hours (high urgency)
   - Provides a reference (the ticket_id)

6. **Persist** via `add_ticket_message` with role='ai' for the customer-facing response
   and `update_ticket_status` with status='escalated'.

Always be empathetic. The customer should feel heard and confident help is coming.
"""

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def llm_call(state: EscalationState, config: RunnableConfig) -> dict:
    """Invoke the LLM (with tools bound) and append its response."""
    logger.debug("llm_call node invoked. message count=%d", len(state["messages"]))

    llm = config.get("configurable", {}).get("llm", None)

    if llm is None:
        logger.error("No 'llm' found in configurable; falling back to default ChatOpenAI.")

    llm_with_tools = llm.bind_tools(ESCALATION_TOOLS)
    messages = [SystemMessage(content=ESCALATION_SYSTEM_PROMPT)] + state["messages"]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
        raise

    return {"messages": [response]}

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(EscalationState)
builder.add_node("llm_call", llm_call)
builder.add_node("tools", ToolNode(ESCALATION_TOOLS))

builder.set_entry_point("llm_call")
builder.add_conditional_edges("llm_call", tools_condition)
builder.add_edge("tools", "llm_call")

escalation_agent = builder.compile()
escalation_agent.name = "escalation"
