
import logging
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agentic.tools.ticket_tools import (
    get_ticket_info,
    update_ticket_status,
    add_ticket_message,
    get_customer_ticket_history,
    get_user_preferences,
    update_user_preferences,
)
from agentic.tools.cultpass_tools import (
    get_cultpass_user_info,
    get_user_reservations,
    get_experience_availability,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ResolverState(TypedDict):
    """Full state carried through the resolver graph."""
    messages: Annotated[list, add_messages]

# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------

RESOLVER_TOOLS = [
    get_ticket_info,
    update_ticket_status,
    add_ticket_message,
    get_customer_ticket_history,
    get_user_preferences,
    update_user_preferences,
    get_cultpass_user_info,
    get_user_reservations,
    get_experience_availability,
]

RESOLVER_SYSTEM_PROMPT = """
You are the **Resolver Agent** for UDA-Hub.

Your goal is to RESOLVE customer support tickets by following these steps:

1. **Understand the issue**: Read the conversation context carefully, including the
   articles already retrieved and shared by the Retriever Agent.

2a. **Use the retrieved knowledge base articles**: The Retriever Agent has already
   searched the knowledge base and shared the relevant articles in the conversation context.
   Use those articles as your primary source of policy and procedure information.

2b. **Check interaction history** (if an external_user_id is available):
   - Call `get_customer_ticket_history` to see if this is a returning customer.
   - If they have prior tickets for the same issue_type, acknowledge it and avoid
     asking them to repeat information already captured (e.g. "I can see you
     contacted us about this before — let me look into what's changed").

2c. **Retrieve and apply customer preferences** (if an external_user_id is available):
   - Call `get_user_preferences` to check stored language, preferred channel, and notes.
   - Apply the preferred language when composing your reply.
   - If you discover new preference information during this interaction (e.g. the
     customer mentions they prefer email updates or speaks French), call
     `update_user_preferences` to persist it for future sessions.

3. **Fetch account context when needed**:
   - Use `get_cultpass_user_info` to check if the user's account is blocked or has an
     active subscription before advising on subscription-related issues.
   - Use `get_user_reservations` for reservation queries.
   - Use `get_experience_availability` when the user asks about specific events.

4. **Compose a clear, empathetic response**:
   - Address the user by feel (warm but professional).
   - Reference the knowledge base content where applicable.
   - Provide concrete next steps.
   - If the account is blocked, inform the user and explain they need to contact support.

5. **Save and finalise**:
   - Call `add_ticket_message` with role='agent' to persist your answer.
   - If resolved: call `update_ticket_status` with status='resolved'.

### Escalation rules — when you CANNOT resolve automatically

Escalate immediately (do NOT attempt a further reply) if ANY of the following apply:
  - The issue involves a billing dispute, charge reversal, or refund request.
  - The customer's account is blocked and requires manual unblocking.
  - No knowledge base article addresses the issue and DB context is insufficient.
  - The customer explicitly requests a human agent.
  - You have already attempted a resolution and the customer reports it did not work.

**When escalating:**
1. Call `update_ticket_status` with status='escalated'.
2. Your final message content MUST be exactly:
   NEEDS_ESCALATION
   (nothing before or after — this exact token is read by the Supervisor to route
   the ticket to the Escalation Agent.)

Always be concise, accurate, and helpful. Never make up policy information — only use
what you find in the knowledge base.
"""

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def llm_call(state: ResolverState, config: RunnableConfig) -> dict:
    """Invoke the LLM (with tools bound) and append its response."""
    logger.debug("llm_call node invoked. message count=%d", len(state["messages"]))

    llm = config.get("configurable", {}).get("llm", None)

    if llm is None:
        logger.error("No 'llm' found in configurable; falling back to default ChatOpenAI.")

    llm_with_tools = llm.bind_tools(RESOLVER_TOOLS)
    messages = [SystemMessage(content=RESOLVER_SYSTEM_PROMPT)] + state["messages"]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
        raise

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    logger.info("LLM response received. has_tool_calls=%s", has_tool_calls)
    return {"messages": [response]}

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _should_continue(state: ResolverState) -> str:
    """Route to tools, END normally, or END with NEEDS_ESCALATION signal."""
    last_msg = state["messages"][-1]

    # If the LLM signalled escalation, stop immediately so the supervisor
    # can read the NEEDS_ESCALATION content and route to the escalation agent.
    content = getattr(last_msg, "content", "") or ""
    if "NEEDS_ESCALATION" in content:
        logger.info("NEEDS_ESCALATION detected — ending resolver graph.")
        return END

    # Normal ReAct loop: continue if there are pending tool calls.
    if getattr(last_msg, "tool_calls", None):
        return "tools"

    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(ResolverState)
builder.add_node("llm_call", llm_call)
builder.add_node("tools", ToolNode(RESOLVER_TOOLS))

builder.set_entry_point("llm_call")
builder.add_conditional_edges("llm_call", _should_continue, ["tools", END])
builder.add_edge("tools", "llm_call")

resolver_agent = builder.compile()
resolver_agent.name = "resolver"
