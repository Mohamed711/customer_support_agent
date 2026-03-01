

import logging
from typing import Annotated, Literal, Optional

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition

from agentic.tools.ticket_tools import get_ticket_info, update_ticket_status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ClassificationOutput(BaseModel):
   """Structured classification produced by the agent."""

   issue_type: Literal[
      "login", "billing", "reservation", "subscription", "account", "general"
   ] = Field(description="Primary category of the support issue")

   urgency: Literal["high", "medium", "low"] = Field(
      description="Urgency level: high=blocked/payment failure, medium=functional degradation, low=informational."
   )

   sentiment: Literal["frustrated", "negative", "neutral", "positive"] = Field(
      description="Detected customer sentiment from their message."
   )

   summary: str = Field(
      description=(
         'One-line summary in the format: '
         '"CLASSIFIED: issue_type=<type>, urgency=<urgency>, sentiment=<sentiment>"'
      )
   )


class ClassifierState(TypedDict):
   """Full state carried through the classifier graph."""
   messages: Annotated[list, add_messages]
   classification: Optional[ClassificationOutput]

# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------

CLASSIFIER_TOOLS = [get_ticket_info, update_ticket_status]

CLASSIFIER_SYSTEM_PROMPT = """
You are the **Classifier Agent** for UDA-Hub, an intelligent customer-support platform.

Your job:
1. Receive the current conversation context (the customer's message and ticket info).
2. Determine the **issue_type** from one of:
   - login        : problems signing in, password reset, 2FA issues
   - billing      : payment failures, refund requests, invoices
   - reservation  : booking, cancellation, waitlist for experiences
   - subscription : plan management, upgrades, downgrades, pausing
   - account      : blocked accounts, profile issues, data requests
   - general      : anything else

3. Assess **urgency** (high / medium / low):
   - high   : blocked accounts, data loss, payment failures
   - medium : functional issues that degrade experience
   - low    : informational or minor questions

4. Detect **sentiment** (frustrated / negative / neutral / positive).

5. Use the `update_ticket_status` tool to persist:
   - status     : 'in_progress'
   - issue_type : as classified above
   - tags       : comma-separated tags including urgency and sentiment

6. Return a concise classification summary so the Supervisor can route correctly.
   Format: "CLASSIFIED: issue_type=<type>, urgency=<urgency>, sentiment=<sentiment>"

Do NOT attempt to resolve the issue — that is the Resolver Agent's job.
Always call `update_ticket_status` to save your classification.

"""

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def extract_classification(state: ClassifierState, config: RunnableConfig) -> dict:
   """
   Run a structured-output pass to populate ClassifierState.classification.
   """
   logger.debug("extract_classification node invoked. message count=%d", len(state["messages"]))

   llm = config.get("configurable", {}).get("llm", None)

   if llm is None:
      logger.warning("No 'llm' found in configurable; falling back to default ChatOpenAI.")
      return {}

   llm_with_tools = llm.bind_tools(CLASSIFIER_TOOLS)
   messages = [SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT)] + state["messages"]

   logger.debug("Invoking structured-output LLM with %d tools.", len(CLASSIFIER_TOOLS))
   llm_with_structured = llm_with_tools.with_structured_output(ClassificationOutput)

   try:
      result: ClassificationOutput = llm_with_structured.invoke(messages)
   except Exception as exc:
      logger.exception("Structured-output LLM call failed: %s", exc)
      raise

   logger.info(
      "Classification complete — issue_type=%s, urgency=%s, sentiment=%s",
      result.issue_type,
      result.urgency,
      result.sentiment,
   )
   return {"classification": result}

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(ClassifierState)
builder.add_node("extract_classification", extract_classification)
builder.add_node("tools", ToolNode(CLASSIFIER_TOOLS))

builder.set_entry_point("extract_classification")
builder.add_conditional_edges(
    "extract_classification",
    tools_condition
)

builder.add_edge("tools", "extract_classification")

classifier_agent = builder.compile()
classifier_agent.name = "classifier"
