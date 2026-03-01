
import logging
from typing import Annotated, List, Optional

from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agentic.tools.knowledge_tools import search_knowledge_base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RetrievedArticle(BaseModel):
    """A single article returned from the knowledge base."""

    title: str = Field(description="Title of the knowledge base article.")
    summary: str = Field(description="Brief summary of what the article covers.")
    relevance: str = Field(
        description="One sentence explaining why this article is relevant to the customer's issue."
    )


class RetrieverOutput(BaseModel):
    """Structured retrieval result produced by the agent."""

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well the retrieved articles answer the customer's issue: "
            "0.8-1.0=fully addressed, 0.6-0.79=partially, 0.4-0.59=tangential, 0.0-0.39=none."
        ),
    )

    articles_found: int = Field(
        description="Total number of relevant articles retrieved from the knowledge base."
    )

    retrieved_articles: List[RetrievedArticle] = Field(
        default_factory=list,
        description="List of articles retrieved and judged relevant.",
    )


class RetrieverState(TypedDict):
    """Full state carried through the retriever graph."""
    messages: Annotated[list, add_messages]
    confidence: Optional[float]
    retrieved_articles: Optional[List[RetrievedArticle]]

# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------

RETRIEVER_TOOLS = [search_knowledge_base]

RETRIEVER_SYSTEM_PROMPT = """
You are Retriever Agent for UDA-Hub.

Your job is to search the knowledge base and evaluate — based on the
content of what you retrieve — whether the knowledge base contains sufficient information
to resolve the customer's issue.

### Steps

1. **Identify the query**: Extract the core topic from the conversation.

2. **Search the knowledge base**: Call `search_knowledge_base` with the
   most relevant keywords. You may call it a second time with rephrased
   keywords if the first results do not seem relevant.

3. **Read and judge the articles**: Carefully read each returned article's
   title, content, and tags. Ask yourself:
   - Does this article directly answer the customer's specific question?
   - Does it cover the exact policy, procedure, or feature they are asking about?
   - Could a support agent use it to compose a complete, accurate reply?

4. **Assign a confidence score** based purely on your reading of the content:
   - 0.8 – 1.0 : Article(s) directly and fully address the customer's question.
   - 0.6 – 0.79: Content is relevant and partially answers the question; a
                  Resolver can fill in the remaining gaps with DB context.
   - 0.4 – 0.59: Content is only tangentially related; unlikely to resolve alone.
   - 0.0 – 0.39: No article meaningfully addresses the issue; escalation needed.

### Rules
- Your confidence score must reflect your assessment of the article CONTENT,
  not technical metadata. Read the articles, do not guess.
- Do NOT attempt to answer the customer's question.
"""

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def retriever(state: RetrieverState, config: RunnableConfig) -> dict:
   """ The retriever agent function """

   logger.debug("retriever node invoked. message count=%d with message %s",
                len(state["messages"]), state["messages"])

   llm = config.get("configurable", {}).get("llm", None)

   if llm is None:
      logger.warning("No 'llm' found in configurable; falling back to default ChatOpenAI.")
      return {}

   llm_with_tools = llm.bind_tools(RETRIEVER_TOOLS)
   messages = [SystemMessage(content=RETRIEVER_SYSTEM_PROMPT)] + state["messages"]

   logger.debug("Invoking structured-output LLM with %d tools.", len(RETRIEVER_TOOLS))
   llm_with_structured = llm_with_tools.with_structured_output(RetrieverOutput)

   try:
      result: RetrieverOutput = llm_with_structured.invoke(messages)
   except Exception as exc:
      logger.exception("Structured-output LLM call failed: %s", exc)
      raise

   logger.info(
      "Retrieval complete — confidence=%.2f, articles_found=%d",
      result.confidence,
      result.articles_found,
   )

   retrieval_msg = AIMessage(
      content=f"RETRIEVAL_RESULT: confidence={result.confidence:.2f}, articles_found={result.articles_found}"
   )
   return {
      "messages": [retrieval_msg],
      "confidence": result.confidence,
      "retrieved_articles": result.retrieved_articles,
   }

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(RetrieverState)
builder.add_node("retriever", retriever)
builder.add_node("tools", ToolNode(RETRIEVER_TOOLS))

builder.set_entry_point("retriever")
builder.add_conditional_edges("retriever", tools_condition)
builder.add_edge("tools", "retriever")

retriever_agent = builder.compile()
retriever_agent.name = "retriever"
