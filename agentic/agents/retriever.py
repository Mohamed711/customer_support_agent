
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

RETRIEVER_SEARCH_PROMPT = """
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
- Call `search_knowledge_base` with relevant keywords extracted from the customer message.
- You may call it a second time with rephrased keywords if first results are not relevant.
- Do NOT attempt to answer the customer's question.
- Once you have searched, stop. Do not call any other tools.
"""

RETRIEVER_EXTRACT_PROMPT = """
Based on the knowledge base search results in this conversation, produce a structured
retrieval assessment:

- **confidence**: How well the retrieved articles answer the customer's question:
  - 0.8 – 1.0 : Article(s) directly and fully address the question.
  - 0.6 – 0.79: Content is relevant and partially answers it.
  - 0.4 – 0.59: Content is only tangentially related.
  - 0.0 – 0.39: No article meaningfully addresses the issue.

- **articles_found**: Total number of relevant articles from the search results.

- **retrieved_articles**: For each relevant article, provide its title, a brief summary,
  and one sentence explaining why it is relevant to this customer's question.

Base your assessment on the actual search result content, not on guesses.
"""

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def llm_call(state: RetrieverState, config: RunnableConfig) -> dict:
    """ReAct node: binds search tools and lets the LLM call search_knowledge_base."""
    logger.debug("llm_call node invoked. message count=%d", len(state["messages"]))

    llm = config.get("configurable", {}).get("llm", None)
    if llm is None:
        logger.warning("No 'llm' found in configurable; skipping.")
        return {}

    llm_with_tools = llm.bind_tools(RETRIEVER_TOOLS)
    messages = [SystemMessage(content=RETRIEVER_SEARCH_PROMPT)] + state["messages"]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
        raise

    return {"messages": [response]}


def _should_continue(state: RetrieverState) -> str:
    """Route to tools if there are pending tool calls, otherwise extract structured output."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "extract_retrieval"


def extract_retrieval(state: RetrieverState, config: RunnableConfig) -> dict:
    """Structured-output node: reads all messages (including tool results) and
    produces a RetrieverOutput with confidence score and article list."""
    logger.debug("extract_retrieval node invoked. message count=%d", len(state["messages"]))

    llm = config.get("configurable", {}).get("llm", None)
    if llm is None:
        logger.warning("No 'llm' found in configurable; skipping.")
        return {}

    llm_structured = llm.with_structured_output(RetrieverOutput)
    messages = [SystemMessage(content=RETRIEVER_EXTRACT_PROMPT)] + state["messages"]

    try:
        result: RetrieverOutput = llm_structured.invoke(messages)
    except Exception as exc:
        logger.exception("Structured extraction failed: %s", exc)
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
builder.add_node("llm_call", llm_call)
builder.add_node("tools", ToolNode(RETRIEVER_TOOLS))
builder.add_node("extract_retrieval", extract_retrieval)

builder.set_entry_point("llm_call")
builder.add_conditional_edges("llm_call", _should_continue, {"tools": "tools", "extract_retrieval": "extract_retrieval"})
builder.add_edge("tools", "llm_call")
builder.add_edge("extract_retrieval", END)

retriever_agent = builder.compile()
retriever_agent.name = "retriever"
