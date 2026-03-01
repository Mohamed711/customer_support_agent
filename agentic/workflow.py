
import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import create_supervisor

from agentic.agents.classifier import classifier_agent
from agentic.agents.retriever import retriever_agent
from agentic.agents.resolver import resolver_agent
from agentic.agents.escalation import escalation_agent

SUPERVISOR_PROMPT = """
    You are the **Supervisor** of UDA-Hub, a universal decision agent \
    for customer support.

    You orchestrate four specialised agents to handle support tickets end-to-end:

    1. classifier agent: Analyzes the customer message and ticket, determines issue type, \
    urgency, and then tags the ticket.

    2. retriever agent: Searches the knowledge base (RAG) and evaluates — based on reading \
    the retrieved article content, It returns a confidence score indicating how well the KB can answer the issue.

    3. resolver agent: Uses the retrieved KB articles (visible in the conversation context \
    from the retriever's output) and CultPass databases to compose an accurate, helpful answer. \
    It returns "NEEDS_ESCALATION" if the issue cannot be resolved automatically.

    4. escalation agent: Handles unresolvable cases: writes a structured escalation note \
    for the human support lead and sends the customer a reassuring hand-off message.

    ### Routing rules (follow STRICTLY):
    - ALWAYS start with classifier agent for NEW incoming messages.
    - After classification, ALWAYS route to retriever agent.
    - After retriever agent responds, determine the urgency from the classifier agent's output, then apply
    the appropriate confidence threshold to the RETRIEVAL_RESULT:

    **High-urgency tickets** (urgency = high):
        * confidence >= 0.75  →  route to resolver agent.
        * confidence <  0.75  →  route DIRECTLY to escalation agent.

    **Normal tickets** (urgency = medium or low):
        * confidence >= 0.60  →  route to resolver agent.
        * confidence <  0.60  →  route DIRECTLY to escalation agent.

    - If resolver returns "NEEDS_ESCALATION" or signals it cannot resolve, route to escalation agent.
    - If resolver successfully answers, return the resolver's final answer directly.
    - For follow-up messages in the same conversation, skip classifier agent unless the topic changes.

    Be decisive. Do not ask the user clarifying questions — delegate to the right agent.
"""

# Load the environment variables from .env file
load_dotenv()
if not os.getenv("VOCAREUM_OPENAPI_KEY"):
    load_dotenv(Path.home() / ".env")

# Setup the LLM model with the appropriate API key and base URL
llm_model = ChatOpenAI(
        model="gpt-4o-mini",
        base_url="https://openai.vocareum.com/v1",
        api_key=os.getenv("VOCAREUM_OPENAPI_KEY"),
        temperature=0
    )

# Create a supervisor agent that orchestrates the classifier, retriever, resolver, and escalation agents
supervisor_graph = create_supervisor(
    agents=[classifier_agent, retriever_agent, resolver_agent, escalation_agent],
    model=llm_model,
    prompt=SUPERVISOR_PROMPT,
    output_mode="last_message",
)

orchestrator = supervisor_graph.compile(checkpointer=MemorySaver())
