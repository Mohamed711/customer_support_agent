"""
main.py
-------
Interactive supervisor agent chat interface for testing the orchestrator.
Demonstrates two scenarios:
    1. A resolvable ticket (subscription pause inquiry)
    2. An escalation case (high-urgency account issue)

Run from the project root:
        python main.py
"""

import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from agentic.workflow import orchestrator, llm_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()
if not os.getenv("VOCAREUM_OPENAPI_KEY"):
    load_dotenv(Path.home() / ".env")

# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------

def chat_interface(agent: CompiledStateGraph, ticket_id: str) -> None:
    """
    Interactive chat interface for the supervisor agent.

    Args:
        agent: The compiled supervisor graph
        ticket_id: Unique thread identifier for the conversation
    """
    print(f"\n{'='*70}")
    print(f"Starting conversation session: {ticket_id}")
    print(f"{'='*70}\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nAssistant: Goodbye!")
            break

        if not user_input:
            continue

        # Build message list
        messages = [HumanMessage(content=user_input)]

        # Create state for the agent
        state = {"messages": messages}

        # Create config with thread_id for memory
        config = {
            "configurable": {
                "thread_id": ticket_id,
                "llm": llm_model,
            },
            "recursion_limit": 25,
        }

        try:
            # Invoke the agent
            result = agent.invoke(state, config=config)

            # Extract and print the last assistant message
            if result.get("messages"):
                last_message = result["messages"][-1]
                content = getattr(last_message, "content", "")
                print(f"\nAssistant: {content}\n")
            else:
                print("\nAssistant: [No response generated]\n")

        except Exception as exc:
            logger.error(f"Error during agent invocation: {exc}", exc_info=True)
            print(f"\nAssistant: I encountered an error. Please try again.\n")


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------

def scenario_resolved() -> None:
    """Scenario 1: Resolvable ticket (subscription cancellation FAQ)"""
    print("\n" + "="*70)
    print("SCENARIO 1: RESOLVABLE - Subscription Cancellation FAQ")
    print("="*70)
    print("Expected: The agent should resolve this using knowledge base info")
    print("="*70 + "\n")

    ticket_id = str(uuid.uuid4())
    user_input = (
        "Hi, I have a basic question: how do I cancel my CultPass subscription? "
        "Please share the cancellation steps and timeline."
    )

    print(f"Customer: {user_input}\n")

    state = {"messages": [HumanMessage(content=user_input)]}
    config = {
        "configurable": {
            "thread_id": ticket_id,
            "llm": llm_model,
        },
        "recursion_limit": 25,
    }

    try:
        result = orchestrator.invoke(state, config=config)
        if result.get("messages"):
            last_message = result["messages"][-1]
            content = getattr(last_message, "content", "")
            print(f"Agent: {content}\n")
    except Exception as exc:
        logger.error(f"Error in resolved scenario: {exc}", exc_info=True)
        print(f"Error: {exc}\n")


def scenario_escalated() -> None:
    """Scenario 2: Escalation case (high-urgency account blocked)"""
    print("\n" + "="*70)
    print("SCENARIO 2: ESCALATION - Account Blocked (High Urgency)")
    print("="*70)
    print("Expected: The agent should escalate this to human support")
    print("="*70 + "\n")

    ticket_id = str(uuid.uuid4())
    user_input = (
        "My account was suddenly blocked without any warning! "
        "I can't access any of my bookings or experiences. "
        "This is urgent and I need immediate help!"
    )

    print(f"Customer: {user_input}\n")

    state = {"messages": [HumanMessage(content=user_input)]}
    config = {
        "configurable": {
            "thread_id": ticket_id,
            "llm": llm_model,
        },
        "recursion_limit": 25,
    }

    try:
        result = orchestrator.invoke(state, config=config)
        if result.get("messages"):
            last_message = result["messages"][-1]
            content = getattr(last_message, "content", "")
            print(f"Agent: {content}\n")
    except Exception as exc:
        logger.error(f"Error in escalation scenario: {exc}", exc_info=True)
        print(f"Error: {exc}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting supervisor agent interface")

    # Run predefined scenarios
    scenario_resolved()
    scenario_escalated()

    # Optional: Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("You can now chat interactively with the supervisor agent.")
    print("Type 'quit', 'exit', or 'q' to end the session.\n")

    ticket_id = str(uuid.uuid4())
    chat_interface(orchestrator, ticket_id)

    logger.info("Supervisor agent session complete")
