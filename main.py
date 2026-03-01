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
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import create_engine

from agentic.workflow import orchestrator, llm_model
from utils import get_session
from data.models import udahub

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
# DB seeding — real tickets so resolver/escalation agents can query the DB
# ---------------------------------------------------------------------------

_DB_PATH = "data/core/udahub.db"
_ACCOUNT_ID = "cultpass"

_TEST_TICKETS = [
    {
        "ticket_id": "T-001",
        "external_user_id": "f556c0",
        "user_name": "Bob Stone",
        "channel": "chat",
        "status": "open",
        "issue_type": "subscription",
        "tags": "subscription, cancel",
        "content": "How do I cancel my CultPass subscription?",
    },
    {
        "ticket_id": "T-003",
        "external_user_id": "a4ab87",
        "user_name": "Alice Kingsley",
        "channel": "chat",
        "status": "open",
        "issue_type": "account",
        "tags": "account, blocked, access",
        "content": (
            "My account was blocked without any notice. "
            "I cannot access any of my booked experiences. Please unblock immediately."
        ),
    },
]


def seed_test_tickets() -> None:
    """Insert test tickets into the DB if they don't already exist."""
    if not Path(_DB_PATH).exists():
        logger.warning("DB not found at %s — skipping ticket seeding.", _DB_PATH)
        return

    engine = create_engine(f"sqlite:///{_DB_PATH}", echo=False)

    with get_session(engine) as session:
        for t in _TEST_TICKETS:
            user = session.query(udahub.User).filter_by(
                account_id=_ACCOUNT_ID,
                external_user_id=t["external_user_id"],
            ).first()

            if not user:
                user = udahub.User(
                    user_id=str(uuid.uuid4()),
                    account_id=_ACCOUNT_ID,
                    external_user_id=t["external_user_id"],
                    user_name=t["user_name"],
                )
                session.add(user)
                session.flush()

            if session.query(udahub.Ticket).filter_by(ticket_id=t["ticket_id"]).first():
                logger.debug("Ticket %s already exists — skipping.", t["ticket_id"])
                continue

            ticket = udahub.Ticket(
                ticket_id=t["ticket_id"],
                account_id=_ACCOUNT_ID,
                user_id=user.user_id,
                channel=t["channel"],
            )
            metadata = udahub.TicketMetadata(
                ticket_id=t["ticket_id"],
                status=t["status"],
                main_issue_type=t["issue_type"],
                tags=t["tags"],
            )
            message = udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=t["ticket_id"],
                role=udahub.RoleEnum.user,
                content=t["content"],
            )
            session.add_all([ticket, metadata, message])
            logger.info("Seeded ticket %s for %s.", t["ticket_id"], t["user_name"])


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
    """Scenario 1: Resolvable ticket — subscription cancellation FAQ with real ticket T-001."""
    print("\n" + "="*70)
    print("SCENARIO 1: RESOLVABLE - Subscription Cancellation FAQ")
    print("="*70)
    print("Expected: The agent should resolve this using knowledge base info")
    print("="*70 + "\n")

    # Use the seeded ticket T-001 so the resolver can look it up in the DB
    ticket_id = "T-001"
    user_input = (
        "Hi, this is ticket T-001. "
        "I have a basic question: how do I cancel my CultPass subscription? "
        "Please share the cancellation steps and timeline."
    )

    print(f"Customer: {user_input}\n")

    state = {"messages": [HumanMessage(content=user_input)]}
    config = {
        "configurable": {
            "thread_id": ticket_id,
            "llm": llm_model,
        },
        "recursion_limit": 35,
    }

    try:
        result = orchestrator.invoke(state, config=config)
        if result.get("messages"):
            last_message = result["messages"][-1]
            content = getattr(last_message, "content", "")
            print(f"Agent: {content}\n")
    except GraphRecursionError as exc:
        logger.error(f"Recursion limit hit in resolved scenario: {exc}")
        print(f"[!] Recursion limit reached: {exc}\n")
    except Exception as exc:
        logger.error(f"Error in resolved scenario: {exc}", exc_info=True)
        print(f"Error: {exc}\n")


def scenario_escalated() -> None:
    """Scenario 2: Escalation case — high-urgency blocked account with real ticket T-003."""
    print("\n" + "="*70)
    print("SCENARIO 2: ESCALATION - Account Blocked (High Urgency)")
    print("="*70)
    print("Expected: The agent should escalate this to human support")
    print("="*70 + "\n")

    # Use the seeded ticket T-003 so the escalation agent can look it up in the DB
    ticket_id = "T-003"
    user_input = (
        "This is ticket T-003. My account was suddenly blocked without any warning! "
        "I can't access any of my bookings or experiences. "
        "This is urgent — please unblock my account immediately!"
    )

    print(f"Customer: {user_input}\n")

    state = {"messages": [HumanMessage(content=user_input)]}
    config = {
        "configurable": {
            "thread_id": ticket_id,
            "llm": llm_model,
        },
        "recursion_limit": 35,
    }

    try:
        result = orchestrator.invoke(state, config=config)
        if result.get("messages"):
            last_message = result["messages"][-1]
            content = getattr(last_message, "content", "")
            print(f"Agent: {content}\n")
    except GraphRecursionError as exc:
        logger.error(f"Recursion limit hit in escalation scenario: {exc}")
        print(f"[!] Recursion limit reached: {exc}\n")
    except Exception as exc:
        logger.error(f"Error in escalation scenario: {exc}", exc_info=True)
        print(f"Error: {exc}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting supervisor agent interface")

    # Seed the DB so resolver/escalation agents can look up tickets
    seed_test_tickets()

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
