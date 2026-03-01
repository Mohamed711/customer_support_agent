"""
main.py
-------
Smoke-tests each agent graph individually by invoking them with representative
inputs and printing the key fields from the returned state.

Run from the project root:
    python main.py
"""

import json
import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from sqlalchemy import create_engine

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
# Environment & LLM
# ---------------------------------------------------------------------------

load_dotenv()
if not os.getenv("VOCAREUM_OPENAPI_KEY"):
    load_dotenv(Path.home() / ".env")

_llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_OPENAPI_KEY"),
    temperature=0,
)


_CONFIG = {"configurable": {"llm": _llm}, "recursion_limit": 20}

# ---------------------------------------------------------------------------
# DB seeding — dummy tickets used by the smoke tests
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
        "tags": "subscription, pause, travel",
        "content": "Can I pause my CultPass subscription for two months while I travel?",
    },
    {
        "ticket_id": "T-002",
        "external_user_id": "88382b",
        "user_name": "Cathy Bloom",
        "channel": "email",
        "status": "open",
        "issue_type": "billing",
        "tags": "billing, refund, double charge",
        "content": "I was charged twice in March. I need a refund immediately.",
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
    """Insert T-001, T-002, T-003 into the DB if they don't already exist."""
    if not Path(_DB_PATH).exists():
        logger.warning("DB not found at %s — skipping ticket seeding.", _DB_PATH)
        return

    engine = create_engine(f"sqlite:///{_DB_PATH}", echo=False)

    with get_session(engine) as session:
        for t in _TEST_TICKETS:
            # Upsert the user
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

            # Skip if ticket already exists
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
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "\n" + "=" * 70 + "\n"

def _section(title: str) -> None:
    print(f"{SEPARATOR}>>> {title}{SEPARATOR}")


def _print_messages(messages: list) -> None:
    for i, msg in enumerate(messages):
        role = type(msg).__name__
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        print(f"  [{i}] {role}: {content[:200]!r}")
        if tool_calls:
            for tc in tool_calls:
                print(f"       tool_call → {tc['name']}({json.dumps(tc.get('args', {}), default=str)[:120]})")


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

# Each scenario is (label, agent_name_for_logging, agent, input_state, extra_check_fn)
# extra_check_fn receives the final state dict and prints important fields.

def run_classifier() -> None:
    from agentic.agents.classifier import classifier_agent

    scenarios = [
        (
            "Subscription downgrade enquiry (low urgency, neutral)",
            {"messages": [HumanMessage(content="Hi, I want to downgrade my CultPass plan from Premium to Basic.")]},
        ),
        (
            "Payment failure (high urgency, frustrated)",
            {"messages": [HumanMessage(content="I was charged twice for my subscription this month! This is ridiculous!")]},
        ),
        (
            "Login problem (medium urgency, negative)",
            {"messages": [HumanMessage(content="I cannot log in. I keep getting 'invalid credentials' even after resetting my password.")]},
        ),
    ]

    _section("CLASSIFIER AGENT")
    for label, state in scenarios:
        print(f"\n--- Scenario: {label} ---")
        try:
            result = classifier_agent.invoke(state, config=_CONFIG)
        except GraphRecursionError as exc:
            print(f"  [!] GraphRecursionError: {exc}")
            continue
        except Exception as exc:
            print(f"  [!] Unexpected error: {exc}")
            continue

        # Messages
        _print_messages(result.get("messages", []))

        # Structured classification output
        clf = result.get("classification")
        if clf:
            print(f"  classification.issue_type : {clf.issue_type}")
            print(f"  classification.urgency    : {clf.urgency}")
            print(f"  classification.sentiment  : {clf.sentiment}")
            print(f"  classification.summary    : {clf.summary}")
        else:
            print("  [!] classification field is None — structured output was not populated")


def run_retriever() -> None:
    from agentic.agents.retriever import retriever_agent

    scenarios = [
        (
            "Subscription cancellation query",
            {"messages": [HumanMessage(content="How do I cancel my CultPass subscription?")]},
        ),
        (
            "Obscure/unlikely topic (expect low confidence)",
            {"messages": [HumanMessage(content="Can I use my CultPass points to buy hardware?")]},
        ),
    ]

    _section("RETRIEVER AGENT")
    for label, state in scenarios:
        print(f"\n--- Scenario: {label} ---")
        try:
            result = retriever_agent.invoke(state, config=_CONFIG)
        except GraphRecursionError as exc:
            print(f"  [!] GraphRecursionError: {exc}")
            continue
        except Exception as exc:
            print(f"  [!] Unexpected error: {exc}")
            continue

        _print_messages(result.get("messages", []))

        confidence = result.get("confidence")
        articles = result.get("retrieved_articles") or []
        print(f"  confidence        : {confidence}")
        print(f"  articles_found    : {len(articles)}")
        for a in articles:
            print(f"    • {a.title!r} — {a.relevance[:100]}")

        if confidence is None:
            print("  [!] confidence field is None — structured output was not populated")


def run_resolver() -> None:
    from agentic.agents.resolver import resolver_agent

    scenarios = [
        (
            "Resolvable: subscription pause question",
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Ticket #T-001. Customer asks: Can I pause my CultPass subscription "
                            "for two months while I travel? "
                            "[Retriever context: KB article 'Subscription Pause Policy' found, "
                            "confidence=0.85]"
                        )
                    )
                ]
            },
        ),
        (
            "Needs escalation: double-charge refund request",
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Ticket #T-002. Customer asks: I was charged twice in March. "
                            "I need a refund immediately. "
                            "[Retriever context: no KB article directly covers manual refunds, "
                            "confidence=0.32]"
                        )
                    )
                ]
            },
        ),
    ]

    _section("RESOLVER AGENT")
    for label, state in scenarios:
        print(f"\n--- Scenario: {label} ---")
        try:
            result = resolver_agent.invoke(state, config=_CONFIG)
        except GraphRecursionError as exc:
            print(f"  [!] GraphRecursionError (ticket not found in DB — expected for synthetic IDs): {exc}")
            continue
        except Exception as exc:
            print(f"  [!] Unexpected error: {exc}")
            continue

        _print_messages(result.get("messages", []))

        last_content = result["messages"][-1].content if result.get("messages") else ""
        if "NEEDS_ESCALATION" in last_content:
            print("  ✓ NEEDS_ESCALATION signal emitted correctly")
        else:
            print("  ✓ Resolver produced a response (no escalation)")


def run_escalation() -> None:
    from agentic.agents.escalation import escalation_agent

    scenarios = [
        (
            "High-urgency escalation: blocked account",
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Ticket #T-003, urgency=high. Customer reports their account was "
                            "blocked without notice and they cannot access any booked experiences. "
                            "Resolver could not unblock automatically."
                        )
                    )
                ]
            },
        ),
    ]

    _section("ESCALATION AGENT")
    for label, state in scenarios:
        print(f"\n--- Scenario: {label} ---")
        try:
            result = escalation_agent.invoke(state, config=_CONFIG)
        except GraphRecursionError as exc:
            print(f"  [!] GraphRecursionError (LLM retried tools after DB misses): {exc}")
            continue
        except Exception as exc:
            print(f"  [!] Unexpected error: {exc}")
            continue

        _print_messages(result.get("messages", []))

        last = result["messages"][-1] if result.get("messages") else None
        if last:
            print(f"\n  Final agent message:\n  {last.content[:500]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting agent smoke tests")

    seed_test_tickets()

    run_classifier()
    run_retriever()
    run_resolver()
    run_escalation()

    logger.info("All smoke tests complete")
