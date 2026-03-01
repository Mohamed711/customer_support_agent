# UDA-Hub — Customer Support Agent

An LLM-powered multi-agent customer support system built with **LangGraph** and **LangChain**.
It orchestrates four specialised agents (Classifier, Retriever, Resolver, Escalation) under a
Supervisor that routes each ticket based on issue urgency and knowledge-base confidence.

---

## Running the Main Script

The `main.py` file runs two predefined end-to-end test scenarios against the full supervisor
graph, then drops into an interactive chat session.

```bash
python main.py
```

It will:
1. Seed two test tickets (T-001, T-003) into the database.
2. Run **Scenario 1** — a resolvable subscription question.
3. Run **Scenario 2** — a high-urgency account blockage that gets escalated.
4. Open an **interactive chat** session (type `quit` to exit).

---

## Example Scenarios

### Scenario 1 — RESOLVABLE: Subscription Cancellation FAQ

**Customer input:**
```
Hi, this is ticket T-001. I have a basic question: how do I cancel my CultPass subscription?
Please share the cancellation steps and timeline.
```

**Agent routing:**
```
Classifier  →  issue_type=subscription, urgency=low, sentiment=neutral
Retriever   →  confidence=0.80, articles_found=1  (threshold for low urgency: ≥0.60 ✓)
Resolver    →  composes answer from KB article + ticket context
```

**Sample agent output:**
```
Agent: To cancel your CultPass subscription, follow these steps:

1. Log in to your CultPass account at cultpass.com.
2. Go to Account Settings → Subscription.
3. Select "Cancel Subscription" and confirm.

Your subscription remains active until the end of the current billing period.
No partial refunds are issued for unused days. Your ticket (T-001) has been
updated and marked as resolved. Let us know if you have any other questions!
```

---

### Scenario 2 — ESCALATION: Account Blocked (High Urgency)

**Customer input:**
```
This is ticket T-003. My account was suddenly blocked without any warning!
I can't access any of my bookings or experiences.
This is urgent — please unblock my account immediately!
```

**Agent routing:**
```
Classifier  →  issue_type=account, urgency=high, sentiment=frustrated
Retriever   →  confidence=0.40, articles_found=5  (threshold for high urgency: ≥0.75 ✗)
Escalation  →  drafts escalation note + sends customer hand-off message
```

**Sample agent output:**
```
Agent: I'm truly sorry to hear your account has been blocked — I completely understand
how frustrating this must be, especially when you cannot access your booked experiences.

I've escalated your case (Ticket #T-003) to our senior support team with HIGH priority.
A human agent will personally review your account and reach out to you within 4 hours.

In the meantime, rest assured that your bookings are safe and will not be affected.
Thank you for your patience, and I'm sorry again for the inconvenience.
```

---

## Testing

### Running Individual Agent Tests

The `tests/agent_testcases.py` file smoke-tests each of the four agents independently
with representative inputs and prints structured output for inspection.

```bash
# Run from the project root
python tests/agent_testcases.py
```

### Running Tool Unit Tests

```bash
# Run all tool tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_get_user_subscription.py
python -m pytest tests/test_search_knowledge_base.py
python -m pytest tests/test_get_ticket_info.py
```

### What Each Test Covers

| Test file | What it verifies |
|---|---|
| `tests/agent_testcases.py` | End-to-end smoke test for each agent: classifier, retriever, resolver, escalation |
| `tests/test_get_user_subscription.py` | `get_user_subscription` tool against real CultPass DB |
| `tests/test_search_knowledge_base.py` | `search_knowledge_base` vector search against ChromaDB |
| `tests/test_get_ticket_info.py` | `get_ticket_info` tool against UdaHub DB |

---

## Further Reading

- **System architecture, agent routing rules, and memory design:**
  [`agentic/design/architecture.md`](agentic/design/architecture.md)

- **Tool descriptions, usage, and MCP server setup:**
  [`agentic/tools/README.md`](agentic/tools/README.md)

---
