# UDA-Hub — System Architecture

## Agent Graph

```
User Input
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                        SUPERVISOR                       │
│                                                         │
│  Routing rules:                                         │
│    1. New message  → Classifier → Retriever             │
│    2. high urgency & confidence ≥ 0.75 → Resolver       │
│    3. high urgency & confidence < 0.75 → Escalation     │
│    4. normal urgency & confidence ≥ 0.60 → Resolver     │
│    5. normal urgency & confidence < 0.60 → Escalation   │
│    6. NEEDS_ESCALATION signal → Escalation Agent        │
└──────────┬──────────────┬──────────────┬────────────────┘
           │              │              │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌──── ▼──────────┐  ┌────────────────┐
    │  CLASSIFIER │ │  RETRIEVER │ │    RESOLVER    │  │   ESCALATION   │
    │             │ │            │ │                │  │                │
    │ Tools:      │ │ Tools:     │ │ Tools:         │  │ Tools:         │
    │ - get_      │ │ - search_  │ │ - get_ticket_  │  │ - get_ticket_  │
    │   ticket_   │ │   knowledge│ │   info         │  │   info         │
    │   info      │ │   _base    │ │ - get_customer_│  │ - get_cultpass_│
    │ - update_   │ │            │ │   ticket_      │  │   user_info    │
    │   ticket_   │ │ Output:    │ │   history      │  │ - add_ticket_  │
    │   status    │ │ RETRIEVAL_ │ │ - get_cultpass_│  │   message      │
    │             │ │ RESULT:    │ │   user_info    │  │ - update_      │
    │             │ │ confidence │ │ - get_user_    │  │   ticket_      │
    │             │ │ =0.0–1.0   │ │   reservations │  │   status       │
    │             │ │            │ │ - get_         │  └────────────────┘
    │             │ │            │ │   experience_  │
    │             │ │            │ │   availability │
    │             │ │            │ │ - add_ticket_  │
    │             │ │            │ │   message      │
    │             │ │            │ │ - update_      │
    │             │ │            │ │   ticket_      │
    │             │ │            │ │   status       │
    └─────────────┘ └────────────┘ └────────────────┘
```

---

## Agents

### 1. Supervisor
- **Role**: Orchestrates all sub-agents; decides routing based on conversation state and retriever confidence.
- **Memory**: LangGraph `MemorySaver` – per-session short-term memory keyed by `thread_id`.

### 2. Classifier Agent
- **Role**: Reads the customer message and classifies the issue.
- **Outputs**: `issue_type`, `urgency` → persisted to `ticket_metadata` via `update_ticket_status`.
- **Issue Types**: login | billing | reservation | subscription | account | general
- **Urgency Levels**: high | medium | low

### 3. Retriever Agent
- **Role**: Searches the knowledge base and **evaluates confidence** based on reading the content of the retrieved articles. The LLM judges how well the KB can answer the customer's question and produces a confidence score (0.0–1.0).
- **Output signal**: `RETRIEVAL_RESULT: confidence=<score>, articles_found=<count>`
- **Routing impact** (urgency-aware dual thresholds):
  - **High urgency**: confidence ≥ 0.75 → Resolver; confidence < 0.75 → Escalation directly.
  - **Normal urgency** (medium / low): confidence ≥ 0.60 → Resolver; confidence < 0.60 → Escalation directly.
  - High-urgency tickets demand a stricter KB match — if the KB is not highly confident, a human agent is more appropriate than a partially-informed automated response.

### 4. Resolver Agent
- **Role**: Composes the final resolution using KB articles already retrieved (present in conversation context) and CultPass DB lookups. Does **not** search the KB — that is the Retriever's job.
- **Personalisation**: Calls `get_customer_ticket_history` (when an `external_user_id` is available) to surface prior interactions, allowing it to acknowledge repeat issues and personalise the response for returning customers.
- **Preferences**: Calls `get_user_preferences` to apply known language/channel preferences; calls `update_user_preferences` when new preferences are discovered during an interaction, persisting them for future sessions.
- **Resolution**: Appends AI response to ticket thread; sets status to `resolved`.
- **Escalation Trigger**: Returns "NEEDS_ESCALATION" when manual intervention is required.

### 5. Escalation Agent
- **Role**: Writes a structured escalation note (for human lead) + empathetic customer message.
- **Outputs**: Sets ticket status to `escalated`; appends both a system note and customer-facing message.
- **Triggered by**: Retriever confidence below the urgency-appropriate threshold (< 0.75 for high urgency, < 0.60 for normal) OR resolver returning "NEEDS_ESCALATION".

---

## Memory Architecture

| Memory Type    | Mechanism                         | Scope           |
|---------------|-----------------------------------|-----------------|
| Short-term     | LangGraph `MemorySaver`           | Per session (thread_id = ticket_id) |
| Long-term      | SQLite `ticket_messages` table    | Permanent; all messages persisted |
| Classification | SQLite `ticket_metadata` table    | Permanent; issue_type, tags, status |
| Customer history | `get_customer_ticket_history` tool | Cross-ticket; retrieved on demand by Resolver to personalise responses for returning customers |
| Preferences    | SQLite `user_preferences` table   | Permanent; per-user language, channel, notes — retrieved and updated by Resolver across sessions |

---

## Tools

| Tool                          | Agent(s)               |
|-------------------------------|------------------------|
| `search_knowledge_base`       | Retriever              |
| `get_ticket_info`             | Classifier, Resolver, Escalation |
| `update_ticket_status`        | Classifier, Resolver, Escalation |
| `add_ticket_message`          | Resolver, Escalation   |
| `get_customer_ticket_history` | Resolver               |
| `get_user_preferences`        | Resolver               |
| `update_user_preferences`     | Resolver               |
| `get_cultpass_user_info`      | Resolver, Escalation   |
| `get_user_reservations`       | Resolver               |
| `get_experience_availability` | Resolver               |

---

## Decision Flow

```
1. User sends message
        ↓
2. Supervisor receives → routes to Classifier
        ↓
3. Classifier analyses issue → updates ticket_metadata (issue_type, urgency, sentiment)
        ↓
4. Supervisor routes to Retriever
        ↓
5. Retriever searches KB → reads article content → LLM rates confidence (0.0–1.0)
   Returns: RETRIEVAL_RESULT: confidence=<score>, articles_found=<count>
        ↓
6. Supervisor applies urgency-aware confidence threshold:
   ┌─────────────────┬───────────────┬──────────────────────────────┐
   │ Urgency         │ Threshold     │ Routing                      │
   ├─────────────────┼───────────────┼──────────────────────────────┤
   │ high            │ 0.75          │ < 0.75 → Escalation (direct) │
   │ medium / low    │ 0.60          │ < 0.60 → Escalation (direct) │
   └─────────────────┴───────────────┴──────────────────────────────┘
        ↓ (confidence meets threshold)
7. Resolver uses retrieved articles + CultPass DB lookups → composes answer
        ↓
8a. Resolver saves response → status = 'resolved' → DONE
8b. Resolver cannot resolve → returns "NEEDS_ESCALATION"
        ↓
6b / 8b. Supervisor routes to Escalation Agent
        ↓
9. Escalation Agent writes human note + customer message → status = 'escalated' → DONE
```

