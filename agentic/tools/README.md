# Tools

This folder has all the tools used by the support agent. Each file is focused on a specific area:

- **cultpass_tools.py** — queries the CultPass database for user info, subscriptions, reservations, and experience availability. Also does similarity search over the experiences vector store.
  - `get_user_general_info` — returns basic profile info for a user (name, email, account status)
  - `get_user_subscription` — returns the user's current subscription tier and status
  - `get_user_reservations` — lists all reservations for a user with experience details and statuses
  - `search_experiences_by_keyword` — runs a vector similarity search over the experiences to find matches for a keyword or phrase
  - `get_experience_availability` — looks up a specific experience by ID and returns its details and available slots

- **knowledge_tools.py** — searches the UdaHub knowledge base using vector similarity to find relevant support articles.
  - `search_knowledge_base` — takes a natural language query and returns the most relevant knowledge articles from the vector store

- **ticket_tools.py** — reads and writes support tickets: fetching ticket details, updating status, and appending messages.
  - `get_ticket_info` — retrieves full ticket details including all messages and metadata
  - `update_ticket_status` — updates the status, issue type, and tags of a ticket
  - `add_ticket_message` — appends a new message to a ticket's conversation thread

- **tools_mcp_server.py** — the MCP server entry point. It registers all tools and runs over stdio so it can be plugged into any MCP-compatible client.

## Running the MCP server

```bash
python agentic/tools/tools_mcp_server.py
```

The server uses `stdio` transport, so it's ready to be used with Claude Desktop or any other MCP client that supports stdio.

## Tests

There are test files under the `tests/` folder to verify the tools work end to end against the actual databases:

```bash
# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_get_user_subscription.py
python -m pytest tests/test_search_knowledge_base.py
python -m pytest tests/test_get_ticket_info.py
```
