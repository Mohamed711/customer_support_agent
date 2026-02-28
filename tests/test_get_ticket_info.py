import os
import sys
import uuid
import pytest
from sqlalchemy import create_engine

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from data.models import udahub
from utils import get_session
import agentic.tools.ticket_tools as tt


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_get_ticket_info(monkeypatch, tmp_path):
    """
    Seed a temporary database with one ticket and confirm get_ticket_info
    returns all fields with the expected values.
    """
    # --- setup temp DB ---
    engine = create_engine(f"sqlite:///{tmp_path / 'test_udahub.db'}", echo=False)
    udahub.Base.metadata.create_all(engine)
    monkeypatch.setattr(tt, "engine", engine)

    ticket_id = str(uuid.uuid4())

    with get_session(engine) as session:
        account_id = str(uuid.uuid4())
        user_id    = str(uuid.uuid4())

        session.add_all([
            udahub.Account(account_id=account_id, account_name="Test Account"),
            udahub.User(user_id=user_id, account_id=account_id,
                        external_user_id="ext-001", user_name="alice"),
            udahub.Ticket(ticket_id=ticket_id, account_id=account_id,
                          user_id=user_id, channel="chat"),
            udahub.TicketMetadata(ticket_id=ticket_id, status="open",
                                  main_issue_type="login", tags="login, access"),
            udahub.TicketMessage(message_id=str(uuid.uuid4()), ticket_id=ticket_id,
                                 role=udahub.RoleEnum.user,
                                 content="I can't log in to my account."),
        ])

    # --- call the tool ---
    result = tt.get_ticket_info(ticket_id)

    # --- assertions ---
    assert result.get("ticket_id")  == ticket_id
    assert result.get("channel")    == "chat"
    assert result.get("user")       == "alice"
    assert result.get("status")     == "open"
    assert result.get("issue_type") == "login"
    assert "login" in result.get("tags", "")
    assert result.get("messages") == [{"role": "user", "content": "I can't log in to my account."}]
