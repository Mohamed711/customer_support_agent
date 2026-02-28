
import os
import sys
import uuid
import logging
from typing import Dict
from sqlalchemy import create_engine

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from utils import get_session
from data.models import udahub
from settings import settings

from agentic.tools.tools_mcp_server import mcp

logger = logging.getLogger(__name__)

engine = create_engine(f"sqlite:///{settings.udahub_db_path}", echo=False)


@mcp.tool()
def get_ticket_info(ticket_id: str) -> Dict:
    """
    Retrieve full details of a support ticket including all messages and metadata.

    Args:
        ticket_id: The UUID of the ticket to retrieve.

    Returns:
        A dictionary with the ticket details, metadata, and conversation history.
        {
            "ticket_id": str,
            "channel": str,
            "created_at": str,
            "user": str,
            "status": str,
            "issue_type": str,
            "tags": str,
            "messages": List[Dict[str, str]] - A list of messages with role and content,
            "error": str - An error message if the ticket is not found or an exception occurs
        }
    """
    try:
        with get_session(engine) as session:
            ticket = session.query(udahub.Ticket).filter_by(ticket_id=ticket_id).first()

            if not ticket:
                return {"error": f"No ticket found with id: {ticket_id}"}

            meta = ticket.ticket_metadata
            user = ticket.user

            messages = [
                {"role": msg.role.name if msg.role else "unknown", "content": msg.content}
                for msg in ticket.messages
            ]

            logger.info(f"Retrieved ticket {ticket_id}")

            return {
                "ticket_id": ticket.ticket_id,
                "channel": ticket.channel,
                "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
                "user": user.user_name if user else "unknown",
                "status": meta.status if meta else None,
                "issue_type": meta.main_issue_type if meta else None,
                "tags": meta.tags if meta else None,
                "messages": messages
            }

    except Exception as e:
        logger.error(f"Error retrieving ticket {ticket_id}: {e}")
        return {"error": f"An error occurred while retrieving ticket {ticket_id}."}


@mcp.tool()
def update_ticket_status(
    ticket_id: str,
    status: str,
    issue_type: str = "",
    tags: str = "",
) -> Dict:
    """
    Update the status and classification metadata of a ticket.

    Args:
        ticket_id: The UUID of the ticket to update.
        status: New status value. Valid values: 'open', 'in_progress', 'resolved', 'escalated', 'closed'.
        issue_type: Main issue category (e.g. 'login', 'billing', 'reservation', 'subscription').
        tags: Comma-separated tags to add or replace (optional).

    Returns:
        A dictionary confirming the update or describing an error.
        {
            "ticket_id": str,
            "status": str,
            "issue_type": str,
            "tags": str,
            "error": str - An error message if the update fails
        }
    """
    try:
        with get_session(engine) as session:
            meta = session.query(udahub.TicketMetadata).filter_by(ticket_id=ticket_id).first()

            if not meta:
                return {"error": f"No ticket metadata found for ticket_id: {ticket_id}"}

            meta.status = status

            if issue_type:
                meta.main_issue_type = issue_type

            if tags:
                existing = set(meta.tags.split(",")) if meta.tags else set()
                new_tags = {t.strip() for t in tags.split(",")}
                meta.tags = ", ".join(sorted(existing | new_tags))

            logger.info(f"Updated ticket {ticket_id} status to '{status}'")

            return {
                "ticket_id": ticket_id,
                "status": meta.status,
                "issue_type": meta.main_issue_type,
                "tags": meta.tags
            }

    except Exception as e:
        logger.error(f"Error updating ticket {ticket_id}: {e}")
        return {"error": f"An error occurred while updating ticket {ticket_id}."}


@mcp.tool()
def add_ticket_message(ticket_id: str, content: str, role: str = "agent") -> Dict:
    """
    Append a new message to a ticket's conversation thread.

    Args:
        ticket_id: The UUID of the ticket.
        content: Message text to append.
        role: Who is sending the message — 'agent', 'user', or 'system' (default: 'agent').

    Returns:
        A dictionary confirming the message was added or describing an error.
        {
            "ticket_id": str,
            "role": str,
            "message_id": str,
            "error": str - An error message if the operation fails
        }
    """
    try:
        with get_session(engine) as session:

            role_lower = role.lower()
            if role_lower not in ("agent", "user", "system"):
                role_lower = "agent"

            role_enum = udahub.RoleEnum[role_lower]
            message_id = str(uuid.uuid4())

            message = udahub.TicketMessage(
                message_id=message_id,
                ticket_id=ticket_id,
                role=role_enum,
                content=content,
            )

            session.add(message)

            logger.info(f"Added message to ticket {ticket_id} as role='{role_lower}'")
            return {
                "ticket_id": ticket_id,
                "role": role_lower,
                "message_id": message_id
            }

    except Exception as e:
        logger.error(f"Error adding message to ticket {ticket_id}: {e}")
        return {"error": f"An error occurred while adding a message to ticket {ticket_id}."}
