
import os
import sys
import logging
from typing import Dict
from sqlalchemy import create_engine

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from utils import get_session
from data.models import cultpass
from settings import settings

from tools.tools_mcp_server import mcp

logger = logging.getLogger(__name__)

engine = create_engine(f"sqlite:///{settings.cultpass_db_path}", echo=False)

@mcp.tool()
def get_user_general_info(user_id: str) -> Dict[str, str]:
    """
    Retrieve general information about a CultPass user.

    Args:
        user_id: The CultPass user ID

    Returns:
        A dictionary with the user's profile information.

        - user_id: str - The CultPass user ID
        - name: str - The user's full name
        - email: str - The user's email address
        - account_status: Literal["ACTIVE", "BLOCKED"] - The user's account status
        - error: str - An error message if the user is not found or an exception occurs
    """
    try:
        with get_session(engine) as session:
            user = session.query(cultpass.User).filter_by(user_id=user_id).first()

            if not user:
                return {"error": f"No CultPass user found with id: {user_id}"}

            blocked_flag = "BLOCKED" if user.is_blocked else "ACTIVE"

            info = {
                "user_id": user.user_id,
                "name": user.full_name,
                "email": user.email,
                "account_status": blocked_flag
            }

            logger.info(f"Retrieved general info for user {user_id}")
            return info

    except Exception as e:
        logger.error(f"Error retrieving general info for user {user_id}: {e}")
        return {"error": f"An error occurred while retrieving general info for user {user_id}."}


@mcp.tool()
def get_user_subscription(user_id: str) -> Dict[str, str]:
    """
    Retrieve the subscription status of a CultPass user

    Args:
        user_id: The CultPass user ID

    Returns:
        A dictionary with the user's subscription information.

        - user_id: str - The CultPass user ID
        - subscription_status: str - The user's subscription status or None if not subscribed
        - subscription_tier: str - The user's subscription tier or None if not subscribed
        - subscription_started_at: str - The start date of the user's subscription in ISO format or None if not subscribed
        - subscription_ended_at: str - The end date of the user's subscription in ISO format or None if not subscribed
        - error: str - An error message if the user is not found or an exception occurs
    """

    try:
        with get_session(engine) as session:
            user = session.query(cultpass.User).filter_by(user_id=user_id).first()

            if not user:
                return {"error": f"No CultPass user found with id: {user_id}"}

            # Check for user subscription:
            if user.subscription:
                logger.info(f"User {user_id} subscription status: {user.subscription}")
                return {
                    "user_id": user_id,
                    "subscription_status": user.subscription.status,
                    "subscription_tier": user.subscription.tier,
                    "subscription_started_at": user.subscription.started_at.isoformat(),
                    "subscription_ended_at": user.subscription.ended_at.isoformat() if user.subscription.ended_at else None
                }
            else:
                logger.info(f"User {user_id} has no active subscription.")
                return {"error": f"User {user_id} has no active subscription."}

    except Exception as e:
        logger.error(f"Error retrieving subscription status for user {user_id}: {e}")
        return {"error": f"An error occurred while retrieving subscription status for user {user_id}."}


@mcp.tool()
def get_user_reservations(user_id: str) -> Dict[str, str]:
    """
    Retrieve the reservation history of a CultPass user.

    Args:
        user_id: The CultPass user ID.

    Returns:
        A dictionary with the user's reservation information.
        {
            "user_id": str - The CultPass user ID
            "reservations": list - A list of the user's reservations with experience details and statuses
            "error": str - An error message if the user is not found or an exception occurs
        }
    """
    try:
        with get_session(engine) as session:
            user = session.query(cultpass.User).filter_by(user_id=user_id).first()

            if not user:
                return {"error": f"No CultPass user found with id: {user_id}"}

            if user.reservations:
                logger.info(f"Retrieved {len(user.reservations)} reservations for user {user_id}")
                reservations = list()

                for res in user.reservations:
                    exp = res.experience
                    reservations.append({
                        "experience_title": exp.title if exp else "N/A",
                        "experience_location": exp.location if exp else "N/A",
                        "experience_time": exp.when.isoformat() if exp and exp.when else "N/A",
                        "experience_is_premium": exp.is_premium if exp else False,
                        "reservation_status": res.status
                    })

                return {
                    "user_id": user_id,
                    "reservations": reservations
                }
            else:
                logger.info(f"User {user_id} has no reservations.")
                return {"error": f"User {user_id} has no reservations."}

    except Exception as e:
        logger.error(f"Error retrieving reservations for user {user_id}: {e}")
        return {"error": f"An error occurred while retrieving reservations for user {user_id}."}


@mcp.tool()
def search_experiences_by_keyword(keyword: str) -> Dict[str, str]:
    """
    Search for CultPass experiences matching a keyword in their title or description.

    """
    pass

@mcp.tool()
def get_experience_availability(experience_id: str) -> Dict[str, str]:
    """
    Check the availability and details of a CultPass experience by ID.

    Args:
        experience_id: The ID of the experience to look up.

    Returns:
        A dictionary with the experience details and availability.
        {
            "experience_title": str - The title of the experience
            "experience_location": str - The location of the experience
            "experience_time": str - The date and time of the experience
            "experience_is_premium": bool - Whether the experience is premium
            "error": str - An error message if the experience is not found or an exception occurs
        }
    """
    try:
        with get_session(engine) as session:
            experience = session.query(cultpass.Experience).filter_by(id=experience_id).first()

            if not experience:
                return {"error": f"No experience found with ID: {experience_id}"}

            logger.info(f"Retrieved experience with ID: {experience_id}")

            return {
                "experience_title": experience.title,
                "experience_location": experience.location if experience else "N/A",
                "experience_time": experience.when.isoformat() if experience and experience.when else "N/A",
                "experience_is_premium": experience.is_premium if experience else False
            }

    except Exception as e:
        logger.error(f"Error retrieving experiences for '{experience_id}': {e}")
        return {"error": f"An error occurred while retrieving experience details for '{experience_id}'."}

