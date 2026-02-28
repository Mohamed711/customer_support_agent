
import os
import sys
import pytest

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from agentic.tools.cultpass_tools import (
    get_user_general_info,
    get_user_subscription,
    get_user_reservations,
    get_experience_availability,
    search_experiences_by_keyword
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_subscribed_user_returns_status():
    """Happy path: user exists and has a subscription — status string is returned."""

    result = get_user_general_info("a4ab87")

    assert result.get("email") == "alice.kingsley@wonderland.com"
    assert result.get("account_status") == "BLOCKED"


def test_check_subscription():
    """Happy path: user exists and has a subscription — status string is returned."""

    result = get_user_subscription("a4ab87")

    assert result.get("subscription_status").lower() == "active"
    assert result.get("subscription_tier").lower() == "premium"


def test_check_reservations():
    """Happy path: user exists and has a subscription — status string is returned."""

    result = get_user_reservations("a4ab87")

    assert isinstance(result.get("reservations"), list)
    assert len(result.get("reservations")) > 0
    assert result.get("reservations")[0].get("experience_title") in [
        "Pelourinho Colonial Walk"]


def test_check_experience_availability():
    """Happy path: experience exists — availability info is returned."""

    result = get_experience_availability("63250a")
    assert result.get("slots_available") > 5


def test_search_experiences_by_keyword():
    """Happy path: keyword matches experiences — list of matching experiences is returned."""

    result = search_experiences_by_keyword("Dance samba")
    print(result)

    assert isinstance(result.get("experiences"), list)
    assert len(result.get("experiences")) > 0

