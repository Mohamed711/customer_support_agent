
import os
import sys

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from agentic.tools.knowledge_tools import search_knowledge_base


def test_search_cancel_subscription():
    """Happy path: query matches a knowledge article — list of articles is returned."""

    result = search_knowledge_base("how to cancel my subscription")
    print(result)

    assert isinstance(result.get("articles"), list)
    assert len(result.get("articles")) > 0

    titles = [article.get("title") for article in result.get("articles")]
    assert "How to Cancel or Pause a Subscription" in titles
