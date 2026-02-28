
import os
import sys
import json
import logging
from typing import Dict

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from settings import settings
from agentic.tools.tools_mcp_server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
def search_knowledge_base(query: str) -> Dict:
    """Search the CultPass knowledge base for articles relevant to the query using vector similarity search.

    Args:
        query: Natural-language question or keywords (e.g. 'cancel subscription').

    Returns:
        A dictionary with the most relevant knowledge articles (up to 3).
        {
            "articles": list - A list of matching articles with title, content, and tags
            "error": str - An error message if an exception occurs
        }
    """
    try:
        embeddings = OpenAIEmbeddings(
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )

        vector_store = Chroma(
            collection_name="udahub_knowledge",
            embedding_function=embeddings,
            persist_directory=settings.knowledge_chroma_db_path
        )

        results = vector_store.similarity_search(query, k=3)

        if not results:
            return {"articles": [], "message": f"No knowledge articles found for the given query."}

        articles = [json.loads(doc.page_content) for doc in results]

        logger.info(f"Found {len(articles)} knowledge articles matching query '{query}'")
        return {"articles": articles}

    except Exception as e:
        logger.error(f"Error searching knowledge base for query '{query}': {e}")
        return {"error": f"An error occurred while searching the knowledge base."}
