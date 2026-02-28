
import os
import sys
import json
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from settings import settings

mcp = FastMCP("CultPass Tools", "Tools for managing CultPass users and experiences.")

# Import tool modules AFTER mcp is defined so @mcp.tool() decorators register correctly
import agentic.tools.cultpass_tools
import agentic.tools.knowledge_tools
import agentic.tools.ticket_tools


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":

    load_dotenv()  # Load environment variables from .env file

    if not os.getenv("VOCAREUM_OPENAPI_KEY"):
        load_dotenv(Path.home() / ".env")  # Try loading from .env.local if not found in .env

    if not os.getenv("VOCAREUM_OPENAPI_KEY"):
        raise ValueError("VOCAREUM_OPENAPI_KEY environment variable is not set. Please set it in .env or .env.local.")

    # Get directory paths for ChromaDB storage
    experience_db_dir = os.path.dirname(settings.experience_chroma_db_path)
    knowledge_db_dir = os.path.dirname(settings.knowledge_chroma_db_path)
    os.makedirs(experience_db_dir, exist_ok=True)
    os.makedirs(knowledge_db_dir, exist_ok=True)

    # Initialize the embeddings
    embeddings = OpenAIEmbeddings(
        base_url="https://openai.vocareum.com/v1",
        api_key=os.getenv("VOCAREUM_OPENAPI_KEY")
    )

    # Create the vector store out of the exist experiences
    if not os.path.exists(settings.experience_chroma_db_path):
        logging.info("Creating ChromaDB vector store for CultPass experiences...")

        with open(settings.experiences_json_path, "r") as f:
            experiences_data = [json.loads(line) for line in f if line.strip()]

        experience_vector_store = Chroma.from_documents(
            documents=[Document(page_content=json.dumps(exp), metadata={"title": exp.get("title")}) for exp in experiences_data],
            collection_name="cultpass_experiences",
            embedding=embeddings,
            persist_directory=settings.experience_chroma_db_path
        )

    # Create the vector store out of the existing knowledge base articles
    if not os.path.exists(settings.knowledge_chroma_db_path):
        logging.info("Creating ChromaDB vector store for UdaHub knowledge base...")

        # Load knowledge base from JSON file
        with open(settings.knowledge_json_path, "r") as f:
            knowledge_data = [json.loads(line) for line in f if line.strip()]

        knowledge_vector_store = Chroma.from_documents(
            documents=[Document(page_content=json.dumps(kb), metadata={"title": kb.get("title")}) for kb in knowledge_data],
            collection_name="udahub_knowledge",
            embedding=embeddings,
            persist_directory=settings.knowledge_chroma_db_path
        )

    mcp.run(transport="stdio")