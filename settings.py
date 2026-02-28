

import os
from dataclasses import dataclass

@dataclass
class Settings:
    """Settings for the application."""

    # Parent path of the project (assumes this file is in the root)
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    # Database paths
    cultpass_db_path = os.path.join(parent_dir, "data", "external", "cultpass.db")
    udahub_db_path = os.path.join(parent_dir, "data", "core", "udahub.db")

    # Chroma DB settings directories
    experience_chroma_db_path = os.path.join(parent_dir, "data", "external", "chroma_experiences", "exp.sqlite3")
    knowledge_chroma_db_path = os.path.join(parent_dir, "data", "core", "chroma_knowledge", "kb.sqlite3")

    # Json file paths for experiences and knowledge base (used for initial vector store population)
    experiences_json_path = os.path.join(parent_dir, "data", "external", "cultpass_experiences.jsonl")
    knowledge_json_path = os.path.join(parent_dir, "data", "external", "cultpass_articles.jsonl")


settings = Settings()