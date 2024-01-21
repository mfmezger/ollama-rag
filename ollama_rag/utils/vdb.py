"""The script to initialize the Qdrant db backend with aleph alpha."""

import os
from typing import Optional

from llama_index.embeddings import OllamaEmbedding
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient

from aleph_alpha_rag.utils.configuration import load_config


@load_config(location="config/db.yml")
def get_db_connection(cfg: DictConfig, collection_name: Optional[str] = None) -> Qdrant:
    """Initializes a connection to the Qdrant DB.

    Args:
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        Qdrant: The Qdrant DB connection.
    """
    embedding = AlephAlphaAsymmetricSemanticEmbedding(
        model=cfg.aleph_alpha_embeddings.model_name,
        normalize=cfg.aleph_alpha_embeddings.normalize,
        compress_to_size=cfg.aleph_alpha_embeddings.compress_to_size,
    )
    qdrant_client = QdrantClient(
        cfg.qdrant.url,
        port=cfg.qdrant.port,
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=cfg.qdrant.prefer_grpc,
    )

    if collection_name is None or collection_name == "":
        collection_name = cfg.qdrant.collection_name_aa

    logger.info(f"USING COLLECTION: {collection_name}")

    vector_db = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db
