"""UIT RAG System - Load chunks & Embedding pipeline."""

from src.config import CHUNK_DIR, EMBEDDING_DIMENSION, EMBEDDING_MODEL, OUTPUT_DIR
from src.setup import (
    embed_chunks,
    load_chunks,
    load_embedding_model,
    save_embeddings,
)

__all__ = [
    "CHUNK_DIR",
    "OUTPUT_DIR",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "load_chunks",
    "load_embedding_model",
    "embed_chunks",
    "save_embeddings",
]