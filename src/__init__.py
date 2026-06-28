"""UIT RAG System - Main package."""

from src.config import (
    CHUNK_DIR,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    OUTPUT_DIR,
    PROJECT_ROOT,
)
from src.llm_utils import get_llm_client, parse_json_safely
from src.pipeline import RAGPipeline, PipelineResult
from src.rag_utils import get_chunk_database, COMMON_MAJORS

__all__ = [
    # Config
    "PROJECT_ROOT",
    "CHUNK_DIR",
    "OUTPUT_DIR",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    # LLM
    "get_llm_client",
    "parse_json_safely",
    # RAG
    "get_chunk_database",
    "COMMON_MAJORS",
    # Pipeline
    "RAGPipeline",
    "PipelineResult",
]
