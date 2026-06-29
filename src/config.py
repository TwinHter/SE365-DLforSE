"""
Configuration module for UIT RAG System.
Centralizes all configuration settings.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CHUNK_DIR = PROJECT_ROOT / "chunk"
OUTPUT_DIR = PROJECT_ROOT / "embedded"

# Embedding model
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_DIMENSION = 768

# Milvus configuration
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "rag_chunks"