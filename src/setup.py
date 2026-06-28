"""
UIT RAG System - Load chunks & Embedding.

Pipeline:
    1. load_chunks()      - Đọc tất cả chunks từ chunk/<category>/*.jsonl
    2. embed_chunks()     - Embed dense vector cho mỗi chunk
    3. save_embeddings() - Lưu kết quả ra JSONL

Chạy:
    python -m src.setup
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from pymilvus import model

from src.config import (
    CHUNK_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    OUTPUT_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Map tên folder trên disk -> label chuẩn
_FOLDER_TO_LABEL = {
    "tuyen_sinh": "tuyensinh",
    "chuong_trinh_dao_tao": "chuongtrinhdaotao",
    "danh_muc_mon_hoc": "danhmucmonhoc",
    "cong_tac_sinh_vien": "hoatdong",
    "nhom_nghien_cuu": "nghiencuu",
}

# Map category thô trong file -> label chuẩn
_CATEGORY_NORMALIZE = {
    "chuong_trinh_dao_tao": "chuongtrinhdaotao",
    "cau_lac_bo": "hoatdong",
    "sinh_vien": "hoatdong",
}


def _normalize_category(raw: Optional[str], folder: str) -> str:
    """Trả về label chuẩn từ raw value hoặc folder name."""
    if raw:
        mapped = _CATEGORY_NORMALIZE.get(str(raw))
        if mapped:
            return mapped
        if str(raw) in _FOLDER_TO_LABEL.values():
            return str(raw)
    return _FOLDER_TO_LABEL.get(folder, folder)


def _extract_major(data: dict, metadata: dict) -> str:
    """Extract major value (string) từ data hoặc metadata."""
    major_field = data.get("major")
    if isinstance(major_field, list) and major_field:
        return str(major_field[0])
    if isinstance(major_field, str) and major_field:
        return major_field

    metadata_majors = metadata.get("majors")
    if isinstance(metadata_majors, list) and metadata_majors:
        return str(metadata_majors[0])

    return ""


def load_chunks(chunk_dir: Path = CHUNK_DIR) -> list[dict[str, Any]]:
    """Load chunks từ tất cả category folders.

    Hỗ trợ 2 schema:
      - Schema A (chuong_trinh_dao_tao, tuyen_sinh): top-level fields
      - Schema B (còn lại): wrapper metadata
    """
    chunks: list[dict[str, Any]] = []
    for category_folder in chunk_dir.iterdir():
        if not category_folder.is_dir():
            continue
        folder = category_folder.name
        for jsonl_file in category_folder.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    metadata = data.get("metadata", {}) or {}
                    raw_year = data.get("year", metadata.get("year"))
                    chunks.append({
                        "chunk_id": data.get("chunk_id", ""),
                        "content": data.get("chunk_content", data.get("content", "")),
                        "category": _normalize_category(
                            data.get("category") or metadata.get("category"),
                            folder,
                        ),
                        "year": raw_year if raw_year is not None else 0,
                        "major": _extract_major(data, metadata),
                        "parent_document": data.get("parent_document", ""),
                    })
    logger.info(f"Loaded {len(chunks)} chunks from {chunk_dir}")
    return chunks


def load_embedding_model():
    """Load Sentence Transformer embedding model."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}...")
    return model.dense.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE,
    )


def embed_chunks(
    chunks: list[dict[str, Any]],
    embed_model,
    batch_size: int = 32,
) -> list[list[float]]:
    """Encode content của tất cả chunks thành dense vectors theo batch."""
    texts = [c["content"] for c in chunks]
    all_vectors: list[list[float]] = []

    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        logger.info(
            f"Embedding batch {i // batch_size + 1}/"
            f"{(total + batch_size - 1) // batch_size}..."
        )
        dense_vectors = embed_model.encode_documents(batch_texts)
        batch_list = (
            dense_vectors.tolist()
            if hasattr(dense_vectors, "tolist")
            else dense_vectors
        )
        all_vectors.extend(batch_list)

    logger.info(f"Embedded {len(all_vectors)} chunks")
    return all_vectors


def _to_list(value: Any) -> Any:
    """Convert numpy arrays/lists to pure Python lists recursively."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return [_to_list(v) for v in value]
    return value


def save_embeddings(
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    output_path: Path,
) -> None:
    """Lưu chunks + dense vectors ra file JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk, vector in zip(chunks, vectors):
            row = {**chunk, "dense_vector": _to_list(vector)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(chunks)} embedded chunks -> {output_path}")


def main() -> None:
    """Chạy full pipeline: load -> embed -> save."""
    chunks = load_chunks()
    embed_model = load_embedding_model()
    vectors = embed_chunks(chunks, embed_model)

    output_path = OUTPUT_DIR / "embedded_chunks.jsonl"
    save_embeddings(chunks, vectors, output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()