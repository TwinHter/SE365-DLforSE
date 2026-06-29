"""Script để embedding all chunks vao file JSONL."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import logging
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_DIRS = [
    Path("chunk/chuong_trinh_dao_tao"),
    Path("chunk/tuyen_sinh"),
    Path("chunk/nhom_nghien_cuu"),
    Path("chunk/danh_muc_mon_hoc"),
    Path("chunk/cong_tac_sinh_vien"),
]

EMBEDDING_FILE = OUTPUT_DIR / "embedded_chunks.jsonl"

def load_all_chunks():
    """Load all chunks from all directories."""
    all_chunks = []
    
    for chunk_dir in CHUNK_DIRS:
        if not chunk_dir.exists():
            logger.warning(f"Directory not found: {chunk_dir}")
            continue
        
        jsonl_files = list(chunk_dir.glob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} files in {chunk_dir}")
        
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # Normalize fields
                            chunk = {
                                "chunk_id": data.get("chunk_id") or data.get("id", ""),
                                "content": data.get("content") or data.get("chunk_content", ""),
                                "category": data.get("category", chunk_dir.name),
                                "year": data.get("year", 0),
                                "major": data.get("major", ""),
                                "parent_document": data.get("parent_document", ""),
                            }
                            all_chunks.append(chunk)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {jsonl_file}")
    
    logger.info(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks

def create_embeddings(chunks):
    """Create embeddings for all chunks."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    embeddings = []
    texts = [c["content"] for c in chunks]
    
    logger.info(f"Creating embeddings for {len(texts)} chunks...")
    
    # Process in batches
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        embeddings.extend(batch_embeddings.tolist())
    
    return embeddings

def main():
    # Load chunks
    print("Dang load chunks...")
    chunks = load_all_chunks()
    
    if not chunks:
        print("Khong co chunk nao de embedding!")
        return
    
    # Create embeddings
    print("Dang tao embeddings...")
    embeddings = create_embeddings(chunks)
    
    # Save to JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dang luu vao {EMBEDDING_FILE}...")
    
    with open(EMBEDDING_FILE, "w", encoding="utf-8") as f:
        for chunk, embedding in zip(chunks, embeddings):
            record = {**chunk, "embedding": embedding}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Da luu {len(chunks)} chunks voi embeddings vao {EMBEDDING_FILE}")

if __name__ == "__main__":
    main()
