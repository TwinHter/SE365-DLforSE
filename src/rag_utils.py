"""
RAG Utilities - Load chunks và Retrieval.
Xử lý việc đọc chunks đã embed và vector search.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import EMBEDDING_DIMENSION, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Category mapping từ file -> chuẩn
_FOLDER_TO_LABEL = {
    "tuyen_sinh": "tuyensinh",
    "chuong_trinh_dao_tao": "chuongtrinhdaotao",
    "danh_muc_mon_hoc": "danhmucmonhoc",
    "cong_tac_sinh_vien": "hoatdong",
    "nhom_nghien_cuu": "nghiencuu",
}

# Common majors - fallback khi major = null hoặc không xác định
COMMON_MAJORS = ["KHMT", "ATTT", "CNPM", "HTTT", "KTMT", "MMT", "CNTT", "CN"]

# Broad categories - những category nên include trong hard filter
# Bao gồm cả null và "chung" vì chunk có thể thuộc nhiều category hoặc không xác định
BROAD_CATEGORIES = ["chung", "tuyensinh", "chuongtrinhdaotao", "danhmucmonhoc", "hoatdong", "nghiencuu", ""]

EMBEDDING_FILE = OUTPUT_DIR / "embedded_chunks.jsonl"


class ChunkDatabase:
    """In-memory database cho chunks đã embed."""

    def __init__(self):
        self.chunks: list[dict[str, Any]] = []
        self.vectors: np.ndarray | None = None
        self.bm25: BM25Search | None = None
        self._loaded = False

    def load(self) -> int:
        """Load all embedded chunks from JSONL file."""
        if self._loaded:
            return len(self.chunks)

        if not EMBEDDING_FILE.exists():
            logger.error(f"Embedding file not found: {EMBEDDING_FILE}")
            return 0

        chunks = []
        vectors = []

        logger.info(f"Loading chunks from {EMBEDDING_FILE}...")

        with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # Extract content
                    content = data.get("content", "") or data.get("chunk_content", "")
                    if not content:
                        continue

                    # Extract vector
                    vector = data.get("dense_vector", [])
                    if not vector:
                        continue

                    # Extract category - chuẩn hóa
                    raw_category = data.get("category", "")
                    category = raw_category if raw_category else "chung"

                    chunks.append({
                        "chunk_id": data.get("chunk_id", f"chunk_{line_num}"),
                        "content": content,
                        "category": category,
                        "year": data.get("year", 0),
                        "major": data.get("major", ""),  # Can be string or list
                        "parent_document": data.get("parent_document", ""),
                    })
                    vectors.append(vector)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        self.chunks = chunks
        self.vectors = np.array(vectors, dtype=np.float32) if vectors else None

        self.bm25 = BM25Search(chunks)
        logger.info(f"Built BM25 index with {len(chunks)} documents")

        logger.info(f"Loaded {len(self.chunks)} chunks with vectors shape: {self.vectors.shape if self.vectors is not None else 'None'}")
        self._loaded = True

        return len(self.chunks)

    def _normalize_major(self, major: str) -> str:
        """Chuẩn hóa major code."""
        if not major:
            return ""
        
        # Handle list format
        if isinstance(major, list):
            if not major:
                return ""
            major = major[0] if major else ""
        
        major = str(major).strip().upper()
        
        # Map alias -> chuẩn
        major_aliases = {
            "CÔNG NGHỆ THÔNG TIN": "CNTT",
            "KHOA HỌC MÁY TÍNH": "KHMT",
            "AN TOÀN THÔNG TIN": "ATTT",
            "KỸ THUẬT PHẦN MỀM": "CNPM",
            "HỆ THỐNG THÔNG TIN": "HTTT",
            "KỸ THUẬT MÁY TÍNH": "KTMT",
            "MẠNG MÁY TÍNH": "MMT",
            "TRÍ TUỆ NHÂN TẠO": "TTNT",
            "KHOA HỌC VÀ KỸ THUẬT THÔNG TIN": "KTTT",
            "CHUNG UIT": "CHUNG",
        }
        
        for alias, code in major_aliases.items():
            if alias in major:
                return code
        
        return major

    def hard_filter(
        self,
        category: str | None = None,
        year: int | None = None,
        major: str | None = None,
        keywords: list[str] | None = None,
    ) -> list[int]:
        """
        Apply hard filters to get candidate chunk indices.
        
        Returns:
            List of indices that match all filters
        """
        if not self._loaded:
            self.load()

        candidate_indices = list(range(len(self.chunks)))

        # Check if query is about tuition vs scholarship (should not filter by major)
        is_tuition_query = False
        is_scholarship_query = False
        if keywords:
            keywords_text = " ".join(keywords).lower()
            tuition_keywords = ["học phí"]
            scholarship_keywords = ["học bổng", "miễn giảm"]
            is_tuition_query = any(kw in keywords_text for kw in tuition_keywords)
            is_scholarship_query = any(kw in keywords_text for kw in scholarship_keywords)

        # Determine if we should skip major filter for tuition queries
        skip_major_filter = is_tuition_query and not is_scholarship_query
        
        if is_tuition_query:
            logger.info(f"Tuition query detected (keywords={keywords}), will skip major filter")
        
        # Calculate year_range early for reuse
        year_range = None
        if year:
            year_range = range(year - 2, year + 3)

        # Normalize major early for reuse in fallbacks
        major_normalized = None
        if major and not skip_major_filter:
            major_normalized = self._normalize_major(major)

        # Filter by category - LUÔN BAO GỒM "chung" và "" (null)
        if category and category not in ["chung", ""]:
            candidate_indices = [
                i for i in candidate_indices
                if self.chunks[i]["category"] in [category, "chung", ""]
            ]
            logger.info(f"After category filter '{category}' (including chung/null): {len(candidate_indices)} candidates")
            # DEBUG: List some candidates after category filter
            sample_cats = [(i, self.chunks[i]["chunk_id"], self.chunks[i]["category"], self.chunks[i]["year"]) 
                          for i in candidate_indices[:10]]
            logger.debug(f"  Sample after category: {sample_cats}")
        else:
            # Nếu không specify category, lấy tất cả
            logger.info(f"No category filter, total candidates: {len(candidate_indices)}")

        # Filter by year - BAO GỒM cả nearby years vì chunk metadata thường là năm tạo document,
        # không phải năm được đề cập trong nội dung (VD: doc 2025 có thể chứa thông tin học phí 2026)
        if year:
            filtered = []
            for i in candidate_indices:
                chunk_year = self.chunks[i]["year"]
                if chunk_year in year_range or chunk_year == 0:
                    filtered.append(i)
            candidate_indices = filtered
            logger.info(f"After year filter '{year}' (range ±2, including year=0): {len(candidate_indices)} candidates")

        # Filter by major - SPECIAL CASE: tuition queries skip major filter entirely
        if major and not skip_major_filter and major_normalized:
            if major_normalized not in ["", "NULL", "NONE"]:
                # Major cụ thể được chỉ định - LẤY cả major đó + chunks có major = "Chung UIT" hoặc "chung"
                include_majors = {major_normalized, "CHUNG UIT"}
                logger.info(f"Major '{major}' normalized to '{major_normalized}', including: {include_majors} + chung")
            else:
                # Major null - lấy tất cả (không filter theo major)
                include_majors = None
                logger.info(f"Major is null, no major filter applied")
            
            if include_majors is not None:
                filtered = []
                for i in candidate_indices:
                    chunk = self.chunks[i]
                    chunk_major = chunk.get("major", "")
                    
                    # Handle both string and list formats
                    if isinstance(chunk_major, list):
                        # For list format like ["Chung UIT"], check if any element matches
                        chunk_major_upper = [m.upper() for m in chunk_major if m]
                    elif chunk_major:
                        chunk_major_upper = [chunk_major.upper()]
                    else:
                        chunk_major_upper = []
                    
                    # Check if major matches or is a general/utility major
                    # Include: major match OR is a general utility major OR major is null/empty (general info)
                    is_general = any(m in ["CHUNG", "CHUNG UIT", "COMMON", "ALL"] for m in chunk_major_upper)
                    is_null_major = len(chunk_major_upper) == 0  # null/None means general info
                    
                    if is_general or is_null_major or any(m in include_majors for m in chunk_major_upper):
                        filtered.append(i)
                
                candidate_indices = filtered
                logger.info(f"After major filter: {len(candidate_indices)} candidates (majors: {include_majors} + chung/CHUNG UIT)")
                
                # Kiểm tra xem có chunk nào match exact với major không
                has_exact_major_match = False
                for i in candidate_indices:
                    chunk_major = self.chunks[i].get("major", "")
                    if isinstance(chunk_major, list):
                        chunk_majors_upper = [m.upper() for m in chunk_major if m]
                    elif chunk_major:
                        chunk_majors_upper = [chunk_major.upper()]
                    else:
                        chunk_majors_upper = []
                    
                    # Exact match (không phải CHUNG)
                    if any(m == major_normalized for m in chunk_majors_upper):
                        has_exact_major_match = True
                        break
                
                # FALLBACK 1: Không có exact match với major trong tuyensinh
                # Tìm thêm trong CTĐT và danhmucmonhoc (nơi thông tin ngành thường nằm)
                if not has_exact_major_match and category in ["tuyensinh", "chuongtrinhdaotao", "chung"]:
                    logger.info(f"No exact major match for '{major_normalized}' in tuyensinh, searching in other categories...")
                    
                    expanded_candidates = set(candidate_indices)
                    
                    # Tìm trong CTĐT
                    for i in range(len(self.chunks)):
                        if self.chunks[i]["category"] == "chuongtrinhdaotao":
                            chunk_major = self.chunks[i].get("major", "")
                            if isinstance(chunk_major, list):
                                chunk_major_upper = [m.upper() for m in chunk_major if m]
                            elif chunk_major:
                                chunk_major_upper = [chunk_major.upper()]
                            else:
                                chunk_major_upper = []
                            
                            is_general = any(m in ["CHUNG", "CHUNG UIT", "COMMON", "ALL"] for m in chunk_major_upper)
                            if is_general or any(m == major_normalized for m in chunk_major_upper):
                                expanded_candidates.add(i)
                    
                    # Tìm trong danhmucmonhoc (quan trọng cho thông tin ngành!)
                    for i in range(len(self.chunks)):
                        if self.chunks[i]["category"] == "danhmucmonhoc":
                            chunk_major = self.chunks[i].get("major", "")
                            if isinstance(chunk_major, list):
                                chunk_major_upper = [m.upper() for m in chunk_major if m]
                            elif chunk_major:
                                chunk_major_upper = [chunk_major.upper()]
                            else:
                                chunk_major_upper = []
                            
                            # Include nếu major match
                            if any(m == major_normalized for m in chunk_major_upper):
                                expanded_candidates.add(i)
                    
                    if len(expanded_candidates) > len(candidate_indices):
                        candidate_indices = list(expanded_candidates)
                        logger.info(f"Expanded to {len(candidate_indices)} candidates from CTĐT and danhmucmonhoc")
        else:
            if skip_major_filter:
                logger.info(f"Tuition query - skipping major filter to include all programs")
            else:
                logger.info(f"No major filter, total candidates: {len(candidate_indices)}")

        # Filter by keywords (content must contain at least one keyword)
        if keywords:
            filtered = []
            for i in candidate_indices:
                content = self.chunks[i]["content"].lower()
                if any(kw.lower() in content for kw in keywords):
                    filtered.append(i)
            candidate_indices = filtered
            logger.info(f"After keyword filter: {len(candidate_indices)} candidates")
            
            # FALLBACK 2: Nếu keyword filter làm giảm quá nhiều kết quả,
            # tìm thêm chunks từ CTĐT và danhmucmonhoc cho major cụ thể (không cần keyword)
            if len(candidate_indices) < 3 and major and category in ["tuyensinh", "chuongtrinhdaotao", "chung"]:
                logger.info(f"Too few results ({len(candidate_indices)}), expanding search for major '{major}'...")
                
                expanded = set(candidate_indices)
                
                # Tìm thêm trong CTĐT và danhmucmonhoc cho major cụ thể (không keyword)
                for i in range(len(self.chunks)):
                    if self.chunks[i]["category"] in ["chuongtrinhdaotao", "danhmucmonhoc"]:
                        # Check year
                        chunk_year = self.chunks[i]["year"]
                        if year and chunk_year not in year_range and chunk_year != 0:
                            continue
                        
                        # Check major
                        chunk_major = self.chunks[i].get("major", "")
                        if isinstance(chunk_major, list):
                            chunk_major_upper = [m.upper() for m in chunk_major if m]
                        elif chunk_major:
                            chunk_major_upper = [chunk_major.upper()]
                        else:
                            chunk_major_upper = []
                        
                        # Include nếu exact match với major hoặc là general
                        is_general = any(m in ["CHUNG", "CHUNG UIT", "COMMON", "ALL"] for m in chunk_major_upper)
                        if is_general or any(m == major_normalized for m in chunk_major_upper):
                            # Apply keyword filter trên chunks mới
                            content = self.chunks[i]["content"].lower()
                            if any(kw.lower() in content for kw in keywords):
                                expanded.add(i)
                
                if len(expanded) > len(candidate_indices):
                    candidate_indices = list(expanded)
                    logger.info(f"After expansion: {len(candidate_indices)} candidates")

        return candidate_indices

    def vector_search(
        self,
        query_vector: list[float],
        candidate_indices: list[int] | None = None,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Search by cosine similarity.
        
        Returns:
            List of (chunk_index, score) tuples sorted by similarity
        """
        if self.vectors is None:
            return []

        query_vec = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        # Normalize query vector
        query_vec_normalized = query_vec / query_norm

        results = []

        indices_to_search = candidate_indices if candidate_indices else range(len(self.chunks))

        for i in indices_to_search:
            chunk_vec = self.vectors[i]
            chunk_norm = np.linalg.norm(chunk_vec)

            if chunk_norm == 0:
                continue

            # Cosine similarity
            similarity = np.dot(query_vec_normalized, chunk_vec / chunk_norm)

            results.append((int(i), float(similarity)))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_chunk(self, index: int) -> dict[str, Any] | None:
        """Get chunk by index."""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None

    def get_chunks_by_indices(self, indices: list[int]) -> list[dict[str, Any]]:
        """Get multiple chunks by indices."""
        return [self.chunks[i] for i in indices if 0 <= i < len(self.chunks)]


class BM25Search:
    """BM25 sparse search for keyword-based retrieval."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.corpus: list[list[str]] = []
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Simple Vietnamese tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r"\w+", text, re.UNICODE)
        return tokens

    def _build_index(self):
        """Build BM25 inverted index."""
        import math

        self.corpus = [self._tokenize(chunk["content"]) for chunk in self.chunks]
        self.corpus_size = len(self.corpus)

        for document in self.corpus:
            freq: dict[str, int] = {}
            for word in document:
                freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.append(freq)

        for document in self.corpus:
            for word in set(document):
                self.idf[word] = self.idf.get(word, 0) + 1

        for word in self.idf:
            self.idf[word] = math.log(
                (self.corpus_size - self.idf[word] + 0.5) / (self.idf[word] + 0.5) + 1
            )

        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.doc_len else 0

    def _calc_bm25(
        self, query_terms: list[str], doc_idx: int, k1: float = 1.5, b: float = 0.75
    ) -> float:
        """Calculate BM25 score for a document."""
        import math

        score = 0.0
        doc_freqs = self.doc_freqs[doc_idx]
        doc_len = self.doc_len[doc_idx]

        for term in query_terms:
            if term not in doc_freqs:
                continue

            freq = doc_freqs[term]
            idf = self.idf.get(term, 0)

            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def search(
        self,
        query: str,
        candidate_indices: list[int] | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Search using BM25.

        Returns:
            List of (chunk_index, bm25_score) tuples sorted by score
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        indices_to_search = candidate_indices if candidate_indices else range(self.corpus_size)

        results: list[tuple[int, float]] = []
        for idx in indices_to_search:
            score = self._calc_bm25(query_terms, idx)
            if score > 0:
                results.append((int(idx), float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def reciprocal_rank_fusion(
    results_list: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion - combine multiple ranked lists.

    Args:
        results_list: List of ranked results [(chunk_idx, score), ...]
        k: RRF parameter (default 60)

    Returns:
        Fused ranked list
    """
    scores: dict[int, float] = {}

    for results in results_list:
        for rank, (idx, _) in enumerate(results):
            rrf_score = 1 / (k + rank + 1)
            scores[idx] = scores.get(idx, 0) + rrf_score

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class CrossEncoderReranker:
    """Cross-Encoder reranker for re-ranking retrieved chunks."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self._loaded = False

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
        self._loaded = True

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Re-rank chunks using cross-encoder scores.

        Args:
            query: The search query
            chunks: List of chunk dictionaries
            top_k: Number of top results to return

        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        self._load_model()

        if not chunks:
            return []

        if self.model is None:
            return [(c, 0.0) for c in chunks[:top_k]]

        pairs = [(query, chunk["content"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [float(s) for s in scores]

        results = list(zip(chunks, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


# Global instance
_db: ChunkDatabase | None = None


def get_chunk_database() -> ChunkDatabase:
    """Get or create the global chunk database."""
    global _db
    if _db is None:
        _db = ChunkDatabase()
    return _db
