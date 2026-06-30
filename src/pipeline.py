"""
RAG Pipeline - Orchestrate the full RAG flow.
Handles: rephrase -> extract JSON A -> hard filter -> retrieval -> rerank -> answer -> support check.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
)
from src.llm_utils import (
    EXTRACT_JSON_A_PROMPT,
    GENERATE_ANSWER_PROMPT,
    GENERATE_ANSWER_NORMAL_PROMPT,
    REPHRASE_PROMPT,
    EVALUATE_RETRIEVAL_PROMPT,
    RETRY_REPHRASE_PROMPT,
    get_llm_client,
    parse_json_safely,
)
from src.rag_utils import (
    CrossEncoderReranker,
    get_chunk_database,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


# ============================================================
# Embedding Model Setup
# ============================================================
_embedding_model = None
_embedding_tokenizer = None


def _get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model, _embedding_tokenizer
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except ImportError as e:
            logger.warning(f"sentence-transformers not installed: {e}")
            _embedding_model = None
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            _embedding_model = None
    return _embedding_model


def generate_embedding(text: str) -> list[float]:
    """Generate embedding for text using the configured model."""
    model = _get_embedding_model()
    if model is not None:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    else:
        # Fallback: return zero vector
        return [0.0] * EMBEDDING_DIMENSION


@dataclass
class PipelineStep:
    """Represents a single step in the pipeline with logs."""
    name: str
    status: str = "pending"  # pending, running, done, error
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    log_messages: list[str] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0
    error: str = ""

    def add_log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {msg}")

    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0


@dataclass
class PipelineResult:
    """Result of the full pipeline."""
    success: bool
    answer: str = ""
    explanation: str = ""
    support_context: str = ""
    is_supported: bool = False
    support_confidence: float = 0.0
    is_about_uit: bool = True
    steps: list[PipelineStep] = field(default_factory=list)
    error: str = ""

    def get_all_logs(self) -> list[tuple[str, str]]:
        """Get all logs from all steps."""
        logs = []
        for step in self.steps:
            logs.append((step.name, f"=== {step.name} ({step.status}) ==="))
            logs.extend((step.name, msg) for msg in step.log_messages)
            if step.error:
                logs.append((step.name, f"ERROR: {step.error}"))
        return logs


class RAGPipeline:
    """Full RAG pipeline with step-by-step execution."""

    def __init__(self, disable_hard_filter: bool = False, no_rag: bool = False, disable_self_rag: bool = False):
        self.llm = get_llm_client()
        self.db = get_chunk_database()
        self.disable_hard_filter = disable_hard_filter
        self.no_rag = no_rag
        self.disable_self_rag = disable_self_rag
        self.result = PipelineResult(success=False)

    @staticmethod
    def _find_out_of_scope_option(options: dict[str, str]) -> str | None:
        """Tìm option letter mang nhãn 'không liên quan UIT / ngoài phạm vi'.

        Trả về letter (vd 'E') hoặc None nếu không tìm thấy.
        Match không phân biệt hoa thường, bỏ dấu (Unicode NFKD).
        """
        import re
        import unicodedata

        def strip_vi(s: str) -> str:
            nfkd = unicodedata.normalize("NFKD", s)
            ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
            return re.sub(r"[^a-z0-9]+", " ", ascii_only.lower()).strip()

        keywords_ascii = {
            "khong lien quan",      # không liên quan
            "khong thuoc",          # không thuộc
            "ngoai pham vi",        # ngoài phạm vi
            "khong lien quan den uit",
            "khong lien quan giao duc",  # không liên quan giáo dục
        }
        for letter, text in options.items():
            normalized = strip_vi(text)
            tokens = normalized.split()
            for kw in keywords_ascii:
                # Match nếu keyword là substring hoặc xuất hiện dưới dạng các token trong normalized.
                if kw in normalized:
                    return letter
        return None

    def _get_step(self, name: str) -> PipelineStep:
        for s in self.result.steps:
            if s.name == name:
                return s
        # If not found, append it dynamically
        new_step = PipelineStep(name=name)
        self.result.steps.append(new_step)
        return new_step

    def run(self, question: str, mode: str = "normal", options: dict | None = None) -> PipelineResult:
        """Run the full RAG pipeline."""
        self.result = PipelineResult(success=False)
        self.result.steps = [
            PipelineStep(name="Load Data"),
            PipelineStep(name="Rephrase Question"),
            PipelineStep(name="Extract JSON A"),
            PipelineStep(name="Hard Filter"),
            PipelineStep(name="Hybrid Search"),
            PipelineStep(name="Generate Answer"),
        ]

        try:
            # Step 1: Load Data
            if self.no_rag:
                self._get_step("Load Data").status = "skipped"
                self._get_step("Load Data").add_log("Load Data skipped in No RAG mode")
            else:
                self._step_load_data()

            # Step 2: Rephrase Question
            if self.no_rag:
                rephrased = question
                self._get_step("Rephrase Question").status = "skipped"
                self._get_step("Rephrase Question").add_log("Rephrase skipped in No RAG mode")
            else:
                rephrased = self._step_rephrase(question)
                if not rephrased:
                    self.result.error = "Failed to rephrase question"
                    return self.result

            # Step 3: Extract JSON A
            json_a = self._step_extract_json_a(rephrased, mode, options)
            if not json_a:
                self.result.error = "Failed to extract structured info"
                return self.result

            # Check if about UIT
            is_about_uit = json_a.get("isAboutUIT", "Yes") == "Yes"
            self.result.is_about_uit = is_about_uit

            if not is_about_uit:
                self._get_step("Generate Answer").status = "done"
                # Trong MCQ mode, tìm option mang nhãn "Không liên quan UIT / ngoài phạm vi / không liên quan đến giáo dục..."
                # để trả về JSON đúng đáp án, thay vì câu từ chối thuần túy.
                if mode == "mcq" and options:
                    no_uit_letter = self._find_out_of_scope_option(options)
                    if no_uit_letter:
                        self.result.answer = json.dumps({
                            "Answer": no_uit_letter,
                            "Explanation": "Câu hỏi này không liên quan đến UIT."
                        }, ensure_ascii=False)
                    else:
                        self.result.answer = "Câu hỏi này không liên quan đến UIT."
                        self.result.explanation = "Hệ thống chỉ trả lời các câu hỏi liên quan đến Trường Đại học Công nghệ Thông tin (UIT)."
                else:
                    self.result.answer = "Câu hỏi này không liên quan đến UIT."
                    self.result.explanation = "Hệ thống chỉ trả lời các câu hỏi liên quan đến Trường Đại học Công nghệ Thông tin (UIT)."
                self.result.success = True
                return self.result

            if self.no_rag:
                self._get_step("Hard Filter").status = "skipped"
                self._get_step("Hard Filter").add_log("Hard Filter skipped in No RAG mode")
                self._get_step("Hybrid Search").status = "skipped"
                self._get_step("Hybrid Search").add_log("Hybrid Search skipped in No RAG mode")
                
                # Step 6: Generate Answer (No RAG)
                answer_data = self._step_generate_answer_no_rag(question, mode, options, json_a)
            else:
                # Step 4: Hard Filter
                if self.disable_hard_filter:
                    candidate_indices = None
                    self._get_step("Hard Filter").status = "done"
                    self._get_step("Hard Filter").add_log("Hard filter disabled by configuration (Ablation Study)")
                else:
                    candidate_indices = self._step_hard_filter(json_a)

                # Step 5: Hybrid Search
                top_chunks = self._step_hybrid_search(rephrased, candidate_indices)

                # Self-RAG Step (if enabled)
                if not self.disable_self_rag and top_chunks:
                    # Dynamically insert Self-RAG Grading and Re-Retrieval steps
                    if "Self-RAG Grading" not in [s.name for s in self.result.steps]:
                        gen_idx = next(i for i, s in enumerate(self.result.steps) if s.name == "Generate Answer")
                        self.result.steps.insert(gen_idx, PipelineStep(name="Self-RAG Grading"))
                        self.result.steps.insert(gen_idx + 1, PipelineStep(name="Re-Retrieval"))

                    # Lượt 1: Đánh giá kết quả tìm kiếm ban đầu
                    is_sufficient = self._step_grade_retrieval(question, top_chunks)

                    if not is_sufficient:
                        retry_step = self._get_step("Re-Retrieval")
                        retry_step.status = "running"
                        retry_step.add_log("Lần 1: Tài liệu thiếu thông tin. Thực hiện Retry 1 (Vẫn dùng Hard Filter, Rephrase truy vấn)...")

                        # Lượt 2 (Retry 1): Rephrase câu truy vấn, giữ nguyên bộ lọc Hard Filter
                        retry_query = self._step_retry_rephrase(question)
                        retry_step.add_log(f"Retry 1: Tìm kiếm lại với câu hỏi đã viết lại: {retry_query}")
                        new_top_chunks = self._step_hybrid_search(retry_query, candidate_indices, step_name="Re-Retrieval")

                        if new_top_chunks:
                            retry_step.add_log("Retry 1: Đang đánh giá tài liệu mới...")
                            is_sufficient_retry1 = self._step_grade_retrieval(question, new_top_chunks, step_name="Re-Retrieval")
                            top_chunks = new_top_chunks

                            # Nếu Retry 1 vẫn không đủ thông tin, và có dùng Hard Filter -> Lượt 3 (Retry 2)
                            if not is_sufficient_retry1 and not self.disable_hard_filter and candidate_indices is not None:
                                retry_step.add_log("Lần 2: Vẫn thiếu thông tin. Thực hiện Retry 2 (Bỏ bộ lọc Hard Filter để tìm trên toàn DB)...")
                                retry_step.add_log(f"Retry 2: Tìm kiếm lại không dùng Hard Filter với câu hỏi: {retry_query}")
                                new_top_chunks_retry2 = self._step_hybrid_search(retry_query, None, step_name="Re-Retrieval")

                                if new_top_chunks_retry2:
                                    retry_step.add_log("Retry 2: Đang đánh giá tài liệu mới...")
                                    is_sufficient_retry2 = self._step_grade_retrieval(question, new_top_chunks_retry2, step_name="Re-Retrieval")
                                    top_chunks = new_top_chunks_retry2
                                else:
                                    retry_step.add_log("Retry 2: Không tìm thấy kết quả mới khi bỏ bộ lọc cứng.")
                        else:
                            # Nếu Retry 1 không tìm thấy gì, và có dùng Hard Filter -> chuyển sang Retry 2 ngay
                            if not self.disable_hard_filter and candidate_indices is not None:
                                retry_step.add_log("Retry 1 không trả về kết quả mới. Thực hiện Retry 2 (Bỏ bộ lọc Hard Filter)...")
                                new_top_chunks_retry2 = self._step_hybrid_search(rephrased, None, step_name="Re-Retrieval")
                                if new_top_chunks_retry2:
                                    retry_step.add_log("Retry 2: Đang đánh giá tài liệu mới...")
                                    is_sufficient_retry2 = self._step_grade_retrieval(question, new_top_chunks_retry2, step_name="Re-Retrieval")
                                    top_chunks = new_top_chunks_retry2
                                else:
                                    retry_step.add_log("Retry 2: Không tìm thấy kết quả mới khi bỏ bộ lọc cứng.")
                            else:
                                retry_step.add_log("Retry 1 không ra kết quả mới và không có Hard Filter để bỏ. Giữ nguyên tài liệu cũ.")

                        retry_step.status = "done"
                    else:
                        self._get_step("Re-Retrieval").status = "skipped"
                        self._get_step("Re-Retrieval").add_log("Tài liệu đầy đủ. Không cần truy xuất lại.")

                if not top_chunks:
                    self.result.error = "Không tìm thấy ngữ cảnh phù hợp"
                    return self.result

                # Step 6: Generate Answer
                answer_data = self._step_generate_answer(question, top_chunks, mode, options, json_a)

            if not answer_data:
                self.result.error = "Failed to generate answer"
                return self.result

            self.result.answer = answer_data.get("Answer", "")
            self.result.explanation = answer_data.get("Explanation", "")
            self.result.support_context = answer_data.get("SupportContext", "")

            self.result.success = True

        except Exception as e:
            logger.exception("Pipeline error")
            self.result.error = str(e)
            # Safe status update for the last step in case of generic failure
            if self.result.steps:
                self.result.steps[-1].status = "error"
                self.result.steps[-1].error = str(e)

        return self.result

    def _step_grade_retrieval(self, question: str, chunks: list[dict], step_name: str = "Self-RAG Grading") -> bool:
        """Evaluate if retrieved chunks are sufficient to answer the question."""
        step = self._get_step(step_name)
        step.status = "running"
        step.start_time = time.time()
        
        # Build context from chunks to pass to grader
        chunks_to_grade = chunks[:10]
        context = "\n\n".join([
            f"[Chunk {i+1}] {c['content']}"
            for i, c in enumerate(chunks_to_grade)
        ])
        
        step.input_data = {"question": question, "chunks_count": len(chunks_to_grade)}
        step.add_log(f"Đang đánh giá {len(chunks_to_grade)} tài liệu đầu tiên...")
        
        try:
            prompt = EVALUATE_RETRIEVAL_PROMPT.format(
                question=question,
                context=context
            )
            
            response, usage = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # low temperature for grading consistency
            )
            
            step.add_log(f"Tokens used: {usage.get('total_tokens', 'N/A')}")
            step.add_log(f"Grader Response (raw):\n{response[:300]}...")
            
            result = parse_json_safely(response)
            
            if result:
                is_sufficient_str = result.get("is_sufficient", "Yes")
                reason = result.get("reason", "")
                
                is_sufficient = is_sufficient_str.strip().lower() == "yes"
                
                step.output_data = result
                step.add_log(f"Kết quả đánh giá: {is_sufficient_str} | Lý do: {reason}")
                
                if is_sufficient:
                    step.status = "done"
                    return True
                else:
                    step.status = "done"
                    return False
            else:
                step.add_log("Không thể parse JSON từ phản hồi của Grader. Mặc định: Đủ thông tin.")
                step.output_data = {"is_sufficient": "Yes", "reason": "Parsing failed, fallback to Yes"}
                step.status = "done"
                return True
                
        except Exception as e:
            step.add_log(f"Lỗi chấm điểm: {e}. Mặc định: Đủ thông tin.")
            step.status = "error"
            step.error = str(e)
            return True
        finally:
            step.end_time = time.time()

    def _step_retry_rephrase(self, question: str) -> str:
        """Rephrase the question with synonyms for re-retrieval."""
        step = self._get_step("Re-Retrieval")
        step.add_log(f"Viết lại câu hỏi tìm kiếm...")
        
        try:
            prompt = RETRY_REPHRASE_PROMPT.format(question=question)
            response, usage = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            
            rephrased = response.strip()
            step.add_log(f"Câu hỏi viết lại: {rephrased}")
            return rephrased
        except Exception as e:
            step.add_log(f"Lỗi rephrase retry: {e}. Sử dụng câu hỏi gốc.")
            return question

    def _step_load_data(self):
        """Load chunks from embedded file."""
        step = self._get_step("Load Data")
        step.status = "running"
        step.start_time = time.time()
        step.add_log("Bắt đầu load chunks...")

        try:
            count = self.db.load()
            step.add_log(f"Đã load {count} chunks")
            step.status = "done"
        except Exception as e:
            step.add_log(f"Lỗi load: {e}")
            step.status = "error"
            step.error = str(e)
        finally:
            step.end_time = time.time()

    def _step_rephrase(self, question: str) -> str | None:
        """Rephrase the question for better retrieval."""
        step = self._get_step("Rephrase Question")
        step.status = "running"
        step.start_time = time.time()
        step.input_data = {"question": question}
        step.add_log(f"Câu hỏi gốc: {question}")

        try:
            prompt = REPHRASE_PROMPT.format(question=question)
            response, usage = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            step.add_log(f"Tokens used: {usage.get('total_tokens', 'N/A')}")
            step.add_log(f"Câu hỏi đã rephrase: {response[:200]}...")
            step.output_data = {"rephrased": response}
            step.status = "done"

            return response

        except Exception as e:
            step.add_log(f"Lỗi: {e}")
            step.status = "error"
            step.error = str(e)
            return None
        finally:
            step.end_time = time.time()

    def _step_extract_json_a(self, rephrased: str, mode: str, options: dict | None) -> dict | None:
        """Extract structured information (JSON A)."""
        step = self._get_step("Extract JSON A")
        step.status = "running"
        step.start_time = time.time()
        step.input_data = {"rephrased": rephrased, "mode": mode}

        # Build prompt with options for MCQ mode
        prompt_base = EXTRACT_JSON_A_PROMPT.format(question=rephrased)

        if mode == "mcq" and options:
            options_text = "\n".join([f"- {k}: {v}" for k, v in options.items()])
            prompt_base += f"\n\nTrong chế độ MCQ, đáp án phải là một trong: {options_text}"

        step.add_log(f"Extracting JSON A...")
        step.add_log(f"Mode: {mode}")

        try:
            response, usage = self.llm.chat(
                messages=[{"role": "user", "content": prompt_base}],
                temperature=0.1,
            )

            step.add_log(f"LLM Response (raw):\n{response[:500]}...")

            json_a = parse_json_safely(response)

            if json_a:
                step.add_log(f"Parsed JSON A: {json_a}")
                step.output_data = json_a
            else:
                step.add_log("Failed to parse JSON, using defaults")
                json_a = {
                    "category": "chung",
                    "year": 2026,
                    "major": None,
                    "isAboutUIT": "Yes",
                }
                step.output_data = json_a

            step.status = "done"
            return json_a

        except Exception as e:
            step.add_log(f"Lỗi: {e}")
            step.status = "error"
            step.error = str(e)
            return None
        finally:
            step.end_time = time.time()

    def _step_hard_filter(self, json_a: dict) -> list[int]:
        """Apply hard filters based on JSON A."""
        step = self._get_step("Hard Filter")
        step.status = "running"
        step.start_time = time.time()

        category = json_a.get("category", "chung")
        year = json_a.get("year", None)
        major = json_a.get("major", "")

        # Extract keywords from question for content filtering
        rephrased = self._get_step("Rephrase Question").output_data.get("rephrased", "")
        keywords = self._extract_keywords(rephrased, category)

        step.add_log(f"Category: {category}")
        step.add_log(f"Year: {year}")
        step.add_log(f"Major: {major}")
        step.add_log(f"Keywords: {keywords}")

        try:
            candidate_indices = self.db.hard_filter(
                category=category,
                year=year,
                major=major if major else None,
                keywords=keywords,
            )

            step.add_log(f"Candidates after filter: {len(candidate_indices)}")
            step.output_data = {"candidate_count": len(candidate_indices)}
            step.status = "done"

            return candidate_indices

        except Exception as e:
            step.add_log(f"Lỗi: {e}, falling back to all chunks")
            step.status = "error"
            step.error = str(e)
            # Fallback: return all chunks
            return list(range(len(self.db.chunks)))
        finally:
            step.end_time = time.time()

    def _extract_keywords(self, text: str, category: str = "chung") -> list[str]:
        """Extract keywords from text for content filtering.
        
        For tuition-related queries, only use the primary keyword to avoid
        conflicting with scholarship chunks.
        """
        text_lower = text.lower()
        
        # Check for tuition vs scholarship primary keywords
        is_tuition_query = "học phí" in text_lower
        is_scholarship_query = "học bổng" in text_lower
        
        # If asking about tuition, don't add "học bổng" as a filter
        # This prevents scholarship chunks from appearing in tuition searches
        if is_tuition_query and not is_scholarship_query:
            return ["học phí"]
        
        # If asking about scholarship, don't add "học phí" as a filter
        if is_scholarship_query and not is_tuition_query:
            return ["học bổng"]
        
        # If both keywords are explicitly mentioned, include both
        found = []
        if is_tuition_query:
            found.append("học phí")
        if is_scholarship_query:
            found.append("học bổng")
        
        # General keywords
        general_keywords = ["miễn giảm", "hỗ trợ tài chính"]
        for kw in general_keywords:
            if kw in text_lower:
                found.append(kw)
        
        return found if found else None

    def _step_hybrid_search(self, query: str, candidate_indices: list[int], step_name: str = "Hybrid Search") -> list[dict]:
        """Hybrid search combining embedding + BM25 + reranking."""
        step = self._get_step(step_name)
        step.status = "running"
        step.start_time = time.time()
        cand_count = len(candidate_indices) if candidate_indices is not None else len(self.db.chunks)
        step.input_data = {"query": query, "candidates": cand_count}

        step.add_log(f"Hybrid searching in {cand_count} candidates...")

        try:
            # 1. Embedding search - Top 50 (semantic coverage)
            try:
                query_vec = generate_embedding(query)
                step.add_log(f"Generated query embedding (dim={len(query_vec)})")

                embedding_results = self.db.vector_search(
                    query_vector=query_vec,
                    candidate_indices=candidate_indices,
                    top_k=50,
                )
                step.add_log(f"Embedding search: {len(embedding_results)} results")
                for idx, (chunk_idx, score) in enumerate(embedding_results[:3]):
                    chunk = self.db.get_chunk(chunk_idx)
                    if chunk:
                        step.add_log(f"  [Emb {idx+1}] Score={score:.3f}: {chunk['chunk_id']}")
            except Exception as e:
                step.add_log(f"Embedding search failed: {e}, using empty results")
                embedding_results = []

            # 2. BM25 search - Top 50 (keyword coverage)
            try:
                bm25_results = self.db.bm25.search(
                    query=query,
                    candidate_indices=candidate_indices,
                    top_k=50,
                )
                step.add_log(f"BM25 search: {len(bm25_results)} results")
                for idx, (chunk_idx, score) in enumerate(bm25_results[:3]):
                    chunk = self.db.get_chunk(chunk_idx)
                    if chunk:
                        step.add_log(f"  [BM25 {idx+1}] Score={score:.3f}: {chunk['chunk_id']}")
            except Exception as e:
                step.add_log(f"BM25 search failed: {e}, using empty results")
                bm25_results = []

            # 3. RRF Fusion - combine to 60 unique
            # Use weighted RRF to preserve quality from high-scoring retrievers
            try:
                fused_results = reciprocal_rank_fusion(
                    [embedding_results, bm25_results],
                    k=60,
                    use_weighted_scores=True
                )
                step.add_log(f"RRF Fusion: {len(fused_results)} unique results (weighted)")
            except Exception as e:
                step.add_log(f"RRF Fusion failed: {e}")
                # Fallback: combine results manually
                all_idx = set(idx for idx, _ in embedding_results) | set(idx for idx, _ in bm25_results)
                fused_results = [(idx, 1.0) for idx in all_idx]

            # 4. Get top 60 candidates for reranking
            top_60_indices = [idx for idx, _ in fused_results[:60]]

            if not top_60_indices:
                step.add_log("No results after fusion")
                step.status = "done"
                return []

            top_60_chunks = self.db.get_chunks_by_indices(top_60_indices)

            # 5. Cross-Encoder Rerank - Top 15 (quality filter)
            try:
                reranker = CrossEncoderReranker()
                reranked = reranker.rerank(query, top_60_chunks, top_k=15)

                step.add_log(f"Reranked: {len(reranked)} final results")
                for idx, (chunk, score) in enumerate(reranked[:5]):
                    step.add_log(f"  [Final {idx+1}] Score={score:.4f}: {chunk['chunk_id']}")
            except Exception as e:
                step.add_log(f"Reranking failed: {e}, using fusion results instead")
                # Fallback: return fusion results without reranking
                reranked = [(chunk, 1.0 / (i + 1)) for i, chunk in enumerate(top_60_chunks[:15])]
                for idx, (chunk, score) in enumerate(reranked[:5]):
                    step.add_log(f"  [Fallback {idx+1}] Score={score:.4f}: {chunk['chunk_id']}")

            step.output_data = {
                "embedding_results": len(embedding_results),
                "bm25_results": len(bm25_results),
                "fused_results": len(fused_results),
                "final_results": len(reranked),
                "chunks": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "category": chunk.get("category"),
                        "year": chunk.get("year"),
                        "major": chunk.get("major"),
                        "content": chunk.get("content")
                    } for chunk, _ in reranked
                ]
            }
            step.status = "done"

            return [chunk for chunk, _ in reranked]

        except Exception as e:
            step.add_log(f"Lỗi: {e}")
            step.status = "error"
            step.error = str(e)
            return []
        finally:
            step.end_time = time.time()

    def _step_generate_answer(
        self,
        question: str,
        chunks: list[dict],
        mode: str,
        options: dict | None,
        json_a: dict | None = None,
    ) -> dict | None:
        """Generate the final answer from chunks."""
        step = self._get_step("Generate Answer")
        step.status = "running"
        step.start_time = time.time()

        # Limit to top 10 chunks for LLM context
        max_context_chunks = 10
        chunks_for_context = chunks[:max_context_chunks]

        # Build context from chunks
        context = "\n\n".join([
            f"[Chunk {i+1}] {c['content']}"
            for i, c in enumerate(chunks_for_context)
        ])

        step.input_data = {"question": question, "chunks_count": len(chunks_for_context)}
        step.add_log(f"Building context from {len(chunks_for_context)} chunks (top {max_context_chunks} of {len(chunks)} retrieved)...")
        step.add_log(f"Context preview: {context[:300]}...")

        try:
            if mode == "mcq":
                # Check if multiple answers expected
                is_multi = False
                if json_a:
                    is_multi = json_a.get("isMultiAnswer", False)
                step.add_log(f"isMultiAnswer: {is_multi}")

                # Build multi-answer context for prompt
                if is_multi:
                    is_multi_context = (
                        "**CHẾ ĐỘ MULTIPLE CHOICE**: Câu hỏi này cho phép CHỌN NHIỀU đáp án đúng.\n"
                        "Hãy chọn TẤT CẢ các đáp án đúng và liệt kê chúng, ví dụ: 'A, B, C'"
                    )
                    answer_format = "A, B, C (danh sách các đáp án đúng, phân cách bằng dấu phẩy)"
                else:
                    is_multi_context = (
                        "**CHẾ ĐỘ SINGLE CHOICE**: Câu hỏi này chỉ có MỘT đáp án đúng.\n"
                        "Hãy chọn duy nhất một đáp án đúng."
                    )
                    answer_format = "X (chỉ một đáp án đúng, ví dụ: B)"

                prompt = GENERATE_ANSWER_PROMPT.format(
                    question=question,
                    is_multi_context=is_multi_context,
                    answer_format=answer_format,
                    context=context,
                )

                if options:
                    options_text = "\n".join([f"- {k}: {v}" for k, v in options.items()])
                    # KHÔNG bắt buộc chọn E khi câu hỏi thuộc UIT
                    # Chỉ chọn E khi câu hỏi KHÔNG thuộc UIT (đã xử lý ở hàm run)
                    prompt += f"\n\n**QUAN TRỌNG - MCQ Mode**: Đáp án phải là MỘT TRONG các lựa chọn sau:\n{options_text}"
            else:
                prompt = GENERATE_ANSWER_NORMAL_PROMPT.format(
                    question=question,
                    context=context,
                )
                answer_format = "Free-text"

            step.add_log(f"Calling LLM to generate answer...")
            step.add_log(f"Answer format: {answer_format}")
            response, usage = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8000,
            )

            step.add_log(f"Tokens used: {usage.get('total_tokens', 'N/A')}")
            step.add_log(f"Raw LLM Response:\n{response}")

            answer_data = parse_json_safely(response)

            if answer_data:
                step.add_log(f"Parsed Answer: {answer_data.get('Answer', '')}")

                # CRITICAL: Nếu LLM xác định câu hỏi THUỘC UIT (isAboutUIT=Yes)
                # thì KHÔNG ĐƯỢC chọn E - force chọn đáp án khác (chỉ áp dụng trong MCQ mode)
                if mode == "mcq" and options and json_a and json_a.get("isAboutUIT", "No") == "Yes":
                    selected = answer_data.get("Answer", "").strip().upper()
                    if selected == "E":
                        step.add_log("WARNING: LLM chọn E cho câu hỏi thuộc UIT! Force chọn đáp án khác...")
                        # Chọn đáp án đầu tiên khác E
                        for key in options.keys():
                            if key.upper() != "E":
                                answer_data["Answer"] = key
                                answer_data["Explanation"] = "(Hệ thống tự động sửa: không chọn E cho câu hỏi thuộc UIT)"
                                step.add_log(f"Force chọn đáp án: {key}")
                                break

                step.output_data = answer_data
            else:
                step.add_log("Failed to parse JSON, using raw response")
                answer_data = {
                    "Answer": response[:500],
                    "Explanation": "",
                    "SupportContext": "",
                }
                step.output_data = answer_data

            step.status = "done"
            return answer_data

        except Exception as e:
            step.add_log(f"Lỗi: {e}")
            step.status = "error"
            step.error = str(e)
            return None
        finally:
            step.end_time = time.time()

    def _step_generate_answer_no_rag(
        self,
        question: str,
        mode: str,
        options: dict | None,
        json_a: dict | None = None,
    ) -> dict | None:
        """Generate answer directly from LLM without any RAG context."""
        step = self._get_step("Generate Answer")
        step.status = "running"
        step.start_time = time.time()
        step.input_data = {"question": question}
        step.add_log("No RAG mode enabled - generating answer directly from LLM...")

        try:
            if mode == "mcq":
                is_multi = False
                if json_a:
                    is_multi = json_a.get("isMultiAnswer", False)
                step.add_log(f"isMultiAnswer: {is_multi}")

                if is_multi:
                    is_multi_context = (
                        "**CHẾ ĐỘ MULTIPLE CHOICE**: Câu hỏi này cho phép CHỌN NHIỀU đáp án đúng.\n"
                        "Hãy chọn TẤT CẢ các đáp án đúng và liệt kê chúng, ví dụ: 'A, B, C'"
                    )
                    answer_format = "A, B, C (danh sách các đáp án đúng, phân cách bằng dấu phẩy)"
                else:
                    is_multi_context = (
                        "**CHẾ ĐỘ SINGLE CHOICE**: Câu hỏi này chỉ có MỘT đáp án đúng.\n"
                        "Hãy chọn duy nhất một đáp án đúng."
                    )
                    answer_format = "X (chỉ một đáp án đúng, ví dụ: B)"

                prompt = f"""Bạn là trợ lý AI thân thiện của Trường Đại học Công nghệ Thông tin (UIT).
Hãy trả lời câu hỏi trắc nghiệm dưới đây dựa trên kiến thức của bạn.

Câu hỏi: {question}
{is_multi_context}

Yêu cầu:
1. Trả lời bằng ngôn ngữ tự nhiên, thân thiện.
2. Trả lời theo định dạng JSON sau (chỉ trả JSON, không thêm văn bản giải thích ngoài JSON):
{{
    "Answer": "{answer_format}",
    "Explanation": "giải thích ngắn gọn lý do chọn đáp án này"
}}
"""

                if options:
                    options_text = "\n".join([f"- {k}: {v}" for k, v in options.items()])
                    prompt += f"\n\n**QUAN TRỌNG - MCQ Mode**: Đáp án phải là MỘT TRONG các lựa chọn sau:\n{options_text}"
            else:
                prompt = f"""Bạn là trợ lý AI thân thiện của Trường Đại học Công nghệ Thông tin (UIT).
Hãy trả lời câu hỏi dưới đây dựa trên kiến thức của bạn một cách TỰ NHIÊN, THÂN THIỆN như đang trò chuyện với sinh viên.

Câu hỏi: {question}

Yêu cầu:
1. Trả lời bằng ngôn ngữ tự nhiên, thân thiện.
2. Trả lời theo định dạng JSON sau (chỉ trả JSON, không thêm văn bản giải thích ngoài JSON):
{{
    "Answer": "câu trả lời chi tiết và đầy đủ trực tiếp cho câu hỏi của người dùng",
    "Explanation": "giải thích chi tiết hoặc thêm thông tin bổ ích (2-3 câu, viết tự nhiên như đang trò chuyện)"
}}
"""

            step.add_log(f"Calling LLM (No RAG) to generate answer...")
            response, usage = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )

            step.add_log(f"Tokens used: {usage.get('total_tokens', 'N/A')}")
            step.add_log(f"Raw LLM Response:\n{response}")

            answer_data = parse_json_safely(response)

            if answer_data:
                step.add_log(f"Parsed Answer: {answer_data.get('Answer', '')}")

                # Force check E logic (same as standard, only for MCQ mode)
                if mode == "mcq" and options and json_a and json_a.get("isAboutUIT", "No") == "Yes":
                    selected = answer_data.get("Answer", "").strip().upper()
                    if selected == "E":
                        step.add_log("WARNING: LLM chọn E cho câu hỏi thuộc UIT! Force chọn đáp án khác...")
                        for key in options.keys():
                            if key.upper() != "E":
                                answer_data["Answer"] = key
                                answer_data["Explanation"] = "(Hệ thống tự động sửa: không chọn E cho câu hỏi thuộc UIT)"
                                step.add_log(f"Force chọn đáp án: {key}")
                                break

                step.output_data = answer_data
            else:
                step.add_log("Failed to parse JSON, using raw response")
                answer_data = {
                    "Answer": response[:500],
                    "Explanation": "",
                }
                step.output_data = answer_data

            step.status = "done"
            return answer_data

        except Exception as e:
            step.add_log(f"Lỗi: {e}")
            step.status = "error"
            step.error = str(e)
            return None
        finally:
            step.end_time = time.time()

