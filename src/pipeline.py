"""
RAG Pipeline - Orchestrate the full RAG flow.
Handles: rephrase -> extract JSON A -> hard filter -> retrieval -> rerank -> answer -> support check.
"""

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
    REPHRASE_PROMPT,
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

    def __init__(self):
        self.llm = get_llm_client()
        self.db = get_chunk_database()
        self.result = PipelineResult(success=False)

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
            self._step_load_data()

            # Step 2: Rephrase Question
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
                self.result.steps[5].status = "done"
                self.result.answer = "Câu hỏi này không liên quan đến UIT."
                self.result.explanation = "Hệ thống chỉ trả lời các câu hỏi liên quan đến Trường Đại học Công nghệ Thông tin (UIT)."
                self.result.success = True
                return self.result

            # Step 4: Hard Filter
            candidate_indices = self._step_hard_filter(json_a)

            # Step 5: Hybrid Search
            top_chunks = self._step_hybrid_search(rephrased, candidate_indices)

            if not top_chunks:
                self.result.error = "Không tìm thấy ngữ cảnh phù hợp"
                return self.result

            # Step 6: Generate Answer
            answer_data = self._step_generate_answer(question, top_chunks, mode, options)
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
            self.result.steps[-1].status = "error"
            self.result.steps[-1].error = str(e)

        return self.result

    def _step_load_data(self):
        """Load chunks from embedded file."""
        step = self.result.steps[0]
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
        step = self.result.steps[1]
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
        step = self.result.steps[2]
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
        step = self.result.steps[3]
        step.status = "running"
        step.start_time = time.time()

        category = json_a.get("category", "chung")
        year = json_a.get("year", None)
        major = json_a.get("major", "")

        # Extract keywords from question for content filtering
        rephrased = self.result.steps[1].output_data.get("rephrased", "")
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

    def _step_hybrid_search(self, query: str, candidate_indices: list[int]) -> list[dict]:
        """Hybrid search combining embedding + BM25 + reranking."""
        step = self.result.steps[4]
        step.status = "running"
        step.start_time = time.time()
        step.input_data = {"query": query, "candidates": len(candidate_indices)}

        step.add_log(f"Hybrid searching in {len(candidate_indices)} candidates...")

        try:
            # 1. Embedding search - Top 10
            try:
                query_vec = generate_embedding(query)
                step.add_log(f"Generated query embedding (dim={len(query_vec)})")

                embedding_results = self.db.vector_search(
                    query_vector=query_vec,
                    candidate_indices=candidate_indices,
                    top_k=10,
                )
                step.add_log(f"Embedding search: {len(embedding_results)} results")
                for idx, (chunk_idx, score) in enumerate(embedding_results[:3]):
                    chunk = self.db.get_chunk(chunk_idx)
                    if chunk:
                        step.add_log(f"  [Emb {idx+1}] Score={score:.3f}: {chunk['chunk_id']}")
            except Exception as e:
                step.add_log(f"Embedding search failed: {e}, using empty results")
                embedding_results = []

            # 2. BM25 search - Top 20 (increased from 10 to capture more relevant results)
            try:
                bm25_results = self.db.bm25.search(
                    query=query,
                    candidate_indices=candidate_indices,
                    top_k=20,
                )
                step.add_log(f"BM25 search: {len(bm25_results)} results")
                for idx, (chunk_idx, score) in enumerate(bm25_results[:3]):
                    chunk = self.db.get_chunk(chunk_idx)
                    if chunk:
                        step.add_log(f"  [BM25 {idx+1}] Score={score:.3f}: {chunk['chunk_id']}")
            except Exception as e:
                step.add_log(f"BM25 search failed: {e}, using empty results")
                bm25_results = []

            # 3. RRF Fusion - combine to 30 unique (increased to capture more BM25 results)
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

            # 4. Get top 30 candidates for reranking (increased from 20)
            top_30_indices = [idx for idx, _ in fused_results[:30]]

            if not top_30_indices:
                step.add_log("No results after fusion")
                step.status = "done"
                return []

            top_30_chunks = self.db.get_chunks_by_indices(top_30_indices)

            # 5. Cross-Encoder Rerank - Top 10 (return more for LLM context)
            try:
                reranker = CrossEncoderReranker()
                reranked = reranker.rerank(query, top_30_chunks, top_k=10)

                step.add_log(f"Reranked: {len(reranked)} final results")
                for idx, (chunk, score) in enumerate(reranked[:5]):
                    step.add_log(f"  [Final {idx+1}] Score={score:.4f}: {chunk['chunk_id']}")
            except Exception as e:
                step.add_log(f"Reranking failed: {e}, using fusion results instead")
                # Fallback: return fusion results without reranking
                reranked = [(chunk, 1.0 / (i + 1)) for i, chunk in enumerate(top_30_chunks[:10])]
                for idx, (chunk, score) in enumerate(reranked[:5]):
                    step.add_log(f"  [Fallback {idx+1}] Score={score:.4f}: {chunk['chunk_id']}")

            step.output_data = {
                "embedding_results": len(embedding_results),
                "bm25_results": len(bm25_results),
                "fused_results": len(fused_results),
                "final_results": len(reranked),
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
    ) -> dict | None:
        """Generate the final answer from chunks."""
        step = self.result.steps[5]
        step.status = "running"
        step.start_time = time.time()

        # Build context from chunks
        context = "\n\n".join([
            f"[Chunk {i+1}] {c['content']}"
            for i, c in enumerate(chunks)
        ])

        step.input_data = {"question": question, "chunks_count": len(chunks)}
        step.add_log(f"Building context from {len(chunks)} chunks...")
        step.add_log(f"Context preview: {context[:300]}...")

        try:
            prompt = GENERATE_ANSWER_PROMPT.format(
                question=question,
                context=context,
            )

            if mode == "mcq" and options:
                options_text = "\n".join([f"- {k}: {v}" for k, v in options.items()])
                prompt += f"\n\n**QUAN TRỌNG - MCQ Mode**: Đáp án phải là MỘT TRONG các lựa chọn sau:\n{options_text}\nNếu không tìm thấy thông tin, chọn đáp án 'E'."

            step.add_log(f"Calling LLM to generate answer...")
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

