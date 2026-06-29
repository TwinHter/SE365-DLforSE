"""
RAG Evaluation Script - Đánh giá hệ thống RAG với testset UIT.

Usage:
    python evaluate.py                          # Chạy tất cả 100 câu
    python evaluate.py --limit 10              # Chạy 10 câu đầu
    python evaluate.py --sample 20              # Lấy ngẫu nhiên 20 câu
    python evaluate.py --question uit_q001      # Chạy 1 câu cụ thể
"""

import argparse
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.pipeline import RAGPipeline
from src.llm_utils import parse_json_safely

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Kết quả đánh giá một câu hỏi."""
    question_id: str
    question: str
    expected_answer: list[str]
    predicted_answer: list[str]
    is_correct: bool
    is_partial_correct: bool  # Đúng một phần (multiple choice)

    # Voting details
    vote_runs: list[list[str]] = field(default_factory=list)  # Đáp án từ 3 lần chạy
    vote_counts: dict = field(default_factory=dict)  # Đếm số lần xuất hiện

    # Performance metrics
    total_duration_ms: float = 0.0
    llm_calls: int = 0
    tokens_used: int = 0

    # Pipeline details
    retrieved_chunks: int = 0
    is_about_uit: bool = True
    pipeline_error: str = ""

    # Raw LLM response for debugging
    raw_answer: str = ""
    raw_json: dict = field(default_factory=dict)

    # Difficulty (if available)
    difficulty: Optional[int] = None
    gold_context: str = ""


class RAGEvaluator:
    """Đánh giá RAG pipeline với testset."""
    
    def __init__(self, testset_path: str, output_dir: str = "results"):
        self.testset_path = Path(testset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.testset = self._load_testset()
        self.pipeline = RAGPipeline()
        
        # Stats
        self.total_llm_calls = 0
        self.total_tokens = 0
        
    def _load_testset(self) -> list[dict]:
        """Load testset từ JSON file."""
        logger.info(f"Loading testset from {self.testset_path}")
        with open(self.testset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} questions")
        return data
    
    def _extract_mcq_answer(self, llm_response: str, options: dict = None) -> list[str]:
        """
        Extract đáp án MCQ từ LLM response.

        Hỗ trợ nhiều format:
        - JSON: {"Answer": "B", ...}
        - JSON: {"Answer": ["B"], ...}
        - Plain text: "B" hoặc "B: text" hoặc "Đáp án: B"
        """
        if not llm_response:
            return []

        # Clean the response
        response = llm_response.strip()

        # Nếu LLM từ chối / xác nhận câu hỏi không liên quan -> không phải đáp án MCQ.
        # Tránh bắt nhầm ký tự đầu tiên là A-E trong câu tiếng Việt (vd: "Câu hỏi..." -> nhầm thành "C").
        refusal_keywords = [
            "không liên quan", "nằm ngoài phạm vi", "không thể trả lời",
            "không đủ thông tin", "không có thông tin", "xin lỗi",
            "không thuộc", "ngoài tầm", "vượt quá phạm vi",
        ]
        if any(kw in response.lower() for kw in refusal_keywords):
            return []

        # CASE 1: Try parsing as JSON first
        json_data = parse_json_safely(llm_response)
        if json_data:
            answer = json_data.get("Answer", "")
            if isinstance(answer, list):
                return self._normalize_answers(answer)
            elif isinstance(answer, str):
                # Check if it's a JSON string
                if answer.startswith('{'):
                    inner_json = parse_json_safely(answer)
                    if inner_json and "Answer" in inner_json:
                        return self._normalize_answers([inner_json["Answer"]])
                # Clean and extract
                answer = answer.strip()
                return self._normalize_answers([answer])
        
        # CASE 2: Parse as text (fallback)
        # Try to find answer patterns like "B: text" or "Đáp án: B" or just "B"
        return self._extract_from_text(llm_response)
    
    def _extract_from_text(self, text: str) -> list[str]:
        """Extract answers from plain text response."""
        answers = []

        # Pattern 1: "B: text" or "B - text" (answer at beginning)
        patterns = [
            r'^([A-E])[\s:.\-]+',  # Starts with letter
            r'([A-E])[\s:.\-]+',   # Letter followed by colon/dash
            r'(?:đáp án|dap an|answer)[:\s]+([A-E](?:\s*,\s*[A-E])*)',
            r'\b([A-E])(?:\s*,\s*([A-E]))*\b',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            for match in matches:
                if isinstance(match, tuple):
                    # Filter out empty matches
                    found = [m for m in match if m and re.match(r'^[A-E]$', m)]
                    answers.extend(found)
                elif isinstance(match, str) and re.match(r'^[A-E]$', match):
                    answers.append(match)

        # Nếu không match pattern nào -> bỏ cuộc, KHÔNG tự ý lấy chữ cái đầu tiên.
        # Fallback cũ (`re.match(r'^([A-E])', text)`) bắt nhầm ký tự đầu của câu tiếng Việt
        # viết hoa (vd: "Câu hỏi..." -> "C", "Ây không..." -> "A") gây ra vote sai.

        # Remove duplicates while preserving order
        seen = set()
        unique_answers = []
        for a in answers:
            if a not in seen:
                seen.add(a)
                unique_answers.append(a)

        return unique_answers
    
    def _normalize_answers(self, answers: list) -> list[str]:
        """Normalize answers to list of uppercase letters."""
        normalized = []
        for ans in answers:
            if isinstance(ans, str):
                # Handle strings like "B" or "B, C" or " B, C "
                ans = ans.strip()
                # If contains comma, split and process each
                if ',' in ans:
                    parts = [p.strip() for p in ans.split(',')]
                    for part in parts:
                        found = re.findall(r'[A-E]', part.upper())
                        normalized.extend(found)
                else:
                    # Extract letters from strings like "B" or "Đáp án: B"
                    found = re.findall(r'[A-E]', ans.upper())
                    normalized.extend(found)
            elif isinstance(ans, (int, float)) and ans in [1, 2, 3, 4, 5]:
                # Map number to letter
                normalized.append(chr(64 + int(ans)))
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for a in normalized:
            if a not in seen:
                seen.add(a)
                result.append(a)
        
        return result
    
    def _compare_answers(self, predicted: list[str], expected: list[str]) -> tuple[bool, bool]:
        """
        Compare predicted vs expected answers.
        
        Returns:
            (exact_match, partial_match)
        """
        pred_set = set(predicted)
        exp_set = set(expected)
        
        # Exact match
        exact = pred_set == exp_set
        
        # Partial match (at least one correct, no wrong answers)
        partial = len(pred_set & exp_set) > 0 and len(pred_set - exp_set) == 0
        
        return exact, partial
    
    def _majority_vote(self, vote_runs: list[list[str]]) -> tuple[list[str], dict]:
        """
        Mode voting: lấy đáp án xuất hiện nhiều nhất qua các lần chạy.

        Args:
            vote_runs: Danh sách các đáp án từ N lần chạy (mỗi lần là list các letters)

        Returns:
            (winning_answer, vote_counts)
        """
        from collections import Counter

        # Convert mỗi list answers thành tuple để hash được
        answer_tuples = [tuple(sorted(a)) for a in vote_runs if a]

        if not answer_tuples:
            return [], {}

        counter = Counter(answer_tuples)
        most_common = counter.most_common(1)[0]
        winning_tuple = most_common[0]

        # Chuyển counts thành dict để hiển thị
        vote_counts = {",".join(k) if k else "(empty)": v for k, v in counter.items()}

        return list(winning_tuple), vote_counts

    def evaluate_single(self, item: dict, n_runs: int = 3) -> EvaluationResult:
        """Đánh giá một câu hỏi với Mode voting N lần."""
        question_id = item.get("id", "unknown")
        question = item.get("question", "")
        expected = item.get("answer", [])
        options = item.get("options", {})
        difficulty = item.get("difficulty")
        gold_context = item.get("gold_context", "")

        logger.info(f"Evaluating {question_id}: {question[:50]}...")

        result = EvaluationResult(
            question_id=question_id,
            question=question,
            expected_answer=expected,
            predicted_answer=[],
            is_correct=False,
            is_partial_correct=False,
            difficulty=difficulty,
            gold_context=gold_context
        )

        start_time = time.time()

        try:
            vote_runs = []
            raw_answers = []
            last_pipeline_result = None

            # Chạy pipeline N lần
            for run_idx in range(n_runs):
                logger.info(f"  Run {run_idx + 1}/{n_runs}...")

                pipeline_result = self.pipeline.run(
                    question=question,
                    mode="mcq",
                    options=options
                )

                last_pipeline_result = pipeline_result
                raw_answers.append(pipeline_result.answer)

                # Extract answer từ response
                if pipeline_result.success and pipeline_result.answer:
                    json_data = parse_json_safely(pipeline_result.answer)
                    if json_data:
                        result.raw_json = json_data
                        predicted = self._extract_mcq_answer(json_data.get("Answer", ""), options)
                    else:
                        predicted = self._extract_mcq_answer(pipeline_result.answer, options)

                    if predicted:
                        vote_runs.append(predicted)
                    else:
                        vote_runs.append([])
                else:
                    vote_runs.append([])

            # Mode voting: lấy đáp án xuất hiện nhiều nhất
            winning_answer, vote_counts = self._majority_vote(vote_runs)

            result.vote_runs = vote_runs
            result.vote_counts = vote_counts
            result.predicted_answer = winning_answer
            result.total_duration_ms = (time.time() - start_time) * 1000
            result.is_about_uit = last_pipeline_result.is_about_uit if last_pipeline_result else True
            result.raw_answer = raw_answers[-1] if raw_answers else ""
            result.pipeline_error = last_pipeline_result.error if last_pipeline_result else ""

            # Count chunks retrieved (từ lần chạy cuối)
            if last_pipeline_result:
                for step in last_pipeline_result.steps:
                    if step.name == "Hybrid Search":
                        result.retrieved_chunks = step.output_data.get("candidate_count", 0)

            # Compare
            exact, partial = self._compare_answers(winning_answer, expected)
            result.is_correct = exact
            result.is_partial_correct = partial

            logger.info(f"  -> Vote runs: {vote_runs}")
            logger.info(f"  -> Vote counts: {vote_counts}")
            logger.info(f"  -> Expected: {expected}, Predicted: {winning_answer}, Correct: {exact}")

            if not last_pipeline_result or not last_pipeline_result.success:
                logger.warning(f"  -> Pipeline failed: {result.pipeline_error}")

        except Exception as e:
            result.total_duration_ms = (time.time() - start_time) * 1000
            result.pipeline_error = str(e)
            logger.error(f"  -> Error: {e}")

        return result
    
    def evaluate_all(
        self,
        limit: Optional[int] = None,
        sample_size: Optional[int] = None,
        question_ids: Optional[list[str]] = None,
        n_runs: int = 3
    ) -> list[EvaluationResult]:
        """Đánh giá tất cả hoặc một phần testset với Mode voting."""

        # Select questions
        questions = self.testset

        if question_ids:
            questions = [q for q in questions if q.get("id") in question_ids]
        elif sample_size and sample_size < len(questions):
            questions = random.sample(questions, sample_size)
        elif limit:
            questions = questions[:limit]

        logger.info(f"Evaluating {len(questions)} questions with {n_runs}-run mode voting...")

        results = []
        for item in questions:
            result = self.evaluate_single(item, n_runs=n_runs)
            results.append(result)

        return results
    
    def generate_report(self, results: list[EvaluationResult]) -> dict:
        """Generate evaluation report."""
        
        total = len(results)
        if total == 0:
            return {"error": "No results to report"}
        
        # Basic stats
        exact_correct = sum(1 for r in results if r.is_correct)
        partial_correct = sum(1 for r in results if r.is_partial_correct)
        errors = sum(1 for r in results if r.pipeline_error)
        
        # Calculate metrics
        accuracy = exact_correct / total
        partial_accuracy = partial_correct / total
        
        # Timing stats
        durations = [r.total_duration_ms for r in results if r.total_duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # By difficulty (if available)
        by_difficulty = {}
        difficulty_groups = {}
        for r in results:
            if r.difficulty is not None:
                if r.difficulty not in difficulty_groups:
                    difficulty_groups[r.difficulty] = []
                difficulty_groups[r.difficulty].append(r)
        
        for diff, group in sorted(difficulty_groups.items()):
            correct = sum(1 for r in group if r.is_correct)
            total_in_group = len(group)
            by_difficulty[str(diff)] = {
                "total": total_in_group,
                "correct": correct,
                "accuracy": correct / total_in_group if total_in_group > 0 else 0
            }
        
        # Errors breakdown
        error_list = [
            {"id": r.question_id, "error": r.pipeline_error}
            for r in results if r.pipeline_error
        ]
        
        report = {
            "summary": {
                "total_questions": total,
                "exact_correct": exact_correct,
                "partial_correct": partial_correct,
                "accuracy_exact": accuracy,
                "accuracy_partial": partial_accuracy,
                "error_count": errors,
                "error_rate": errors / total
            },
            "timing": {
                "avg_duration_ms": avg_duration,
                "min_duration_ms": min_duration,
                "max_duration_ms": max_duration,
                "total_duration_seconds": sum(durations) / 1000 if durations else 0
            },
            "by_difficulty": by_difficulty,
            "errors": error_list,
            "results": [asdict(r) for r in results]
        }
        
        return report
    
    def save_report(self, results: list[EvaluationResult], filename: Optional[str] = None) -> str:
        """Lưu report ra file."""
        report = self.generate_report(results)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        
        # Also print summary
        summary = report["summary"]
        timing = report["timing"]
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Exact Match Accuracy: {summary['accuracy_exact']:.2%} ({summary['exact_correct']}/{summary['total_questions']})")
        print(f"Partial Match Accuracy: {summary['accuracy_partial']:.2%} ({summary['partial_correct']}/{summary['total_questions']})")
        print(f"Error Rate: {summary['error_rate']:.2%} ({summary['error_count']} errors)")
        print(f"\nTiming:")
        print(f"  Average: {timing['avg_duration_ms']:.0f}ms")
        print(f"  Min: {timing['min_duration_ms']:.0f}ms")
        print(f"  Max: {timing['max_duration_ms']:.0f}ms")
        
        if report.get("by_difficulty"):
            print(f"\nBy Difficulty:")
            for diff, stats in report["by_difficulty"].items():
                print(f"  Level {diff}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        if report.get("errors"):
            print(f"\nErrors ({len(report['errors'])}):")
            for err in report["errors"][:5]:
                print(f"  - {err['id']}: {err['error'][:50]}...")

        # Show voting details
        print(f"\nVoting Details (first 5 questions):")
        for r in results[:5]:
            runs_str = " | ".join([",".join(a) if a else "(empty)" for a in r.vote_runs])
            winning = ",".join(r.predicted_answer) if r.predicted_answer else "(empty)"
            match = "OK" if r.is_correct else ("~" if r.is_partial_correct else "X")
            print(f"  {match} {r.question_id}: [{runs_str}] -> {winning} | expected: {r.expected_answer}")

        print("=" * 60)

        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system with UIT testset (Mode Voting)")
    parser.add_argument("--testset", default="testset/uit_rag_testset_50.json",
                        help="Path to testset JSON file (default: 50 questions)")
    parser.add_argument("--output", default=None, help="Output filename")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N questions")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample N questions")
    parser.add_argument("--question", action="append", default=None,
                        help="Run specific question IDs (can repeat)")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for reports")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs for mode voting (default: 3)")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RAGEvaluator(
        testset_path=args.testset,
        output_dir=args.output_dir
    )

    # Run evaluation
    results = evaluator.evaluate_all(
        limit=args.limit,
        sample_size=args.sample,
        question_ids=args.question,
        n_runs=args.n_runs
    )

    # Save and print report
    output_path = evaluator.save_report(results, args.output)
    print(f"\nFull report: {output_path}")


if __name__ == "__main__":
    main()
