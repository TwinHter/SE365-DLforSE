"""
LLM Utilities - DeepSeek API Client.
Handles all LLM calls with detailed logging.
"""

import json
import logging
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class DeepSeekClient:
    """Client for DeepSeek API compatible endpoints."""

    def __init__(self):
        self.api_url = os.getenv("DEEPSEEK_API_URL", "https://api.xah.io/v1/chat/completions")
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "deepseek-v4-flash")
        self.provider = os.getenv("LLM_PROVIDER", "deepseek")

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"DeepSeekClient initialized with model: {self.model}")
        logger.info(f"API URL: {self.api_url}")

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> tuple[str, list[dict]]:
        """
        Call LLM and return (response_text, full_usage).

        Returns:
            tuple: (response_text, usage_info)
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            logger.info(f"LLM call successful - tokens: {usage.get('total_tokens', 'N/A')}")
            return content, usage

        except requests.exceptions.Timeout:
            logger.error("LLM API timeout")
            return "Xin lỗi, yêu cầu bị timeout. Vui lòng thử lại.", {}
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API error: {e}")
            return f"Lỗi API: {str(e)}", {}
        except (KeyError, IndexError) as e:
            logger.error(f"LLM response parse error: {e}")
            return f"Lỗi parse response: {str(e)}", {}


# Global client instance
_client: DeepSeekClient | None = None


def get_llm_client() -> DeepSeekClient:
    """Get or create the global LLM client."""
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client


def parse_json_safely(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Remove markdown code block markers
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the text
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """Bạn là trợ lý AI của Trường Đại học Công nghệ Thông tin (UIT).
Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và trích xuất thông tin cấu trúc.

Luôn trả lời bằng tiếng Việt và dựa trên ngữ cảnh được cung cấp.
Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ điều đó."""


REPHRASE_PROMPT = """Hãy diễn đạt lại câu hỏi sau thành dạng rõ ràng, chi tiết hơn để phục vụ việc tìm kiếm trong tài liệu:

QUAN TRỌNG: Giữ nguyên phạm vi của câu hỏi gốc. KHÔNG tự ý thêm các cụm từ như:
- "tại các trường đại học"
- "ở Việt Nam"
- "công lập và tư thục"
- Hoặc bất kỳ thông tin nào không có trong câu hỏi gốc

Chỉ diễn đạt lại chính xác những gì câu hỏi hỏi, thêm chi tiết từ chính câu hỏi.

Câu hỏi gốc: {question}

Câu hỏi đã diễn đạt lại (viết bằng tiếng Việt, đầy đủ và rõ ràng, GIỮ NGUYÊN phạm vi):"""


EXTRACT_JSON_A_PROMPT = """Dựa vào câu hỏi sau, trích xuất thông tin cấu trúc:

Câu hỏi: {question}

Trả lời theo định dạng JSON sau (chỉ trả JSON, không giải thích):
{{
    "question": "câu hỏi đã chuẩn hóa",
    "category": "tuyensinh | chuongtrinhdaotao | danhmucmonhoc | hoatdong | nghiencuu | chung",
    "year": năm (mặc định 2026),
    "major": "KHMT | ATTT | CNPM | HTTT | KTMT | MMT | CNTT | TTNT | ... hoặc null nếu không xác định được",
    "isAboutUIT": "Yes" hoặc "No",
    "reasoning": "giải thích ngắn về why bạn chọn category, year và major"
}}

Quy tắc xác định major:
- "Khoa học máy tính", "Khoa hoc may tinh", "KHMT", "Computer Science" -> "KHMT"
- "An toàn thông tin", "ATTT", "Cybersecurity" -> "ATTT"
- "Công nghệ phần mềm", "CNPM", "Software Engineering" -> "CNPM"
- "Hệ thống thông tin", "HTTT", "Information Systems" -> "HTTT"
- "Kỹ thuật máy tính", "KTMT", "Computer Engineering" -> "KTMT"
- "Mạng máy tính", "MMT", "Networking" -> "MMT"
- "Công nghệ thông tin", "CNTT", "IT" -> "CNTT"
- "Trí tuệ nhân tạo", "TTNT", "AI" -> "TTNT"
- Nếu hỏi về ngành/chuyên ngành cụ thể, LUÔN xác định major
- Nếu câu hỏi chung chung không liên quan đến ngành cụ thể -> null

Quy tắc xác định isAboutUIT:
- "Yes": Câu hỏi đề cập UIT, các thuật ngữ chung của UIT (trường, khoa, ngành, học phí, tuyển sinh, chương trình đào tạo...), HOẶC câu hỏi không nói rõ trường nào nhưng ngữ cảnh gợi ý là UIT
- "No": Câu hỏi đề cập RÕ RÀNG một trường đại học CỤ THỂ KHÁC (VD: ĐH Bách Khoa, ĐH KHTN, FPT, VNU...), HOẶC câu hỏi hoàn toàn không liên quan đến giáo dục đại học

QUAN TRỌNG: Nếu câu hỏi hỏi về "ngành KHMT", "học phí", "tuyển sinh", "chương trình đào tạo" mà không nói tên trường cụ thể -> IS UIT vì hệ thống này phục vụ UIT!

Quy tắc xác định category:
- Nếu câu hỏi về tuyển sinh, điểm chuẩn, học phí -> category = "tuyensinh"
- Nếu câu hỏi về chương trình đào tạo, khung chương trình -> category = "chuongtrinhdaotao"
- Nếu câu hỏi về môn học, tín chỉ, mã môn -> category = "danhmucmonhoc"
- Nếu câu hỏi về hoạt động sinh viên, câu lạc bộ -> category = "hoatdong"
- Nếu câu hỏi về nghiên cứu khoa học, nhóm nghiên cứu -> category = "nghiencuu"
- Các câu hỏi khác, chung chung -> category = "chung"
"""


GENERATE_ANSWER_PROMPT = """Dựa vào ngữ cảnh sau để trả lời câu hỏi:

Câu hỏi: {question}

Ngữ cảnh (các đoạn tài liệu liên quan):
---
{context}
---

Trả lời theo định dạng JSON sau:
{{
    "Answer": "câu trả lời ngắn gọn (1-2 câu)",
    "Explanation": "giải thích chi tiết dựa trên ngữ cảnh",
    "SupportContext": "trích dẫn ngắn từ ngữ cảnh hỗ trợ câu trả lời"
}}

Yêu cầu:
- Answer phải ngắn gọn, đủ ý
- Explanation phải dựa TRỰC TIẾP vào ngữ cảnh được cung cấp
- Không bịa đặt thông tin không có trong ngữ cảnh
- Nếu ngữ cảnh không đủ -> nói rõ không tìm thấy thông tin
"""


SUPPORT_CHECK_PROMPT = """Đánh giá xem câu trả lời sau có được hỗ trợ đầy đủ bởi ngữ cảnh không:

Câu hỏi: {question}
Câu trả lời: {answer}
Ngữ cảnh: {context}

Trả lời theo định dạng JSON:
{{
    "isSupported": true/false,
    "confidence": 0.0 - 1.0,
    "reason": "giải thích ngắn"
}}

Nếu isSupported = false, hệ thống sẽ tự động tìm thêm ngữ cảnh và thử lại."""
