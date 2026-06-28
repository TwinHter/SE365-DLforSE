# UIT RAG System - Hệ thống Tra cứu Nội quy & Quy chế UIT

## Tổng quan Đề tài

**Mục tiêu:** Xây dựng hệ thống RAG chuyên biệt để tra cứu thông tin về Nội quy & Quy chế UIT (tuyển sinh, học phí, quy chế học vụ), khắc phục hallucination của LLM khi hỏi về các quy chế chuyên sâu, liên tục thay đổi theo năm (2024-2026).

**Vấn đề cốt lõi:** LLM như ChatGPT thường bị ảo giác hoặc thiếu thông tin cập nhật cục bộ khi hỏi về các quy chế chuyên sâu của UIT.

---

## 5 Chiến thuật Cốt lõi (Advanced RAG)

### 1. Table-to-Text Transformation

**Vấn đề:** Bảng biểu thô (Markdown/HTML) khi đưa vào Embedding rất dễ bị mất ngữ nghĩa liên kết giữa các hàng và cột.

**Chiến thuật:** Thay vì lưu bảng thô, hệ thống tự động chuyển dịch cấu trúc bảng thành các câu văn xuôi hoàn chỉnh, giàu ngữ cảnh mang tính tường thuật.

**Ví dụ:**
- Input (bảng thô): `| Mã ngành | Học phí |`
- Output (văn xuôi): "Đối với ngành Kỹ thuật Phần mềm hệ Chất lượng cao, định mức học phí áp dụng là X đồng tín chỉ. Đối với ngành Khoa học Máy tính hệ chính quy, định mức học phí là Y đồng tín chỉ..."

---

### 2. Hierarchical Chunking & Context Preservation

**Vấn đề:** Các chunk nhỏ khi đứng độc lập bị "mất gốc" - thiếu ngữ cảnh cấp cao.

**Chiến thuật:**
- Chia cắt văn bản theo cấu trúc phân cấp: **Chương > Điều > Khoản**
- Tự động nối tiêu đề bối cảnh cấp cao vào đầu mỗi chunk nhỏ
- Giúp Embedding hiểu chính xác ngữ cảnh của từng đoạn

**Ví dụ:**
- Chunk nhỏ: "Sinh viên phải đóng học phí trước ngày 15/9"
- Với context: "**Chương 3: Học phí và Thanh toán** | **Điều 12: Thời hạn đóng học phí** | Sinh viên phải đóng học phí trước ngày 15/9"

---

### 3. Adaptive RAG (Hard Filtering qua LLM Router)

**Vấn đề:** Không gian tìm kiếm quá rộng, nhiễu từ các năm khác, các ngành khác.

**Chiến thuật:**
- LLM Router đóng vai trò "bảo vệ cửa ngõ"
- Trích xuất **entities** liên quan để làm metadata filter
- Khóa chặt không gian tìm kiếm (đúng năm 2025, đúng ngành Khoa học Máy tính)
- Loại bỏ hoàn toàn nhiễu từ các năm/ngành khác

**Output Format:**
```json
{
  "label": "tuyensinh | chuongtrinhdaotao | danhmucmonhoc | hoatdong | nghiencuu | chung",
  "year": 2025,
  "entities": ["Khoa học Máy tính", "hệ Chất lượng cao"],
  "major_mapping": {
    "extracted": "Khoa học Máy tính",
    "code": "KHMT",
    "matched_chunks": ["KHMT"]
  }
}
```

---

### 4. Hybrid Search & Cross-Encoder Reranking

**Vấn đề:** Dense search (semantic) hoặc Sparse search (keyword) đơn lẻ đều không đủ chính xác.

**Chiến thuật:**
- **Dense Search:** Vector similarity (Cosine, Dot Product) - tìm theo ngữ nghĩa
- **Sparse Search:** BM25 - tìm theo từ khóa chính xác
- **RRF (Reciprocal Rank Fusion):** Trộn kết quả từ 2 bộ tìm kiếm
- **Cross-Encoder Reranker:** Chấm điểm lại độ tương quan sâu giữa câu hỏi và chunk

---

### 5. Self-RAG (Vòng lặp tự đánh giá - Critique)

**Vấn đề:** LLM có thể tạo câu trả lời không được hỗ trợ bởi context hoặc không đúng trọng tâm.

**Chiến thuật:** Sau khi LLM "nháp" câu trả lời, hệ thống kiểm tra nghiêm ngặt:
1. **Faithfulness:** Câu trả lời có được hỗ trợ 100% bởi Context không? (Chống ảo giác)
2. **Answer Relevance:** Có thực sự trả lời đúng trọng tâm câu hỏi không?
3. **Nếu vi phạm:** Buộc thử lại hoặc truy xuất thêm context

---

## 6 Nhãn Phẳng (Flat Labels)

| Label | Mô tả | Ví dụ |
|-------|-------|-------|
| `tuyensinh` | Thông tin tuyển sinh | Điểm chuẩn, học phí, phương thức xét tuyển đầu vào |
| `chuongtrinhdaotao` | Chương trình đào tạo | Khung chương trình, lộ trình học, giới thiệu chung về ngành |
| `danhmucmonhoc` | Danh mục môn học | Mã môn học, số tín chỉ, môn tiên quyết, nội dung môn học |
| `hoatdong` | Hoạt động sinh viên | Phong trào đoàn hội, chiến dịch tình nguyện, câu lạc bộ, sự kiện |
| `nghiencuu` | Nghiên cứu khoa học | Nghiên cứu khoa học sinh viên, cuộc thi học thuật, bài báo |
| `chung` | Thông tin chung | Chào hỏi, quy chế chung, hoặc **null / không xác định** |

> **Lưu ý:** Nếu `category` là `null` hoặc không thuộc 5 nhãn trên, mặc định gán label = `chung` hoặc là áp dụng cho tất cả.

---

## Kiến trúc Pipeline

```
User Query
    │
    ▼
┌─────────────────┐
│   LLM Router    │ ← Trích xuất label, year, entities + major_mapping
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Hard Filter (Adaptive)      │ ← Lọc theo label + year + major_code
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│       Hybrid Search             │ ← Chiến thuật 4
│  Dense (Vector) + Sparse (BM25) │
│         + RRF Fusion            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Cross-Encoder Reranker        │ ← Top 20 → Top 5
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      LLM Generator              │ ← Sinh câu trả lời
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Self-RAG Critique          │ ← Chiến thuật 5
│  Faithfulness + Relevance Check │
└────────┬────────────────────────┘
         │
         ▼
   Output + Citation + [chunk_id]
```

---

## Cấu trúc Project

```
uit_rag_system/
├── docs/                  # Tài liệu và notes
├── chunk/                 # Các chunk đã xử lý theo 6 nhãn
│   ├── tuyensinh/
│   ├── chuongtrinhdaotao/
│   ├── danhmucmonhoc/
│   ├── hoatdong/
│   ├── nghiencuu/
│   └── chung/
├── src/                   # Source code
│   ├── config.py          # Cấu hình chung
│   ├── major_matcher.py  # Major Matching (CTĐT ↔ Danh mục môn học)
│   ├── table2text.py      # Table-to-Text Transformation
│   ├── chunker.py         # Hierarchical Chunking
│   ├── router.py          # LLM Router
│   ├── retrieval.py       # Hybrid Search
│   ├── reranker.py        # Cross-Encoder Reranker
│   ├── generator.py        # LLM Generator
│   └── critique.py        # Self-RAG Critique
├── testset/               # Dataset đánh giá
└── app.py                 # Streamlit Demo
```

---

## Nguyên tắc Nền tảng

### DRY (Don't Repeat Yourself)
Không bao giờ lặp lại code. Nếu thấy mình copy-paste logic từ file này sang file khác, cần rút ra thành hàm hoặc class dùng chung.

### SRP (Single Responsibility Principle)
Mỗi hàm, file, hoặc class chỉ nên làm **đúng một việc**. Chia nhỏ thay vì gộp chung.

### Hard Filter First
Luôn áp dụng Hard Filter (từ Router) **TRƯỚC** khi hybrid search để giảm nhiễu và tăng tốc độ.

### Context Preservation
Mọi chunk phải giữ được ngữ cảnh cấp cao (tiêu đề Chương/Điều) để không bị "mất gốc".

---

## Metadata Schema (Milvus)

| Field | Type | Mô tả |
|-------|------|-------|
| `chunk_id` | PK | Unique identifier |
| `vector` | Dense Float | Embedding vector (semantic) |
| `sparse_vector` | Sparse | BM25 sparse vector |
| `content` | Text | Nội dung đã qua Table-to-Text + Hierarchical Chunking |
| `category` | String | Một trong 6 nhãn phẳng |
| `year` | Int | Năm áp dụng (2024, 2025, 2026) |
| `major_code` | String | Mã viết tắt ngành (VD: `KHMT`, `HTTT`, `CNPM`) |
| `major_fullname` | String | Tên đầy đủ ngành (VD: `Khoa học Máy tính`) |
| `program_type` | String | `chinh_quy`, `chat_luong_cao`, `lien_thong`, `viet_nhat` |
| `parent_document` | String | Tài liệu gốc |

---

## 7 Major Matching System

**Vấn đề:** Cùng một ngành nhưng có 2 cách gọi khác nhau:
- **CTĐT:** Tên đầy đủ tiếng Việt (VD: "Khoa học Máy tính", "Hệ thống Thông tin")
- **Danh mục môn học:** Mã viết tắt (VD: `KHMT`, `HTTT`)

Hệ thống cần matching chính xác để query đúng chunks.

**Quy tắc cốt lõi:**
- Khi `category = chuongtrinhdaotao` → `major` phải nằm trong **16 ngành CTĐT** (MỖI NGÀNH LÀ 1 ENTITY RIÊNG BIỆT)
- Khi `category = danhmucmonhoc` → `major` phải nằm trong **14 mã viết tắt** (raw value trong metadata chunk)

> **Quan trọng:** Trong metadata chunk thực tế, `majors` của `danhmucmonhoc` được lưu dưới dạng **raw name** có dạng `"Tên đầy đủ (CODE)"` (VD: `"Hệ thống thông tin (HTTT)"`, `"Khoa học máy tính (KHMT)"`). Hệ thống cần normalize về dạng chuẩn (chỉ mã) trước khi filter.

---

### 7.1 Phân biệt Majors theo Category

**QUAN TRỌNG:** Majors được phân thành 2 loại riêng biệt tùy theo category, **không được trộn lẫn**:

#### A. CHƯƠNG TRÌNH ĐÀO TẠO (`chuongtrinhdaotao`)
→ **16 ngành CTĐT** - MỖI NGÀNH LÀ 1 ENTITY RIÊNG BIỆT
→ Theo Danh mục Tuyển sinh UIT (data thực tế từ `chunk/chuong_trinh_dao_tao/chunks_ctdt.jsonl`)

| # | Tên ngành đầy đủ | Mã ngành |
|---|------------------|----------|
| 1 | An toàn Thông tin | 7440202 |
| 2 | Công nghệ Thông tin | 7340102 |
| 3 | Công nghệ Thông tin (Liên thông) | 7340103 |
| 4 | Công nghệ Thông tin (Việt - Nhật) | - |
| 5 | Công nghệ Thông tin (Văn bằng 2) | - |
| 6 | Hệ thống Thông tin | 7440101 |
| 7 | Hệ thống Thông tin (Chương trình tiên tiến) | 7440102 |
| 8 | Khoa học Máy tính | 7340101 |
| 9 | Kỹ thuật Hệ thống Máy tính (Liên kết Newcastle) | 7440128TN |
| 10 | Kỹ thuật Phần mềm | 7440104 |
| 11 | Mạng máy tính và An toàn thông tin (Liên kết BCU) | 7480102 |
| 12 | Mạng máy tính và Truyền thông dữ liệu | 7480101 |
| 13 | Thiết kế Vi mạch | 7520203 |
| 14 | Thương mại điện tử | 7340101TN |
| 15 | Truyền thông đa phương tiện | 7340104 |
| 16 | Trí tuệ Nhân tạo | 7480203 |

#### B. DANH MỤC MÔN HỌC (`danhmucmonhoc`)
→ **14 raw values** trong `metadata.majors` của chunk (theo data thực tế từ `chunk/danh_muc_mon_hoc/chunk_final_daa_courses.jsonl`)

| # | Raw value (metadata.majors) | Mã chuẩn | Ghi chú |
|---|------------------------------|----------|---------|
| 1 | `BMAV` | `BMAV` | Bộ môn Anh Văn (Tiếng Anh) - common |
| 2 | `BMTL` | `BMTL` | Bộ môn Ngoại ngữ - common |
| 3 | `GDQP` | `GDQP` | Giáo dục Quốc phòng - common |
| 4 | `Giáo dục thể chất (GDTC)` | `GDTC` | Thể dục - common |
| 5 | `Hệ thống thông tin (HTTT)` | `HTTT` | Hệ thống thông tin |
| 6 | `Khoa học máy tính (KHMT)` | `KHMT` | Khoa học Máy tính |
| 7 | `Khoa học và Kỹ thuật thông tin (KTTT)` | `KTTT` | Khoa học và Kỹ thuật thông tin |
| 8 | `Kỹ thuật máy tính (KTMT)` | `KTMT` | Kỹ thuật Máy tính |
| 9 | `Kỹ thuật phần mềm (CNPM)` | `CNPM` | Kỹ thuật Phần mềm |
| 10 | `MMT` | `MMT` | Mạng máy tính |
| 11 | `Mạng máy tính và truyền thông dữ liệu (MMT&TT)` | `MMT&TT` | Mạng máy tính và TT dữ liệu |
| 12 | `P.ĐTĐH` | `P.ĐTĐH` | Phòng Đào tạo ĐH - alias |
| 13 | `PĐTĐH` | `PĐTĐH` | Phòng Đào tạo ĐH - common |
| 14 | `TTNN` | `TTNN` | Cơ sở ngành (Tiếng Nhật) - common |

**Nhóm ngành chuyên ngành (8 mã):** `KHMT`, `HTTT`, `CNPM`, `KTMT`, `MMT`, `MMT&TT`, `KTTT`, `TTNN`

**Nhóm môn chung (6 mã):** `BMAV`, `BMTL`, `PĐTĐH`, `P.ĐTĐH`, `GDTC`, `GDQP`

> **Lưu ý 1:** `TTNN` được phân loại vào nhóm chuyên ngành theo data, nhưng vẫn là môn chung (common) mà **tất cả sinh viên** đều phải học.
>
> **Lưu ý 2:** `PĐTĐH` và `P.ĐTĐH` là cùng một đơn vị hành chính — phải chuẩn hóa về `PĐTĐH` khi filter.
>
> **Lưu ý 3:** Các raw value dạng `"Tên đầy đủ (CODE)"` (VD: `"Hệ thống thông tin (HTTT)"`) cần được normalize về mã chuẩn (`HTTT`) trước khi filter Milvus.

---

### 7.2 Mapping CTĐT ↔ Danh mục

| Tên CTĐT | Mã trong Danh mục |
|----------|-------------------|
| An toàn Thông tin | `MMT` |
| Công nghệ Thông tin | `KHMT`, `HTTT`, `CNPM`, `MMT` |
| Công nghệ Thông tin (Liên thông) | `KHMT`, `HTTT`, `CNPM`, `MMT` |
| Công nghệ Thông tin (Việt - Nhật) | `KHMT`, `HTTT`, `CNPM`, `MMT` |
| Công nghệ Thông tin (Văn bằng 2) | `KHMT`, `HTTT`, `CNPM`, `MMT` |
| Hệ thống Thông tin | `HTTT` |
| Hệ thống Thông tin (Chương trình tiên tiến) | `HTTT` |
| Khoa học Máy tính | `KHMT` |
| Kỹ thuật Hệ thống Máy tính (Liên kết Newcastle) | `KTMT` |
| Kỹ thuật Phần mềm | `CNPM` |
| Mạng máy tính và An toàn thông tin (Liên kết BCU) | `MMT` |
| Mạng máy tính và Truyền thông dữ liệu | `MMT`, `MMT&TT` |
| Thiết kế Vi mạch | `KTMT` |
| Thương mại điện tử | `CNPM` |
| Truyền thông đa phương tiện | `CNPM` |
| Trí tuệ Nhân tạo | `KHMT`, `TTNN` |

---

### 7.3 Fallback Logic (Khi không match được)

1. **Partial Match:** Nếu user hỏi "HTTT" nhưng chunk có "Hệ thống Thông tin" → **MATCH**
2. **Substring Match:** "Hệ thống" → `HTTT`, "Khoa học" → `KHMT`
3. **Alias Match:** "CNTT" → `Công nghệ Thông tin` → map sang `KHMT`, `HTTT`, `CNPM`, `MMT`
4. **Common Courses Detection:**
   - Nếu query chứa `PĐTĐH`, `P.ĐTĐH` → map sang `PĐTĐH` (unified)
   - Nếu query là môn Tiếng Anh, Ngoại ngữ → match với `BMAV`, `BMTL`
   - Nếu query là Thể chất, Quốc phòng → match với `GDTC`, `GDQP`
   - Nếu query có "tiếng nhật", "nhật bản", "việt - nhật" → match với `TTNN`
5. **Broad Match (CTĐT):** Nếu label = `chuongtrinhdaotao` + major không xác định → lấy tất cả chunks thuộc CTĐT
6. **Broad Match (Danh mục):** Nếu label = `danhmucmonhoc` + major không xác định → fallback về major phổ biến nhất (`KHMT`, `CNPM`, `HTTT`)
7. **Thứ tự ưu tiên cho CNTT variants:** Vì `"Công nghệ Thông tin"` là substring của `"Công nghệ Thông tin (Việt - Nhật)"`, khi trích xuất major phải check các biến thể **dài hơn trước** (Việt - Nhật, Văn bằng 2, Liên thông), sau đó mới đến CNTT chính quy.

---

### 7.4 Query Routing Examples

**Query:** "Môn học nào về Hệ thống Thông tin?"

```json
{
  "label": "danhmucmonhoc",
  "year": 2025,
  "entities": ["HTTT", "Hệ thống Thông tin"],
  "major_mapping": {
    "extracted": "Hệ thống Thông tin",
    "code": "HTTT",
    "matched_chunks": "HTTT"
  }
}
```

**Query:** "Chương trình đào tạo Khoa học Máy tính"

```json
{
  "label": "chuongtrinhdaotao",
  "year": 2025,
  "entities": ["Khoa học Máy tính"],
  "major_mapping": {
    "extracted": "Khoa học Máy tính",
    "matched_chunks": ["7340101"]
  }
}
```

**Query:** "Môn Tiếng Anh nâng cao có mấy tín chỉ?"

```json
{
  "label": "danhmucmonhoc",
  "year": 2025,
  "entities": ["Tiếng Anh"],
  "major_mapping": {
    "extracted": "Tiếng Anh",
    "code": "BMAV",
    "is_common_course": true,
    "matched_chunks": ["BMAV"]
  }
}
```

**Query:** "Chương trình đào tạo Công nghệ Thông tin Việt - Nhật"

```json
{
  "label": "chuongtrinhdaotao",
  "year": 2025,
  "entities": ["Công nghệ Thông tin (Việt - Nhật)"],
  "major_mapping": {
    "extracted": "Công nghệ Thông tin (Việt - Nhật)",
    "matched_chunks": ["7340102VN"]
  }
}
```

> **Giải thích:** Khi `label = chuongtrinhdaotao`, `matched_chunks` chứa **mã ngành tuyển sinh** (VD: `7340101`, `7340102VN`, `7440102`...) để filter chính xác trên `metadata.major` của chunks CTĐT. Bảng `CTDT_CODE_TO_NAME` cho phép lookup ngược từ mã → tên ngành khi cần hiển thị.

**Query:** "Môn học do P.ĐTĐH quản lý"

```json
{
  "label": "danhmucmonhoc",
  "year": 2025,
  "entities": ["P.ĐTĐH"],
  "major_mapping": {
    "extracted": "P.ĐTĐH",
    "code": "PĐTĐH",
    "is_common_course": true,
    "note": "P.ĐTĐH và PĐTĐH là cùng một đơn vị",
    "matched_chunks": ["PĐTĐH"]
  }
}
```

---

## 8 Evaluation Metrics

| Metric | Mô tả |
|--------|-------|
| **Recall@k** | Retrieval Accuracy - chunk đúng có trong Top k? |
| **Citation Accuracy** | Trích dẫn đúng điều khoản gốc? |
| **Answer Correctness** | Câu trả lời chính xác? |
| **Faithfulness Rate** | Tỷ lệ câu trả lời được hỗ trợ 100% bởi context |
| **Major Match Rate** | Tỷ lệ major được match đúng khi query |
