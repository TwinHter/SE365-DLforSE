# UIT RAG System

Hệ thống **Retrieval-Augmented Generation** chuyên tra cứu Nội quy & Quy chế UIT (tuyển sinh, học phí, chương trình đào tạo, danh mục môn học, hoạt động sinh viên, nhóm nghiên cứu). Giải quyết hiện tượng LLM hallucination khi trả lời các quy chế chuyên sâu, thay đổi theo năm.

---

## Cấu trúc Project

```
uit_rag_system/
├── chunk/                          # ← Dữ liệu chính dùng cho RAG
│   ├── tuyen_sinh/chunks_tuyensinh.jsonl
│   ├── chuong_trinh_dao_tao/chunks_ctdt.jsonl
│   ├── danh_muc_mon_hoc/chunk_final_daa_courses.jsonl
│   ├── cong_tac_sinh_vien/chunk_final_uit_*.jsonl
│   └── nhom_nghien_cuu/chunk_final_*.jsonl
├── src/
│   ├── config.py                   # Đường dẫn, embedding model
│   └── setup.py                    # Pipeline load → embed → save
├── testset/uit_rag_testset_80_balanced_userstyle.json
├── volumes/                        # Bind-mount cho Docker
├── docker-compose.yml              # Milvus standalone
├── requirements.txt
└── README.md
```

---

## 1. Dữ liệu `chunk/*.jsonl`

Mỗi dòng JSONL là 1 chunk đã qua Table-to-Text + Context Preservation, sẵn sàng để embed.

### 1.1 Mapping folder → nhãn phẳng

| Folder | Label | Schema field đặc trưng |
|---|---|---|
| `tuyen_sinh/` | `tuyensinh` | `chunk_id`, `parent_document`, `hierarchical_path`, `content`, `category=tuyensinh`, `major=null`, `cohort`, `year` |
| `chuong_trinh_dao_tao/` | `chuongtrinhdaotao` | `chunk_id`, `parent_document`, `hierarchical_path`, `content`, `category=chuong_trinh_dao_tao`, `major=[tên đầy đủ]`, `cohort`, `year`, `url` |
| `danh_muc_mon_hoc/` | `danhmucmonhoc` | `id`, `content`, `metadata{majors,category,year}`, `chunk_id`, `chunk_content` (đã enrich ngữ cảnh phân cấp + từ khoá) |
| `cong_tac_sinh_vien/` | `hoatdong` | `id`, `content`, `metadata{category,major,year}`, `chunk_id`, `chunk_content` |
| `nhom_nghien_cuu/` | `nghiencuu` | `id`, `content`, `metadata{category,major,year,status}`, `chunk_id`, `chunk_content` |
| `chung/` (chưa có) | `chung` | Áp dụng cho mọi ngành / null |

### 1.2 Xét logic chunk theo domain

**`tuyen_sinh` + `chuong_trinh_dao_tao`** (chunk schema A — top-level fields)

- Văn bản gốc lấy từ `tuyensinh.uit.edu.vn` / `student.uit.edu.vn`, đã parse thủ công / semi-auto.
- Mỗi chunk = một đoạn văn xuôi đã qua **Table-to-Text**: ví dụ bảng "Mã ngành | Học phí" được chuyển thành câu "Đối với ngành … định mức học phí là …".
- Giữ **`hierarchical_path`** để bảo toàn ngữ cảnh cấp cao khi embed (Chiến thuật 2 — Context Preservation).
- `chunk_id` có dạng `<domain>_<slug>_<year>_<index>` (VD: `ctdt_cntt_2024_0001`).
- `major` để dạng list cho phép filter theo ngành; `null` ở `tuyensinh` (áp dụng chung).

**`danh_muc_mon_hoc` + `cong_tac_sinh_vien` + `nhom_nghien_cuu`** (chunk schema B — wrapper `metadata`)

- Crawl HTML bằng các script trong `raw/<domain>/crawl/`, sau đó parse sang JSONL.
- Mỗi record chứa **`content`** (gốc) và **`chunk_content`** (đã enrich).
- Cấu trúc `chunk_content` chuẩn hoá 4 lớp ngữ cảnh (xem `crawl_daa_courses.py::build_content`):
  ```
  Tài liệu: <parent_document>
  Phân cấp: <hierarchical_path>
  Loại dữ liệu: <category>; năm dữ liệu: <year>; chương trình: <program>
  Ngữ cảnh truy xuất: <mô tả>
  Nội dung gốc: <content>
  Từ khoá hỗ trợ tìm kiếm: <keyword1>; <keyword2>; …
  ```
- `metadata.majors` ở `danh_mucmonhoc` được lưu dạng `"Tên đầy đủ (CODE)"` (VD: `"Hệ thống thông tin (HTTT)"`). Giữ nguyên không cần normalize.
- `chunk_id` có dạng `<label>_<slug>_<year>_<index>` (VD: `danhmucmonhoc_httt_2026_0001`).

### 1.3 Chuẩn hoá category

Được chuẩn hoá trong `src/setup.py::_FOLDER_TO_LABEL` / `_CATEGORY_NORMALIZE`:

```
tuyen_sinh               → tuyensinh
chuong_trinh_dao_tao     → chuongtrinhdaotao
danh_muc_mon_hoc         → danhmucmonhoc
cong_tac_sinh_vien       → hoatdong
cau_lac_bo / sinh_vien   → hoatdong
nhom_nghien_cuu          → nghiencuu
null / không xác định    → chung
```

### 1.4 Ví dụ một chunk `danh_muc_mon_hoc`

```json
{
  "id": 1,
  "link": "https://daa.uit.edu.vn/danh-muc-mon-hoc-dai-hoc",
  "parent_document": "Danh mục môn học đại học UIT",
  "content": "Môn học Hệ thống thông tin kế toán (ACCT3603) có tên tiếng Anh là Accounting Information Systems. Môn học này hiện đang mở lớp. Môn học thuộc đơn vị quản lý chuyên môn Hệ thống thông tin (HTTT). Loại môn học là CN. Số tín chỉ lý thuyết là 3, số tín chỉ thực hành là 0.",
  "metadata": {
    "category": "danhmucmonhoc",
    "majors": ["Hệ thống thông tin (HTTT)"],
    "subcategory": "Chuyên ngành (CN)",
    "year": 2026,
    "program": null
  },
  "chunk_id": "danhmucmonhoc_httt_2026_0001",
  "chunk_content": "Tài liệu: Danh mục môn học đại học UIT.\nPhân cấp: Danh mục môn học đại học UIT > Hệ thống thông tin (HTTT) > Chuyên ngành (CN) > Hệ thống thông tin kế toán (ACCT3603).\nLoại dữ liệu: danhmucmonhoc; năm dữ liệu: 2026; chương trình: Không xác định.\nNgữ cảnh truy xuất: …\nNội dung gốc: …\nTừ khoá hỗ trợ tìm kiếm: ACCT3603; Hệ thống thông tin kế toán; HTTT; …"
}
```

### 1.5 Ví dụ một chunk `chuong_trinh_dao_tao`

```json
{
  "chunk_id": "ctdt_cntt_2024_0001",
  "parent_document": "Chương trình đào tạo Cử nhân ngành Công nghệ Thông tin Khóa 19 - 2024",
  "hierarchical_path": "CTĐT UIT / Cử nhân ngành Công nghệ Thông tin / Giới thiệu chung",
  "content": "Chương trình đào tạo Cử nhân chính quy ngành Công nghệ Thông tin áp dụng từ Khóa 19 năm 2024 tại Trường Đại học Công nghệ Thông tin (UIT) đặt ra mục tiêu đào tạo …",
  "category": "chuong_trinh_dao_tao",
  "sub_category": "chi_tiet_nganh",
  "major": ["Công nghệ Thông tin"],
  "cohort": 19,
  "year": 2024,
  "url": "https://student.uit.edu.vn/content/cu-nhan-nganh-cong-nghe-thong-tin-ap-dung-tu-khoa-19-2024"
}
```

---

## 2. Pipeline Embedding (`src/setup.py`)

```
chunk/<category>/*.jsonl
        │
        │ load_chunks()         → chuẩn hoá category, rút major/year
        ▼
list[dict]  (chunk_id, content, category, year, major, parent_document)
        │
        │ load_embedding_model() → keepitreal/vietnamese-sbert (768d, CPU)
        │ embed_chunks(batch=32) → list[list[float]]
        ▼
embedded/embedded_chunks.jsonl  (chunk + dense_vector)
```

### Chạy

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

python -m src.setup
# → embedded/embedded_chunks.jsonl
```

Cấu hình tại `src/config.py`:

```python
CHUNK_DIR        = PROJECT_ROOT / "chunk"
OUTPUT_DIR       = PROJECT_ROOT / "embedded"
EMBEDDING_MODEL  = "keepitreal/vietnamese-sbert"
EMBEDDING_DEVICE = "cpu"             # đổi sang "cuda" nếu có GPU
EMBEDDING_DIMENSION = 768
```

---

## 3. Khởi tạo Docker (Milvus)

Stack `docker-compose.yml`: **etcd** (metadata) + **minio** (object storage) + **milvus-standalone** (gRPC :19530, metrics :9091).

```bash
# 1. Tạo volume (chỉ cần làm lần đầu)
mkdir -p volumes/etcd volumes/milvus volumes/minio

# 2. Khởi tạo stack
docker compose up -d
docker compose ps

# 3. Tuỳ chọn: trỏ volume ra thư mục khác
#    PowerShell:  $env:DOCKER_VOLUME_DIR = "$PWD\volumes"
#    Bash:        export DOCKER_VOLUME_DIR=$PWD/volumes

# 4. Kết nối từ Python
python -c "from pymilvus import MilvusClient; print(MilvusClient(uri='http://localhost:19530').list_collections())"
```

Thao tác thường dùng:

```bash
docker compose logs -f milvus-standalone   # xem log
docker compose stop                        # dừng (giữ data)
docker compose down                        # tắt container (giữ volume)
docker compose down -v                     # xoá toàn bộ data
```

---

## 4. 6 Nhãn phẳng

| Label | Mô tả |
|---|---|
| `tuyensinh` | Tuyển sinh, học phí, phương thức xét tuyển |
| `chuongtrinhdaotao` | Khung CTĐT, lộ trình học |
| `danhmucmonhoc` | Mã môn, tín chỉ, môn tiên quyết |
| `hoatdong` | CLB, tình nguyện, handbook SV |
| `nghiencuu` | Nhóm NC, giảng viên hướng dẫn |
| `chung` | Quy chế chung / null |

---

## 5. Hướng dẫn chạy nhanh

```bash
# Cài môi trường
python -m venv venv && .\venv\Scripts\activate
pip install -r requirements.txt

# Khởi tạo Milvus
docker compose up -d

# Sinh embedding từ chunk/*.jsonl
python -m src.setup
# → embedded/embedded_chunks.jsonl

# Nạp vào Milvus + chạy retrieval (router → hybrid search → reranker)
# (sẽ bổ sung app.py / script nạp collection)
```

---

## 6. Đánh giá

- Bộ test: `testset/uit_rag_testset_80_balanced_userstyle.json` (80 câu hỏi).
- Metrics: Recall@k, Citation Accuracy, Answer Correctness, Faithfulness Rate, Major Match Rate.
