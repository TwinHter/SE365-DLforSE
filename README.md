# UIT RAG System - Hệ thống Tra cứu Nội quy & Quy chế UIT 🎓🤖

Hệ thống **Retrieval-Augmented Generation (RAG)** nâng cao chuyên tra cứu Nội quy & Quy chế của Trường Đại học Công nghệ Thông tin (UIT) (bao gồm tuyển sinh, học phí, chương trình đào tạo, danh mục môn học, hoạt động sinh viên, và nhóm nghiên cứu). 

Dự án áp dụng các kỹ thuật RAG tiên tiến (Advanced RAG) nhằm giải quyết triệt để hiện tượng ảo giác (hallucination) của Mô hình ngôn ngữ lớn (LLM) đối với các văn bản quy chế phức tạp, chuyên sâu và thường xuyên thay đổi theo khóa học/năm học (từ 2024 đến 2026).

---

## 🚀 5 Chiến thuật RAG Nâng cao Đang Áp Dụng

1. **Table-to-Text Transformation**: Bảng biểu (Markdown/HTML) được chuyển đổi tự động thành các đoạn văn xuôi giàu ngữ cảnh, giúp mô hình embedding nắm bắt chính xác mối quan hệ giữa các hàng/cột thay vì chỉ embed bảng thô.
2. **Hierarchical Chunking & Context Preservation**: Phân đoạn tài liệu theo cấu trúc **Chương > Điều > Khoản** và tự động đính kèm thông tin bối cảnh cấp cao vào từng chunk nhỏ để tránh hiện tượng "mất bối cảnh".
3. **Adaptive RAG (Hard Filtering qua LLM Router)**: LLM đóng vai trò phân tích truy vấn của người dùng, xác định nhãn phẳng (`category`), năm (`year`) và ngành học (`major`) để lọc cứng (hard filter) không gian tìm kiếm trước khi thực hiện truy xuất, giảm tối đa nhiễu thông tin chéo năm/ngành.
4. **Hybrid Search & Cross-Encoder Reranking**: Kết hợp tìm kiếm ngữ nghĩa Dense (Vector Search trên Milvus) và tìm kiếm từ khóa Sparse (BM25 Search) qua thuật toán **Reciprocal Rank Fusion (RRF)**, sau đó dùng **Cross-Encoder Reranker** để tái xếp hạng độ tương quan sâu giữa câu hỏi và các ứng viên.
5. **Self-RAG (Vòng lặp tự đánh giá - Critique)**: Kiểm tra nghiêm ngặt câu trả lời nháp từ LLM dựa trên hai tiêu chí: **Faithfulness** (độ trung thực - câu trả lời có được hỗ trợ 100% bởi context không) và **Relevance** (độ liên quan - câu trả lời có đúng trọng tâm câu hỏi không). Nếu không đạt, hệ thống sẽ tự động điều chỉnh và thử lại (Retry).

### 📊 Số lượng chunk qua từng giai đoạn của Pipeline

Luồng truy xuất và lọc dữ liệu (Retrieval Pipeline) được tối ưu hóa số lượng chunk qua từng giai đoạn để đạt hiệu quả cao nhất:

| Giai đoạn | Hành động | Số lượng chunk còn lại | Mô tả / Vai trò |
| :--- | :--- | :---: | :--- |
| **0. Toàn bộ CSDL** | Khởi tạo | **~1,500+** | Toàn bộ dữ liệu chunk từ tất cả các danh mục trong hệ thống. |
| **1. Hard Filter** | LLM Router + Metadata | **Biến thiên (n)** | Chỉ giữ lại các chunk khớp nhãn (`category`), năm (`year`), hoặc mã ngành (`major`). Loại bỏ hoàn toàn nhiễu từ các năm hoặc ngành khác. |
| **2. Song song Search** | Dense & Sparse Search | **Top 50 mỗi luồng** | Thực hiện song song: tìm kiếm vector (Dense Search) chọn ra **Top 50** và tìm kiếm từ khóa BM25 (Sparse Search) chọn ra **Top 50** từ tập đã lọc cứng. |
| **3. RRF Fusion** | Reciprocal Rank Fusion | **Tối đa Top 60** | Ghép và chuẩn hóa độ ưu tiên từ 2 luồng kết quả trên bằng thuật toán RRF (có trọng số), loại bỏ trùng lặp và lấy ra **Top 60** chunk ứng viên. |
| **4. Reranker** | Cross-Encoder Reranking | **Top 15** | Sử dụng mô hình Cross-Encoder chuyên dụng để chấm điểm tương quan sâu giữa câu hỏi và Top 60 ứng viên, chỉ giữ lại **Top 15** chunk chất lượng nhất. |
| **5. Generator** | Sinh câu trả lời | **Top 10** | Đưa Top 10 chunk này vào Prompt Context của LLM để tổng hợp câu trả lời chính xác, kèm theo trích dẫn nguồn (`chunk_id`). |

---

## 📂 Cấu trúc Project

```text
uit_rag_system/
├── chunk/                          # Dữ liệu chính dùng cho RAG (đã được chunk)
│   ├── tuyen_sinh/                 # Chunks tuyển sinh (.jsonl)
│   ├── chuong_trinh_dao_tao/       # Chunks chương trình đào tạo (.jsonl)
│   ├── danh_muc_mon_hoc/           # Chunks danh mục môn học (.jsonl)
│   ├── cong_tac_sinh_vien/         # Chunks công tác sinh viên (.jsonl)
│   └── nhom_nghien_cuu/            # Chunks nhóm nghiên cứu (.jsonl)
├── src/                            # Mã nguồn hệ thống RAG
│   ├── app.py                      # Giao diện Streamlit chi tiết
│   ├── config.py                   # Cấu hình đường dẫn, embedding model, Milvus
│   ├── llm_utils.py                # Client LLM (DeepSeek, OpenAI), Prompts, parse JSON
│   ├── pipeline.py                 # Điều phối toàn bộ luồng RAG (Step 1-8)
│   └── rag_utils.py                # Database chunk, Hard Filter, BM25, RRF, Reranker
├── testset/                        # Bộ dữ liệu đánh giá hệ thống (JSON)
│   ├── uit_rag_testset_50.json     # Bộ 50 câu hỏi trắc nghiệm
│   └── uit_rag_testset_100.json    # Bộ 100 câu hỏi trắc nghiệm
├── volumes/                        # Thư mục lưu trữ dữ liệu Docker (Milvus, MinIO, etcd)
├── app.py                          # Streamlit entry point chính
├── create_embeddings.py            # Script sinh vector embedding từ chunks
├── load_to_milvus.py               # Script nạp chunks & vectors vào Milvus DB
├── reload_milvus.bat               # Batch script tự động chạy Re-embed & Load Milvus (Windows)
├── run.bat                         # Batch script khởi động Streamlit nhanh (Windows)
├── docker-compose.yml              # Cấu hình Milvus Standalone Docker
├── requirements.txt                # Danh sách thư viện Python cần thiết
├── evaluate.py                     # Script đánh giá hiệu năng hệ thống trên bộ testset
├── calculate_metrics.py            # Script tính toán chỉ số F1, Precision, Recall, Accuracy
├── compare_ablation.py             # Script so sánh ablation study (Có/Không Hard Filter)
└── README.md                       # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 🛠️ Hướng dẫn Khởi chạy Project (Step-by-Step)

Dưới đây là hướng dẫn chi tiết cách cài đặt và vận hành hệ thống trên máy tính của bạn.

### Bước 1: Chuẩn bị Môi trường Python

1. Đảm bảo bạn đã cài đặt Python (phiên bản khuyến nghị: `>= 3.10`).
2. Mở Terminal (CMD / PowerShell trên Windows) tại thư mục dự án và tạo môi trường ảo:
   ```bash
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   * **Trên Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Trên Windows (CMD):**
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   * **Trên macOS / Linux:**
     ```bash
     source venv/bin/activate
     ```
4. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

### Bước 2: Cấu hình Biến môi trường (`.env`)

Tạo file `.env` tại thư mục gốc của dự án (hoặc chỉnh sửa file `.env` đã có) và nhập thông tin API Key của LLM Provider mà bạn sử dụng:

```env
# LLM Provider: deepseek, openai, anthropic
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash

# DeepSeek API (Hoặc sử dụng OpenRouter / OpenAI)
DEEPSEEK_API_URL=https://api.xah.io/v1/chat/completions
DEEPSEEK_API_KEY=sk-xxxx... # Thay bằng API Key thật của bạn

# OpenAI API (Phương án dự phòng)
OPENAI_API_KEY=your_openai_key_here
OPENAI_API_URL=https://api.openai.com/v1/chat/completions

# Anthropic API (Phương án dự phòng)
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Bước 3: Khởi chạy Database Vector (Milvus)

Hệ thống RAG sử dụng **Milvus Standalone** làm cơ sở dữ liệu vector. Bạn cần khởi động nó thông qua Docker:

1. Đảm bảo máy của bạn đã chạy **Docker Desktop** (hoặc Docker Daemon).
2. Chạy lệnh sau tại thư mục gốc để khởi động các container (Milvus, MinIO, etcd) ở chế độ chạy ngầm (`-d`):
   ```bash
   docker compose up -d
   ```
3. Kiểm tra xem các container đã khởi động thành công chưa:
   ```bash
   docker compose ps
   ```
   *Bạn sẽ thấy các dịch vụ `milvus-standalone`, `milvus-minio`, và `milvus-etcd` ở trạng thái `Up`.*

### Bước 4: Tạo Embeddings và Nạp Dữ liệu vào Milvus

Để hệ thống có dữ liệu quy chế tra cứu, bạn cần chạy bước nhúng vector (embedding) dữ liệu từ các file trong thư mục `chunk/` và nạp vào Milvus:

* **Cách 1: Sử dụng Script tự động (Khuyên dùng cho Windows)**
  Double-click vào file `reload_milvus.bat` hoặc chạy lệnh:
  ```cmd
  .\reload_milvus.bat
  ```
  *Script này sẽ thực thi tuần tự: sinh vector embeddings → xoá collection cũ trong Milvus (nếu có) → tạo collection mới → nạp toàn bộ chunks kèm vector vào Milvus.*

* **Cách 2: Chạy thủ công các lệnh Python**
  Nếu bạn dùng macOS/Linux hoặc muốn chạy từng phần:
  1. Sinh vector embeddings từ dữ liệu chunks thô:
     ```bash
     python create_embeddings.py
     ```
     *(Kết quả được lưu tại thư mục `embedded/embedded_chunks.jsonl`)*
  2. Nạp dữ liệu embeddings vừa sinh vào Milvus:
     ```bash
     python load_to_milvus.py
     ```

### Bước 5: Khởi động Ứng dụng Streamlit (Web UI)

Sau khi dữ liệu đã được nạp thành công vào Milvus, bạn có thể khởi chạy ứng dụng chatbot:

* **Trên Windows:** Chạy trực tiếp file batch:
  ```cmd
  .\run.bat
  ```
* **Chạy thủ công qua câu lệnh:**
  ```bash
  streamlit run app.py
  ```

Sau khi chạy thành công, giao diện Web UI sẽ tự động mở trên trình duyệt của bạn (thường tại địa chỉ `http://localhost:8501`). Tại đây bạn có thể chọn chế độ chat thông thường (**Normal**) hoặc làm bài thi trắc nghiệm quy chế (**MCQ**).

---

## 📊 Hướng dẫn Đánh giá & So sánh (Evaluation & Ablation Studies)

Hệ thống cung cấp sẵn các công cụ để tự động đánh giá chất lượng câu trả lời RAG trên bộ câu hỏi chuẩn (testset).

### 1. Chạy đánh giá hệ thống (Evaluation)
Để chạy thử nghiệm đánh giá câu trả lời (sử dụng phương pháp Mode Voting - chạy 3 lần lấy số đông):
```bash
# Đánh giá mặc định trên bộ testset 50 câu
python evaluate.py --testset testset/uit_rag_testset_50.json

# Hoặc chạy trên bộ 100 câu
python evaluate.py --testset testset/uit_rag_testset_100.json

# Chỉ chạy thử nghiệm 5 câu đầu tiên để test tốc độ
python evaluate.py --limit 5
```
*Kết quả báo cáo chi tiết sẽ được xuất ra một file JSON nằm trong thư mục `results/`.*

### 2. Tính toán các chỉ số chi tiết (Calculate Metrics)
Sau khi có file JSON kết quả đánh giá (ví dụ: `results/evaluation_report_20260630_014236.json`), bạn có thể tính các chỉ số F1, Precision, Recall, Accuracy cho cả câu hỏi Single-choice và Multiple-choice:
```bash
python calculate_metrics.py results/evaluation_report_xxxxxx_xxxxxx.json
```

### 3. Đánh giá kiểm chứng thành phần (Ablation Study)
Để so sánh trực tiếp hiệu quả của việc **Có Hard Filter** vs **Không có Hard Filter** (để chứng minh giá trị của chiến thuật Adaptive RAG):
1. Chạy đánh giá không có Hard Filter:
   ```bash
   python evaluate.py --testset testset/uit_rag_testset_50.json --disable-hard-filter --output results/ablation_report.json
   ```
2. Chạy so sánh chênh lệch hiệu năng giữa 2 báo cáo:
   ```bash
   python compare_ablation.py results/evaluation_report_baseline.json results/ablation_report.json --output results/compare_result.md
   ```

---

## 🏷️ 6 Nhãn Phẳng phân loại dữ liệu (Flat Labels)

| Nhãn phẳng | Mô tả | Nguồn thư mục tương ứng |
|:---|:---|:---|
| `tuyensinh` | Thông tin tuyển sinh, học phí, phương thức xét tuyển đầu vào. | `chunk/tuyen_sinh/` |
| `chuongtrinhdaotao` | Khung chương trình đào tạo, lộ trình học, môn định hướng của các ngành. | `chunk/chuong_trinh_dao_tao/` |
| `danhmucmonhoc` | Chi tiết mã môn học, số tín chỉ lý thuyết/thực hành, môn tiên quyết/môn học trước. | `chunk/danh_muc_mon_hoc/` |
| `hoatdong` | Phong trào Đoàn - Hội, chiến dịch tình nguyện, câu lạc bộ, cẩm nang sinh viên. | `chunk/cong_tac_sinh_vien/` |
| `nghiencuu` | Nhóm nghiên cứu khoa học, thông tin giảng viên hướng dẫn, bài báo khoa học. | `chunk/nhom_nghien_cuu/` |
| `chung` | Chào hỏi thông thường, quy chế chung áp dụng cho mọi khóa/ngành, hoặc không xác định. | (Tất cả nguồn hoặc fallback) |
