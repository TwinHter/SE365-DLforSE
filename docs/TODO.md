# TODO - UIT RAG System

## 5 Chiến thuật Cốt lõi

### Chiến thuật 3: Adaptive RAG (LLM Router + Hard Filter)
- [x] Viết prompt cho Router trích xuất entities (label, year, specific entities)
- [x] Chuyển đổi JSON filters thành Milvus expr
- [ ] Test với câu hỏi theo năm khác nhau (2024, 2025, 2026)
- [ ] Test với câu hỏi theo ngành khác nhau

### Chiến thuật 4: Hybrid Search & Cross-Encoder Reranking
- [x] Implement Dense Search với Milvus (vector similarity)
- [x] Implement Sparse Search với BM25
- [x] Implement RRF (Reciprocal Rank Fusion) trộn kết quả
- [x] Tích hợp Cross-Encoder (BAAI/bge-reranker-large)
- [x] Lọc Top 20 → Top 5 chunks
- [ ] Tune trọng số giữa dense và sparse (alpha)

### Chiến thuật 5: Self-RAG Critique
- [ ] Viết prompt Faithfulness Checker
- [ ] Viết prompt Answer Relevance Checker
- [ ] Implement vòng lặp: nếu fail → retry hoặc từ chối
- [ ] Đánh giá Faithfulness Rate trên testset

---

## Thiết lập Nền tảng

- [x] Cài đặt Milvus (pymilvus[model] - Milvus Lite)
- [x] Định nghĩa Schema Milvus (chunk_id, vector, sparse_vector, category, year, parent_document)
- [x] Cấu hình BM25 cho Sparse Search
- [x] Gom và chuẩn bị dữ liệu chunk theo 6 nhãn
- [x] Insert dữ liệu vào Milvus

---

## Retrieval Pipeline

- [x] Hybrid Search (Dense + Sparse với RRF)
- [x] Áp dụng Hard Filter theo metadata từ Router
- [x] Cross-Encoder Reranker (BAAI/bge-reranker-large)
- [x] Lọc Top 20 → Top 5 chunks

---

## Generator

- [] Viết prompt cho LLM sinh câu trả lời
- [] Ép trích nguồn [chunk_id] ở cuối câu
- [] Xử lý trường hợp "Không tìm thấy"
- [] Tích hợp Generator vào pipeline chính (app.py)

---

## Evaluation

- [ ] Tạo 50 câu hỏi test với gold_chunk_id
  - [ ] Câu hỏi tra cứu đơn giản
  - [ ] Câu hỏi so sánh theo năm
  - [ ] Câu hỏi dạng bảng
  - [ ] Câu hỏi bẫy/gây nhiễu
- [ ] Chạy Hit Rate / Recall@5
- [ ] Tính Faithfulness Rate
- [ ] Tính Citation Accuracy
- [ ] Tính Answer Correctness
- [ ] Xuất kết quả ra file JSON

---

## UI (Streamlit Demo)

- [ ] Giao diện chat cơ bản
- [ ] Hiển thị metadata filter (label, year, entities)
- [ ] Hiển thị các bước trong pipeline
- [ ] Hiển thị chunks được lấy ra và điểm Rerank
- [ ] Hiển thị citation [chunk_id] trong câu trả lời
- [ ] Hiển thị Self-RAG critique feedback

---

## 6 Nhãn cần xử lý

- [x] `tuyensinh` - Điểm chuẩn, học phí, phương thức xét tuyển
- [x] `chuongtrinhdaotao` - Khung chương trình, lộ trình học
- [x] `danhmucmonhoc` - Mã môn, tín chỉ, môn tiên quyết
- [x] `hoatdong` - Phong trào, tình nguyện, CLB, sự kiện
- [x] `nghiencuu` - NCKH sinh viên, cuộc thi học thuật
- [x] `chung` - Chào hỏi, quy chế chung, null/không xác định

---

## Priority Order

1. **P0 - Core Pipeline:** Router → Retrieval → Reranker → Generator
2. **P1 - Quality:** Table-to-Text + Hierarchical Chunking
3. **P2 - Safety:** Self-RAG Critique
4. **P3 - Evaluation:** Testset + Metrics
5. **P4 - UX:** Streamlit Demo
