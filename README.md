# AI-Powered Software Testing Automation App (AI Test Agent)

## 1) Abstract

Project này xây dựng một nguyên mẫu nền tảng cho bài toán tự động hoá kiểm tra phần mềm bằng AI, tập trung vào hai khối chức năng chính: tổ chức tri thức từ tài liệu dự án bằng truy hồi tăng cường sinh (RAG) và thu thập nội dung web có kiểm soát phục vụ truy vấn/đối chiếu. Cụ thể, hệ thống thiết lập quy trình nạp tài liệu PDF, phân đoạn nội dung, tạo embedding và lưu trữ trong VectorDB (Chroma) để hỗ trợ truy vấn theo ngữ cảnh bằng mô hình ngôn ngữ chạy cục bộ (Ollama). Song song, một khung search/scraper dựa trên Playwright được sử dụng để truy cập trang, lấy HTML và trích xuất văn bản, tạo nguồn dữ liệu bổ sung cho kho tri thức. Kết quả của đề tài là một pipeline có thể tái sử dụng cho việc xây dựng trợ lý kiểm thử dựa trên tri thức dự án, đồng thời làm nền cho các bước mở rộng tiếp theo như sinh test case và điều khiển thao tác UI.

---

# Mục tiêu

## 1) Khối RAG + VectorDB

- Nạp tài liệu **PDF** (ví dụ: `docs/ds-tool.pdf`) bằng loader.
- Tiền xử lý và **chia nhỏ (chunking)** nội dung tài liệu.
- Tạo **embedding** bằng mô hình chạy cục bộ qua Ollama (ví dụ: `embeddinggemma:latest`).
- Lưu trữ và quản lý **VectorDB Chroma** (persist vào thư mục `vector_db/`).
- Truy vấn VectorDB để **truy xuất ngữ cảnh liên quan** theo câu hỏi (retrieval).
- Dùng LLM chạy cục bộ qua Ollama (ví dụ: `qwen3:0.6b`) để tạo câu trả lời dựa trên context (RAG QA cơ bản).

## 2) Khối thu thập nội dung web

- Tìm URL theo từ khoá bằng script tìm kiếm (ví dụ `search.py`).
- Dùng **Playwright** để:
  - Truy cập trang web (`goto`).
  - Lấy **HTML sau render** (`page.content()`).
- Dùng **lxml** để:
  - Trích xuất nội dung text từ HTML theo selector.
  - Chuẩn hoá văn bản đầu ra phục vụ lưu trữ/đối chiếu.

## 3) Khối tích hợp pipeline

- Tạo pipeline tối thiểu:
  - (PDF → VectorDB) + (Web HTML → text)
  - Cho phép dùng chung một “luồng” nhập dữ liệu và truy vấn.
- Viết entrypoint chạy theo kịch bản (CLI đơn giản):
  - Chọn nguồn: PDF hoặc URL.
  - Thực hiện: lấy dữ liệu → chuẩn hoá → truy vấn RAG → in kết quả.
- Chuẩn hoá cấu trúc output tối thiểu:
  - Lưu văn bản đã trích xuất (từ PDF/HTML) để kiểm tra lại.
  - Lưu kết quả truy vấn và context đã retrieve.
