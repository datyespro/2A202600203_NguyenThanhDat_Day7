# Tổng Hợp Công Việc Hoàn Thành Lab 7

Trong quá trình thực hiện Lab 7, chúng ta đã hoàn thành xuất sắc các mục tiêu đề ra, bao gồm cả phần bắt buộc (My Approach) và các phần nâng cao (Chatbot, Chunking, Clean Data). Dưới đây là bảng tổng hợp khối lượng công việc đã thực hiện để bạn có cái nhìn tổng quan.

## 1. Hoàn thành Source Code (`src/`)

Toàn bộ các TODO trong `src` framework đã được code hoàn chỉnh và vượt qua **100% test cases (42/42 PASSED)**.

| Component | Thực hiện | Chi tiết |
| :--- | :--- | :--- |
| `chunking.py` | ✅ Hoàn thành | - Đã implement `SentenceChunker` tách theo các bộ dấu câu câu phổ biến `. `, `! `, `? `.<br>- Đã implement `RecursiveChunker` xử lý fallback mượt mà theo độ ưu tiên separator.<br>- Đã implement `compute_similarity` chuẩn công thức Cosine Similarity. |
| `store.py` | ✅ Hoàn thành | - Thiết kế `EmbeddingStore` in-memory dùng list of dicts bảo vệ bằng logic update/xoá chính xác (`add_documents`, `search`, `delete_document`).<br>- Thêm tính năng Lọc Metadata trước khi tìm kiếm (`search_with_filter`). |
| `agent.py` | ✅ Hoàn thành | - Hoàn thiện RAG flow: Gom chunk từ store -> Dựng Prompt kèm CONTEXT -> Gọi LLM.<br>- Prompt được thiết kế linh hoạt, xử lý case không có context chứ không hardcode cứng nhắc. |
| `embeddings.py`| ✅ Verify | Xác nhận hỗ trợ `OpenAIEmbedder` qua thư viện `openai` và model `text-embedding-3-small`. |

## 2. Tiền xử lý Dữ liệu (Mở rộng)

Nhận thấy cấu trúc file dữ liệu thô (extract từ .docx sang .md) bị lộn xộn, đứt dòng và lẫn tạp âm, chúng ta đã tự phát triển một pipeline làm sạch dữ liệu:

*   **Script `clean.py`**:
    *   Sử dụng regex để lọc rác: Xóa bỏ các metadata dư thừa (`<img>`, tiêu chuẩn TCVN, Quốc hiệu, Tiêu đề luật lặp lại).
    *   Tự ghép các dòng vô ý bị cắt đứt giữa câu, nối dòng liên tục dựa trên liên từ hoặc dấu câu kết thúc.
    *   Giữ lại nguyên vẹn cấu trúc đánh dấu quan trọng (ví dụ: `**Chương ...**`, `**Điều ...**`).
*   **Kết quả (`luat_ai_cleaned.md`)**: File MD đầu ra sạch sẽ (hơn 500 dòng), bố cục theo đoạn chuẩn chỉnh phù hợp để đọc bằng phần mềm và phân tách cho RAG.

## 3. Kiến trúc Xử lý & RAG Pipeline (Mở rộng)

Thay vì chỉ viết test chay, chúng ta đã tạo một hệ thống thật sự hoạt động mạnh mẽ chia làm 2 giai đoạn (Tách file để dễ maintain):

*   **`law_chunking.py` (Parser & Store Builder)**:
    *   Đảm nhận vai trò đọc file clean, tự động phát hiện `Chương` và `Điều`, đưa thành cấu trúc Metadata bài bản.
    *   Cấu hình `FixedSizeChunker(chunk_size=800, overlap=150)` là chiến thuật chunking cho file Luật AI.
    *   Kết nối với API Embeddings và lưu bộ vector store xuống file `vector_store.json`.
*   **`chatbot.py` (RAG Chat Application)**:
    *   Load vector store đã dump (nhanh và rẻ vì không gọi lại API Embed).
    *   *Intent Detection (Hệ thống điều hướng)*: Tính năng mới tự nhận diện tin nhắn là "casual greeting/chat" để LLM nói chuyện bình thường hay là "luật/thông tin" để kích hoạt vector search.
    *   Kéo Agent vào tương tác QA tự nhiên, tích hợp Fix lỗi encoding console Windows (UTF-8 Surrogate Problem) nên chạy tốt khi gõ tiếng Việt có dấu và Emoji terminal.

## 4. Kiểm thử Benchmark và Hoàn tất Report

Tại file `report/REPORT.md`:

1.  **Warm-Up & Chunking Math**: Điền đầy đủ lý thuyết so sánh Cosine Similarity so với Euclidean distance.
2.  **Domain Selection**: Nêu bật tại sao Luật Trí tuệ Nhân tạo VN lại tuyệt vời cho Chunking (Mỗi Điều là 1 atomic unit). Bảng thống kê kích cỡ document và Metadata Schema gồm `chuong`, `dieu_title`.
3.  **Chunking Strategy**: Giải thích chiến thuật Custom (Sliding Window kết hợp chia theo cấu trúc Luật).
4.  **My Approach**: Báo cáo chi tiết code đã viết trong gói `src/` và dán kết quả 42 Pass Tests.
5.  **Benchmark Queries**: Viết sẵn 5 câu hỏi khó, test RAG thật (ví dụ về Định nghĩa AI Rủi ro cao lấy chính xác từ Điều 13, Khoản...).

*(Lưu ý: Các phần thuộc về nhận xét các bạn khác trong "So sánh với thành viên khác" hay "Điều hay nhất học được" bạn sẽ tự bổ sung trong markdown).*
