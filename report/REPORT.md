# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thành Đạt
**Nhóm:** D2-C401
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nghĩa là hai vector biểu diễn văn bản có góc hợp bởi rất nhỏ (gần 0 độ), thể hiện ngữ nghĩa cốt lõi của hai đoạn văn bản rất giống nhau dù từ ngữ có thể viết khác đi.

**Ví dụ HIGH similarity:**
- Sentence A: "Hôm nay trời đẹp và có nắng."
- Sentence B: "Thời tiết hôm nay rất tuyệt vời, mây quang và tạnh ráo."
- Tại sao tương đồng: Cả hai câu đều miêu tả cùng một trạng thái thời tiết tốt trong ngày hôm nay.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi rất thích ăn mì Ý."
- Sentence B: "Ngôn ngữ lập trình Python thật thú vị."
- Tại sao khác: Hai câu bàn về hai chủ đề hoàn toàn không liên đới (ẩm thực và công nghệ) nên vector sẽ gần như vuông góc nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine đánh giá lường *hướng* thay vì *độ lớn* của vector. Với văn bản, độ dài của văn bản (tạo vector dài hơn) không quyết định nhiều bằng hướng ngữ nghĩa tổng thể. Vì vậy dùng cosine sẽ tránh được nhiễu loạn liên quan tới số lượng thống kê từ.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Chunk đầu chứa 500 ký tự. Các chunk tiếp theo sẽ tịnh tiến một khoảng (500 - 50) = 450 ký tự. Tổng số = 1 + ceil((10000 - 500) / 450) = 1 + 22 = 23 chunks.
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap tăng lên 100, thì tiến tịnh chỉ còn 400 ký tự mỗi step, khiến số chunk tăng lên (~25 chunks). Ta thường cần overlap nhiều hơn bởi các đường cắt cứng dễ chẻ đôi câu, làm mất ngữ nghĩa. Chèn lấp sẽ khôi phục lại trọn vẹn ý tiếp nối.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Pháp luật Việt Nam (Dự thảo Luật Trí tuệ nhân tạo 2025).

**Tại sao nhóm chọn domain này?**
> Đây là một chủ đề cực hot và có cấu trúc văn bản rất chặt chẽ (Chương, Điều, Khoản). Việc thử nghiệm RAG trên văn bản luật giúp đánh giá chính xác khả năng truy xuất thông tin cụ thể và tính logic của Agent.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Luật AI - Chương 1: Quy định chung | du-thao-Luat-AI.md | ~340 | `chuong: Chương I`, `dieu: Điều 1–6`, `category: law` |
| 2 | Luật AI - Chương 2: Phân loại rủi ro | du-thao-Luat-AI.md | ~11,000 | `chuong: Chương II`, `dieu: Điều 8–14`, `category: law` |
| 3 | Luật AI - Chương 3: Hạ tầng &amp; Chủ quyền | du-thao-Luat-AI.md | ~7,500 | `chuong: Chương III`, `dieu: Điều 15–17`, `category: law` |
| 4 | Luật AI - Chương 4: Hệ sinh thái | du-thao-Luat-AI.md | ~8,000 | `chuong: Chương IV`, `dieu: Điều 18–24`, `category: law` |
| 5 | Luật AI - Chương 5: Đạo đức &amp; Trách nhiệm | du-thao-Luat-AI.md | ~3,500 | `chuong: Chương V`, `dieu: Điều 25–26`, `category: law` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `chuong` | string | `Chương II` | Filter theo Chương — thu hẹp phạm vi tìm kiếm, tránh nhiễu giữa các Chương. |
| `chuong_title` | string | `Phân loại và quản lý hệ thống AI theo rủi ro` | Bổ sung ngữ cảnh vĩ mô ngay trong chunk, giúp LLM biết mình đang xử lý lĩnh vực nào. |
| `dieu` | string | `Điều 8` | Truy xuất chính xác theo số Điều — thường là trọng tâm khi user hỏi về quy định cụ thể. |
| `dieu_title` | string | `Mức độ rủi ro của hệ thống trí tuệ nhân tạo` | Hiển thị tên Điều giúp Agent có thể trích dẫn nguồn pháp lý chính xác trong câu trả lời. |
| `category` | string | `law` | Dùng để lọc domain (nếu hệ thống mở rộng sang nhiều loại tài liệu). |
| `language` | string | `vi` | Hỗ trợ multi-language filter nếu sau này hệ thống có thêm tài liệu tiếng Anh. |
| `source` | string | `data/luat_ai_cleaned.md` | Truy xuất nguồn gốc file gốc — hỗ trợ debug và kiểm tra tính toàn vẹn dữ liệu. |
| `chunk_index` | int | `0` | Biết chunk này là đoạn thứ mấy trong Điều — hỗ trợ sắp xếp lại context theo thứ tự khi cần. |
| `total_chunks` | int | `3` | Biết Điều này được chia thành bao nhiêu chunk — đánh giá được độ dài của Điều đó. |
| `doc_id` | string | `dieu_8` | ID duy nhất của Document gốc (trước khi chunk) — dùng để `delete_document` hoặc group lại. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu `luat_ai_cleaned.md`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Luật AI | RecursiveChunker (recursive) | 171 | 59.56 | Excellent (Rất hội tụ). |
| Luật AI | DocumentStructureChunker  (document_structure) | 71 | ~409 | Tốt, giữ được đơn vị điều/mục và nội dung pháp lý đầy đủ hơn. |
| Luật AI | Custom Strategy (Logic Document Parse nguyên cấu trúc + FixedSizeChunker) | 89 | ~682 | Bảo toàn ngữ cảnh tốt. |
| Luật AI | Sematic chunking (semantic_chunking) | 89 | ~682 | Bảo toàn ngữ cảnh tốt. |
| Luật AI | Document-Structure + Overlap (Hybrid) (document_structure_overlap) | 100 | ~190 | Tốt, giữ được đơn vị điều/mục và nội dung pháp lý đầy đủ hơn. |
| Luật AI | Fixed-Size (fixed_size) | 121 | ~496 | Trung bình (câu có thể bị cắt ngang nếu xui xẻo ráp nối không đều ở phần biên) |

### Strategy Của Tôi

**Loại:** Custom Strategy kết hợp (Logic Document Parse nguyên cấu trúc + FixedSizeChunker)

**Mô tả cách hoạt động:**
> File `law_chunking.py` đọc tài liệu đã xử lý sạch `luat_ai_cleaned.md`, dùng Regex để tự động phát hiện và tách riêng cấu trúc (**Chương**, **Điều**) để đóng gói mỗi Điều này thành 1 Document đơn mang đủ metadata. Lớp bao ngoài này sau đó lại chạy qua `FixedSizeChunker(chunk_size=800, overlap=150)` trích xuất nhỏ sâu trong nội dung. 

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Do Đặc thù Luật Trí tuệ Nhân tạo có cấu trúc ràng buộc chặt chẽ, nếu không tách từng "Điều" sẽ dễ xảy ra tình trạng câu hỏi bị lẫn thông tin từ các Điều với nhau. Việc bảo toàn metadata chuẩn của Điều kết hợp cùng một size cố định giúp LLM lúc Retrieval biết mình đang truy xuất chính xác nội dung từ điều/khoản nào.

**Code snippet (nếu custom):**
```python
# Parse Document bằng cách bắt regex nguyên thủy của MD file
chuong_match = re.match(r'^\*\*Chương\s+([IVXLC]+)\*\*$', stripped)
dieu_match = re.match(r'^\*\*Điều\s+(\d+\+?)\.\s*(.+?)\*\*$', stripped)
# Lồng thêm FixedSizeChunker để chunk mịn hơn từng dòng của Doc
chunker = FixedSizeChunker(chunk_size=800, overlap=150)
chunks = chunker.chunk(doc.content)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|-------------------|
| `luat_ai_cleaned.md` | Best baseline: FixedSizeChunker | 101 | ~688 chars | Trung bình — 5/5 relevant top-3, nhưng không có metadata Điều/Chương nên Agent khó trích dẫn nguồn chính xác. |
| `luat_ai_cleaned.md` | **Custom (Regex Parse + FixedSize)** | **89** | **~682 chars** | **Xuất sắc — 5/5 relevant top-3, score trung bình 0.57–0.74, Agent trích dẫn chính xác tên Điều do metadata đầy đủ.** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Chunk Count | Avg Length | Retrieval Quality | Điểm mạnh | Điểm yếu |
|-----------|----------|-------------|------------|-------------------|-----------|----------|
| **Nguyễn Thành Đạt** (tôi) | Custom: Regex Parse + FixedSizeChunker (800, overlap=150) | 89 | ~682 chars | Xuất sắc — 5/5 top-3 relevant, score 0.57–0.74 | Metadata cực chuẩn (Điều/Chương), Agent trích dẫn chính xác nguồn pháp lý | Pipeline phức tạp hơn, phụ thuộc format file đầu vào |
| **Hoàng Ngọc Anh** | DocumentStructureChunker | 71 | ~409 chars | Tốt nhất về độ gọn — giữ được đơn vị Điều/Mục, nội dung pháp lý đầy đủ | Chunk nhỏ gọt và sạch theo cấu trúc tài liệu | Avg Length thấp, có thể thiếu context nếu nội dung Điều dài |
| **Nguyễn Anh Đức** | Document-Structure Hybrid | 87 | Năng động | Xuất sắc — Giữ đúng tên Điều làm metadata/header ở mọi chunk | Kết hợp linh hoạt giữa cấu trúc và độ dài chunk | Avg Length không cố định, khó dự đoán kích thước retrieval |
| **Nguyễn Hoàng Việt** | DocumentStructureChunker (Chương I–III) | 71 | ~409 chars | Tốt — bảo toàn đơn vị Điều/Mục | Tách sạch theo Chương, context vẫn đầy đủ | Chỉ cover một phần tài liệu (Chương I–III) |
| **Đậu Văn Quyền** | RecursiveChunker | 171 | ~59.56 chars | Excellent (Rất hội tụ) | Chunk rất nhỏ, độ chính xác cosine cao | Chunk quá nhỏ có thể thiếu ngữ cảnh câu |
| **Vũ Duy Linh** | FixedSizeChunker (`fixed_size`) | 121 | ~496.83 chars | Trung bình | Đơn giản, dễ triển khai, không cần parse cấu trúc | Câu có thể bị cắt ngang tại biên chunk nếu không may |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với domain Luật Trí tuệ Nhân tạo, chiến thuật **Document-Structure** (tách theo Chương/Điều) — được áp dụng bởi Hoàng Ngọc Anh, Nguyễn Anh Đức, Nguyễn Hoàng Việt và Nguyễn Thành Đạt — cho thấy ưu thế rõ ràng so với baseline, vì văn bản luật có đơn vị ngữ nghĩa tự nhiên là từng "Điều". Strategy thuần RecursiveChunker của Quyền đạt Chunk Count cao nhưng avg length quá nhỏ (~59 chars), dễ bị mất ngữ cảnh. Xét tổng thể, cách tốt nhất là kết hợp: **Parse cấu trúc Điều làm metadata** + **FixedSizeChunker với overlap trên nội dung** để vừa giữ context vừa đảm bảo kích thước chunk đồng nhất cho retrieval chính xác.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex kết hợp tìm kiếm dấu nhắc hết câu `(?<=[.!?])\s+` làm điểm phân tách. Có xử lý edge case cẩn thận để giữ nguyên dấu ngắt câu cuối, rồi loop qua từng sentence ghép mạch liên hồi miễn là đoạn đang nối ngắn hơn `chunk_size` quy định.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Dựa trên thuật toán đệ quy bóc tách text theo dạng giảm dần độ ưu tiên chia cách `["\n\n", "\n", " ", ""]`. Base case đệ quy rơi vào lúc string của fragment `<= chunk_size` hoặc đã duyệt hết mảng separators -> dừng lại.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Store lưu list of dict trong bộ nhớ (In-memory `_store`), vừa lưu doc vừa encode. Khâu search lấy vector query đã encode, tính Cosine Similarity chay với các vector document lưu từ trước, sort điểm lớn nhất và slice top K.

**`search_with_filter` + `delete_document`** — approach:
> Xuyên suốt pipeline ta thực hiện Filter `metadata` *trước* khi chạy Similarity hòng thu hẹp tập tính toán vector nhằm tối ưu tốc độ. `delete_document` khá basic qua cách thao tác list comprehension bỏ bớt trùng khớp theo document id truyền vô.

### KnowledgeBaseAgent

**`answer`** — approach:
> Tổ chức thiết kế logic Prompt mở cho AI (Dynamic): Bot có năng lực điều hướng Intent. Có input Context vào bot chèn context đó làm nền. Nếu mảng rỗng như lúc End User chào đón giao lưu, hệ thống hạ thành chat tự nhiên không khiên cưỡng.

### Test Results

```
tests/test_solution.py::TestFixedSizeChunker::test_chunk_with_overlap PASSED [  2%]
...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.07s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Hệ thống AI rủi ro cao gây thiệt hại | Phải bồi thường tổn thất do AI gây ra | high | 0.81 | Đúng |
| 2 | Phạm vi điều chỉnh của Luật AI | Thời tiết hôm nay mát mẻ | low | 0.12 | Đúng |
| 3 | Quyền của người dùng trí tuệ nhân tạo | Không được phân biệt đối xử khi dùng AI | high | 0.76 | Đúng |
| 4 | Trí tuệ nhân tạo tạo sinh | Thuật toán AI thế hệ mới | high | 0.85 | Đúng |
| 5 | Các cơ quan nhà nước kiểm soát AI | Công thức nấu ăn ngon mỗi ngày | low | -0.05 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp 4 bất ngờ vì dù không có từ ngữ (keyword) nào trùng khớp hoàn toàn ("trí tuệ nhân tạo tạo sinh" vs "thuật toán AI thế hệ mới") nhưng score rất cao (0.85). Nó cho thấy Embedding model không hoạt động theo kiểu so khớp từ, mà thực sự nhúng *khái niệm* và *ngữ cảnh* vào không gian vector đa chiều, giúp nhận diện sự đồng nghĩa sâu xa.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Phạm vi điều chỉnh của Luật AI là gì? | Quy định về nghiên cứu, phát triển, cung cấp, triển khai sử dụng AI tại Việt Nam. Không áp dụng cho quốc phòng an ninh và cơ yếu. |
| 2 | Ai chịu trách nhiệm bồi thường khi AI rủi ro cao gây thiệt hại? | Bên triển khai phải chịu trách nhiệm bồi thường trước cho người bị thiệt hại. Sau đó có thể yêu cầu nhà phát triển hoàn trả. |
| 3 | AI tạo sinh có phải dán nhãn không? | Phải gắn nhãn dễ nhận biết hoặc đánh dấu ở dạng máy đọc đối với âm thanh, hình ảnh, video do AI tạo ra. |
| 4 | Các hành vi bị cấm trong lĩnh vực AI? | Lợi dụng AI để vi phạm pháp luật, sử dụng yếu tố giả mạo thao túng con người, tạo nội dung nguy hại đến an ninh quốc gia. |
| 5 | Các mức độ rủi ro của AI? | Gồm 3 mức: Rủi ro cao, rủi ro trung bình và rủi ro thấp. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Phạm vi điều chỉnh của Luật AI là gì? | Điều 1: Phạm vi điều chỉnh... | 0.62 | Yes | Trả lời chính xác về các đối tượng quản lý và nhóm miễn trừ. |
| 2 | Ai chịu trách nhiệm bồi thường khi AI rủi ro cao gây thiệt hại? | Điều 28: Xử lý vi phạm và bồi thường... | 0.57 | Yes | Chính xác: Bên triển khai bồi thường trước cho cá nhân bị thiệt hại. |
| 3 | AI tạo sinh có phải dán nhãn không? | Điều 33: Quy trình xử lý... | 0.39 | Yes | Bắt buộc phải gắn nhãn/đánh dấu ở dạng máy đọc cho đầu ra do máy tạo. |
| 4 | Các hành vi bị cấm trong lĩnh vực AI? | Điều 6: Các hành vi bị cấm... | 0.50 | Yes | Nêu đúng các lệnh cấm như thao túng tâm lý, tạo fake video. |
| 5 | Các mức độ rủi ro của AI? | Điều 8: Mức độ rủi ro của hệ thống... | 0.74 | Yes | Đề cập đúng 3 bậc phân loại: Rủi ro Cao, TB, và Thấp. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Mình học được tư duy xử lý edge-cases khi biến chuỗi bị rỗng/null, cũng như cách các bạn chia nhỏ hàm tính Similarity để code clean hơn thay vì gom hết vào một khối to.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Có nhóm sử dụng tích hợp thêm Plotly/Matplotlib để visualize các text vector thực tế trên một không gian 2D. Điều này giúp trực quan hóa được khoảng cách (độ tương đồng ngữ nghĩa) giữa các chunks quá tốt.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Mình xem xét kết hợp **HyDE (Hypothetical Document Embeddings)**. Thay vì dùng trực tiếp câu hỏi ngắn gọn của user để query đống văn bản luật vốn dĩ dài và phức tạp, mình sẽ bắt LLM viết ra hẳn một văn bản "giả định" từ câu hỏi đó, dùng văn bản đó đo distance với vector store hứa hẹn kết quả sẽ mịn hơn rất nhiều.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **88 / 100** |
