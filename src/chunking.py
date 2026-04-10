from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        # Chia theo dấu câu ranh giới: ". ", "! ", "? " hoặc ".\n"
        pattern = r'(\. |\! |\? |\.\n)'
        parts = re.split(pattern, text)
        
        sentences = []
        current_sentence = ""
        
        for part in parts:
            current_sentence += part
            # Nếu part là dấu phân cách, câu hiện tại đã hoàn chỉnh
            if re.match(pattern, part):
                sentences.append(current_sentence.strip())
                current_sentence = ""
                
        # Dọn dẹp khoảng trắng dư thừa và thêm phần cuối cùng nếu có
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_text = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            chunks.append(chunk_text)
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Gọi hàm đệ quy với toàn bộ đoạn text và danh sách separators ban đầu
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Xong: Nếu đoạn text đã ngắn hơn chunk_size thì trả về luôn
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Xong/Đường cùng: Đã thử hết tất cả separator mà vẫn dài -> cắt cứng theo độ dài
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        # Lấy separator để thử lần này, và lưu lại danh sách cho các lần sau
        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Cắt bằng khoảng trắng vô hình (`""`) sẽ lỗi trong Python nên ta xử lý bằng cách cắt cứng luôn
        if separator == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        # THỬ NGHIỆM: Cắt text bằng separator hiện đại
        splits = current_text.split(separator)

        # Nếu đoạn text không chứa separator này, thử separator tiếp theo
        if len(splits) == 1:
            return self._split(current_text, next_separators)

        # GOM LẠI: Từng phần rất nhỏ sau khi cắt sẽ được nối lại với nhau
        # Miễn sao không được vượt quá `chunk_size`
        good_chunks = []
        current_chunk = ""

        for part in splits:
            # Nguy hiểm: Bản thân 1 phần sau khi bị cắt VẪN dài hơn chunk_size
            if len(part) > self.chunk_size:
                # Đẩy tất cả những gì gom được trước đó vào kết quả
                if current_chunk:
                    good_chunks.append(current_chunk)
                    current_chunk = ""
                    
                # Đệ quy: mang phần cứng đầu này đi cắt tiếp bằng separator cấp dưới
                good_chunks.extend(self._split(part, next_separators))
                continue

            # Tính toán xem nếu ghép nối lại thì độ dài mới là bao nhiêu
            # (Chỉ cộng thêm chiều dài dấu separator nếu biến current_chunk không bị rỗng)
            new_len = len(current_chunk) + len(separator) + len(part) if current_chunk else len(part)

            # Nếu ghép vào bị quá tải -> Lưu cái cũ, gán part thành điểm khởi đầu cho chuỗi mới
            if new_len > self.chunk_size:
                good_chunks.append(current_chunk)
                current_chunk = part
            else:
                # Dư sức -> Cứ nối part vào
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part

        # Lưu lại miếng cuối cùng sau vòng lặp
        if current_chunk:
            good_chunks.append(current_chunk)

        return good_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    dot_product = _dot(vec_a, vec_b)
    norm_a = sum(x * x for x in vec_a) ** 0.5
    norm_b = sum(x * x for x in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        if not text:
            return {}

        # Khởi tạo 3 chiến lược chunking
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=20)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        # Chạy từng chiến lược
        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        # Hàm phụ trợ tính trung bình độ dài
        def _get_avg_len(chunks: list[str]) -> float:
            if not chunks:
                return 0.0
            return sum(len(c) for c in chunks) / len(chunks)

        # Trả về kết quả dưới dạng dictionary
        return {
            "fixed_size": {
                "count": len(fixed_chunks),
                "avg_length": _get_avg_len(fixed_chunks),
                "chunks": fixed_chunks,
            },
            "by_sentences": {
                "count": len(sentence_chunks),
                "avg_length": _get_avg_len(sentence_chunks),
                "chunks": sentence_chunks,
            },
            "recursive": {
                "count": len(recursive_chunks),
                "avg_length": _get_avg_len(recursive_chunks),
                "chunks": recursive_chunks,
            },
        }
