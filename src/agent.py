from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Bước 1: Lấy các đoạn tài liệu có liên quan nhất từ Vector Store
        results = self.store.search(query=question, top_k=top_k)

        # Bước 2: Trích xuất nội dung văn bản (content) và ghép lại làm ngữ cảnh (context)
        context_chunks = [result["content"] for result in results]
        context = "\n---\n".join(context_chunks)

        # Bước 3: Xây dựng câu Prompt nạp vào cho LLM
        prompt = (
            "Dưới đây là các đoạn pháp luật liên quan được trích xuất từ "
            "Dự thảo Luật Trí tuệ nhân tạo Việt Nam:\n\n"
            f"[CONTEXT]\n{context}\n\n"
            f"[QUESTION]\n{question}\n\n"
            "Hãy trả lời câu hỏi dựa trên context trên. "
            "Trích dẫn Điều, Khoản cụ thể nếu có. "
            "Nếu context không đủ thông tin, hãy nói thẳng và gợi ý người dùng hỏi cụ thể hơn."
        )

        # Bước 4: Gọi LLM tạo sinh câu trả lời
        return self.llm_fn(prompt)
