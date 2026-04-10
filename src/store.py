from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Tạo bản ghi lưu trữ tiêu chuẩn từ doc và tính toán embedding
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata.copy() if doc.metadata else {},
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # Tìm kiếm trong bộ nhớ (In-memory search)
        query_emb = self._embedding_fn(query)
        results = []
        for record in records:
            score = _dot(query_emb, record["embedding"])
            # Ta tạo record mới để nhét thêm field 'score' vào cho kết quả
            res_record = record.copy()
            res_record["score"] = score
            results.append(res_record)
            
        # Sắp xếp giảm dần theo điểm số similarity
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)
            
            # (Phần rỗng để dành nếu sau này bạn khởi tạo ChromaDB)
            if self._use_chroma and self._collection is not None:
                pass

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # Chuyển xuống hàm _search_records xử lý chung
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        filtered_records = []
        if metadata_filter:
            for record in self._store:
                # Kiểm tra xem record này có chứa đầy đủ điều kiện filter không
                match = True
                for k, v in metadata_filter.items():
                    if record["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
        else:
            filtered_records = self._store
            
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        initial_len = len(self._store)
        
        # Xoá document bằng cách chỉ giữ lại những record KHÔNG thoả mãn doc_id
        self._store = [
            record for record in self._store
            # Ở bài Lab này, document ID có thể nằm trực tiếp ở record['id'] hoặc trong metadata['doc_id']
            if record["id"] != doc_id and record["metadata"].get("doc_id") != doc_id
        ]
        
        return len(self._store) < initial_len
