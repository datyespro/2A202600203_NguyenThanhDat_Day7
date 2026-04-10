"""
Law Document Parser, Chunker & Vector Store Builder

Quy trình xử lý dữ liệu Luật AI:
  1. Parse file Luật AI đã clean → tách từng Điều thành Document với metadata
  2. Chunking theo sliding window + overlap (FixedSizeChunker)
  3. Embedding bằng OpenAI → lưu vector store ra file JSON

Chạy:  python law_chunking.py
Kết quả: data/vector_store.json (dùng cho chatbot.py)
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

# Fix Windows terminal encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from src.chunking import FixedSizeChunker
from src.embeddings import OpenAIEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

# ─── Config ───────────────────────────────────────────────────────────
DATA_FILE = Path("data/luat_ai_cleaned.md")
STORE_FILE = Path("data/vector_store.json")
CHUNK_SIZE = 800
OVERLAP = 150


# ─── Step 1: Parse dữ liệu Luật AI với metadata ──────────────────────

def parse_law_document(filepath: Path) -> list[Document]:
    """
    Đọc file Luật AI đã clean, parse ra từng Điều luật
    và gán metadata (chương, điều) cho mỗi đoạn.

    Nhận diện cấu trúc bằng regex:
      - **Chương I**, **Chương II**, ... → metadata "chuong"
      - **Điều 1. Tên điều** → metadata "dieu", "dieu_title"

    Returns:
        Danh sách Document, mỗi Document = 1 Điều luật hoàn chỉnh.
    """
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    documents: list[Document] = []
    current_chuong = ""
    current_chuong_title = ""
    current_dieu = ""
    current_dieu_title = ""
    current_content_lines: list[str] = []

    def flush():
        """Lưu Điều hiện tại thành Document nếu có nội dung."""
        if current_dieu and current_content_lines:
            content = "\n".join(current_content_lines).strip()
            if content:
                doc_id = re.sub(r'[^a-zA-Z0-9_]', '_', current_dieu.lower().replace(" ", "_"))
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    metadata={
                        "chuong": current_chuong,
                        "chuong_title": current_chuong_title,
                        "dieu": current_dieu,
                        "dieu_title": current_dieu_title,
                        "category": "law",
                        "language": "vi",
                        "source": str(filepath),
                    }
                ))

    for line in lines:
        stripped = line.strip()

        # Phát hiện Chương: **Chương I**, **Chương II**, ...
        chuong_match = re.match(r'^\*\*Chương\s+([IVXLC]+)\*\*$', stripped)
        if chuong_match:
            current_chuong = f"Chương {chuong_match.group(1)}"
            continue

        # Phát hiện tiêu đề Chương (dòng bold ngay sau **Chương X**)
        if current_chuong and not current_chuong_title and stripped.startswith("**") and stripped.endswith("**") and "Điều" not in stripped:
            current_chuong_title = stripped.strip("*").strip()
            continue

        # Phát hiện Điều: **Điều 1. Phạm vi điều chỉnh**
        dieu_match = re.match(r'^\*\*Điều\s+(\d+\+?)\.\s*(.+?)\*\*$', stripped)
        if dieu_match:
            flush()
            current_dieu = f"Điều {dieu_match.group(1)}"
            current_dieu_title = dieu_match.group(2).strip()
            current_content_lines = [f"{current_dieu}. {current_dieu_title}"]
            continue

        # Tích lũy nội dung vào Điều hiện tại
        if current_dieu and stripped:
            current_content_lines.append(stripped)

    flush()
    return documents


# ─── Step 2: Chunking với Sliding Window + Overlap ────────────────────

def chunk_documents(
    docs: list[Document],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Document]:
    """
    Chia mỗi Document (1 Điều) thành nhiều chunks nhỏ hơn
    bằng FixedSizeChunker (sliding window + overlap).
    """
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    chunked_docs: list[Document] = []

    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc.id}_chunk_{i}",
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_id": doc.id,
                }
            )
            chunked_docs.append(chunk_doc)

    return chunked_docs


# ─── Step 3: Embed & Save Vector Store ────────────────────────────────

def save_vector_store(store: EmbeddingStore, filepath: Path) -> None:
    """Lưu vector store (đã embed) ra file JSON."""
    data = {
        "collection_name": store._collection_name,
        "records": store._store,  # list[dict] gồm id, content, metadata, embedding
    }
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_embedder():
    """Tạo embedder: ưu tiên OpenAI, fallback về mock."""
    try:
        embedder = OpenAIEmbedder()
        embedder("test")
        print("  ✅ Sử dụng OpenAI Embedder (text-embedding-3-small)")
        return embedder
    except Exception as e:
        print(f"  ⚠️  OpenAI Embedder lỗi: {e}")
        print("  📦 Fallback → Mock Embedder")
        return _mock_embed


# ─── Main: build pipeline ────────────────────────────────────────────

def build_store():
    """Pipeline chính: parse → chunk → embed → save."""

    load_dotenv(override=False)

    print("=" * 60)
    print("📦 BUILD VECTOR STORE — Luật Trí tuệ nhân tạo")
    print("=" * 60)

    # 1. Parse
    print(f"\n📂 Đang parse {DATA_FILE}...")
    if not DATA_FILE.exists():
        print(f"❌ Không tìm thấy file: {DATA_FILE}")
        print("   Chạy 'python clean.py' trước để tạo file.")
        return 1

    law_docs = parse_law_document(DATA_FILE)
    print(f"  📄 {len(law_docs)} Điều luật")

    for doc in law_docs[:3]:
        print(f"     • {doc.metadata['dieu']}: {doc.metadata['dieu_title']} ({len(doc.content)} chars)")
    if len(law_docs) > 3:
        print(f"     • ... và {len(law_docs) - 3} Điều khác")

    # 2. Chunk
    print(f"\n✂️  Chunking (sliding window: size={CHUNK_SIZE}, overlap={OVERLAP})...")
    chunked_docs = chunk_documents(law_docs, CHUNK_SIZE, OVERLAP)
    avg_len = sum(len(d.content) for d in chunked_docs) / len(chunked_docs)
    print(f"  📦 {len(chunked_docs)} chunks (avg {avg_len:.0f} chars/chunk)")

    # 3. Embed
    print(f"\n🧮 Khởi tạo Embedding...")
    embedder = create_embedder()

    print(f"\n💾 Embedding {len(chunked_docs)} chunks...")
    store = EmbeddingStore(collection_name="luat_ai", embedding_fn=embedder)
    store.add_documents(chunked_docs)
    print(f"  ✅ Store size: {store.get_collection_size()} documents")

    # 4. Save
    print(f"\n💿 Lưu vector store → {STORE_FILE}...")
    save_vector_store(store, STORE_FILE)
    file_size = STORE_FILE.stat().st_size / 1024
    print(f"  ✅ Đã lưu ({file_size:.0f} KB)")

    print(f"\n🎉 Xong! Chạy 'python chatbot.py' để bắt đầu chat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(build_store())
