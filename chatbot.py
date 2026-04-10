"""
Chatbot RAG Luật AI

Chatbot hỏi đáp về Dự thảo Luật Trí tuệ nhân tạo Việt Nam.
Load vector store đã build sẵn từ law_chunking.py, rồi chat.

Chuẩn bị:  python law_chunking.py   (chạy 1 lần để build store)
Chạy:      python chatbot.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

# Fix Windows terminal encoding (cp1252 can't render emoji/Vietnamese)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from src.embeddings import OpenAIEmbedder, _mock_embed
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

# ─── Config ───────────────────────────────────────────────────────────
STORE_FILE = Path("data/vector_store.json")
TOP_K = 3
LLM_MODEL = "gpt-4o-mini"


# ─── Load Vector Store từ file ────────────────────────────────────────

def load_vector_store(filepath: Path, embedding_fn) -> EmbeddingStore:
    """Load vector store đã build sẵn từ file JSON."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    store = EmbeddingStore(
        collection_name=data["collection_name"],
        embedding_fn=embedding_fn,
    )
    # Nạp trực tiếp records đã embed (không cần embed lại)
    store._store = data["records"]
    return store


# ─── Embedder (chỉ dùng cho query, không embed lại docs) ─────────────

def create_embedder():
    """Tạo embedder cho search query."""
    try:
        embedder = OpenAIEmbedder()
        embedder("test")
        print("  ✅ OpenAI Embedder (cho search query)")
        return embedder
    except Exception as e:
        print(f"  ⚠️  OpenAI Embedder lỗi: {e}")
        print("  📦 Fallback → Mock Embedder")
        return _mock_embed


# ─── Intent Detection ────────────────────────────────────────────────

# Các pattern nhận diện tin nhắn chào hỏi / hội thoại thông thường
_CASUAL_PATTERNS = re.compile(
    r'^(hi|hello|hey|xin chào|chào|alo|ok|okay|cảm ơn|thanks|thank you|'
    r'bạn là ai|bạn tên gì|giới thiệu|help|giúp|hướng dẫn|instructions?)\b',
    re.IGNORECASE | re.UNICODE,
)

_LAW_KEYWORDS = re.compile(
    r'(điều|khoản|luật|quy định|trí tuệ nhân tạo|ai|rủi ro|cấm|'
    r'trách nhiệm|nhà cung cấp|bên triển khai|hệ thống|chương|minh bạch)',
    re.IGNORECASE | re.UNICODE,
)

def is_casual(text: str) -> bool:
    """Trả về True nếu câu hỏi là hội thoại thông thường, không phải hỏi về luật."""
    text = text.strip()
    # Quá ngắn và không chứa từ khóa pháp luật → casual
    if len(text) <= 20 and not _LAW_KEYWORDS.search(text):
        return True
    # Khớp pattern chào hỏi
    if _CASUAL_PATTERNS.match(text):
        return True
    return False


# ─── LLM Function ────────────────────────────────────────────────────

SYSTEM_RAG = (
    "Bạn là trợ lý pháp luật chuyên về Luật Trí tuệ nhân tạo Việt Nam. "
    "Hãy trả lời câu hỏi dựa trên nội dung pháp luật được cung cấp trong [CONTEXT]. "
    "Trích dẫn Điều, Khoản cụ thể khi trả lời. "
    "Nếu context không đủ thông tin, hãy nói thẳng và gợi ý người dùng hỏi cụ thể hơn. "
    "Trả lời bằng tiếng Việt, ngắn gọn, chính xác."
)

SYSTEM_CASUAL = (
    "Bạn là trợ lý AI chuyên về Luật Trí tuệ nhân tạo Việt Nam. "
    "Hãy chào hỏi và trả lời tự nhiên, thân thiện bằng tiếng Việt. "
    "Giới thiệu mình là chatbot hỗ trợ tìm hiểu Dự thảo Luật AI Việt Nam khi cần."
)


def create_llm_fn(model: str = LLM_MODEL):
    """Trả về dict gồm 2 hàm: rag_fn (có context) và chat_fn (không context)."""
    try:
        from openai import OpenAI
        client = OpenAI()

        def _call_stream(system: str, user_content: str) -> str:
            """Gọi OpenAI với stream=True, in từng token ngay khi nhận được."""
            full = []
            with client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.3,
                max_tokens=1024,
                stream=True,
            ) as stream:
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        print(token, end="", flush=True)
                        full.append(token)
            print()  # xuống dòng sau khi stream xong
            return "".join(full)

        rag_fn  = lambda prompt:    _call_stream(SYSTEM_RAG,    prompt)
        chat_fn = lambda question:   _call_stream(SYSTEM_CASUAL, question)

        print(f"  ✅ {model} (streaming)")
        return rag_fn, chat_fn

    except Exception as e:
        print(f"  ⚠️  OpenAI Chat lỗi: {e}")
        print("  📦 Fallback → Demo LLM")

        demo = lambda text: f"[DEMO — cần OpenAI key] {text[:200]}..."
        return demo, demo


# ─── Interactive Chatbot ──────────────────────────────────────────────

def run_chatbot():
    """Load store → chat loop."""

    load_dotenv(override=False)

    print("=" * 60)
    print("🤖  CHATBOT LUẬT TRÍ TUỆ NHÂN TẠO VIỆT NAM")
    print("=" * 60)

    # 1. Load vector store
    print(f"\n📂 Load vector store từ {STORE_FILE}...")
    if not STORE_FILE.exists():
        print(f"❌ Chưa có vector store!")
        print(f"   Chạy 'python law_chunking.py' trước để build.")
        return 1

    print(f"\n🧮 Khởi tạo Embedder...")
    embedder = create_embedder()

    store = load_vector_store(STORE_FILE, embedding_fn=embedder)
    print(f"  📦 Đã load {store.get_collection_size()} chunks")

    # 2. LLM
    print(f"\n🧠 Khởi tạo LLM...")
    rag_fn, chat_fn = create_llm_fn()

    # 3. Agent (dùng rag_fn)
    agent = KnowledgeBaseAgent(store=store, llm_fn=rag_fn)

    # 4. Chat loop
    print("\n" + "=" * 60)
    print("💬 Chatbot sẵn sàng! Gõ câu hỏi về Luật AI.")
    print("   Lệnh: 'quit' = thoát | 'info' = thống kê")
    print("=" * 60)

    while True:
        try:
            question = input("\n📝 Câu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Tạm biệt!")
            break

        if not question:
            continue

        # Sanitize surrogate chars (Windows stdin)
        question = question.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
        question = question.replace("\ufffd", "").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("👋 Tạm biệt!")
            break

        if question.lower() == "info":
            print(f"  📦 Store: {store.get_collection_size()} chunks")
            print(f"  📁 Source: {STORE_FILE}")
            continue

        # --- Intent detect: casual → trả lời tự nhiên, không RAG ---
        if is_casual(question):
            print(f"\n💬 Trả lời: ", end="", flush=True)
            chat_fn(question)   # stream tự in ra, không cần print()
            continue

        # --- Câu hỏi về luật → RAG ---
        print(f"🔍 Đang tìm kiếm trong {store.get_collection_size()} chunks...")

        search_results = store.search(question, top_k=TOP_K)
        print(f"\n📎 Top {min(TOP_K, len(search_results))} chunks liên quan:")
        for i, r in enumerate(search_results[:3], 1):
            dieu = r['metadata'].get('dieu', '?')
            score = r.get('score', 0)
            preview = r['content'][:80].replace('\n', ' ')
            print(f"   {i}. [{dieu}] (score={score:.3f}) {preview}...")

        print(f"\n📖 Trả lời: ", end="", flush=True)
        agent.answer(question, top_k=TOP_K)   # stream tự in ra trong rag_fn



    return 0


if __name__ == "__main__":
    raise SystemExit(run_chatbot())
