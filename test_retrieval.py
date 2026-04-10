"""Test Retrieval Quality for Chunking Strategy section in REPORT.md"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.store import EmbeddingStore
from src.embeddings import OpenAIEmbedder, _mock_embed
from src.models import Document

# ── Load vector store đã dump ──────────────────────────────
data = json.load(open("data/vector_store.json", encoding="utf-8"))

try:
    embedder = OpenAIEmbedder()
    embedder("test")
    print("  [Embedder] OpenAI")
except Exception as e:
    print(f"  [Embedder] Mock fallback ({e})")
    embedder = _mock_embed

store = EmbeddingStore(collection_name="luat_ai", embedding_fn=embedder)
store._store = data["records"]
print(f"  [Store] {len(store._store)} chunks loaded\n")

# ── 5 Benchmark queries ────────────────────────────────────
queries = [
    ("Phạm vi điều chỉnh của Luật AI là gì?",                "dieu_1"),
    ("Ai chịu trách nhiệm bồi thường khi AI rủi ro cao gây thiệt hại?", "dieu_28"),
    ("AI tạo sinh có phải dán nhãn không?",                   "dieu_10"),
    ("Các hành vi bị cấm trong lĩnh vực AI?",                 "dieu_6"),
    ("Các mức độ rủi ro của hệ thống trí tuệ nhân tạo?",      "dieu_8"),
]

hits = 0
print(f"{'Query':<55} {'Top1 ID':<30} {'Score':>6}  {'Relevant?'}")
print("-"*110)

for q, expected_key in queries:
    results = store.search(q, top_k=3)
    top1 = results[0]
    score = round(top1["score"], 3)
    top1_id = top1["id"]
    dieu_num = expected_key.split("_")[1]          # "1", "28", "10"...
    expected_dieu_str = f"Điều {dieu_num}"         # "Điều 1", "Điều 28"...
    # Relevant nếu metadata['dieu'] trong top-3 khớp với expected
    relevant = any(
        r["metadata"].get("dieu", "") == expected_dieu_str
        for r in results
    )
    if relevant:
        hits += 1
    print(f"{q:<55} {top1_id:<30} {score:>6}  {'YES' if relevant else 'NO'}")

print("-"*110)
hit_rate = hits / len(queries)
print(f"\n>>> Relevant in Top-3: {hits} / {len(queries)}")
print(f">>> Retrieval Quality: {hit_rate*10:.1f}/10  ({hit_rate*100:.0f}%)")

# ── Thống kê Chunking ─────────────────────────────────────
lengths = [len(r["content"]) for r in data["records"]]
print(f"\n=== Chunking Stats (Custom Strategy) ===")
print(f"  Chunk Count : {len(lengths)}")
print(f"  Avg Length  : {sum(lengths)/len(lengths):.1f} chars")
print(f"  Min/Max     : {min(lengths)} / {max(lengths)} chars")
