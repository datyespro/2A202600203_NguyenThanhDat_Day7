from pathlib import Path
from dotenv import load_dotenv
from src import OpenAIEmbedder

load_dotenv(dotenv_path=Path('.env'), override=False)
embedder = OpenAIEmbedder()
print(f'Backend: {embedder._backend_name}')
print(f'Vector length: {len(embedder("embedding smoke test"))}')
