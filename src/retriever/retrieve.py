import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from .reranker import Reranker

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INDEX_PATH = BASE_DIR / "data/embeddings/faiss_index.index"
TEXT_PATH = BASE_DIR / "data/embeddings/texts.csv"


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.texts = pd.read_csv(TEXT_PATH)

        self.reranker = Reranker()

    def search(self, query, k=5):
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        # 🔥 Reduced search space (less noise)
        D, I = self.index.search(query_vec, 20)

        docs = self.texts.iloc[I[0]]["text"].tolist()

        # 🔥 Remove short/noisy chunks
        docs = [d for d in docs if len(d.split()) > 30]

        # 🔥 Rerank
        docs = self.reranker.rerank(query, docs)

        # 🔥 Limit size
        docs = docs[:k]
        docs = [d[:300] for d in docs]

        return docs