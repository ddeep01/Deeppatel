import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

from src.retriever.reranker import Reranker

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INDEX_PATH = BASE_DIR / "data/embeddings/faiss.index"
DATA_PATH = BASE_DIR / "data/embeddings/data.csv"


class Retriever:
    def __init__(self):
        print("🔹 Loading Embedding Model...")

        # 🔥 UPGRADE: Better embedding model
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        print("🔹 Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("🔹 Loading dataset...")
        self.df = pd.read_csv(DATA_PATH)

        # 🔥 NEW: Reranker
        self.reranker = Reranker()

    def search(self, query, top_k=10):
        # =========================
        # 🔍 STEP 1: EMBEDDING SEARCH
        # =========================
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        results = self.df.iloc[indices[0]].copy()

        # =========================
        # 🔥 STEP 2: FILTER NOISE
        # =========================
        results = results[
            (results["answer"].str.len() > 40) &
            (~results["question"].str.lower().str.contains("which of the following")) &
            (~results["question"].str.lower().str.contains("true or false")) &
            (~results["question"].str.lower().str.contains("mcq"))
        ]

        if results.empty:
            return results

        # =========================
        # 🔥 STEP 3: RERANK (VERY IMPORTANT)
        # =========================
        docs = results["answer"].tolist()

        reranked_docs = self.reranker.rerank(query, docs)

        # Keep top 5 best docs
        final_docs = reranked_docs[:4]

        # =========================
        # 🔥 STEP 4: MAP BACK TO DATAFRAME
        # =========================
        final_results = results[results["answer"].isin(final_docs)]

        # Preserve order after reranking
        final_results["rank"] = final_results["answer"].apply(lambda x: final_docs.index(x))
        final_results = final_results.sort_values("rank")

        return final_results.head(5)