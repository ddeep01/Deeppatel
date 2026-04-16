import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INDEX_PATH = BASE_DIR / "data/embeddings/faiss.index"
DATA_PATH = BASE_DIR / "data/embeddings/data.csv"


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

        self.index = faiss.read_index(str(INDEX_PATH))
        self.df = pd.read_csv(DATA_PATH)

    def search(self, query, top_k=8):  # 🔥 increased top_k
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        results = self.df.iloc[indices[0]]

        return results