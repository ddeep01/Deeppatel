import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data/processed/clean_data.csv"
EMBED_DIR = BASE_DIR / "data/embeddings"

INDEX_PATH = EMBED_DIR / "faiss.index"
DATA_PATH = EMBED_DIR / "data.csv"


def generate_embeddings():
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings, dtype="float32")

    # FAISS (cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    df.to_csv(DATA_PATH, index=False)

    print("✅ Embeddings + FAISS ready")


if __name__ == "__main__":
    generate_embeddings()