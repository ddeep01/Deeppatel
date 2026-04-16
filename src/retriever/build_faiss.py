import faiss
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

EMBED_PATH = BASE_DIR / "data/embeddings/embeddings.npy"
INDEX_PATH = BASE_DIR / "data/embeddings/faiss_index.index"


def build_index():
    embeddings = np.load(EMBED_PATH).astype("float32")

    dim = embeddings.shape[1]
    nlist = 100  # clusters

    print("Training FAISS IVF index...")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    index.train(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    print("✅ IVF FAISS index created")


if __name__ == "__main__":
    build_index()