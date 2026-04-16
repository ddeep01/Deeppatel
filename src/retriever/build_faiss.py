import faiss
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

EMBED_PATH = BASE_DIR / "data/embeddings/embeddings.npy"
INDEX_PATH = BASE_DIR / "data/embeddings/faiss_index.index"


def build_index():
    embeddings = np.load(EMBED_PATH)

    dim = embeddings.shape[1]

    nlist = 256     # clusters
    m = 32          # PQ compression

    quantizer = faiss.IndexFlatIP(dim)

    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        nlist,
        m,
        8
    )

    print("Training index...")
    index.train(embeddings)

    print("Adding vectors...")
    index.add(embeddings)

    index.nprobe = 10  # search accuracy/speed tradeoff

    faiss.write_index(index, str(INDEX_PATH))

    print("✅ FAISS IVF-PQ index ready")


if __name__ == "__main__":
    build_index()