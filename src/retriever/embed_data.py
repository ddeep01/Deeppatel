import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =========================
# BASE PATH
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data/processed/chunked_data.csv"
EMBED_DIR = BASE_DIR / "data/embeddings"
EMBED_PATH = EMBED_DIR / "embeddings.npy"
TEXT_PATH = EMBED_DIR / "texts.csv"


def generate_embeddings():
    print("Loading chunked data...")
    df = pd.read_csv(INPUT_PATH)

    # ✅ Ensure directory exists
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["text"].fillna("").astype(str).tolist()

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)

    # =========================
    # SAVE FILES
    # =========================
    np.save(EMBED_PATH, embeddings)
    df.to_csv(TEXT_PATH, index=False)

    print("✅ Embeddings saved successfully!")
    print(f"Embeddings: {EMBED_PATH}")
    print(f"Texts: {TEXT_PATH}")


if __name__ == "__main__":
    generate_embeddings()