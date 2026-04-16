import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data/processed/qa_dataset_large.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/chunked_data.csv"


def chunk_text(text, chunk_size=150, overlap=40):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk.split()) < 25:
            continue

        chunks.append(chunk)

    return chunks


def create_chunks():
    df = pd.read_csv(INPUT_PATH)

    df["question"] = df["question"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)

    all_chunks = []

    for i, (q, a) in enumerate(zip(df["question"], df["answer"])):
        text = (q + " " + a).strip()

        if not text:
            continue

        chunks = chunk_text(text)

        for c in chunks:
            all_chunks.append({"text": c})

        if i % 10000 == 0:
            print(f"Processed {i}")

    pd.DataFrame(all_chunks).to_csv(OUTPUT_PATH, index=False)

    print("✅ Chunking done")


if __name__ == "__main__":
    create_chunks()