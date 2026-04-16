import pandas as pd
from pathlib import Path

# =========================
# BASE PATH
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data/processed/qa_dataset_large.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/chunked_data.csv"


# =========================
# CHUNK FUNCTION
# =========================
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        # ✅ Skip very small chunks (important for quality)
        if len(chunk.split()) < 20:
            continue

        chunks.append(chunk)

    return chunks


# =========================
# MAIN FUNCTION
# =========================
def create_chunks():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    # =========================
    # FIX: HANDLE NaN + TYPES
    # =========================
    df["question"] = df["question"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)

    print("Total rows:", len(df))

    all_chunks = []

    # =========================
    # FAST LOOP (ZIP INSTEAD OF ITERROWS)
    # =========================
    for i, (q, a, source) in enumerate(zip(df["question"], df["answer"], df["source"])):
        combined = (q + " " + a).strip()

        # Skip empty rows
        if not combined:
            continue

        chunks = chunk_text(combined)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": source
            })

        if i % 10000 == 0:
            print(f"Processed {i}")

    # =========================
    # CREATE DATAFRAME
    # =========================
    chunk_df = pd.DataFrame(all_chunks)

    print("Total chunks created:", len(chunk_df))

    # =========================
    # SAVE
    # =========================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunk_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Chunking complete")
    print(f"Saved to: {OUTPUT_PATH}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    create_chunks()