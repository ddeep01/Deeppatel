import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data/processed/qa_dataset_large.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/clean_data.csv"


def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").strip()


def prepare():
    df = pd.read_csv(INPUT_PATH)

    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)

    df = df[(df["question"] != "") & (df["answer"] != "")]

    df = df.drop_duplicates(subset=["question"])

    df = df[df["answer"].str.split().str.len() > 5]

    # 🔥 KEY FIX: separator improves embedding
    df["text"] = df["question"] + " [SEP] " + df["answer"]

    df = df[["question", "answer", "text"]]

    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Clean data ready:", len(df))


if __name__ == "__main__":
    prepare()