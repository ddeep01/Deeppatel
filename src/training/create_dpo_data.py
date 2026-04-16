import pandas as pd
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data/processed/clean_data.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/dpo_data.jsonl"


def create_dpo():
    df = pd.read_csv(INPUT_PATH)

    data = []

    answers = df["answer"].tolist()

    for i, row in df.iterrows():
        question = row["question"]
        correct = row["answer"]

        # ❌ create wrong answer randomly
        wrong = random.choice(answers)

        if wrong == correct:
            continue

        data.append({
            "prompt": question,
            "chosen": correct,
            "rejected": wrong
        })

        if i % 10000 == 0:
            print(f"Processed {i}")

    pd.DataFrame(data).to_json(OUTPUT_PATH, orient="records", lines=True)

    print("✅ DPO dataset created")


if __name__ == "__main__":
    create_dpo()