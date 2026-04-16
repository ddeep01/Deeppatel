import os
import xml.etree.ElementTree as ET
import pandas as pd
from datasets import load_dataset, load_from_disk
from multiprocessing import Pool, cpu_count
from pathlib import Path

# =========================
# BASE PATH (CRITICAL FIX)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# =========================
# CLEAN FUNCTION
# =========================
def clean_text(text):
    if text is None:
        return ""
    text = text.replace("Key Points", "")
    text = text.replace("\n", " ")
    return text.strip()


# =========================
# PUBMEDQA
# =========================
def load_pubmedqa():
    path = BASE_DIR / "data/raw/pubmedqa"
    print(f"Loading PubMedQA from: {path}")

    ds = load_from_disk(str(path))

    data = []
    for item in ds["train"]:
        data.append({
            "question": clean_text(item.get("question", "")),
            "answer": clean_text(item.get("long_answer", "")),
            "source": "pubmedqa"
        })

    return pd.DataFrame(data)


# =========================
# MEDQUAD (FIXED + DEBUG)
# =========================
def process_xml_file(file_path):
    local_data = []

    try:
        for event, elem in ET.iterparse(file_path, events=("end",)):
            if elem.tag == "QAPair":
                q = elem.findtext("Question")
                a = elem.findtext("Answer")

                if q and a:
                    local_data.append({
                        "question": clean_text(q),
                        "answer": clean_text(a),
                        "source": "medquad"
                    })

                elem.clear()

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return local_data


def load_medquad():
    base_path = BASE_DIR / "data/raw/MedQuAD"
    file_paths = []

    print(f"\nSearching MedQuAD in: {base_path}")

    # 🔥 FIX: convert Path → str
    for root, _, files in os.walk(str(base_path)):
        print(f"Scanning: {root}")  # DEBUG
        for file in files:
            if file.endswith(".xml"):
                file_paths.append(os.path.join(root, file))

    print(f"\nTotal XML files found: {len(file_paths)}")

    if len(file_paths) == 0:
        print("⚠️ WARNING: No XML files found. Check dataset download!")
        return pd.DataFrame()

    num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} CPU cores...")

    with Pool(num_workers) as pool:
        results = pool.map(process_xml_file, file_paths)

    data = [item for sublist in results for item in sublist]

    df = pd.DataFrame(data)

    print(f"MedQuAD loaded samples: {len(df)}")

    return df


# =========================
# MEDMCQA (FAST VERSION)
# =========================
def load_medmcqa():
    print("\nLoading MedMCQA from HuggingFace...")

    dataset = load_dataset("medmcqa")["train"]
    df = pd.DataFrame(dataset)

    options = df[["opa", "opb", "opc", "opd"]].values
    correct = df["cop"].values

    answers = [
        options[i][correct[i]] if correct[i] < 4 else options[i][0]
        for i in range(len(df))
    ]

    df_final = pd.DataFrame({
        "question": df["question"].apply(clean_text),
        "answer": [clean_text(a) for a in answers],
        "source": "medmcqa"
    })

    return df_final


# =========================
# MERGE EVERYTHING
# =========================
def merge_all():
    print("Current Working Directory:", os.getcwd())
    print("Base Directory:", BASE_DIR)

    # ---- Load datasets ----
    print("\nLoading PubMedQA...")
    df_pubmed = load_pubmedqa()
    print("PubMedQA:", len(df_pubmed))

    print("\nLoading MedQuAD...")
    df_medquad = load_medquad()
    print("MedQuAD:", len(df_medquad))

    print("\nLoading MedMCQA...")
    df_medmcqa = load_medmcqa()
    print("MedMCQA:", len(df_medmcqa))

    # 🔥 OPTIONAL: Balance dataset (recommended)
    if len(df_medquad) > 0:
        df_medquad = df_medquad.sample(min(50000, len(df_medquad)))

    # ---- Merge ----
    print("\nMerging datasets...")
    df = pd.concat([df_pubmed, df_medquad, df_medmcqa], ignore_index=True)

    # ---- Clean ----
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=["question"])

    df = df[df["question"] != ""]
    df = df[df["answer"] != ""]

    # ---- Save ----
    output_dir = BASE_DIR / "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / "qa_dataset_large.csv"
    df.to_csv(output_path, index=False)

    print("\n✅ FINAL DATASET SIZE:", len(df))
    print(f"Saved to: {output_path}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    merge_all()