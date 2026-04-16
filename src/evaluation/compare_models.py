import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import re

from sentence_transformers import SentenceTransformer, util
from src.pipeline.rag_pipeline import RAGPipeline

# =========================
# LOAD SIMILARITY MODEL
# =========================
sim_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# =========================
# NORMALIZATION
# =========================
def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# =========================
# F1 SCORE
# =========================
def compute_f1(pred, truth):
    pred_tokens = normalize(pred).split()
    truth_tokens = normalize(truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)

# =========================
# EXACT MATCH
# =========================
def exact_match(pred, truth):
    return int(normalize(pred) == normalize(truth))

# =========================
# HALLUCINATION
# =========================
def is_hallucinated(answer, context):
    answer_words = set(normalize(answer).split())
    context_words = set(normalize(context).split())

    overlap = len(answer_words & context_words)

    return int(overlap < 3)

# =========================
# SEMANTIC SIMILARITY ⭐
# =========================
def semantic_similarity(pred, truth):
    emb1 = sim_model.encode(pred, convert_to_tensor=True)
    emb2 = sim_model.encode(truth, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2)

    return float(score)

# =========================
# EVALUATE MODEL
# =========================
def evaluate_model(model_name, model_path, sample_size=100):
    print(f"\n🔍 Evaluating: {model_name}")

    rag = RAGPipeline(model_name=model_path)

    df = pd.read_csv("data/processed/clean_data.csv")
    df = df.sample(sample_size)

    f1_scores, em_scores, hallucinations, similarities = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        ground_truth = row["answer"]

        result = rag.query(question)

        pred = result["answer"]
        context = " ".join(result["sources"])

        f1_scores.append(compute_f1(pred, ground_truth))
        em_scores.append(exact_match(pred, ground_truth))
        hallucinations.append(is_hallucinated(pred, context))
        similarities.append(semantic_similarity(pred, ground_truth))

    return {
        "Model": model_name,
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "Exact Match": sum(em_scores) / len(em_scores),
        "Hallucination Rate": sum(hallucinations) / len(hallucinations),
        "Semantic Similarity": sum(similarities) / len(similarities)
    }

# =========================
# PLOT
# =========================
def plot_results(df):
    metrics = ["F1 Score", "Exact Match", "Hallucination Rate", "Semantic Similarity"]

    df.set_index("Model")[metrics].plot(kind="bar")

    plt.title("Model Comparison (Baseline vs DPO)")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("evaluation_plot.png")
    plt.show()

# =========================
# MAIN
# =========================
def compare():
    results = []

    baseline = evaluate_model(
        "Baseline",
        "mistralai/Mistral-7B-Instruct-v0.1"
    )

    dpo = evaluate_model(
        "DPO",
        "models/dpo_model"
    )

    results.append(baseline)
    results.append(dpo)

    df_results = pd.DataFrame(results)

    print("\n📊 FINAL COMPARISON TABLE:\n")
    print(df_results)

    df_results.to_csv("evaluation_results.csv", index=False)

    plot_results(df_results)


if __name__ == "__main__":
    compare()