import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import re
import time

from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from src.pipeline.rag_pipeline import RAGPipeline

# =========================
# LOAD MODEL ONCE
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
    return int(overlap < 5)


# =========================
# 🔥 GROUNDING SCORE (NEW)
# =========================
def grounding_score(answer, context):
    answer_words = set(normalize(answer).split())
    context_words = set(normalize(context).split())

    if len(answer_words) == 0:
        return 0

    overlap = len(answer_words & context_words)

    return overlap / len(answer_words)


# =========================
# SEMANTIC SIMILARITY
# =========================
def semantic_similarity_batch(preds, truths):
    emb1 = sim_model.encode(preds, convert_to_tensor=True)
    emb2 = sim_model.encode(truths, convert_to_tensor=True)

    scores = util.cos_sim(emb1, emb2)
    return [float(scores[i][i]) for i in range(len(preds))]


# =========================
# BLEU SCORE
# =========================
def compute_bleu(pred, truth):
    return sentence_bleu([normalize(truth).split()], normalize(pred).split())


# =========================
# EVALUATE MODEL
# =========================
def evaluate_model(model_name, model_path, sample_size=100):
    print(f"\n🔍 Evaluating: {model_name}")

    rag = RAGPipeline(model_name=model_path)

    df = pd.read_csv("data/processed/clean_data.csv")
    df = df.sample(sample_size)

    preds, truths, contexts = [], [], []

    f1_scores = []
    em_scores = []
    hallucinations = []
    bleu_scores = []
    grounding_scores = []
    latencies = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        ground_truth = row["answer"]

        # =========================
        # ⏱ LATENCY START
        # =========================
        start_time = time.time()

        result = rag.query(question)

        latency = time.time() - start_time
        latencies.append(latency)

        pred = result["answer"]
        context = " ".join(result["sources"])

        preds.append(pred)
        truths.append(ground_truth)
        contexts.append(context)

        # =========================
        # METRICS
        # =========================
        f1_scores.append(compute_f1(pred, ground_truth))
        em_scores.append(exact_match(pred, ground_truth))
        hallucinations.append(is_hallucinated(pred, context))
        bleu_scores.append(compute_bleu(pred, ground_truth))
        grounding_scores.append(grounding_score(pred, context))

    similarities = semantic_similarity_batch(preds, truths)

    return {
        "Model": model_name,
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "Exact Match": sum(em_scores) / len(em_scores),
        "Hallucination Rate": sum(hallucinations) / len(hallucinations),
        "Semantic Similarity": sum(similarities) / len(similarities),
        "BLEU Score": sum(bleu_scores) / len(bleu_scores),
        "Grounding Score": sum(grounding_scores) / len(grounding_scores),
        "Latency (sec)": sum(latencies) / len(latencies),
    }


# =========================
# PLOT
# =========================
def plot_results(df):
    metrics = [
        "F1 Score",
        "Exact Match",
        "Hallucination Rate",
        "Semantic Similarity",
        "BLEU Score",
        "Grounding Score",
        "Latency (sec)"
    ]

    df.set_index("Model")[metrics].plot(kind="bar")

    plt.title("Baseline vs LoRA vs DPO")
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

    lora = evaluate_model(
        "LoRA",
        "models/lora_model"
    )

    dpo = evaluate_model(
        "DPO",
        "models/dpo_model"
    )

    results.extend([baseline, lora, dpo])

    df_results = pd.DataFrame(results)

    print("\n📊 FINAL COMPARISON TABLE:\n")
    print(df_results)

    df_results.to_csv("evaluation_results.csv", index=False)

    plot_results(df_results)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    compare()