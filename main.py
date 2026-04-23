from src.pipeline.rag_pipeline import RAGPipeline


def format_answer(text):
    # Clean unwanted phrases
    bad_phrases = [
        "ONLY from the provided context",
        "Do NOT use prior knowledge",
        "If answer is not clearly in context",
        "Final Answer:",
        "Answer:"
    ]

    for phrase in bad_phrases:
        text = text.replace(phrase, "")

    text = " ".join(text.strip().split())

    # Limit to 2 sentences
    sentences = text.split(".")
    text = ". ".join(sentences[:2]).strip()

    if not text.endswith("."):
        text += "."

    return text


if __name__ == "__main__":
    print("🚀 Medical RAG System")

    print("\nSelect Model:")
    print("1. Baseline")
    print("2. Final (DPO)")

    choice = input("Enter choice (1/2): ")

    if choice == "2":
        rag = RAGPipeline(model_name="models/dpo_model")
        print("✅ Using DPO Model")

    else:
        rag = RAGPipeline()
        print("✅ Using Baseline Model")

    while True:
        query = input("\n💬 Enter question (or 'exit'): ")

        if query.lower() == "exit":
            break

        result = rag.query(query)

        clean_answer = format_answer(result["answer"])

        print("\n🧠 ANSWER:")
        print(clean_answer)

        print("\n📚 SOURCES:")
        for s in result["sources"][:3]:
            print("-", s)