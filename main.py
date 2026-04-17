from src.pipeline.rag_pipeline import RAGPipeline


def format_answer(text):
    # Clean unwanted prompt leakage
    bad_phrases = [
        "ONLY from the provided context",
        "Do NOT use prior knowledge",
        "If answer is not clearly in context"
    ]

    for phrase in bad_phrases:
        text = text.replace(phrase, "")

    text = text.strip()

    # Format into readable paragraph
    return "\n".join([line.strip() for line in text.split("\n") if line.strip()])


if __name__ == "__main__":
    print("🚀 Medical RAG System")

    print("\nSelect Model:")
    print("1. Baseline")
    print("2. DPO (Recommended)")

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