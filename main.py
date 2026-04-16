from src.pipeline.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline()

    while True:
        query = input("\nEnter question: ")

        result = rag.query(query)

        print("\n=== ANSWER ===")
        print(result["answer"])

        print("\n=== SOURCES ===")
        for s in result["sources"]:
            print("-", s)