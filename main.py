from src.pipeline.rag_pipeline import rag_pipeline

if __name__ == "__main__":
    while True:
        query = input("\nAsk medical q  uestion: ")

        if query.lower() == "exit":
            break

        result = rag_pipeline(query)

        print("\n💡 Answer:")
        print(result["answer"])

        print("\n📚 Sources:")
        for c in result["context"]:
            print("-", c[:120])