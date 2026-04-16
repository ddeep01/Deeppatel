from src.retriever.retrieve import Retriever
from src.generator.generate import Generator


class RAGPipeline:
    def __init__(self, model_name=None):
        """
        model_name:
        - None → Baseline model
        - "models/dpo_model" → DPO model
        """
        print(f"\n🚀 Initializing RAG Pipeline with model: {model_name}")

        self.retriever = Retriever()
        self.generator = Generator(model_name=model_name)

    def query(self, question):
        # 🔍 Step 1: Retrieve documents
        results = self.retriever.search(question)

        # 🔥 Step 2: Build context (IMPORTANT)
        context = "\n\n".join(results["answer"].tolist())

        # 🤖 Step 3: Generate answer
        answer = self.generator.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": results["question"].tolist()
        }