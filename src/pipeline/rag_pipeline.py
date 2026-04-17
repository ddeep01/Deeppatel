from src.retriever.retrieve import Retriever
from src.generator.generate import Generator


class RAGPipeline:
    def __init__(self, model_name=None):
        print(f"\n🚀 Initializing RAG Pipeline with model: {model_name}")

        self.retriever = Retriever()
        self.generator = Generator(model_name=model_name)

    # =========================
    # 🔥 DOMAIN FILTER
    # =========================
    def is_medical_query(self, query):
        medical_keywords = [
            "disease", "symptom", "treatment", "infection", "pain",
            "diabetes", "cancer", "virus", "bacteria", "lung",
            "heart", "fever", "covid", "pneumonia", "health"
        ]

        query = query.lower()

        return any(word in query for word in medical_keywords)

    def query(self, question):
        # =========================
        # 🔥 DOMAIN CHECK
        # =========================
        if not self.is_medical_query(question):
            return {
                "question": question,
                "answer": "I don't know",
                "sources": []
            }

        # 🔍 Step 1: Retrieve
        results = self.retriever.search(question)

        if results.empty:
            return {
                "question": question,
                "answer": "I don't know",
                "sources": []
            }

        # 🔥 Step 2: Context
        context = "\n\n".join(results["answer"].tolist())

        # =========================
        # 🔥 RELEVANCE CHECK
        # =========================
        if len(context.strip()) < 30:
            return {
                "question": question,
                "answer": "I don't know",
                "sources": results["question"].tolist()
            }

        # 🤖 Step 3: Generate
        answer = self.generator.generate(question, context)

        # =========================
        # 🔥 FALLBACK (VERY IMPORTANT)
        # =========================
        if answer == "I don't know":
            small_context = "\n\n".join(results["answer"].tolist()[:2])
            answer = self.generator.generate(question, small_context)

        return {
            "question": question,
            "answer": answer,
            "sources": results["question"].tolist()
        }