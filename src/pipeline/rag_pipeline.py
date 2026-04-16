from src.retriever.retrieve import Retriever
from src.generator.generate import generate_answer

retriever = Retriever()


# 🔥 Query normalization
def clean_query(query):
    return query.lower().strip()


def rag_pipeline(query):
    query = clean_query(query)

    docs = retriever.search(query)

    context = "\n".join(docs)

    answer = generate_answer(query, context)

    return {
        "answer": answer,
        "context": docs
    }