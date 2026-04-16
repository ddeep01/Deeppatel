from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_answer(query, context):
    prompt = f"""
You are a medical expert.

Answer clearly and correctly.

Use ONLY the given context.
If context is not relevant, say "I don't know".

Give a short, factual answer.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer