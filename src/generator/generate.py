from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


class Generator:
    def __init__(self, model_name=None):
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"

        print(f"🔹 Loading base model: {base_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 🔥 If DPO model → load adapter
        if model_name is not None and model_name != base_model_name:
            print(f"🔹 Loading DPO adapter: {model_name}")
            self.model = PeftModel.from_pretrained(base_model, model_name)
        else:
            print("🔹 Using baseline model")
            self.model = base_model

    def generate(self, query, context):
        prompt = f"""
You are a medical assistant.

STRICT RULES:
- Answer ONLY from the provided context
- Do NOT use prior knowledge
- If answer is not clearly in context → say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.2,
            do_sample=False
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer" in decoded:
            answer = decoded.split("Answer")[-1].strip(": \n")
        else:
            answer = decoded.strip()

        if len(answer.split()) < 3:
            return "I don't know"

        return answer