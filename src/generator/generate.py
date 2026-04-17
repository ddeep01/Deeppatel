from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
from pathlib import Path


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

        # =========================
        # 🔥 LOAD DPO MODEL
        # =========================
        if model_name is not None and model_name != base_model_name:
            BASE_DIR = Path(__file__).resolve().parent.parent.parent
            model_path = BASE_DIR / model_name
            model_path = str(model_path)

            print(f"📂 Loading DPO adapter from: {model_path}")

            if not os.path.exists(model_path):
                raise ValueError(f"❌ Model path not found: {model_path}")

            self.model = PeftModel.from_pretrained(
                base_model,
                model_path,
                local_files_only=True
            )
        else:
            print("🔹 Using baseline model")
            self.model = base_model

    def generate(self, query, context):
        prompt = f"""
You are a medical expert.

Answer the question in 2–3 short sentences.

Rules:
- Be concise
- Do NOT repeat instructions
- If unsure, say "I don't know"

Do NOT guess

Context:
{context[:1200]}

Question:
{query}

Final Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # =========================
        # 🔥 EXTRACT ANSWER
        # =========================
        if "Final Answer:" in decoded:
            answer = decoded.split("Final Answer:")[-1]
        elif "Answer:" in decoded:
            answer = decoded.split("Answer:")[-1]
        else:
            answer = decoded

        answer = " ".join(answer.strip().split())

        # =========================
        # 🔥 LIMIT TO 2 SENTENCES
        # =========================
        sentences = answer.split(".")
        answer = ". ".join(sentences[:2]).strip()

        if not answer.endswith("."):
            answer += "."

        # =========================
        # 🔥 UNKNOWN HANDLING
        # =========================
        if "i don't know" in answer.lower():
            return "I don't know"

        if len(answer.split()) < 5:
            return "I don't know"

        return answer