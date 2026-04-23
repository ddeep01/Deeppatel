import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from trl import DPOTrainer, DPOConfig

# =========================
# CONFIG
# =========================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
LORA_PATH = "models/lora_model"   # 🔥 IMPORTANT
OUTPUT_DIR = "models/dpo_model"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data/processed/dpo_data.jsonl"


# =========================
# LOAD MODEL (FIXED)
# =========================
def load_model():
    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("🔹 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("🔹 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        is_trainable=True
    )

    return model, tokenizer


# =========================
# LOAD DATA
# =========================
def load_data():
    print(f"📂 Loading dataset from: {DATA_PATH}")

    dataset = load_dataset(
        "json",
        data_files=str(DATA_PATH)
    )["train"]

    def format_example(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    dataset = dataset.map(format_example)

    # 🔥 Reduce size for faster training (optional)
    dataset = dataset.select(range(min(5000, len(dataset))))

    print("📊 Dataset size:", len(dataset))

    return dataset


# =========================
# TRAIN
# =========================
def train():
    model, tokenizer = load_model()
    dataset = load_data()

    print("🚀 Starting DPO training...")

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,  # 🔥 lower LR for stability
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    trainer.train()

    print("💾 Saving DPO model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ DPO Training Complete")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()