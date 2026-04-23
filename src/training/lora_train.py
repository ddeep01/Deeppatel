import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from pathlib import Path
from trl import SFTTrainer, SFTConfig

# =========================
# CONFIG
# =========================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
OUTPUT_DIR = "models/lora_model"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data/processed/qa_dataset_large.csv"


# =========================
# LOAD MODEL
# =========================
def load_model():
    print("🔹 Loading base model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )

    return model, tokenizer


# =========================
# LOAD DATA
# =========================
def load_data():
    print(f"📂 Loading dataset from: {DATA_PATH}")

    dataset = load_dataset("csv", data_files=str(DATA_PATH))["train"]

    def format_example(example):
        return {
            "text": f"Question: {example['question']}\nAnswer: {example['answer']}"
        }

    dataset = dataset.map(format_example)

    # 🔥 Reduce size (important for GPU)
    dataset = dataset.select(range(min(10000, len(dataset))))

    print("📊 Dataset size:", len(dataset))

    return dataset


# =========================
# APPLY LORA
# =========================
def apply_lora(model):
    print("🔧 Applying LoRA...")

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


# =========================
# TOKENIZE DATASET (NEW FIX)
# =========================
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    dataset = dataset.map(tokenize_function, batched=True)
    return dataset


# =========================
# TRAIN
# =========================
def train():
    model, tokenizer = load_model()
    dataset = load_data()

    model = apply_lora(model)

    # 🔥 VERY IMPORTANT
    model.config.use_cache = False

    # 🔥 TOKENIZE (CRITICAL FIX)
    dataset = tokenize_dataset(dataset, tokenizer)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    print("🚀 Starting LoRA training...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    trainer.train()

    print("💾 Saving model...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ LoRA Training Complete")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()