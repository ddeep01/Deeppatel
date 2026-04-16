import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from pathlib import Path
from trl import DPOConfig 
# =========================
# CONFIG
# =========================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
OUTPUT_DIR = "models/dpo_model"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data/processed/dpo_data.jsonl"


# =========================
# LOAD MODEL
# =========================
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 🔥 Important fix
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
    print(f"Loading dataset from: {DATA_PATH}")

    dataset = load_dataset(
        "json",
        data_files=str(DATA_PATH)
    )

    def format_example(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    dataset = dataset["train"].map(format_example)

    # 🔥 Optional: use small subset for testing
    dataset = dataset.select(range(min(5000, len(dataset))))

    print("Dataset size:", len(dataset))

    return dataset


# =========================
# APPLY LORA
# =========================
def apply_lora(model):
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
# TRAIN
# =========================
def train():
    model, tokenizer = load_model()
    dataset = load_data()

    model = apply_lora(model)

    training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none")

    trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer)

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ DPO Training Complete")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()