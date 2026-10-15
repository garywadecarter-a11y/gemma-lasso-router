print("Starting simple math LoRA training...", flush=True)

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "google/gemma-3-1b-it"

Path("adapters").mkdir(exist_ok=True)

class JsonlTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line)["text"])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.rows[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

print("Adding LoRA...", flush=True)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

print("Loading dataset...", flush=True)
dataset = JsonlTextDataset("data/math_train.jsonl", tokenizer)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

print("Training...", flush=True)
max_steps = 100

for step, batch in enumerate(loader, start=1):
    batch = {k: v.to(model.device) for k, v in batch.items()}

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"step {step} loss {loss.item():.4f}", flush=True)

    if step >= max_steps:
        break

print("Saving adapter...", flush=True)
model.save_pretrained("adapters/math")
tokenizer.save_pretrained("adapters/math")

print("Saved math adapter to adapters/math", flush=True)