import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model with GPU/CPU/disk offload...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={
        0: "5GiB",       # your RTX 3060 has 6GB; leave some headroom
        "cpu": "20GiB",  # adjust lower if your system RAM is smaller
    },
    offload_folder="offload/mistral",
    low_cpu_mem_usage=True,
)

model.eval()

prompt = "[INST] Explain LoRA, meaning Low-Rank Adaptation for neural network fine-tuning, in one concise paragraph. [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...", flush=True)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))