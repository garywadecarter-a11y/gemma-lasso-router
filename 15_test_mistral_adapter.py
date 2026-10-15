import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER = "zjudai/flowertune-finance-lora-mistral-7b-instruct-v0.3"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model with offload...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={
        0: "5GiB",
        "cpu": "20GiB",
    },
    offload_folder="offload/mistral",
    low_cpu_mem_usage=True,
)

base_model.eval()

print("Loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
    adapter_name="finance",
    device_map="auto",
    max_memory={
        0: "5GiB",
        "cpu": "20GiB",
    },
    offload_folder="offload/mistral_peft",
)

model.eval()


def generate(prompt, use_adapter=False, max_new_tokens=80):
    formatted = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    if use_adapter:
        model.set_adapter("finance")
        context = torch.no_grad()
    else:
        context = model.disable_adapter()

    with context:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = "Explain what financial sentiment analysis is in one concise paragraph."

print("\n================ BASE ================", flush=True)
print(generate(prompt, use_adapter=False))

print("\n================ FINANCE ADAPTER ================", flush=True)
print(generate(prompt, use_adapter=True))