import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER = "zjudai/flowertune-finance-lora-mistral-7b-instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

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

def generate(prompt, use_adapter=False, max_new_tokens=120):
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

prompts = [
    "Classify the financial sentiment as bullish, bearish, or neutral: The company beat earnings expectations and raised full-year guidance.",
    "Classify the financial sentiment as bullish, bearish, or neutral: Revenue declined, margins compressed, and management warned of weaker demand next quarter.",
    "Explain in one concise paragraph why rising interest rates can reduce equity valuations.",
    "Given this headline, identify the likely market impact: Central bank signals rate cuts may come sooner than expected.",
    "Summarize the risk in this statement: The company has high short-term debt maturities and negative free cash flow.",
]

for prompt in prompts:
    print("\n" + "=" * 80)
    print("PROMPT:", prompt)

    print("\n--- BASE ---")
    print(generate(prompt, use_adapter=False))

    print("\n--- FINANCE ADAPTER ---")
    print(generate(prompt, use_adapter=True))