import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

ADAPTERS = {
    "finance": "zjudai/flowertune-finance-lora-mistral-7b-instruct-v0.3",
    "brainstorm": "DevQuasar/Mistral-7B-Instruct-v0.3_brainstorm-v3.1_adapter",
    "orpo_sft": "chchen/Mistral-7B-Instruct-v0.3-ORPO-SFT",
    "generic_sft": "yifanxie/mistralai-Mistral-7B-Instruct-v0.3-1719254473",
}

OFFLOAD_BASE = "offload/mistral"
OFFLOAD_PEFT = "offload/mistral_peft"

MAX_MEMORY = {
    0: "5GiB",
    "cpu": "20GiB",
}

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model with offload...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=MAX_MEMORY,
    offload_folder=OFFLOAD_BASE,
    low_cpu_mem_usage=True,
)

base_model.eval()

loaded_adapters = []

first_name = None
first_repo = None

for name, repo in ADAPTERS.items():
    first_name = name
    first_repo = repo
    break

print(f"Loading first adapter: {first_name} -> {first_repo}", flush=True)

model = PeftModel.from_pretrained(
    base_model,
    first_repo,
    adapter_name=first_name,
    device_map="auto",
    max_memory=MAX_MEMORY,
    offload_folder=OFFLOAD_PEFT,
)

loaded_adapters.append(first_name)

for name, repo in list(ADAPTERS.items())[1:]:
    print(f"\nTrying to load adapter: {name} -> {repo}", flush=True)

    try:
        model.load_adapter(
            repo,
            adapter_name=name,
            device_map="auto",
            max_memory=MAX_MEMORY,
            offload_folder=OFFLOAD_PEFT,
        )
        loaded_adapters.append(name)
        print(f"SUCCESS: loaded {name}", flush=True)

    except Exception as e:
        print(f"FAILED: could not load {name}", flush=True)
        print(str(e)[:800], flush=True)

model.eval()

print("\nLoaded adapters:", loaded_adapters, flush=True)


def generate(prompt, mode="base", max_new_tokens=90):
    formatted = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    if mode == "base":
        context = model.disable_adapter()
    else:
        model.set_adapter(mode)
        context = torch.no_grad()

    with context:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


TESTS = [
    {
        "label": "finance",
        "prompt": "Classify the financial sentiment as bullish, bearish, or neutral: The company beat earnings expectations and raised full-year guidance.",
    },
    {
        "label": "brainstorm",
        "prompt": "I want to build a small AI project around dynamic LoRA routing. Help me brainstorm directions by asking useful questions.",
    },
    {
        "label": "generic",
        "prompt": "Give a concise answer: what is the difference between fine-tuning and retrieval augmented generation?",
    },
]

for test in TESTS:
    print("\n" + "=" * 90)
    print("TEST:", test["label"])
    print("PROMPT:", test["prompt"])

    print("\n--- BASE ---")
    print(generate(test["prompt"], mode="base"))

    for adapter_name in loaded_adapters:
        print(f"\n--- ADAPTER: {adapter_name} ---")
        print(generate(test["prompt"], mode=adapter_name))