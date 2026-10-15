import pickle
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "google/gemma-3-1b-it"

# Load router
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("router/l1_router.pkl", "rb") as f:
    router = pickle.load(f)

# Load base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Load first adapter as PeftModel
model = PeftModel.from_pretrained(
    base_model,
    "adapters/math",
    adapter_name="math",
)

# Load second adapter into same model
model.load_adapter(
    "adapters/json",
    adapter_name="json",
)

model.eval()

def choose_adapter(prompt: str) -> str:
    x = embedder.encode([prompt])
    pred = router.predict(x)[0]
    return pred

def generate(prompt: str, max_new_tokens: int = 120):
    adapter = choose_adapter(prompt)

    # Dynamically activate LoRA adapter weights
    model.set_adapter(adapter)

    chat_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return adapter, decoded

if __name__ == "__main__":
    test_prompts = [
        "What is 37 * 42? Give only the final answer.",
        "Return valid JSON only with name=Gary, age=35, city=Dallas.",
        "Calculate 123 times 77.",
        "Create a JSON object for name Alice, age 29, city Austin.",
    ]

    for prompt in test_prompts:
        adapter, answer = generate(prompt)
        print("=" * 80)
        print("PROMPT:", prompt)
        print("ROUTED TO:", adapter)
        print("ANSWER:")
        print(answer)
        