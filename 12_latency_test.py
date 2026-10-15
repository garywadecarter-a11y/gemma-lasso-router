import time
import pickle
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "google/gemma-3-1b-it"

ADAPTERS = [
    "math",
    "harder_math",
    "code",
    "sql",
    "not_enough_info",
    "concise_explain",
]

PROMPTS = [
    "Return valid JSON only with name=Gary, age=35, city=Dallas.",
    "Write a Python function named count_vowels that counts vowels in string s. Return code only.",
    "Write SQL to show each customer_id and total order amount from the orders table. Return SQL only.",
    "Answer using only the context.\n\nContext: The warranty lasts 12 months and covers manufacturing defects.\n\nQuestion: Does it cover water damage?",
    "Explain LoRA in one concise paragraph.",
    "What is 37 * 42? Give only the final answer.",
    "What is 15% of 840? Return only the number.",
]

ROUNDS = 3
MAX_NEW_TOKENS = 80

print("Loading router/embedder...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("router/multi_l1_router.pkl", "rb") as f:
    router = pickle.load(f)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "adapters/math", adapter_name="math")

for adapter in ADAPTERS:
    if adapter == "math":
        continue
    model.load_adapter(f"adapters/{adapter}", adapter_name=adapter)

model.eval()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def choose_route(prompt):
    x = embedder.encode([prompt])
    return router.predict(x)[0]


def generate(prompt, route):
    chat_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    if route == "base":
        context = model.disable_adapter()
    else:
        model.set_adapter(route)
        context = torch.no_grad()

    with context:
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    return outputs


def time_call(fn, *args):
    sync()
    start = time.perf_counter()
    result = fn(*args)
    sync()
    end = time.perf_counter()
    return end - start, result


# Warmup
print("Warming up...")
for p in PROMPTS[:2]:
    generate(p, "base")
    generate(p, "code")
    choose_route(p)

print("\nTesting router-only latency...")
router_times = []

for _ in range(ROUNDS):
    for p in PROMPTS:
        elapsed, route = time_call(choose_route, p)
        router_times.append(elapsed)

print(f"Router-only avg seconds: {sum(router_times) / len(router_times):.4f}")


def benchmark_mode(mode):
    times = []
    routes = {}

    for _ in range(ROUNDS):
        for p in PROMPTS:
            if mode == "routed":
                route_time, route = time_call(choose_route, p)
                routes[route] = routes.get(route, 0) + 1
                gen_time, _ = time_call(generate, p, route)
                elapsed = route_time + gen_time
            else:
                elapsed, _ = time_call(generate, p, mode)

            times.append(elapsed)

    avg = sum(times) / len(times)
    return avg, routes


print("\nTesting generation latency...")

modes = [
    "base",
    "code",
    "sql",
    "not_enough_info",
    "concise_explain",
    "routed",
]

for mode in modes:
    avg, routes = benchmark_mode(mode)
    print(f"{mode:18s} avg seconds per prompt: {avg:.3f}")
    if routes:
        print("  routed choices:", routes)