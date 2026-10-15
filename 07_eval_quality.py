import json
import random
import re
import pickle
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "google/gemma-3-1b-it"
N_MATH = 50
N_JSON = 50
random.seed(123)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("router/l1_router.pkl", "rb") as f:
    router = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "adapters/math", adapter_name="math")
model.load_adapter("adapters/json", adapter_name="json")
model.eval()

def choose_adapter(prompt):
    x = embedder.encode([prompt])
    return router.predict(x)[0]

def clean_answer(decoded):
    if "model\n" in decoded:
        return decoded.split("model\n")[-1].strip()
    return decoded.strip()

def generate_with_mode(prompt, mode):
    chat_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    if mode == "routed":
        adapter = choose_adapter(prompt)
        model.set_adapter(adapter)
        context = torch.no_grad()
    elif mode == "math":
        model.set_adapter("math")
        context = torch.no_grad()
    elif mode == "json":
        model.set_adapter("json")
        context = torch.no_grad()
    elif mode == "base":
        context = model.disable_adapter()
    else:
        raise ValueError(mode)

    with context:
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_answer(decoded)

def math_score(answer, expected):
    nums = re.findall(r"-?\d+", answer)
    if not nums:
        return False
    return int(nums[0]) == expected

def json_score(answer, expected):
    match = re.search(r"\{.*?\}", answer, re.DOTALL)
    if not match:
        return False
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return False
    return obj == expected

math_prompts = []
for _ in range(N_MATH):
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    prompt = random.choice([
        f"What is {a} * {b}? Give only the final answer.",
        f"Calculate {a} times {b}.",
        f"Solve: {a} x {b}. Return only the number.",
        f"What do you get if you multiply {a} by {b}?",
    ])
    math_prompts.append((prompt, a * b))

json_prompts = []
names = ["Gary", "Alice", "Bob", "Dina", "Evan", "Fatima", "Mia", "Noah"]
cities = ["Dallas", "Austin", "Chicago", "Seattle", "Denver", "Boston"]

for _ in range(N_JSON):
    name = random.choice(names)
    age = random.randint(18, 80)
    city = random.choice(cities)
    expected = {"name": name, "age": age, "city": city}
    prompt = random.choice([
        f"Return valid JSON only with name={name}, age={age}, city={city}.",
        f"Create a JSON object with name {name}, age {age}, city {city}.",
        f"Output only JSON: name={name}, age={age}, city={city}.",
        f"Format this as JSON with fields name, age, city: {name}, {age}, {city}.",
    ])
    json_prompts.append((prompt, expected))

modes = ["base", "math", "json", "routed"]
results = {mode: {"math": 0, "json": 0} for mode in modes}

print("Running evaluation...")

for mode in modes:
    print(f"\nTesting mode: {mode}")

    for prompt, expected in math_prompts:
        ans = generate_with_mode(prompt, mode)
        if math_score(ans, expected):
            results[mode]["math"] += 1

    for prompt, expected in json_prompts:
        ans = generate_with_mode(prompt, mode)
        if json_score(ans, expected):
            results[mode]["json"] += 1

print("\n================ RESULTS ================")
print(f"Math questions: {N_MATH}")
print(f"JSON questions: {N_JSON}\n")

for mode in modes:
    math_acc = results[mode]["math"] / N_MATH
    json_acc = results[mode]["json"] / N_JSON
    overall = (results[mode]["math"] + results[mode]["json"]) / (N_MATH + N_JSON)

    print(f"{mode.upper()}")
    print(f"  Math accuracy: {math_acc:.1%}")
    print(f"  JSON accuracy: {json_acc:.1%}")
    print(f"  Overall:       {overall:.1%}")