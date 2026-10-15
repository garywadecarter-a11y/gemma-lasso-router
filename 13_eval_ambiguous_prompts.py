import json
import random
import re
import pickle
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "google/gemma-3-1b-it"
random.seed(789)

ADAPTERS = [
    "math",
    "harder_math",
    "code",
    "sql",
    "not_enough_info",
    "concise_explain",
]

MODES = [
    "base",
    "routed",
]

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("router/multi_l1_router.pkl", "rb") as f:
    router = pickle.load(f)

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


def choose_route(prompt):
    x = embedder.encode([prompt])
    return router.predict(x)[0]


def clean_answer(decoded):
    if "model\n" in decoded:
        return decoded.split("model\n")[-1].strip()
    return decoded.strip()


def generate(prompt, mode, max_new_tokens=120):
    route = mode

    if mode == "routed":
        route = choose_route(prompt)

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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return route, clean_answer(decoded)


def extract_code(answer):
    fence = re.search(r"```(?:python)?\s*(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    idx = answer.find("def ")
    if idx == -1:
        return ""
    return answer[idx:].strip()


def run_code_test(answer, func_name, tests):
    code = extract_code(answer)

    if "def " not in code:
        return False

    safe_globals = {
        "__builtins__": {
            "len": len,
            "sum": sum,
            "max": max,
            "min": min,
            "range": range,
            "set": set,
            "list": list,
            "sorted": sorted,
            "True": True,
            "False": False,
            "round": round,
            "dict": dict,
            "str": str,
            "int": int,
            "float": float,
        }
    }

    safe_locals = {}

    try:
        exec(code, safe_globals, safe_locals)
    except Exception:
        return False

    if func_name not in safe_locals:
        return False

    fn = safe_locals[func_name]

    try:
        for args, expected in tests:
            result = fn(*args)
            if result != expected:
                return False
    except Exception:
        return False

    return True


def score_json_obj(answer, expected):
    match = re.search(r"\{.*?\}", answer, re.DOTALL)
    if not match:
        return False

    try:
        obj = json.loads(match.group(0))
    except Exception:
        return False

    return obj == expected


def normalize_sql(answer):
    text = answer.lower()
    text = re.sub(r"```(?:sql)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_sql_terms(answer, required):
    text = normalize_sql(answer)
    return all(term.lower() in text for term in required)


def score_has_refusal(answer):
    text = answer.lower()
    return (
        "not have enough information" in text
        or "does not mention" in text
        or "not stated" in text
        or "not provided" in text
    )


def score_concise(answer):
    return 5 <= len(answer.split()) <= 90


tasks = []

# 1. Math + JSON
tasks.append({
    "name": "math_json",
    "prompt": "Calculate 37 * 42 and return valid JSON only with fields operation and result.",
    "score": lambda ans: score_json_obj(ans, {"operation": "37 * 42", "result": 1554}),
})

tasks.append({
    "name": "math_json",
    "prompt": "What is 15% of 840? Return JSON only with fields percent, base, and answer.",
    "score": lambda ans: score_json_obj(ans, {"percent": 15, "base": 840, "answer": 126}),
})

# 2. Code + math
tasks.append({
    "name": "code_math",
    "prompt": "Write a Python function named discounted_price that takes price and returns the price after a 20% discount. Return code only.",
    "score": lambda ans: run_code_test(ans, "discounted_price", [((100,), 80.0), ((50,), 40.0)]),
})

tasks.append({
    "name": "code_math",
    "prompt": "Write a Python function named multiply_then_add that returns a * b + c. Return code only.",
    "score": lambda ans: run_code_test(ans, "multiply_then_add", [((2, 3, 4), 10), ((5, 6, 7), 37)]),
})

# 3. Code + JSON
tasks.append({
    "name": "code_json",
    "prompt": "Write a Python function named make_person that returns a dictionary with keys name, age, and city. Return code only.",
    "score": lambda ans: run_code_test(ans, "make_person", [(("Gary", 35, "Dallas"), {"name": "Gary", "age": 35, "city": "Dallas"})]),
})

tasks.append({
    "name": "code_json",
    "prompt": "Write a Python function named make_result that takes x and returns {'result': x}. Return code only.",
    "score": lambda ans: run_code_test(ans, "make_result", [((7,), {"result": 7}), ((12,), {"result": 12})]),
})

# 4. SQL + concise explanation
tasks.append({
    "name": "sql_explain",
    "prompt": "Write SQL to calculate total order amount by customer_id from orders, then explain it in one brief sentence.",
    "score": lambda ans: score_sql_terms(ans, ["select", "customer_id", "sum(", "amount", "from orders", "group by"]) and score_concise(ans),
})

tasks.append({
    "name": "sql_explain",
    "prompt": "Write SQL using a CTE to calculate total claim_amount by policy_id from claims, then briefly explain it.",
    "score": lambda ans: score_sql_terms(ans, ["with", "policy_id", "sum(", "claim_amount", "from claims", "group by"]) and score_concise(ans),
})

# 5. Context + JSON
tasks.append({
    "name": "context_json_present",
    "prompt": "Use only the context and return JSON only.\n\nContext: The flight departs at 7:20 AM and boarding starts at 6:45 AM.\n\nQuestion: When does boarding start?\n\nReturn fields answer and source.",
    "score": lambda ans: score_json_obj(ans, {"answer": "6:45 AM", "source": "context"}),
})

tasks.append({
    "name": "context_json_absent",
    "prompt": "Use only the context and return JSON only.\n\nContext: The warranty lasts 12 months and covers manufacturing defects.\n\nQuestion: Does it cover water damage?\n\nReturn fields answer and source.",
    "score": lambda ans: ("water" in ans.lower()) and ("not" in ans.lower() or "enough" in ans.lower() or "does not" in ans.lower()),
})

# 6. Context + concise explanation
tasks.append({
    "name": "context_concise_absent",
    "prompt": "Answer using only the context. Keep it brief.\n\nContext: The museum is open Tuesday through Sunday from 10 AM to 5 PM.\n\nQuestion: How much does admission cost?",
    "score": lambda ans: score_has_refusal(ans) and score_concise(ans),
})

tasks.append({
    "name": "context_concise_present",
    "prompt": "Answer using only the context. Keep it brief.\n\nContext: The museum is open Tuesday through Sunday from 10 AM to 5 PM.\n\nQuestion: When is the museum open?",
    "score": lambda ans: ("tuesday" in ans.lower() and "sunday" in ans.lower() and "10" in ans and "5" in ans),
})


results = {mode: {} for mode in MODES}
route_counts = {}

for mode in MODES:
    print(f"\nTesting mode: {mode}", flush=True)

    for task in tasks:
        name = task["name"]
        prompt = task["prompt"]
        scorer = task["score"]

        route, ans = generate(prompt, mode)

        if mode == "routed":
            route_counts[route] = route_counts.get(route, 0) + 1

        ok = scorer(ans)

        if name not in results[mode]:
            results[mode][name] = [0, 0]

        results[mode][name][1] += 1
        if ok:
            results[mode][name][0] += 1

        print(f"\n[{mode}] task={name} route={route} ok={ok}")
        print("PROMPT:", prompt)
        print("ANSWER:", ans[:500].replace("\n", "\\n"))

print("\n================ AMBIGUOUS RESULTS ================")

for mode in MODES:
    total_correct = 0
    total_count = 0

    print(f"\n{mode.upper()}")

    for name in sorted(results[mode]):
        correct, count = results[mode][name]
        total_correct += correct
        total_count += count
        print(f"  {name:24s}: {correct}/{count} = {correct / count:.1%}")

    print(f"  {'OVERALL':24s}: {total_correct}/{total_count} = {total_correct / total_count:.1%}")

if route_counts:
    print("\nRouted adapter choices:")
    for route, count in sorted(route_counts.items(), key=lambda x: x[0]):
        print(f"  {str(route):18s}: {count}")