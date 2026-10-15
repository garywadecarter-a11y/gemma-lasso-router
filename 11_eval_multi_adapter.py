import json
import random
import re
import pickle
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "google/gemma-3-1b-it"
random.seed(456)

# Increase later if this runs fast enough.
N_PER_TASK = 10

ADAPTERS = [
    "math",
    "harder_math",
    "code",
    "sql",
    "not_enough_info",
    "concise_explain",
]

# Keep this simple for now: main comparison is base vs routed.
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
    pred = router.predict(x)[0]
    return pred


def clean_answer(decoded):
    if "model\n" in decoded:
        return decoded.split("model\n")[-1].strip()
    return decoded.strip()


def generate(prompt, mode, max_new_tokens=100):
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


def extract_first_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    val = float(nums[0])
    if val.is_integer():
        return int(val)
    return val


def close_enough(a, b, tol=1e-4):
    if a is None:
        return False
    return abs(float(a) - float(b)) <= tol


def score_math(answer, expected):
    val = extract_first_number(answer)
    return close_enough(val, expected)


def score_json(answer, expected):
    match = re.search(r"\{.*?\}", answer, re.DOTALL)
    if not match:
        return False

    try:
        obj = json.loads(match.group(0))
    except Exception:
        return False

    return obj == expected


# ============================================================
# Improved code scoring: execute generated function
# ============================================================

def extract_code(answer):
    fence = re.search(r"```(?:python)?\s*(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    idx = answer.find("def ")
    if idx == -1:
        return ""

    return answer[idx:].strip()


def score_code(answer, test_spec):
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
        }
    }

    safe_locals = {}

    try:
        exec(code, safe_globals, safe_locals)
    except Exception:
        return False

    func_name = test_spec["func_name"]
    tests = test_spec["tests"]

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


# ============================================================
# Improved SQL scoring: stricter structural checks
# ============================================================

def normalize_sql(answer):
    text = answer.lower()
    text = re.sub(r"```(?:sql)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_sql(answer, spec):
    text = normalize_sql(answer)

    required = spec.get("required", [])
    forbidden = spec.get("forbidden", [])
    one_of = spec.get("one_of", [])

    if not text.startswith("select") and not text.startswith("with"):
        return False

    for term in required:
        if term.lower() not in text:
            return False

    for term in forbidden:
        if term.lower() in text:
            return False

    for options in one_of:
        if not any(opt.lower() in text for opt in options):
            return False

    return True


def score_nei(answer, expected_kind):
    text = answer.lower()
    refusal = (
        "not have enough information" in text
        or "does not mention" in text
        or "not stated" in text
        or "not provided" in text
    )

    if expected_kind == "absent":
        return refusal

    return not refusal


def score_concise(answer):
    words = answer.split()
    return 5 <= len(words) <= 80


# ============================================================
# Build eval tasks
# ============================================================

tasks = []

# ----------------------------
# JSON/base tasks
# ----------------------------

names = ["Gary", "Alice", "Bob", "Dina", "Evan", "Fatima"]
cities = ["Dallas", "Austin", "Chicago", "Seattle", "Denver", "Boston"]

for _ in range(N_PER_TASK):
    name = random.choice(names)
    age = random.randint(18, 80)
    city = random.choice(cities)

    expected = {
        "name": name,
        "age": age,
        "city": city,
    }

    prompt = random.choice([
        f"Return valid JSON only with name={name}, age={age}, city={city}.",
        f"Create a JSON object with name {name}, age {age}, city {city}.",
        f"Output only JSON: name={name}, age={age}, city={city}.",
    ])

    tasks.append(("json", prompt, lambda ans, expected=expected: score_json(ans, expected)))


# ----------------------------
# Simple math tasks
# ----------------------------

for _ in range(N_PER_TASK):
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    expected = a * b

    prompt = random.choice([
        f"What is {a} * {b}? Give only the final answer.",
        f"Calculate {a} times {b}.",
        f"Solve: {a} x {b}. Return only the number.",
    ])

    tasks.append(("math", prompt, lambda ans, expected=expected: score_math(ans, expected)))


# ----------------------------
# Harder math tasks
# ----------------------------

for _ in range(N_PER_TASK):
    kind = random.choice(["percent", "discount", "average", "multi"])

    if kind == "percent":
        pct = random.choice([8, 12, 15, 18, 25, 30])
        base = random.randint(100, 3000)
        expected = base * pct / 100
        prompt = f"What is {pct}% of {base}? Return only the number."

    elif kind == "discount":
        price = random.randint(100, 2000)
        pct = random.choice([10, 15, 20, 25, 30])
        expected = price * (1 - pct / 100)
        prompt = f"An item costs {price} and is discounted by {pct}%. What is the final price?"

    elif kind == "average":
        nums = [random.randint(10, 200) for _ in range(4)]
        expected = sum(nums) / len(nums)
        prompt = f"What is the average of {', '.join(map(str, nums))}?"

    else:
        a = random.randint(5, 50)
        b = random.randint(5, 50)
        c = random.randint(2, 10)
        d = random.randint(1, 50)
        expected = (a + b) * c - d
        prompt = f"What is ({a} + {b}) * {c} - {d}?"

    tasks.append(("harder_math", prompt, lambda ans, expected=expected: score_math(ans, expected)))


# ----------------------------
# Code tasks with actual execution tests
# ----------------------------

code_eval_prompts = [
    (
        "Write a Python function named count_vowels that counts vowels in string s. Return code only.",
        {
            "func_name": "count_vowels",
            "tests": [
                (("hello",), 2),
                (("sky",), 0),
                (("AEIOU",), 5),
            ],
        },
    ),
    (
        "Write a Python function named is_palindrome that returns True if string s is a palindrome and False otherwise. Return code only.",
        {
            "func_name": "is_palindrome",
            "tests": [
                (("racecar",), True),
                (("hello",), False),
                (("madam",), True),
            ],
        },
    ),
    (
        "Write a Python function named remove_duplicates that removes duplicates from items while preserving order. Return code only.",
        {
            "func_name": "remove_duplicates",
            "tests": [
                (([1, 2, 2, 3, 1],), [1, 2, 3]),
                ((["a", "b", "a"],), ["a", "b"]),
                (([],), []),
            ],
        },
    ),
    (
        "Write a Python function named second_largest that returns the second largest unique number in nums. Return code only.",
        {
            "func_name": "second_largest",
            "tests": [
                (([1, 5, 3, 9],), 5),
                (([10, 10, 8, 7],), 8),
                (([-1, -5, -3],), -3),
            ],
        },
    ),
    (
        "Write a Python function named only_even that returns only even numbers from nums. Return code only.",
        {
            "func_name": "only_even",
            "tests": [
                (([1, 2, 3, 4],), [2, 4]),
                (([7, 9],), []),
                (([0, 2, 8],), [0, 2, 8]),
            ],
        },
    ),
]

for _ in range(N_PER_TASK):
    prompt, test_spec = random.choice(code_eval_prompts)
    tasks.append(("code", prompt, lambda ans, test_spec=test_spec: score_code(ans, test_spec)))


# ----------------------------
# SQL tasks with stricter checks
# ----------------------------

sql_eval_prompts = [
    (
        "Write SQL to show each customer_id and total order amount from the orders table. Return SQL only.",
        {
            "required": ["select", "customer_id", "sum(", "amount", "from orders", "group by customer_id"],
            "forbidden": ["join"],
        },
    ),
    (
        "Write SQL to join customers and orders by customer_id and show customer name and order amount. Return SQL only.",
        {
            "required": ["select", "join", "customers", "orders", "customer_id", "amount"],
            "one_of": [["c.name", "customers.name", "name"]],
        },
    ),
    (
        "Write SQL using a CTE to calculate total claim_amount by policy_id from claims. Return SQL only.",
        {
            "required": ["with", "policy_id", "sum(", "claim_amount", "from claims", "group by policy_id", "select"],
        },
    ),
    (
        "Write SQL to rank orders by amount within each customer_id. Return SQL only.",
        {
            "required": ["select", "rank()", "over", "partition by customer_id", "order by amount", "from orders"],
        },
    ),
    (
        "Write SQL to find customers who have no orders. Return SQL only.",
        {
            "required": ["select", "customers", "left join", "orders", "customer_id", "where"],
            "one_of": [["is null", "order_id is null"]],
        },
    ),
]

for _ in range(N_PER_TASK):
    prompt, spec = random.choice(sql_eval_prompts)
    tasks.append(("sql", prompt, lambda ans, spec=spec: score_sql(ans, spec)))


# ----------------------------
# Not enough info / groundedness tasks
# ----------------------------

nei_eval = [
    (
        "Answer using only the context.\n\nContext: The flight departs at 7:20 AM and boarding starts at 6:45 AM.\n\nQuestion: What gate does the flight leave from?",
        "absent",
    ),
    (
        "Answer using only the context.\n\nContext: The flight departs at 7:20 AM and boarding starts at 6:45 AM.\n\nQuestion: When does boarding start?",
        "present",
    ),
    (
        "Use only the information below. If not stated, say you do not have enough information.\n\nContext: The warranty lasts 12 months and covers manufacturing defects.\n\nQuestion: Does it cover water damage?",
        "absent",
    ),
    (
        "Use only the information below. If not stated, say you do not have enough information.\n\nContext: The hotel check-in time is 3 PM and checkout is 11 AM.\n\nQuestion: What time is checkout?",
        "present",
    ),
]

for _ in range(N_PER_TASK):
    prompt, kind = random.choice(nei_eval)
    tasks.append(("not_enough_info", prompt, lambda ans, kind=kind: score_nei(ans, kind)))


# ----------------------------
# Concise explanation tasks
# ----------------------------

for _ in range(N_PER_TASK):
    topic = random.choice(["PCA", "overfitting", "LoRA", "gradient descent", "transformers", "SQL"])

    prompt = random.choice([
        f"Explain {topic} in one concise paragraph.",
        f"Define {topic} in plain English.",
        f"What is {topic}? Keep the answer brief.",
    ])

    tasks.append(("concise_explain", prompt, score_concise))


random.shuffle(tasks)


# ============================================================
# Run eval
# ============================================================

results = {mode: {} for mode in MODES}
route_counts = {}

for mode in MODES:
    print(f"\nTesting mode: {mode}", flush=True)

    for category, prompt, scorer in tasks:
        if category not in results[mode]:
            results[mode][category] = [0, 0]

        route, ans = generate(prompt, mode)

        if mode == "routed":
            route_counts[route] = route_counts.get(route, 0) + 1

        ok = scorer(ans)

        results[mode][category][1] += 1
        if ok:
            results[mode][category][0] += 1

        print(".", end="", flush=True)

    print("", flush=True)


print("\n================ RESULTS ================")

categories = sorted({cat for cat, _, _ in tasks})

for mode in MODES:
    total_correct = 0
    total_count = 0

    print(f"\n{mode.upper()}")

    for cat in categories:
        correct, count = results[mode].get(cat, [0, 0])
        acc = correct / count if count else 0

        total_correct += correct
        total_count += count

        print(f"  {cat:18s}: {acc:6.1%}  ({correct}/{count})")

    overall = total_correct / total_count if total_count else 0
    print(f"  {'OVERALL':18s}: {overall:6.1%}  ({total_correct}/{total_count})")


if route_counts:
    print("\nRouted adapter choices:")
    for route, count in sorted(route_counts.items(), key=lambda x: x[0]):
        print(f"  {route:18s}: {count}")