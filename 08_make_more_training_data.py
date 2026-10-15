import json
import random
from pathlib import Path

random.seed(123)
Path("data").mkdir(exist_ok=True)

def write_jsonl(path, rows):
    random.shuffle(rows)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

def chat_text(prompt, response):
    return {
        "text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
    }

def clean_num(x):
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return str(round(x, 4)).rstrip("0").rstrip(".")
    return str(x)

# ============================================================
# HARDER MATH DATA
# ============================================================

math_rows = []

for _ in range(10000):
    kind = random.choice([
        "multiply", "two_step_add", "two_step_subtract", "percent",
        "discount", "average", "ratio", "rate_time", "unit_price",
        "parentheses", "profit", "remaining", "increase", "decrease"
    ])

    if kind == "multiply":
        a, b = random.randint(10, 999), random.randint(10, 999)
        prompt = random.choice([
            f"What is {a} * {b}? Return only the final number.",
            f"Calculate {a} times {b}. Give just the answer.",
            f"Solve {a} x {b}. No explanation."
        ])
        ans = a * b

    elif kind == "two_step_add":
        a, b, c = random.randint(10, 300), random.randint(2, 80), random.randint(1, 500)
        prompt = random.choice([
            f"A store sells {a} items per day for {b} days, then sells {c} more. What is the total?",
            f"Compute {a} * {b} + {c}. Return only the number.",
            f"There are {b} boxes with {a} items each plus {c} extras. How many total?"
        ])
        ans = a * b + c

    elif kind == "two_step_subtract":
        a, b, c = random.randint(10, 200), random.randint(2, 80), random.randint(1, 500)
        start = a * b + c
        prompt = random.choice([
            f"A warehouse had {start} boxes. It shipped {a} boxes per day for {b} days. How many remain?",
            f"Compute {start} - ({a} * {b}). Return only the number.",
            f"Start with {start}, subtract {a} repeated {b} times. What remains?"
        ])
        ans = c

    elif kind == "percent":
        pct = random.choice([4, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40])
        base = random.randint(50, 5000)
        prompt = random.choice([
            f"What is {pct}% of {base}? Return only the number.",
            f"Calculate {pct} percent of {base}.",
            f"A quantity is {base}. Find {pct}% of it."
        ])
        ans = base * pct / 100

    elif kind == "discount":
        price = random.randint(20, 3000)
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60])
        prompt = random.choice([
            f"An item costs {price} dollars and is discounted by {pct}%. What is the final price?",
            f"Calculate the sale price of {price} after a {pct}% discount.",
            f"Price {price}, discount {pct}%. Return the final price only."
        ])
        ans = price * (1 - pct / 100)

    elif kind == "average":
        nums = [random.randint(5, 300) for _ in range(random.choice([3, 4, 5, 6]))]
        joined = ", ".join(map(str, nums))
        prompt = random.choice([
            f"What is the average of {joined}?",
            f"Find the mean of these numbers: {joined}.",
            f"Compute the arithmetic average of {joined}."
        ])
        ans = sum(nums) / len(nums)

    elif kind == "ratio":
        a, b = random.randint(2, 25), random.randint(2, 25)
        scale = random.randint(3, 80)
        first = a * scale
        second = b * scale
        prompt = random.choice([
            f"The ratio is {a}:{b}. If the first quantity is {first}, what is the second?",
            f"Red to blue is {a}:{b}. There are {first} red items. How many blue items?",
            f"In a {a} to {b} ratio, the first value is {first}. Find the second value."
        ])
        ans = second

    elif kind == "rate_time":
        rate, time = random.randint(5, 200), random.randint(2, 100)
        prompt = random.choice([
            f"A machine produces {rate} units per hour for {time} hours. How many units are produced?",
            f"At {rate} miles per hour for {time} hours, how many miles are traveled?",
            f"Compute output at rate {rate} for {time} periods."
        ])
        ans = rate * time

    elif kind == "unit_price":
        qty = random.randint(2, 50)
        unit = random.randint(2, 200)
        total = qty * unit
        prompt = random.choice([
            f"{qty} items cost {total} dollars total. What is the price per item?",
            f"Total cost is {total} for {qty} units. Find unit price.",
            f"Divide {total} by {qty}. Return the unit price."
        ])
        ans = total / qty

    elif kind == "parentheses":
        a, b, c, d = random.randint(2, 100), random.randint(2, 100), random.randint(2, 20), random.randint(1, 100)
        prompt = random.choice([
            f"What is ({a} + {b}) * {c} - {d}?",
            f"Add {a} and {b}, multiply by {c}, then subtract {d}.",
            f"Solve: ({a} + {b}) times {c} minus {d}."
        ])
        ans = (a + b) * c - d

    elif kind == "profit":
        revenue = random.randint(500, 10000)
        cost = random.randint(100, revenue - 1)
        prompt = random.choice([
            f"Revenue is {revenue} and cost is {cost}. What is profit?",
            f"A business earns {revenue} and spends {cost}. Find profit.",
            f"Compute profit = {revenue} - {cost}."
        ])
        ans = revenue - cost

    elif kind == "remaining":
        start = random.randint(100, 10000)
        used = random.randint(1, start - 1)
        prompt = random.choice([
            f"There were {start} units and {used} were used. How many remain?",
            f"Start with {start}. Remove {used}. Return remaining amount.",
            f"What is left after subtracting {used} from {start}?"
        ])
        ans = start - used

    elif kind == "increase":
        base = random.randint(50, 5000)
        inc = random.choice([5, 10, 12, 15, 20, 25, 30])
        prompt = random.choice([
            f"Increase {base} by {inc}%. What is the new value?",
            f"A value of {base} rises by {inc} percent. Find the final value.",
            f"Compute {base} after a {inc}% increase."
        ])
        ans = base * (1 + inc / 100)

    else:
        base = random.randint(50, 5000)
        dec = random.choice([5, 10, 12, 15, 20, 25, 30])
        prompt = random.choice([
            f"Decrease {base} by {dec}%. What is the new value?",
            f"A value of {base} falls by {dec} percent. Find the final value.",
            f"Compute {base} after a {dec}% decrease."
        ])
        ans = base * (1 - dec / 100)

    math_rows.append(chat_text(prompt, clean_num(ans)))

write_jsonl("data/harder_math_train.jsonl", math_rows)

# ============================================================
# CODE DATA
# ============================================================

code_rows = []

code_templates = [
    ("Write a Python function named {fname} that returns {a} + {b}. Return code only.",
     "def {fname}({a}, {b}):\n    return {a} + {b}"),
    ("Write a Python function named {fname} that returns {a} - {b}. Return code only.",
     "def {fname}({a}, {b}):\n    return {a} - {b}"),
    ("Write a Python function named {fname} that returns {a} * {b}. Return code only.",
     "def {fname}({a}, {b}):\n    return {a} * {b}"),
    ("Write a Python function named {fname} that returns True if {n} is even. Return code only.",
     "def {fname}({n}):\n    return {n} % 2 == 0"),
    ("Write a Python function named {fname} that returns True if {n} is odd. Return code only.",
     "def {fname}({n}):\n    return {n} % 2 != 0"),
    ("Write a Python function named {fname} that returns the larger of {a} and {b}. Return code only.",
     "def {fname}({a}, {b}):\n    return max({a}, {b})"),
    ("Write a Python function named {fname} that returns the smaller of {a} and {b}. Return code only.",
     "def {fname}({a}, {b}):\n    return min({a}, {b})"),
    ("Write a Python function named {fname} that returns the length of {items}. Return code only.",
     "def {fname}({items}):\n    return len({items})"),
    ("Write a Python function named {fname} that returns the first element of {items}. Return code only.",
     "def {fname}({items}):\n    return {items}[0]"),
    ("Write a Python function named {fname} that returns the last element of {items}. Return code only.",
     "def {fname}({items}):\n    return {items}[-1]"),
    ("Write a Python function named {fname} that reverses string {s}. Return code only.",
     "def {fname}({s}):\n    return {s}[::-1]"),
    ("Write a Python function named {fname} that lowercases string {s}. Return code only.",
     "def {fname}({s}):\n    return {s}.lower()"),
    ("Write a Python function named {fname} that uppercases string {s}. Return code only.",
     "def {fname}({s}):\n    return {s}.upper()"),
    ("Write a Python function named {fname} that returns only positive numbers from {nums}. Return code only.",
     "def {fname}({nums}):\n    return [n for n in {nums} if n > 0]"),
    ("Write a Python function named {fname} that returns only negative numbers from {nums}. Return code only.",
     "def {fname}({nums}):\n    return [n for n in {nums} if n < 0]"),
    ("Write a Python function named {fname} that squares every number in {nums}. Return code only.",
     "def {fname}({nums}):\n    return [n * n for n in {nums}]"),
    ("Write a Python function named {fname} that returns the sum of {nums}. Return code only.",
     "def {fname}({nums}):\n    return sum({nums})"),
    ("Write a Python function named {fname} that returns the average of {nums}. Return code only.",
     "def {fname}({nums}):\n    return sum({nums}) / len({nums})"),
    ("Write a Python function named {fname} that counts how often {item} appears in {items}. Return code only.",
     "def {fname}({items}, {item}):\n    return {items}.count({item})"),
    ("Write a Python function named {fname} that returns the value for {key} in dictionary {d}. Return code only.",
     "def {fname}({d}, {key}):\n    return {d}.get({key})"),
    ("Write a Python function named {fname} that removes duplicates from {items} while preserving order. Return code only.",
     "def {fname}({items}):\n    seen = set()\n    result = []\n    for item in {items}:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"),
    ("Write a Python function named {fname} that checks if string {s} is a palindrome. Return code only.",
     "def {fname}({s}):\n    cleaned = {s}.lower()\n    return cleaned == cleaned[::-1]"),
    ("Write a Python function named {fname} that counts vowels in string {s}. Return code only.",
     "def {fname}({s}):\n    return sum(1 for ch in {s}.lower() if ch in 'aeiou')"),
]

verbs = ["calculate", "compute", "get", "return", "find", "make", "build", "check", "create"]
objects = ["value", "result", "answer", "output", "number", "item", "total", "score", "flag"]

for _ in range(8000):
    pt, rt = random.choice(code_templates)
    values = {
        "fname": f"{random.choice(verbs)}_{random.choice(objects)}_{random.randint(1, 999999)}",
        "a": random.choice(["a", "x", "first", "left", "num1"]),
        "b": random.choice(["b", "y", "second", "right", "num2"]),
        "n": random.choice(["n", "number", "value", "x"]),
        "items": random.choice(["items", "values", "records", "data", "lst"]),
        "nums": random.choice(["nums", "numbers", "values", "amounts", "scores"]),
        "s": random.choice(["s", "text", "word", "name", "message"]),
        "d": random.choice(["d", "data", "record", "mapping"]),
        "key": random.choice(["key", "field", "name"]),
        "item": random.choice(["item", "value", "target"]),
    }
    code_rows.append(chat_text(pt.format(**values), rt.format(**values)))

write_jsonl("data/code_train.jsonl", code_rows)

# ============================================================
# SQL DATA
# ============================================================

sql_rows = []

schemas = {
    "customers": ["customer_id", "name", "state", "age", "signup_date"],
    "orders": ["order_id", "customer_id", "amount", "order_date", "status"],
    "employees": ["employee_id", "department", "salary", "hire_date", "manager_id"],
    "products": ["product_id", "name", "price", "category", "inventory"],
    "claims": ["claim_id", "policy_id", "claim_amount", "claim_date", "status"],
    "policies": ["policy_id", "customer_id", "premium", "status", "effective_date"],
    "transactions": ["transaction_id", "account_id", "amount", "transaction_date", "merchant"],
    "accounts": ["account_id", "customer_id", "balance", "status", "opened_date"],
}

sql_tasks = [
    "select_all", "select_cols", "where", "count", "sum", "avg", "group_count",
    "order", "limit", "date_filter", "join_customer_orders", "join_policy_claims",
    "having", "case_when", "cte", "window_rank"
]

for _ in range(8000):
    task = random.choice(sql_tasks)
    table = random.choice(list(schemas.keys()))
    cols = schemas[table]
    col = random.choice(cols)

    if task == "select_all":
        prompt = f"Write SQL to select all columns from {table}. Return SQL only."
        resp = f"SELECT *\nFROM {table};"

    elif task == "select_cols":
        c1, c2 = random.sample(cols, 2)
        prompt = f"Write SQL to select {c1} and {c2} from {table}. Return SQL only."
        resp = f"SELECT {c1}, {c2}\nFROM {table};"

    elif task == "where":
        val = random.choice(["'active'", "'closed'", "'pending'", "'TX'", "100", "500"])
        prompt = f"Write SQL to select all rows from {table} where {col} = {val}. Return SQL only."
        resp = f"SELECT *\nFROM {table}\nWHERE {col} = {val};"

    elif task == "count":
        prompt = f"Write SQL to count rows in {table}. Return SQL only."
        resp = f"SELECT COUNT(*) AS row_count\nFROM {table};"

    elif task == "sum":
        num = random.choice(["amount", "salary", "price", "inventory", "claim_amount", "premium", "balance", "age"])
        prompt = f"Write SQL to calculate total {num} from {table}. Return SQL only."
        resp = f"SELECT SUM({num}) AS total_{num}\nFROM {table};"

    elif task == "avg":
        num = random.choice(["amount", "salary", "price", "inventory", "claim_amount", "premium", "balance", "age"])
        prompt = f"Write SQL to calculate average {num} from {table}. Return SQL only."
        resp = f"SELECT AVG({num}) AS avg_{num}\nFROM {table};"

    elif task == "group_count":
        prompt = f"Write SQL to count rows in {table} grouped by {col}. Return SQL only."
        resp = f"SELECT {col}, COUNT(*) AS row_count\nFROM {table}\nGROUP BY {col};"

    elif task == "order":
        prompt = f"Write SQL to select all rows from {table} ordered by {col}. Return SQL only."
        resp = f"SELECT *\nFROM {table}\nORDER BY {col};"

    elif task == "limit":
        n = random.choice([5, 10, 20, 50, 100])
        prompt = f"Write SQL to return the first {n} rows from {table}. Return SQL only."
        resp = f"SELECT *\nFROM {table}\nLIMIT {n};"

    elif task == "date_filter":
        year = random.choice([2022, 2023, 2024, 2025, 2026])
        dcol = random.choice(["order_date", "hire_date", "claim_date", "effective_date", "transaction_date", "opened_date", "signup_date"])
        prompt = f"Write SQL to select rows from {table} where {dcol} is on or after January 1, {year}. Return SQL only."
        resp = f"SELECT *\nFROM {table}\nWHERE {dcol} >= '{year}-01-01';"

    elif task == "join_customer_orders":
        prompt = "Write SQL to show each customer_id and their total order amount using customers and orders. Return SQL only."
        resp = "SELECT c.customer_id, SUM(o.amount) AS total_amount\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nGROUP BY c.customer_id;"

    elif task == "join_policy_claims":
        prompt = "Write SQL to show each policy_id and total claim amount using policies and claims. Return SQL only."
        resp = "SELECT p.policy_id, SUM(c.claim_amount) AS total_claim_amount\nFROM policies p\nJOIN claims c ON p.policy_id = c.policy_id\nGROUP BY p.policy_id;"

    elif task == "having":
        threshold = random.choice([100, 500, 1000, 5000])
        prompt = f"Write SQL to find customer_id values from orders with total amount greater than {threshold}. Return SQL only."
        resp = f"SELECT customer_id, SUM(amount) AS total_amount\nFROM orders\nGROUP BY customer_id\nHAVING SUM(amount) > {threshold};"

    elif task == "case_when":
        prompt = "Write SQL to classify orders as high_value when amount >= 1000 and normal otherwise. Return SQL only."
        resp = "SELECT order_id,\n       CASE WHEN amount >= 1000 THEN 'high_value' ELSE 'normal' END AS order_type\nFROM orders;"

    elif task == "cte":
        prompt = "Write SQL using a CTE to calculate total order amount by customer_id. Return SQL only."
        resp = "WITH customer_totals AS (\n    SELECT customer_id, SUM(amount) AS total_amount\n    FROM orders\n    GROUP BY customer_id\n)\nSELECT *\nFROM customer_totals;"

    else:
        prompt = "Write SQL to rank orders by amount within each customer_id. Return SQL only."
        resp = "SELECT order_id,\n       customer_id,\n       amount,\n       RANK() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS amount_rank\nFROM orders;"

    sql_rows.append(chat_text(prompt, resp))

write_jsonl("data/sql_train.jsonl", sql_rows)

# ============================================================
# NOT ENOUGH INFO / GROUNDEDNESS DATA
# ============================================================

nei_rows = []

context_bank = [
    ("The museum is open Tuesday through Sunday from 10 AM to 5 PM. It is closed on Mondays.",
     "Is the museum open on Monday?", "No. The context says the museum is closed on Mondays.",
     "How much does admission cost?", "I do not have enough information. The context does not mention admission cost."),
    ("The warranty lasts 12 months from purchase and covers manufacturing defects.",
     "How long does the warranty last?", "The warranty lasts 12 months from purchase.",
     "Does it cover water damage?", "I do not have enough information. The context does not say whether water damage is covered."),
    ("The flight departs at 7:20 AM and boarding starts at 6:45 AM.",
     "When does boarding start?", "Boarding starts at 6:45 AM.",
     "What gate does the flight leave from?", "I do not have enough information. The context does not mention the gate."),
    ("The hotel check-in time is 3 PM and checkout is 11 AM.",
     "What time is checkout?", "Checkout is at 11 AM.",
     "Is breakfast included?", "I do not have enough information. The context does not mention breakfast."),
    ("The policy premium is $120 per month and the deductible is $500.",
     "What is the deductible?", "The deductible is $500.",
     "Does the policy include dental coverage?", "I do not have enough information. The context does not mention dental coverage."),
]

for _ in range(8000):
    context, q1, a1, q2, a2 = random.choice(context_bank)
    if random.random() < 0.45:
        question, answer = q1, a1
    else:
        question, answer = q2, a2

    prompt = random.choice([
        f"Answer using only the context.\n\nContext: {context}\n\nQuestion: {question}",
        f"Use only the information below. If not stated, say you do not have enough information.\n\nContext: {context}\n\nQuestion: {question}",
        f"Based only on this context, answer the question.\n\nContext: {context}\n\nQuestion: {question}",
    ])
    nei_rows.append(chat_text(prompt, answer))

write_jsonl("data/not_enough_info_train.jsonl", nei_rows)

# ============================================================
# CONCISE EXPLANATION DATA
# ============================================================

concise_rows = []

topics = {
    "photosynthesis": "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make sugar and oxygen.",
    "inflation": "Inflation is when prices rise over time, reducing the buying power of money.",
    "interest rates": "Interest rates are the cost of borrowing money or the reward for saving it.",
    "machine learning": "Machine learning is when computers learn patterns from data to make predictions or decisions.",
    "random forests": "A random forest combines many decision trees to make more stable predictions.",
    "PCA": "PCA reduces many variables into fewer dimensions while keeping as much variation as possible.",
    "LASSO": "LASSO is a regression method that can shrink some coefficients to zero, making the model simpler.",
    "SQL": "SQL is a language used to query and manage data in relational databases.",
    "transformers": "Transformers are neural networks that use attention to decide which parts of input matter most.",
    "LoRA": "LoRA fine-tunes a model by training small low-rank weight updates instead of changing all weights.",
    "overfitting": "Overfitting happens when a model learns training examples too specifically and performs worse on new data.",
    "gradient descent": "Gradient descent improves a model by repeatedly adjusting weights to reduce error.",
    "embeddings": "Embeddings turn text into vectors that capture similarity or meaning.",
    "recursion": "Recursion is when a function solves a problem by calling itself on smaller versions of the problem.",
    "regularization": "Regularization discourages overly complex models so they generalize better to new data.",
}

styles = [
    "Explain {topic} in one concise paragraph.",
    "Give a simple one-paragraph explanation of {topic}.",
    "Explain {topic} clearly in 3 sentences or fewer.",
    "What is {topic}? Keep the answer brief.",
    "Define {topic} in plain English.",
    "Explain {topic} without jargon.",
]

for _ in range(5000):
    topic, explanation = random.choice(list(topics.items()))
    prompt = random.choice(styles).format(topic=topic)
    concise_rows.append(chat_text(prompt, explanation))

write_jsonl("data/concise_explain_train.jsonl", concise_rows)

print("Wrote:")
print(f"data/harder_math_train.jsonl: {len(math_rows)}")
print(f"data/code_train.jsonl: {len(code_rows)}")
print(f"data/sql_train.jsonl: {len(sql_rows)}")
print(f"data/not_enough_info_train.jsonl: {len(nei_rows)}")
print(f"data/concise_explain_train.jsonl: {len(concise_rows)}")