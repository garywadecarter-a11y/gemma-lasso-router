import random
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

random.seed(42)
Path("router").mkdir(exist_ok=True)

prompts = []
labels = []

def add(prompt, label):
    prompts.append(prompt)
    labels.append(label)

# ============================================================
# BASE ROUTE
# Things base Gemma may already handle well, especially JSON/simple chat
# ============================================================

names = ["Gary", "Alice", "Bob", "Dina", "Evan", "Fatima", "Mia", "Noah"]
cities = ["Dallas", "Austin", "Chicago", "Seattle", "Denver", "Boston"]

for _ in range(800):
    name = random.choice(names)
    age = random.randint(18, 80)
    city = random.choice(cities)

    template = random.choice([
        "Return valid JSON only with name={name}, age={age}, city={city}.",
        "Create a JSON object containing name, age, and city for {name}.",
        "Output JSON only: name {name}, age {age}, city {city}.",
        "Format this as JSON: {name}, {age}, {city}.",
    ])

    add(template.format(name=name, age=age, city=city), "base")

for _ in range(400):
    add(random.choice([
        "Say hello in a friendly way.",
        "Write a short thank-you message.",
        "What is the capital of France?",
        "Give me three colors.",
        "List three common fruits.",
        "Rewrite this sentence politely: send me the file now.",
        "Translate hello to Spanish.",
    ]), "base")


# ============================================================
# MATH ROUTE
# Simple arithmetic
# ============================================================

for _ in range(1000):
    a = random.randint(2, 999)
    b = random.randint(2, 999)

    template = random.choice([
        "What is {a} * {b}?",
        "Calculate {a} times {b}.",
        "Give only the answer: {a} multiplied by {b}.",
        "Solve this arithmetic problem: {a} x {b}.",
    ])

    add(template.format(a=a, b=b), "math")


# ============================================================
# HARDER MATH ROUTE
# Word problems, percentages, multi-step
# ============================================================

for _ in range(1000):
    kind = random.choice(["percent", "discount", "average", "rate", "multi"])

    if kind == "percent":
        pct = random.choice([5, 10, 15, 20, 25, 30, 35])
        base = random.randint(50, 5000)
        add(random.choice([
            f"What is {pct}% of {base}?",
            f"Calculate {pct} percent of {base}.",
            f"A quantity is {base}. Find {pct}% of it.",
        ]), "harder_math")

    elif kind == "discount":
        price = random.randint(20, 3000)
        pct = random.choice([5, 10, 15, 20, 25, 30, 40])
        add(random.choice([
            f"An item costs {price} dollars and is discounted by {pct}%. What is the final price?",
            f"Calculate the sale price of {price} after a {pct}% discount.",
        ]), "harder_math")

    elif kind == "average":
        nums = [random.randint(5, 300) for _ in range(random.choice([3, 4, 5]))]
        joined = ", ".join(map(str, nums))
        add(random.choice([
            f"What is the average of {joined}?",
            f"Find the mean of these numbers: {joined}.",
        ]), "harder_math")

    elif kind == "rate":
        rate = random.randint(5, 200)
        time = random.randint(2, 100)
        add(random.choice([
            f"A machine produces {rate} units per hour for {time} hours. How many units are produced?",
            f"At {rate} miles per hour for {time} hours, how many miles are traveled?",
        ]), "harder_math")

    else:
        a, b, c, d = random.randint(2, 100), random.randint(2, 100), random.randint(2, 20), random.randint(1, 100)
        add(random.choice([
            f"What is ({a} + {b}) * {c} - {d}?",
            f"Add {a} and {b}, multiply by {c}, then subtract {d}.",
        ]), "harder_math")


# ============================================================
# CODE ROUTE
# Python function prompts
# ============================================================

for _ in range(1000):
    fname = f"func_{random.randint(1, 999999)}"

    add(random.choice([
        f"Write a Python function named {fname} that returns the sum of two inputs a and b. Return code only.",
        f"Write a Python function named {fname} that checks if n is even. Return code only.",
        f"Write a Python function named {fname} that reverses a string s. Return code only.",
        f"Write a Python function named {fname} that returns only positive numbers from nums. Return code only.",
        f"Write a Python function named {fname} that removes duplicates from a list while preserving order. Return code only.",
        f"Write a Python function named {fname} that counts vowels in a string. Return code only.",
        f"Write Python code for a function named {fname} that returns the largest number in a list.",
        f"Create a Python function named {fname} that returns True if a string is a palindrome.",
    ]), "code")


# ============================================================
# SQL ROUTE
# SQL generation prompts
# ============================================================

for _ in range(1000):
    table = random.choice(["customers", "orders", "employees", "products", "claims", "policies", "transactions"])
    col = random.choice(["customer_id", "amount", "status", "state", "salary", "claim_amount", "order_date"])

    add(random.choice([
        f"Write SQL to select all columns from {table}. Return SQL only.",
        f"Write SQL to count rows in {table}. Return SQL only.",
        f"Write SQL to select all rows from {table} where {col} = 'active'. Return SQL only.",
        f"Write SQL to calculate total amount from orders by customer_id. Return SQL only.",
        f"Write SQL to join customers and orders by customer_id. Return SQL only.",
        f"Write SQL using a CTE to calculate total order amount by customer_id. Return SQL only.",
        f"Write SQL to rank orders by amount within each customer_id. Return SQL only.",
        f"Write SQL to group {table} by {col} and count rows. Return SQL only.",
    ]), "sql")


# ============================================================
# NOT ENOUGH INFO ROUTE
# Context-grounded Q&A prompts
# ============================================================

for _ in range(1000):
    context = random.choice([
        "The museum is open Tuesday through Sunday from 10 AM to 5 PM. It is closed on Mondays.",
        "The warranty lasts 12 months from purchase and covers manufacturing defects.",
        "The flight departs at 7:20 AM and boarding starts at 6:45 AM.",
        "The hotel check-in time is 3 PM and checkout is 11 AM.",
        "The policy premium is $120 per month and the deductible is $500.",
    ])

    question = random.choice([
        "What information is missing?",
        "How much does admission cost?",
        "Does it cover water damage?",
        "What gate does the flight leave from?",
        "Is breakfast included?",
        "Does the policy include dental coverage?",
        "Answer this using only the provided context.",
    ])

    add(random.choice([
        f"Answer using only the context.\n\nContext: {context}\n\nQuestion: {question}",
        f"Use only the information below. If not stated, say you do not have enough information.\n\nContext: {context}\n\nQuestion: {question}",
        f"Based only on this context, answer the question.\n\nContext: {context}\n\nQuestion: {question}",
    ]), "not_enough_info")


# ============================================================
# CONCISE EXPLAIN ROUTE
# Brief/simple explanation prompts
# ============================================================

topics = [
    "photosynthesis", "inflation", "interest rates", "machine learning",
    "random forests", "PCA", "LASSO", "SQL", "transformers", "LoRA",
    "overfitting", "gradient descent", "embeddings", "regularization"
]

for _ in range(1000):
    topic = random.choice(topics)
    add(random.choice([
        f"Explain {topic} in one concise paragraph.",
        f"Give a simple one-paragraph explanation of {topic}.",
        f"Explain {topic} clearly in 3 sentences or fewer.",
        f"What is {topic}? Keep the answer brief.",
        f"Define {topic} in plain English.",
        f"Explain {topic} without jargon.",
    ]), "concise_explain")


print(f"Total router examples: {len(prompts)}")
print("Labels:", sorted(set(labels)))

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = embedder.encode(prompts, show_progress_bar=True)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

router = LogisticRegression(
    penalty="l1",
    solver="saga",
    C=0.5,
    max_iter=5000,
    random_state=42,
)

router.fit(X_train, y_train)

pred = router.predict(X_test)
print(classification_report(y_test, pred))

with open("router/multi_l1_router.pkl", "wb") as f:
    pickle.dump(router, f)

with open("router/multi_router_labels.txt", "w", encoding="utf-8") as f:
    for label in sorted(set(labels)):
        f.write(label + "\n")

print("Saved router to router/multi_l1_router.pkl")