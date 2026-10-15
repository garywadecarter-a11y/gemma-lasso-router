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

# Math routing examples
for _ in range(1000):
    a = random.randint(2, 999)
    b = random.randint(2, 999)
    template = random.choice([
        "What is {a} * {b}?",
        "Calculate {a} times {b}.",
        "Give only the answer: {a} multiplied by {b}.",
        "Solve this arithmetic problem: {a} x {b}.",
    ])
    prompts.append(template.format(a=a, b=b))
    labels.append("math")

# JSON routing examples
names = ["Alice", "Bob", "Carlos", "Dina", "Evan", "Fatima"]
cities = ["Dallas", "Chicago", "Boston", "Seattle", "Austin", "Denver"]

for _ in range(1000):
    name = random.choice(names)
    age = random.randint(18, 80)
    city = random.choice(cities)
    template = random.choice([
        "Return valid JSON with name={name}, age={age}, city={city}.",
        "Create a JSON object containing name, age, and city for {name}.",
        "Output JSON only: name {name}, age {age}, city {city}.",
        "Format this as JSON: {name}, {age}, {city}.",
    ])
    prompts.append(template.format(name=name, age=age, city=city))
    labels.append("json")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = embedder.encode(prompts, show_progress_bar=True)
y = np.array(labels)

X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
    X, y, prompts, test_size=0.2, random_state=42, stratify=y
)

# This is the LASSO-like part:
# Logistic regression with L1 penalty creates sparse coefficients.
router = LogisticRegression(
    penalty="l1",
    solver="saga",
    C=0.5,
    max_iter=3000,
    random_state=42,
)

router.fit(X_train, y_train)

pred = router.predict(X_test)
print(classification_report(y_test, pred))

with open("router/embedder_name.txt", "w", encoding="utf-8") as f:
    f.write("sentence-transformers/all-MiniLM-L6-v2")

with open("router/l1_router.pkl", "wb") as f:
    pickle.dump(router, f)

print("Saved router to router/l1_router.pkl")