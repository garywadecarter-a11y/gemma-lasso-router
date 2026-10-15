import json
import random
from pathlib import Path

random.seed(42)

Path("data").mkdir(exist_ok=True)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

math_rows = []

for _ in range(500):
    a = random.randint(2, 99)
    b = random.randint(2, 99)
    answer = a * b

    prompt = f"What is {a} * {b}? Give only the final answer."
    response = str(answer)

    math_rows.append({
        "text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
    })

json_rows = []

names = ["Alice", "Bob", "Carlos", "Dina", "Evan", "Fatima"]
cities = ["Dallas", "Chicago", "Boston", "Seattle", "Austin", "Denver"]

for _ in range(500):
    name = random.choice(names)
    age = random.randint(18, 80)
    city = random.choice(cities)

    prompt = (
        f"Return valid JSON only with these fields: name, age, city. "
        f"Use name={name}, age={age}, city={city}."
    )

    response = json.dumps({
        "name": name,
        "age": age,
        "city": city
    })

    json_rows.append({
        "text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
    })

write_jsonl("data/math_train.jsonl", math_rows)
write_jsonl("data/json_train.jsonl", json_rows)

print("Wrote data/math_train.jsonl and data/json_train.jsonl")