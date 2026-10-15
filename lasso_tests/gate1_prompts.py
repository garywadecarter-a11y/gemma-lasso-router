"""Test prompts for Gate 1 / Gate 2.

MIXED prompts need two of Gary's routes. Ground-truth routes are a human
judgment (mine), not a measured fact. SINGLE prompts need one route and are
deliberately worded differently from the router's training templates.
Each prompt may carry a `check` spec so Gate 2 can grade answers automatically.
"""
import random

ROUTES = ["base", "code", "concise_explain", "harder_math", "math", "not_enough_info", "sql"]
NAMES = ["Gary", "Alice", "Priya", "Marcus", "Lena", "Omar", "Sofia", "Jin"]
CITIES = ["Dallas", "Austin", "Denver", "Boston", "Seattle", "Phoenix", "Atlanta"]
TOPICS = ["recursion", "hash maps", "list comprehensions", "sorting", "binary search", "string slicing"]
SQLTOPICS = [("a CTE", "using a CTE to total order amount by customer_id"),
             ("window functions", "to rank orders by amount within each customer_id"),
             ("GROUP BY", "to count claims by status"),
             ("a LEFT JOIN", "to list every customer and their orders, including customers with no orders")]
TABLES = ["orders", "claims", "policies", "customers", "products"]


def fn():
    return random.choice(["calc", "compute", "solve", "get", "make"]) + "_" + random.choice(
        ["total", "price", "value", "result", "amount"]) + str(random.randint(1, 99))


def mixed_prompts(n_per=15, seed=7):
    random.seed(seed)
    out = []

    def add(p, routes, check=None):
        out.append({"prompt": p, "routes": sorted(routes), "kind": "mixed", "check": check})

    for _ in range(n_per):  # sql + harder_math
        p = random.choice([10, 15, 20, 25, 30])
        n = random.randint(50, 500)
        t = random.choice([
            (f"Write SQL that returns order_id and the amount after a {p}% discount from the orders table. Return SQL only.",
             {"type": "sql", "query_expect": f"SELECT order_id, amount * (1 - {p}/100.0) FROM orders ORDER BY order_id"}),
            (f"Write a SQL query that returns customer_id and {p} percent of each customer's total order amount from the orders table. Return SQL only.",
             {"type": "sql", "query_expect": f"SELECT customer_id, SUM(amount) * {p}/100.0 FROM orders GROUP BY customer_id ORDER BY customer_id"}),
            (f"Using SQL, list the order_id of orders whose amount after a {p}% markup is greater than {n}. Return SQL only.",
             {"type": "sql", "query_expect": f"SELECT order_id FROM orders WHERE amount * (1 + {p}/100.0) > {n} ORDER BY order_id"}),
            (f"Write SQL to compute the average amount in the orders table and then reduce it by {p}%. Return SQL only.",
             {"type": "sql", "query_expect": f"SELECT AVG(amount) * (1 - {p}/100.0) FROM orders"}),
        ])
        add(t[0], ["sql", "harder_math"], t[1])

    for _ in range(n_per):  # code + harder_math
        f = fn(); p = random.choice([10, 15, 20, 25, 40])
        t = random.choice([
            (f"Write a Python function named {f}(price) that returns the final price after a {p}% discount. Return code only.",
             {"type": "code", "fn": f, "tests": [([100], 100 * (1 - p / 100)), ([250], 250 * (1 - p / 100))]}),
            (f"Write a Python function named {f}(nums) that returns {p}% of the average of the list nums. Return code only.",
             {"type": "code", "fn": f, "tests": [([[10, 20, 30]], 20 * p / 100), ([[5, 15]], 10 * p / 100)]}),
            (f"Write a Python function named {f}(rate, hours) that returns the total units produced by a machine making rate units per hour for hours hours. Return code only.",
             {"type": "code", "fn": f, "tests": [([12, 5], 60), ([7, 3], 21)]}),
            (f"Write a Python function named {f}(a, b, c, d) that returns (a + b) * c - d. Return code only.",
             {"type": "code", "fn": f, "tests": [([1, 2, 3, 4], 5), ([10, 5, 2, 1], 29)]}),
        ])
        add(t[0], ["code", "harder_math"], t[1])

    for _ in range(n_per):  # not_enough_info + harder_math
        x = random.randint(80, 300); p = random.choice([10, 20, 25]); k = random.randint(2, 6)
        t = random.choice([
            (f"Answer using only the context.\n\nContext: The policy premium is ${x} per month and the deductible is $500.\n\nQuestion: What is the yearly premium after a {p}% increase? Give only the number.",
             {"type": "num", "expect": round(x * 12 * (1 + p / 100), 2)}),
            (f"Use only the information below. If not stated, say you do not have enough information.\n\nContext: The hotel charges ${x} per night.\n\nQuestion: What is the cost of {k} nights with a {p}% discount? Give only the number.",
             {"type": "num", "expect": round(x * k * (1 - p / 100), 2)}),
            (f"Based only on this context, answer the question.\n\nContext: The flight departs at 7:20 AM and boarding starts at 6:45 AM.\n\nQuestion: How many miles will the plane travel at {x} miles per hour?",
             {"type": "nei"}),
            (f"Answer using only the context.\n\nContext: The warranty lasts 12 months and covers manufacturing defects.\n\nQuestion: What is {p}% of the repair cost?",
             {"type": "nei"}),
        ])
        add(t[0], ["not_enough_info", "harder_math"], t[1])

    for _ in range(n_per):  # base(json) + harder_math
        nm = random.choice(NAMES); c = random.choice(CITIES); pr = random.randint(40, 900); p = random.choice([10, 20, 25])
        nums = [random.randint(5, 100) for _ in range(3)]
        t = random.choice([
            (f"Return valid JSON only with name={nm} and discounted_price equal to {pr} after a {p}% discount.",
             {"type": "json", "key": "discounted_price", "expect": round(pr * (1 - p / 100), 2)}),
            (f"Output JSON only with fields city={c} and average equal to the mean of {nums[0]}, {nums[1]}, {nums[2]}.",
             {"type": "json", "key": "average", "expect": round(sum(nums) / 3, 4)}),
        ])
        add(t[0], ["base", "harder_math"], t[1])

    for _ in range(n_per):  # code + concise_explain
        f = fn(); tp = random.choice(TOPICS)
        add(random.choice([
            f"Write a Python function named {f} that reverses a string, then explain in one sentence how it works.",
            f"Explain {tp} briefly and write a tiny Python function named {f} that demonstrates it.",
            f"Write a Python function named {f} that removes duplicates from a list, and give a short plain-English explanation.",
        ]), ["code", "concise_explain"])

    for _ in range(n_per):  # sql + concise_explain
        a, b = random.choice(SQLTOPICS)
        add(random.choice([
            f"Explain {a} in plain English, then write SQL {b}.",
            f"Briefly explain {a} and write SQL {b}.",
            f"In two sentences, explain what {a} does in SQL, then show SQL {b}.",
        ]), ["sql", "concise_explain"])

    for _ in range(n_per):  # code + sql
        f = fn(); tb = random.choice(TABLES)
        add(random.choice([
            f"Write a Python function named {f} that uses sqlite3 to run a SQL query counting rows in {tb}. Return code only.",
            f"Write a Python function named {f} that returns a SQL query string selecting all rows from {tb} where status = 'active'.",
            f"Write Python code with a function named {f} that executes SQL to sum amount by customer_id from {tb}.",
        ]), ["code", "sql"])

    for _ in range(n_per):  # not_enough_info + concise_explain
        add(random.choice([
            "Using only the context, briefly explain whether the warranty covers water damage.\n\nContext: The warranty lasts 12 months from purchase and covers manufacturing defects.",
            "Based only on this context, give a short plain-English explanation of when the museum is open.\n\nContext: The museum is open Tuesday through Sunday from 10 AM to 5 PM. It is closed on Mondays.",
            "Using only the context, explain simply whether dental is covered.\n\nContext: The policy premium is $120 per month and the deductible is $500.",
        ]), ["not_enough_info", "concise_explain"])

    for _ in range(n_per):  # math + code
        f = fn(); a, b = random.randint(11, 99), random.randint(11, 99)
        add(f"Write a Python function named {f} that multiplies two numbers, and also tell me what {a} * {b} is.",
            ["math", "code"])

    for _ in range(n_per):  # math + sql
        a, b = random.randint(11, 99), random.randint(11, 99)
        add(f"Write SQL that multiplies quantity by price for each row in orders, and also compute {a} times {b}.",
            ["math", "sql"])

    for _ in range(n_per):  # base(json) + code
        f = fn(); nm = random.choice(NAMES)
        add(f"Write a Python function named {f} that returns a JSON-style dict with name={nm}, age, and city. Return code only.",
            ["base", "code"])
    return out


def single_prompts(n_per=10, seed=11):
    random.seed(seed)
    out = []

    def add(p, r, check=None):
        out.append({"prompt": p, "routes": [r], "kind": "single", "check": check})

    for _ in range(n_per):
        a, b = random.randint(12, 999), random.randint(12, 999)
        add(random.choice([f"Quick one: {a} x {b} = ?", f"Multiply {a} by {b} and give just the result.",
                           f"I need the product of {a} and {b}."]), "math", {"type": "num", "expect": a * b})
    for _ in range(n_per):
        x = random.randint(100, 2000); p = random.choice([12, 18, 35])
        if random.random() < 0.5:
            add(f"My bill is {x} dollars and I tip {p} percent. How much is the tip? Number only.",
                "harder_math", {"type": "num", "expect": round(x * p / 100, 2)})
        else:
            add(f"Take {p} percent off {x}. What's left? Number only.",
                "harder_math", {"type": "num", "expect": round(x * (1 - p / 100), 2)})
    for _ in range(n_per):
        f = fn()
        add(random.choice([f"Code a Python function {f}(xs) that gives back the smallest item in xs.",
                           f"I need Python: a function {f}(s) that counts uppercase letters in s."]), "code")
    for _ in range(n_per):
        tb = random.choice(TABLES)
        add(random.choice([f"Give me a query that pulls the 10 most recent rows from {tb} by created_at.",
                           f"Query to find duplicate customer_id values in {tb}?"]), "sql")
    for _ in range(n_per):
        add(random.choice(["Context: The gym opens at 6 AM on weekdays.\n\nQuestion: Is there a pool? Only use the context.",
                           "Context: Shipping takes 3-5 business days.\n\nQuestion: What carrier is used? Only use what's given."]),
            "not_enough_info")
    for _ in range(n_per):
        add(random.choice(["In plain words, what's a neural network?", "Short version: what does a credit score measure?",
                           "Keep it brief: what is compound interest?"]), "concise_explain")
    for _ in range(n_per):
        add(random.choice([f"Give me a JSON object for {random.choice(NAMES)} who lives in {random.choice(CITIES)}.",
                           "Write a friendly two-line welcome message for new coworkers."]), "base")
    return out
