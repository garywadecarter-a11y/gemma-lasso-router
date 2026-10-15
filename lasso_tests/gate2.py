"""Gate 2: do lasso-selected, blended LoRA adapters give better Gemma 3 1B
answers than Gary's single-route router?

Conditions (per prompt):
  base            no adapter
  router_top1     Gary's L1 logistic router, one adapter (current system)
  sim_top2        two most similar route keys, adapters summed at weight 1 each
  lasso_set       lasso-selected routes (alpha=5e-4), summed at weight 1 each
  lasso_weighted  lasso-selected routes, weights = lasso coef / max coef
  oracle_set      the true routes (cheat / ceiling), weight 1 each
"base" in a route set means "no adapter for that part".
Blends use PEFT add_weighted_adapter(combination_type="cat"), i.e. the LoRA
deltas are added exactly: dW = sum_i w_i * B_i A_i.

Grading is automatic: SQL is executed in SQLite against a toy orders table and
compared with a reference query; Python is executed against unit tests; numeric
and JSON answers are compared within tolerance; "not enough info" answers are
matched by phrase.  Results stream to gate2_results.jsonl (resumable).
"""
import json, os, re, sys, time, sqlite3, subprocess, tempfile, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, Lasso
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gate1_prompts as GP, gate1 as G

torch.set_num_threads(2)
GEMMA = "/home/claude/models/gemma-3-1b-it"
ADIR = "/mnt/user-data/uploads/gemma-lasso-router/adapters"
OUT = os.environ.get("GATE2_OUT", "gate2_results.jsonl")
ALPHA = 5e-4
MAXTOK = 140
CONDS = os.environ.get("GATE2_CONDS", "base,router_top1,sim_top2,lasso_set,lasso_weighted,oracle_set").split(",")

# ---------------- prompts ----------------
items = [it for it in GP.mixed_prompts() if it["check"]] + \
        [it for it in GP.single_prompts() if it["check"]]

# ---------------- routing ----------------
emb = SentenceTransformer(G.MINILM, device="cpu")
tp, ty = G.router_training_data()
Xtr = emb.encode(tp, batch_size=64, normalize_embeddings=True); ty = np.array(ty)
routes = sorted(set(ty))
keys = G.unit(np.stack([Xtr[ty == r].mean(0) for r in routes]))
lr = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=5000, random_state=42).fit(Xtr, ty)
X = emb.encode([it["prompt"] for it in items], batch_size=64, normalize_embeddings=True)


def plan(i, cond):
    """Return {route: weight} for prompt i under a condition."""
    x = X[i]; s = x @ keys.T
    if cond == "base":
        return {}
    if cond == "router_top1":
        return {lr.predict(x[None])[0]: 1.0}
    if cond == "sim_top2":
        return {routes[j]: 1.0 for j in np.argsort(-s)[:2]}
    if cond in ("lasso_set", "lasso_weighted"):
        w = Lasso(alpha=ALPHA, positive=True, fit_intercept=False, max_iter=20000).fit(keys.T, x).coef_
        nz = np.flatnonzero(w > 1e-6)
        if len(nz) == 0:
            nz = [int(np.argmax(s))]; w = np.ones(len(routes))
        if cond == "lasso_set":
            return {routes[j]: 1.0 for j in nz}
        return {routes[j]: float(w[j] / w[nz].max()) for j in nz}
    if cond == "oracle_set":
        return {r: 1.0 for r in items[i]["routes"]}
    # "average" variants: same sets, weights rescaled to sum to 1
    if cond in ("sim_top2_avg", "lasso_avg", "oracle_avg"):
        src = {"sim_top2_avg": "sim_top2", "lasso_avg": "lasso_weighted", "oracle_avg": "oracle_set"}[cond]
        p = {r: w for r, w in plan(i, src).items() if r != "base"}
        if not p:
            return {}
        tot = sum(p.values())
        return {r: w / tot for r, w in p.items()}


# ---------------- model ----------------
tok = AutoTokenizer.from_pretrained(GEMMA)
base = AutoModelForCausalLM.from_pretrained(GEMMA, dtype=torch.float32)
ADS = ["code", "concise_explain", "harder_math", "math", "not_enough_info", "sql"]
model = PeftModel.from_pretrained(base, f"{ADIR}/{ADS[0]}", adapter_name=ADS[0])
for a in ADS[1:]:
    model.load_adapter(f"{ADIR}/{a}", adapter_name=a)
model.eval()
_blend_cache = {}


def activate(p):
    """p: {route: weight}. 'base' entries mean no adapter for that part."""
    p = {r: w for r, w in p.items() if r != "base" and w > 0}
    if not p:
        return None
    if len(p) == 1 and abs(list(p.values())[0] - 1) < 1e-9:
        model.set_adapter(list(p)[0]); return list(p)[0]
    key = tuple(sorted((r, round(w, 3)) for r, w in p.items()))
    if key not in _blend_cache:
        name = "blend%d" % len(_blend_cache)
        model.add_weighted_adapter(list(dict(key)), [w for _, w in key], name, combination_type="cat")
        _blend_cache[key] = name
    model.set_adapter(_blend_cache[key]); return _blend_cache[key]


def generate(prompt, p):
    chat = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    ids = tok(chat, return_tensors="pt")
    name = activate(p)
    ctx = model.disable_adapter() if name is None else torch.no_grad()
    with ctx, torch.no_grad():
        o = model.generate(**ids, max_new_tokens=MAXTOK, do_sample=False, repetition_penalty=1.1,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------- grading ----------------
ORDERS = [(1, 101, 120.0), (2, 101, 80.0), (3, 102, 250.0), (4, 103, 40.0), (5, 102, 310.0), (6, 104, 95.5)]


def db():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE orders(order_id INTEGER, customer_id INTEGER, amount REAL, quantity INTEGER, price REAL, status TEXT, order_date TEXT)")
    c.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
                  [(a, b, m, 2, m / 2, "active", "2026-01-0%d" % a) for a, b, m in ORDERS])
    return c


def norm_rows(rows):
    out = []
    for r in rows:
        out.append(tuple(round(float(v), 2) if isinstance(v, (int, float)) and v is not None else v for v in r))
    return sorted(out, key=lambda t: tuple(str(v) for v in t))


def extract_sql(ans):
    m = re.search(r"```(?:sql)?\s*(.*?)```", ans, re.S | re.I)
    s = m.group(1) if m else ans
    m2 = re.search(r"((?:WITH|SELECT)\b.*?)(;|$)", s, re.S | re.I)
    return m2.group(1).strip() if m2 else s.strip()


def grade_sql(ans, chk):
    exp = norm_rows(db().execute(chk["query_expect"]).fetchall())
    try:
        got = norm_rows(db().execute(extract_sql(ans)).fetchall())
    except Exception as e:
        return False, "sql_error: %s" % str(e)[:80]
    # compare on numeric columns only (column order/naming may differ)
    def nums(rows):
        return sorted(tuple(sorted(v for v in r if isinstance(v, float))) for r in rows)
    ok = got == exp or nums(got) == nums(exp)
    return ok, "rows_match" if ok else "rows_differ"


def extract_code(ans):
    m = re.search(r"```(?:python)?\s*(.*?)```", ans, re.S | re.I)
    if m:
        return m.group(1)
    i = ans.find("def ")
    return ans[i:] if i >= 0 else ""


def grade_code(ans, chk):
    code = extract_code(ans)
    if "def " not in code:
        return False, "no_function"
    harness = code + "\n\nimport json, math\n_f = globals().get(%r)\n" % chk["fn"] + \
        "if _f is None:\n    import types\n    _c=[v for k,v in list(globals().items()) if isinstance(v, types.FunctionType) and not k.startswith('_')]\n    _f=_c[0] if _c else None\n" + \
        "_ok=True\nfor a, e in %r:\n    try:\n        r=_f(*a)\n        _ok = _ok and abs(float(r)-float(e)) <= 1e-6 + 1e-4*abs(float(e))\n    except Exception as ex:\n        _ok=False\nprint('RESULT', _ok)\n" % (chk["tests"],)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness); path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=10)
        ok = "RESULT True" in r.stdout
        return ok, "tests_pass" if ok else ("tests_fail" if "RESULT" in r.stdout else "code_error")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def first_number(t):
    t = t.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", t)
    return float(m[0]) if m else None


def grade_num(ans, chk):
    v = first_number(ans)
    ok = v is not None and abs(v - chk["expect"]) <= 0.01 + 1e-3 * abs(chk["expect"])
    return ok, "num=%s" % v


NEI = ["not enough information", "don't have enough", "do not have enough", "not provided", "not mentioned",
       "cannot be determined", "can't be determined", "not stated", "doesn't say", "does not say", "no information",
       "cannot determine", "can't determine", "not specified", "doesn't mention", "does not mention", "unable to", "doesn't provide", "does not provide", "not given", "isn't provided", "is not provided", "doesn't give", "does not give", "no mention", "doesn't state", "does not state", "doesn't include", "does not include"]


def grade_nei(ans, chk):
    a = ans.lower().replace("\u2019", "'")
    ok = any(p in a for p in NEI)
    return ok, "nei" if ok else "answered"


def grade_json(ans, chk):
    m = re.search(r"\{.*\}", ans, re.S)
    if not m:
        return False, "no_json"
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return False, "bad_json"
    v = obj.get(chk["key"])
    try:
        ok = abs(float(v) - chk["expect"]) <= 0.01 + 1e-3 * abs(chk["expect"])
    except Exception:
        return False, "missing_key"
    return ok, "val=%s" % v


GRADERS = {"sql": grade_sql, "code": grade_code, "num": grade_num, "nei": grade_nei, "json": grade_json}

# ---------------- run ----------------
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l); done.add((r["i"], r["cond"]))
t0 = time.time(); n = 0
with open(OUT, "a") as f:
    for i, it in enumerate(items):
        for cond in CONDS:
            if (i, cond) in done:
                continue
            p = plan(i, cond)
            ans = generate(it["prompt"], p)
            ok, note = GRADERS[it["check"]["type"]](ans, it["check"])
            rec = {"i": i, "cond": cond, "kind": it["kind"], "pair": "+".join(it["routes"]),
                   "check": it["check"]["type"], "plan": p, "correct": bool(ok), "note": note,
                   "prompt": it["prompt"], "answer": ans}
            f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
            if n % 10 == 0:
                print(f"{n} done, {time.time() - t0:.0f}s elapsed", flush=True)
print("FINISHED", flush=True)
