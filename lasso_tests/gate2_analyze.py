import json, collections, numpy as np

import os, sys
_f = sys.argv[1] if len(sys.argv) > 1 else "gate2_results.jsonl"
rows = [json.loads(l) for l in open(_f)]
if len(sys.argv) > 2:   # merge a second results file (extra conditions)
    rows += [json.loads(l) for l in open(sys.argv[2])]
_src = open("gate2.py").read()
_g = _src[_src.index("NEI = ["):_src.index("def grade_json")]
_ns = {}; exec(_g, _ns)
import re as _re
import gate1_prompts as _GP
_items = [it for it in _GP.mixed_prompts() if it["check"]] + [it for it in _GP.single_prompts() if it["check"]]
for r in rows:
    if r["check"] == "nei":
        r["correct"] = bool(_ns["grade_nei"](r["answer"], {})[0])
    if r["check"] == "num":   # lenient: expected value appears anywhere in the answer
        e = _items[r["i"]]["check"]["expect"]
        vals = [float(v) for v in _re.findall(r"-?\d+(?:\.\d+)?", r["answer"].replace(",", ""))]
        r["correct"] = any(abs(v - e) <= 0.01 + 1e-3 * abs(e) for v in vals)
CONDS = ["base", "router_top1", "sim_top2", "lasso_set", "lasso_weighted", "oracle_set"]
if len(sys.argv) > 2:
    CONDS += ["sim_top2_avg", "lasso_avg", "oracle_avg"]
by = collections.defaultdict(dict)
for r in rows:
    by[r["i"]][r["cond"]] = r
complete = [i for i, d in by.items() if all(c in d for c in CONDS)]
rng = np.random.default_rng(0)


def acc(cond, idx):
    return np.mean([by[i][cond]["correct"] for i in idx]) if idx else float("nan")


def boot_diff(a, b, idx, B=5000):
    """paired bootstrap 95% CI for acc(a) - acc(b)"""
    x = np.array([by[i][a]["correct"] for i in idx], float) - np.array([by[i][b]["correct"] for i in idx], float)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


groups = {"all": complete,
          "mixed": [i for i in complete if by[i]["base"]["kind"] == "mixed"],
          "single": [i for i in complete if by[i]["base"]["kind"] == "single"]}
for pr in sorted({by[i]["base"]["pair"] for i in complete}):
    groups["  " + pr] = [i for i in complete if by[i]["base"]["pair"] == pr]

print(f"complete prompts: {len(complete)} of 80\n")
print("%-34s %4s " % ("group", "n") + " ".join("%14s" % c for c in CONDS))
for g, idx in groups.items():
    print("%-34s %4d " % (g, len(idx)) + " ".join("%13.1f%%" % (100 * acc(c, idx)) for c in CONDS))

print("\nPaired differences (all prompts), mean and 95% bootstrap CI, percentage points:")
for a, b in [("lasso_set", "router_top1"), ("lasso_weighted", "router_top1"), ("lasso_set", "sim_top2"),
             ("lasso_set", "base"), ("oracle_set", "router_top1"), ("router_top1", "base"), ("sim_top2", "base")]:
    for g in ["all", "mixed", "single"]:
        m, lo, hi = boot_diff(a, b, groups[g])
        print("  %-15s - %-12s [%-6s] %+6.1f  (%+.1f, %+.1f)" % (a, b, g, 100 * m, 100 * lo, 100 * hi))

print("\nAvg adapters used:")
for c in CONDS:
    print("  %-15s %.2f" % (c, np.mean([len([k for k in by[i][c]["plan"] if k != "base"]) for i in complete])))
