import json, collections, numpy as np, os


def load(f):
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def ms(v):
    return f"{np.mean(v):.3f}±{np.std(v):.3f}"


# ---------------- Test 1 ----------------
T1 = load("t1.jsonl")
if T1:
    print("TEST 1 — knowledge spread over two partial pieces (3 seeds)")
    for r in T1: r["ret"] = 0.5 * (r["old_acc"] + r["add_acc"])
    for lr in sorted({r["lr"] for r in T1}):
        print(f"\n write lr={lr}")
        g = collections.defaultdict(list)
        for r in T1:
            if r["lr"] == lr: g[(r["read"], r["param"])].append(r)
        for (rd, p), rs in g.items():
            print(f"  {rd:12s} {str(p):6s} seen {ms([x['new_seen'] for x in rs])}  held-out {ms([x['new_heldout'] for x in rs])}"
                  f"  retention {ms([x['ret'] for x in rs])}  n={len(rs)}")
    # best per switch with retention >= 0.99
    print("\n Best per switch (retention >= 0.99), by write lr:")
    for lr in sorted({r["lr"] for r in T1}):
        for rd in ["oracle_one", "oracle_both", "nearest1", "nearest_sum", "lasso"]:
            g = collections.defaultdict(list)
            for r in T1:
                if r["lr"] == lr and r["read"] == rd: g[r["param"]].append(r)
            ok = [(p, rs) for p, rs in g.items() if np.mean([x["ret"] for x in rs]) >= 0.99]
            if ok:
                p, rs = max(ok, key=lambda t: np.mean([x["new_heldout"] + 4 * x["new_seen"] for x in t[1]]))
                print(f"  lr={lr:<4} {rd:12s} param={str(p):6s} seen {ms([x['new_seen'] for x in rs])}  held-out {ms([x['new_heldout'] for x in rs])}")

# ---------------- Test 2 ----------------
T2 = load("t2.jsonl")
if T2:
    print("\n\nTEST 2 — indirect use: taught only CITY, asked COUNTRY via think-out-loud (3 seeds)")
    b = T2[0]["before"]
    print(f"  before: old people composed {b['old_country_composed']:.3f}, new people country {b['new_country']:.3f} (chance 0.10)")
    g = collections.defaultdict(list)
    for r in T2: g[(r["method"], r["lr"], json.dumps(r["read"]))].append(r)
    keys = ["new_city", "new_country", "new_country_and_city", "old_city", "old_country_composed", "add"]
    print("  %-8s %-6s %-20s " % ("method", "lr", "switch") + " ".join(f"{k:>22s}" for k in keys))
    for (m, lr, rd), rs in sorted(g.items()):
        print("  %-8s %-6s %-20s " % (m, lr, rd) + " ".join(f"{ms([x[k] for x in rs]):>22s}" for k in keys))

# ---------------- Test 3 ----------------
T3 = load("t3.jsonl")
if T3:
    print("\n\nTEST 3 — bounded growth (1 slot per fact; 400 facts)")
    for r in T3: r["ret"] = 0.5 * (r["old_acc"] + r["add_acc"])
    print(" Best non-oracle switch per (strategy, budget, lr) subject to retention >= 0.99; overall new-fact accuracy = 0.8*seen + 0.2*held-out")
    g = collections.defaultdict(list)
    for r in T3: g[(r["strategy"], r["budget"], r["lr"], r["read"], json.dumps(r["param"]))].append(r)
    best = {}
    for (st, B, lr, rd, p), rs in g.items():
        score = np.mean([0.8 * x["new_seen"] + 0.2 * x["new_heldout"] for x in rs])
        ret = np.mean([x["ret"] for x in rs])
        k = (st, B, rd == "oracle")
        if ret >= 0.99 and (k not in best or score > best[k][0]):
            best[k] = (score, lr, rd, p, rs)
    for B in [400, 200, 100, 50]:
        for st in ["evict", "merge", "lasso_reuse"]:
            for orc in [True, False]:
                if (st, B, orc) in best:
                    sc, lr, rd, p, rs = best[(st, B, orc)]
                    print(f"  budget {B:3d} ({B/400:.0%} of facts)  {st:12s} {'oracle switch' if orc else rd+' '+p:22s} lr={lr:<4} "
                          f"new {ms([0.8*x['new_seen']+0.2*x['new_heldout'] for x in rs])}  n={len(rs)}")
