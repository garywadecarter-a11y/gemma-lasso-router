import json, collections, numpy as np, glob
R = [json.loads(l) for f in glob.glob("mvl_*.jsonl") for l in open(f)]
for r in R:
    r["ret"] = 0.5 * (r["old_acc"] + r["add_acc"])
    r["new"] = 0.8 * r["new_seen"] + 0.2 * r["new_heldout"]
g = collections.defaultdict(list)
for r in R:
    g[(r["method"], r["budget"], r["k"], r["lr"], r["steps"], r["tau"])].append(r)
print("Best config per method and budget (tuned on mean over seeds), retention floor shown; new = 0.8*seen + 0.2*held-out\n")
for floor in [0.99, 0.97]:
    print(f"retention >= {floor}")
    print("  budget  %-13s %-22s %-14s %-14s %-14s n" % ("method", "config", "new", "seen", "held-out"))
    for B in [200, 100, 50]:
        for m in ["memoir", "lasso_recipe"]:
            c = [(k, rs) for k, rs in g.items() if k[0] == m and k[1] == B and np.mean([x["ret"] for x in rs]) >= floor and len(rs) == 3]
            if not c:
                print(f"  {B:4d}    {m:13s} (none)"); continue
            k, rs = max(c, key=lambda t: np.mean([x["new"] for x in t[1]]))
            f = lambda key: f"{np.mean([x[key] for x in rs]):.3f}±{np.std([x[key] for x in rs]):.3f}"
            cfg = f"k={k[2]} lr={k[3]} st={k[4]} tau={k[5]}"
            print(f"  {B:4d}    {m:13s} {cfg:22s} {f('new'):14s} {f('new_seen'):14s} {f('new_heldout'):14s} {len(rs)}")
    print()
# per-seed paired difference at best configs (floor 0.99)
print("Per-seed lasso - memoir (best configs, retention >= 0.99):")
for B in [200, 100, 50]:
    best = {}
    for m in ["memoir", "lasso_recipe"]:
        c = [(k, rs) for k, rs in g.items() if k[0] == m and k[1] == B and np.mean([x["ret"] for x in rs]) >= 0.99 and len(rs) == 3]
        if c:
            best[m] = {x["seed"]: x["new"] for x in max(c, key=lambda t: np.mean([x["new"] for x in t[1]]))[1]}
    if len(best) == 2:
        d = [best["lasso_recipe"][s] - best["memoir"][s] for s in sorted(best["memoir"])]
        print(f"  budget {B}: " + " ".join(f"{v:+.3f}" for v in d))
