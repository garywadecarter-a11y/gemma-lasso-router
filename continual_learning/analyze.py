"""Learning-vs-forgetting tradeoff per method.
retention = mean(old_acc/base_old, add_acc/base_add)  (1.0 = no forgetting)
Each config is averaged over seeds first (with seed std), then per method we read
off its Pareto frontier."""
import json, sys, collections, numpy as np

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "sweep.jsonl")]
cfg = collections.defaultdict(list)
for r in rows:
    r["ret"] = 0.5 * (r["old_acc"] / r["base_old"] + r["add_acc"] / r["base_add"])
    cfg[(r["method"], r["lr"], r["t"], r["lam"])].append(r)

pts = collections.defaultdict(list)   # method -> [(new, ret, new_sd, ret_sd, label, n)]
for (m, lr, t, lam), rs in cfg.items():
    nw = np.array([r["new_acc"] for r in rs]); rt = np.array([r["ret"] for r in rs])
    lab = f"lr={lr}" + (f",t={t}" if t else "") + (f",lam={lam}" if lam else "")
    pts[m].append((nw.mean(), rt.mean(), nw.std(), rt.std(), lab, len(rs)))

METHODS = ["full_ft", "mem_dense", "tfidf", "kl", "grad", "ratio", "glasso"]
print("Best retention achievable while learning at least X of the new facts (mean over seeds):\n")
TH = [0.2, 0.4, 0.6, 0.8, 0.9]
print("%-10s" % "method" + "".join("   new>=%.1f          " % x for x in TH))
for m in METHODS:
    line = "%-10s" % m
    for x in TH:
        ok = [p for p in pts[m] if p[0] >= x]
        if ok:
            b = max(ok, key=lambda p: p[1])
            line += "   %.3f±%.3f (n%.2f) " % (b[1], b[3], b[0])
        else:
            line += "   %-19s" % "  -"
    print(line)

print("\nBest new-fact learning while keeping retention at least Y:\n")
TY = [0.99, 0.97, 0.95, 0.90]
print("%-10s" % "method" + "".join("   ret>=%.2f           " % y for y in TY))
for m in METHODS:
    line = "%-10s" % m
    for y in TY:
        ok = [p for p in pts[m] if p[1] >= y]
        if ok:
            b = max(ok, key=lambda p: p[0])
            line += "   %.3f±%.3f [%s]" % (b[0], b[2], b[4][:14])
        else:
            line += "   %-21s" % "  -"
    print(line)

print("\nAll configs:")
for m in METHODS:
    for p in sorted(pts[m], key=lambda p: -p[0]):
        print("  %-10s %-24s new %.3f±%.3f  ret %.3f±%.3f  seeds %d" % (m, p[4], p[0], p[2], p[1], p[3], p[5]))


# ---- single-number summary: area under each method's frontier, per seed ----
def frontier_area(points, ref=0.8):
    xs = np.linspace(0, 1, 1001)
    best = np.array([max([r for (n, r) in points if n >= x] or [ref]) for x in xs])
    return float(np.clip(best - ref, 0, None).mean())

print("\nArea under learning-vs-retention frontier (retention above 0.80), per seed; higher is better:")
seeds = sorted({r["seed"] for r in rows})
areas = {}
for m in METHODS:
    a = [frontier_area([(r["new_acc"], r["ret"]) for r in rows if r["method"] == m and r["seed"] == s]) for s in seeds]
    areas[m] = a
    print("  %-10s %s   mean %.4f ± %.4f" % (m, " ".join("%.4f" % v for v in a), np.mean(a), np.std(a)))
for a, b in [("glasso", "kl"), ("glasso", "tfidf"), ("kl", "tfidf"), ("ratio", "grad")]:
    d = np.array(areas[a]) - np.array(areas[b])
    print("  %s - %s: per-seed %s  mean %+.4f" % (a, b, " ".join("%+.4f" % v for v in d), d.mean()))
