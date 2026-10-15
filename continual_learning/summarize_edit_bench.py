"""Summarize edit_bench.py results: best threshold per method by average of rel/gen/loc,
plus the result at locality >= 0.95 (so a method can't win by ignoring locality)."""
import json, glob, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else "edit_results"
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    r = json.load(open(f))
    if not isinstance(r, dict) or "by_tau" not in r:
        continue
    name = os.path.basename(f)[:-5]
    best = max(r["by_tau"].items(), key=lambda kv: kv[1]["average"])
    ok = [kv for kv in r["by_tau"].items() if kv[1]["locality"] >= 0.95]
    best95 = max(ok, key=lambda kv: kv[1]["average"]) if ok else None
    rows.append((name, r.get("memory_floats", 0), best, best95, r.get("edit_seconds", 0)))

lines = ["| Run | Memory (millions of numbers) | Best tau | Reliability | Generalization | Locality | Average | Best with locality >= 0.95 | Edit time (min) |",
         "|---|---|---|---|---|---|---|---|---|"]
for name, mem, (tau, m), b95, sec in rows:
    s95 = f"{b95[1]['average']:.3f} (tau {b95[0]})" if b95 else "n/a"
    lines.append(f"| {name} | {mem/1e6:.2f} | {tau} | {m['reliability']:.3f} | {m['generalization']:.3f} | "
                 f"{m['locality']:.3f} | {m['average']:.3f} | {s95} | {sec/60:.0f} |")
text = "\n".join(lines)
print(text)
open(os.path.join(d, "SUMMARY.md"), "w").write(text + "\n")
